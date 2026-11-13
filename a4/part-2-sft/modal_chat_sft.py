"""Launch nanochat SFT on Modal from a pinned pretrained checkpoint.

Example:
    NANOCHAT_REPO=/path/to/nanochat modal run modal_chat_sft.py \
        --volume a4-nanochat-volume --base-path a4 \
        --model-tag d20-baseline-8-H100 --model-step 9960 \
        --output-tag a4-original-sft --gpus 8

The repository mounted into the image must contain scripts/chat_sft.py. Additional
datasets can be supplied by modifying that script in the selected repository and
passing its CLI arguments through --sft-args.
"""

import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import modal


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent == Path("/root"):
    REPO = Path("/root/nanochat")
else:
    # Supports the launcher in either a4/modal/ or a4/part-2-sft/.
    repo_candidates = (
        SCRIPT_PATH.parents[2] / "a3" / "nanochat",
        SCRIPT_PATH.parents[1] / "a3" / "nanochat",
    )
    default_repo = next((candidate for candidate in repo_candidates if (candidate / "scripts" / "chat_sft.py").is_file()), repo_candidates[0])
    REPO = Path(os.environ.get("NANOCHAT_REPO", default_repo)).expanduser().resolve()
if not (REPO / "scripts" / "chat_sft.py").is_file():
    raise FileNotFoundError(f"Set NANOCHAT_REPO to a nanochat checkout: {REPO}")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy", "tiktoken", "filelock", "psutil", "pyarrow", "requests",
        "jinja2", "datasets", "kernels==0.11.7", "huggingface-hub==0.34.4",
        "rustbpe", "wandb",
    )
    .add_local_dir(REPO, "/root/nanochat", ignore=[".venv", ".git", "**/__pycache__", "**/.pytest_cache"])
)
app = modal.App("nanochat-a4-sft", image=image)


@app.function(timeout=12 * 3600, cpu=8)
def train(
    base_path: str,
    model_tag: str,
    model_step: int,
    output_tag: str,
    run_name: str,
    gpus: int,
    load_optimizer: bool,
    sft_args: tuple[str, ...],
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str,
):
    base_dir = Path("/vol") / base_path
    checkpoint_dir = base_dir / "base_checkpoints" / model_tag
    prefix = f"{model_step:06d}"
    required = [
        checkpoint_dir / f"model_{prefix}.pt",
        checkpoint_dir / f"meta_{prefix}.json",
        base_dir / "tokenizer" / "tokenizer.pkl",
        base_dir / "tokenizer" / "token_bytes.pt",
    ]
    if load_optimizer:
        required.extend(checkpoint_dir / f"optim_{prefix}_rank{rank}.pt" for rank in range(gpus))
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    metadata = json.loads((checkpoint_dir / f"meta_{prefix}.json").read_text())
    if metadata["step"] != model_step:
        raise ValueError("Checkpoint step and metadata disagree.")
    if metadata["model_config"]["vocab_size"] <= 0:
        raise ValueError("Invalid vocabulary size in checkpoint metadata.")

    run_dir = base_dir / "sft_runs" / output_tag
    output_dir = run_dir / "chatsft_checkpoints" / model_tag
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output tag already has files: {output_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("base_checkpoints", "tokenizer"):
        link = run_dir / name
        if not link.exists():
            link.symlink_to(base_dir / name, target_is_directory=True)

    log_dir = base_dir / "logs"
    result_dir = base_dir / "results"
    wandb_dir = base_dir / "wandb"
    for directory in (log_dir, result_dir, wandb_dir):
        directory.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        NANOCHAT_BASE_DIR=str(run_dir),
        WANDB_MODE=wandb_mode,
        WANDB_DIR=str(wandb_dir),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
    )
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    # The upstream script hardcodes its W&B project and omits processed tokens.
    # Instrument only the container's copy so local nanochat remains unchanged.
    script_path = Path("/root/nanochat/scripts/chat_sft.py")
    script = script_path.read_text()
    project_call = 'wandb.init(project="nanochat-sft", name=args.run, config=user_config)'
    flops_field = '"total_training_flops": flops_so_far,'
    if script.count(project_call) != 1 or script.count(flops_field) != 3:
        raise RuntimeError("Unexpected chat_sft.py logging code; update launcher instrumentation.")
    script = script.replace(
        project_call,
        'wandb.init(project=os.environ["WANDB_PROJECT"], name=args.run, config=user_config)',
    )
    script = script.replace(
        flops_field,
        flops_field + '\n            "total_training_tokens": args.total_batch_size * step,',
    )
    script_path.write_text(script)
    env["WANDB_PROJECT"] = wandb_project
    launcher = (
        ["torchrun", "--standalone", f"--nproc_per_node={gpus}", "-m", "scripts.chat_sft", "--"]
        if gpus > 1 else ["python", "-m", "scripts.chat_sft"]
    )
    command = launcher + [
        f"--model-tag={model_tag}",
        f"--model-step={model_step}",
        f"--load-optimizer={int(load_optimizer)}",
        f"--run={run_name}",
        *sft_args,
    ]
    print("Launching:", shlex.join(command), flush=True)
    started = time.time()
    log_path = log_dir / f"{output_tag}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd="/root/nanochat", env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    result = {
        "command": command,
        "input_checkpoint": str(checkpoint_dir),
        "input_step": model_step,
        "output_checkpoint": str(output_dir),
        "gpus": gpus,
        "load_optimizer": load_optimizer,
        "wandb_mode": wandb_mode,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "returncode": returncode,
        "elapsed_seconds": time.time() - started,
        "log_path": str(log_path),
    }
    (result_dir / f"{output_tag}.json").write_text(json.dumps(result, indent=2))
    if returncode:
        raise RuntimeError(f"SFT failed with exit code {returncode}; see {log_path}")
    expected = list(output_dir.glob("model_*.pt"))
    if len(expected) != 1:
        raise RuntimeError(f"Expected one final SFT checkpoint in {output_dir}")
    return result


@app.local_entrypoint()
def main(
    volume: str = "a4-nanochat-volume",
    base_path: str = "a4",
    model_tag: str = "d20-baseline-8-H100",
    model_step: int = 9960,
    output_tag: str = "a4-original-sft",
    run_name: str = "a4-original-sft",
    gpus: int = 8,
    load_optimizer: bool = True,
    sft_args: str = "",
    wandb_mode: str = "offline",
    wandb_secret: str = "",
    wandb_project: str = "",
    wandb_entity: str = "",
):
    if not base_path or Path(base_path).is_absolute() or ".." in Path(base_path).parts:
        raise ValueError("base_path must be a relative path within the volume.")
    for label, value in (("model_tag", model_tag), ("output_tag", output_tag)):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{label} must be a single directory name.")
    if model_step < 0:
        raise ValueError("model_step must be nonnegative.")
    if not run_name or run_name == "dummy":
        raise ValueError("run_name must enable W&B logging.")
    if gpus not in (1, 2, 4, 8):
        raise ValueError("gpus must be 1, 2, 4, or 8.")
    if wandb_mode not in ("offline", "online"):
        raise ValueError("wandb_mode must be offline or online.")
    if wandb_mode == "online" and not wandb_secret:
        raise ValueError("Online W&B requires the name of a Modal secret with WANDB_API_KEY.")
    if not wandb_project:
        raise ValueError("Pass a W&B project with --wandb-project.")
    extra_args = tuple(shlex.split(sft_args))
    controlled = ("--model-tag", "--model-step", "--load-optimizer", "--run")
    if any(arg.split("=", 1)[0] in controlled for arg in extra_args):
        raise ValueError("Pass checkpoint and run options through the launcher's named arguments.")
    options = {
        "gpu": f"H100!:{gpus}" if gpus > 1 else "H100!",
        "volumes": {"/vol": modal.Volume.from_name(volume)},
        "timeout": 12 * 3600,
    }
    if wandb_secret:
        options["secrets"] = [modal.Secret.from_name(wandb_secret)]
    result = train.with_options(**options).remote(
        base_path, model_tag, model_step, output_tag, run_name, gpus,
        load_optimizer, extra_args, wandb_mode, wandb_project, wandb_entity,
    )
    print(json.dumps(result, indent=2))
