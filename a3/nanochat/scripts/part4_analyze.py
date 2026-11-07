"""Recalculate paired Part 4 intervals and figures from the archived results."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2] / "part4"
METHOD = ("Miller (2024), Section 4.2. Condition-wise paired mean difference B-A and normal "
          "95% confidence interval: mean +/- 1.96 times the sample standard deviation "
          "of paired trial differences divided by sqrt(n).")


def paired_summary(payload, source="passkey-competing-paired.json"):
    rows = payload["results"]
    steps = sorted({row["step"] for row in rows})
    if len(steps) != 2:
        raise ValueError("Expected paired predictions from two checkpoints")
    results = paired_rows([r for r in rows if r["step"] == steps[0]],
                          [r for r in rows if r["step"] == steps[1]])
    return dict(method=METHOD, source=source, results=results)


def paired_rows(reference_rows, compared_rows):
    """Compare matched prompts without relying on distinct training step numbers."""
    grouped = {}
    for model, rows in enumerate((reference_rows, compared_rows)):
        for row in rows:
            key = (row["input_tokens"], row["distance_tokens"])
            group = grouped.setdefault(key, {})
            if model in group:
                raise ValueError("Duplicate paired condition")
            cases = row["case_results"]
            if len(cases) != row["trials"] or len(cases) < 2:
                raise ValueError("Invalid paired trial count")
            actual = sum(c["predicted_index"] == c["target_index"] for c in cases)
            if actual != row["correct"]:
                raise ValueError("Saved total does not match per-trial predictions")
            if not math.isclose(row["accuracy"], actual / len(cases), abs_tol=1e-12):
                raise ValueError("Saved accuracy does not match per-trial predictions")
            group[model] = cases
    results = []
    for (length, distance), group in sorted(grouped.items()):
        if set(group) != {0, 1} or len(group[0]) != len(group[1]):
            raise ValueError("Missing paired trials")
        pairs = list(zip(group[0], group[1], strict=True))
        outcomes = []
        for a, b in pairs:
            if any(a[key] != b[key] for key in ("seed", "candidate_token_ids", "target_index")):
                raise ValueError("The paired prompts or expected answers do not match")
            outcomes.append((int(a["predicted_index"] == a["target_index"]),
                             int(b["predicted_index"] == b["target_index"])))
        differences = [b - a for a, b in outcomes]
        gain = 100 * statistics.mean(differences)
        se = 100 * statistics.stdev(differences) / math.sqrt(len(differences))
        results.append(dict(input_tokens=length, distance_tokens=distance, n=len(outcomes),
                            both_correct=outcomes.count((1, 1)), a_only=outcomes.count((1, 0)),
                            b_only=outcomes.count((0, 1)), both_wrong=outcomes.count((0, 0)),
                            difference_pp=gain, se_pp=se, ci95_pp=[gain - 1.96 * se, gain + 1.96 * se]))
    return results


def baseline_summary(single, four, baseline_single, baseline_four):
    """Rebuild the saved A/B/baseline table and B-minus-baseline paired intervals."""
    if single["seed"] != baseline_single["seed"] or four["seed"] != baseline_four["seed"]:
        raise ValueError("Evaluation seeds do not match")
    key = lambda row: (row["input_tokens"], row["distance_tokens"])
    sources = (("single_a", single, 1260), ("single_b", single, 2520),
               ("single_baseline", baseline_single, 2520), ("four_a", four, 1260),
               ("four_b", four, 2520), ("four_baseline", baseline_four, 2520))
    indices = {}
    for name, payload, step in sources:
        selected = [r for r in payload["results"] if r["step"] == step]
        indices[name] = {key(r): r for r in selected}
        if len(indices[name]) != len(selected):
            raise ValueError("Duplicate evaluation condition")
    conditions = set(indices["four_b"])
    if len(conditions) != 9 or any(set(index) != conditions for index in indices.values()):
        raise ValueError("Missing matched evaluation conditions")
    intervals = paired_rows(indices["four_baseline"].values(), indices["four_b"].values())
    results = []
    for interval in intervals:
        condition = key(interval)
        row = dict(input_tokens=condition[0], distance_tokens=condition[1], trials=128)
        for name, index in indices.items():
            record = index[condition]
            if (record["trials"] != 128 or not 0 <= record["correct"] <= 128 or
                    not math.isclose(record["accuracy"], record["correct"] / 128, abs_tol=1e-12)):
                raise ValueError("Invalid saved evaluation total")
            row[name + "_correct"] = record["correct"]
            row[name + "_accuracy_pct"] = 100 * record["accuracy"]
        row.update(b_minus_baseline_pp=interval["difference_pp"],
                   ci95_low_pp=interval["ci95_pp"][0], ci95_high_pp=interval["ci95_pp"][1],
                   b_only=interval["b_only"], baseline_only=interval["a_only"])
        results.append(row)
    method = ("For each condition, d_i = correct_B_i - correct_baseline_i. Difference in percentage "
              "points = 100 * mean(d). Approximate pointwise 95% paired interval = 100 * (mean(d) "
              "+/- 1.96 * sample_stdev(d) / sqrt(128)). Intervals describe prompt variation for two fixed checkpoints.")
    return dict(comparison="Checkpoint B (512 then 2048) versus Part 2 d12-baseline (2048 throughout)",
                training_steps_per_final_checkpoint=2520, training_tokens_per_final_checkpoint=1321205760,
                single_record_seed=single["seed"], four_record_seed=four["seed"], method=method,
                sources=["passkey-results.json", "passkey-competing-paired.json",
                         "part2-baseline-single-record.json", "part2-baseline-four-record-paired.json"],
                results=results)


def read_training_log(path):
    text = Path(path).read_text(encoding="utf-8")
    train = [(int(s), float(v)) for s, v in re.findall(r"step (\d+)/\d+[^\n]*?loss: ([\d.]+)", text)]
    validation = [(int(s), float(v)) for s, v in re.findall(r"Step (\d+) \| Validation bpb: ([\d.]+)", text)]
    if not train or not validation:
        raise ValueError(f"Missing loss observations in {path}")
    return dict(train=train, validation=validation)


def plot_figures(payload, summary, log_dir, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    blue, orange = "#285e80", "#ac4b15"
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for y, row in enumerate(summary["results"]):
        gain = row["difference_pp"]
        lo, hi = row["ci95_pp"]
        ax.errorbar(gain, y, xerr=[[gain - lo], [hi - gain]], fmt="o", color=blue, capsize=4)
        label = Decimal(str(gain)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        ax.text(1.02, y, f"{label:+.1f}", transform=ax.get_yaxis_transform(), va="center")
    ax.text(1.02, 1.025, "B - A", transform=ax.transAxes, fontsize=11)
    ax.set_yticks(range(len(summary["results"])),
                  [f"{r['input_tokens']} / {r['distance_tokens']}" for r in summary["results"]])
    ax.invert_yaxis()
    ax.set_xlim(-16, 44)
    ax.axvline(0, color="0.4", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Accuracy difference B - A (percentage points)")
    ax.set_ylabel("Input length / code distance (tokens)")
    ax.grid(axis="x", alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    fig.savefig(output_dir / "paired-accuracy-difference.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    for step, label, color, marker in [(1260, "Checkpoint A", blue, "o"), (2520, "Checkpoint B", orange, "s")]:
        rows = sorted((r for r in payload["results"] if r["step"] == step and r["input_tokens"] == 2048),
                      key=lambda r: r["distance_tokens"])
        ax.plot([r["distance_tokens"] for r in rows], [100 * r["accuracy"] for r in rows],
                marker=marker, color=color, label=label)
    ax.axhline(100 * payload["chance"], color="0.4", linestyle="--", label="Chance (25%)")
    ax.set(xlabel="Code distance from end of prompt (tokens)", ylabel="Accuracy",
           xticks=[128, 384, 768, 1536], yticks=[0, 25, 50, 75, 100], ylim=(0, 105))
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "four-record-accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for length, color, marker in [(512, blue, "o"), (2048, orange, "s")]:
        curve = read_training_log(log_dir / f"d12-context-part4-{length}.log")
        for ax, key in zip(axes, ("train", "validation"), strict=True):
            points = [(s, v) for s, v in curve[key] if s >= 250]
            ax.plot(*zip(*points), color=color, marker=marker if key == "validation" else None,
                    markersize=4, label=f"{length}-token stage")
            ax.axvline(1260, color="0.4", linestyle="--", linewidth=1)
            ax.grid(axis="y", alpha=0.2)
            ax.set_xticks([250, 750, 1260, 1750, 2520])
            ax.set_xlabel("Optimizer step")
    axes[0].set(title="Training loss", ylabel="Nats per token")
    axes[1].set(title="Validation loss", ylabel="Bits per byte")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_dir / "training-validation-loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/passkey-competing-paired.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--summary-only", action="store_true", help="skip plots and the matplotlib dependency")
    parser.add_argument("--include-baseline", action="store_true", help="also rebuild the Part 2 baseline comparison")
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    summary = paired_summary(payload, args.results.name)
    destination = args.output_dir / "results/passkey-paired-summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.include_baseline:
        read = lambda name: json.loads((args.results.parent / name).read_text(encoding="utf-8"))
        comparison = baseline_summary(read("passkey-results.json"), payload,
                                      read("part2-baseline-single-record.json"),
                                      read("part2-baseline-four-record-paired.json"))
        path = destination.with_name("part2-baseline-comparison")
        path.with_suffix(".json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison["results"][0]))
            writer.writeheader()
            writer.writerows(comparison["results"])
    if not args.summary_only:
        plot_figures(payload, summary, ROOT / "logs", args.output_dir / "figures")
    print(f"Recalculated {len(summary['results'])} paired intervals: {destination}")


if __name__ == "__main__":
    main()
