"""
Figures for the Part 5 d20 Diff V2 run, made from the training log (the same metrics the run sent to W&B).

    python plot_d20.py d20/d20-diff-v2.log d20/d20-diff-v2.json OUT_DIR

The run had three attempts (it was resumed from the step-2,500 and step-6,500 checkpoints).
Each figure colours the attempts separately, so the overlapping steps stay visible.
"""

import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG, META, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ATTEMPT_COLORS = [BLUE, ORANGE, AQUA]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

STEP_RE = re.compile(r"^step (\d+)/\d+ .*?\| loss: ([\d.]+) \| lrm: ([\d.]+) .*?\| tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+)")
VAL_RE = re.compile(r"^Step (\d+) \| Validation bpb: ([\d.]+)")
ATTEMPT_RE = re.compile(r"^# attempt at .*resume_from=(\S+)")

attempts = []          # each: {"resume": int|None, "train": [(step, loss, lrm, tok, mfu)], "val": [(step, bpb)]}
for line in open(LOG, encoding="utf-8", errors="replace"):
    m = ATTEMPT_RE.match(line)
    if m:
        attempts.append({"resume": None if m.group(1) == "None" else int(m.group(1)), "train": [], "val": []})
        continue
    if not attempts:
        continue
    m = STEP_RE.match(line)
    if m:
        s, loss, lrm, tok, mfu = m.groups()
        attempts[-1]["train"].append((int(s), float(loss), float(lrm), int(tok.replace(",", "")), float(mfu)))
        continue
    m = VAL_RE.match(line)
    if m:
        attempts[-1]["val"].append((int(m.group(1)), float(m.group(2))))

meta = json.load(open(META))
labels = []
for i, a in enumerate(attempts):
    lo = a["train"][0][0]
    hi = max(a["train"][-1][0], max((v[0] for v in a["val"]), default=0))
    labels.append(f"Attempt {i + 1}: steps {lo:,}–{hi:,}")


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=INK2)


def restarts(ax):
    for a in attempts[1:]:
        ax.axvline(a["resume"], color=INK2, linestyle=":", linewidth=1)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote", name)


# 1. training loss
fig, ax = plt.subplots(figsize=(9, 4.8))
for a, c, lab in zip(attempts, ATTEMPT_COLORS, labels):
    xs = [t[0] for t in a["train"] if t[0] >= 50]
    ys = [t[1] for t in a["train"] if t[0] >= 50]
    ax.plot(xs, ys, color=c, linewidth=1.2, label=lab)
restarts(ax)
style(ax, "d20 Diff V2: training loss (smoothed, as logged)", "Step", "Train loss (nats/token)")
ax.legend(frameon=False, labelcolor=INK)
save(fig, "d20_train_loss.png")

# 2. validation bpb
fig, ax = plt.subplots(figsize=(9, 4.8))
merged = {}
for a in attempts:
    for s, v in a["val"]:
        merged[s] = v
ms = [k for k in sorted(merged) if k > 0]
ax.plot(ms, [merged[k] for k in ms], color="#b5b4ae", linewidth=1.2, linestyle="--", zorder=1)
for a, c, lab in zip(attempts, ATTEMPT_COLORS, labels):
    pts = [v for v in a["val"] if v[0] > 0]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, linewidth=2, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1.5, label=lab)
restarts(ax)
final_step, final_bpb = attempts[-1]["val"][-1]
ax.annotate(f"final {final_bpb:.4f}", (final_step, final_bpb), xytext=(0, -18), textcoords="offset points",
            ha="center", color=INK, fontsize=10)
ax.set_ylim(0.712, None)
style(ax, "d20 Diff V2: validation bpb every 500 steps (step 0 = 3.169, not shown)", "Step", "Validation bits per byte")
ax.legend(frameon=False, labelcolor=INK)
save(fig, "d20_val_bpb.png")

# 3. throughput
fig, ax = plt.subplots(figsize=(9, 4.8))
for a, c, lab in zip(attempts, ATTEMPT_COLORS, labels):
    pts = [t for t in a["train"] if t[0] > 0 and (a["resume"] is None or t[0] > a["resume"] + 2)]
    w = 25
    tok = [t[3] / 1e3 for t in pts]
    med = [sorted(tok[max(0, i - w // 2):i + w // 2 + 1])[len(tok[max(0, i - w // 2):i + w // 2 + 1]) // 2] for i in range(len(tok))]
    ax.plot([t[0] for t in pts], med, color=c, linewidth=1.5, label=lab)
restarts(ax)
ax.axhline(meta["tok_per_sec"] / 1e3, color=INK2, linestyle="--", linewidth=1)
ax.annotate(f"mean {meta['tok_per_sec'] / 1e3:,.0f}k tok/s, {meta['mfu_pct']:.1f}% MFU", (attempts[-1]["train"][-1][0], meta["tok_per_sec"] / 1e3),
            xytext=(0, -14), ha="right", textcoords="offset points", color=INK2, fontsize=9)
ax.set_ylim(meta["tok_per_sec"] / 1e3 * 0.97, meta["tok_per_sec"] / 1e3 * 1.03)
style(ax, "d20 Diff V2: training throughput on 8× H100 (rolling median of 25 steps)", "Step", "Tokens per second (thousands)")
ax.legend(frameon=False, labelcolor=INK, loc="lower right")
save(fig, "d20_throughput.png")

# 4. learning-rate multiplier
fig, ax = plt.subplots(figsize=(9, 3.6))
for a, c, lab in zip(attempts, ATTEMPT_COLORS, labels):
    ax.plot([t[0] for t in a["train"]], [t[2] for t in a["train"]], color=c, linewidth=2, label=lab)
restarts(ax)
style(ax, "d20 Diff V2: learning-rate multiplier (logged to 2 decimals)", "Step", "LR multiplier")
ax.legend(frameon=False, labelcolor=INK)
save(fig, "d20_lr_schedule.png")

# 5. CORE per task
tasks = sorted(meta["core_tasks"].items(), key=lambda kv: kv[1]["centered"])
fig, ax = plt.subplots(figsize=(9, 7.5))
names = [k for k, _ in tasks]
vals = [v["centered"] for _, v in tasks]
ax.barh(names, vals, color=BLUE, height=0.7)
for y, v in enumerate(vals):
    ax.text(v + (0.008 if v >= 0 else -0.008), y, f"{v:.3f}", va="center", ha="left" if v >= 0 else "right",
            color=INK, fontsize=8)
ax.axvline(0, color=INK2, linewidth=1)
ax.axvline(meta["core"], color=ORANGE, linestyle="--", linewidth=1.2)
ax.text(meta["core"] + 0.006, -1.0, f"CORE = {meta['core']:.4f}", color=INK, fontsize=9)
ax.set_xlim(min(vals) - 0.06, max(vals) + 0.08)
ax.set_ylim(-1.6, len(vals) - 0.4)
style(ax, "d20 Diff V2: CORE by task (centered accuracy, 500 items each)", "Centered accuracy", "")
ax.grid(axis="y", visible=False)
save(fig, "d20_core_tasks.png")

# merged val curve (later attempts override overlapping steps) for the README / writer
json.dump({"val_bpb_by_step": {str(k): merged[k] for k in sorted(merged)},
           "attempts": [{"label": l, "resume_from": a["resume"]} for l, a in zip(labels, attempts)]},
          open(os.path.join(OUT, "d20_curves_summary.json"), "w"), indent=1)
print("wrote d20_curves_summary.json")
