#!/usr/bin/env python3
"""
Plot Granite 3.1 1B-A400M baseline benchmark results.
Reads lm-evaluation-harness result JSONs and saves a slide-ready
horizontal bar chart to figures/baseline_results.{png,svg}.
"""
import json
import os

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWLEDGE_FILE = os.path.join(
    BASE, "results", "baseline_knowledge_medium2_20261103T131441Z", "results.json"
)
REASONING_FILE = os.path.join(
    BASE, "results", "baseline_reasoning_medium_20261103T145939Z", "results.json"
)

with open(KNOWLEDGE_FILE) as fh:
    kd = json.load(fh)
with open(REASONING_FILE) as fh:
    rd = json.load(fh)

kr = kd["results"]
rr = rd["results"]

# ── Task definitions (chart order: top → bottom) ─────────────────────────────
TASKS = [
    {
        "name":   "ARC-Easy",
        "metric": "normalised accuracy",
        "val":    kr["arc_easy"]["acc_norm,none"],
        "se":     kr["arc_easy"]["acc_norm_stderr,none"],
        "chance": 0.25,
    },
    {
        "name":   "MMLU",
        "metric": "accuracy",
        "val":    kr["mmlu_continuation"]["acc,none"],
        "se":     kr["mmlu_continuation"]["acc_stderr,none"],
        "chance": 0.25,
    },
    {
        "name":   "TriviaQA",
        "metric": "exact match",
        "val":    kr["triviaqa"]["exact_match,remove_whitespace"],
        "se":     kr["triviaqa"]["exact_match_stderr,remove_whitespace"],
        "chance": None,
    },
    {
        "name":   "GSM8K",
        "metric": "exact match, strict",
        "val":    rr["gsm8k"]["exact_match,strict-match"],
        "se":     rr["gsm8k"]["exact_match_stderr,strict-match"],
        "chance": None,
    },
    {
        "name":   "HellaSwag",
        "metric": "normalised accuracy",
        "val":    rr["hellaswag"]["acc_norm,none"],
        "se":     rr["hellaswag"]["acc_norm_stderr,none"],
        "chance": 0.25,
    },
    {
        "name":   "Winogrande",
        "metric": "accuracy",
        "val":    rr["winogrande"]["acc,none"],
        "se":     rr["winogrande"]["acc_stderr,none"],
        "chance": 0.5,
    },
]

# Reverse so ARC-Easy appears at the top of the horizontal chart
tasks = TASKS[::-1]

# ── Font ─────────────────────────────────────────────────────────────────────
def _pick_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("IBM Plex Sans", "Arial", "Helvetica"):
        if name in available:
            return name
    return "sans-serif"

matplotlib.rcParams["font.family"] = _pick_font()

# ── Colours ───────────────────────────────────────────────────────────────────
BAR_COLOR    = "#3573DF"
ERR_COLOR    = "#16202E"
CHANCE_COLOR = "#EA6C13"
INK          = "#16202E"
GREY         = "#8A8A8A"

# ── Figure: 16 × 4.5 in @ 317 dpi  →  ~3200 × 1430 px after tight crop ──────
fig, ax = plt.subplots(figsize=(16, 4.5))
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

n = len(tasks)
ypos = np.arange(n) * 0.85     # tighter row spacing
BAR_H = 0.42

vals = [t["val"] for t in tasks]
err2 = [2.0 * t["se"] for t in tasks]

# Bars + error bars
ax.barh(
    ypos, vals,
    height=BAR_H,
    color=BAR_COLOR,
    xerr=err2,
    error_kw=dict(ecolor=ERR_COLOR, elinewidth=1.3, capsize=5, capthick=1.3),
    zorder=3,
)

# Chance ticks (short orange vertical line across each bar)
for i, t in enumerate(tasks):
    if t["chance"] is not None:
        ax.plot(
            [t["chance"], t["chance"]],
            [ypos[i] - BAR_H / 2, ypos[i] + BAR_H / 2],
            color=CHANCE_COLOR,
            linewidth=3.0,
            zorder=5,
            solid_capstyle="butt",
            clip_on=False,
        )

# Value labels to the right: "0.590 ± 0.016"
for i, t in enumerate(tasks):
    x_start = t["val"] + err2[i] + 0.016
    ax.text(
        x_start, ypos[i],
        f"{t['val']:.3f} ± {t['se']:.3f}",
        va="center", ha="left",
        fontsize=12.5,
        color=INK,
        clip_on=False,
    )

# Y-axis labels: bold task name on top line, grey metric on bottom line
trans = blended_transform_factory(ax.transAxes, ax.transData)
for i, t in enumerate(tasks):
    ax.text(
        -0.025, ypos[i] + 0.125,
        t["name"],
        va="center", ha="right",
        fontsize=14, fontweight="bold",
        color=INK,
        transform=trans,
        clip_on=False,
    )
    ax.text(
        -0.025, ypos[i] - 0.125,
        t["metric"],
        va="center", ha="right",
        fontsize=10,
        color=GREY,
        transform=trans,
        clip_on=False,
        style="italic",
    )

# ── Axes styling ──────────────────────────────────────────────────────────────
ax.set_xlim(0, 0.8)
ax.set_ylim(ypos[0] - 0.5, ypos[-1] + 0.5)
ax.set_yticks([])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
ax.tick_params(axis="x", labelsize=13, length=5, color=GREY)
ax.xaxis.label.set_color(GREY)

for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GREY)
ax.spines["bottom"].set_linewidth(0.8)

ax.grid(False)

# ── Caption ───────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.012,
    "Dark line: ±2 standard errors. Orange tick: chance. "
    "First-N subsets, except GSM8K and Winogrande (full sets).",
    ha="center", va="bottom",
    fontsize=10, color=GREY,
    style="italic",
)

# ── Layout & save ─────────────────────────────────────────────────────────────
plt.subplots_adjust(left=0.30, right=0.78, top=0.95, bottom=0.12)

figures_dir = os.path.join(BASE, "figures")
os.makedirs(figures_dir, exist_ok=True)

png_path = os.path.join(figures_dir, "baseline_results.png")
svg_path = os.path.join(figures_dir, "baseline_results.svg")

fig.savefig(png_path, dpi=317, bbox_inches="tight", transparent=True)
fig.savefig(svg_path, bbox_inches="tight", transparent=True)

print(f"Saved {png_path}")
print(f"Saved {svg_path}")
