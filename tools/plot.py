"""Paper-style frontier figure from results/summary.json -> docs/frontier.{png,pdf}.

(a) KLD to BF16 on mixed-domain text, (b) task accuracy, (c) decode throughput, all vs GGUF size.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

R = Path(__file__).resolve().parents[1]
rows = [r for r in json.load(open(R / "results/summary.json"))["rows"] if not r["ablation"] and r["file_gb"] < 7]
# r["short"]: '4.0 bits' for ours, the recipe name for baselines (set by tools/names.py)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
# Okabe-Ito (colour-blind safe)
C_OURS, C_K, C_IQ, C_BASE = "#009E73", "#0072B2", "#E69F00", "#D55E00"


def style(r):
    m = r["model"]
    if r["ours"]:
        return dict(marker="o", color=C_OURS, s=30, z=4)
    if m.endswith("Q4_K_M"):
        return dict(marker="*", color=C_BASE, s=130, z=5)
    if "IQ" in m:
        return dict(marker="^", color=C_IQ, s=34, z=3)
    return dict(marker="s", color=C_K, s=28, z=3)


def short(r):
    return r["short"]


# label offsets (points) per panel, tuned to avoid overlaps
OFF = {
    "mix_kld": {"2.6 bits": (6, -2), "3.0 bits": (6, 1), "3.2 bits": (-31, -9), "3.6 bits": (6, -4), "4.0 bits": (6, 3),
                "4.4 bits": (6, -6), "UD-IQ2_M": (5, 3), "UD-Q2_K_XL": (5, 3), "UD-IQ3_XXS": (5, 3), "Q3_K_M": (5, 3),
                "UD-Q3_K_XL": (5, 3), "Q4_K_M": (-8, 7)},
    "tasks_public4": {"2.6 bits": (6, -3), "3.0 bits": (5, -9), "3.2 bits": (-12, 6), "3.6 bits": (-26, 6), "4.0 bits": (-14, 6),
                      "4.4 bits": (4, -10), "UD-IQ2_M": (5, -3), "UD-Q2_K_XL": (5, -8), "UD-IQ3_XXS": (5, -9),
                      "Q3_K_M": (5, -9), "UD-Q3_K_XL": (-30, 6), "Q4_K_M": (-10, -12)},
    "decode_tok_s": {"2.6 bits": (6, -2), "3.0 bits": (5, 2), "3.2 bits": (-30, -10), "3.6 bits": (5, 2), "4.0 bits": (-6, 6),
                     "4.4 bits": (5, 3), "UD-IQ2_M": (-14, 6), "UD-Q2_K_XL": (5, -8), "UD-IQ3_XXS": (5, -9),
                     "Q3_K_M": (5, -8), "UD-Q3_K_XL": (5, -8), "Q4_K_M": (-12, 7)},
}


def panel(ax, key, ylabel, title, log=False, ylim=None):
    ours = sorted([r for r in rows if r["ours"]], key=lambda r: r["file_gb"])
    ax.plot([r["file_gb"] for r in ours], [r[key] for r in ours], color=C_OURS, lw=1.2, zorder=2)
    for r in rows:
        st = style(r)
        ax.scatter(r["file_gb"], r[key], marker=st["marker"], s=st["s"], color=st["color"], zorder=st["z"],
                   edgecolors="white", linewidths=0.6)
        dx, dy = OFF[key].get(short(r), (5, 3))
        ax.annotate(short(r), (r["file_gb"], r[key]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7, color=st["color"], zorder=6)
    if log:
        ax.set_yscale("log")
        ax.set_yticks([0.02, 0.03, 0.05, 0.1, 0.2])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(3.35, 6.15)
    ax.set_xlabel("GGUF size (GB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, which="major", lw=0.4, alpha=0.35)


fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))
panel(axes[0], "mix_kld", r"KLD to BF16 $\downarrow$ (log)", "(a) Output fidelity", log=True, ylim=(0.02, 0.22))
panel(axes[1], "tasks_public4", r"Task accuracy (%) $\uparrow$", "(b) Downstream tasks", ylim=(66.5, 80.5))
panel(axes[2], "decode_tok_s", r"Decode throughput (tok/s) $\uparrow$", "(c) Speed (A100, CUDA)", ylim=(105, 156))
handles = [Line2D([], [], color=C_OURS, marker="o", lw=1.2, ms=5, label="ShiftAddQwen-3.5-9B-$x$bits (ours)"),
           Line2D([], [], color=C_BASE, marker="*", lw=0, ms=10, label="UnslothAI/Qwen-3.5-9B-Q4_K_M (reference)"),
           Line2D([], [], color=C_K, marker="s", lw=0, ms=5, label="UnslothAI/Qwen-3.5-9B-{Q3_K_M, UD-Q*_K_XL}"),
           Line2D([], [], color=C_IQ, marker="^", lw=0, ms=5.5, label="UnslothAI/Qwen-3.5-9B-UD-IQ*")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=8)
fig.tight_layout(w_pad=1.6)
(R / "docs").mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(R / "docs" / f"frontier.{ext}", dpi=300)
print("wrote docs/frontier.png, docs/frontier.pdf")
