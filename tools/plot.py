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
    "font.size": 21, "axes.labelsize": 22, "axes.titlesize": 28, "axes.labelweight": "bold", "xtick.labelsize": 19, "ytick.labelsize": 19,
    "axes.linewidth": 1.1, "xtick.major.width": 1.1, "ytick.major.width": 1.1, "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
# Okabe-Ito (colour-blind safe)
C_OURS, C_K, C_IQ, C_BASE = "#009E73", "#0072B2", "#E69F00", "#D55E00"


def style(r):
    m = r["model"]
    if r["ours"]:
        return dict(marker="o", color=C_OURS, s=110, z=4)
    if m.endswith("Q4_K_M"):
        return dict(marker="*", color=C_BASE, s=460, z=5)
    if "IQ" in m:
        return dict(marker="^", color=C_IQ, s=125, z=3)
    return dict(marker="s", color=C_K, s=100, z=3)


def short(r):
    return r["short"]


# label offsets (points) per panel, tuned to avoid overlaps
OFF = {
    "mix_kld": {"2.6 bits": (10, -4), "3.0 bits": (10, 1), "3.2 bits": (-52, -16), "3.6 bits": (10, -6), "4.0 bits": (10, 5),
                "4.4 bits": (10, -10), "UD-IQ2_M": (8, 5), "UD-Q2_K_XL": (8, 5), "UD-IQ3_XXS": (8, 5), "Q3_K_M": (8, 5),
                "UD-Q3_K_XL": (8, 5), "Q4_K_M": (-13, 12)},
    "tasks_public4": {"2.6 bits": (10, -5), "3.0 bits": (8, -16), "3.2 bits": (-21, 10), "3.6 bits": (-44, 10), "4.0 bits": (-23, 10),
                      "4.4 bits": (6, -17), "UD-IQ2_M": (8, -5), "UD-Q2_K_XL": (8, -13), "UD-IQ3_XXS": (8, -16),
                      "Q3_K_M": (8, -16), "UD-Q3_K_XL": (-51, 10), "Q4_K_M": (-17, -21)},
    "decode_tok_s": {"2.6 bits": (10, -4), "3.0 bits": (8, 4), "3.2 bits": (-51, -17), "3.6 bits": (8, 4), "4.0 bits": (-10, 10),
                     "4.4 bits": (8, 5), "UD-IQ2_M": (-23, 10), "UD-Q2_K_XL": (8, -13), "UD-IQ3_XXS": (8, -16),
                     "Q3_K_M": (8, -13), "UD-Q3_K_XL": (8, -13), "Q4_K_M": (-21, 12)},
}


def panel(ax, key, ylabel, title, log=False, ylim=None):
    ours = sorted([r for r in rows if r["ours"]], key=lambda r: r["file_gb"])
    ax.plot([r["file_gb"] for r in ours], [r[key] for r in ours], color=C_OURS, lw=2.6, zorder=2)
    for r in rows:
        st = style(r)
        if key == "tasks_public4" and r.get("tasks_ci"):  # bootstrap 95% CI over the task items
            ax.errorbar(r["file_gb"], r[key], yerr=[[r[key] - r["tasks_ci"][0]], [r["tasks_ci"][1] - r[key]]],
                        fmt="none", ecolor=st["color"], elinewidth=1.4, capsize=4, alpha=0.55, zorder=1)
        ax.scatter(r["file_gb"], r[key], marker=st["marker"], s=st["s"], color=st["color"], zorder=st["z"],
                   edgecolors="white", linewidths=0.6)
        dx, dy = OFF[key].get(short(r), (5, 3))
        ax.annotate(short(r), (r["file_gb"], r[key]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=17, color=st["color"], zorder=6)
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
    ax.set_title(title, loc="center", fontweight="bold", pad=10)
    ax.grid(True, which="major", lw=0.4, alpha=0.35)


fig, axes = plt.subplots(1, 3, figsize=(24, 7.6))
panel(axes[0], "mix_kld", r"KLD to BF16 $\downarrow$ (log)", "(a) Output fidelity", log=True, ylim=(0.02, 0.22))
panel(axes[1], "tasks_public4", r"Task accuracy (%) $\uparrow$", "(b) Downstream tasks (95% CI)", ylim=(63, 83))
panel(axes[2], "decode_tok_s", r"Decode throughput (tok/s) $\uparrow$", "(c) Speed (A100, CUDA)", ylim=(105, 156))
handles = [Line2D([], [], color=C_OURS, marker="o", lw=2.6, ms=13, label="ShiftAddQwen-3.5-9B-$x$bits (ours)"),
           Line2D([], [], color=C_BASE, marker="*", lw=0, ms=24, label="UnslothAI/Qwen-3.5-9B-Q4_K_M (reference)"),
           Line2D([], [], color=C_K, marker="s", lw=0, ms=13, label="UnslothAI/Qwen-3.5-9B-{Q3_K_M, UD-Q*_K_XL}"),
           Line2D([], [], color=C_IQ, marker="^", lw=0, ms=14, label="UnslothAI/Qwen-3.5-9B-UD-IQ*")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.19), fontsize=22, prop={"weight": "bold", "size": 22},
           handletextpad=0.6, columnspacing=3.0)
fig.tight_layout(w_pad=0.6)
(R / "docs").mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(R / "docs" / f"frontier.{ext}", dpi=300)
print("wrote docs/frontier.png, docs/frontier.pdf")
