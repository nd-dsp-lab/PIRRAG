"""Probe-selection figure (fig:probe_selection).

Held-out recall@10 of equal-size clustering vs. number of probed clusters p, against
the unbalanced k-means IVF-Flat baseline at the same scale. Sized for a full-width
IEEE figure* (\\textwidth = 7.16 in), so fonts print at their nominal size.

Run from the repo root:
  .venv-embed/bin/python figures/make_probe_selection.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.9,
    "legend.frameon": False, "pdf.fonttype": 42,
})
BLUE, GRAY, RED = "#0F4D92", "#767676", "#B64342"

# (panel name as used in the paper, sweep file, cluster width n, chosen p, x-range, label offset)
# "700K" / "1.1M" are the real vector counts of the HF top-1M / top-10M indices.
CFG = [
    ("65K", "wiki-rag/cluster_size_sweep/cluster_size_sweep.json", 128, 43, (0, 130), (8, -16)),
    ("700K", "wiki-rag/cluster_size_sweep_1M/cluster_size_sweep.json", 256, 232, (0, 700), (7, -15)),
    ("1.1M", "wiki-rag/cluster_size_sweep_10M/cluster_size_sweep.json", 256, 546, (0, 1300), (7, -15)),
]

fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.1), sharey=True)
for ax, (name, path, n, p_sel, xlim, offset) in zip(axes, CFG):
    d = json.load(open(path))
    row = next(r for r in d["rows"] if r["cluster_size"] == n)
    target = 100 * d["target_recall"]
    pts = sorted((int(p), 100 * r) for p, r in row["recall_curve"].items() if xlim[0] <= int(p) <= xlim[1])

    ax.plot([p for p, _ in pts], [r for _, r in pts], "-o", color=BLUE, lw=1.8, ms=4, zorder=3)
    ax.axhline(target, color=RED, ls="--", lw=1.1)
    ax.axvline(p_sel, color=GRAY, ls=":", lw=0.9)
    y = 100 * row["recall_curve"][str(p_sel)]
    ax.plot([p_sel], [y], "o", ms=8.5, mfc="white", mec=BLUE, mew=1.9, zorder=4)
    ax.annotate(f"p = {p_sel}", (p_sel, y), textcoords="offset points", xytext=offset,
                fontsize=10.5, color=BLUE, fontweight="bold")
    ax.set_title(f"{name}  (n = {n})\nbaseline {target:.1f}%", fontsize=10.5, linespacing=1.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(93, 100.3)
    ax.set_xlabel("clusters probed p")

axes[0].set_ylabel("held-out recall@10 (%)")
axes[1].legend(
    [Line2D([], [], color=BLUE, marker="o", lw=1.8, ms=4),
     Line2D([], [], color=RED, ls="--", lw=1.1),
     Line2D([], [], color=BLUE, marker="o", ms=6, mfc="white", mew=1.5, lw=0)],
    ["equal-size clustering", "unbalanced IVF baseline", "chosen p"],
    loc="lower right", fontsize=9, handlelength=1.8,
)
fig.tight_layout(pad=0.4, w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(f"figures/probe_selection.{ext}", dpi=300)
