"""Figures for the rewritten pre-processing section (equal-size clustering).

Reads the measured sweep outputs directly; nothing is typed in by hand.

  fig:probes         wiki-rag/verify_p43/verify_p43.json
  fig:cluster_width  wiki-rag/cluster_size_sweep/cluster_size_sweep.json (rows)
                     wiki-rag/verify_p43/verify_p43.json (baseline fetch range)

Run from the repo root:
  python paper_preprocessing/make_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "gray": "#767676",
    "dark": "#272727",
}
CENTROIDS_PER_CT = 8  # CKKS packing, N = 16384, B = 8 (Table IV)
COL_W = 3.5  # IEEE column width, inches

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def load(rel):
    return json.loads((ROOT / rel).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_probes():
    v = load("wiki-rag/verify_p43/verify_p43.json")
    base = {int(p): r for p, r in v["baseline"]["recall_by_p"].items()}
    bal = {int(p): r for p, r in v["n128"]["recall_by_p"].items()}
    K_base, K_bal = v["baseline"]["K"], v["n128"]["K"]
    s = 128
    m_sel, m_base = 43, 100
    target = base[m_base]
    cands = v["baseline"]["candidates_p100"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    bx, by = zip(*sorted(base.items()))
    ex, ey = zip(*sorted(bal.items()))
    ax.plot(bx, by, color=PALETTE["red_strong"], lw=1.3, marker="s", ms=3,
            label=f"$k$-means, variable sizes ($|\\mathbf{{C}}|$ = {K_base:,})")
    ax.plot(ex, ey, color=PALETTE["blue_main"], lw=1.3, marker="o", ms=2.6,
            label=f"Balanced, $s$ = {s} ($|\\mathbf{{C}}|$ = {K_bal})")

    ax.axhline(target, color=PALETTE["gray"], lw=0.8, ls="--", zorder=0)
    ax.text(203, target + 0.0012, f"target {target:.3f}", ha="right", va="bottom",
            fontsize=6.5, color=PALETTE["gray"])
    ax.axvline(m_sel, color=PALETTE["dark"], lw=0.8, ls=":", zorder=0)

    ax.plot([m_base], [base[m_base]], "o", ms=6, mfc="none", mec=PALETTE["red_strong"], mew=1.2)
    ax.plot([m_sel], [bal[m_sel]], "o", ms=6, mfc=PALETTE["blue_main"], mec="black", mew=0.8, zorder=5)

    ax.annotate(f"$m$ = {m_sel}: {m_sel}$\\times${s} = {m_sel * s:,} embeddings,\nevery query",
                xy=(m_sel, bal[m_sel]), xytext=(58, 0.905), fontsize=6.5,
                color=PALETTE["blue_main"],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=PALETTE["blue_main"]))
    ax.annotate(f"$m$ = {m_base}: {cands['min']:,}–{cands['max']:,} embeddings,\ndepending on the query",
                xy=(m_base, base[m_base]), xytext=(112, 0.885), fontsize=6.5,
                color=PALETTE["red_strong"],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=PALETTE["red_strong"]))

    ax.set_xlim(0, 205)
    ax.set_ylim(0.845, 0.99)
    ax.set_xlabel("Clusters probed per query, $m$")
    ax.set_ylabel("Recall@10")
    ax.legend(loc="lower right", handlelength=1.8, borderaxespad=0.2)
    save(fig, "probe-selection")


def fig_cluster_width():
    sweep = load("wiki-rag/cluster_size_sweep/cluster_size_sweep.json")
    v = load("wiki-rag/verify_p43/verify_p43.json")
    cands = v["baseline"]["candidates_p100"]

    rows = [dict(label=f"$k$-means\n$|\\mathbf{{C}}|$=4,096, $m$=100",
                 ct=math.ceil(4096 / CENTROIDS_PER_CT), rec=cands["mean"],
                 lo=cands["min"], hi=cands["max"], kind="base")]
    for r in sweep["rows"]:
        rows.append(dict(label=f"$s$={r['cluster_size']}\n$|\\mathbf{{C}}|$={r['nlist']:,}, $m$={r['min_nprobe']}",
                         ct=math.ceil(r["nlist"] / CENTROIDS_PER_CT), rec=r["candidates"],
                         kind="sel" if r["cluster_size"] == 128 else "bal"))
    color = {"base": PALETTE["red_strong"], "bal": PALETTE["blue_secondary"], "sel": PALETTE["blue_main"]}
    alpha = {"base": 1.0, "bal": 0.45, "sel": 1.0}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True,
                                 gridspec_kw=dict(wspace=0.12))
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        kw = dict(color=color[r["kind"]], alpha=alpha[r["kind"]], height=0.62,
                  edgecolor="black", linewidth=0.5)
        a1.barh(yi, r["ct"], **kw)
        a1.text(r["ct"] + 12, yi, f"{r['ct']}", va="center", fontsize=6.5)
        a2.barh(yi, r["rec"], **kw)
        if r["kind"] == "base":
            a2.errorbar(r["rec"], yi, xerr=[[r["rec"] - r["lo"]], [r["hi"] - r["rec"]]],
                        fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
            a2.text(r["hi"] + 250, yi, f"mean {r['rec']:,.0f}\n({r['lo']:,}–{r['hi']:,})", va="center", fontsize=6.5)
        else:
            a2.text(r["rec"] + 150, yi, f"{r['rec']:,}", va="center", fontsize=6.5)

    a1.set_yticks(y)
    a1.set_yticklabels([r["label"] for r in rows], fontsize=6.5)
    a1.tick_params(axis="y", length=0)
    a1.set_xlim(0, 640)
    a2.set_xlim(0, 8600)
    a2.tick_params(axis="y", length=0)
    a1.set_xlabel("FHE stage:\nciphertexts per query")
    a2.set_xlabel("PIR stage:\nembeddings fetched per query")
    a2.set_xticks([0, 2500, 5000, 7500])
    a2.set_xticklabels(["0", "2.5k", "5k", "7.5k"])
    save(fig, "cluster-width")


if __name__ == "__main__":
    fig_probes()
    fig_cluster_width()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
