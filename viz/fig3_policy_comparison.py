# viz/fig3_policy_comparison.py
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# --- publication style ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
})

OUT_PDF = "results/figs/fig3_policy_comparison.pdf"
OUT_PNG = "results/figs/fig3_policy_comparison.png"

POL_ORDER = ["lnn", "gru", "astar_coord", "greedy"]
LABELS = {"lnn": "LNN", "gru": "GRU", "astar_coord": "A* (coord.)", "greedy": "Greedy"}

# Okabe–Ito palette
C_SUCCESS = "#0072B2"  # blue
C_BATT    = "#E69F00"  # orange
C_STEPS   = "#4D4D4D"  # dark gray

def load_many(policy: str):
    pats = [f"results/multi_delivery_metrics_{policy}_*.json",
            f"results/multi_delivery_metrics_{policy}.json"]
    files = sorted(glob.glob(pats[0])) or ([pats[1]] if os.path.exists(pats[1]) else [])
    vals = []
    for p in files:
        try:
            with open(p, "r") as f:
                vals.append(json.load(f))
        except Exception:
            pass
    return vals

def mean_ci(xs, alpha=0.05):
    xs = np.asarray(xs, dtype=float)
    m = float(xs.mean())
    n = len(xs)
    if n <= 1:
        return m, None, n
    sd = xs.std(ddof=1)
    se = sd / np.sqrt(n)
    try:
        from scipy.stats import t as student_t
        tcrit = float(student_t.ppf(1 - alpha / 2, df=n - 1))
    except Exception:
        tcrit = 2.13 if n < 30 else 1.96
    return m, tcrit * se, n

def annotate_bars(ax, bars, fmt="{:.1f}", dy=0.8):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width()/2.0, v + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=8)

def _yerr_with_nan(errs):
    return np.array([np.nan if e is None else float(e) for e in errs], dtype=float)

def main():
    Path("results/figs").mkdir(parents=True, exist_ok=True)

    # Load metrics
    data = {}
    for pol in POL_ORDER:
        arr = load_many(pol)
        if arr:
            data[pol] = {
                "succ": [100 * float(d["success_rate"])   for d in arr],
                "steps": [float(d["avg_steps"])           for d in arr],
                "batt": [100 * float(d["batt_fail_rate"]) for d in arr],
            }
    assert data, "No metrics found. Run evaluation first."

    names = [LABELS[p] for p in data.keys()]
    succ_stats  = [mean_ci(data[p]["succ"])  for p in data.keys()]
    batt_stats  = [mean_ci(data[p]["batt"])  for p in data.keys()]
    steps_stats = [mean_ci(data[p]["steps"]) for p in data.keys()]

    succ_mean  = [m for (m, e, n) in succ_stats]
    succ_err   = [e for (m, e, n) in succ_stats]
    batt_mean  = [m for (m, e, n) in batt_stats]
    batt_err   = [e for (m, e, n) in batt_stats]
    steps_mean = [m for (m, e, n) in steps_stats]
    steps_err  = [e for (m, e, n) in steps_stats]
    nseeds     = [n for (_, _, n) in succ_stats]

    x = np.arange(len(names))
    w = 0.34

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.6, 5.4), sharex=True, gridspec_kw={"hspace": 0.10}
    )

    # ---------- Top panel ----------
    b1 = ax_top.bar(x - w/2, succ_mean, width=w, color=C_SUCCESS, edgecolor="black",
                    linewidth=0.4, label="Success %", zorder=3)

    batt_draw = []
    hatch_idx = []
    stub = 0.6
    for i, v in enumerate(batt_mean):
        if np.isclose(v, 0.0):
            batt_draw.append(stub); hatch_idx.append(i)
        else:
            batt_draw.append(v)

    b2 = ax_top.bar(x + w/2, batt_draw, width=w, color=C_BATT, edgecolor="black",
                    linewidth=0.4, label="Batt-fail %", zorder=3)
    for i in hatch_idx:
        b2[i].set_hatch("..")
        b2[i].set_alpha(0.8)

    # Error bars
    ax_top.errorbar(x - w/2, succ_mean, yerr=_yerr_with_nan(succ_err), fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=3, zorder=4)
    ax_top.errorbar(x + w/2, batt_mean, yerr=_yerr_with_nan(batt_err), fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=3, zorder=4)

    annotate_bars(ax_top, b1, fmt="{:.1f}", dy=0.8)
    for i, rect in enumerate(b2):
        ax_top.text(rect.get_x() + rect.get_width()/2.0, rect.get_height() + 0.8,
                    f"{batt_mean[i]:.1f}", ha="center", va="bottom", fontsize=8)

    ax_top.set_ylabel("Percentage (%)")
    ax_top.set_ylim(0, max(max(succ_mean), max(batt_mean)) * 1.30)
    ax_top.grid(axis="y", ls=":", alpha=0.3, zorder=0)
    ax_top.legend(loc="upper right", frameon=True, fancybox=True, facecolor="white")
    ax_top.set_title("Policy comparison")

    # ---------- Bottom panel ----------
    b3 = ax_bot.bar(x, steps_mean, width=0.45, color=C_STEPS, edgecolor="black",
                    linewidth=0.4, label="Avg. steps", zorder=3)
    ax_bot.errorbar(x, steps_mean, yerr=_yerr_with_nan(steps_err), fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=3, zorder=4)

    ax_bot.set_ylabel("Steps")
    ax_bot.set_ylim(0, max(steps_mean) * 1.15)
    ax_bot.grid(axis="y", ls=":", alpha=0.3, zorder=0)
    for rect in b3:
        ax_bot.text(rect.get_x() + rect.get_width()/2.0, rect.get_height() + 0.6,
                    f"{rect.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax_bot.set_xticks(x, names)
    seeds_note = " / ".join([f"{LABELS[p]}: N={n}" for p, n in zip(data.keys(), nseeds)])


    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_PDF} and {OUT_PNG}")

if __name__ == "__main__":
    main()