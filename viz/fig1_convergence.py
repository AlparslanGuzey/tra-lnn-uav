import os, csv, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

# --- publication style ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
})

CURVE_CSV = "results/train_curve_multi_delivery.csv"
OUT_PDF   = "results/figs/fig1_convergence.pdf"
OUT_PNG   = "results/figs/fig1_convergence.png"

def load_curve(path):
    ep, tl, ta, va = [], [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            ep.append(int(row["epoch"]))
            tl.append(float(row["train_loss"]))
            ta.append(float(row["train_acc"]))
            va.append(float(row["val_acc"]))
    return np.array(ep), np.array(tl), np.array(ta), np.array(va)

def main():
    if not os.path.exists(CURVE_CSV):
        raise FileNotFoundError(f"Missing {CURVE_CSV}. Re-run training.")
    os.makedirs("results/figs", exist_ok=True)

    ep, tr_loss, tr_acc, val_acc = load_curve(CURVE_CSV)
    best_i = int(np.argmax(val_acc))
    best_ep, best_val = int(ep[best_i]), float(val_acc[best_i])

    fig, axL = plt.subplots(figsize=(6.4, 4.2))
    axR = axL.twinx()

    # Draw lines with higher z-order so they sit above grid
    axL.set_axisbelow(True)
    h_loss, = axL.plot(ep, tr_loss, lw=2.0, ls="-", label="Train loss", zorder=3)
    h_val,  = axR.plot(ep, val_acc, lw=2.0, ls="--", label="Val. accuracy", zorder=3)

    # Mark best epoch (vertical guide + point)
    axL.axvline(best_ep, lw=1.0, ls=":", zorder=2)
    axR.scatter([best_ep], [best_val], s=28, zorder=4)

    # Labels / limits
    axL.set_xlabel("Epoch")
    axL.set_ylabel("Training loss")
    axR.set_ylabel("Validation accuracy")
    axR.set_ylim(0.0, 1.0)
    axL.grid(True, ls=":", alpha=0.35)
    axL.set_title("LNN Convergence on Multi-UAV Delivery (10×10)")

    # Legend outside (top center)
    handles = [h_loss, h_val]
    labels = ["Train loss", "Val. accuracy"]
    leg = axL.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5, 1.20), ncol=2, frameon=True,
                     borderaxespad=0.6)
    leg.set_zorder(5)

    # Best-epoch annotation: place in axes space to avoid collisions
    # Decide left/right placement based on where best_ep lies
    frac_x = (best_ep - ep.min()) / (ep.max() - ep.min() + 1e-9)
    place_left = frac_x > 0.6  # if marker is toward the right, place text on the left
    x_off = -0.10 if place_left else 0.10
    anchor = (0.15, 0.20) if place_left else (0.75, 0.20)

    txt = f"Best epoch = {best_ep}\nVal. acc = {best_val:.3f}"
    axR.annotate(
        txt,
        xy=(best_ep, best_val), xycoords="data",
        xytext=anchor, textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.8),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9)
    )

    # Give right y-label a little breathing room
    axR.yaxis.set_label_coords(1.08, 0.5)

    # Tighten layout with extra top margin for the legend
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_PDF} and {OUT_PNG}")

if __name__ == "__main__":
    main()
