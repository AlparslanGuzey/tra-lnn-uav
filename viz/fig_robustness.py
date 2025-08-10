# viz/fig_robustness.py
import os, json, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "savefig.dpi": 300,
})

OUT_PDF = 'results/figs/fig_robustness_dyn.pdf'
OUT_PNG = 'results/figs/fig_robustness_dyn.png'

def load_metric(path):
    with open(path,'r') as f: return json.load(f)

def main():
    os.makedirs('results/figs', exist_ok=True)
    # Expect files saved by the multi-seed runner, e.g.:
    # results/robust_nDyn{n}_metrics_{policy}_seed{S}.json
    policies = ['lnn','gru','astar_coord','greedy']
    n_dyn_vals = sorted({int(fn.split('robust_nDyn')[-1].split('_')[0])
                         for fn in os.listdir('results') if fn.startswith('robust_nDyn')})
    assert n_dyn_vals, "No robustness files found."

    fig, ax = plt.subplots(figsize=(6.2,4.0))
    for pol in policies:
        means, cies = [], []
        for nd in n_dyn_vals:
            files = [f for f in os.listdir('results')
                     if f.startswith(f'robust_nDyn{nd}_metrics_{pol}') and f.endswith('.json')]
            if not files: continue
            xs = [load_metric(os.path.join('results', f))['success_rate'] for f in files]
            xs = np.array(xs, dtype=float)
            mean = xs.mean(); ci = 1.96*xs.std(ddof=1)/np.sqrt(len(xs)) if len(xs)>1 else None
            means.append(mean); cies.append(ci)
        if not means: continue
        ax.plot(n_dyn_vals[:len(means)], np.array(means)*100, lw=2.0, label=pol.upper())
        if all(c is not None for c in cies):
            ax.fill_between(n_dyn_vals[:len(means)],
                            (np.array(means)-np.array(cies))*100,
                            (np.array(means)+np.array(cies))*100,
                            alpha=0.15)

    ax.set_xlabel('# Dynamic obstacles'); ax.set_ylabel('Success (%)')
    ax.set_title('Robustness vs Dynamic Obstacle Density')
    ax.grid(True, ls=':', alpha=0.35); ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_PDF} and {OUT_PNG}")

if __name__ == '__main__':
    main()