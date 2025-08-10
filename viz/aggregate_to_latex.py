# viz/aggregate_to_latex.py
import os, glob, json, numpy as np

def mean_ci(xs):
    xs = np.asarray(xs, dtype=float)
    m = xs.mean()
    if len(xs)<=1: return m, None
    se = xs.std(ddof=1)/np.sqrt(len(xs)); return m, 1.96*se

def collect(pattern):
    rows=[]
    for p in glob.glob(pattern):
        with open(p,'r') as f: rows.append(json.load(f))
    return rows

def main():
    pols = ['lnn','gru','astar_coord','greedy']
    res = {}
    for pol in pols:
        rows = collect(f"results/multi_delivery_metrics_{pol}_*.json") or \
               collect(f"results/multi_delivery_metrics_{pol}.json")
        if not rows: continue
        succ = [r['success_rate'] for r in rows]
        steps= [r['avg_steps'] for r in rows]
        batt = [r['batt_fail_rate'] for r in rows]
        res[pol] = {
            'succ': mean_ci(succ),
            'steps': mean_ci(steps),
            'batt': mean_ci(batt),
            'n': len(rows)
        }
    # emit LaTeX
    pol_map = {'lnn':'LNN','gru':'GRU','astar_coord':'A* (coord.)','greedy':'Greedy'}
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Policy & Success (\%) & Avg. steps & Batt-fail (\%) \\",
             r"\midrule"]
    for pol in ['lnn','gru','astar_coord','greedy']:
        if pol not in res: continue
        s, sc = res[pol]['succ']
        t, tc = res[pol]['steps']
        b, bc = res[pol]['batt']
        def f(m, c, pct=False):
            if pct: m *= 100; c = (c*100) if c is not None else None
            return f"{m:.2f} $\pm$ {c:.2f}" if c is not None else f"{m:.2f}"
        lines.append(f"{pol_map[pol]} & {f(s,sc,True)} & {f(t,tc)} & {f(b,bc,True)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = "\n".join(lines)
    with open('results/metrics_table.tex','w') as f: f.write(out)
    print("Wrote results/metrics_table.tex")

if __name__ == '__main__':
    main()