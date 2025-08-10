# viz/make_tables.py
import os, glob, json, re
import numpy as np
import pandas as pd
import torch

# ---- Optional: import your model classes to count parameters ----
from train.train_bc_multi_delivery import LiquidPolicy5
from train.train_bc_gru_multi import GRUPolicy5

RESULTS_DIR = "results"
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

# ---------- helpers ----------
def fmt_pct(x, digits=1):
    if x is None or np.isnan(x): return "–"
    return f"{100.0*float(x):.{digits}f}"

def bold_best(series, higher_is_better=True, mask_valid=None):
    # returns list of strings, bolding the best non-oracle, non-empty entries
    vals = pd.to_numeric(series, errors="coerce")
    if mask_valid is None:
        mask_valid = ~vals.isna()
    if higher_is_better:
        best_idx = vals[mask_valid].idxmax() if mask_valid.any() else None
    else:
        best_idx = vals[mask_valid].idxmin() if mask_valid.any() else None
    out = []
    for i, s in enumerate(series):
        if i == best_idx and s != "–":
            out.append(f"\\textbf{{{s}}}")
        else:
            out.append(s)
    return out

def read_metrics():
    rows = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "multi_delivery_metrics_*.json")):
        with open(path, "r") as f:
            d = json.load(f)
        # parse policy and optional tag from filename
        # e.g., multi_delivery_metrics_gru.json
        # or    multi_delivery_metrics_gru_baseline.json
        base = os.path.basename(path)
        m = re.match(r"multi_delivery_metrics_(\w+)(?:_(.+))?\.json", base)
        policy = d.get("policy", m.group(1) if m else "unknown")
        tag = m.group(2) if (m and m.group(2)) else ""
        rows.append({
            "file": base,
            "policy": policy,
            "tag": tag,
            "episodes": d.get("episodes", np.nan),
            "success_rate": d.get("success_rate", np.nan),
            "avg_steps": d.get("avg_steps", np.nan),
            "batt_fail_rate": d.get("batt_fail_rate", np.nan)
        })
    return pd.DataFrame(rows).sort_values(["policy","tag"]).reset_index(drop=True)

def policy_pretty(policy, tag):
    name = {
        "lnn": "LNN (ours)",
        "gru": "GRU",
        "greedy": "Greedy heuristic",
        "astar_coord": "A* (coordinated, oracle)"
    }.get(policy, policy)
    if policy == "gru" and tag:
        # if you evaluated with --tag baseline / dagger / etc., reflect it
        name += f" ({tag})"
    return name

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def trainset_sizes():
    sizes = {}
    # BC (single-step) dataset
    p1 = os.path.join(RESULTS_DIR, "multi_delivery.npz")
    if os.path.exists(p1):
        d = np.load(p1)
        sizes["BC_samples"] = len(d["X"])
        sizes["BC_featdim"] = d["X"].shape[1]
    # Seq dataset
    p2 = os.path.join(RESULTS_DIR, "multi_delivery_seq.npz")
    if os.path.exists(p2):
        d = np.load(p2)
        sizes["SEQ_samples"] = len(d["X"])
        sizes["SEQ_T"]       = d["X"].shape[1]
        sizes["SEQ_featdim"] = d["X"].shape[2]
    return sizes

# ---------- Table 1: Overall comparison ----------
def make_table1(metrics_df: pd.DataFrame):
    # choose the latest record for each (policy,tag) combo
    # (if you re-ran same tag multiple times, last write wins naturally)
    if metrics_df.empty:
        print("No metrics files found for Table 1.")
        return
    rows = []
    for _, r in metrics_df.iterrows():
        rows.append({
            "Method": policy_pretty(r["policy"], r["tag"]),
            "Episodes (N)": int(r["episodes"]) if not np.isnan(r["episodes"]) else "–",
            "Success ↑ (%)": fmt_pct(r["success_rate"]),
            "Avg steps ↓": f"{float(r['avg_steps']):.3f}" if not np.isnan(r["avg_steps"]) else "–",
            "Battery fail ↓ (%)": fmt_pct(r["batt_fail_rate"])
        })
    df = pd.DataFrame(rows)
    # sort to a nice order
    order = ["A* (coordinated, oracle)", "LNN (ours)"]
    df["__order"] = df["Method"].apply(lambda x: 0 if x in order else 1)
    df = df.sort_values(["__order","Method"]).drop(columns="__order")

    # Bold best non-oracle for Success and Steps and Batt-fail
    non_oracle_mask = df["Method"] != "A* (coordinated, oracle)"
    # Prepare numeric series
    succ_num = pd.to_numeric(df["Success ↑ (%)"].str.replace("%",""), errors="coerce")
    steps_num = pd.to_numeric(df["Avg steps ↓"], errors="coerce")
    batt_num = pd.to_numeric(df["Battery fail ↓ (%)"].str.replace("%",""), errors="coerce")

    df["Success ↑ (%)"] = bold_best(df["Success ↑ (%)"], True, mask_valid=non_oracle_mask & succ_num.notna())
    df["Avg steps ↓"]   = bold_best(df["Avg steps ↓"],   False, mask_valid=non_oracle_mask & steps_num.notna())
    df["Battery fail ↓ (%)"] = bold_best(df["Battery fail ↓ (%)"], False, mask_valid=non_oracle_mask & batt_num.notna())

    # Save
    df.to_csv(os.path.join(TABLES_DIR, "table1_overall_comparison.csv"), index=False)
    latex = df.to_latex(index=False, escape=False, column_format="lcccc", longtable=False)
    latex = latex.replace("\\toprule", "\\toprule\n\\addlinespace[0.25em]")
    latex = latex.replace("\\bottomrule", "\\addlinespace[0.25em]\n\\bottomrule")
    with open(os.path.join(TABLES_DIR, "table1_overall_comparison.tex"), "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Overall performance on 10×10 multi-UAV delivery. "
                "Bold indicates best non-oracle.}\n\\label{tab:overall}\n" + latex + "\n\\end{table}\n")
    print("Wrote Table 1 → results/tables/table1_overall_comparison.{csv,tex}")
    return df

# ---------- Table 2: If you have multiple seeds/tags, aggregate ----------
def make_table2(metrics_df: pd.DataFrame):
    # Expect files like multi_delivery_metrics_lnn_seedX.json or use --tag to label runs
    if metrics_df.empty:
        return
    has_tags = metrics_df["tag"].astype(str).str.len().gt(0).any()
    if not has_tags:
        print("No tags found; skipping Table 2 (robustness across seeds/tags).")
        return

    grp = metrics_df.groupby(["policy"])
    rows = []
    for pol, g in grp:
        sr = g["success_rate"].dropna().values
        st = g["avg_steps"].dropna().values
        bf = g["batt_fail_rate"].dropna().values
        if len(sr) == 0: continue
        rows.append({
            "Method": policy_pretty(pol, ""),
            "Success ↑ (%)": f"{100*np.mean(sr):.1f} ± {100*np.std(sr, ddof=1):.1f}",
            "Avg steps ↓":    f"{np.mean(st):.2f} ± {np.std(st, ddof=1):.2f}" if len(st)>0 else "–",
            "Battery fail ↓ (%)": f"{100*np.mean(bf):.1f} ± {100*np.std(bf, ddof=1):.1f}" if len(bf)>0 else "–",
            "Runs": len(g)
        })
    if not rows:
        print("No multi-tag data to aggregate; skipping Table 2.")
        return
    df = pd.DataFrame(rows).sort_values("Method")
    df.to_csv(os.path.join(TABLES_DIR, "table2_robustness.csv"), index=False)
    latex = df.to_latex(index=False, escape=False, column_format="lcccc", longtable=False)
    with open(os.path.join(TABLES_DIR, "table2_robustness.tex"), "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Robustness across seeds/tags (mean ± std).}\n"
                "\\label{tab:robustness}\n" + latex + "\n\\end{table}\n")
    print("Wrote Table 2 → results/tables/table2_robustness.{csv,tex}")
    return df

# ---------- Table 3: Model size & dataset ----------
def make_table3():
    # Instantiate with feature dims from saved datasets if possible
    sizes = trainset_sizes()
    # sensible defaults if missing
    featdim_bc  = sizes.get("BC_featdim", 18)     # your env emits 18 feats in multi-delivery
    featdim_seq = sizes.get("SEQ_featdim", 18)

    lnn = LiquidPolicy5(in_dim=featdim_bc)
    gru = GRUPolicy5(in_dim=featdim_seq)

    rows = [{
        "Model": "LNN (ours)",
        "Params (M)": f"{count_params(lnn)/1e6:.3f}",
        "Input dim": featdim_bc,
        "Train set (BC)": f"{sizes.get('BC_samples','–')}",
        "Seq set (T×N)": f"{sizes.get('SEQ_T','–')}×{sizes.get('SEQ_samples','–')}"
    },{
        "Model": "GRU",
        "Params (M)": f"{count_params(gru)/1e6:.3f}",
        "Input dim": featdim_seq,
        "Train set (BC)": f"{sizes.get('BC_samples','–')}",
        "Seq set (T×N)": f"{sizes.get('SEQ_T','–')}×{sizes.get('SEQ_samples','–')}"
    }]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "table3_modelsize_dataset.csv"), index=False)
    latex = df.to_latex(index=False, escape=False, column_format="lcccc", longtable=False)
    with open(os.path.join(TABLES_DIR, "table3_modelsize_dataset.tex"), "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Model size and dataset characteristics.}\n"
                "\\label{tab:modelsize}\n" + latex + "\n\\end{table}\n")
    print("Wrote Table 3 → results/tables/table3_modelsize_dataset.{csv,tex}")
    return df

def main():
    md = read_metrics()
    if md.empty:
        print("No metrics JSONs found under results/. Run eval first.")
        return
    print("Found metrics files:\n", "\n ".join(md["file"].tolist()))
    t1 = make_table1(md)
    t2 = make_table2(md)
    t3 = make_table3()
    # Also dump a merged CSV for archival
    md.to_csv(os.path.join(TABLES_DIR, "raw_metrics_flat.csv"), index=False)
    print("Wrote raw metrics dump → results/tables/raw_metrics_flat.csv")

if __name__ == "__main__":
    main()