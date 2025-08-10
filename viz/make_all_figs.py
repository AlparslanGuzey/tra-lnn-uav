import os, subprocess, sys

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RESULTS   = os.path.join(REPO_ROOT, "results")
FIGS      = os.path.join(RESULTS, "figs")

def run(cmd):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def ensure():
    os.makedirs(FIGS, exist_ok=True)
    os.environ["PYTHONPATH"] = REPO_ROOT

    # 1) dataset
    if not os.path.exists(os.path.join(RESULTS, "multi_delivery.npz")):
        run([sys.executable, "-m", "dataset.make_dataset_multi_delivery", "--episodes", "3000", "--seed", "7"])

    # 2) training (curve for Fig 1)
    if not os.path.exists(os.path.join(RESULTS, "train_curve_multi_delivery.csv")):
        run([sys.executable, "-m", "train.train_bc_multi_delivery", "--epochs", "25", "--batch_size", "4096", "--seed", "42"])

    # 3) eval (metrics for Fig 3)
    if not os.path.exists(os.path.join(RESULTS, "multi_delivery_metrics_lnn.json")):
        run([sys.executable, "-m", "eval.eval_multi_delivery", "--policy", "lnn", "--episodes", "500", "--seed", "123"])
    if not os.path.exists(os.path.join(RESULTS, "multi_delivery_metrics_greedy.json")):
        run([sys.executable, "-m", "eval.eval_multi_delivery", "--policy", "greedy", "--episodes", "500", "--seed", "123"])

def make_figs():
    run([sys.executable, "-m", "viz.fig1_convergence"])
    run([sys.executable, "-m", "viz.fig2_scenario_snapshot"])
    run([sys.executable, "-m", "viz.fig3_policy_comparison"])
    print("Saved figures to", FIGS)

if __name__ == "__main__":
    ensure()
    make_figs()
