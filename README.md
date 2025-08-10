# TRA-LNN-UAV: Liquid Neural Network Planner (Imitation Learning Edition)

This repository provides a minimal, **fully runnable** codebase to train and evaluate a
**Liquid Neural Network (LNN)** policy for **dynamic, obstacle-aware** UAV path planning
in a 2D grid. It starts with **imitation learning** (behavior cloning) from an A* expert,
and is organized to later plug in PPO/RL or multi-UAV coordination.

> Target: a clean pipeline that produces (1) a trained LNN classifier for 9-way moves,
> (2) evaluation metrics (success rate, path length, energy), and (3) publication-ready figures.

## Scenario (S2-lite for code)
- 2D grid (default 40x40).
- Static rectangular obstacles.
- **Moving obstacles** (patrol between waypoints).
- **No-fly zones** that toggle on/off during the episode.
- **Wind field** (piecewise constant) affecting energy cost.
- Single-UAV in this starter; structure supports extending to multi-UAV.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 1) Generate imitation dataset from A* expert
python dataset/make_dataset.py --n_episodes 500 --seed 42

# 2) Train LNN via behavior cloning
python train/train_bc.py --epochs 20 --batch_size 512 --seed 42

# 3) Evaluate metrics and create figs
python eval/eval_policy.py --policy lnn --n_episodes 200 --seed 123
python viz/plot_paths.py  # saves sample trajectories to results/figs
```

Artifacts are saved into `results/`:
- `bc_lnn.pt` – trained model
- `metrics.json` – success rate, violations, energy, etc.
- `figs/` – bar charts and trajectory overlays

## Repo layout
```
envs/         # Grid env with dynamic obstacles, no-fly toggles, wind, energy model
lnn/          # Liquid layer + policy heads
planners/     # A* expert
dataset/      # Data generation from expert
train/        # Behavior cloning training
eval/         # Evaluation script
viz/          # Plotting utilities (paper figs 1–3)
results/      # Saved models, metrics, and figures
baselines/    # (stub) space for PPO-LSTM or RRT* variants later
```

## Requirements
- Python 3.9+
- PyTorch (CPU ok), NumPy, Matplotlib

## Next steps
- Add PPO variant for fine-tuning.
- Extend to **multi-UAV** (neighbor features + coordination bias).
