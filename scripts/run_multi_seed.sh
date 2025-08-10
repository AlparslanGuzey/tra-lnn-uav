#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)"

EPISODES=${EPISODES:-3000}
EVAL_EP=${EVAL_EP:-1000}
SEEDS=${SEEDS:-"7 11 13 17 19"}
T=${T:-8}

mkdir -p results/figs

# 1) Base dataset + LNN/GRU training per seed
for S in $SEEDS; do
  echo "== Seed $S: dataset =="
  python -m dataset.make_dataset_multi_delivery --episodes $EPISODES --seed $S --out results/multi_delivery_$S.npz
  python -m dataset.make_dataset_multi_delivery_seq --episodes $EPISODES --seed $S --T $T --out results/multi_delivery_seq_$S.npz

  echo "== Seed $S: train LNN =="
  python -m train.train_bc_multi_delivery --data results/multi_delivery_$S.npz --epochs 25 --batch_size 4096 --seed 42 --out_prefix results/multi_delivery_lnn_seed$S

  echo "== Seed $S: train GRU =="
  python -m train.train_bc_gru_multi --data results/multi_delivery_seq_$S.npz --epochs 25 --batch_size 2048 --seed 42 --out_prefix results/multi_delivery_gru_seed$S

  # copy best weights to canonical names for eval convenience
  cp results/multi_delivery_lnn_seed$S.pt results/multi_delivery_lnn.pt
  cp results/multi_delivery_gru_seed$S.pt results/multi_delivery_gru.pt

  echo "== Seed $S: eval =="
  python -m eval.eval_multi_delivery --policy lnn  --episodes $EVAL_EP --seed 123 --tag seed$S \
    && mv results/multi_delivery_metrics_lnn_seed$S.json results/multi_delivery_metrics_lnn_seed$S.json
  python -m eval.eval_multi_delivery --policy gru  --episodes $EVAL_EP --seed 123 --tag seed$S
  python -m eval.eval_multi_delivery --policy greedy --episodes $EVAL_EP --seed 123 --tag seed$S
  python -m eval.eval_multi_delivery --policy astar_coord --episodes $EVAL_EP --seed 123 --tag seed$S
done

# 2) Robustness sweep over # dynamics (re-uses last seed’s weights)
for ND in 0 2 5 8; do
  for S in $SEEDS; do
    python -m dataset.make_dataset_multi_delivery --episodes 1000 --seed $S --n_dyn $ND --out results/robust_tmp.npz
    # quick eval on env constructed internally (the eval script spawns env fresh each episode, so we store tag only)
    python -m eval.eval_multi_delivery --policy lnn --episodes 500 --seed 123 --tag robust_nDyn${ND}_seed${S} \
      && mv results/multi_delivery_metrics_lnn_robust_nDyn${ND}_seed${S}.json results/robust_nDyn${ND}_metrics_lnn_seed${S}.json
    python -m eval.eval_multi_delivery --policy greedy --episodes 500 --seed 123 --tag robust_nDyn${ND}_seed${S} \
      && mv results/multi_delivery_metrics_greedy_robust_nDyn${ND}_seed${S}.json results/robust_nDyn${ND}_metrics_greedy_seed${S}.json
    python -m eval.eval_multi_delivery --policy astar_coord --episodes 500 --seed 123 --tag robust_nDyn${ND}_seed${S} \
      && mv results/multi_delivery_metrics_astar_coord_robust_nDyn${ND}_seed${S}.json results/robust_nDyn${ND}_metrics_astar_coord_seed${S}.json
    python -m eval.eval_multi_delivery --policy gru --episodes 500 --seed 123 --tag robust_nDyn${ND}_seed${S} \
      && mv results/multi_delivery_metrics_gru_robust_nDyn${ND}_seed${S}.json results/robust_nDyn${ND}_metrics_gru_seed${S}.json
  done
done

# 3) Figures + LaTeX table
python -m viz.fig1_convergence
python -m viz.fig2_scenario_snapshot
python -m viz.fig3_policy_comparison
python -m viz.fig_robustness
python -m viz.aggregate_to_latex

echo "Done. Figures in results/figs/ and table at results/metrics_table.tex"