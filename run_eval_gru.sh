#!/usr/bin/env bash
set -e
export PYTHONPATH=$(pwd)
rm -f results/gru_perm.json
python -m eval.eval_multi_delivery --policy gru --episodes 100 --seed 123 --full_perm
python -m eval.eval_multi_delivery --policy gru --episodes 1000 --seed 123
python -m viz.fig3_policy_comparison
