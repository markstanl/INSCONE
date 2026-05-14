#!/usr/bin/env bash
# run_experiments.sh
# Reproduces three experiments for the SCONE energy MGT detection paper.
#
# Exp 1: INSCONE (lambda=0.005, scone-temporal, pi-aware proximal/distal split)
# Exp 2: Baseline (lambda=0.0, scone-temporal, same wild budget)
# Exp 3: Standard SCONE (lambda=0.005, uniform push, no pi-aware split)
# Exp 4: Fair ablation (lambda=0.0, scone-temporal-ablate, wild budget → train)
#
# Budget multiplier is set once here — applies uniformly to all experiments.
# Double all budgets: set BUDGET_MULT=2.0
set -e
BUDGET_MULT=${BUDGET_MULT:-1.0}
EPOCHS=${EPOCHS:-20}
SEED=${SEED:-42}
DEVICE=${DEVICE:-1}
BASE_ARGS="
  --epochs $EPOCHS
  --seed $SEED
  --devices $DEVICE
  --budget_multiplier $BUDGET_MULT
  --m_in -7.0
  --m_out -2.0
  --alpha_energy 0.001
  --scone_warmup 4
  --wild_ratios 0.1 0.6 0.3
  --buffer 0.1
  --train_id_budget 10000
  --wild_budget 10000
  --eval_budget 1000
  --dev_budget 5000
  --test_budget 5000
  --warmup_steps 500
"
echo "======================================================"
echo " Exp 1: INSCONE (lambda=0.005, scone-temporal)"
echo "======================================================"
python -m scone.scone_energy.train \
  $BASE_ARGS \
  --split scone-temporal \
  --lambda_scone 0.005 \
  --model inscone \
  --checkpoint_name exp1_inscone

echo "======================================================"
echo " Exp 2: Baseline (lambda=0.0, scone-temporal)"
echo "======================================================"
python -m scone.scone_energy.train \
  $BASE_ARGS \
  --split scone-temporal \
  --lambda_scone 0.0 \
  --model inscone \
  --checkpoint_name exp2_baseline

echo "======================================================"
echo " Exp 3: Standard SCONE (lambda=0.005, uniform push)"
echo "======================================================"
python -m scone.scone_energy.train \
  $BASE_ARGS \
  --split scone-temporal \
  --lambda_scone 0.005 \
  --model scone \
  --checkpoint_name exp3_standard_scone

echo "======================================================"
echo " Exp 4: Fair ablation (lambda=0.0, scone-temporal-ablate)"
echo " Wild budget transferred to train_id_budget."
echo "======================================================"
ABLATE_TRAIN=$(python3 -c "print(int(20000 * $BUDGET_MULT))")
python -m scone.scone_energy.train \
  $BASE_ARGS \
  --split scone-temporal-ablate \
  --lambda_scone 0.0 \
  --train_id_budget $ABLATE_TRAIN \
  --wild_budget 1 \
  --checkpoint_name exp4_ablation

echo "======================================================"
echo " All experiments done."
echo "======================================================"