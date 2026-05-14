#!/usr/bin/env bash
# run_ablation.sh
# Ablation sweep over wild_ratios and buffer for INSCONE.
# Runs 3 seeds per configuration.
#
# Configurations:
#   Baseline
#   (0.1, 0.6, 0.3) buffer=0.0
#   (0.1, 0.6, 0.3) buffer=0.1  ← best config
#   (0.1, 0.6, 0.3) buffer=0.2
#   (0.0, 0.7, 0.3) buffer=0.0
#   (0.0, 1.0, 0.0) buffer=0.0

set -e
BUDGET_MULT=${BUDGET_MULT:-1.0}
EPOCHS=${EPOCHS:-20}
DEVICE=${DEVICE:-1}

BASE_ARGS="
  --epochs $EPOCHS
  --devices $DEVICE
  --budget_multiplier $BUDGET_MULT
  --m_in -7.0
  --m_out -2.0
  --alpha_energy 0.001
  --scone_warmup 4
  --train_id_budget 10000
  --wild_budget 10000
  --eval_budget 1000
  --dev_budget 5000
  --test_budget 5000
  --warmup_steps 500
  --lambda_scone 0.005
  --model inscone
  --split scone-temporal
  --silent
"

for SEED in 42 43 44; do

  echo "======================================================"
  echo " Baseline (lambda=0.0) seed=$SEED"
  echo "======================================================"
  python -m inscone.inscone_energy.train \
    $BASE_ARGS \
    --lambda_scone 0.0 \
    --model inscone \
    --wild_ratios 0.0 0.7 0.3 \
    --buffer 0.0 \
    --seed $SEED \
    --checkpoint_name ablation_baseline_s${SEED}

  echo "======================================================"
  echo " (0.1, 0.6, 0.3) buffer=0.0 seed=$SEED"
  echo "======================================================"
  python -m inscone.inscone_energy.train \
    $BASE_ARGS \
    --wild_ratios 0.1 0.6 0.3 \
    --buffer 0.0 \
    --seed $SEED \
    --checkpoint_name ablation_063_b0_s${SEED}

  echo "======================================================"
  echo " (0.1, 0.6, 0.3) buffer=0.1 seed=$SEED"
  echo "======================================================"
  python -m inscone.inscone_energy.train \
    $BASE_ARGS \
    --wild_ratios 0.1 0.6 0.3 \
    --buffer 0.1 \
    --seed $SEED \
    --checkpoint_name ablation_063_b1_s${SEED}

  echo "======================================================"
  echo " (0.1, 0.6, 0.3) buffer=0.2 seed=$SEED"
  echo "======================================================"
  python -m inscone.inscone_energy.train \
    $BASE_ARGS \
    --wild_ratios 0.1 0.6 0.3 \
    --buffer 0.2 \
    --seed $SEED \
    --checkpoint_name ablation_063_b2_s${SEED}

  echo "======================================================"
  echo " (0.0, 0.7, 0.3) buffer=0.0 seed=$SEED"
  echo "======================================================"
  python -m inscone.inscone_energy.train \
    $BASE_ARGS \
    --wild_ratios 0.0 0.7 0.3 \
    --buffer 0.0 \
    --seed $SEED \
    --checkpoint_name ablation_073_b0_s${SEED}

  echo "======================================================"
  echo " (0.0, 1.0, 0.0) buffer=0.0 seed=$SEED"
  echo "======================================================"
  python -m inscone.inscone_energy.train \
    $BASE_ARGS \
    --wild_ratios 0.0 1.0 0.0 \
    --buffer 0.0 \
    --seed $SEED \
    --checkpoint_name ablation_100_b0_s${SEED}

done

echo "======================================================"
echo " Ablation done."
echo "======================================================"