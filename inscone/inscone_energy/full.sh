#!/usr/bin/env bash
# run_inscone_full.sh
# Single INSCONE experiment using full dataset via FullEnergyRAIDDataModule.
# lambda_scone=0.01 accounts for ~10:1 train:wild ratio vs 0.005 in budgeted runs.
set -e

EPOCHS=${EPOCHS:-20}
SEED=${SEED:-42}
DEVICE=${DEVICE:-1}

python -m inscone.inscone_energy.train_full \
  --epochs $EPOCHS \
  --seed $SEED \
  --devices $DEVICE \
  --split scone-temporal \
  --model inscone \
  --lambda_scone 0.01 \
  --m_in -7.0 \
  --m_out -2.0 \
  --alpha_energy 0.001 \
  --scone_warmup 4 \
  --wild_ratios 0.1 0.6 0.3 \
  --wild_size 20000 \
  --test_pool_cap 10000 \
  --buffer 0.1 \
  --train_ratio 0.90 \
  --val_ratio 0.05 \
  --test_ratio 0.05 \
  --warmup_steps 500 \
  --checkpoint_name inscone_full

echo "done."
