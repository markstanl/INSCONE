# OOD Generalization (INSCONE)

This directory contains all code related to the INSCONE approach described in our paper: *INSCONE: Unknown-Aware Detection of LLM-Generated Text via Informed Wild Data*.

## Method Overview

INSCONE adapts the SCONE wild-data framework to the text domain. Rather than uniformly pushing all wild samples toward high energy, INSCONE applies a proximal anchor on covariate OOD samples (unseen LLMs) and a distal anchor on semantic OOD samples (human text), exploiting known wild batch mixing proportions $(\pi_{id}, \pi_c, \pi_s)$ to avoid the covariate/semantic conflation that degrades standard SCONE in text.

## Replication

Set up a Python environment (see `environment.yml`) and run:

```bash
bash scone/scone_energy/run_exp.sh
```

For the hyperparameter ablation over wild ratios and buffer size:

```bash
bash scone/scone_energy/run_ablation.sh
```

## Main Results

Evaluated on the RAID benchmark with a scone-temporal split (10,000 training samples, 10,000 wild samples).

| Method         | AUROC  | Wild FPR95 | Zero-Shot FPR95 |
|----------------|--------|------------|-----------------|
| Baseline       | 0.9577 | 0.1646     | 0.2862          |
| Baseline (fair)| 0.9571 | 0.1552     | 0.3166          |
| SCONE          | 0.9164 | 0.3848     | 0.4270          |
| INSCONE (ours) | 0.9568 | 0.1764     | **0.2252**      |

INSCONE achieves the best zero-shot FPR95, improving 6.1 points over the naive baseline and 9.1 points over standard SCONE.

## Ablation

Sweep over wild ratios and buffer size (3 seeds, small-scale):

| Wild Ratios $(\pi)$ | Buffer $\delta$ | AUROC  | Wild FPR95 | ZS FPR95 |
|---------------------|-----------------|--------|------------|----------|
| Baseline (no wild)  |                 | 0.9175 | 0.6094     | 0.5860   |
| Baseline (fair)     |                 | 0.9224 | 0.1942     | 0.4228   |
| (0.1, 0.6, 0.3)     | 0.0             | 0.9345 | 0.2265     | 0.3166   |
| (0.1, 0.6, 0.3)     | **0.1**         | **0.9400** | **0.1827** | **0.2991** |
| (0.1, 0.6, 0.3)     | 0.2             | 0.9406 | 0.2506     | 0.3802   |
| (0.0, 0.7, 0.3)     | 0.0             | 0.9316 | 0.2440     | 0.3188   |
| (0.0, 1.0, 0.0)     | 0.0             | 0.9116 | 0.3337     | 0.4142   |

Best configuration: $\pi = (0.1, 0.6, 0.3)$, $\delta = 0.1$.

## Key Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Encoder | princeton-nlp/unsup-simcse-roberta-base |
| $m_{in}$ | -7.0 |
| $m_{out}$ | -2.0 |
| $\lambda_{INSCONE}$ | 0.005 |
| Buffer $\delta$ | 0.1 |
| Wild ratios $\pi$ | (0.1, 0.6, 0.3) |
| SCONE warmup | 4 epochs |
| Optimizer | AdamW |