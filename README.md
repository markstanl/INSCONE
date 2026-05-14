# INSCONE: Unknown-Aware Detection of LLM-Generated Text via Informed Wild Data

Implementation and experiments for INSCONE, an informed wild-data energy detector for machine-generated text (MGT) detection with strong zero-shot generalization to unseen LLM families.

**Dataset:** [markstanl/RAID-Plus](https://huggingface.co/datasets/markstanl/RAID-Plus)

---

## Repository Structure

```
.
├── inscone/inscone_energy/   # INSCONE method (main contribution)
├── energy/               # shared encoder, datamodule, test harness
└── raid_plus/          # RAID+ regeneration pipeline
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## INSCONE

### Method

INSCONE adapts the SCONE wild-data framework (Bai et al., 2023, ICML) to the text domain. Rather than uniformly pushing all wild samples toward high energy, INSCONE applies a **proximal anchor** on covariate OOD samples (unseen LLMs) and a **distal anchor** on semantic OOD samples (human text), exploiting known wild batch mixing proportions $(\pi_{id}, \pi_c, \pi_s)$ to avoid the covariate/semantic conflation that degrades standard SCONE in text.

The energy surface is shaped by four losses: contrastive (SimCLR-style), classification (K-class LLM family head), energy margin (explicit in/out margins), and the INSCONE wild loss (proximal/distal quantile split).

### Reproducing Main Results

```bash
# main experiments: INSCONE vs Baseline vs Standard SCONE vs Fair Ablation
bash inscone/inscone_energy/run_exp.sh 2>&1 | tee logs/run_exp.log

# hyperparameter ablation (3 seeds, wild_ratios x buffer sweep)
bash inscone/inscone_energy/run_ablation.sh 2>&1 | tee logs/run_ablation.log

# filter tqdm noise from logs
grep -v "it/s]" logs/run_exp.log > logs/run_exp_clean.log
```

### Main Results

Evaluated on RAID (scone-temporal split, 10k train, 10k wild). Epoch selected by `mean(id_retention AUROC, wild_mem AUROC)` — a principled criterion that does not touch the zero-shot pillar.

| Method | OVERALL AUROC ↑ | Wild FPR95 ↓ | Zero-Shot FPR95 ↓ |
|---|---|---|---|---|
| Standard SCONE | 0.9298 | 0.4702 | 0.4486 |
| Fair Ablation (20k labeled) | 0.9371 | 0.1552 | 0.3166 |
| Baseline (10k) | 0.9520 | 0.1646 | 0.2862 |
| **INSCONE (ours)** | **0.9506** | 0.1764 | **0.2252** |

INSCONE achieves the best zero-shot FPR95, improving **6.1 points** over baseline and **20.3 points** over standard SCONE, at near-identical AUROC.


## RAID-Plus Dataset

RAID-Plus regenerates RAID prompts using frontier models absent from the original benchmark, providing an evaluation set for testing detector behavior against contemporary LLMs.

| Model | Provider | Samples |
|---|---|---|
| Gemini-3.1-Pro | Vertex AI | 2,000 |
| DeepSeek-V3 | DeepSeek | 2,000 |
| Gemma-3-27B | Vertex AI | 2,000 |
| LLaMA-3.3-70B | Together.ai | 2,000 |

Dataset available at [markstanl/RAID-Plus](https://huggingface.co/datasets/markstanl/RAID-Plus).


## Citation

```bibtex
@misc{stanley2025inscone,
  title={INSCONE: Unknown-Aware Detection of LLM-Generated Text via Informed Wild Data},
  author={Stanley, Mark and Syed, Samad and Abboud, Masa and Khatoon, Saira and Khan, Fairoz},
  year={2025},
  url={https://github.com/markstanl/INSCONE}
}
```