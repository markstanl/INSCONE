# Dataset Pipeline — Modern LLM Regeneration

Regenerates RAID and M4 prompts using modern LLMs via LiteLLM.
Outputs are saved as Parquet (full runs) or JSON (debug), and can be
uploaded directly to HuggingFace.

## Setup

```bash
# install dependencies
# IMPORTANT: pin litellm away from compromised versions 1.82.7 and 1.82.8
pip install "litellm>=1.83.0" google-cloud-aiplatform pyarrow huggingface_hub datasets tenacity tqdm

# authenticate with GCP (needed for Gemini / Gemma via Vertex AI)
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project-id

# set API keys for other providers
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
export TOGETHERAI_API_KEY=...
```

## Models

| Model | Provider |
|---|---|
| gemini-3.1-pro | Vertex AI (GCP) |
| gemma-3-27b | Google AI Studio |
| gpt-5.4 | OpenAI |
| claude-sonnet-4-6 | Anthropic |
| deepseek-v3 | DeepSeek |
| llama-3.3-70b | Together.ai |

## Usage

```bash
# debug — 5 prompts, cheapest model, JSON output
python datamodules/pipeline.py --dataset raid --mode debug
python datamodules/pipeline.py --dataset m4   --mode debug

# full run — all prompts, all models, Parquet output
python datamodules/pipeline.py --dataset raid --mode full
python datamodules/pipeline.py --dataset m4   --mode full

# specific models only
python datamodules/pipeline.py --dataset raid --mode full --models gemini-3.1-pro deepseek-v3

# with RAID adversarial attacks
git clone https://github.com/liamdugan/raid.git
export RAID_REPO=/path/to/raid
python datamodules/pipeline.py --dataset raid --mode full --attacks homoglyph whitespace

# upload to HuggingFace after generation
python datamodules/pipeline.py --dataset raid --mode full --hf-repo mabb123/raid-modern-llms
```

## Output

- `outputs/raid_<timestamp>_modern_llms.parquet` — full generation run
- `outputs/raid_<timestamp>_modern_llms_sample.json` — first 20 rows for debugging
- `outputs/debug_raid_<timestamp>.json` — debug mode output

## HuggingFace Dataset

The generated dataset is publicly available at:
[mabb123/raid-modern-llms](https://huggingface.co/datasets/mabb123/raid-modern-llms)