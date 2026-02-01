# SFT vs DPO Comparison Study

A comparison study of Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) for instruction-following language models using Mistral-7B.

## Overview

This project trains and evaluates three model variants:

1. **Base Model** - Mistral-7B-v0.1 (no fine-tuning)
2. **SFT Model** - Base + Supervised Fine-Tuning on instruction data
3. **SFT+DPO Model** - SFT model + Direct Preference Optimization on preference pairs

## Training Methods

### Supervised Fine-Tuning (SFT)
- Trains the model to follow instructions using (instruction, response) pairs
- Dataset: Alpaca-GPT4 (~10K examples)
- LoRA configuration: r=16, alpha=32
- Learning rate: 2e-5
- Epochs: 3

### Direct Preference Optimization (DPO)
- Trains the model to prefer "chosen" responses over "rejected" ones
- Dataset: UltraFeedback preference pairs
- Starts from the SFT model checkpoint
- Learning rate: 5e-7 (much lower than SFT)
- Beta (KL penalty): 0.1
- Epochs: 1

## Project Structure

```
.
├── scripts/
│   ├── train_sft.py         # SFT training with LoRA
│   ├── train_dpo.py         # DPO training on top of SFT
│   ├── evaluate.py          # Evaluate all model variants
│   ├── upload_results.py    # Upload models to Hugging Face Hub
│   └── run_full_pipeline.sh # End-to-end training pipeline
├── configs/                  # Training configurations
├── results/                  # Evaluation outputs
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (A100 recommended)
- ~40GB GPU memory for training

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Full Pipeline
```bash
bash scripts/run_full_pipeline.sh
```

### Run Individual Steps

**Train SFT model:**
```bash
python scripts/train_sft.py
```

**Train DPO model (requires SFT model):**
```bash
python scripts/train_dpo.py
```

**Evaluate all models:**
```bash
python scripts/evaluate.py
```

**Upload to Hugging Face Hub:**
```bash
python scripts/upload_results.py
```

## Evaluation

The evaluation script tests models on diverse prompts across categories:
- Instruction following
- Reasoning
- Creative writing
- Factual knowledge
- Safety (refusal behavior)
- Coding

Results are saved to:
- `results/evaluation_results.json` - Full JSON results
- `results/side_by_side_comparison.txt` - Human-readable comparison

## Infrastructure

Designed to run on Lambda Labs GPU instances. Estimated costs:
- Training time: ~10-12 hours
- Cost: ~$10-13 on a single A100
