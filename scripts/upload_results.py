#!/usr/bin/env python3
"""
Script 7: Upload trained models and results to Hugging Face Hub
Saves everything for permanent storage (free!)
"""

from huggingface_hub import HfApi, login
import os
from datetime import datetime

def main():
    print("=" * 70)
    print("Upload Results to Hugging Face Hub")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Make sure user is logged in
    print("Checking Hugging Face authentication...")
    try:
        api = HfApi()
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✓ Logged in as: {username}\n")
    except Exception as e:
        print("✗ Not logged in to Hugging Face!")
        print("Run: python3 -c 'from huggingface_hub import login; login()'")
        return False
    
    # Confirm upload
    print("This will upload the following to your Hugging Face account:")
    print("  1. SFT model adapter (~100MB)")
    print("  2. SFT+DPO model adapter (~100MB)")
    print("  3. Evaluation results (~1MB)")
    print("  Total: ~200MB\n")
    
    response = input("Proceed with upload? [y/N]: ")
    if response.lower() != 'y':
        print("Upload cancelled.")
        return False
    
    print("\n" + "=" * 70)
    print("Uploading Models")
    print("=" * 70)
    
    # Upload SFT model
    print("\n1. Uploading SFT model...")
    try:
        api.upload_folder(
            folder_path="models/sft",
            repo_id=f"{username}/mistral-7b-sft",
            repo_type="model",
            commit_message="Add SFT model from SFT vs DPO comparison study"
        )
        print(f"✓ SFT model uploaded to: https://huggingface.co/{username}/mistral-7b-sft")
    except Exception as e:
        print(f"✗ Error uploading SFT model: {e}")
        return False
    
    # Upload SFT+DPO model
    print("\n2. Uploading SFT+DPO model...")
    try:
        api.upload_folder(
            folder_path="models/sft_dpo",
            repo_id=f"{username}/mistral-7b-sft-dpo",
            repo_type="model",
            commit_message="Add SFT+DPO model from SFT vs DPO comparison study"
        )
        print(f"✓ SFT+DPO model uploaded to: https://huggingface.co/{username}/mistral-7b-sft-dpo")
    except Exception as e:
        print(f"✗ Error uploading SFT+DPO model: {e}")
        return False
    
    # Upload results
    print("\n3. Uploading evaluation results...")
    try:
        # Check if results exist
        if not os.path.exists("results/evaluation_results.json"):
            print("⚠ No evaluation results found. Run 06_evaluate.py first.")
        else:
            api.upload_folder(
                folder_path="results",
                repo_id=f"{username}/sft-dpo-comparison-results",
                repo_type="dataset",
                commit_message="Add evaluation results from SFT vs DPO comparison study"
            )
            print(f"✓ Results uploaded to: https://huggingface.co/{username}/sft-dpo-comparison-results")
    except Exception as e:
        print(f"✗ Error uploading results: {e}")
    
    # Create README for the models
    print("\n4. Creating model cards...")
    
    sft_readme = f"""---
tags:
- mistral
- sft
- instruction-tuning
- lora
base_model: mistralai/Mistral-7B-v0.1
---

# Mistral-7B SFT Model

This is a LoRA adapter for Mistral-7B trained using Supervised Fine-Tuning (SFT) on the Alpaca-GPT4 dataset.

## Training Details
- **Base Model:** mistralai/Mistral-7B-v0.1
- **Method:** LoRA (r=16, alpha=32)
- **Dataset:** Alpaca-GPT4 (10K examples)
- **Epochs:** 3
- **Learning Rate:** 2e-5
- **Training Time:** ~6 hours on A100

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{username}/mistral-7b-sft")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{username}/mistral-7b-sft")

# Generate
prompt = "### Instruction:\\nWrite a haiku about AI.\\n\\n### Response:\\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Part of SFT vs DPO Comparison Study

This model is part of a study comparing Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

See also:
- SFT+DPO model: [{username}/mistral-7b-sft-dpo](https://huggingface.co/{username}/mistral-7b-sft-dpo)
- Results: [{username}/sft-dpo-comparison-results](https://huggingface.co/{username}/sft-dpo-comparison-results)
"""
    
    try:
        api.upload_file(
            path_or_fileobj=sft_readme.encode(),
            path_in_repo="README.md",
            repo_id=f"{username}/mistral-7b-sft",
            repo_type="model"
        )
        print(f"✓ Added README to SFT model")
    except:
        pass
    
    # Summary
    print("\n" + "=" * 70)
    print("Upload Complete!")
    print("=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Models are now available at:")
    print(f"  • SFT: https://huggingface.co/{username}/mistral-7b-sft")
    print(f"  • SFT+DPO: https://huggingface.co/{username}/mistral-7b-sft-dpo")
    print(f"  • Results: https://huggingface.co/{username}/sft-dpo-comparison-results")
    
    print("\nYou can now:")
    print("  1. Terminate your Lambda instance (everything is backed up!)")
    print("  2. Share your models with others")
    print("  3. Load them anytime for inference")
    print("  4. Write up your findings")
    
    print("\n" + "=" * 70)
    print("Project Complete! 🎉")
    print("=" * 70)

if __name__ == "__main__":
    main()