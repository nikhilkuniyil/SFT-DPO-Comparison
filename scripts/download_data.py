"""
Script: Download SFT and DPO datasets from Hugging Face Hub
"""

from datasets import load_dataset
import os
from datetime import datetime

def main():
    print("=" * 70)
    print("SFT vs DPO Study - Dataset Download")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    print("✓ Created data directories\n")
    
    # Download SFT dataset (Alpaca-GPT4)
    print("-" * 70)
    print("Downloading SFT Dataset: Alpaca-GPT4")
    print("-" * 70)
    print("Source: vicgalle/alpaca-gpt4")
    print("Size: 10,000 instruction-response pairs")
    print("Purpose: Supervised Fine-Tuning\n")
    
    try:
        sft_dataset = load_dataset(
            "vicgalle/alpaca-gpt4",
            split="train[:10000]"
        )
        sft_dataset.save_to_disk("data/raw/sft_alpaca")
        print(f"✓ Successfully downloaded {len(sft_dataset):,} SFT examples")
        print(f"✓ Saved to: data/raw/sft_alpaca/")
        
        # Show example
        print("\nExample SFT entry:")
        print(f"  Instruction: {sft_dataset[0]['instruction'][:100]}...")
        print(f"  Output: {sft_dataset[0]['output'][:100]}...")
        
    except Exception as e:
        print(f"✗ Error downloading SFT dataset: {e}")
        return False
    
    print("\n")
    
    # Download DPO dataset (UltraFeedback)
    print("-" * 70)
    print("Downloading DPO Dataset: UltraFeedback")
    print("-" * 70)
    print("Source: openbmb/UltraFeedback")
    print("Size: 5,000 preference pairs")
    print("Purpose: Direct Preference Optimization\n")
    
    try:
        dpo_dataset = load_dataset(
            "openbmb/UltraFeedback",
            split="train[:5000]"
        )
        dpo_dataset.save_to_disk("data/raw/dpo_ultrafeedback")
        print(f"✓ Successfully downloaded {len(dpo_dataset):,} DPO examples")
        print(f"✓ Saved to: data/raw/dpo_ultrafeedback/")
        
        # Show example
        print("\nExample DPO entry:")
        print(f"  Instruction: {dpo_dataset[0]['instruction'][:100]}...")
        print(f"  Completions: {len(dpo_dataset[0]['completions'])} responses with ratings")
        
    except Exception as e:
        print(f"✗ Error downloading DPO dataset: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print(f"SFT examples: {len(sft_dataset):,}")
    print(f"DPO examples: {len(dpo_dataset):,}")
    print(f"Total storage: ~5GB")
    print(f"\nNext step: Run scripts/02_preprocess_data.py")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)