#!/usr/bin/env python3
"""
Script 2: Preprocess datasets into training format
- SFT: Format as instruction-response pairs
- DPO: Extract chosen/rejected preference pairs
"""

from datasets import load_from_disk, Dataset
import os
from datetime import datetime

def preprocess_sft():
    """Preprocess Alpaca-GPT4 for SFT training"""
    print("-" * 70)
    print("Preprocessing SFT Dataset")
    print("-" * 70)
    
    # Load raw data
    dataset = load_from_disk("data/raw/sft_alpaca")
    print(f"Loaded {len(dataset):,} raw examples\n")
    
    def format_alpaca(example):
        """
        Convert Alpaca format to training format with instruction template
        """
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]
        
        # Create prompt with instruction template
        if input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        return {
            "text": prompt + output,
            "prompt": prompt,
            "response": output
        }
    
    # Apply formatting
    print("Applying instruction template...")
    processed_dataset = dataset.map(
        format_alpaca,
        remove_columns=dataset.column_names,
        desc="Formatting SFT data"
    )
    
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    processed_dataset.save_to_disk("data/processed/sft_data")
    
    print(f"✓ Processed {len(processed_dataset):,} SFT examples")
    print(f"✓ Saved to: data/processed/sft_data/")
    
    # Show example
    print("\nExample formatted text:")
    print(processed_dataset[0]["text"][:400] + "...\n")
    
    return len(processed_dataset)

def preprocess_dpo():
    """Preprocess UltraFeedback for DPO training"""
    print("-" * 70)
    print("Preprocessing DPO Dataset")
    print("-" * 70)
    
    # Load raw data
    dataset = load_from_disk("data/raw/dpo_ultrafeedback")
    print(f"Loaded {len(dataset):,} raw examples\n")
    
    def extract_preference_pair(example):
        """
        Extract chosen/rejected pairs from UltraFeedback
        Chosen = highest rated response
        Rejected = lowest rated response
        """
        instruction = example["instruction"]
        completions = example["completions"]
        
        if len(completions) < 2:
            return None
        
        # Sort by overall score
        sorted_completions = sorted(
            completions,
            key=lambda x: x.get("overall_score", 0),
            reverse=True
        )
        
        # Get best and worst responses
        chosen = sorted_completions[0]["response"]
        rejected = sorted_completions[-1]["response"]
        
        # Calculate rating difference
        chosen_score = sorted_completions[0].get("overall_score", 0)
        rejected_score = sorted_completions[-1].get("overall_score", 0)
        rating_diff = chosen_score - rejected_score
        
        # Only include pairs with clear quality difference
        if rating_diff < 1.5:
            return None
        
        return {
            "prompt": instruction,
            "chosen": chosen,
            "rejected": rejected,
            "rating_diff": rating_diff
        }
    
    # Apply extraction
    print("Extracting preference pairs...")
    processed_data = []
    valid_count = 0
    
    for example in dataset:
        result = extract_preference_pair(example)
        if result is not None:
            processed_data.append(result)
            valid_count += 1
    
    # Create dataset
    processed_dataset = Dataset.from_list(processed_data)
    
    # Save processed data
    processed_dataset.save_to_disk("data/processed/dpo_data")
    
    print(f"✓ Processed {len(processed_dataset):,} valid preference pairs")
    print(f"✓ Filtered out {len(dataset) - len(processed_dataset):,} low-quality pairs")
    print(f"✓ Saved to: data/processed/dpo_data/")
    
    # Show example
    print("\nExample preference pair:")
    print(f"Prompt: {processed_dataset[0]['prompt'][:150]}...")
    print(f"\nChosen (rating: {processed_dataset[0]['rating_diff']:.1f} higher):")
    print(f"{processed_dataset[0]['chosen'][:150]}...")
    print(f"\nRejected:")
    print(f"{processed_dataset[0]['rejected'][:150]}...\n")
    
    return len(processed_dataset)

def verify_data():
    """Verify preprocessed data is valid"""
    print("-" * 70)
    print("Verifying Processed Data")
    print("-" * 70)
    
    # Check SFT data
    try:
        sft_data = load_from_disk("data/processed/sft_data")
        assert "text" in sft_data.column_names
        assert len(sft_data) > 0
        print(f"✓ SFT data verified: {len(sft_data):,} examples")
    except Exception as e:
        print(f"✗ SFT data verification failed: {e}")
        return False
    
    # Check DPO data
    try:
        dpo_data = load_from_disk("data/processed/dpo_data")
        assert "prompt" in dpo_data.column_names
        assert "chosen" in dpo_data.column_names
        assert "rejected" in dpo_data.column_names
        assert len(dpo_data) > 0
        print(f"✓ DPO data verified: {len(dpo_data):,} preference pairs")
    except Exception as e:
        print(f"✗ DPO data verification failed: {e}")
        return False
    
    print("✓ All data verified successfully!\n")
    return True

def main():
    print("=" * 70)
    print("SFT vs DPO Study - Data Preprocessing")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Preprocess SFT data
    sft_count = preprocess_sft()
    print()
    
    # Preprocess DPO data
    dpo_count = preprocess_dpo()
    print()
    
    # Verify everything
    if not verify_data():
        return False
    
    # Summary
    print("=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"SFT examples ready: {sft_count:,}")
    print(f"DPO preference pairs ready: {dpo_count:,}")
    print(f"\nNext step: Run scripts/03_download_model.py")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)