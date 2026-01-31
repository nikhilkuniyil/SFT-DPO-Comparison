"""
Script: Train DPO (Direct Preference Optimization) model
Trains on UltraFeedback preference pairs using the SFT model as starting point
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_from_disk
from peft import PeftModel, LoraConfig, get_peft_model
import wandb
import os
from datetime import datetime

def main():
    print("=" * 70)
    print("DPO Training with LoRA")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize Weights & Biases
    wandb.init(
        project="sft-dpo-comparison",
        name="dpo-training",
        config={
            "model": "mistral-7b",
            "method": "dpo",
            "base_model": "sft",
            "learning_rate": 5e-7,
            "beta": 0.1,
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation": 8
        }
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("models/sft")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded\n")
    
    # Load SFT model (trained in previous step)
    print("Loading SFT model as starting point...")
    model = AutoModelForCausalLM.from_pretrained(
        "models/base",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load the SFT LoRA adapter
    model = PeftModel.from_pretrained(model, "models/sft")
    print("✓ SFT model loaded\n")
    
    # Load reference model (same as model, used for KL penalty)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        "models/base",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    ref_model = PeftModel.from_pretrained(ref_model, "models/sft")
    print("✓ Reference model loaded\n")
    
    # Load DPO dataset
    print("Loading DPO dataset...")
    dataset = load_from_disk("data/dpo_data")
    print(f"✓ Loaded {len(dataset):,} preference pairs\n")
    
    # Show example
    print("Example preference pair:")
    print(f"Prompt: {dataset[0]['prompt'][:100]}...")
    print(f"Chosen: {dataset[0]['chosen'][:100]}...")
    print(f"Rejected: {dataset[0]['rejected'][:100]}...\n")
    
    # DPO Training configuration
    print("Setting up DPO training...")
    dpo_config = DPOConfig(
        output_dir="models/sft_dpo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,      # Effective batch size = 16
        learning_rate=5e-7,                 # Much lower than SFT
        beta=0.1,                           # KL divergence penalty
        max_prompt_length=512,
        max_length=1024,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        optim="adamw_torch",
        report_to="wandb",
        logging_dir="logs/dpo",
        remove_unused_columns=False,
        push_to_hub=False
    )
    
    # Create DPO trainer
    print("Creating DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        beta=0.1
    )
    
    print("=" * 70)
    print("Starting DPO Training")
    print("=" * 70)
    print(f"Total preference pairs: {len(dataset):,}")
    print(f"Epochs: 1")
    print(f"Batch size: 2 × 8 = 16 (effective)")
    print(f"Steps: ~{len(dataset) // 16}")
    print(f"Learning rate: 5e-7 (much lower than SFT)")
    print(f"Beta (KL penalty): 0.1")
    print(f"Estimated time: ~4 hours")
    print("=" * 70 + "\n")
    
    print("DPO Training Objective:")
    print("  Maximize: log(σ(β * log(π/π_ref) chosen - β * log(π/π_ref) rejected))")
    print("  Where:")
    print("    π = policy model (being trained)")
    print("    π_ref = reference model (frozen SFT model)")
    print("    β = controls how much we can deviate from reference")
    print("    σ = sigmoid function")
    print("\n")
    
    # Train!
    dpo_trainer.train()
    
    # Save final model
    print("\nSaving model...")
    dpo_trainer.save_model("models/sft_dpo")
    tokenizer.save_pretrained("models/sft_dpo")
    
    # Save metrics
    print("Saving training metrics...")
    metrics = dpo_trainer.state.log_history
    with open("models/sft_dpo/training_metrics.txt", "w") as f:
        f.write(str(metrics))
    
    print("\n" + "=" * 70)
    print("DPO Training Complete!")
    print("=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved to: models/sft_dpo/")
    print(f"\nAll models trained:")
    print(f"  1. Base: models/base/ (no training)")
    print(f"  2. SFT: models/sft/ (supervised fine-tuning)")
    print(f"  3. SFT+DPO: models/sft_dpo/ (preference optimization)")
    print(f"\nNext step: Run scripts/06_evaluate.py")
    print("=" * 70)
    
    wandb.finish()

if __name__ == "__main__":
    main()