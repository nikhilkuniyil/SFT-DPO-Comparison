"""
Script: Train SFT (Supervised Fine-Tuning) model using LoRA
Trains on Alpaca-GPT4 instruction-following dataset
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import wandb
import os
from datetime import datetime

def format_instruction(example):
    """Format data for instruction tuning"""
    return {"text": example["text"]}

def main():
    print("=" * 70)
    print("SFT Training with LoRA")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize Weights & Biases
    wandb.init(
        project="sft-dpo-comparison",
        name="sft-training",
        config={
            "model": "mistral-7b",
            "method": "sft",
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-5,
            "epochs": 3,
            "batch_size": 4,
            "gradient_accumulation": 4
        }
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("models/base")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer loaded\n")
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "models/base",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ Base model loaded\n")
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,                              # Rank
        lora_alpha=32,                     # Alpha scaling
        target_modules=[                   # Which layers to apply LoRA
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj"
        ],
        lora_dropout=0.05,                 # Dropout for regularization
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA configured")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}\n")
    
    # Load dataset
    print("Loading SFT dataset...")
    dataset = load_from_disk("data/sft_data")
    print(f"✓ Loaded {len(dataset):,} training examples\n")
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    print("✓ Dataset tokenized\n")
    
    # Training arguments
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir="models/sft",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,      # Effective batch size = 16
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,                 # Only keep 2 checkpoints
        fp16=True,                          # Mixed precision training
        optim="adamw_torch",
        report_to="wandb",
        logging_dir="logs/sft",
        push_to_hub=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print("=" * 70)
    print("Starting SFT Training")
    print("=" * 70)
    print(f"Total examples: {len(tokenized_dataset):,}")
    print(f"Epochs: 3")
    print(f"Batch size: 4 × 4 = 16 (effective)")
    print(f"Steps per epoch: ~{len(tokenized_dataset) // 16}")
    print(f"Total steps: ~{3 * (len(tokenized_dataset) // 16)}")
    print(f"Estimated time: ~6 hours")
    print("=" * 70 + "\n")
    
    # Train!
    trainer.train()
    
    # Save final model
    print("\nSaving model...")
    trainer.save_model("models/sft")
    tokenizer.save_pretrained("models/sft")
    
    # Save metrics
    print("Saving training metrics...")
    metrics = trainer.state.log_history
    with open("models/sft/training_metrics.txt", "w") as f:
        f.write(str(metrics))
    
    print("\n" + "=" * 70)
    print("SFT Training Complete!")
    print("=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved to: models/sft/")
    print(f"Next step: Run scripts/05_train_dpo.py")
    print("=" * 70)
    
    wandb.finish()
