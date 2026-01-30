"""
Script: Download Mistral-7B base model from Hugging Face
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datetime import datetime

def main():
    print("=" * 70)
    print("SFT vs DPO Study - Download Base Model")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create model directory
    os.makedirs("models/base", exist_ok=True)
    
    # Model info
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"Model: {model_name}")
    print(f"Size: ~14GB")
    print(f"License: Apache 2.0\n")
    
    print("-" * 70)
    print("Downloading Model")
    print("-" * 70)
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained("models/base")
        print("✓ Tokenizer downloaded")
        
        # Download model
        print("\nDownloading model (this may take 5-10 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model.save_pretrained("models/base")
        print("✓ Model downloaded")
        
        # Verify
        print("\n" + "-" * 70)
        print("Verifying Model")
        print("-" * 70)
        
        # Check files exist
        base_files = os.listdir("models/base")
        required_files = ["config.json", "tokenizer.json"]
        
        for file in required_files:
            if file in base_files:
                print(f"✓ Found {file}")
            else:
                print(f"✗ Missing {file}")
                return False
        
        # Check model size
        total_size = sum(
            os.path.getsize(os.path.join("models/base", f))
            for f in base_files
            if os.path.isfile(os.path.join("models/base", f))
        )
        size_gb = total_size / (1024**3)
        print(f"✓ Total size: {size_gb:.2f} GB")
        
        # Test inference
        print("\n" + "-" * 70)
        print("Testing Model Inference")
        print("-" * 70)
        
        test_prompt = "Once upon a time"
        print(f"Test prompt: '{test_prompt}'")
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nGenerated: {generated_text}")
        print("\n✓ Model inference working!")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Location: models/base/")
    print(f"Size: {size_gb:.2f} GB")
    print(f"\nNext step: Run scripts/train_sft.py")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)