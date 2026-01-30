#!/bin/bash
# Master script to run the entire SFT vs DPO training pipeline
# Run this on Lambda Labs after setup

set -e  # Exit on any error

echo "======================================================================"
echo "SFT vs DPO Comparison Study - Full Training Pipeline"
echo "======================================================================"
echo "This will take approximately 10-12 hours and cost ~\$10-13"
echo ""
echo "Pipeline steps:"
echo "  1. Download datasets (10 min)"
echo "  2. Preprocess data (5 min)"
echo "  3. Download base model (10 min)"
echo "  4. Train SFT model (6 hours)"
echo "  5. Train DPO model (4 hours)"
echo "  6. Evaluate all models (1 hour)"
echo "  7. Upload results to HF Hub (5 min)"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Pipeline cancelled."
    exit 0
fi

START_TIME=$(date +%s)

echo ""
echo "======================================================================"
echo "Step 1/7: Downloading Datasets"
echo "======================================================================"
python3 scripts/01_download_data.py
if [ $? -ne 0 ]; then
    echo "Error downloading data. Exiting."
    exit 1
fi

echo ""
echo "======================================================================"
echo "Step 2/7: Preprocessing Data"
echo "======================================================================"
python3 scripts/preprocess_data.py
if [ $? -ne 0 ]; then
    echo "Error preprocessing data. Exiting."
    exit 1
fi

echo ""
echo "======================================================================"
echo "Step 3/7: Downloading Base Model"
echo "======================================================================"
python3 scripts/03_download_model.py
if [ $? -ne 0 ]; then
    echo "Error downloading model. Exiting."
    exit 1
fi

echo ""
echo "======================================================================"
echo "Step 4/7: Training SFT Model (~6 hours)"
echo "======================================================================"
echo "This is the longest step. You can safely disconnect and check back later."
echo "Training will continue in the background."
echo ""

# Run in background with logging
nohup python3 scripts/04_train_sft.py > logs/sft_training.log 2>&1 &
SFT_PID=$!
echo "SFT training started (PID: $SFT_PID)"
echo "Monitor progress: tail -f logs/sft_training.log"
echo "Or check WandB: https://wandb.ai"
echo ""
echo "Waiting for SFT training to complete..."

# Wait for SFT to finish
wait $SFT_PID
if [ $? -ne 0 ]; then
    echo "Error in SFT training. Check logs/sft_training.log"
    exit 1
fi
echo "✓ SFT training complete!"

echo ""
echo "======================================================================"
echo "Step 5/7: Training DPO Model (~4 hours)"
echo "======================================================================"

# Run in background with logging
nohup python3 scripts/05_train_dpo.py > logs/dpo_training.log 2>&1 &
DPO_PID=$!
echo "DPO training started (PID: $DPO_PID)"
echo "Monitor progress: tail -f logs/dpo_training.log"
echo ""
echo "Waiting for DPO training to complete..."

# Wait for DPO to finish
wait $DPO_PID
if [ $? -ne 0 ]; then
    echo "Error in DPO training. Check logs/dpo_training.log"
    exit 1
fi
echo "✓ DPO training complete!"

echo ""
echo "======================================================================"
echo "Step 6/7: Evaluating Models"
echo "======================================================================"
python3 scripts/06_evaluate.py
if [ $? -ne 0 ]; then
    echo "Error during evaluation. Exiting."
    exit 1
fi

echo ""
echo "======================================================================"
echo "Step 7/7: Uploading Results to Hugging Face Hub"
echo "======================================================================"
python3 scripts/07_upload_results.py

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "======================================================================"
echo "Pipeline Complete! 🎉"
echo "======================================================================"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved:"
echo "  • models/sft/ - SFT model adapter"
echo "  • models/sft_dpo/ - SFT+DPO model adapter"
echo "  • results/ - Evaluation results"
echo ""
echo "Uploaded to Hugging Face Hub:"
echo "  • Check your HF profile for the uploaded models"
echo ""
echo "Next steps:"
echo "  1. Review results/side_by_side_comparison.txt"
echo "  2. Analyze the differences between models"
echo "  3. Write up your findings"
echo "  4. Terminate Lambda instance to stop billing"
echo ""
echo "======================================================================"