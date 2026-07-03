#!/bin/bash
# Quick setup and training script for Kaggle TPU environment
# 
# Usage: bash run_training.sh

set -e

echo "🚀 Gemma 4 Quantized Training on Kaggle TPUs"
echo "=============================================="

# Step 1: Setup environment
echo ""
echo "Step 1: Setting up Python environment..."
pip install --quiet -r requirements_gemma4_training.txt

# Step 2: Verify TPU availability
echo ""
echo "Step 2: Verifying TPU availability..."
python << 'EOF'
import jax
print(f"✓ JAX version: {jax.__version__}")
print(f"✓ Available devices: {len(jax.devices())}")
print(f"✓ Device types: {[d.device_kind for d in jax.devices()]}")
EOF

# Step 3: Quick validation
echo ""
echo "Step 3: Running quick validation..."
python examples_quickstart.py

# Step 4: Training with default settings
echo ""
echo "Step 4: Starting training..."
echo "⚠️  Using synthetic data for this demo"
echo ""

python gemma4_quantized_trainer.py \
  --model_size=e2b \
  --quant_type=ternary \
  --batch_size=4 \
  --num_epochs=1 \
  --warmup_steps=100 \
  --checkpoint_dir=/tmp/gemma4_checkpoints \
  --use_synthetic_data=True

echo ""
echo "✓ Training completed!"
echo ""
echo "Next steps:"
echo "1. Update --dataset_path to point to your training data"
echo "2. Adjust --batch_size based on memory availability"
echo "3. Increase --num_epochs for production training"
echo ""
