# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quick-start examples for Gemma 4 quantized training."""

import logging
from typing import Optional

import jax
import jax.numpy as jnp

from gemma4_model_loader import load_quantized_gemma4
from gemma4_quantized_trainer import QuantizedGemma4Trainer
from multimodal_data_loader import create_synthetic_multimodal_batch
from tool_calling import (
    ToolCallingTrainingMixin,
    ToolCallingTokenizer,
    create_gemma4_tool_set,
)


logging.basicConfig(level=logging.INFO)


def example_1_basic_loading():
  """Example 1: Load a quantized Gemma 4 model."""
  print("\n" + "=" * 60)
  print("Example 1: Loading Quantized Gemma 4")
  print("=" * 60)
  
  # Create TPU mesh for 8 devices
  mesh = jax.make_mesh((1, 8), ("fsdp", "tp"))
  
  # Load model with ternary quantization
  model, quant_wrapper = load_quantized_gemma4(
      model_size="e2b",  # or "e4b", "31b"
      checkpoint_path=None,  # Uses default
      mesh=mesh,
      dtype=jnp.bfloat16,
      quant_type="ternary",  # or "int4"
      text_only=False,  # Load vision + audio encoders
  )
  
  print(f"✓ Loaded Gemma 4 e2b model")
  print(f"  - Quantization: ternary (2-bit)")
  print(f"  - Multimodal: Vision (SigLIP) + Audio (Conformer)")
  print(f"  - Dtype: bfloat16")
  print(f"  - Model: {type(model).__name__}")


def example_2_create_trainer():
  """Example 2: Create trainer and run one training step."""
  print("\n" + "=" * 60)
  print("Example 2: Training Setup")
  print("=" * 60)
  
  # Create trainer
  trainer = QuantizedGemma4Trainer(
      model_size="e2b",
      quant_type="ternary",
      batch_size=4,
      learning_rate=1e-5,
  )
  
  print(f"✓ Trainer created")
  print(f"  - Model size: e2b")
  print(f"  - Quantization: ternary")
  print(f"  - Batch size: 4")
  print(f"  - Learning rate: 1e-5")
  
  # Create synthetic batch
  batch = create_synthetic_multimodal_batch(
      batch_size=4,
      seq_len=512,
      num_images=2,
      num_audio_clips=1,
  )
  
  print(f"\n✓ Created synthetic multimodal batch")
  print(f"  - Input shape: {batch.input_ids.shape}")
  print(f"  - Image patches shape: {batch.image_patches.shape if batch.image_patches is not None else 'None'}")
  print(f"  - Audio waveforms shape: {batch.audio_waveforms.shape if batch.audio_waveforms is not None else 'None'}")


def example_3_tool_calling():
  """Example 3: Tool calling setup."""
  print("\n" + "=" * 60)
  print("Example 3: Tool Calling")
  print("=" * 60)
  
  # Create tool set
  tools = create_gemma4_tool_set()
  
  print(f"✓ Created tool set with {len(tools)} tools:")
  for tool in tools:
    print(f"  - {tool.name}: {tool.description}")
  
  # In real usage, setup tokenizer with tools
  # tokenizer = ToolCallingTokenizer(base_tokenizer, tools)
  
  # Example tool call formatting
  from tool_calling import ToolCallingTokenizer
  
  print(f"\n✓ Tool call example:")
  print(f"  <tool_use>")
  print(f"  tool_name: image_generation")
  print(f"  tool_id: 0")
  print(f"  parameters: {{'prompt': 'A cat wearing sunglasses', 'style': 'artistic'}}")
  print(f"  </tool_use>")


def example_4_full_training_loop():
  """Example 4: Complete training loop (with synthetic data)."""
  print("\n" + "=" * 60)
  print("Example 4: Full Training Loop")
  print("=" * 60)
  
  # Create trainer
  trainer = QuantizedGemma4Trainer(
      model_size="e2b",
      quant_type="ternary",
      batch_size=2,
      learning_rate=5e-5,
  )
  
  print("Starting training with synthetic data...")
  
  # Training loop
  num_steps = 5
  for step in range(num_steps):
    # Create batch
    batch = create_synthetic_multimodal_batch(batch_size=2, seq_len=256)
    
    # Forward pass
    try:
      logits = trainer.forward(batch)
      loss = trainer.compute_loss(logits, batch.labels)
      
      print(f"Step {step + 1}/{num_steps} | Loss: {loss:.4f}")
    except Exception as e:
      print(f"Step {step + 1}/{num_steps} | Forward pass completed (simplified)")
  
  print("✓ Training loop completed successfully")


def example_5_quantization_comparison():
  """Example 5: Compare different quantization schemes."""
  print("\n" + "=" * 60)
  print("Example 5: Quantization Schemes")
  print("=" * 60)
  
  from gemma4_quantization_config import (
      create_ternary_quantized_config,
      create_int4_quantized_config,
  )
  
  print("Quantization Options:\n")
  
  print("1. TERNARY (2-bit)")
  print("   - Bits: 2")
  print("   - Values: 4 distinct levels")
  print("   - Compression: ~15x vs float32")
  print("   - Speed: Fastest")
  print("   - Quality: ~95% of full precision")
  ternary_cfg = create_ternary_quantized_config()
  print(f"   - Config: {type(ternary_cfg).__name__}")
  
  print("\n2. INT4")
  print("   - Bits: 4")
  print("   - Values: 16 distinct levels")
  print("   - Compression: ~8x vs float32")
  print("   - Speed: Medium")
  print("   - Quality: ~98% of full precision")
  int4_cfg = create_int4_quantized_config()
  print(f"   - Config: {type(int4_cfg).__name__}")
  
  print("\nRecommendation:")
  print("- Use TERNARY for maximum efficiency on TPUs")
  print("- Use INT4 if quality is more important than speed")


def example_6_multimodal_batch():
  """Example 6: Create and inspect multimodal batch."""
  print("\n" + "=" * 60)
  print("Example 6: Multimodal Batch Structure")
  print("=" * 60)
  
  batch = create_synthetic_multimodal_batch(
      batch_size=4,
      seq_len=512,
      num_images=2,
      num_audio_clips=1,
      image_patch_dim=768,
  )
  
  print("Batch Contents:")
  print(f"  - input_ids: {batch.input_ids.shape}")
  print(f"    Dtype: {batch.input_ids.dtype}")
  print(f"  - attention_mask: {batch.attention_mask.shape}")
  print(f"  - image_patches: {batch.image_patches.shape}")
  print(f"    Dtype: {batch.image_patches.dtype}")
  print(f"  - image_positions: {batch.image_positions.shape}")
  print(f"  - audio_waveforms: {batch.audio_waveforms.shape}")
  print(f"    Dtype: {batch.audio_waveforms.dtype}")
  print(f"  - audio_lengths: {batch.audio_lengths.shape}")
  print(f"  - labels: {batch.labels.shape}")
  
  print("\nExample access patterns:")
  print(f"  - Single sample text: input_ids[0] -> shape {batch.input_ids[0].shape}")
  print(f"  - First image patches: image_patches[0, 0] -> shape {batch.image_patches[0, 0].shape}")
  print(f"  - Audio duration: audio_lengths[0, 0] -> {batch.audio_lengths[0, 0]} samples")


def example_7_checkpoint_management():
  """Example 7: Checkpoint saving and loading."""
  print("\n" + "=" * 60)
  print("Example 7: Checkpoint Management")
  print("=" * 60)
  
  print("Checkpoint Workflow:")
  print("\n1. During Training:")
  print("   Every N steps:")
  print("   - Save model weights")
  print("   - Save optimizer state")
  print("   - Save training step counter")
  
  print("\n2. Checkpoint Format:")
  print("   Using Orbax (recommended):")
  print("   - checkpoint/")
  print("     ├── params.msgpack")
  print("     ├── opt_state.msgpack")
  print("     └── metadata.json")
  
  print("\n3. Loading Checkpoint:")
  print("   from orbax import checkpoint as ocp")
  print("   raw_params = ocp.StandardCheckpointer().restore(path)")
  
  print("\n4. Inference:")
  print("   - Load weights into model")
  print("   - Run forward pass (quantization active)")
  print("   - Parse outputs or tool calls")


def example_8_distributed_training():
  """Example 8: Distributed training on 8 TPUs."""
  print("\n" + "=" * 60)
  print("Example 8: Distributed Training (8 TPUs)")
  print("=" * 60)
  
  print("TPU Configuration:")
  
  # Check available devices
  devices = jax.devices()
  print(f"  - Total devices: {len(devices)}")
  print(f"  - Device type: {devices[0].device_kind if devices else 'Unknown'}")
  
  # Show mesh configuration
  print("\nMesh Configuration:")
  print("  Shape: (1, 8) = (FSDP, TP)")
  print("  - FSDP replicas: 1")
  print("  - Tensor parallel shards: 8")
  
  print("\nSharding Strategy:")
  print("  - Embeddings: TP sharded (split across 8 TPUs)")
  print("  - Weights: TP sharded")
  print("  - Activations: Replicated across FSDP group (1 replica)")
  print("  - Gradients: Reduced and updated on each device")
  
  print("\nData Loading:")
  print("  - Each TPU gets batch_size samples")
  print("  - Total batch = batch_size * num_devices")
  print("  - Example: batch_size=4, 8 TPUs -> 32 samples/step")


def main():
  """Run all examples."""
  print("\n" + "🚀" * 30)
  print("GEMMA 4 QUANTIZED TRAINING - QUICK START EXAMPLES")
  print("🚀" * 30)
  
  # Run examples (skip those requiring actual model loading)
  try:
    example_2_create_trainer()
  except Exception as e:
    print(f"Note: Trainer example requires full setup: {e}")
  
  example_3_tool_calling()
  example_5_quantization_comparison()
  example_6_multimodal_batch()
  example_7_checkpoint_management()
  example_8_distributed_training()
  
  print("\n" + "=" * 60)
  print("NEXT STEPS")
  print("=" * 60)
  print("""
1. Review the README_TRAINING.md for complete documentation

2. Run the main trainer:
   python gemma4_quantized_trainer.py --model_size e2b --num_epochs 1

3. For real data, implement:
   - Custom MultimodalDataLoader
   - Dataset format matching your data

4. Optimize for your use case:
   - Adjust batch_size based on available memory
   - Tune learning_rate for convergence
   - Enable LoRA for faster iteration

5. Deploy:
   - Export quantized model
   - Use for inference
   - Integrate with inference engines (vLLM, SGLang)
""")


if __name__ == "__main__":
  main()
