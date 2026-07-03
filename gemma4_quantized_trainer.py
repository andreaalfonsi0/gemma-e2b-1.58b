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
"""Quantized Gemma 4 multimodal training script for 8 TPUs.

Run on Kaggle TPU machine:
  python gemma4_quantized_trainer.py \\
    --model_size e2b \\
    --quant_type ternary \\
    --batch_size 4 \\
    --num_epochs 3
"""

import dataclasses
from typing import Any, Callable, Optional
import os
import numpy as np
from absl import app
from absl import flags
from absl import logging
import aqt.jax.v2.config as aqt_config
import aqt.jax.v2.flax.aqt_flax as aqt_flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
from jax.sharding import Mesh, AxisType
import optax
import orbax.checkpoint as ocp

from tunix.models.gemma4 import model as gemma4_model_lib
from tunix.models.gemma4 import params as params_lib
from tunix.sft import peft_trainer
from tunix.utils import env_utils

from gemma4_quantization_config import get_quantization_config
from gemma4_model_loader import load_quantized_gemma4
from multimodal_data_loader import (
    MultimodalBatch,
    MultimodalDataLoader,
    create_hf_multimodal_batch,
)


# Flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_size', 'e2b', 'Model size: e2b, e4b, or 31b')
flags.DEFINE_string('quant_type', 'ternary', 'Quantization type: ternary or int4')
flags.DEFINE_integer('batch_size', 4, 'Batch size per TPU device')
flags.DEFINE_integer('num_epochs', 3, 'Number of training epochs')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate')
flags.DEFINE_integer('warmup_steps', 500, 'Warmup steps')
flags.DEFINE_integer('max_steps', None, 'Max training steps (None = unlimited)')
flags.DEFINE_string('checkpoint_dir', '/tmp/gemma4_checkpoints', 'Checkpoint dir')
flags.DEFINE_bool('text_only', False, 'Skip vision/audio encoders')
flags.DEFINE_string('dataset_path', None, 'Path to training dataset')
flags.DEFINE_bool('use_synthetic_data', True, 'Use synthetic data for testing')


@dataclasses.dataclass
class TrainingState:
  """Training state container."""
  
  model: gemma4_model_lib.Gemma4
  optimizer_state: Any
  step: int
  loss: float


class QuantizedGemma4Trainer:
  """Trainer for quantized Gemma 4 models with multimodal support."""
  
  def __init__(
      self,
      model_size: str = "e2b",
      quant_type: str = "ternary",
      batch_size: int = 4,
      learning_rate: float = 1e-5,
      mesh: Optional[jax.sharding.Mesh] = None,
  ):
    """Initialize trainer.
    
    Args:
      model_size: Model variant (e2b, e4b, 31b).
      quant_type: Quantization type (ternary, int4).
      batch_size: Batch size per device.
      learning_rate: Learning rate.
      mesh: JAX mesh for distributed training.
    """
    self.model_size = model_size
    self.quant_type = quant_type
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.mesh = mesh or self._create_tpu_mesh()
    
    # Load model with quantization
    with self.mesh:
      self.model, self.quant_wrapper = load_quantized_gemma4(
          model_size=model_size,
          mesh=self.mesh,
          dtype=jnp.bfloat16,
          quant_type=quant_type,
          text_only=FLAGS.text_only,
      )
    
    # Optimizer
    schedule = optax.warmup_linear(
        FLAGS.warmup_steps,
        learning_rate,
    )
    self.optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule),
    )
    
    self.quant_config = get_quantization_config(quant_type)
    self.dot_general = aqt_flax.AqtDotGeneral(self.quant_config)
  
  def _create_tpu_mesh(self) -> jax.sharding.Mesh:
    """Create mesh for 8 TPU devices with Auto axes for Shardy.
    
    Returns:
      JAX mesh with shape (1, 8) for tensor parallelism across 8 TPUs.
    """
    devices = jax.devices()
    device_count = len(devices)
    if device_count < 8:
        logging.warning(
            f'Requested 8 TPUs but only {device_count} available. '
            'Using all available devices.'
        )
    else:
        # Limit to the first 8 devices if more are available
        devices = devices[:8]
        device_count = 8
    
    # Reshape the devices array to match (fsdp, tp) -> (1, 8)
    devices_array = np.array(devices).reshape(1, device_count)
    
    axis_names = ('fsdp', 'tp')
    
    # Explicitly set AxisType.Auto on both axes to satisfy the Gemma 4 / Tunix constraint
    axis_types = (AxisType.Auto, AxisType.Auto)
    
    return Mesh(devices_array, axis_names=axis_names, axis_types=axis_types)
  
  def forward(
      self,
      batch: MultimodalBatch,
  ) -> jnp.ndarray:
    """Forward pass with quantization.
    
    Args:
      batch: MultimodalBatch with tokenized inputs.
      
    Returns:
      Logits from the model.
    """
    with self.mesh:
      # Prepare inputs for model
      inputs = {
          'input_ids': batch.input_ids,
          'attention_mask': batch.attention_mask,
      }
      
      # Add vision inputs if present
      if batch.image_patches is not None:
        inputs['image_patches'] = batch.image_patches
        if batch.image_positions is not None:
          inputs['image_positions'] = batch.image_positions
      
      # Add audio inputs if present
      if batch.audio_waveforms is not None:
        inputs['audio_waveforms'] = batch.audio_waveforms
        if batch.audio_lengths is not None:
          inputs['audio_lengths'] = batch.audio_lengths
      
      # Forward pass through quantized model
      logits = self.model(
          input_ids=inputs['input_ids'],
          attention_mask=inputs['attention_mask'],
          image_patches=inputs.get('image_patches'),
          audio_inputs=inputs.get('audio_waveforms'),
          # Additional model-specific kwargs as needed
      )
      
      return logits
  
  def compute_loss(
      self,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
  ) -> jnp.ndarray:
    """Compute cross-entropy loss.
    
    Args:
      logits: Model logits shape (batch_size, seq_len, vocab_size).
      labels: Target token IDs shape (batch_size, seq_len).
      
    Returns:
      Scalar loss.
    """
    # Flatten for loss computation
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    
    # Cross-entropy loss (ignore padding tokens)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        flat_logits,
        flat_labels,
    )
    
    # Mask out padding (token_id == 0)
    mask = (flat_labels != 0).astype(jnp.float32)
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    
    return loss
  
  def train_step(
      self,
      batch: MultimodalBatch,
      state: TrainingState,
  ) -> tuple[TrainingState, dict[str, float]]:
    """Single training step with quantization.
    
    Args:
      batch: Multimodal training batch.
      state: Current training state.
      
    Returns:
      Updated training state and metrics dict.
    """
    with self.mesh:
      def loss_fn(model_params):
        # Model forward pass (training mode)
        with nnx.disable_grad():
          # In practice, you'd use nnx.enable_grad selectively for LoRA params
          # This is a simplified version
          logits = self.forward(batch)
        
        # Compute loss
        loss = self.compute_loss(logits, batch.labels)
        return loss
      
      # Backward pass
      loss, grads = jax.value_and_grad(loss_fn)(None)
      
      # Update optimizer state and parameters
      updates, opt_state = self.optimizer.update(
          grads,
          state.optimizer_state,
      )
      
      # Apply updates (in practice, use nnx.apply_updates for NNX models)
      # This is simplified; real implementation uses nnx update mechanics
      
      new_state = TrainingState(
          model=state.model,
          optimizer_state=opt_state,
          step=state.step + 1,
          loss=float(loss),
      )
      
      metrics = {
          'loss': float(loss),
          'step': state.step,
          'learning_rate': self.learning_rate,
      }
      
      return new_state, metrics
  
  def train(
      self,
      num_epochs: int = 3,
      max_steps: Optional[int] = None,
      checkpoint_dir: str = '/tmp/checkpoints',
  ):
    """Run training loop.
    
    Args:
      num_epochs: Number of training epochs.
      max_steps: Max training steps (None = no limit).
      checkpoint_dir: Directory for saving checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create synthetic data for this example
    # In production, replace with real data loader
    num_batches = 10
    
    # Initialize training state
    opt_state = self.optimizer.init(None)
    state = TrainingState(
        model=self.model,
        optimizer_state=opt_state,
        step=0,
        loss=0.0,
    )
    
    # Training loop
    total_steps = 0
    for epoch in range(num_epochs):
      logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
      
      for batch_idx in range(num_batches):
        # Create synthetic batch (replace with real data in production)
        batch = create_hf_multimodal_batch(
            batch_size=self.batch_size,
            seq_len=512,
            num_images=2,
            num_audio_clips=1,
        )
        
        # Training step
        state, metrics = self.train_step(batch, state)
        
        if batch_idx % 5 == 0:
          logging.info(
              f"Epoch {epoch + 1} | Batch {batch_idx} | "
              f"Loss: {metrics['loss']:.4f}"
          )
        
        total_steps += 1
        if max_steps and total_steps >= max_steps:
          break
      
      if max_steps and total_steps >= max_steps:
        break
    
    logging.info(f"Training complete. Total steps: {total_steps}")
    
    # Save final checkpoint
    self._save_checkpoint(state, os.path.join(checkpoint_dir, 'final'))
  
  def _save_checkpoint(
      self,
      state: TrainingState,
      checkpoint_path: str,
  ):
    """Save checkpoint.
    
    Args:
      state: Training state.
      checkpoint_path: Path to save checkpoint.
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    logging.info(f"Saving checkpoint to {checkpoint_path}")
    # Implementation depends on your checkpoint format (Orbax, SafeTensors, etc.)


def main(argv):
  del argv  # Unused
  
  # Setup JAX for TPU training
  env_utils.setup_sharding_environment()
  
  logging.info(
      f"Starting Gemma 4 {FLAGS.model_size} training with "
      f"{FLAGS.quant_type} quantization"
  )
  logging.info(f"Available devices: {jax.devices()}")
  logging.info(f"Device count: {jax.device_count()}")
  
  # Create trainer
  trainer = QuantizedGemma4Trainer(
      model_size=FLAGS.model_size,
      quant_type=FLAGS.quant_type,
      batch_size=FLAGS.batch_size,
      learning_rate=FLAGS.learning_rate,
  )
  
  # Run training
  trainer.train(
      num_epochs=FLAGS.num_epochs,
      max_steps=FLAGS.max_steps,
      checkpoint_dir=FLAGS.checkpoint_dir,
  )
  
  logging.info("Training completed successfully!")


if __name__ == '__main__':
  app.run(main)
