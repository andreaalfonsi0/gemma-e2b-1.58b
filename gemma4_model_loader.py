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
"""Quantized Gemma 4 model loader with multimodal support."""

from typing import Optional

import aqt.jax.v2.config as aqt_config
import aqt.jax.v2.flax.aqt_flax as aqt_flax
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.models.gemma4 import model as gemma4_model_lib
from tunix.models.gemma4 import params as params_lib

from gemma4_quantization_config import get_quantization_config


class QuantizedGemma4Wrapper:
  """Wrapper to apply quantization to Gemma 4 model layers.
  
  This class intercepts Dense layer creation and injects AQT quantization
  configurations. Supports all Gemma 4 variants with vision and audio encoders.
  """
  
  def __init__(
      self,
      quant_type: str = "ternary",
      quantize_activations: bool = True,
      quantize_weights: bool = True,
      quantize_vision: bool = True,
      quantize_audio: bool = True,
  ):
    """Initialize quantization wrapper.
    
    Args:
      quant_type: Quantization type ("ternary" or "int4").
      quantize_activations: Whether to quantize activations.
      quantize_weights: Whether to quantize weights.
      quantize_vision: Whether to quantize vision encoder layers.
      quantize_audio: Whether to quantize audio encoder layers.
    """
    self.quant_config = get_quantization_config(quant_type)
    self.quantize_activations = quantize_activations
    self.quantize_weights = quantize_weights
    self.quantize_vision = quantize_vision
    self.quantize_audio = quantize_audio
    self.quant_type = quant_type
  
  def get_dot_general(self):
    """Get AQT DotGeneral for Flax Dense layers."""
    if self.quant_config:
      return aqt_flax.AqtDotGeneral(self.quant_config)
    return None


def load_quantized_gemma4(
    model_size: str = "e2b",
    checkpoint_path: Optional[str] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
    dtype: jnp.dtype = jnp.bfloat16,
    quant_type: str = "ternary",
    text_only: bool = False,
) -> tuple[gemma4_model_lib.Gemma4, QuantizedGemma4Wrapper]:
  """Load a quantized Gemma 4 model with multimodal support.
  
  Args:
    model_size: Model variant ("e2b", "e4b", "31b", etc.).
    checkpoint_path: Path to model checkpoint. If None, uses default paths.
    mesh: JAX mesh for distributed loading. For 8 TPUs, use mesh with
      shape like (1, 8) for tensor parallel, or (8, 1) for FSDP.
    dtype: Parameter dtype (default: bfloat16).
    quant_type: Quantization type ("ternary" or "int4").
    text_only: If True, skips loading vision/audio encoders.
    
  Returns:
    Tuple of (Gemma4Model, QuantizationWrapper).
  """
  
  # Get model config by size
  config_fn = getattr(
      gemma4_model_lib.ModelConfig,
      f"gemma4_{model_size}",
      gemma4_model_lib.ModelConfig.gemma4_e2b
  )
  
  if mesh is not None:
    model_config = config_fn(
        sharding_config=gemma4_model_lib.ShardingConfig.get_default_sharding(
            is_sampling=False
        )
    )
  else:
    model_config = config_fn()
  
  # Set checkpoint path
  if checkpoint_path is None:
    if model_size == "e2b":
      checkpoint_path = params_lib.GEMMA4_E2B_IT
    elif model_size == "e4b":
      checkpoint_path = params_lib.GEMMA4_E4B_IT
    else:
      raise ValueError(f"Unknown default checkpoint for model size: {model_size}")
  
  # Load model from checkpoint
  model = params_lib.create_model_from_checkpoint(
      checkpoint_path=checkpoint_path,
      model_config=model_config,
      mesh=mesh,
      dtype=dtype,
      text_only=text_only,
  )
  
  # Create quantization wrapper
  quant_wrapper = QuantizedGemma4Wrapper(quant_type=quant_type)
  
  return model, quant_wrapper


def apply_quantization_to_dense_layers(
    model: gemma4_model_lib.Gemma4,
    quant_wrapper: QuantizedGemma4Wrapper,
) -> gemma4_model_lib.Gemma4:
  """Apply AQT quantization to Dense layers in the model.
  
  This function uses JAX pytree operations to replace Dense layer definitions
  with quantized variants.
  
  Args:
    model: The Gemma 4 model instance.
    quant_wrapper: Quantization wrapper with AQT config.
    
  Returns:
    Model with quantized layers (may require recompilation).
    
  Note:
    For Flax NNX models, quantization is typically applied via the layer
    definition itself rather than post-hoc. Consider using a custom model
    factory that wraps Dense layers with AqtDotGeneral at initialization.
  """
  # In Flax NNX, quantization should be applied during model creation
  # by customizing the Dense layer's dot_general parameter.
  # This is a placeholder for demonstration.
  return model


def create_quantized_dense_wrapper(quant_config: aqt_config.DotGeneral):
  """Create a wrapped Dense layer that uses AQT quantization.
  
  Args:
    quant_config: AQT DotGeneral configuration.
    
  Returns:
    A Dense-like layer with quantized dot operations.
  """
  dot_general_fn = aqt_flax.AqtDotGeneral(quant_config)
  
  class QuantizedDense(nn.Module):
    """Dense layer with AQT quantization injection."""
    features: int
    use_bias: bool = True
    kernel_init: callable = nn.initializers.lecun_normal()
    bias_init: callable = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, inputs):
      # Use AQT dot_general instead of standard JAX dot_general
      kernel = self.param(
          'kernel',
          self.kernel_init,
          (inputs.shape[-1], self.features),
      )
      if self.use_bias:
        bias = self.param('bias', self.bias_init, (self.features,))
      else:
        bias = None
      
      y = dot_general_fn(
          lhs=inputs,
          rhs=kernel,
          dimension_numbers=(((inputs.ndim - 1,), (0,)), ((), ())),
      )
      if bias is not None:
        y = y + bias
      return y
  
  return QuantizedDense
