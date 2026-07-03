# Copyright 2026 Google LLC
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
"""AQT v2 quantization configurations for Gemma 4 models."""

from typing import Optional
import aqt.jax.v2.config as aqt_config

def create_ternary_quantized_config(
    calibration_steps: int = 100,
    use_symmetric: bool = True,
) -> aqt_config.DotGeneral:
  """Create ternary (2-bit) quantized configuration for efficient training.
  
  Uses the standard robust AQT builder method to create an initial config 
  and modifies the bit depth down to 2-bit (ternary) for Gemma 4.
  """
  # Generate a standard default floating/integer config template safely
  config = aqt_config.fully_quantized(fwd_bits=2, bwd_bits=None)
  
  # Ensure the forward pass uses symmetric quantization parameters (2-bit ternary)
  if hasattr(config.fwd.lhs, 'quantizer') and hasattr(config.fwd.lhs.quantizer, 'numerics'):
    config.fwd.lhs.quantizer.numerics.bits = 2
    config.fwd.rhs.quantizer.numerics.bits = 2
  elif hasattr(config.fwd.lhs, 'numerics'):
    config.fwd.lhs.numerics.bits = 2
    config.fwd.rhs.numerics.bits = 2

  return config


def create_int4_quantized_config(
    calibration_steps: int = 100,
) -> aqt_config.DotGeneral:
  """Create INT4 quantized configuration for better accuracy than ternary."""
  # Generate a standard 4-bit config template safely
  config = aqt_config.fully_quantized(fwd_bits=4, bwd_bits=None)
  
  return config


def get_quantization_config(
    quant_type: str = "ternary",
    **kwargs
) -> aqt_config.DotGeneral:
  """Factory function to get quantization config by type."""
  if quant_type == "ternary":
    return create_ternary_quantized_config(**kwargs)
  elif quant_type == "int4":
    return create_int4_quantized_config(**kwargs)
  else:
    raise ValueError(
        f"Unknown quantization type: {quant_type}. "
        "Choose from: 'ternary', 'int4'"
    )
