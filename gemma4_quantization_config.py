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
from aqt.jax.v2 import calibration
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics
import jax.numpy as jnp


def create_ternary_quantized_config(
    calibration_steps: int = 100,
    use_symmetric: bool = True,
) -> aqt_config.DotGeneral:
  """Create ternary (2-bit) quantized configuration for efficient training.
  
  Ternary quantization uses 2-bits which allows for 4 distinct values,
  providing a good balance between model compression and accuracy.
  Optimized for Gemma 4 models on TPUs.
  
  Args:
    calibration_steps: Number of steps for calibration statistics gathering.
    use_symmetric: Whether to use symmetric quantization (recommended for QAT).
    
  Returns:
    DotGeneral config ready for injection into Flax Dense layers.
  """
  
  # Ternary (2-bit) quantization config
  ternary_numerics = int_numerics.IntSymmetric(
      bits=2,
      preserve_zero=True,
      preserve_max_val=True,
      clip=True,
      clip_gradient=True,
      round=True,
      noise_fn=None,  # No noise for deterministic training
  )
  
  # Statistics calibration using absolute max
  calib = calibration.AbsMaxCalibration()
  
  # Forward pass: Quantize activations and weights
  fwd_tensor = aqt_config.Tensor(
      quantizer=ternary_numerics,  # FIXED: Changed from numerics= to quantizer=
      calib_shared_axes=-1,
      scale_stop_grad=True,
      calibration=calib,
      po2_scale=False,
      use_fake_quant=False,
  )
  
  fwd_dot_general_raw = aqt_config.DotGeneralRaw(
      lhs=fwd_tensor,  # Activations
      rhs=fwd_tensor,  # Weights
  )
  
  # Backward pass: Use no quantization for stability
  bwd_no_quant = aqt_config.Tensor(
      quantizer=no_numerics.NoNumerics(),  # FIXED: Changed from numerics= to quantizer=
      calib_shared_axes=-1,
      scale_stop_grad=True,
      calibration=calib,
  )
  
  dlhs_dot_general_raw = aqt_config.DotGeneralRaw(
      lhs=bwd_no_quant,  # Gradient w.r.t. activations
      rhs=fwd_tensor,    # Weights (frozen during backprop for efficiency)
  )
  
  drhs_dot_general_raw = aqt_config.DotGeneralRaw(
      lhs=bwd_no_quant,  # Gradient w.r.t. activations
      rhs=bwd_no_quant,  # No gradient quantization
  )
  
  return aqt_config.DotGeneral(
      fwd=fwd_dot_general_raw,
      dlhs=dlhs_dot_general_raw,
      drhs=drhs_dot_general_raw,
  )


def create_int4_quantized_config(
    calibration_steps: int = 100,
) -> aqt_config.DotGeneral:
  """Create INT4 quantized configuration for better accuracy than ternary.
  
  INT4 provides 16 distinct values, offering better accuracy with minimal
  overhead compared to ternary quantization.
  
  Args:
    calibration_steps: Number of steps for calibration statistics gathering.
    
  Returns:
    DotGeneral config ready for injection into Flax Dense layers.
  """
  
  int4_numerics = int_numerics.IntSymmetric(
      bits=4,
      preserve_zero=True,
      preserve_max_val=True,
      clip=True,
      clip_gradient=True,
      round=True,
      noise_fn=None,
  )
  
  calib = calibration.AbsMaxCalibration()
  
  # Forward pass quantization
  fwd_tensor = aqt_config.Tensor(
      quantizer=int4_numerics,  # FIXED: Changed from numerics= to quantizer=
      calib_shared_axes=-1,
      scale_stop_grad=True,
      calibration=calib,
      po2_scale=False,
      use_fake_quant=False,
  )
  
  fwd_dot_general_raw = aqt_config.DotGeneralRaw(
      lhs=fwd_tensor,
      rhs=fwd_tensor,
  )
  
  # Backward pass with INT4 for weights, unquantized for gradients
  bwd_no_quant = aqt_config.Tensor(
      quantizer=no_numerics.NoNumerics(),  # FIXED: Changed from numerics= to quantizer=
      calib_shared_axes=-1,
      scale_stop_grad=True,
      calibration=calib,
  )
  
  dlhs_dot_general_raw = aqt_config.DotGeneralRaw(
      lhs=bwd_no_quant,
      rhs=fwd_tensor,
  )
  
  drhs_dot_general_raw = aqt_config.DotGeneralRaw(
      lhs=bwd_no_quant,
      rhs=bwd_no_quant,
  )
  
  return aqt_config.DotGeneral(
      fwd=fwd_dot_general_raw,
      dlhs=dlhs_dot_general_raw,
      drhs=drhs_dot_general_raw,
  )


def get_quantization_config(
    quant_type: str = "ternary",
    **kwargs
) -> aqt_config.DotGeneral:
  """Factory function to get quantization config by type.
  
  Args:
    quant_type: One of "ternary" (2-bit) or "int4".
    **kwargs: Additional arguments passed to config creators.
    
  Returns:
    DotGeneral config.
    
  Raises:
    ValueError: If quant_type is not recognized.
  """
  if quant_type == "ternary":
    return create_ternary_quantized_config(**kwargs)
  elif quant_type == "int4":
    return create_int4_quantized_config(**kwargs)
  else:
    raise ValueError(
        f"Unknown quantization type: {quant_type}. "
        "Choose from: 'ternary', 'int4'"
    )
