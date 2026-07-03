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

import inspect
from typing import Optional

import aqt.jax.v2.config as aqt_config
from aqt.jax.v2 import calibration
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics
import jax.numpy as jnp


def _safe_create_tensor(numerics_obj, **kwargs) -> aqt_config.Tensor:
  """Helper to safely instantiate aqt_config.Tensor across varying AQT versions."""
  # Inspect the Tensor constructor to see what keyword it accepts
  init_params = inspect.signature(aqt_config.Tensor.__init__).parameters
  
  if 'quantizer' in init_params:
    kwargs['quantizer'] = numerics_obj
  elif 'numerics' in init_params:
    kwargs['numerics'] = numerics_obj
  else:
    # Fallback: if the field name changed entirely, try finding a parameter 
    # that isn't one of the standard metadata fields.
    known_metadata = {'self', 'calib_shared_axes', 'scale_stop_grad', 'calibration', 'po2_scale', 'use_fake_quant'}
    possible_fields = [p for p in init_params if p not in known_metadata]
    if possible_fields:
      kwargs[possible_fields[0]] = numerics_obj
      
  return aqt_config.Tensor(**kwargs)


def create_ternary_quantized_config(
    calibration_steps: int = 100,
    use_symmetric: bool = True,
) -> aqt_config.DotGeneral:
  """Create ternary (2-bit) quantized configuration for efficient training."""
  
  ternary_numerics = int_numerics.IntSymmetric(
      bits=2,
      preserve_zero=True,
      preserve_max_val=True,
      clip=True,
      clip_gradient=True,
      round=True,
      noise_fn=None,
  )
  
  calib = calibration.AbsMaxCalibration()
  
  # Forward pass: Quantize activations and weights
  fwd_tensor = _safe_create_tensor(
      numerics_obj=ternary_numerics,
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
  
  # Backward pass: Use no quantization for stability
  bwd_no_quant = _safe_create_tensor(
      numerics_obj=no_numerics.NoNumerics(),
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


def create_int4_quantized_config(
    calibration_steps: int = 100,
) -> aqt_config.DotGeneral:
  """Create INT4 quantized configuration for better accuracy than ternary."""
  
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
  
  fwd_tensor = _safe_create_tensor(
      numerics_obj=int4_numerics,
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
  
  bwd_no_quant = _safe_create_tensor(
      numerics_obj=no_numerics.NoNumerics(),
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
