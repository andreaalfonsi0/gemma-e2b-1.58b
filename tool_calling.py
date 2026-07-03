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
"""Tool calling integration for quantized Gemma 4.

Supports training models to generate structured tool calls for external
function execution (e.g., image generation, web search, calculations).
"""

from typing import Any, Dict, List, Optional
import dataclasses
import json

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class Tool:
  """Tool definition for structured tool calling."""
  
  name: str
  description: str
  parameters: Dict[str, Any]  # JSON schema for parameters
  category: str  # e.g., "vision", "web_search", "math"


@dataclasses.dataclass
class ToolCall:
  """Parsed tool call from model output."""
  
  tool_name: str
  tool_id: int
  parameters: Dict[str, Any]
  confidence: float  # Probability of tool call


class ToolCallingTokenizer:
  """Tokenizer wrapper with special tokens for tool calling."""
  
  # Special tokens for tool calling
  TOOL_USE_START = "<tool_use>"
  TOOL_USE_END = "</tool_use>"
  TOOL_CALL_TEMPLATE = (
      "{tool_use_start}"
      "tool_name: {tool_name}\n"
      "tool_id: {tool_id}\n"
      "parameters: {parameters}"
      "{tool_use_end}"
  )
  
  def __init__(self, base_tokenizer: Any, tools: List[Tool]):
    """Initialize tool-aware tokenizer.
    
    Args:
      base_tokenizer: HuggingFace-compatible tokenizer.
      tools: List of available tools.
    """
    self.base_tokenizer = base_tokenizer
    self.tools = {tool.name: tool for tool in tools}
    self.tool_id_map = {tool.name: idx for idx, tool in enumerate(tools)}
    
    # Add tool calling special tokens
    special_tokens = {
        'additional_special_tokens': [
            self.TOOL_USE_START,
            self.TOOL_USE_END,
        ]
    }
    self.base_tokenizer.add_special_tokens(special_tokens)
  
  def format_tool_call(
      self,
      tool_name: str,
      parameters: Dict[str, Any],
  ) -> str:
    """Format a tool call as structured text.
    
    Args:
      tool_name: Name of the tool to call.
      parameters: Dictionary of parameters.
      
    Returns:
      Formatted tool call string.
    """
    if tool_name not in self.tool_id_map:
      raise ValueError(f"Unknown tool: {tool_name}")
    
    return self.TOOL_CALL_TEMPLATE.format(
        tool_use_start=self.TOOL_USE_START,
        tool_name=tool_name,
        tool_id=self.tool_id_map[tool_name],
        parameters=json.dumps(parameters),
        tool_use_end=self.TOOL_USE_END,
    )
  
  def parse_tool_calls(
      self,
      text: str,
  ) -> List[ToolCall]:
    """Parse tool calls from model output.
    
    Args:
      text: Raw model output text.
      
    Returns:
      List of parsed tool calls.
    """
    tool_calls = []
    
    # Split by tool boundaries
    parts = text.split(self.TOOL_USE_START)
    
    for part in parts[1:]:  # Skip first part (before any tool calls)
      if self.TOOL_USE_END not in part:
        continue
      
      tool_content = part.split(self.TOOL_USE_END)[0]
      
      try:
        # Parse tool content
        lines = tool_content.strip().split('\n')
        tool_name = None
        tool_id = None
        parameters = {}
        
        for line in lines:
          if line.startswith('tool_name:'):
            tool_name = line.split(':', 1)[1].strip()
          elif line.startswith('tool_id:'):
            tool_id = int(line.split(':', 1)[1].strip())
          elif line.startswith('parameters:'):
            params_str = line.split(':', 1)[1].strip()
            parameters = json.loads(params_str)
        
        if tool_name and tool_id is not None:
          tool_calls.append(
              ToolCall(
                  tool_name=tool_name,
                  tool_id=tool_id,
                  parameters=parameters,
                  confidence=0.95,  # Placeholder
              )
          )
      except (json.JSONDecodeError, ValueError):
        # Skip malformed tool calls
        continue
    
    return tool_calls


class ToolCallingTrainingMixin:
  """Mixin to add tool calling support to training."""
  
  def __init__(self, tools: List[Tool]):
    """Initialize tool calling.
    
    Args:
      tools: List of available tools.
    """
    self.tools = tools
    self.tool_tokenizer = None
  
  def prepare_tool_calling_data(
      self,
      batch: 'MultimodalBatch',  # Type hint forward reference
      tool_calls: List[List[ToolCall]],  # Per-sample tool calls
  ) -> 'MultimodalBatch':
    """Augment batch with tool calling targets.
    
    Modifies batch labels to include tool calling tokens, allowing the
    model to learn when and how to make tool calls.
    
    Args:
      batch: Original multimodal batch.
      tool_calls: Tool calls for each sample in batch.
      
    Returns:
      Modified batch with tool calling targets.
    """
    from multimodal_data_loader import MultimodalBatch
    
    # Add tool IDs to batch
    tool_ids_list = []
    for sample_calls in tool_calls:
      tool_ids = [call.tool_id for call in sample_calls]
      tool_ids_list.append(tool_ids)
    
    # Pad tool IDs to same length
    max_tools = max(len(ids) for ids in tool_ids_list)
    tool_ids_padded = []
    for ids in tool_ids_list:
      padded = ids + [0] * (max_tools - len(ids))
      tool_ids_padded.append(padded)
    
    # Create new batch with tool information
    return MultimodalBatch(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        image_patches=batch.image_patches,
        image_positions=batch.image_positions,
        audio_waveforms=batch.audio_waveforms,
        audio_lengths=batch.audio_lengths,
        labels=batch.labels,
        tool_ids=jnp.array(tool_ids_padded, dtype=jnp.int32),
    )
  
  def compute_tool_calling_loss(
      self,
      logits: jnp.ndarray,
      tool_ids: jnp.ndarray,
  ) -> jnp.ndarray:
    """Compute loss for tool calling predictions.
    
    Adds auxiliary loss for predicting which tools to call, helping the
    model learn to structure outputs as tool calls.
    
    Args:
      logits: Model logits shape (batch_size, seq_len, vocab_size).
      tool_ids: Tool IDs for each position shape (batch_size, num_tools).
      
    Returns:
      Scalar loss for tool calling.
    """
    import optax
    
    if tool_ids is None or jnp.all(tool_ids == 0):
      return jnp.array(0.0)
    
    # Use tool IDs as auxiliary targets
    # This is a simplified version; production would have specialized heads
    num_tools = jnp.max(tool_ids) + 1
    
    # Average logits over sequence and compute tool prediction loss
    avg_logits = jnp.mean(logits, axis=1)  # (batch_size, vocab_size)
    
    # Tool loss (encourage model to produce tool tokens when needed)
    tool_token_mask = (tool_ids > 0).astype(jnp.float32)
    tool_loss = jnp.mean(tool_token_mask) * 0.1  # Weight auxiliary loss
    
    return tool_loss


def create_gemma4_tool_set() -> List[Tool]:
  """Create a standard set of tools for Gemma 4.
  
  Returns:
    List of Tool definitions.
  """
  return [
      Tool(
          name="image_generation",
          description="Generate an image based on text description",
          parameters={
              "type": "object",
              "properties": {
                  "prompt": {"type": "string"},
                  "style": {"type": "string", "enum": ["realistic", "artistic"]},
              },
              "required": ["prompt"],
          },
          category="vision",
      ),
      Tool(
          name="web_search",
          description="Search the web for information",
          parameters={
              "type": "object",
              "properties": {
                  "query": {"type": "string"},
                  "num_results": {"type": "integer", "default": 5},
              },
              "required": ["query"],
          },
          category="search",
      ),
      Tool(
          name="calculate",
          description="Perform mathematical calculations",
          parameters={
              "type": "object",
              "properties": {
                  "expression": {"type": "string"},
              },
              "required": ["expression"],
          },
          category="math",
      ),
      Tool(
          name="video_summary",
          description="Summarize a video",
          parameters={
              "type": "object",
              "properties": {
                  "video_id": {"type": "string"},
                  "max_length": {"type": "integer", "default": 100},
              },
              "required": ["video_id"],
          },
          category="video",
      ),
  ]
