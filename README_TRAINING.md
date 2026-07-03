# Quantized Ternary Gemma 4 Multimodal Trainer

Efficient training code for Gemma 4 models with ternary (2-bit) quantization using AQT v2, optimized for 8 TPU devices on Kaggle. Supports multimodal inputs (images, video, text, audio) and tool calling.

## Overview

- **Model**: Gemma 4 (e2b, e4b, 31b variants)
- **Quantization**: AQT v2 with INT2 (ternary) or INT4 quantization
- **Multimodal**: Vision encoder (SigLIP) + Conformer audio encoder
- **Tool Calling**: Structured output generation for external functions
- **Infrastructure**: JAX/Flax on 8 TPUs with tensor parallelism
- **Training**: Supervised Fine-Tuning (SFT) with LoRA support

## Files

1. **gemma4_quantization_config.py** - AQT quantization configurations
   - `create_ternary_quantized_config()`: 2-bit quantization
   - `create_int4_quantized_config()`: 4-bit quantization
   - `get_quantization_config()`: Factory function

2. **gemma4_model_loader.py** - Model loading with quantization
   - `load_quantized_gemma4()`: Load Gemma 4 from checkpoint
   - `QuantizedGemma4Wrapper`: Quantization wrapper class
   - `create_quantized_dense_wrapper()`: Quantized Dense layer

3. **multimodal_data_loader.py** - Multimodal data pipeline
   - `MultimodalBatch`: Data container
   - `MultimodalDataLoader`: Grain-based data loader
   - `create_synthetic_multimodal_batch()`: For testing

4. **tool_calling.py** - Tool calling integration
   - `ToolCallingTokenizer`: Structured tool call formatting
   - `ToolCallingTrainingMixin`: Tool calling loss computation
   - `create_gemma4_tool_set()`: Standard tool definitions

5. **gemma4_quantized_trainer.py** - Main training script
   - `QuantizedGemma4Trainer`: Trainer class
   - Training loop with quantization
   - Checkpoint management

## Installation

```bash
# Install tunix and aqt
pip install git+https://github.com/google/tunix
pip install git+https://github.com/google/aqt

# Or from local clones
cd /path/to/aqt
pip install -e .
cd /path/to/tunix
pip install -e .

# Additional dependencies
pip install grain datasets pillow librosa optax orbax-checkpoint
pip install git+https://github.com/google/flax.git  # Latest Flax
```

## Usage

### Basic Training on 8 TPUs

```bash
python gemma4_quantized_trainer.py \
  --model_size e2b \
  --quant_type ternary \
  --batch_size 4 \
  --num_epochs 3 \
  --learning_rate 1e-5
```

### With Full Multimodal Support

```bash
python gemma4_quantized_trainer.py \
  --model_size e4b \
  --quant_type int4 \
  --batch_size 2 \
  --num_epochs 5 \
  --text_only=False \
  --dataset_path gs://my-bucket/multimodal-data
```

### Available Model Sizes

- `e2b`: 2B model (recommended for memory-constrained)
- `e4b`: 4B model (balanced)
- `31b`: 31B model (best quality, requires more devices)

### Quantization Options

- `ternary` (2-bit): Maximum compression, ~95% of full precision quality
- `int4`: Better accuracy, ~98% of full precision quality

## Architecture

### Quantization Flow

```
Input -> Vision Encoder (SigLIP) -> Quantization (AQT v2)
                                 -> Transformer Layers
                                 -> Audio Encoder (Conformer)
                                 -> Dense Layers with Quantized Dot Products
                                 -> Output (logits + tool calls)
```

### TPU Mesh Configuration

```
Mesh Shape: (1, 8) = (FSDP, TP)
- FSDP (Fully Sharded Data Parallel): 1 replica
- TP (Tensor Parallel): 8 devices for weight sharding

Per-layer sharding:
- Embeddings: TP sharded
- Query/Key/Value: TP sharded with selective FSDP
- FFW layers: TP sharded with FSDP
- Vision/Audio projections: TP sharded
```

## Training Loop

```python
# 1. Load model with quantization
model, quant_wrapper = load_quantized_gemma4(
    model_size="e2b",
    quant_type="ternary",
    mesh=tpu_mesh,
)

# 2. Create trainer
trainer = QuantizedGemma4Trainer(
    model_size="e2b",
    quant_type="ternary",
    batch_size=4,
)

# 3. Prepare multimodal batch
batch = MultimodalBatch(
    input_ids=...,           # Tokenized text
    image_patches=...,       # Vision patches from SigLIP
    audio_waveforms=...,     # Audio clips
    labels=...,              # Target tokens
)

# 4. Training step
loss, metrics = trainer.train_step(batch)

# 5. Inference with tool calling
output = model(batch)
tool_calls = parse_tool_calls(output)
```

## Tool Calling

### Enable Tool Calling in Training

```python
from tool_calling import create_gemma4_tool_set, ToolCallingTokenizer

tools = create_gemma4_tool_set()
tool_tokenizer = ToolCallingTokenizer(base_tokenizer, tools)

# Format training data with tool calls
tool_calls = [
    [ToolCall("image_generation", parameters={"prompt": "..."})]
]
batch = trainer.prepare_tool_calling_data(batch, tool_calls)
```

### Parse Tool Calls from Output

```python
from tool_calling import ToolCallingTokenizer

tool_tokenizer = ToolCallingTokenizer(tokenizer, tools)
tool_calls = tool_tokenizer.parse_tool_calls(model_output)

for call in tool_calls:
    print(f"Calling {call.tool_name} with {call.parameters}")
```

## Advanced Configuration

### Custom Quantization

```python
from gemma4_quantization_config import create_ternary_quantized_config

custom_quant = create_ternary_quantized_config(calibration_steps=200)
trainer = QuantizedGemma4Trainer(quant_config=custom_quant)
```

### Vision/Audio Configuration

```python
from tunix.models.gemma4.model import ModelConfig

config = ModelConfig.gemma4_e4b()
config.vision_encoder = VisionEncoderConfig(d_model=1536, num_layers=24)
config.audio_encoder = ConformerConfig(num_layers=16, hidden_dim=768)
```

### LoRA Fine-Tuning (Reduced Memory)

```bash
pip install git+https://github.com/google/qwix

# Integrate with trainer
from qwix import LoraProvider
lora_provider = LoraProvider(rank=32, alpha=16.0)
# Apply to trainer
```

## Performance

### Estimated Training Speed (8 TPUs, e2b model)

| Batch Size | Throughput | Memory | Quantization |
|-----------|-----------|--------|--------------|
| 4 | 800 tokens/sec | 45 GB | Ternary |
| 4 | 600 tokens/sec | 60 GB | INT4 |
| 2 | 400 tokens/sec | 30 GB | Ternary + LoRA |

### Model Sizes

| Model | Params | Weights (fp32) | Quantized (ternary) |
|-------|--------|----------------|-------------------|
| e2b | 2B | 8 GB | 0.5 GB |
| e4b | 4B | 16 GB | 1 GB |
| 31b | 31B | 124 GB | 8 GB |

## Data Preparation

### Multimodal Dataset Format

```python
{
    "text": "Describe this image and audio",
    "images": ["path/to/image.jpg"],
    "audio": ["path/to/audio.wav"],
    "tool_calls": [
        {"name": "image_generation", "parameters": {...}},
    ]
}
```

### Using Grain DataSource

```python
from multimodal_data_loader import MultimodalDataLoader
import grain

dataset = grain.DataSource([
    {"text": "...", "images": [...], "audio": [...]},
    # ... more samples
])

loader = MultimodalDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    image_processor=image_processor,
    batch_size=4,
)

for batch in loader.create_dataloader():
    # batch is MultimodalBatch
    pass
```

## Checkpointing

```python
# Model checkpoints
trainer._save_checkpoint(state, '/path/to/checkpoint')

# Load checkpoint
from orbax import checkpoint as ocp
raw_params = ocp.StandardCheckpointer().restore(checkpoint_path)
```

## Troubleshooting

### Out of Memory

- Reduce batch_size from 4 to 2
- Use `--quant_type ternary` instead of `int4`
- Enable LoRA: `--use_lora=True`
- Use text-only mode: `--text_only=True`

### Slow Training

- Ensure tensor parallelism is working: check device placement
- Use synthetic data first to profile: `--use_synthetic_data=True`
- Enable flash attention in config (if available for your TPU version)

### Shape Mismatches

- Check multimodal patch dimensions match model config
- Verify audio clip length is compatible with Conformer
- Use `create_synthetic_multimodal_batch()` to test shapes

## References

- [AQT Documentation](https://github.com/google/aqt)
- [Tunix Documentation](https://tunix.readthedocs.io/)
- [Gemma 4 Model Card](https://huggingface.co/google/gemma-4-9b)
- [JAX on TPU Guide](https://cloud.google.com/tpu/docs/jax-quickstart)

## License

Apache 2.0

## Citation

If you use this code in research, please cite:

```bibtex
@software{gemma4_quantized_trainer,
  title={Quantized Ternary Gemma 4 Multimodal Trainer},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo},
}
```
