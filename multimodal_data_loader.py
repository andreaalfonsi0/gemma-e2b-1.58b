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
"""Multimodal data loader for Gemma 4 training."""

from typing import Any, Dict, Optional, Tuple
import dataclasses

import grain
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclasses.dataclass
class MultimodalBatch:
  """Batch container for multimodal data."""
  
  # Text inputs
  input_ids: jnp.ndarray  # Shape: (batch_size, seq_len)
  attention_mask: jnp.ndarray  # Shape: (batch_size, seq_len)
  
  # Vision inputs (optional)
  image_patches: Optional[jnp.ndarray] = None  # Shape: (batch_size, num_patches, patch_dim)
  image_positions: Optional[jnp.ndarray] = None  # Relative positions of image patches
  
  # Audio inputs (optional)
  audio_waveforms: Optional[jnp.ndarray] = None  # Shape: (batch_size, num_clips, samples)
  audio_lengths: Optional[jnp.ndarray] = None  # Shape: (batch_size, num_clips)
  
  # Labels for training
  labels: Optional[jnp.ndarray] = None  # Shape: (batch_size, seq_len)
  
  # Tool calling info (optional)
  tool_ids: Optional[jnp.ndarray] = None  # Tool identifiers if applicable


class MultimodalDataLoader:
  """Efficient data loader for multimodal Gemma 4 training on TPUs.
  
  Handles batching of text, images, videos, and audio with proper padding
  and sharding for distributed training.
  """
  
  def __init__(
      self,
      dataset: Any,  # grain.DataSource or similar
      tokenizer: Any,
      image_processor: Any,
      audio_processor: Optional[Any] = None,
      batch_size: int = 8,
      max_text_length: int = 2048,
      max_images: int = 4,
      max_audio_clips: int = 2,
      num_workers: int = 8,
      shuffle: bool = True,
      seed: int = 42,
  ):
    """Initialize multimodal data loader.
    
    Args:
      dataset: grain.DataSource with multimodal samples.
      tokenizer: Tokenizer (HuggingFace-compatible).
      image_processor: Image processor for vision encoding.
      audio_processor: Audio processor for audio encoding.
      batch_size: Batch size per device.
      max_text_length: Maximum text sequence length.
      max_images: Maximum number of images per sample.
      max_audio_clips: Maximum number of audio clips per sample.
      num_workers: Number of parallel data loading workers.
      shuffle: Whether to shuffle the dataset.
      seed: Random seed for reproducibility.
    """
    self.dataset = dataset
    self.tokenizer = tokenizer
    self.image_processor = image_processor
    self.audio_processor = audio_processor
    self.batch_size = batch_size
    self.max_text_length = max_text_length
    self.max_images = max_images
    self.max_audio_clips = max_audio_clips
    self.num_workers = num_workers
    self.shuffle = shuffle
    self.seed = seed
    self.rng = np.random.RandomState(seed)
  
  def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single multimodal sample.
    
    Args:
      sample: Raw sample dict with 'text', 'images', 'audio' keys.
      
    Returns:
      Processed sample with tokenized text and encoded vision/audio.
    """
    processed = {}
    
    # Process text
    text = sample.get('text', '')
    text_encoding = self.tokenizer(
        text,
        max_length=self.max_text_length,
        padding='max_length',
        truncation=True,
        return_tensors='np',
    )
    processed['input_ids'] = text_encoding['input_ids'][0]
    processed['attention_mask'] = text_encoding['attention_mask'][0]
    
    # Process images
    images = sample.get('images', [])
    if images and self.image_processor:
      # Handle variable number of images by padding
      image_patches_list = []
      image_positions_list = []
      
      for img in images[:self.max_images]:
        if isinstance(img, str):  # Path
          img = Image.open(img).convert('RGB')
        elif not isinstance(img, Image.Image):
          img = Image.fromarray(img)
        
        # Process image
        processed_img = self.image_processor(img)
        if isinstance(processed_img, tuple):
          patches, positions = processed_img
        else:
          patches = processed_img
          positions = None
        
        image_patches_list.append(patches)
        if positions is not None:
          image_positions_list.append(positions)
      
      # Pad image list to max_images
      if image_patches_list:
        patches_shape = image_patches_list[0].shape
        while len(image_patches_list) < self.max_images:
          image_patches_list.append(np.zeros(patches_shape, dtype=np.float32))
        processed['image_patches'] = np.stack(image_patches_list)
        
        if image_positions_list:
          while len(image_positions_list) < self.max_images:
            image_positions_list.append(
                np.zeros(image_positions_list[0].shape, dtype=np.int32)
            )
          processed['image_positions'] = np.stack(image_positions_list)
    
    # Process audio
    audio_samples = sample.get('audio', [])
    if audio_samples and self.audio_processor:
      audio_waveforms = []
      audio_lengths = []
      
      for audio in audio_samples[:self.max_audio_clips]:
        if isinstance(audio, str):  # Path
          import librosa
          waveform, sr = librosa.load(audio, sr=16000)
        else:
          waveform = audio
        
        # Processor will resample/normalize
        processed_audio = self.audio_processor(waveform)
        audio_waveforms.append(processed_audio['waveform'])
        audio_lengths.append(len(waveform))
      
      # Pad audio list
      if audio_waveforms:
        waveform_shape = audio_waveforms[0].shape
        while len(audio_waveforms) < self.max_audio_clips:
          audio_waveforms.append(np.zeros(waveform_shape, dtype=np.float32))
        processed['audio_waveforms'] = np.stack(audio_waveforms)
        processed['audio_lengths'] = np.array(
            audio_lengths + [0] * (self.max_audio_clips - len(audio_lengths)),
            dtype=np.int32,
        )
    
    # Labels (usually same as input_ids shifted by 1 for next-token prediction)
    processed['labels'] = processed['input_ids'].copy()
    
    # Tool IDs if present
    if 'tool_id' in sample:
      processed['tool_ids'] = np.array([sample['tool_id']], dtype=np.int32)
    
    return processed
  
  def _collate_batch(self, batch: list[Dict[str, Any]]) -> MultimodalBatch:
    """Collate a batch of processed samples.
    
    Args:
      batch: List of processed samples.
      
    Returns:
      MultimodalBatch with stacked arrays.
    """
    batch_dict = {key: [] for key in batch[0].keys()}
    
    for sample in batch:
      for key, value in sample.items():
        batch_dict[key].append(value)
    
    # Stack and convert to jax arrays
    stacked = {}
    for key, values in batch_dict.items():
      if key in ['image_patches', 'image_positions', 'audio_waveforms', 'audio_lengths']:
        if values[0] is not None:
          stacked[key] = jnp.array(np.stack(values))
        else:
          stacked[key] = None
      else:
        stacked[key] = jnp.array(np.stack(values))
    
    return MultimodalBatch(
        input_ids=stacked['input_ids'],
        attention_mask=stacked.get('attention_mask'),
        image_patches=stacked.get('image_patches'),
        image_positions=stacked.get('image_positions'),
        audio_waveforms=stacked.get('audio_waveforms'),
        audio_lengths=stacked.get('audio_lengths'),
        labels=stacked.get('labels'),
        tool_ids=stacked.get('tool_ids'),
    )
  
  def create_dataloader(self):
    """Create a Grain dataloader for training.
    
    Returns:
      Iterable grain dataloader.
    """
    # Create mapping dataset with processing
    data_source = grain.MapDataset.source(self.dataset)
    data_source = data_source.map(self._process_sample)
    
    if self.shuffle:
      data_source = data_source.shuffle(
          buffer_size=1000,
          seed=self.seed,
      )
    
    # Batch and collate
    data_source = data_source.batch(self.batch_size)
    data_source = data_source.map(self._collate_batch)
    
    return data_source


def create_synthetic_multimodal_batch(
    batch_size: int = 4,
    seq_len: int = 512,
    num_images: int = 2,
    num_audio_clips: int = 1,
    image_patch_dim: int = 768,
) -> MultimodalBatch:
  """Create a synthetic multimodal batch for testing.
  
  Useful for debugging and verifying model shape compatibility before
  loading real data.
  
  Args:
    batch_size: Batch size.
    seq_len: Text sequence length.
    num_images: Number of images per sample.
    num_audio_clips: Number of audio clips per sample.
    image_patch_dim: Dimension of image patches.
    
  Returns:
    Synthetic MultimodalBatch.
  """
  rng = np.random.RandomState(42)
  
  return MultimodalBatch(
      input_ids=jnp.array(
          rng.randint(0, 262144, (batch_size, seq_len)), dtype=jnp.int32
      ),
      attention_mask=jnp.ones((batch_size, seq_len), dtype=jnp.float32),
      image_patches=jnp.array(
          rng.randn(batch_size, num_images, image_patch_dim).astype(np.float32)
      ),
      image_positions=jnp.array(
          rng.randint(0, 256, (batch_size, num_images, 2)), dtype=jnp.int32
      ),
      audio_waveforms=jnp.array(
          rng.randn(batch_size, num_audio_clips, 16000).astype(np.float32)
      ),
      audio_lengths=jnp.array(
          rng.randint(8000, 16000, (batch_size, num_audio_clips)), dtype=jnp.int32
      ),
      labels=jnp.array(
          rng.randint(0, 262144, (batch_size, seq_len)), dtype=jnp.int32
      ),
  )

def create_hf_multimodal_batch(
    batch_size: int = 4,
    seq_len: int = 512,
    num_images: int = 2,
    num_audio_clips: int = 1,
    image_patch_dim: int = 768,
) -> Any:  # Returns MultimodalBatch
    """Create a multimodal batch fetching real data from Hugging Face.
    
    Streams real text and images from 'whyen-wang/coco_captions' 
    and real audio from 'PolyAI/minds14'.
    """
    # 1. Load streaming datasets
    # Using MS COCO for clean, reliable Image-Text pairs
    vl_dataset = load_dataset("whyen-wang/coco_captions", split="train", streaming=True)
    # Using a Speech dataset for Audio
    audio_dataset = load_dataset("PolyAI/minds14", name="en-US", split="train", streaming=True, trust_remote_code=True)
    
    # 2. Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    
    vl_iter = iter(vl_dataset)
    audio_iter = iter(audio_dataset)
    
    texts = []
    images = []
    audios = []
    audio_lens = []
    
    # Fetch exact amounts of real data required to fill the dimensions
    for _ in range(batch_size):
        # Fetch Text (COCO provides a list of captions; we take the first one)
        first_vl_sample = next(vl_iter)
        texts.append(first_vl_sample['captions'][0])
        
        # Fetch multiple Images for this sample to fulfill `num_images` requirement
        sample_images = [first_vl_sample['image']]
        for _ in range(num_images - 1):
            sample_images.append(next(vl_iter)['image'])
        images.append(sample_images)
        
        # Fetch multiple Audio clips for this sample
        sample_audio = []
        sample_audio_lens = []
        for _ in range(num_audio_clips):
            a_sample = next(audio_iter)
            audio_array = a_sample['audio']['array']
            sample_audio.append(audio_array)
            sample_audio_lens.append(len(audio_array))
            
        audios.append(sample_audio)
        audio_lens.append(sample_audio_lens)

    # 3. Process Text (Tokenization)
    tokens = tokenizer(
        texts, 
        max_length=seq_len, 
        padding="max_length", 
        truncation=True, 
        return_tensors="np"
    )
    
    # 4. Process Images 
    processed_images = np.zeros((batch_size, num_images, image_patch_dim), dtype=np.float32)
    for b in range(batch_size):
        for n in range(num_images):
            # Convert to RGB and shrink significantly to simulate flat patches
            img = images[b][n].convert("RGB").resize((16, 16)) 
            img_array = np.array(img, dtype=np.float32).flatten()
            
            # Align flattened image to requested patch dimension
            if len(img_array) > image_patch_dim:
                img_array = img_array[:image_patch_dim]
            else:
                img_array = np.pad(img_array, (0, image_patch_dim - len(img_array)))
                
            processed_images[b, n, :] = img_array

    # 5. Process Audio
    max_audio_samples = 16000
    processed_audio = np.zeros((batch_size, num_audio_clips, max_audio_samples), dtype=np.float32)
    processed_audio_lengths = np.zeros((batch_size, num_audio_clips), dtype=np.int32)
    
    for b in range(batch_size):
        for n in range(num_audio_clips):
            audio_arr = audios[b][n]
            if len(audio_arr) > max_audio_samples:
                audio_arr = audio_arr[:max_audio_samples]
            else:
                audio_arr = np.pad(audio_arr, (0, max_audio_samples - len(audio_arr)))
                
            processed_audio[b, n, :] = audio_arr
            processed_audio_lengths[b, n] = min(audio_lens[b][n], max_audio_samples)

    # 6. Generate sequential placeholder coordinates for image positions
    image_positions = np.zeros((batch_size, num_images, 2), dtype=np.int32)
    for b in range(batch_size):
        for n in range(num_images):
            image_positions[b, n] = [n, n] 

    # Note: Returning a dictionary here for structural compatibility, 
    # but in your code, this maps perfectly to `MultimodalBatch(...)`.
    return {
        "input_ids": jnp.array(tokens["input_ids"], dtype=jnp.int32),
        "attention_mask": jnp.array(tokens["attention_mask"], dtype=jnp.float32),
        "image_patches": jnp.array(processed_images),
        "image_positions": jnp.array(image_positions, dtype=jnp.int32),
        "audio_waveforms": jnp.array(processed_audio),
        "audio_lengths": jnp.array(processed_audio_lengths, dtype=jnp.int32),
        "labels": jnp.array(tokens["input_ids"].copy(), dtype=jnp.int32),
    }
