# Fast MLX Whisper Usage Guide

## Installation

```bash
# Install from the fast_mlx_whisper directory
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

```python
from fast_whisper_mlx import FastWhisperMLX

# Create model
model = FastWhisperMLX(
    model_name="large-v3",
    batch_size=8,
    use_vad=True
)

# Transcribe
result = model.transcribe("audio.wav", language="en")
print(result['text'])
```

## Advanced Usage

### Using Local Model Path
```python
model = FastWhisperMLX(
    model_name="/path/to/mlx_models/large-v3",
    batch_size=16,
    use_vad=True,
    max_segment_duration=10.0
)
```

### Disable VAD for Maximum Speed
```python
model = FastWhisperMLX(
    model_name="large-v3",
    batch_size=16,
    use_vad=False  # Faster but may hallucinate
)
```

### VAD-Only Transcription
```python
from vad_transcribe_final import transcribe_with_vad_final

result = transcribe_with_vad_final(
    audio="audio.wav",
    model_path="large-v3",
    vad_threshold=0.5,
    max_segment_duration=8.0,
    language="en"
)
```

## Performance Tips

1. **Batch Size**: Increase batch size for better throughput (8-16 recommended)
2. **VAD**: Enable VAD to prevent hallucinations, disable for maximum speed
3. **Segment Duration**: 5-10 seconds is optimal for accuracy
4. **Model**: Use local model path to avoid download overhead

## API Reference

### FastWhisperMLX

**Parameters:**
- `model_name`: Model size or path to local model
- `batch_size`: Number of segments to process in parallel (default: 8)
- `use_vad`: Enable VAD segmentation (default: True)
- `vad_threshold`: VAD activation threshold 0-1 (default: 0.5)
- `max_segment_duration`: Maximum segment length in seconds (default: 10.0)

**Methods:**
- `transcribe(audio, language=None, **kwargs)`: Transcribe audio file or numpy array
- `detect_language(audio)`: Detect language of audio

### Results Dictionary

```python
{
    'text': str,                    # Full transcription
    'segments': List[Dict],         # Text segments
    'language': str,                # Detected/specified language
    'processing_time': float,       # Time in seconds
    'vad_segments': List[Dict],     # VAD segment boundaries
}
```

## Limitations

- Word-level timestamps not fully implemented (infrastructure ready)
- Requires Apple Silicon for Metal acceleration
- First model load takes ~1 second

## Troubleshooting

### PyTorch Issues
If you encounter PyTorch loading errors, ensure you have a compatible version:
```bash
pip install torch>=2.0.0,<2.6.0
```

### Model Download
Models are cached in `~/.cache/whisper/mlx_models/`. Pre-download:
```python
from lightning_whisper_mlx import download_model
download_model("large-v3")
```

### Memory Issues
Reduce batch size if encountering memory errors:
```python
model = FastWhisperMLX(batch_size=4)  # Lower batch size
```