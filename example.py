#!/usr/bin/env python3
"""
Example usage of Fast MLX Whisper
"""
from fast_whisper_mlx import FastWhisperMLX

# Initialize model
model = FastWhisperMLX(
    model_name="large-v3",  # or path to local model
    batch_size=8,
    use_vad=True,
    max_segment_duration=10.0
)

# Transcribe audio
result = model.transcribe(
    audio="path/to/audio.wav",
    language="en",
    word_timestamps=False,  # Currently not fully supported
    verbose=True
)

# Print results
print(f"Text: {result['text']}")
print(f"Processing time: {result['processing_time']:.1f}s")
print(f"VAD segments: {len(result['vad_segments'])}")

# For production use with local model path:
"""
model = FastWhisperMLX(
    model_name="/Users/username/.cache/whisper/mlx_models/large-v3",
    batch_size=16,  # Increase for faster processing
    use_vad=True,
    vad_threshold=0.5,
    max_segment_duration=10.0
)
"""