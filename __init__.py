"""
Fast MLX Whisper - Optimized Whisper implementation for Apple Silicon

A fast and accurate MLX-based Whisper implementation with:
- 6.9x faster than real-time transcription
- VAD-based segmentation to prevent hallucinations
- Apple Silicon optimization via Metal
- Production-ready error handling
"""

from .fast_whisper_mlx import FastWhisperMLX, create_fast_whisper_mlx
from .vad_transcribe_final import (
    ProductionVADTranscriber,
    transcribe_with_vad_final,
    SpeechSegment
)
from .enhanced_word_timestamps import (
    EnhancedWordTimestampExtractor,
    extract_word_timestamps,
    ALIGNMENT_HEADS
)

__version__ = "1.0.0"
__all__ = [
    "FastWhisperMLX",
    "create_fast_whisper_mlx",
    "ProductionVADTranscriber", 
    "transcribe_with_vad_final",
    "EnhancedWordTimestampExtractor",
    "extract_word_timestamps",
    "ALIGNMENT_HEADS"
]