#!/usr/bin/env python3
"""
Enhanced word-level timestamp implementation for MLX Whisper
Combines best practices from WhisperX and latest Whisper improvements
"""

import numpy as np
import mlx.core as mx
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

from .audio import SAMPLE_RATE, HOP_LENGTH, TOKENS_PER_SECOND
from .timing import WordTiming, find_alignment, merge_punctuations
from .tokenizer import Tokenizer
from .whisper import Whisper


# Alignment heads configuration for different model sizes
ALIGNMENT_HEADS = {
    "tiny": [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]],
    "base": [[3, 1], [4, 2], [4, 3], [4, 7], [5, 1], [5, 2], [5, 4], [5, 6]],
    "small": [[5, 3], [5, 9], [8, 0], [8, 4], [8, 7], [8, 8], [9, 0], [9, 7], [9, 9], [10, 5]],
    "medium": [[13, 15], [15, 4], [15, 15], [16, 1], [20, 0], [23, 4]],
    "large": [[10, 12], [13, 17], [16, 11], [16, 12], [16, 13], [17, 15], [17, 16], 
              [18, 4], [18, 11], [18, 19], [19, 11], [21, 2], [21, 3], [22, 3], 
              [22, 9], [22, 12], [23, 5], [23, 7], [23, 13], [25, 5]],
    "large-v2": [[10, 12], [13, 17], [16, 11], [16, 12], [16, 13], [17, 15], [17, 16], 
                 [18, 4], [18, 11], [18, 19], [19, 11], [21, 2], [21, 3], [22, 3], 
                 [22, 9], [22, 12], [23, 5], [23, 7], [23, 13], [25, 5]],
    "large-v3": [[10, 12], [13, 17], [16, 11], [16, 12], [16, 13], [17, 15], [17, 16], 
                 [18, 4], [18, 11], [18, 19], [19, 11], [21, 2], [21, 3], [22, 3], 
                 [22, 9], [22, 12], [23, 5], [23, 7], [23, 13], [25, 5]]
}


@dataclass
class EnhancedWordTiming:
    """Enhanced word timing with confidence and alignment quality metrics"""
    word: str
    start: float
    end: float
    confidence: float
    tokens: List[int]
    alignment_confidence: float = 1.0
    is_reliable: bool = True


class EnhancedWordTimestampExtractor:
    """
    Enhanced word-level timestamp extraction for MLX Whisper.
    Implements best practices from WhisperX and recent improvements.
    """
    
    def __init__(
        self,
        model: Whisper,
        tokenizer: Tokenizer,
        model_name: str = "large-v3",
        median_filter_width: int = 7,
        qk_scale: float = 1.0,
        max_word_duration: float = 2.0,
        min_word_duration: float = 0.02
    ):
        """
        Initialize enhanced word timestamp extractor.
        
        Args:
            model: MLX Whisper model
            tokenizer: Whisper tokenizer
            model_name: Model size for alignment head selection
            median_filter_width: Width of median filter for attention weights
            qk_scale: Scale factor for cross-attention softmax
            max_word_duration: Maximum allowed word duration
            min_word_duration: Minimum allowed word duration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = self._normalize_model_name(model_name)
        self.median_filter_width = median_filter_width
        self.qk_scale = qk_scale
        self.max_word_duration = max_word_duration
        self.min_word_duration = min_word_duration
        
        # Set alignment heads based on model size
        self._set_alignment_heads()
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to match alignment head keys"""
        # Handle various model name formats
        if "tiny" in model_name:
            return "tiny"
        elif "base" in model_name:
            return "base"
        elif "small" in model_name:
            return "small"
        elif "medium" in model_name:
            return "medium"
        elif "large-v3" in model_name:
            return "large-v3"
        elif "large-v2" in model_name:
            return "large-v2"
        elif "large" in model_name:
            return "large"
        else:
            warnings.warn(f"Unknown model name: {model_name}, defaulting to large-v3")
            return "large-v3"
    
    def _set_alignment_heads(self):
        """Set alignment heads for the model based on model size"""
        if self.model_name in ALIGNMENT_HEADS:
            alignment_heads = ALIGNMENT_HEADS[self.model_name]
            # Convert to mx.array format expected by the model
            self.model.alignment_heads = mx.array(alignment_heads)
        else:
            warnings.warn(f"No alignment heads for model: {self.model_name}")
            # Use default alignment heads from large model
            self.model.alignment_heads = mx.array(ALIGNMENT_HEADS["large"])
    
    def extract_word_timestamps(
        self,
        segments: List[Dict],
        mel: mx.array,
        num_frames: int,
        time_offset: float = 0.0
    ) -> List[Dict]:
        """
        Extract word-level timestamps for segments.
        
        Args:
            segments: List of transcription segments
            mel: Mel spectrogram
            num_frames: Number of audio frames
            time_offset: Time offset for segment
            
        Returns:
            Segments with added word-level timestamps
        """
        if not segments:
            return segments
        
        # Extract all tokens from segments
        all_tokens = []
        segment_token_bounds = []
        
        for segment in segments:
            tokens = [t for t in segment.get("tokens", []) if t < self.tokenizer.eot]
            segment_token_bounds.append((len(all_tokens), len(all_tokens) + len(tokens)))
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return segments
        
        # Get word alignments
        try:
            alignments = self._get_word_alignments(all_tokens, mel, num_frames)
            
            # Assign alignments to segments
            segment_idx = 0
            for i, (start_idx, end_idx) in enumerate(segment_token_bounds):
                segment_alignments = []
                
                # Collect alignments for this segment
                current_token_count = 0
                for alignment in alignments:
                    token_count = len(alignment.tokens)
                    if current_token_count >= start_idx and current_token_count < end_idx:
                        # Adjust timing with offset
                        enhanced_timing = EnhancedWordTiming(
                            word=alignment.word,
                            start=time_offset + alignment.start,
                            end=time_offset + alignment.end,
                            confidence=alignment.probability,
                            tokens=alignment.tokens,
                            alignment_confidence=self._calculate_alignment_confidence(alignment),
                            is_reliable=self._is_timing_reliable(alignment)
                        )
                        segment_alignments.append(enhanced_timing)
                    
                    current_token_count += token_count
                    if current_token_count >= end_idx:
                        break
                
                # Convert to dict format and add to segment
                segments[i]["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "confidence": round(w.confidence, 4),
                        "alignment_confidence": round(w.alignment_confidence, 4),
                        "is_reliable": w.is_reliable
                    }
                    for w in segment_alignments
                ]
                
                # Update segment boundaries based on words
                if segment_alignments:
                    segments[i]["start"] = segment_alignments[0].start
                    segments[i]["end"] = segment_alignments[-1].end
        
        except Exception as e:
            warnings.warn(f"Failed to extract word timestamps: {e}")
            # Return segments without word timestamps
        
        return segments
    
    def _get_word_alignments(
        self,
        tokens: List[int],
        mel: mx.array,
        num_frames: int
    ) -> List[WordTiming]:
        """Get word alignments using cross-attention weights"""
        # Use the existing find_alignment function with our parameters
        alignments = find_alignment(
            model=self.model,
            tokenizer=self.tokenizer,
            text_tokens=tokens,
            mel=mel,
            num_frames=num_frames,
            medfilt_width=self.median_filter_width,
            qk_scale=self.qk_scale
        )
        
        # Post-process alignments
        alignments = self._post_process_alignments(alignments)
        
        return alignments
    
    def _post_process_alignments(self, alignments: List[WordTiming]) -> List[WordTiming]:
        """Post-process alignments to improve quality"""
        if not alignments:
            return alignments
        
        # Calculate median word duration for reference
        durations = [a.end - a.start for a in alignments]
        median_duration = np.median(durations) if durations else 0.5
        
        # Fix unrealistic word durations
        for i, alignment in enumerate(alignments):
            duration = alignment.end - alignment.start
            
            # Fix too short words
            if duration < self.min_word_duration:
                # Extend to minimum duration
                center = (alignment.start + alignment.end) / 2
                alignment.start = center - self.min_word_duration / 2
                alignment.end = center + self.min_word_duration / 2
            
            # Fix too long words
            elif duration > self.max_word_duration:
                # Check if it's a sentence boundary
                is_sentence_end = any(alignment.word.endswith(p) for p in ".!?")
                
                if is_sentence_end:
                    # Truncate from the end
                    alignment.end = alignment.start + min(self.max_word_duration, median_duration * 2)
                else:
                    # Center and truncate
                    center = (alignment.start + alignment.end) / 2
                    new_duration = min(self.max_word_duration, median_duration * 2)
                    alignment.start = center - new_duration / 2
                    alignment.end = center + new_duration / 2
            
            # Ensure no overlap with previous word
            if i > 0 and alignment.start < alignments[i-1].end:
                gap = 0.01  # 10ms gap
                alignment.start = alignments[i-1].end + gap
        
        return alignments
    
    def _calculate_alignment_confidence(self, alignment: WordTiming) -> float:
        """Calculate confidence score for word alignment"""
        # Base confidence from token probabilities
        confidence = alignment.probability
        
        # Adjust based on word duration
        duration = alignment.end - alignment.start
        if duration < 0.05 or duration > 2.0:
            confidence *= 0.8  # Reduce confidence for unusual durations
        
        # Adjust based on word length
        word_length = len(alignment.word.strip())
        expected_duration = word_length * 0.08  # Rough estimate
        duration_ratio = duration / expected_duration if expected_duration > 0 else 1.0
        
        if 0.5 < duration_ratio < 2.0:
            confidence *= 1.0
        else:
            confidence *= 0.9
        
        return min(1.0, confidence)
    
    def _is_timing_reliable(self, alignment: WordTiming) -> bool:
        """Check if word timing is reliable"""
        duration = alignment.end - alignment.start
        
        # Check duration bounds
        if duration < 0.02 or duration > 3.0:
            return False
        
        # Check probability
        if alignment.probability < 0.5:
            return False
        
        return True


def extract_word_timestamps(
    segments: List[Dict],
    model: Whisper,
    tokenizer: Tokenizer,
    mel: mx.array,
    num_frames: int,
    model_name: str = "large-v3",
    **kwargs
) -> List[Dict]:
    """
    Convenience function to extract word timestamps.
    
    Args:
        segments: Transcription segments
        model: Whisper model
        tokenizer: Tokenizer
        mel: Mel spectrogram
        num_frames: Number of frames
        model_name: Model size name
        **kwargs: Additional parameters
        
    Returns:
        Segments with word timestamps
    """
    extractor = EnhancedWordTimestampExtractor(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        **kwargs
    )
    
    return extractor.extract_word_timestamps(
        segments=segments,
        mel=mel,
        num_frames=num_frames
    )