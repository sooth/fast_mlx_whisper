#!/usr/bin/env python3
"""
Production-ready VAD-enhanced transcription for MLX Whisper
Implements Phase 1 & 2 of the WhisperX port plan with proven improvements
"""

import numpy as np
import mlx.core as mx
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import warnings
import re

from .transcribe import transcribe_audio as original_transcribe
from .audio import SAMPLE_RATE, log_mel_spectrogram, N_SAMPLES
from .silero_vad import SileroVAD


@dataclass
class SpeechSegment:
    """Represents a segment of speech detected by VAD"""
    start: float
    end: float
    start_sample: int
    end_sample: int
    

class ProductionVADTranscriber:
    """
    Production-ready Whisper transcription with VAD preprocessing.
    Addresses hallucination issues through smart segmentation.
    """
    
    def __init__(
        self,
        model_path: str = "mlx-community/whisper-large-v3",
        vad_threshold: float = 0.5,
        min_speech_duration: float = 0.2,
        min_silence_duration: float = 0.3,
        speech_pad: float = 0.3,
        max_segment_duration: float = 8.0,  # Reduced from 10s based on testing
        optimal_segment_duration: float = 5.0,  # Sweet spot for accuracy
        hallucination_words: List[str] = ["the", "and", "a", "you", "thank"],
        hallucination_threshold: int = 20  # Max repetitions before considered hallucination
    ):
        self.model_path = model_path
        self.vad_threshold = vad_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.speech_pad = speech_pad
        self.max_segment_duration = max_segment_duration
        self.optimal_segment_duration = optimal_segment_duration
        self.hallucination_words = hallucination_words
        self.hallucination_threshold = hallucination_threshold
        
        # Initialize VAD
        self.vad = SileroVAD(
            threshold=vad_threshold,
            min_speech_duration_ms=int(min_speech_duration * 1000),
            min_silence_duration_ms=int(min_silence_duration * 1000)
        )
        
    def detect_speech_segments(self, audio: np.ndarray) -> List[SpeechSegment]:
        """
        Detect speech segments with smart splitting for optimal transcription.
        """
        # Get VAD timestamps
        vad_segments = self.vad(audio, return_seconds=True)
        
        if not vad_segments:
            warnings.warn("No speech detected in audio")
            return []
        
        segments = []
        audio_duration = len(audio) / SAMPLE_RATE
        
        for segment in vad_segments:
            start = max(0, segment['start'] - self.speech_pad)
            end = min(audio_duration, segment['end'] + self.speech_pad)
            duration = end - start
            
            # Smart splitting based on duration
            if duration > self.max_segment_duration:
                # Calculate optimal number of chunks
                num_chunks = int(np.ceil(duration / self.optimal_segment_duration))
                chunk_duration = duration / num_chunks
                
                for i in range(num_chunks):
                    chunk_start = start + i * chunk_duration
                    chunk_end = start + (i + 1) * chunk_duration
                    
                    # Add small overlap to avoid cutting words
                    if i > 0:
                        chunk_start -= 0.1  # 100ms overlap
                    if i < num_chunks - 1:
                        chunk_end += 0.1
                    
                    segments.append(SpeechSegment(
                        start=max(0, chunk_start),
                        end=min(end, chunk_end),
                        start_sample=int(max(0, chunk_start) * SAMPLE_RATE),
                        end_sample=int(min(end, chunk_end) * SAMPLE_RATE)
                    ))
            else:
                segments.append(SpeechSegment(
                    start=start,
                    end=end,
                    start_sample=int(start * SAMPLE_RATE),
                    end_sample=int(end * SAMPLE_RATE)
                ))
        
        return self._merge_overlapping_segments(segments)
    
    def _merge_overlapping_segments(self, segments: List[SpeechSegment]) -> List[SpeechSegment]:
        """Merge overlapping segments while respecting max duration."""
        if not segments:
            return segments
        
        merged = []
        current = segments[0]
        
        for segment in segments[1:]:
            # Check if can merge without exceeding max duration
            potential_duration = segment.end - current.start
            
            if (segment.start <= current.end + 0.1 and 
                potential_duration <= self.max_segment_duration):
                # Merge
                current.end = max(current.end, segment.end)
                current.end_sample = max(current.end_sample, segment.end_sample)
            else:
                merged.append(current)
                current = segment
        
        merged.append(current)
        return merged
    
    def detect_hallucination(self, text: str) -> bool:
        """
        Quick hallucination detection based on repetition patterns.
        """
        if not text or len(text) < 20:
            return False
        
        words = text.lower().split()
        
        # Check for extreme repetition of common words
        for word in self.hallucination_words:
            if words.count(word) > self.hallucination_threshold:
                return True
        
        # Check for pattern like "word word word word"
        if len(words) > 4:
            # Check if 80% of words are the same
            most_common = max(set(words), key=words.count)
            if words.count(most_common) / len(words) > 0.8:
                return True
        
        return False
    
    def transcribe_segment_safe(
        self,
        audio: np.ndarray,
        segment: SpeechSegment,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Transcribe a segment with fallback for hallucinations.
        """
        # First attempt with temperature 0
        # Filter out parameters that would conflict
        filtered_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['temperature', 'batch_size', 'compression_ratio_threshold', 
                        'no_speech_threshold', 'condition_on_previous_text', 
                        'language', 'verbose']
        }
        
        result = original_transcribe(
            audio=audio,
            path_or_hf_repo=self.model_path,
            language=language,
            temperature=0.0,
            compression_ratio_threshold=2.0,
            no_speech_threshold=0.3,
            condition_on_previous_text=False,
            batch_size=kwargs.get('batch_size', 1),
            verbose=False,
            **filtered_kwargs
        )
        
        text = result.get('text', '').strip()
        
        # If hallucination detected, try higher temperature
        if self.detect_hallucination(text):
            result = original_transcribe(
                audio=audio,
                path_or_hf_repo=self.model_path,
                language=language,
                temperature=0.2,
                compression_ratio_threshold=2.0,
                no_speech_threshold=0.3,
                condition_on_previous_text=False,
                batch_size=kwargs.get('batch_size', 1),
                verbose=False,
                **filtered_kwargs
            )
            
            # If still hallucinating, segment is probably problematic
            if self.detect_hallucination(result.get('text', '')):
                # Return empty result for this segment
                return {'text': '[inaudible]', 'segments': []}
        
        return result
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = False,
        **kwargs
    ) -> Dict:
        """
        Transcribe audio with VAD preprocessing and smart segmentation.
        """
        # Load audio if path provided
        if isinstance(audio, str):
            from .audio import load_audio
            audio_array = np.array(load_audio(audio))
        else:
            audio_array = audio
        
        # Detect speech segments
        if verbose:
            print("Detecting speech segments with VAD...")
        
        segments = self.detect_speech_segments(audio_array)
        
        if not segments:
            return {
                "text": "",
                "segments": [],
                "language": language or "en"
            }
        
        if verbose:
            total_speech = sum(s.end - s.start for s in segments)
            print(f"Found {len(segments)} speech segments, total duration: {total_speech:.1f}s")
            print(f"Average segment duration: {total_speech/len(segments):.1f}s")
        
        # Transcribe each segment
        all_segments = []
        all_text = []
        
        for i, segment in enumerate(segments):
            if verbose:
                duration = segment.end - segment.start
                print(f"\nSegment {i+1}/{len(segments)}: {segment.start:.1f}s - {segment.end:.1f}s ({duration:.1f}s)")
            
            # Extract audio segment
            segment_audio = audio_array[segment.start_sample:segment.end_sample]
            
            # Skip very short segments
            if len(segment_audio) < SAMPLE_RATE * 0.1:
                continue
            
            # Transcribe with safety checks
            result = self.transcribe_segment_safe(
                segment_audio, segment, language, task=task, **kwargs
            )
            
            # Adjust timestamps
            if isinstance(result, dict) and result.get('segments'):
                for seg in result['segments']:
                    if isinstance(seg, dict):
                        # Dictionary format segment
                        seg['start'] = seg.get('start', 0) + segment.start
                        seg['end'] = seg.get('end', 0) + segment.start
                        
                        if 'words' in seg:
                            for word in seg['words']:
                                word['start'] += segment.start
                                word['end'] += segment.start
                        
                        all_segments.append(seg)
                    elif isinstance(seg, list) and len(seg) == 3:
                        # List format [start_seek, end_seek, text]
                        # Convert to dictionary format with adjusted timestamps
                        dict_seg = {
                            'start': seg[0] / 1000.0 + segment.start,  # Convert ms to seconds
                            'end': seg[1] / 1000.0 + segment.start,
                            'text': seg[2]
                        }
                        all_segments.append(dict_seg)
                    else:
                        # Unknown format, skip
                        warnings.warn(f"Unknown segment format: {type(seg)}")
            
            # Collect text
            if isinstance(result, dict):
                segment_text = result.get('text', '').strip()
                if segment_text and segment_text != '[inaudible]':
                    # Remove potential overlap at boundaries
                    if i > 0 and all_text and len(segment_text.split()) > 3:
                        # Check if first words match last words of previous segment
                        prev_words = all_text[-1].split()[-3:] if all_text else []
                        curr_words = segment_text.split()[:3]
                        
                        # Simple overlap detection
                        if prev_words and curr_words and prev_words[-1].lower() == curr_words[0].lower():
                            # Skip first word
                            segment_text = ' '.join(segment_text.split()[1:])
                    
                    all_text.append(segment_text)
        
        # Combine results
        final_result = {
            "text": " ".join(all_text),
            "segments": all_segments,
            "language": language or 'en',
            "vad_segments": [{
                "start": s.start,
                "end": s.end
            } for s in segments]
        }
        
        if verbose:
            print(f"\nTranscription complete. Total text length: {len(final_result['text'])} chars")
        
        return final_result


def transcribe_with_vad_final(
    audio: Union[str, np.ndarray],
    model_path: str = "mlx-community/whisper-large-v3",
    vad_threshold: float = 0.5,
    max_segment_duration: float = 8.0,
    **kwargs
) -> Dict:
    """
    Production-ready VAD-enhanced transcription.
    
    Args:
        audio: Path to audio file or numpy array
        model_path: Path to MLX Whisper model
        vad_threshold: VAD activation threshold (0-1)
        max_segment_duration: Maximum segment duration before splitting
        **kwargs: Additional arguments for transcription
        
    Returns:
        Transcription results with text, segments, and metadata
    """
    transcriber = ProductionVADTranscriber(
        model_path=model_path,
        vad_threshold=vad_threshold,
        max_segment_duration=max_segment_duration
    )
    
    return transcriber.transcribe(audio, **kwargs)