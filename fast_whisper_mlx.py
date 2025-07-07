#!/usr/bin/env python3
"""
Fast and accurate MLX Whisper implementation with precise word-level timestamps
Combines all improvements: VAD, enhanced timestamps, and optimizations
"""

import numpy as np
import mlx.core as mx
from typing import List, Dict, Optional, Union, Tuple
import time
import warnings

from .vad_transcribe_final import ProductionVADTranscriber
from .enhanced_word_timestamps import EnhancedWordTimestampExtractor, ALIGNMENT_HEADS
from .transcribe import transcribe_audio, ModelHolder
from .audio import SAMPLE_RATE, load_audio, log_mel_spectrogram, pad_or_trim, N_FRAMES
from .tokenizer import get_tokenizer, LANGUAGES
from .load_models import load_model
from .decoding import DecodingOptions, DecodingResult


class FastWhisperMLX:
    """
    Fast and accurate MLX Whisper with word-level timestamps.
    
    Features:
    - VAD-based segmentation to prevent hallucinations
    - Enhanced word-level timestamp extraction
    - Optimized for speed on Apple Silicon
    - Production-ready error handling
    """
    
    def __init__(
        self,
        model_name: str = "large-v3",
        device: Optional[str] = None,  # For compatibility
        compute_type: Optional[str] = None,  # For compatibility
        batch_size: int = 8,
        use_vad: bool = True,
        vad_threshold: float = 0.5,
        max_segment_duration: float = 10.0
    ):
        """
        Initialize Fast MLX Whisper.
        
        Args:
            model_name: Model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Ignored (for API compatibility)
            compute_type: Ignored (for API compatibility)
            batch_size: Batch size for processing
            use_vad: Whether to use VAD segmentation
            vad_threshold: VAD activation threshold
            max_segment_duration: Maximum segment duration for VAD
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_vad = use_vad
        
        # Map model names to MLX paths
        self.model_path = self._get_model_path(model_name)
        
        # Load model
        self.dtype = mx.float16
        self.model = load_model(self.model_path, dtype=self.dtype)
        
        # Initialize VAD if requested
        if use_vad:
            self.vad_transcriber = ProductionVADTranscriber(
                model_path=self.model_path,
                vad_threshold=vad_threshold,
                max_segment_duration=max_segment_duration,
                optimal_segment_duration=5.0
            )
        
        # Cache tokenizer
        self._tokenizer = None
    
    def _get_model_path(self, model_name: str) -> str:
        """Get MLX model path from model name"""
        # Handle local paths
        if "/" in model_name or model_name.startswith("~"):
            return model_name
        
        # Standard mappings
        model_mapping = {
            "tiny": "mlx-community/whisper-tiny",
            "tiny.en": "mlx-community/whisper-tiny.en",
            "base": "mlx-community/whisper-base",
            "base.en": "mlx-community/whisper-base.en",
            "small": "mlx-community/whisper-small",
            "small.en": "mlx-community/whisper-small.en",
            "medium": "mlx-community/whisper-medium",
            "medium.en": "mlx-community/whisper-medium.en",
            "large": "mlx-community/whisper-large",
            "large-v1": "mlx-community/whisper-large-v1",
            "large-v2": "mlx-community/whisper-large-v2",
            "large-v3": "mlx-community/whisper-large-v3",
        }
        
        return model_mapping.get(model_name, f"mlx-community/whisper-{model_name}")
    
    def get_tokenizer(self, language: Optional[str] = None, task: str = "transcribe"):
        """Get or create tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(
                self.model.is_multilingual,
                num_languages=self.model.num_languages,
                language=language,
                task=task
            )
        return self._tokenizer
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: Optional[int] = None,
        best_of: Optional[int] = None,
        temperature: Union[float, List[float]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        word_timestamps: bool = True,
        condition_on_previous_text: bool = False,
        initial_prompt: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ) -> Dict:
        """
        Transcribe audio with word-level timestamps.
        
        Args:
            audio: Path to audio file or numpy array
            language: Language code (None for auto-detection)
            task: "transcribe" or "translate"
            beam_size: Beam size for beam search
            best_of: Number of candidates for sampling
            temperature: Temperature(s) for sampling
            word_timestamps: Extract word-level timestamps
            condition_on_previous_text: Use previous text as context
            initial_prompt: Initial prompt for decoder
            verbose: Print progress
            **kwargs: Additional decoding options
            
        Returns:
            Dictionary with transcription results including word timestamps
        """
        start_time = time.time()
        
        # Load audio if path
        if isinstance(audio, str):
            if verbose:
                print(f"Loading audio from: {audio}")
            audio_array = np.array(load_audio(audio))
        else:
            audio_array = audio
        
        # Use VAD transcription if enabled
        if self.use_vad:
            if verbose:
                print("Using VAD-based transcription...")
            
            # VAD transcriber handles everything
            result = self.vad_transcriber.transcribe(
                audio=audio_array,
                language=language,
                task=task,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                verbose=verbose
            )
            
            # Extract segments from VAD result
            segments = result.get('segments', [])
            vad_segments = result.get('vad_segments', [])
            
            # Convert segments to dictionary format if needed
            dict_segments = []
            for seg in segments:
                if isinstance(seg, dict):
                    dict_segments.append(seg)
                elif isinstance(seg, list) and len(seg) == 3:
                    # Convert [start_seek, end_seek, text] to dictionary
                    dict_segments.append({
                        'start': seg[0] / 1000.0,  # Convert ms to seconds
                        'end': seg[1] / 1000.0,
                        'text': seg[2],
                        'tokens': []  # Will be filled if word timestamps requested
                    })
            segments = dict_segments
            
        else:
            # Standard transcription without VAD
            result = transcribe_audio(
                audio=audio_array,
                path_or_hf_repo=self.model_path,
                language=language,
                task=task,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                batch_size=self.batch_size,
                verbose=verbose,
                **kwargs
            )
            segments = result.get('segments', [])
            vad_segments = []
            
            # Convert segments to dictionary format if needed
            dict_segments = []
            for seg in segments:
                if isinstance(seg, dict):
                    dict_segments.append(seg)
                elif isinstance(seg, list) and len(seg) == 3:
                    # Convert [start_seek, end_seek, text] to dictionary
                    dict_segments.append({
                        'start': seg[0] / 1000.0,  # Convert ms to seconds
                        'end': seg[1] / 1000.0,
                        'text': seg[2],
                        'tokens': []  # Will be filled if word timestamps requested
                    })
            segments = dict_segments
        
        # Add word timestamps if requested
        if word_timestamps and segments:
            if verbose:
                print("Extracting word-level timestamps...")
            
            try:
                # Get mel spectrogram for the full audio
                mel = log_mel_spectrogram(audio_array, n_mels=self.model.dims.n_mels)
                
                # Get tokenizer
                tokenizer = self.get_tokenizer(language or result.get('language'), task)
                
                # Initialize word timestamp extractor
                extractor = EnhancedWordTimestampExtractor(
                    model=self.model,
                    tokenizer=tokenizer,
                    model_name=self.model_name
                )
                
                # Process each segment
                for i, segment in enumerate(segments):
                    if 'tokens' in segment:
                        # For VAD segments, we need to extract the relevant mel frames
                        if vad_segments and i < len(vad_segments):
                            vad_seg = vad_segments[i]
                            start_frame = int(vad_seg['start'] * mel.shape[-1] / (len(audio_array) / SAMPLE_RATE))
                            end_frame = int(vad_seg['end'] * mel.shape[-1] / (len(audio_array) / SAMPLE_RATE))
                            segment_mel = mel[:, start_frame:end_frame]
                            num_frames = end_frame - start_frame
                        else:
                            # Use full mel for non-VAD segments
                            segment_mel = mel
                            num_frames = mel.shape[-1]
                        
                        # Extract word timestamps for this segment
                        segments[i:i+1] = extractor.extract_word_timestamps(
                            segments[i:i+1],
                            segment_mel,
                            num_frames,
                            time_offset=segment.get('start', 0)
                        )
                
                result['segments'] = segments
                
            except Exception as e:
                warnings.warn(f"Failed to extract word timestamps: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Add processing time
        result['processing_time'] = time.time() - start_time
        
        # Ensure consistent output format
        if 'text' not in result:
            result['text'] = ' '.join(seg.get('text', '') for seg in segments)
        
        if verbose:
            print(f"Transcription completed in {result['processing_time']:.1f}s")
            if word_timestamps:
                total_words = sum(len(seg.get('words', [])) for seg in segments)
                print(f"Extracted timestamps for {total_words} words")
        
        return result
    
    def detect_language(self, audio: Union[str, np.ndarray]) -> Tuple[str, Dict[str, float]]:
        """
        Detect language of audio.
        
        Args:
            audio: Path to audio file or numpy array
            
        Returns:
            Detected language code and probability distribution
        """
        # Load audio if path
        if isinstance(audio, str):
            audio_array = np.array(load_audio(audio))
        else:
            audio_array = audio
        
        # Get mel spectrogram
        mel = log_mel_spectrogram(audio_array, n_mels=self.model.dims.n_mels)
        mel = pad_or_trim(mel, N_FRAMES)
        
        # Detect language
        _, probs = self.model.detect_language(mel)
        
        # Get top language
        language = max(probs, key=probs.get)
        
        return language, probs
    
    def transcribe_with_vad(
        self,
        audio: Union[str, np.ndarray],
        **kwargs
    ) -> Dict:
        """Force transcription with VAD regardless of initialization setting"""
        original_use_vad = self.use_vad
        self.use_vad = True
        
        try:
            result = self.transcribe(audio, **kwargs)
        finally:
            self.use_vad = original_use_vad
        
        return result


def create_fast_whisper_mlx(
    model_name: str = "large-v3",
    **kwargs
) -> FastWhisperMLX:
    """
    Create a Fast MLX Whisper instance.
    
    Args:
        model_name: Model size
        **kwargs: Additional initialization parameters
        
    Returns:
        FastWhisperMLX instance
    """
    return FastWhisperMLX(model_name=model_name, **kwargs)