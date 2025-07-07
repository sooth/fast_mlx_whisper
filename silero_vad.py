#!/usr/bin/env python3
"""
Silero VAD wrapper for MLX
Provides Voice Activity Detection for audio preprocessing
"""

import torch
import numpy as np
from typing import List, Dict, Union
import warnings

# Suppress torch warnings when running on Apple Silicon
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class SileroVAD:
    """
    Wrapper for Silero Voice Activity Detection model.
    Uses PyTorch for now as Silero doesn't have MLX version yet.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 300,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30
    ):
        """
        Initialize Silero VAD.
        
        Args:
            threshold: Speech probability threshold
            sampling_rate: Audio sampling rate (must be 16000 for Silero)
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence to split segments
            window_size_samples: Window size for VAD processing
            speech_pad_ms: Padding around speech segments
        """
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        
        # Load Silero VAD model
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.model.eval()
        except Exception as e:
            warnings.warn(f"Failed to load Silero VAD model: {e}")
            self.model = None
            self.utils = None
    
    def __call__(
        self,
        audio: np.ndarray,
        return_seconds: bool = True
    ) -> List[Dict[str, Union[float, int]]]:
        """
        Detect speech segments in audio.
        
        Args:
            audio: Audio waveform as numpy array
            return_seconds: Return timestamps in seconds (True) or samples (False)
            
        Returns:
            List of speech segments with start/end times
        """
        if self.model is None:
            # Fallback: return entire audio as one segment
            duration = len(audio) / self.sampling_rate
            return [{
                'start': 0.0 if return_seconds else 0,
                'end': duration if return_seconds else len(audio)
            }]
        
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = torch.tensor(audio).float()
        
        # Ensure audio is 1D
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()
        
        # Get speech timestamps using Silero VAD
        try:
            # Use the get_speech_timestamps utility
            get_speech_timestamps = self.utils[0]
            
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                threshold=self.threshold,
                sampling_rate=self.sampling_rate,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                window_size_samples=self.window_size_samples,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=return_seconds
            )
            
            return speech_timestamps
            
        except Exception as e:
            warnings.warn(f"VAD processing failed: {e}")
            # Fallback: return entire audio
            duration = len(audio) / self.sampling_rate
            return [{
                'start': 0.0 if return_seconds else 0,
                'end': duration if return_seconds else len(audio)
            }]
    
    def get_speech_probs(self, audio: np.ndarray) -> np.ndarray:
        """
        Get frame-wise speech probabilities.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Array of speech probabilities
        """
        if self.model is None:
            return np.ones(len(audio) // self.window_size_samples)
        
        # Convert to torch
        audio_tensor = torch.from_numpy(audio).float()
        
        # Process in chunks
        num_chunks = len(audio) // self.window_size_samples
        probs = []
        
        for i in range(num_chunks):
            start = i * self.window_size_samples
            end = start + self.window_size_samples
            chunk = audio_tensor[start:end]
            
            with torch.no_grad():
                prob = self.model(chunk, self.sampling_rate).item()
                probs.append(prob)
        
        return np.array(probs)


# Alternative pure NumPy implementation for fallback
class SimpleEnergyVAD:
    """
    Simple energy-based VAD as fallback when Silero is not available.
    Much less accurate but works without external dependencies.
    """
    
    def __init__(
        self,
        threshold: float = 0.02,
        frame_duration_ms: int = 30,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 300,
        sampling_rate: int = 16000
    ):
        self.threshold = threshold
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sampling_rate * frame_duration_ms / 1000)
        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self.min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
        self.sampling_rate = sampling_rate
    
    def __call__(
        self,
        audio: np.ndarray,
        return_seconds: bool = True
    ) -> List[Dict[str, Union[float, int]]]:
        """
        Detect speech using energy thresholding.
        """
        # Calculate frame energies
        num_frames = len(audio) // self.frame_samples
        energies = []
        
        for i in range(num_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples
            frame = audio[start:end]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Dynamic threshold based on energy statistics
        if len(energies) > 0:
            energy_threshold = max(
                self.threshold,
                np.percentile(energies, 30)  # Bottom 30% considered silence
            )
        else:
            energy_threshold = self.threshold
        
        # Detect speech frames
        speech_frames = energies > energy_threshold
        
        # Apply minimum duration constraints
        segments = []
        in_speech = False
        start_frame = 0
        silence_count = 0
        
        for i, is_speech in enumerate(speech_frames):
            if not in_speech and is_speech:
                # Start of speech
                in_speech = True
                start_frame = i
                silence_count = 0
            elif in_speech and not is_speech:
                # Possible end of speech
                silence_count += 1
                if silence_count >= self.min_silence_frames:
                    # Confirm end of speech
                    duration = i - start_frame - silence_count
                    if duration >= self.min_speech_frames:
                        # Valid speech segment
                        start_time = start_frame * self.frame_duration_ms / 1000
                        end_time = (i - silence_count) * self.frame_duration_ms / 1000
                        
                        segments.append({
                            'start': start_time if return_seconds else int(start_time * self.sampling_rate),
                            'end': end_time if return_seconds else int(end_time * self.sampling_rate)
                        })
                    in_speech = False
            elif in_speech and is_speech:
                # Continue speech
                silence_count = 0
        
        # Handle final segment
        if in_speech:
            duration = len(speech_frames) - start_frame
            if duration >= self.min_speech_frames:
                start_time = start_frame * self.frame_duration_ms / 1000
                end_time = len(speech_frames) * self.frame_duration_ms / 1000
                
                segments.append({
                    'start': start_time if return_seconds else int(start_time * self.sampling_rate),
                    'end': end_time if return_seconds else int(end_time * self.sampling_rate)
                })
        
        # If no segments detected, return entire audio
        if not segments:
            duration = len(audio) / self.sampling_rate
            segments = [{
                'start': 0.0 if return_seconds else 0,
                'end': duration if return_seconds else len(audio)
            }]
        
        return segments