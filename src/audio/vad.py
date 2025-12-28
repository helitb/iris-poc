"""
Voice Activity Detection using Silero VAD.

Silero VAD is lightweight, fast, and works well for real-time applications.
https://github.com/snakers4/silero-vad
"""

from typing import List, Optional, Tuple
import numpy as np

from .types import SpeechSegment, SAMPLE_RATE


class SileroVAD:
    """
    Silero VAD wrapper for speech detection.
    
    Usage:
        vad = SileroVAD()
        segments = vad.process(audio_buffer)
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,  # 32ms at 16kHz
        speech_pad_ms: int = 30,
        merge_gap_ms: int = 250,
        drop_short_ms: int = 0,
        keep_gap_silence: bool = False,
        debug: bool = True,
    ):
        """
        Args:
            threshold: Speech probability threshold (0-1)
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence to split segments
            window_size_samples: VAD window size (512 or 1024 for 16kHz)
            speech_pad_ms: Padding around speech segments
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.merge_gap_ms = merge_gap_ms
        self.drop_short_ms = drop_short_ms
        self.keep_gap_silence = keep_gap_silence
        self.debug = debug
        
        self._model = None
        self._utils = None

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[DEBUG][SileroVAD] {message}")
    
    def _load_model(self):
        """Lazy load Silero VAD model."""
        if self._model is not None:
            return
        
        try:
            import torch
            
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,  # Use PyTorch model
                trust_repo=True,
            )
            
            self._model = model
            self._utils = utils
            
            # Get utility functions
            (
                self._get_speech_timestamps,
                self._save_audio,
                self._read_audio,
                self._VADIterator,
                self._collect_chunks,
            ) = utils
            
        except ImportError as e:
            raise ImportError(
                "Silero VAD requires torch. Install with: pip install torch"
            ) from e
        
        self._debug(
            f"Loaded Silero VAD (threshold={self.threshold}, "
            f"min_speech={self.min_speech_duration_ms}ms, "
            f"min_silence={self.min_silence_duration_ms}ms)"
        )
    
    def _merge_segments(
        self,
        segments: List["SpeechSegment"],
        sample_rate: int,
    ) -> List["SpeechSegment"]:
        if not segments:
            return []

        gap_sec = self.merge_gap_ms / 1000.0
        min_len_sec = self.drop_short_ms / 1000.0

        merged: List["SpeechSegment"] = []
        cur = segments[0]

        for nxt in segments[1:]:
            gap = nxt.start_sec - cur.end_sec

            if gap <= gap_sec:
                # Merge timing bounds
                new_start = cur.start_sec
                new_end = nxt.end_sec

                if self.keep_gap_silence:
                    # pad silence (zeros) to preserve relative time inside audio field
                    pad_samples = int(round(gap * sample_rate))
                    if pad_samples < 0:
                        pad_samples = 0
                    cur_audio = cur.audio
                    nxt_audio = nxt.audio
                    cur.audio = np.concatenate([cur_audio, np.zeros(pad_samples, dtype=cur_audio.dtype), nxt_audio])
                else:
                    # speech-only concatenation (best for embeddings)
                    cur.audio = np.concatenate([cur.audio, nxt.audio])

                cur.start_sec = new_start
                cur.end_sec = new_end
            else:
                # finalize current
                if (cur.end_sec - cur.start_sec) >= min_len_sec:
                    merged.append(cur)
                cur = nxt

        # finalize last
        if (cur.end_sec - cur.start_sec) >= min_len_sec:
            merged.append(cur)

        return merged

    def process(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> List[SpeechSegment]:
        """
        Detect speech segments in audio buffer.
        
        Args:
            audio: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate (should be 16000)
            
        Returns:
            List of SpeechSegment with detected speech regions
        """
        self._load_model()
        
        import torch
        
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        self._debug(
            f"Processing audio: samples={len(audio)}, sr={sample_rate}, "
            f"window={self.window_size_samples}"
        )
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio)
        
        # Get speech timestamps
        speech_timestamps = self._get_speech_timestamps(
            audio_tensor,
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            window_size_samples=self.window_size_samples,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,  # Return sample indices
        )
        self._debug(f"Silero returned {len(speech_timestamps)} speech regions")
        
        # Convert to SpeechSegment objects
        segments = []
        for ts in speech_timestamps:
            start_sample = ts['start']
            end_sample = ts['end']
            
            # Extract audio chunk
            audio_chunk = audio[start_sample:end_sample].copy()
            
            self._debug(
                f"Detected segment: start={start_sample/sample_rate:.2f}s, "
                f"end={end_sample/sample_rate:.2f}s, "
                f"duration={(end_sample - start_sample)/sample_rate:.2f}s"
            )
            
            segments.append(SpeechSegment(
                start_sec=start_sample / sample_rate,
                end_sec=end_sample / sample_rate,
                audio=audio_chunk,
            ))
        
        # Merge close segments to stabilize embeddings
        if self.merge_gap_ms and self.merge_gap_ms > 0:
            segments = self._merge_segments(segments, sample_rate=sample_rate)
            self._debug(f"Merged to {len(segments)} segments (gap={self.merge_gap_ms}ms)")
        else:
            self._debug(f"Detected {len(segments)} segments (no merge)")

        return segments
    
    def reset_states(self):
        """Reset VAD model states (for streaming)."""
        if self._model is not None:
            self._model.reset_states()



# =============================================================================
# FACTORY
# =============================================================================

def create_vad(backend: str = "silero", **kwargs) -> SileroVAD:
    """
    Create a VAD processor.
    
    Args:
        backend: "silero" 
        **kwargs: Backend-specific configuration
        
    Returns:
        VAD processor instance
    """
    return SileroVAD(**kwargs)
