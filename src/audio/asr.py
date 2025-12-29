"""
Automatic Speech Recognition for Hebrew using faster-whisper.

Primary model: ivrit-ai/faster-whisper-v2-d4 (optimized for Hebrew)
Fallback: OpenAI Whisper models via faster-whisper
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


from .types import ASRResult, SAMPLE_RATE


class HebrewASR:
    """
    Hebrew ASR using faster-whisper with ivrit-ai model.
    
    Usage:
        asr = HebrewASR()
        result = asr.transcribe(audio)
    """
    
    # Model options for Hebrew
    MODELS = {
        "ivrit-v2-d4": "ivrit-ai/faster-whisper-v2-d4",  # Best for Hebrew
        "ivrit-v2-d3": "ivrit-ai/faster-whisper-v2-d3",
        "whisper-large-v3": "large-v3",  # Multilingual fallback
        "whisper-medium": "medium",
        "whisper-small": "small",  # Lightweight option
    }
    
    def __init__(
        self,
        model_name: str = "whisper-medium",
        device: str = "auto",  # "cpu", "cuda", "auto"
        compute_type: str = "auto",  # "float16", "int8", "auto"
        language: str = "he",
        beam_size: int = 5,
        vad_filter: bool = True,
        debug: bool = False,
        cpu_threads: int | None = None,
        num_workers: int = 1,
    ):
        """
        Args:
            model_name: Model identifier (see MODELS)
            device: Compute device
            compute_type: Quantization type for inference
            language: Target language code
            beam_size: Beam search width
            vad_filter: Whether to use built-in VAD filtering
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.debug = debug
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        
        self._model = None

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[DEBUG][ASR] {message}")
    
    def _load_model(self):
        """Lazy load the ASR model."""
        if self._model is not None:
            return
        
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper required. Install with: pip install faster-whisper"
            ) from e
        
        # Resolve model path
        model_path = self.MODELS.get(self.model_name, self.model_name)
        
        # Determine device
        device = self.device
        compute_type = self.compute_type
        
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"Loading ASR model: {model_path} on {device} ({compute_type})")
        self._debug(
            f"Initializing faster-whisper model={model_path} device={device} "
            f"compute_type={compute_type}"
        )
        
        # macOS/conda environments sometimes double-load OpenMP (torch + ctranslate2)
        # which raises an abort; allow duplicates as a pragmatic workaround.
        import os
        if os.environ.get("KMP_DUPLICATE_LIB_OK") is None:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            self._debug("Set KMP_DUPLICATE_LIB_OK=TRUE to avoid OpenMP conflict")

        # Limit thread explosion on CPU to reduce crash risk
        if device == "cpu":
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                if os.environ.get(var) is None:
                    os.environ[var] = "1"
            cpu_threads = self.cpu_threads or max(1, (os.cpu_count() or 2) // 2)
        else:
            cpu_threads = 0  # let GPU handle threads
        
        self._model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=self.num_workers,
        )
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        word_timestamps: bool = True,
    ) -> ASRResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio samples (float32, mono)
            sample_rate: Sample rate of audio
            word_timestamps: Whether to extract word-level timing
            
        Returns:
            ASRResult with transcription and metadata
        """
        self._debug(
            f"SKIPPING Transcribing audio len={len(audio)} sr={sample_rate} "
            f"DEBUG DEBUG"
        )
        return ASRResult(
            text="ASR transcription skipped in debug mode.",
            language=self.language,
            confidence=None,
            words=[],
        )
    '''
        self._load_model()
        self._debug(
            f"Transcribing audio len={len(audio)} sr={sample_rate} "
            f"word_timestamps={word_timestamps}"
        )
        
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, SAMPLE_RATE)
            self._debug(f"Resampled audio from {sample_rate} to {SAMPLE_RATE}")
        
        # Run transcription
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            word_timestamps=word_timestamps,
        )
        
        # Collect results
        full_text = []
        words = []
        total_confidence = 0.0
        segment_count = 0
        
        for segment in segments:
            full_text.append(segment.text.strip())
            
            # Track confidence
            if hasattr(segment, 'avg_logprob'):
                # Convert log probability to confidence
                confidence = np.exp(segment.avg_logprob)
                total_confidence += confidence
                segment_count += 1
            
            # Collect word timestamps if available
            if word_timestamps and hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    words.append({
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                        "probability": getattr(word, 'probability', None),
                    })
        
        # Compute average confidence
        avg_confidence = total_confidence / segment_count if segment_count > 0 else None
        self._debug(
            f"Transcription complete: segments={segment_count}, "
            f"avg_confidence={avg_confidence}, text_len={len(' '.join(full_text))}"
        )
        
        return ASRResult(
            text=" ".join(full_text),
            language=info.language if hasattr(info, 'language') else self.language,
            confidence=avg_confidence,
            words=words,
        )
        '''
        
    def _resample(
        self,
        audio: np.ndarray,
        source_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)
        except ImportError:
            # Simple resampling fallback
            ratio = target_sr / source_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)
    
    def transcribe_with_diarization(
        self,
        audio: np.ndarray,
        speaker_segments: List[Tuple[float, float, str]],
    ) -> List[Tuple[str, str, float, float]]:
        """
        Transcribe with pre-computed speaker diarization.
        
        Args:
            audio: Full audio buffer
            speaker_segments: List of (start_sec, end_sec, speaker_id)
            
        Returns:
            List of (speaker_id, text, start_sec, end_sec)
        """
        results = []
        
        for start, end, speaker in speaker_segments:
            start_sample = int(start * SAMPLE_RATE)
            end_sample = int(end * SAMPLE_RATE)
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) > SAMPLE_RATE * 0.3:  # Min 300ms
                result = self.transcribe(segment_audio, word_timestamps=False)
                if not result.is_empty:
                    results.append((speaker, result.text, start, end))
        
        return results


class WhisperASR:
    """
    Alternative ASR using OpenAI Whisper directly (not faster-whisper).
    Useful for environments where faster-whisper has issues.
    """
    
    def __init__(
        self,
        model_size: str = "small",
        language: str = "en",
        device: str = "auto",
        debug: bool = False,
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.debug = debug
        self._model = None

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[DEBUG][WhisperASR] {message}")
    
    def _load_model(self):
        if self._model is not None:
            return
        
        try:
            import whisper as openai_whisper  # pip package: openai-whisper
        except ImportError as e:
            raise ImportError(
                "OpenAI Whisper required. Install with: pip install \"openai-whisper>=20231117\""
            ) from e

        # Avoid OpenMP duplicate load crashes (whisper + torch on macOS)
        import os
        if os.environ.get("KMP_DUPLICATE_LIB_OK") is None:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            self._debug("Set KMP_DUPLICATE_LIB_OK=TRUE to avoid OpenMP conflict")
        
        device = self.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        self._debug(f"Loading OpenAI Whisper model={self.model_size} device={device}")
        self._model = openai_whisper.load_model(self.model_size, device=device)
    
    def transcribe(self, audio: np.ndarray) -> ASRResult:
        """Transcribe using OpenAI Whisper."""
        self._load_model()
        self._debug(f"Transcribing audio len={len(audio)}")
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        result = self._model.transcribe(
            audio,
            language=self.language,
            fp16=False,  # Use fp32 for CPU
        )
        self._debug(f"Transcription complete; text_len={len(result.get('text', ''))}")
        
        return ASRResult(
            text=result["text"].strip(),
            language=result.get("language", self.language),
            confidence=None,
            words=[],
        )


# =============================================================================
# LINGUISTIC ANALYSIS HELPERS
# =============================================================================

def compute_verbal_complexity(word_count: int) -> str:
    """
    Determine verbal complexity from word count.
    Maps to schema.VerbalComplexity values.
    """
    if word_count == 0:
        return "vocalization"
    elif word_count == 1:
        return "single_word"
    elif word_count <= 3:
        return "phrase"
    elif word_count <= 8:
        return "sentence"
    else:
        return "multi_sentence"


def detect_question(text: str) -> bool:
    """Detect if text is a question (Hebrew or English)."""
    # Hebrew question markers
    hebrew_question_words = ['מה', 'מי', 'איפה', 'למה', 'מתי', 'איך', 'האם', 'כמה']
    
    # Check punctuation
    if text.strip().endswith('?'):
        return True
    
    # Check Hebrew question words
    words = text.split()
    if words and words[0] in hebrew_question_words:
        return True
    
    return False


def detect_protest_markers(text: str) -> bool:
    """Detect protest/refusal markers in Hebrew."""
    protest_words = ['לא', 'אל', 'לא רוצה', 'עזוב', 'די', 'מספיק', 'לך']
    text_lower = text.lower()
    return any(word in text_lower for word in protest_words)


# =============================================================================
# FACTORY
# =============================================================================

def create_asr(
    backend: str = "faster-whisper",
    model: str = "ivrit-v2-d4",
    **kwargs,
) -> "HebrewASR | WhisperASR":
    """
    Create an ASR processor.
    
    Args:
        backend: "faster-whisper" (recommended) or "whisper"
        model: Model name/size
        **kwargs: Backend-specific configuration
        
    Returns:
        ASR processor instance
    """
    if backend == "faster-whisper":
        return HebrewASR(model_name=model, **kwargs)
    elif backend == "whisper":
        return WhisperASR(model_size=model, **kwargs)
    else:
        raise ValueError(f"Unknown ASR backend: {backend}")
