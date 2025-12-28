"""
Internal types for the audio processing pipeline.

These are intermediate representations that flow through pipeline stages.
They are converted to schema.py event types before leaving the audio module.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List
import numpy as np


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

SAMPLE_RATE = 16000  # 16kHz standard for speech recognition
WINDOW_SIZE_SEC = 30  # Rolling buffer size
OVERLAP_SEC = 10  # Overlap between windows
MIN_UTTERANCE_DURATION_SEC = 0.5  # Minimum utterance length
MAX_UTTERANCE_GAP_SEC = 1.0  # Max silence within same utterance


# =============================================================================
# VAD OUTPUT
# =============================================================================

@dataclass
class SpeechSegment:
    """
    Output from VAD stage.
    Contains raw audio for the detected speech region.
    """
    start_sec: float  # Offset from buffer start in seconds
    end_sec: float
    audio: np.ndarray  # Raw audio samples for this segment
    
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec
    
    @property
    def duration_ms(self) -> int:
        return int(self.duration_sec * 1000)


# =============================================================================
# DIARIZATION OUTPUT
# =============================================================================

@dataclass
class DiarizedSegment:
    """
    SpeechSegment with speaker identity assigned.
    """
    start_sec: float
    end_sec: float
    audio: np.ndarray
    speaker_id: str  # e.g., "speaker_0", "speaker_1"
    embedding: Optional[np.ndarray] = None  # Speaker embedding vector
    
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


# =============================================================================
# UTTERANCE (GROUPED SEGMENTS)
# =============================================================================

@dataclass
class Utterance:
    """
    Grouped consecutive segments from the same speaker.
    Ready for ASR and prosody analysis.
    """
    speaker_id: str
    start_sec: float
    end_sec: float
    audio: np.ndarray  # Concatenated audio from all segments
    
    # Metadata
    num_segments: int = 1
    has_internal_pauses: bool = False
    
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec
    
    @property
    def duration_ms(self) -> int:
        return int(self.duration_sec * 1000)
    
    def to_absolute_time(self, buffer_start: datetime) -> tuple[datetime, datetime]:
        """Convert relative times to absolute timestamps."""
        start = buffer_start + timedelta(seconds=self.start_sec)
        end = buffer_start + timedelta(seconds=self.end_sec)
        return start, end


# =============================================================================
# PROSODY FEATURES
# =============================================================================

@dataclass
class ProsodyResult:
    """
    Prosodic features extracted from an utterance.
    """
    # Pitch (F0)
    pitch_mean_hz: Optional[float] = None
    pitch_std_hz: Optional[float] = None
    pitch_min_hz: Optional[float] = None
    pitch_max_hz: Optional[float] = None
    pitch_contour: Optional[str] = None  # "rising", "falling", "flat", "variable"
    
    # Intensity
    intensity_mean_db: Optional[float] = None
    intensity_std_db: Optional[float] = None
    intensity_range_db: Optional[float] = None
    
    # Timing
    speech_rate_syl_per_sec: Optional[float] = None
    speech_rate_category: Optional[str] = None  # "slow", "normal", "fast"
    articulation_rate: Optional[float] = None
    
    # Voice quality
    jitter: Optional[float] = None  # Pitch perturbation
    shimmer: Optional[float] = None  # Amplitude perturbation
    hnr_db: Optional[float] = None  # Harmonics-to-noise ratio
    voice_quality: Optional[str] = None  # "modal", "breathy", "tense", "creaky"


# =============================================================================
# ASR RESULT
# =============================================================================

@dataclass
class ASRResult:
    """
    Transcription result from ASR model.
    """
    text: str
    language: str = "he"  # Hebrew
    confidence: Optional[float] = None
    
    # Word-level timing (if available)
    words: List[dict] = field(default_factory=list)  # [{"word": "שלום", "start": 0.1, "end": 0.3}]
    
    @property
    def word_count(self) -> int:
        if self.words:
            return len(self.words)
        return len(self.text.split()) if self.text else 0
    
    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()


# =============================================================================
# ENRICHED UTTERANCE (FINAL PIPELINE OUTPUT)
# =============================================================================

@dataclass
class EnrichedUtterance:
    """
    Fully processed utterance with all extracted features.
    This is the final output before conversion to schema events.
    
    Note: After conversion, the audio and transcript are discarded.
    Only structured metadata persists.
    """
    # Identity
    speaker_id: str
    start_sec: float
    end_sec: float
    
    # ASR
    transcript: Optional[str] = None
    word_count: int = 0
    asr_confidence: Optional[float] = None
    
    # Prosody
    prosody: Optional[ProsodyResult] = None
    
    # Vocalization type
    vocal_type: str = "speech"  # "speech", "cry", "laugh", "scream", "hum", "whisper"
    
    # ASD-relevant patterns
    is_echolalia: bool = False
    echolalia_type: Optional[str] = None  # "immediate", "delayed"
    echolalia_similarity: Optional[float] = None
    echolalia_source_offset_ms: Optional[int] = None  # How long ago was the echoed utterance
    
    is_perseveration: bool = False
    perseveration_count: Optional[int] = None  # How many times repeated
    
    # Turn-taking
    gap_before_ms: Optional[int] = None
    is_overlap: bool = False
    previous_speaker_id: Optional[str] = None
    
    # Embedding for similarity comparisons (not persisted)
    _embedding: Optional[np.ndarray] = field(default=None, repr=False)
    
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec
    
    @property
    def duration_ms(self) -> int:
        return int(self.duration_sec * 1000)
    
    def to_absolute_times(self, buffer_start: datetime) -> tuple[datetime, datetime]:
        """Convert relative times to absolute timestamps."""
        start = buffer_start + timedelta(seconds=self.start_sec)
        end = buffer_start + timedelta(seconds=self.end_sec)
        return start, end


# =============================================================================
# ACOUSTIC EVENT (NON-SPEECH)
# =============================================================================

@dataclass
class AcousticEvent:
    """
    Non-speech audio event detected in the stream.
    """
    start_sec: float
    end_sec: float
    event_type: str  # "bell", "door", "crash", "music", etc.
    confidence: float = 0.5
    intensity: str = "moderate"  # "low", "moderate", "high"
    
    @property
    def duration_ms(self) -> int:
        return int((self.end_sec - self.start_sec) * 1000)
