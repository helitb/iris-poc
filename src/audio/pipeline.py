"""Main audio processing pipeline orchestrator.

Coordinates all stages:
VAD → Diarization → Segmentation → ASR → Prosody → Enrichment → Conversion

Critical privacy principle: Raw audio never leaves this module.
Only structured events are returned.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
import sys
import numpy as np

# Allow running as a script (python audio/pipeline.py) by ensuring the package
# root is on sys.path so the relative imports below resolve.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "audio"

from .types import (
    Utterance, EnrichedUtterance,
    AcousticEvent, SAMPLE_RATE, WINDOW_SIZE_SEC, OVERLAP_SEC,
)
from .vad import SileroVAD, create_vad
from .diarization import SpeakerDiarizer, create_diarizer
from .segmentation import UtteranceSegmenter, TurnTakingAnalyzer, create_segmenter
from .asr import HebrewASR, create_asr
from .prosody import ProsodyExtractor, create_prosody_extractor
from .enrichment import UtteranceEnricher, create_enricher
from .conversion import EventConverter, create_converter



@dataclass
class PipelineConfig:
    """Configuration for the audio pipeline."""
    
    # Audio settings
    sample_rate: int = SAMPLE_RATE
    window_size_sec: float = WINDOW_SIZE_SEC
    overlap_sec: float = OVERLAP_SEC
    debug: bool = False
    
    # Component backends
    vad_backend: str = "silero"
    diarization_backend: str = "speechbrain"
    asr_model: str = "whisper-medium" #"ivrit-v2-d4"
    
    # Feature flags
    enable_prosody: bool = True
    enable_echolalia_detection: bool = False
    enable_perseveration_detection: bool = False
    
    # Privacy
    retain_transcripts: bool = True  # Should be False in production
    
    # Processing    
    min_utterance_duration: float = 0.5
    max_utterance_gap: float = 1.0


@dataclass
class PipelineResult:
    """Result from processing an audio window."""
    
    # Events (privacy-safe output)
    speech_events: List[EnrichedUtterance] = field(default_factory=list)
    ambient_events: List[AcousticEvent] = field(default_factory=list)
    
    # Metadata
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    processing_time_ms: int = 0
    
    # Stats
    num_utterances: int = 0
    speakers_detected: int = 0
    
    @property
    def all_events(self) -> List:
        """All events in chronological order."""
        return sorted(
            self.speech_events + self.ambient_events,
            key=lambda e: e.timestamp
        )


class AudioPipeline:
    """
    Main audio processing pipeline.
    
    Usage:
        pipeline = AudioPipeline()
        
        # Process a 30-second audio window
        result = pipeline.process_window(audio_buffer, timestamp)
        
        # Events are privacy-safe - raw audio is discarded
        for event in result.speech_events:
            print(event)
    
    Streaming usage:
        for event in pipeline.process_streaming(audio_buffer, timestamp):
            yield event  # Events emitted as they're ready
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        
        # Initialize components (lazy loading)
        self._vad: Optional[SileroVAD] = None
        self._diarizer: Optional[SpeakerDiarizer] = None
        self._segmenter: Optional[UtteranceSegmenter] = None
        self._asr: Optional[HebrewASR] = None
        self._prosody: Optional[ProsodyExtractor] = None
        self._enricher: Optional[UtteranceEnricher] = None
        self._converter: Optional[EventConverter] = None
        
        # Turn-taking analyzer (stateful across windows)
        self._turn_analyzer = TurnTakingAnalyzer()
        
        # Track last utterance for cross-window context
        self._last_utterance: Optional[Utterance] = None
    
    def _debug(self, message: str) -> None:
        """Lightweight debug logger."""
        if self.config.debug:
            print(f"[DEBUG][AudioPipeline] {message}")
    
    # =========================================================================
    # COMPONENT ACCESS (Lazy initialization)
    # =========================================================================
    
    @property
    def vad(self) -> SileroVAD:
        if self._vad is None:
            self._vad = create_vad(self.config.vad_backend)
        return self._vad
    
    @property
    def diarizer(self) -> SpeakerDiarizer:
        if self._diarizer is None:
            self._diarizer = create_diarizer(self.config.diarization_backend)
        return self._diarizer
    
    @property
    def segmenter(self) -> UtteranceSegmenter:
        if self._segmenter is None:
            self._segmenter = create_segmenter(
                min_duration_sec=self.config.min_utterance_duration,
                max_gap_sec=self.config.max_utterance_gap,
            )
        return self._segmenter
    
    @property
    def asr(self) -> HebrewASR:
        if self._asr is None:
            self._asr = create_asr(model=self.config.asr_model)
        return self._asr
    
    @property
    def prosody(self) -> Optional[ProsodyExtractor]:
        if self._prosody is None and self.config.enable_prosody:
            self._prosody = create_prosody_extractor()
        return self._prosody
    
    @property
    def enricher(self) -> UtteranceEnricher:
        if self._enricher is None:
            self._enricher = create_enricher(
                asr=self.asr,
                prosody=self.prosody,
                enable_echolalia=self.config.enable_echolalia_detection,
                enable_perseveration=self.config.enable_perseveration_detection,
            )
        return self._enricher
    
    @property
    def converter(self) -> EventConverter:
        if self._converter is None:
            self._converter = create_converter(
                retain_transcripts=self.config.retain_transcripts,
            )
        return self._converter
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    def process_window(
        self,
        audio: np.ndarray,
        window_start: datetime,
        on_event: Optional[Callable[[EnrichedUtterance], None]] = None,
    ) -> PipelineResult:
        """
        Process a single audio window through the full pipeline.
        
        Args:
            audio: Audio samples (float32, mono, 16kHz)
            window_start: Absolute timestamp of window start
            on_event: Callback for each event as it's created
            
        Returns:
            PipelineResult with all extracted events
            
        Note:
            Raw audio is discarded after processing.
            Only structured events are returned.
        """
        import time
        start_time = time.time()
        
        # Validate input
        audio = self._validate_audio(audio)
        self._debug(
            f"Processing window starting {window_start.isoformat()} "
            f"with {len(audio)} samples (~{len(audio) / self.config.sample_rate:.2f}s)"
        )
        
        # Stage 1: Voice Activity Detection
        speech_segments = self.vad.process(audio, self.config.sample_rate)
        self._debug(
            f"VAD produced {len(speech_segments)} segments "
            f"durations={[round(s.duration_sec, 2) for s in speech_segments]}"
        )
        
        if not speech_segments:
            self._debug("No speech detected; returning empty result")
            return PipelineResult(
                window_start=window_start,
                window_end=window_start,
                processing_time_ms=int((time.time() - start_time) * 1000),
            )
        
        # Stage 2: Speaker Diarization
        diarized = self.diarizer.process(speech_segments)
        self._debug(
            f"Diarization produced {len(diarized)} segments across "
            f"{self.diarizer.get_speaker_count()} speakers"
        )
        
        # Stage 3: Utterance Segmentation
        utterances = self.segmenter.process(diarized)
        self._debug(
            f"Segmentation produced {len(utterances)} utterances "
            f"durations={[round(u.duration_sec, 2) for u in utterances]}"
        )
        
        # Stage 4-5: ASR, Prosody, Enrichment
        enriched = []
        previous = self._last_utterance
        
        for idx, utterance in enumerate(utterances, start=1):
            self._debug(
                f"Enriching utterance {idx}/{len(utterances)} "
                f"speaker={utterance.speaker_id} "
                f"duration={utterance.duration_sec:.2f}s segments={utterance.num_segments}"
            )
            result = self.enricher.process(utterance, previous)
            enriched.append(result)
            previous = utterance
        
        # Update last utterance for next window
        if utterances:
            self._last_utterance = utterances[-1]
        
        # Stage 6: Convert to schema events
        events = self.converter.convert_batch(enriched, window_start)
        self._debug(f"Converted {len(events)} events from enriched utterances")
        
        # Callback for streaming
        if on_event:
            for event in events:
                on_event(event)
        
        # CRITICAL: Discard raw audio
        del audio
        del speech_segments
        del diarized
        del utterances
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        self._debug(
            f"Processing complete in {processing_time}ms; "
            f"utterances={len(enriched)} speakers={self.diarizer.get_speaker_count()}"
        )
        
        return PipelineResult(
            speech_events=events,
            ambient_events=[],  # TODO: acoustic event detection
            window_start=window_start,
            window_end=window_start,  # TODO: calculate properly
            processing_time_ms=processing_time,
            num_utterances=len(enriched),
            speakers_detected=self.diarizer.get_speaker_count(),
        )
    
    def process_streaming(
        self,
        audio: np.ndarray,
        window_start: datetime,
    ) -> Generator[EnrichedUtterance, None, None]:
        """
        Process audio window with streaming event output.
        
        Yields events as they're ready, enabling real-time processing.
        
        Args:
            audio: Audio samples
            window_start: Absolute timestamp
            
        Yields:
            EnrichedUtterance objects as they're created
        """
        audio = self._validate_audio(audio)
        
        # VAD
        speech_segments = self.vad.process(audio, self.config.sample_rate)
        if not speech_segments:
            return
        
        # Diarization
        diarized = self.diarizer.process(speech_segments)
        
        # Segmentation
        utterances = self.segmenter.process(diarized)
        
        # Process and yield each utterance
        previous = self._last_utterance
        
        for utterance in utterances:
            # Enrich
            enriched = self.enricher.process(utterance, previous)
            
            # Convert
            event = self.converter.convert_utterance(enriched, window_start)
            
            # Yield immediately
            yield event
            
            previous = utterance
        
        # Update state
        if utterances:
            self._last_utterance = utterances[-1]
        
        # Cleanup
        del audio
    
    def _validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """Validate and normalize audio input."""
        # Ensure numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        return audio
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def reset_session(self):
        """
        Reset all stateful components for a new session.
        
        Call this when starting observation of a new classroom/group.
        """
        self.diarizer.reset()
        self.enricher.reset()
        self.converter.reset()
        self._turn_analyzer.reset()
        self._last_utterance = None
    
    def get_session_stats(self) -> dict:
        """Get statistics for current session."""
        return {
            "speakers_detected": self.diarizer.get_speaker_count(),
            "speaker_stats": self.diarizer.get_speaker_stats(),
            "turn_stats": self._turn_analyzer.get_speaker_stats(),
            "actors": [a.id for a in self.converter.get_actors()],
        }


class RollingBufferPipeline:
    """
    Pipeline with rolling buffer management.
    
    Handles the 30-second rolling window with 10-second overlap
    as specified in the architecture.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_events: Optional[Callable[[List[EnrichedUtterance]], None]] = None,
    ):
        self.config = config or PipelineConfig()
        self.pipeline = AudioPipeline(config)
        self.on_events = on_events
        
        # Rolling buffer
        window_samples = int(self.config.window_size_sec * self.config.sample_rate)
        self._buffer = np.zeros(window_samples, dtype=np.float32)
        self._buffer_pos = 0
        self._window_start: Optional[datetime] = None
        
        # Processing interval
        self._process_interval_samples = int(
            (self.config.window_size_sec - self.config.overlap_sec) 
            * self.config.sample_rate
        )
        self._samples_since_process = 0
    
    def add_samples(
        self,
        samples: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> Optional[PipelineResult]:
        """
        Add audio samples to the buffer.
        
        Automatically triggers processing when buffer is ready.
        
        Args:
            samples: New audio samples
            timestamp: Timestamp of first sample (optional)
            
        Returns:
            PipelineResult if processing was triggered, else None
        """
        if timestamp and self._window_start is None:
            self._window_start = timestamp
        
        # Add to buffer (circular)
        samples = samples.astype(np.float32)
        
        for sample in samples:
            self._buffer[self._buffer_pos] = sample
            self._buffer_pos = (self._buffer_pos + 1) % len(self._buffer)
            self._samples_since_process += 1
        
        # Check if we should process
        if self._samples_since_process >= self._process_interval_samples:
            return self._process_buffer()
        
        return None
    
    def _process_buffer(self) -> PipelineResult:
        """Process the current buffer."""
        # Get buffer in correct order
        if self._buffer_pos == 0:
            audio = self._buffer.copy()
        else:
            audio = np.concatenate([
                self._buffer[self._buffer_pos:],
                self._buffer[:self._buffer_pos],
            ])
        
        # Process
        window_start = self._window_start or datetime.now()
        result = self.pipeline.process_window(audio, window_start)
        
        # Callback
        if self.on_events and result.speech_events:
            self.on_events(result.speech_events)
        
        # Update state
        self._samples_since_process = 0
        
        # Advance window start by process interval
        if self._window_start:
            interval_sec = self.config.window_size_sec - self.config.overlap_sec
            from datetime import timedelta
            self._window_start += timedelta(seconds=interval_sec)
        
        return result
    
    def flush(self) -> Optional[PipelineResult]:
        """Process any remaining buffered audio."""
        if self._samples_since_process > self.config.sample_rate:  # At least 1 second
            return self._process_buffer()
        return None
    
    def reset(self):
        """Reset buffer and pipeline state."""
        self._buffer.fill(0)
        self._buffer_pos = 0
        self._window_start = None
        self._samples_since_process = 0
        self.pipeline.reset_session()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline(
    config: Optional[PipelineConfig] = None,
    **kwargs,
) -> AudioPipeline:
    """Create an audio pipeline with optional configuration."""
    if config is None:
        config = PipelineConfig(**kwargs)
    return AudioPipeline(config)


def process_audio_file(
    filepath: str,
    config: Optional[PipelineConfig] = None,
) -> List[EnrichedUtterance]:
    """
    Process an audio file through the pipeline.
    
    Convenience function for testing/development.
    """
    import soundfile as sf
    
    # Load audio
    audio, sample_rate = sf.read(filepath, dtype='float32')
    
    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        print(f"Resampling from {sample_rate} to {SAMPLE_RATE}")

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Process
    pipeline = create_pipeline(config)
    result = pipeline.process_window(audio, datetime.now())
    
    return result.speech_events

if __name__ == "__main__":
    # Simple test
    print(f"Testing audio pipeline with sample file... file: ../data/audio/slice1_audio.wav")
    with open("../data/audio/slice1_audio.wav", "rb") as f:
        test_events = process_audio_file("../data/audio/slice1_audio.wav", config=PipelineConfig(debug=True))
        for event in test_events:
            print(event)
    print(f"Processed {len(test_events)} events from test audio file.")