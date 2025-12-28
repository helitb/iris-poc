"""
IRIS Audio Processing Pipeline

Privacy-first audio processing for autism support classrooms.

Architecture:
    Audio Buffer → VAD → Diarization → Segmentation → ASR → Prosody → Enrichment → Events
    
    Raw audio is discarded immediately after processing.
    Only structured, anonymized events are returned.

Quick Start:
    from iris.audio import AudioPipeline, PipelineConfig
    
    # Create pipeline
    pipeline = AudioPipeline()
    
    # Process 30-second audio window
    result = pipeline.process_window(audio_buffer, timestamp)
    
    # Get events (privacy-safe output)
    for event in result.speech_events:
        print(f"{event.speaker.id}: {event.word_count} words")

Streaming:
    for event in pipeline.process_streaming(audio, timestamp):
        yield event  # Events emitted in real-time

Components can also be used individually:
    from iris.audio import create_vad, create_asr, create_diarizer
"""

# Pipeline
from .pipeline import (
    AudioPipeline,
    RollingBufferPipeline,
    PipelineConfig,
    PipelineResult,
    create_pipeline,
    process_audio_file,
)

# Types
from .types import (
    # Configuration
    SAMPLE_RATE,
    WINDOW_SIZE_SEC,
    OVERLAP_SEC,
    MIN_UTTERANCE_DURATION_SEC,
    MAX_UTTERANCE_GAP_SEC,
    
    # Pipeline types
    SpeechSegment,
    DiarizedSegment,
    Utterance,
    EnrichedUtterance,
    ProsodyResult,
    ASRResult,
    AcousticEvent,
)

# Components
from .vad import SileroVAD, create_vad
from .diarization import SpeakerDiarizer, SimpleDiarizer, create_diarizer
from .segmentation import UtteranceSegmenter, TurnTakingAnalyzer, create_segmenter
from .asr import HebrewASR, WhisperASR, create_asr
from .prosody import ProsodyExtractor, create_prosody_extractor
from .enrichment import (
    UtteranceEnricher,
    EcholaliaDetector,
    PerseverationDetector,
    SentenceEmbedder,
    create_enricher,
)
from .conversion import EventConverter, create_converter, utterances_to_events

__all__ = [
    # Pipeline
    "AudioPipeline",
    "RollingBufferPipeline", 
    "PipelineConfig",
    "PipelineResult",
    "create_pipeline",
    "process_audio_file",
    
    # Types
    "SAMPLE_RATE",
    "WINDOW_SIZE_SEC",
    "OVERLAP_SEC",
    "MIN_UTTERANCE_DURATION_SEC",
    "MAX_UTTERANCE_GAP_SEC",
    "SpeechSegment",
    "DiarizedSegment",
    "Utterance",
    "EnrichedUtterance",
    "ProsodyResult",
    "ASRResult",
    "AcousticEvent",
    
    # VAD
    "SileroVAD",
    "create_vad",
    
    # Diarization
    "SpeakerDiarizer",
    "SimpleDiarizer",
    "create_diarizer",
    
    # Segmentation
    "UtteranceSegmenter",
    "TurnTakingAnalyzer",
    "create_segmenter",
    
    # ASR
    "HebrewASR",
    "WhisperASR",
    "create_asr",
    
    # Prosody
    "ProsodyExtractor",
    "create_prosody_extractor",
    
    # Enrichment
    "UtteranceEnricher",
    "EcholaliaDetector",
    "PerseverationDetector",
    "SentenceEmbedder",
    "create_enricher",
    
    # Conversion
    "EventConverter",
    "create_converter",
    "utterances_to_events",
]
