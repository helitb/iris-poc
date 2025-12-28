# IRIS POC PIPELINE


┌──────────────────────────────────────────────────────────────┐
│                    EDGE DEVICE ONLY                           │
│                  (Nothing leaves device)                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  L1: Raw Sensor Data (Audio + Video Buffers)                 │
│  ↓                                                            │
│  Process locally → Generate structured events                │
│  ↓                                                            │
│  L2: Structured Event Logs (Anonymized)                      │
│  → speech_events, gaze_events, proximity_events, etc.        │
│  → Store locally, NO PII                                     │
│                                                               │
│  ✅ Discard all raw audio/video immediately                  │
│                                                               │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ ONLY sanitized L2 events can exit
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                    CLOUD (Optional)                           │
│                                                               │
│  L3: Narrative Reconstruction                                │
│  → Input: Anonymized event logs (no audio, no video)        │
│  → Output: Parent stories, clinical reports                  │
│                                                               │
└──────────────────────────────────────────────────────────────┘



# Audio pipeline

Audio Buffer (30s) 
    ↓
┌─────────────────────────────────────────────────────────────┐
│ types.py      Internal pipeline types (SpeechSegment,       │
│               DiarizedSegment, Utterance, EnrichedUtterance)│
├─────────────────────────────────────────────────────────────┤
│ vad.py        Silero VAD (primary) + WebRTC (lightweight)   │
│               → List[SpeechSegment]                         │
├─────────────────────────────────────────────────────────────┤
│ diarization   Resemblyzer/SpeechBrain/Simple backends       │
│               → List[DiarizedSegment] with speaker IDs      │
├─────────────────────────────────────────────────────────────┤
│ segmentation  Group segments → Utterances                    │
│               + TurnTakingAnalyzer                          │
├─────────────────────────────────────────────────────────────┤
│ asr.py        HebrewASR (faster-whisper + ivrit-ai)         │
│               → ASRResult with transcript + word timing     │
├─────────────────────────────────────────────────────────────┤
│ prosody.py    Parselmouth/Praat extraction                  │
│               → ProsodyResult (pitch, intensity, jitter...) │
├─────────────────────────────────────────────────────────────┤
│ enrichment    EcholaliaDetector + PerseverationDetector     │
│               + VocalTypeClassifier → EnrichedUtterance     │
├─────────────────────────────────────────────────────────────┤
│ conversion    EnrichedUtterance → schema.SpeechEvent        │
│               ⚠️  PRIVACY BOUNDARY: transcripts discarded   │
├─────────────────────────────────────────────────────────────┤
│ pipeline.py   AudioPipeline orchestrator                    │
│               + RollingBufferPipeline for streaming         │
└─────────────────────────────────────────────────────────────┘
    ↓
List[SpeechEvent]  (schema-compliant, privacy-safe)

# Embedded Configuration

PRODUCTION_CONFIG = {
    "hardware": "jetson_orin_nano_8gb",
    
    "models": {
        "multimodal": "llama_3.2_vision_11b_int4",  
        "asr": "whisper_small_int8",  # 500MB
        "prosody": "parselmouth",  # CPU
        "cv": "opencv_mediapipe"  # CPU
    },
    
    "task_allocation": {
        "llama_handles": [
            "video_frame_analysis",
            "pattern_detection", 
            "interaction_enrichment",
            "optional_local_narrative"  # If user doesn't want cloud
        ],
        "whisper_handles": ["asr"],
        "cpu_handles": ["vad", "prosody", "cv_detection"]
    },
    
    "memory_usage": {
        "llama": "6-7GB",
        "whisper": "500MB",
        "other": "500-800MB",
        "total": "~7.8GB / 8GB"  
    }
}

