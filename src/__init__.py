"""
IRIS - Intelligent Room Insight System

Privacy-first behavioral observation for autism support classrooms.

Architecture:
    Layer 1: Raw sensor events from ASR/CV models
    Layer 2: LLM-inferred compound events with semantic meaning

Modules:
    - schema: Event data types
    """


# Schema
from .schema import (
    # Common
    ActorRole, ClassroomZone, Intensity,
    Actor, Location, EventBase,
    
    # Layer 1 - Audio
    SpeechTarget, VocalType, VerbalComplexity,
    ProsodyFeatures, SpeechEvent, AmbientAudioEvent,
    
    # Layer 1 - Video  
    ProximityChange, ProximityLevel, ProximityEvent,
    GazeDirection, GazeEvent,
    PostureType, MovementType, BodyOrientation, PostureEvent,
    ObjectAction, ObjectEvent,
    
    # Layer 2 - LLM Inferred
    ContentType, CommunicativeIntent, BehaviorCategory,
    EmotionalState, TriggerType,
    BehavioralEvent, InteractionEvent, ContextEvent,
    InteractionType, InteractionQuality,
    ActivityType, ClassroomClimate,
    
    # Session
    SessionMetadata, Session,
    
    # Type aliases
    Layer1Event, Layer2Event, Event,
)

__version__ = "0.0.1"
__all__ = [
    # Common
    "ActorRole", "ClassroomZone", "Intensity",
    "Actor", "Location", "EventBase",
    
    # Layer 1 - Audio
    "SpeechTarget", "VocalType", "VerbalComplexity", 
    "ProsodyFeatures", "SpeechEvent", "AmbientAudioEvent",
    
    # Layer 1 - Video
    "ProximityChange", "ProximityLevel", "ProximityEvent",
    "GazeDirection", "GazeEvent",
    "PostureType", "MovementType", "BodyOrientation", "PostureEvent",
    "ObjectAction", "ObjectEvent",
    
    # Layer 2 - LLM Inferred
    "ContentType", "CommunicativeIntent", "BehaviorCategory",
    "EmotionalState", "TriggerType",
    "BehavioralEvent", "InteractionEvent", "ContextEvent",
    "InteractionType", "InteractionQuality",
    "ActivityType", "ClassroomClimate",
    
    # Type aliases
    "Layer1Event", "Layer2Event", "Event",
]
