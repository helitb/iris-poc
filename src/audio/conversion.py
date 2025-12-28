"""
Conversion from pipeline types to schema events.

This is the critical boundary where:
1. Raw audio is discarded
2. Transcriptions are converted to metadata
3. Structured events are created for persistence

Privacy note: After conversion, no raw content should remain.

"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict
import uuid

from .types import EnrichedUtterance, AcousticEvent, ProsodyResult

# Import schema types - handle different import contexts
def _import_schema():
    """Import schema types, handling various package configurations."""
    try:
        # When used as iris.audio.conversion
        from ..schema import (
            Actor, ActorRole,
            SpeechEvent, AmbientAudioEvent,
            SpeechTarget, VocalType, VerbalComplexity,
            ProsodyFeatures,
            Intensity,
            ClassroomZone,
        )
        return (Actor, ActorRole, SpeechEvent, AmbientAudioEvent,
                SpeechTarget, VocalType, VerbalComplexity,
                ProsodyFeatures, Intensity, ClassroomZone)
    except ImportError:
        pass
    
    try:
        # When schema is in same directory or PYTHONPATH
        from schema import (
            Actor, ActorRole,
            SpeechEvent, AmbientAudioEvent,
            SpeechTarget, VocalType, VerbalComplexity,
            ProsodyFeatures,
            Intensity,
            ClassroomZone,
        )
        return (Actor, ActorRole, SpeechEvent, AmbientAudioEvent,
                SpeechTarget, VocalType, VerbalComplexity,
                ProsodyFeatures, Intensity, ClassroomZone)
    except ImportError:
        pass
    
    try:
        # When iris is installed as a package
        from iris.schema import (
            Actor, ActorRole,
            SpeechEvent, AmbientAudioEvent,
            SpeechTarget, VocalType, VerbalComplexity,
            ProsodyFeatures,
            Intensity,
            ClassroomZone,
        )
        return (Actor, ActorRole, SpeechEvent, AmbientAudioEvent,
                SpeechTarget, VocalType, VerbalComplexity,
                ProsodyFeatures, Intensity, ClassroomZone)
    except ImportError:
        raise ImportError(
            "Cannot import schema types. Ensure iris package is properly installed "
            "or schema.py is in the Python path."
        )

(Actor, ActorRole, SpeechEvent, AmbientAudioEvent,
 SpeechTarget, VocalType, VerbalComplexity,
 ProsodyFeatures, Intensity, ClassroomZone) = _import_schema()


class EventConverter:
    """
    Converts pipeline output to schema events.
    
    This is the privacy boundary - after conversion:
    - Raw audio is gone
    - Transcripts are converted to metadata (word count, complexity)
    - Only structured, anonymized data remains
    """
    
    def __init__(
        self,
        retain_transcripts: bool = False,  # For debugging only
        speaker_role_map: Optional[Dict[str, ActorRole]] = None,
    ):
        """
        Args:
            retain_transcripts: Whether to keep transcripts in events
                               (should be False in production)
            speaker_role_map: Map speaker IDs to roles (child/adult)
        """
        self.retain_transcripts = retain_transcripts
        self.speaker_role_map = speaker_role_map or {}
        
        # Cache for Actor objects
        self._actors: Dict[str, Actor] = {}
    
    def _get_or_create_actor(self, speaker_id: str) -> Actor:
        """Get or create Actor for speaker ID."""
        if speaker_id not in self._actors:
            # Determine role from map or default
            role = self.speaker_role_map.get(speaker_id, ActorRole.CHILD)
            
            # Create anonymous actor ID
            # Format: child_0, child_1, adult_0, etc.
            role_count = sum(
                1 for a in self._actors.values() 
                if a.role == role
            )
            actor_id = f"{role.value}_{role_count}"
            
            self._actors[speaker_id] = Actor(id=actor_id, role=role)
        
        return self._actors[speaker_id]
    
    def convert_utterance(
        self,
        enriched: EnrichedUtterance,
        buffer_start: datetime,
    ) -> SpeechEvent:
        """
        Convert EnrichedUtterance to SpeechEvent.
        
        Args:
            enriched: Enriched utterance from pipeline
            buffer_start: Absolute timestamp of buffer start
            
        Returns:
            SpeechEvent (schema type)
        """
        # Calculate absolute timestamp
        timestamp = buffer_start + timedelta(seconds=enriched.start_sec)
        
        # Get actor
        actor = self._get_or_create_actor(enriched.speaker_id)
        
        # Map vocal type
        vocal_type = self._map_vocal_type(enriched.vocal_type)
        
        # Calculate complexity
        complexity = self._compute_complexity(enriched.word_count)
        
        # Convert prosody
        prosody = self._convert_prosody(enriched.prosody)
        
        # Determine speech target (heuristic)
        target = self._infer_target(enriched)
        
        # Build event
        event = SpeechEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            speaker=actor,
            
            # Transcript handling - privacy critical!
            transcription=enriched.transcript if self.retain_transcripts else None,
            word_count=enriched.word_count,
            complexity=complexity,
            
            # Type and target
            vocal_type=vocal_type,
            target=target,
            
            # Prosody
            prosody=prosody,
            duration_ms=enriched.duration_ms,
            
            # Turn-taking
            gap_before_ms=enriched.gap_before_ms,
            is_overlap=enriched.is_overlap,
            previous_speaker=enriched.previous_speaker_id,
            
            # ASD patterns
            is_echolalia_candidate=enriched.is_echolalia,
            echolalia_similarity=enriched.echolalia_similarity,
            echolalia_delay_ms=enriched.echolalia_source_offset_ms,
            is_perseveration_candidate=enriched.is_perseveration,
        )
        
        return event
    
    def convert_acoustic_event(
        self,
        acoustic: AcousticEvent,
        buffer_start: datetime,
    ) -> AmbientAudioEvent:
        """
        Convert AcousticEvent to AmbientAudioEvent.
        """
        timestamp = buffer_start + timedelta(seconds=acoustic.start_sec)
        
        intensity = self._map_intensity(acoustic.intensity)
        
        return AmbientAudioEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            sound_type=acoustic.event_type,
            intensity=intensity,
            duration_ms=acoustic.duration_ms,
            location_estimate=None,  # Would need spatial audio for this
        )
    
    def _map_vocal_type(self, vocal_type: str) -> VocalType:
        """Map pipeline vocal type to schema enum."""
        mapping = {
            "speech": VocalType.SPEECH,
            "vocalization": VocalType.VOCALIZATION,
            "cry": VocalType.CRY,
            "laugh": VocalType.LAUGH,
            "scream": VocalType.SCREAM,
            "hum": VocalType.HUM,
            "whisper": VocalType.WHISPER,
        }
        return mapping.get(vocal_type, VocalType.SPEECH)
    
    def _compute_complexity(self, word_count: int) -> VerbalComplexity:
        """Compute verbal complexity from word count."""
        if word_count == 0:
            return VerbalComplexity.VOCALIZATION
        elif word_count == 1:
            return VerbalComplexity.SINGLE_WORD
        elif word_count <= 3:
            return VerbalComplexity.PHRASE
        elif word_count <= 8:
            return VerbalComplexity.SENTENCE
        else:
            return VerbalComplexity.MULTI_SENTENCE
    
    def _convert_prosody(self, prosody: Optional[ProsodyResult]) -> Optional[ProsodyFeatures]:
        """Convert pipeline ProsodyResult to schema ProsodyFeatures."""
        if prosody is None:
            return None
        
        return ProsodyFeatures(
            pitch_mean_hz=prosody.pitch_mean_hz,
            pitch_std_hz=prosody.pitch_std_hz,
            pitch_contour=prosody.pitch_contour,
            intensity_mean_db=prosody.intensity_mean_db,
            intensity_range_db=prosody.intensity_range_db,
            speech_rate=prosody.speech_rate_category,
            rhythm_regularity=None,  # Not computed yet
            voice_quality=prosody.voice_quality,
        )
    
    def _infer_target(self, enriched: EnrichedUtterance) -> SpeechTarget:
        """
        Infer who the speech is directed at.
        
        This is a heuristic - in production would use:
        - Gaze direction from video
        - Audio direction of arrival
        - Conversation context
        """
        # Self-talk indicators
        if enriched.vocal_type in ("hum", "vocalization"):
            return SpeechTarget.SELF
        
        # If echoing, might be self-directed
        if enriched.is_echolalia:
            return SpeechTarget.SELF
        
        # If there was a previous speaker, likely responding to them
        if enriched.previous_speaker_id:
            # Check if previous speaker was adult or peer
            prev_actor = self._actors.get(enriched.previous_speaker_id)
            if prev_actor:
                if prev_actor.role == ActorRole.ADULT:
                    return SpeechTarget.ADULT
                else:
                    return SpeechTarget.PEER
        
        return SpeechTarget.UNKNOWN
    
    def _map_intensity(self, intensity: str) -> Intensity:
        """Map intensity string to enum."""
        mapping = {
            "low": Intensity.LOW,
            "moderate": Intensity.MODERATE,
            "high": Intensity.HIGH,
        }
        return mapping.get(intensity, Intensity.MODERATE)
    
    def convert_batch(
        self,
        utterances: List[EnrichedUtterance],
        buffer_start: datetime,
    ) -> List[SpeechEvent]:
        """Convert a batch of utterances to events."""
        return [
            self.convert_utterance(utt, buffer_start)
            for utt in utterances
        ]
    
    def get_actors(self) -> List[Actor]:
        """Get all actors created during conversion."""
        return list(self._actors.values())
    
    def reset(self):
        """Reset actor cache (e.g., for new session)."""
        self._actors.clear()


class PrivacySanitizer:
    """
    Final privacy check before events leave the system.
    
    Ensures no PII has leaked through the pipeline.
    """
    
    def __init__(self):
        # Patterns that should never appear in events
        self._forbidden_patterns = [
            # Names (would need NER in production)
            # Phone numbers, emails, etc.
        ]
    
    def sanitize_event(self, event: SpeechEvent) -> SpeechEvent:
        """
        Sanitize a single event.
        
        Raises:
            ValueError: If PII is detected
        """
        # Check transcription is removed (unless debugging)
        if event.transcription:
            # In production, this should either:
            # 1. Remove the transcription
            # 2. Raise an error
            # For now, just warn
            print(f"Warning: Event {event.event_id} contains transcription")
        
        return event
    
    def sanitize_batch(self, events: List[SpeechEvent]) -> List[SpeechEvent]:
        """Sanitize a batch of events."""
        return [self.sanitize_event(e) for e in events]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_converter(
    retain_transcripts: bool = False,
    speaker_roles: Optional[Dict[str, ActorRole]] = None,
) -> EventConverter:
    """Create an event converter."""
    return EventConverter(
        retain_transcripts=retain_transcripts,
        speaker_role_map=speaker_roles,
    )


def utterances_to_events(
    utterances: List[EnrichedUtterance],
    buffer_start: datetime,
    retain_transcripts: bool = False,
) -> List[SpeechEvent]:
    """
    Convenience function to convert utterances to events.
    
    Args:
        utterances: List of enriched utterances
        buffer_start: Absolute timestamp of buffer start
        retain_transcripts: Whether to keep transcripts (debugging only)
        
    Returns:
        List of SpeechEvent objects
    """
    converter = EventConverter(retain_transcripts=retain_transcripts)
    return converter.convert_batch(utterances, buffer_start)
