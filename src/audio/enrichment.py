"""
Utterance enrichment for ASD-relevant pattern detection.

Detects:
- Echolalia (immediate and delayed)
- Perseveration (topic/phrase repetition)
- Vocal type classification
- Turn-taking anomalies
"""

from typing import List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass
import numpy as np

from .types import (
    Utterance, EnrichedUtterance, ASRResult, ProsodyResult,
    SAMPLE_RATE,
)
from .asr import HebrewASR, compute_verbal_complexity
from .prosody import ProsodyExtractor


@dataclass
class UtteranceRecord:
    """Record of a past utterance for pattern matching."""
    speaker_id: str
    text: str
    embedding: Optional[np.ndarray]
    timestamp_sec: float
    
    
class SentenceEmbedder:
    """
    Generate sentence embeddings for similarity comparison.
    Uses multilingual model for Hebrew support.
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            model_name: Sentence-transformers model name
        """
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required. Install with: "
                "pip install sentence-transformers"
            ) from e
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        self._load_model()
        return self._model.encode(text, convert_to_numpy=True)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


class EcholaliaDetector:
    """
    Detect echolalic speech patterns.
    
    Types:
    - Immediate echolalia: Direct repetition of recent speech
    - Delayed echolalia: Repetition of previously heard phrases
    
    Uses both exact matching and semantic similarity.
    """
    
    def __init__(
        self,
        immediate_threshold: float = 0.85,  # High similarity for immediate
        delayed_threshold: float = 0.75,  # Lower threshold for delayed
        immediate_window_sec: float = 10.0,  # Look back window for immediate
        history_size: int = 100,  # Number of utterances to track
    ):
        self.immediate_threshold = immediate_threshold
        self.delayed_threshold = delayed_threshold
        self.immediate_window_sec = immediate_window_sec
        self.history_size = history_size
        
        self._history: Deque[UtteranceRecord] = deque(maxlen=history_size)
        self._embedder: Optional[SentenceEmbedder] = None
    
    def _get_embedder(self) -> SentenceEmbedder:
        if self._embedder is None:
            self._embedder = SentenceEmbedder()
        return self._embedder
    
    def check_echolalia(
        self,
        text: str,
        speaker_id: str,
        current_time_sec: float,
        embedding: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[int]]:
        """
        Check if utterance is echolalic.
        
        Args:
            text: Transcribed text
            speaker_id: Speaker ID
            current_time_sec: Current timestamp
            embedding: Pre-computed embedding (optional)
            
        Returns:
            (is_echolalia, echolalia_type, similarity, source_offset_ms)
        """
        if not text or len(text.strip()) < 2:
            return False, None, None, None
        
        # Get embedding if not provided
        if embedding is None:
            try:
                embedding = self._get_embedder().embed(text)
            except Exception:
                embedding = None
        
        best_match = None
        best_similarity = 0.0
        best_offset = None
        
        for record in self._history:
            # Skip own utterances for immediate echolalia detection
            # (we're looking for echoing others)
            
            # Calculate time offset
            offset_sec = current_time_sec - record.timestamp_sec
            offset_ms = int(offset_sec * 1000)
            
            # Compute similarity
            if embedding is not None and record.embedding is not None:
                similarity = self._get_embedder().embedding_similarity(
                    embedding, record.embedding
                )
            else:
                # Fallback to exact/substring matching
                similarity = self._text_similarity(text, record.text)
            
            # Check if this is a better match
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = record
                best_offset = offset_ms
        
        # Determine echolalia type
        if best_match is None:
            return False, None, None, None
        
        offset_sec = best_offset / 1000.0 if best_offset else 0
        
        # Immediate echolalia: high similarity, recent, typically from different speaker
        if (best_similarity >= self.immediate_threshold and 
            offset_sec <= self.immediate_window_sec and
            best_match.speaker_id != speaker_id):
            return True, "immediate", best_similarity, best_offset
        
        # Delayed echolalia: moderate similarity, any time
        if best_similarity >= self.delayed_threshold:
            return True, "delayed", best_similarity, best_offset
        
        return False, None, best_similarity, best_offset
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity fallback."""
        # Normalize
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # Exact match
        if t1 == t2:
            return 1.0
        
        # Substring match
        if t1 in t2 or t2 in t1:
            shorter = min(len(t1), len(t2))
            longer = max(len(t1), len(t2))
            return shorter / longer if longer > 0 else 0.0
        
        # Word overlap
        words1 = set(t1.split())
        words2 = set(t2.split())
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def add_to_history(
        self,
        text: str,
        speaker_id: str,
        timestamp_sec: float,
        embedding: Optional[np.ndarray] = None,
    ):
        """Add utterance to history for future comparisons."""
        if not text or len(text.strip()) < 2:
            return
        
        if embedding is None:
            try:
                embedding = self._get_embedder().embed(text)
            except Exception:
                embedding = None
        
        self._history.append(UtteranceRecord(
            speaker_id=speaker_id,
            text=text,
            embedding=embedding,
            timestamp_sec=timestamp_sec,
        ))
    
    def reset(self):
        """Clear history."""
        self._history.clear()


class PerseverationDetector:
    """
    Detect perseverative speech patterns.
    
    Perseveration: Repetitive focus on same topic/phrase by same speaker,
    beyond what's contextually appropriate.
    """
    
    def __init__(
        self,
        repetition_threshold: int = 3,  # Min repetitions to flag
        similarity_threshold: float = 0.7,
        window_size: int = 20,  # Utterances to consider
    ):
        self.repetition_threshold = repetition_threshold
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        
        # Track per-speaker history
        self._speaker_history: dict[str, Deque[str]] = {}
        self._embedder: Optional[SentenceEmbedder] = None
    
    def _get_embedder(self) -> SentenceEmbedder:
        if self._embedder is None:
            self._embedder = SentenceEmbedder()
        return self._embedder
    
    def check_perseveration(
        self,
        text: str,
        speaker_id: str,
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if utterance is perseverative.
        
        Returns:
            (is_perseveration, repetition_count)
        """
        if not text or len(text.strip()) < 2:
            return False, None
        
        # Get speaker history
        history = self._speaker_history.get(speaker_id, deque(maxlen=self.window_size))
        
        if not history:
            return False, None
        
        # Count similar utterances
        similar_count = 0
        text_lower = text.lower().strip()
        
        for past_text in history:
            similarity = self._compute_similarity(text_lower, past_text.lower().strip())
            if similarity >= self.similarity_threshold:
                similar_count += 1
        
        is_perseveration = similar_count >= self.repetition_threshold
        return is_perseveration, similar_count if is_perseveration else None
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between texts."""
        # Exact match
        if text1 == text2:
            return 1.0
        
        # Word overlap (Jaccard)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def add_to_history(self, text: str, speaker_id: str):
        """Add utterance to speaker's history."""
        if not text:
            return
        
        if speaker_id not in self._speaker_history:
            self._speaker_history[speaker_id] = deque(maxlen=self.window_size)
        
        self._speaker_history[speaker_id].append(text)
    
    def reset(self):
        """Clear all history."""
        self._speaker_history.clear()


class VocalTypeClassifier:
    """
    Classify vocalization type from audio features.
    
    Categories:
    - speech: Intelligible words
    - vocalization: Non-word vocal sounds
    - cry: Crying
    - laugh: Laughter
    - scream: Screaming/yelling
    - hum: Humming
    - whisper: Whispering
    """
    
    def __init__(self):
        self._classifier = None
    
    def classify(
        self,
        audio: np.ndarray,
        transcript: Optional[str],
        prosody: Optional[ProsodyResult],
    ) -> str:
        """
        Classify the type of vocalization.
        
        Uses combination of:
        - ASR confidence (low = non-speech)
        - Prosodic features
        - Audio energy patterns
        """
        # If we have valid transcript, it's speech
        if transcript and len(transcript.strip()) > 0:
            # Check for whisper based on intensity
            if prosody and prosody.intensity_mean_db and prosody.intensity_mean_db < 50:
                return "whisper"
            return "speech"
        
        # Analyze audio for non-speech classification
        return self._classify_from_audio(audio, prosody)
    
    def _classify_from_audio(
        self,
        audio: np.ndarray,
        prosody: Optional[ProsodyResult],
    ) -> str:
        """Classify non-speech vocalizations from audio."""
        # Simple heuristic classification
        # In production, use a trained classifier
        
        rms = np.sqrt(np.mean(audio ** 2))
        zcr = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
        
        # High energy + high pitch variation = scream/cry
        if prosody:
            if prosody.pitch_std_hz and prosody.pitch_std_hz > 100:
                if prosody.intensity_mean_db and prosody.intensity_mean_db > 70:
                    return "scream"
                return "cry"
            
            # Low pitch variation + regular pattern = hum
            if prosody.pitch_std_hz and prosody.pitch_std_hz < 20:
                return "hum"
        
        # High ZCR can indicate laughter
        if zcr > 0.1:
            return "laugh"
        
        return "vocalization"


class UtteranceEnricher:
    """
    Main enrichment pipeline that combines all analyzers.
    
    Usage:
        enricher = UtteranceEnricher()
        enriched = enricher.process(utterance, buffer_start_time)
    """
    
    def __init__(
        self,
        asr: Optional[HebrewASR] = None,
        prosody_extractor: Optional[ProsodyExtractor] = None,
        enable_echolalia: bool = True,
        enable_perseveration: bool = True,
    ):
        self.asr = asr
        self.prosody_extractor = prosody_extractor
        
        self.echolalia_detector = EcholaliaDetector() if enable_echolalia else None
        self.perseveration_detector = PerseverationDetector() if enable_perseveration else None
        self.vocal_classifier = VocalTypeClassifier()
        
        self._embedder: Optional[SentenceEmbedder] = None
    
    def _get_embedder(self) -> SentenceEmbedder:
        if self._embedder is None:
            self._embedder = SentenceEmbedder()
        return self._embedder
    
    def process(
        self,
        utterance: Utterance,
        previous_utterance: Optional[Utterance] = None,
    ) -> EnrichedUtterance:
        """
        Process a single utterance through all enrichment stages.
        
        Args:
            utterance: Input utterance with audio
            previous_utterance: Previous utterance for turn-taking analysis
            
        Returns:
            EnrichedUtterance with all extracted features
        """
        # ASR
        transcript = None
        word_count = 0
        asr_confidence = None
        
        if self.asr:
            try:
                asr_result = self.asr.transcribe(utterance.audio)
                transcript = asr_result.text
                word_count = asr_result.word_count
                asr_confidence = asr_result.confidence
            except Exception as e:
                print(f"Warning: ASR failed: {e}")
        
        # Prosody
        prosody = None
        if self.prosody_extractor:
            try:
                prosody = self.prosody_extractor.extract(utterance.audio)
            except Exception as e:
                print(f"Warning: Prosody extraction failed: {e}")
        
        # Vocal type
        vocal_type = self.vocal_classifier.classify(utterance.audio, transcript, prosody)
        
        # Embedding for pattern detection
        embedding = None
        if transcript:
            try:
                embedding = self._get_embedder().embed(transcript)
            except Exception:
                pass
        
        # Echolalia detection
        is_echolalia = False
        echolalia_type = None
        echolalia_similarity = None
        echolalia_offset = None
        
        if self.echolalia_detector and transcript:
            is_echo, echo_type, echo_sim, echo_offset = self.echolalia_detector.check_echolalia(
                transcript,
                utterance.speaker_id,
                utterance.start_sec,
                embedding,
            )
            is_echolalia = is_echo
            echolalia_type = echo_type
            echolalia_similarity = echo_sim
            echolalia_offset = echo_offset
            
            # Add to history for future detection
            self.echolalia_detector.add_to_history(
                transcript,
                utterance.speaker_id,
                utterance.start_sec,
                embedding,
            )
        
        # Perseveration detection
        is_perseveration = False
        perseveration_count = None
        
        if self.perseveration_detector and transcript:
            is_persev, persev_count = self.perseveration_detector.check_perseveration(
                transcript,
                utterance.speaker_id,
            )
            is_perseveration = is_persev
            perseveration_count = persev_count
            
            # Add to history
            self.perseveration_detector.add_to_history(transcript, utterance.speaker_id)
        
        # Turn-taking analysis
        gap_before_ms = None
        is_overlap = False
        previous_speaker = None
        
        if previous_utterance:
            gap_sec = utterance.start_sec - previous_utterance.end_sec
            gap_before_ms = int(gap_sec * 1000)
            is_overlap = gap_sec < 0
            previous_speaker = previous_utterance.speaker_id
        
        return EnrichedUtterance(
            speaker_id=utterance.speaker_id,
            start_sec=utterance.start_sec,
            end_sec=utterance.end_sec,
            transcript=transcript,
            word_count=word_count,
            asr_confidence=asr_confidence,
            prosody=prosody,
            vocal_type=vocal_type,
            is_echolalia=is_echolalia,
            echolalia_type=echolalia_type,
            echolalia_similarity=echolalia_similarity,
            echolalia_source_offset_ms=echolalia_offset,
            is_perseveration=is_perseveration,
            perseveration_count=perseveration_count,
            gap_before_ms=gap_before_ms,
            is_overlap=is_overlap,
            previous_speaker_id=previous_speaker,
            _embedding=embedding,
        )
    
    def process_batch(
        self,
        utterances: List[Utterance],
    ) -> List[EnrichedUtterance]:
        """Process multiple utterances, maintaining context."""
        enriched = []
        previous = None
        
        for utterance in utterances:
            result = self.process(utterance, previous)
            enriched.append(result)
            previous = utterance
        
        return enriched
    
    def reset(self):
        """Reset all detectors (e.g., for new session)."""
        if self.echolalia_detector:
            self.echolalia_detector.reset()
        if self.perseveration_detector:
            self.perseveration_detector.reset()


# =============================================================================
# FACTORY
# =============================================================================

def create_enricher(
    asr: Optional[HebrewASR] = None,
    prosody: Optional[ProsodyExtractor] = None,
    **kwargs,
) -> UtteranceEnricher:
    """Create an utterance enricher."""
    return UtteranceEnricher(asr=asr, prosody_extractor=prosody, **kwargs)
