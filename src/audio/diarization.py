"""
Speaker Diarization using embedding-based clustering.

Assigns speaker IDs to speech segments based on voice similarity.
Uses lightweight embedding models suitable for edge deployment.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .types import SpeechSegment, DiarizedSegment, SAMPLE_RATE


@dataclass
class SpeakerProfile:
    """Accumulated speaker profile from multiple utterances."""
    speaker_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    utterance_count: int = 0
    
    def add_embedding(self, embedding: np.ndarray):
        """Add a new embedding and update centroid."""
        self.embeddings.append(embedding)
        self.utterance_count += 1
        self._update_centroid()
    
    def _update_centroid(self):
        """Recalculate centroid from all embeddings."""
        if self.embeddings:
            self.centroid = np.mean(self.embeddings, axis=0)
    
    def similarity(self, embedding: np.ndarray) -> float:
        """Cosine similarity to this speaker's centroid."""
        if self.centroid is None:
            return 0.0
        return cosine_similarity(embedding, self.centroid)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SpeakerDiarizer:
    """
    Speaker diarization using voice embeddings and online clustering.
    
    Supports multiple embedding backends:
    - speechbrain: High quality, heavier
    - resemblyzer: Good balance of quality and speed
    - pyannote: State-of-the-art but requires license for some features
    
    Usage:
        diarizer = SpeakerDiarizer()
        diarized = diarizer.process(speech_segments)
    """
    
    def __init__(
        self,
        backend: str = "speechbrain",
        similarity_threshold: float = 0.75,
        max_speakers: int = 10,
        min_segment_duration: float = 0.5,
        debug: bool = True,
    ):
        """
        Args:
            backend: Embedding model backend
            similarity_threshold: Minimum similarity to match existing speaker
            max_speakers: Maximum number of speakers to track
            min_segment_duration: Minimum segment duration for embedding
        """
        self.backend = backend
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.min_segment_duration = min_segment_duration
        self.debug = debug
        
        self._encoder = None
        self._speakers: Dict[str, SpeakerProfile] = {}
        self._next_speaker_id = 0

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[DEBUG][Diarizer] {message}")
    
    def _load_encoder(self):
        """Lazy load the embedding model."""
        if self._encoder is not None:
            return
        
        self._debug(f"Loading encoder backend={self.backend}")
        
        if self.backend == "resemblyzer":
            self._load_resemblyzer()
        elif self.backend == "speechbrain":
            self._load_speechbrain()
        elif self.backend == "pyannote":
            self._load_pyannote()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _load_resemblyzer(self):
        """Load Resemblyzer embedding model."""
        ## TBD FUTURE
        return    
    
    def _load_speechbrain(self):
        """Load SpeechBrain ECAPA-TDNN encoder (high quality)."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self._encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
            )
            self._embed_fn = self._embed_speechbrain
        except ImportError as e:
            raise ImportError(
                "SpeechBrain required. Install with: pip install speechbrain"
            ) from e
        except TypeError as e:
            # Common when huggingface_hub >=0.24 removed use_auth_token
            try:
                import huggingface_hub  # type: ignore
                hf_version = getattr(huggingface_hub, "__version__", "unknown")
            except Exception:
                hf_version = "unknown"
            raise RuntimeError(
                "SpeechBrain encoder load failed due to huggingface_hub API change "
                f"(version={hf_version}). Pin huggingface_hub<0.24.0, e.g.: "
                "'pip install \"huggingface_hub<0.24\"', or switch "
                "PipelineConfig.diarization_backend to 'simple' for a lightweight fallback."
            ) from e
        self._debug("Loaded SpeechBrain encoder (ecapa-voxceleb)")
    
    def _load_pyannote(self):
        """Load pyannote embedding model."""
        # TBD FUTURE
        return
        
    
    def _embed_speechbrain(self, audio: np.ndarray) -> np.ndarray:
        """Get embedding using SpeechBrain."""
        import torch
        # SpeechBrain expects torch tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        embedding = self._encoder.encode_batch(audio_tensor)
        return embedding.squeeze().numpy()
    
    
    def get_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio segment.
        
        Returns None if segment is too short.
        """
        self._load_encoder()
        
        duration = len(audio) / SAMPLE_RATE
        if duration < self.min_segment_duration:
            self._debug(
                f"Segment duration {duration:.2f}s below min "
                f"{self.min_segment_duration}s; skipping embedding"
            )
            return None
        
        try:
            return self._embed_fn(audio)
        except Exception as e:
            print(f"Warning: Embedding extraction failed: {e}")
            return None
    
    def _match_or_create_speaker(self, embedding: np.ndarray) -> str:
        """
        Match embedding to existing speaker or create new one.
        
        Returns speaker_id.
        """
        best_match = None
        best_similarity = 0.0
        
        # Find best matching speaker
        for speaker_id, profile in self._speakers.items():
            sim = profile.similarity(embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_match = speaker_id
        
        # Check if match is good enough
        if best_match and best_similarity >= self.similarity_threshold:
            self._speakers[best_match].add_embedding(embedding)
            self._debug(
                f"Matched speaker {best_match} (similarity={best_similarity:.2f})"
            )
            return best_match
        
        # Create new speaker if under limit
        if len(self._speakers) < self.max_speakers:
            new_id = f"speaker_{self._next_speaker_id}"
            self._next_speaker_id += 1
            self._speakers[new_id] = SpeakerProfile(speaker_id=new_id)
            self._speakers[new_id].add_embedding(embedding)
            self._debug(
                f"Created new speaker {new_id} "
                f"(similarity={best_similarity:.2f})"
            )
            return new_id
        
        # Over limit - assign to closest speaker
        if best_match:
            self._speakers[best_match].add_embedding(embedding)
            self._debug(
                f"Assigned to closest speaker {best_match} "
                f"(over limit, similarity={best_similarity:.2f})"
            )
            return best_match
        
        # Fallback
        self._debug("No suitable speaker match; marking as unknown")
        return "speaker_unknown"
    
    def process(self, segments: List[SpeechSegment]) -> List[DiarizedSegment]:
        """
        Assign speaker IDs to speech segments.
        
        Args:
            segments: List of SpeechSegment from VAD
            
        Returns:
            List of DiarizedSegment with speaker assignments
        """
        self._load_encoder()
        self._debug(f"Processing {len(segments)} VAD segments")
        
        diarized = []
        
        for segment in segments:
            embedding = self.get_embedding(segment.audio)
            
            if embedding is not None:
                speaker_id = self._match_or_create_speaker(embedding)
            else:
                # Too short for reliable embedding
                speaker_id = "speaker_unknown"
                embedding = None
                self._debug(
                    f"Segment {segment.start_sec:.2f}-{segment.end_sec:.2f}s "
                    "too short for embedding"
                )
            
            diarized.append(DiarizedSegment(
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                audio=segment.audio,
                speaker_id=speaker_id,
                embedding=embedding,
            ))
        
        self._debug(f"Diarized into {len(diarized)} segments; speakers={len(self._speakers)}")
        return diarized
    
    def reset(self):
        """Reset speaker profiles (e.g., for new session)."""
        self._speakers.clear()
        self._next_speaker_id = 0
    
    def get_speaker_count(self) -> int:
        """Return number of detected speakers."""
        return len(self._speakers)
    
    def get_speaker_stats(self) -> Dict[str, int]:
        """Return utterance count per speaker."""
        return {
            sid: profile.utterance_count 
            for sid, profile in self._speakers.items()
        }


class SimpleDiarizer:
    """
    Simplified diarizer for testing without heavy dependencies.
    Uses basic audio features for speaker differentiation.
    """
    
    def __init__(self, similarity_threshold: float = 0.8, debug: bool = False):
        self.similarity_threshold = similarity_threshold
        self._speakers: Dict[str, np.ndarray] = {}
        self._next_id = 0
        self.debug = debug

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[DEBUG][SimpleDiarizer] {message}")
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract simple acoustic features."""
        # Basic features: pitch proxy, energy, zero-crossing rate
        features = []
        
        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        features.append(rms)
        
        # Zero-crossing rate
        zcr = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
        features.append(zcr)
        
        # Spectral centroid proxy (simple)
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
        if np.sum(fft) > 0:
            centroid = np.sum(freqs * fft) / np.sum(fft)
        else:
            centroid = 0
        features.append(centroid / 8000)  # Normalize
        
        return np.array(features)
    
    def process(self, segments: List[SpeechSegment]) -> List[DiarizedSegment]:
        """Simple feature-based diarization."""
        self._debug(f"Processing {len(segments)} segments (simple diarizer)")
        diarized = []
        
        for segment in segments:
            features = self._extract_features(segment.audio)
            speaker_id = self._match_speaker(features)
            
            diarized.append(DiarizedSegment(
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                audio=segment.audio,
                speaker_id=speaker_id,
                embedding=features,
            ))
        
        self._debug(f"Diarized into {len(diarized)} segments; speakers={len(self._speakers)}")
        return diarized
    
    def _match_speaker(self, features: np.ndarray) -> str:
        """Match features to speaker or create new."""
        best_match = None
        best_sim = 0.0
        
        for sid, stored in self._speakers.items():
            sim = cosine_similarity(features, stored)
            if sim > best_sim:
                best_sim = sim
                best_match = sid
        
        if best_match and best_sim >= self.similarity_threshold:
            # Update running average
            self._speakers[best_match] = 0.9 * self._speakers[best_match] + 0.1 * features
            return best_match
        
        # New speaker
        new_id = f"speaker_{self._next_id}"
        self._next_id += 1
        self._speakers[new_id] = features
        return new_id
    
    def reset(self):
        self._speakers.clear()
        self._next_id = 0


# =============================================================================
# FACTORY
# =============================================================================

def create_diarizer(
    backend: str = "resemblyzer",
    **kwargs
) -> "SpeakerDiarizer | SimpleDiarizer":
    """
    Create a speaker diarizer.
    
    Args:
        backend: "resemblyzer", "speechbrain", "pyannote", or "simple"
        **kwargs: Backend-specific configuration
        
    Returns:
        Diarizer instance
    """
    if backend == "simple":
        return SimpleDiarizer(**kwargs)
    else:
        return SpeakerDiarizer(backend=backend, **kwargs)
