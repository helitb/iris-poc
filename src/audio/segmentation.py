"""
Utterance segmentation and grouping.

Groups consecutive speech segments from the same speaker into utterances,
handling gaps, overlaps, and speaker changes.
"""

from typing import List, Optional, Tuple
import numpy as np

from .types import (
    DiarizedSegment, Utterance,
    SAMPLE_RATE, MIN_UTTERANCE_DURATION_SEC, MAX_UTTERANCE_GAP_SEC,
)


class UtteranceSegmenter:
    """
    Groups consecutive diarized segments into utterances.
    
    Rules:
    - Consecutive segments from same speaker are merged
    - Gaps longer than max_gap_sec create new utterance
    - Short segments (< min_duration) may be discarded or merged
    
    Usage:
        segmenter = UtteranceSegmenter()
        utterances = segmenter.process(diarized_segments)
    """
    
    def __init__(
        self,
        min_duration_sec: float = MIN_UTTERANCE_DURATION_SEC,
        max_gap_sec: float = MAX_UTTERANCE_GAP_SEC,
        merge_short_segments: bool = True,
    ):
        """
        Args:
            min_duration_sec: Minimum utterance duration to keep
            max_gap_sec: Maximum gap within same-speaker utterance
            merge_short_segments: Whether to merge short segments with neighbors
        """
        self.min_duration_sec = min_duration_sec
        self.max_gap_sec = max_gap_sec
        self.merge_short_segments = merge_short_segments
    
    def process(self, segments: List[DiarizedSegment]) -> List[Utterance]:
        """
        Group segments into utterances.
        
        Args:
            segments: List of diarized segments (should be sorted by time)
            
        Returns:
            List of Utterance objects
        """
        if not segments:
            return []
        
        # Sort by start time
        segments = sorted(segments, key=lambda s: s.start_sec)
        
        utterances = []
        current: Optional[_UtteranceBuilder] = None
        
        for segment in segments:
            if current is None:
                # Start first utterance
                current = _UtteranceBuilder(segment)
            
            elif self._should_merge(current, segment):
                # Extend current utterance
                current.add_segment(segment)
            
            else:
                # Finish current, start new
                utt = current.build()
                if utt.duration_sec >= self.min_duration_sec:
                    utterances.append(utt)
                current = _UtteranceBuilder(segment)
        
        # Don't forget the last one
        if current is not None:
            utt = current.build()
            if utt.duration_sec >= self.min_duration_sec:
                utterances.append(utt)
        
        # Optional: merge isolated short segments
        if self.merge_short_segments:
            utterances = self._merge_short_utterances(utterances)
        
        return utterances
    
    def _should_merge(self, current: "_UtteranceBuilder", segment: DiarizedSegment) -> bool:
        """Determine if segment should be merged into current utterance."""
        # Different speaker = new utterance
        if segment.speaker_id != current.speaker_id:
            return False
        
        # Gap too long = new utterance
        gap = segment.start_sec - current.end_sec
        if gap > self.max_gap_sec:
            return False
        
        return True
    
    def _merge_short_utterances(self, utterances: List[Utterance]) -> List[Utterance]:
        """
        Merge very short utterances with adjacent ones from same speaker.
        """
        if len(utterances) < 2:
            return utterances
        
        merged = []
        i = 0
        
        while i < len(utterances):
            current = utterances[i]
            
            # Check if this is a short utterance that can be merged
            if (current.duration_sec < self.min_duration_sec * 2 and 
                i + 1 < len(utterances)):
                
                next_utt = utterances[i + 1]
                
                # Merge with next if same speaker and close
                if (next_utt.speaker_id == current.speaker_id and
                    next_utt.start_sec - current.end_sec < self.max_gap_sec * 2):
                    
                    # Create merged utterance
                    merged_audio = np.concatenate([current.audio, next_utt.audio])
                    merged_utt = Utterance(
                        speaker_id=current.speaker_id,
                        start_sec=current.start_sec,
                        end_sec=next_utt.end_sec,
                        audio=merged_audio,
                        num_segments=current.num_segments + next_utt.num_segments,
                        has_internal_pauses=True,
                    )
                    merged.append(merged_utt)
                    i += 2
                    continue
            
            merged.append(current)
            i += 1
        
        return merged


class _UtteranceBuilder:
    """Helper class to build an utterance from segments."""
    
    def __init__(self, initial_segment: DiarizedSegment):
        self.speaker_id = initial_segment.speaker_id
        self.start_sec = initial_segment.start_sec
        self.end_sec = initial_segment.end_sec
        self.segments: List[DiarizedSegment] = [initial_segment]
    
    def add_segment(self, segment: DiarizedSegment):
        """Add a segment to this utterance."""
        self.segments.append(segment)
        self.end_sec = max(self.end_sec, segment.end_sec)
    
    def build(self) -> Utterance:
        """Build the final Utterance object."""
        # Concatenate audio from all segments
        audio_parts = [s.audio for s in self.segments]
        combined_audio = np.concatenate(audio_parts)
        
        # Check for internal pauses
        has_pauses = len(self.segments) > 1
        if has_pauses:
            # Check if there are actual gaps
            for i in range(1, len(self.segments)):
                gap = self.segments[i].start_sec - self.segments[i-1].end_sec
                if gap > 0.1:  # 100ms gap
                    has_pauses = True
                    break
            else:
                has_pauses = False
        
        return Utterance(
            speaker_id=self.speaker_id,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            audio=combined_audio,
            num_segments=len(self.segments),
            has_internal_pauses=has_pauses,
        )


class OverlapDetector:
    """
    Detect and handle overlapping speech from multiple speakers.
    """
    
    def __init__(self, overlap_threshold_sec: float = 0.1):
        """
        Args:
            overlap_threshold_sec: Minimum overlap duration to consider
        """
        self.overlap_threshold = overlap_threshold_sec
    
    def find_overlaps(
        self,
        utterances: List[Utterance],
    ) -> List[Tuple[Utterance, Utterance, float, float]]:
        """
        Find overlapping utterances.
        
        Returns:
            List of (utt1, utt2, overlap_start, overlap_end)
        """
        overlaps = []
        
        # Sort by start time
        sorted_utts = sorted(utterances, key=lambda u: u.start_sec)
        
        for i, utt1 in enumerate(sorted_utts):
            for utt2 in sorted_utts[i+1:]:
                # If utt2 starts after utt1 ends, no more overlaps possible
                if utt2.start_sec >= utt1.end_sec:
                    break
                
                # Calculate overlap
                overlap_start = max(utt1.start_sec, utt2.start_sec)
                overlap_end = min(utt1.end_sec, utt2.end_sec)
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration >= self.overlap_threshold:
                    overlaps.append((utt1, utt2, overlap_start, overlap_end))
        
        return overlaps
    
    def mark_overlaps(self, utterances: List[Utterance]) -> List[Utterance]:
        """
        Mark utterances that overlap with others.
        Returns same utterances with potential modifications.
        """
        overlaps = self.find_overlaps(utterances)
        
        # Create a set of utterance IDs that have overlaps
        overlapping_ids = set()
        for utt1, utt2, _, _ in overlaps:
            overlapping_ids.add(id(utt1))
            overlapping_ids.add(id(utt2))
        
        # We can't modify dataclass instances easily,
        # so this is more for analysis
        return utterances


class TurnTakingAnalyzer:
    """
    Analyze turn-taking patterns between speakers.
    """
    
    def __init__(self):
        self.turn_history: List[Tuple[str, float, float]] = []
    
    def analyze_gap(
        self,
        current_utterance: Utterance,
        previous_utterance: Optional[Utterance],
    ) -> Tuple[Optional[int], bool, Optional[str]]:
        """
        Analyze the gap before current utterance.
        
        Returns:
            (gap_ms, is_overlap, previous_speaker_id)
        """
        if previous_utterance is None:
            return None, False, None
        
        gap_sec = current_utterance.start_sec - previous_utterance.end_sec
        gap_ms = int(gap_sec * 1000)
        
        is_overlap = gap_sec < 0
        
        return gap_ms, is_overlap, previous_utterance.speaker_id
    
    def classify_turn_taking(self, gap_ms: int) -> str:
        """
        Classify the type of turn-taking based on gap duration.
        
        Categories:
        - "overlap": negative gap (simultaneous speech)
        - "latch": very quick response (< 200ms)
        - "normal": typical response time (200ms - 1s)
        - "delayed": longer pause (1s - 3s)
        - "extended_pause": very long pause (> 3s)
        """
        if gap_ms < 0:
            return "overlap"
        elif gap_ms < 200:
            return "latch"
        elif gap_ms < 1000:
            return "normal"
        elif gap_ms < 3000:
            return "delayed"
        else:
            return "extended_pause"
    
    def add_turn(self, speaker_id: str, start_sec: float, end_sec: float):
        """Record a turn for history tracking."""
        self.turn_history.append((speaker_id, start_sec, end_sec))
    
    def get_speaker_stats(self) -> dict:
        """Get speaking statistics per speaker."""
        stats = {}
        
        for speaker_id, start, end in self.turn_history:
            if speaker_id not in stats:
                stats[speaker_id] = {
                    'turn_count': 0,
                    'total_duration': 0.0,
                    'turns': [],
                }
            
            stats[speaker_id]['turn_count'] += 1
            stats[speaker_id]['total_duration'] += (end - start)
            stats[speaker_id]['turns'].append((start, end))
        
        # Calculate averages
        for speaker_id, data in stats.items():
            if data['turn_count'] > 0:
                data['avg_turn_duration'] = data['total_duration'] / data['turn_count']
        
        return stats
    
    def reset(self):
        """Clear turn history."""
        self.turn_history.clear()


# =============================================================================
# FACTORY
# =============================================================================

def create_segmenter(**kwargs) -> UtteranceSegmenter:
    """Create an utterance segmenter."""
    return UtteranceSegmenter(**kwargs)
