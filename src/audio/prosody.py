"""
Prosody extraction using Parselmouth (Praat in Python).

Extracts pitch, intensity, voice quality, and timing features
that are clinically relevant for autism assessment.
"""

from typing import Optional, Tuple
import numpy as np

from .types import ProsodyResult, SAMPLE_RATE


class ProsodyExtractor:
    """
    Extract prosodic features from speech using Praat/Parselmouth.
    
    Features extracted:
    - Pitch (F0): mean, std, min, max, contour
    - Intensity: mean, std, range
    - Voice quality: jitter, shimmer, HNR
    - Timing: speech rate estimation
    
    Usage:
        extractor = ProsodyExtractor()
        prosody = extractor.extract(audio)
    """
    
    def __init__(
        self,
        pitch_floor: float = 75.0,  # Hz - lower for adult males
        pitch_ceiling: float = 500.0,  # Hz - higher for children
        time_step: float = 0.01,  # 10ms analysis window
    ):
        """
        Args:
            pitch_floor: Minimum pitch to detect (Hz)
            pitch_ceiling: Maximum pitch to detect (Hz)
            time_step: Analysis time step (seconds)
        """
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.time_step = time_step
        
        self._parselmouth = None
    
    def _load_parselmouth(self):
        """Lazy load parselmouth."""
        if self._parselmouth is not None:
            return
        
        try:
            import parselmouth
            self._parselmouth = parselmouth
        except ImportError as e:
            raise ImportError(
                "Parselmouth required. Install with: pip install praat-parselmouth"
            ) from e
    
    def _create_sound(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
        """Create Parselmouth Sound object from numpy array."""
        self._load_parselmouth()
        
        # Ensure float64 for Praat
        if audio.dtype != np.float64:
            audio = audio.astype(np.float64)
        
        return self._parselmouth.Sound(audio, sampling_frequency=sample_rate)
    
    def extract(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> ProsodyResult:
        """
        Extract prosodic features from audio.
        
        Args:
            audio: Audio samples (float, mono)
            sample_rate: Sample rate
            
        Returns:
            ProsodyResult with extracted features
        """
        self._load_parselmouth()
        
        sound = self._create_sound(audio, sample_rate)
        
        # Extract pitch
        pitch_features = self._extract_pitch(sound)
        
        # Extract intensity
        intensity_features = self._extract_intensity(sound)
        
        # Extract voice quality
        voice_quality = self._extract_voice_quality(sound)
        
        # Estimate speech rate
        speech_rate = self._estimate_speech_rate(sound)
        
        return ProsodyResult(
            # Pitch
            pitch_mean_hz=pitch_features.get('mean'),
            pitch_std_hz=pitch_features.get('std'),
            pitch_min_hz=pitch_features.get('min'),
            pitch_max_hz=pitch_features.get('max'),
            pitch_contour=pitch_features.get('contour'),
            
            # Intensity
            intensity_mean_db=intensity_features.get('mean'),
            intensity_std_db=intensity_features.get('std'),
            intensity_range_db=intensity_features.get('range'),
            
            # Voice quality
            jitter=voice_quality.get('jitter'),
            shimmer=voice_quality.get('shimmer'),
            hnr_db=voice_quality.get('hnr'),
            voice_quality=voice_quality.get('quality_label'),
            
            # Speech rate
            speech_rate_syl_per_sec=speech_rate.get('syllables_per_sec'),
            speech_rate_category=speech_rate.get('category'),
            articulation_rate=speech_rate.get('articulation_rate'),
        )
    
    def _extract_pitch(self, sound) -> dict:
        """Extract pitch (F0) features."""
        pm = self._parselmouth
        
        try:
            pitch = sound.to_pitch_ac(
                time_step=self.time_step,
                pitch_floor=self.pitch_floor,
                pitch_ceiling=self.pitch_ceiling,
            )
            
            # Get pitch values (excluding unvoiced frames)
            pitch_values = pitch.selected_array['frequency']
            voiced_values = pitch_values[pitch_values > 0]
            
            if len(voiced_values) == 0:
                return {'contour': 'unvoiced'}
            
            mean_pitch = float(np.mean(voiced_values))
            std_pitch = float(np.std(voiced_values))
            min_pitch = float(np.min(voiced_values))
            max_pitch = float(np.max(voiced_values))
            
            # Determine contour shape
            contour = self._classify_pitch_contour(voiced_values)
            
            return {
                'mean': mean_pitch,
                'std': std_pitch,
                'min': min_pitch,
                'max': max_pitch,
                'contour': contour,
            }
            
        except Exception as e:
            print(f"Warning: Pitch extraction failed: {e}")
            return {}
    
    def _classify_pitch_contour(self, pitch_values: np.ndarray) -> str:
        """Classify the overall pitch contour shape."""
        if len(pitch_values) < 3:
            return "flat"
        
        # Compare first and last third
        third = len(pitch_values) // 3
        if third == 0:
            return "flat"
        
        start_mean = np.mean(pitch_values[:third])
        end_mean = np.mean(pitch_values[-third:])
        
        # Calculate variance
        variance = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) > 0 else 0
        
        # Thresholds
        rise_threshold = 1.1  # 10% rise
        fall_threshold = 0.9  # 10% fall
        variable_threshold = 0.15  # 15% coefficient of variation
        
        if variance > variable_threshold:
            return "variable"
        elif end_mean > start_mean * rise_threshold:
            return "rising"
        elif end_mean < start_mean * fall_threshold:
            return "falling"
        else:
            return "flat"
    
    def _extract_intensity(self, sound) -> dict:
        """Extract intensity features."""
        try:
            intensity = sound.to_intensity(
                minimum_pitch=self.pitch_floor,
                time_step=self.time_step,
            )
            
            intensity_values = intensity.values[0]
            # Filter out very low values (silence)
            voiced_intensity = intensity_values[intensity_values > 40]  # dB threshold
            
            if len(voiced_intensity) == 0:
                return {}
            
            mean_int = float(np.mean(voiced_intensity))
            std_int = float(np.std(voiced_intensity))
            range_int = float(np.max(voiced_intensity) - np.min(voiced_intensity))
            
            return {
                'mean': mean_int,
                'std': std_int,
                'range': range_int,
            }
            
        except Exception as e:
            print(f"Warning: Intensity extraction failed: {e}")
            return {}
    
    def _extract_voice_quality(self, sound) -> dict:
        """Extract voice quality measures (jitter, shimmer, HNR)."""
        pm = self._parselmouth
        
        try:
            # Create PointProcess for voice analysis
            point_process = pm.praat.call(
                sound, "To PointProcess (periodic, cc)",
                self.pitch_floor, self.pitch_ceiling
            )
            
            # Jitter (pitch perturbation)
            jitter = pm.praat.call(
                point_process, "Get jitter (local)",
                0, 0, 0.0001, 0.02, 1.3
            )
            
            # Shimmer (amplitude perturbation)
            shimmer = pm.praat.call(
                [sound, point_process], "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )
            
            # Harmonics-to-Noise Ratio
            harmonicity = sound.to_harmonicity()
            hnr_values = harmonicity.values[0]
            hnr_values = hnr_values[hnr_values > -200]  # Filter undefined values
            hnr = float(np.mean(hnr_values)) if len(hnr_values) > 0 else None
            
            # Classify voice quality
            quality_label = self._classify_voice_quality(jitter, shimmer, hnr)
            
            return {
                'jitter': float(jitter) if not np.isnan(jitter) else None,
                'shimmer': float(shimmer) if not np.isnan(shimmer) else None,
                'hnr': hnr,
                'quality_label': quality_label,
            }
            
        except Exception as e:
            print(f"Warning: Voice quality extraction failed: {e}")
            return {}
    
    def _classify_voice_quality(
        self,
        jitter: Optional[float],
        shimmer: Optional[float],
        hnr: Optional[float],
    ) -> str:
        """Classify voice quality based on acoustic measures."""
        # Typical thresholds (these vary by age/gender)
        # Higher jitter/shimmer = more perturbed voice
        # Lower HNR = more noise in voice
        
        if jitter is None or shimmer is None:
            return "modal"  # Default
        
        if hnr is not None and hnr < 10:
            return "breathy"  # Low HNR indicates breathiness
        
        if jitter > 0.02 or shimmer > 0.1:
            return "tense"  # High perturbation indicates tension
        
        if jitter < 0.005 and shimmer < 0.03:
            return "modal"  # Normal voice
        
        if hnr is not None and hnr > 20:
            return "modal"  # Good HNR
        
        return "creaky"  # Default for irregular patterns
    
    def _estimate_speech_rate(self, sound) -> dict:
        """
        Estimate speech rate using intensity-based syllable detection.
        """
        try:
            intensity = sound.to_intensity(minimum_pitch=self.pitch_floor)
            
            # Get intensity values
            int_values = intensity.values[0]
            times = intensity.xs()
            
            # Find peaks (syllable nuclei)
            peaks = self._find_intensity_peaks(int_values, times)
            
            # Calculate duration (excluding silence)
            duration = sound.get_total_duration()
            
            if duration > 0 and len(peaks) > 0:
                syllables_per_sec = len(peaks) / duration
                
                # Categorize
                if syllables_per_sec < 3:
                    category = "slow"
                elif syllables_per_sec < 6:
                    category = "normal"
                else:
                    category = "fast"
                
                return {
                    'syllables_per_sec': float(syllables_per_sec),
                    'category': category,
                    'articulation_rate': float(syllables_per_sec),
                }
            
            return {}
            
        except Exception as e:
            print(f"Warning: Speech rate estimation failed: {e}")
            return {}
    
    def _find_intensity_peaks(
        self,
        intensity: np.ndarray,
        times: np.ndarray,
        min_dip: float = 2.0,  # dB
    ) -> list:
        """Find intensity peaks (potential syllable nuclei)."""
        peaks = []
        
        if len(intensity) < 3:
            return peaks
        
        # Simple peak detection
        for i in range(1, len(intensity) - 1):
            if intensity[i] > intensity[i-1] and intensity[i] > intensity[i+1]:
                # Check if it's a significant peak
                left_dip = intensity[i] - intensity[i-1]
                right_dip = intensity[i] - intensity[i+1]
                if left_dip > min_dip or right_dip > min_dip:
                    peaks.append(times[i])
        
        return peaks


def extract_prosody_simple(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """
    Simple prosody extraction without Parselmouth.
    Fallback for environments without Praat.
    """
    features = {}
    
    # Basic intensity (RMS)
    rms = np.sqrt(np.mean(audio ** 2))
    features['intensity_rms'] = float(rms)
    
    # Zero-crossing rate (rough pitch proxy)
    zcr = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
    features['zcr'] = float(zcr)
    
    # Simple energy contour
    frame_size = int(sample_rate * 0.025)  # 25ms frames
    hop_size = int(sample_rate * 0.010)  # 10ms hop
    
    energies = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        energies.append(np.sqrt(np.mean(frame ** 2)))
    
    if energies:
        features['energy_mean'] = float(np.mean(energies))
        features['energy_std'] = float(np.std(energies))
    
    return features


# =============================================================================
# FACTORY
# =============================================================================

def create_prosody_extractor(**kwargs) -> ProsodyExtractor:
    """Create a prosody extractor instance."""
    return ProsodyExtractor(**kwargs)
