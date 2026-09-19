from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class PitchAlgorithm(ABC):
    def __init__(self, sample_rate: int, hop_size: int, fmin: float, fmax: float):
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if hop_size <= 0:
            raise ValueError(f"Hop size must be positive, got {hop_size}")

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

    @property
    def supports_continuous_periodicity(self) -> bool:
        return isinstance(self, ContinuousPitchAlgorithm)

    def _validate_audio(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            raise ValueError("Empty audio input")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite values")
        if np.any(np.abs(audio) > 1.0):
            raise ValueError("Audio must be normalized to [-1.0, 1.0]")

    def _compute_target_times(self, audio_length: int) -> np.ndarray:
        n_hops = audio_length // self.hop_size
        return np.arange(n_hops) * (self.hop_size / self.sample_rate)

    def _align_to_grid(
        self,
        algorithm_times: np.ndarray,
        values: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        if len(algorithm_times) == 0:
            return np.zeros_like(target_times)
        return np.interp(target_times, algorithm_times, values, left=0.0, right=0.0)

    def _sanity_check(
        self, pitch: np.ndarray, periodicity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        periodicity = np.nan_to_num(periodicity, nan=0.0)
        pitch = np.nan_to_num(pitch, nan=0.0)

        voiced = periodicity > 0
        pitch[~voiced] = 0.0
        # 推定後の [fmin, fmax] クリップは **行わない**。RMVPE の出力は 360 ビン固定
        # （約 31.7–2005.5 Hz）で、レンジはモデルに渡らない。ここでクリップすると
        # レンジ外の値が捨てられるのではなく境界値に張り付き、実在しない平坦な
        # ロングトーンを作ってしまう（pyworld / praat 時代の名残）。

        periodicity = np.clip(periodicity, 0.0, 1.0)
        return pitch, periodicity

    def notes_from_pitch_contour(
        self,
        pitch_contour: np.ndarray,
        voicing_contour: np.ndarray,
        split_semitone_threshold: float = 0.8,
        min_note_duration: float = 0.05,
        unvoiced_grace_period: float = 0.02,
    ) -> List[Dict[str, float]]:
        frame_period = self.hop_size / self.sample_rate
        notes = []
        current_note_segment = None
        unvoiced_frames_count = 0

        valid_voiced_frames = (
            (voicing_contour > 0)
            & (pitch_contour >= self.fmin)
            & (pitch_contour <= self.fmax)
        )

        midi_contour = np.full_like(pitch_contour, np.nan)
        valid_indices = np.where(valid_voiced_frames)[0]
        if len(valid_indices) > 0:
            midi_contour[valid_indices] = 69 + 12 * np.log2(
                pitch_contour[valid_indices] / 440.0
            )

        for i, is_voiced in enumerate(valid_voiced_frames):
            t = i * frame_period
            if is_voiced:
                unvoiced_frames_count = 0
                midi_pitch = midi_contour[i]
                if current_note_segment is None:
                    current_note_segment = {
                        "start": t,
                        "end": t + frame_period,
                        "samples": [midi_pitch],
                    }
                else:
                    current_median = np.median(current_note_segment["samples"])
                    pitch_deviation = abs(midi_pitch - current_median)
                    if pitch_deviation >= split_semitone_threshold:
                        notes.append(current_note_segment)
                        current_note_segment = {
                            "start": t,
                            "end": t + frame_period,
                            "samples": [midi_pitch],
                        }
                    else:
                        current_note_segment["samples"].append(midi_pitch)
                        current_note_segment["end"] = t + frame_period
            else:
                if current_note_segment is not None:
                    unvoiced_frames_count += 1
                    unvoiced_duration = unvoiced_frames_count * frame_period
                    if unvoiced_duration >= unvoiced_grace_period:
                        notes.append(current_note_segment)
                        current_note_segment = None
                    else:
                        current_note_segment["end"] = t + frame_period

        if current_note_segment is not None:
            notes.append(current_note_segment)
        if not notes:
            return []

        processed_notes = []
        for segment in notes:
            duration = segment["end"] - segment["start"]
            if duration >= min_note_duration and segment["samples"]:
                median_pitch = np.median(segment["samples"])
                processed_notes.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "midi_pitch": round(median_pitch),
                    }
                )
        if not processed_notes:
            return []

        final_notes = [processed_notes[0]]
        epsilon = 1e-9
        for current_note in processed_notes[1:]:
            previous_note = final_notes[-1]
            gap = current_note["start"] - previous_note["end"]
            if (
                gap <= frame_period + epsilon
                and previous_note["midi_pitch"] == current_note["midi_pitch"]
            ):
                previous_note["end"] = current_note["end"]
            else:
                final_notes.append(current_note)

        return final_notes

    def extract_pitch(
        self,
        audio: np.ndarray,
        thresholds: Optional[Union[float, List[float]]] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]],
        List[Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]],
    ]:
        if thresholds is None:
            thresholds = [self._get_default_threshold()]
        elif isinstance(thresholds, (int, float)):
            thresholds = [float(thresholds)]
        else:
            thresholds = list(thresholds)

        if self.supports_continuous_periodicity:
            results = self._extract_continuous_multiple_thresholds(audio, thresholds)
        else:
            results = self._extract_threshold_multiple_thresholds(audio, thresholds)

        return results[0] if len(results) == 1 else results

    @abstractmethod
    def _get_default_threshold(self) -> float:
        pass

    @classmethod
    def get_name(cls) -> str:
        return getattr(cls, "_name", cls.__name__.replace("PitchAlgorithm", ""))


class ContinuousPitchAlgorithm(PitchAlgorithm):
    @abstractmethod
    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def _get_default_threshold(self) -> float:
        return 0.5

    def extract_continuous_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_audio(audio)
        times, pitch, periodicity = self._extract_raw_pitch_and_periodicity(audio)
        pitch, periodicity = self._sanity_check(pitch, periodicity)
        target_times = self._compute_target_times(len(audio))
        aligned_pitch = self._align_to_grid(times, pitch, target_times)
        aligned_periodicity = self._align_to_grid(times, periodicity, target_times)
        return aligned_pitch, aligned_periodicity

    def _extract_continuous_multiple_thresholds(
        self, audio: np.ndarray, thresholds: List[float]
    ) -> List[Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]]:
        pitch, confidence = self.extract_continuous_periodicity(audio)
        results = []
        for threshold in thresholds:
            voicing = (confidence >= threshold).astype(bool)
            notes = self.notes_from_pitch_contour(pitch, voicing)
            results.append((pitch, voicing, notes))
        return results


class ThresholdPitchAlgorithm(PitchAlgorithm):
    @abstractmethod
    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def _get_default_threshold(self) -> float:
        return 0.5

    def _extract_threshold_multiple_thresholds(
        self, audio: np.ndarray, thresholds: List[float]
    ) -> List[Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]]:
        self._validate_audio(audio)
        results = []
        target_times = self._compute_target_times(len(audio))
        for threshold in thresholds:
            times, pitch, periodicity = self._extract_pitch_with_threshold(
                audio, threshold
            )
            pitch, periodicity = self._sanity_check(pitch, periodicity)
            aligned_pitch = self._align_to_grid(times, pitch, target_times)
            aligned_periodicity = self._align_to_grid(times, periodicity, target_times)
            notes = self.notes_from_pitch_contour(
                aligned_pitch, aligned_periodicity.astype(bool)
            )
            results.append((aligned_pitch, aligned_periodicity.astype(bool), notes))
        return results

    def extract_continuous_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support continuous periodicity extraction. "
            f"Use extract_pitch() instead or check supports_continuous_periodicity property."
        )
