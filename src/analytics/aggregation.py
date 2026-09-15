"""
Temporal OCR aggregation - the feature that turns a demo into a system.

The problem
-----------
A single frame's OCR is a noisy measurement. The same plate, read across four
frames, might give:

    frame 100  UP32AB1234  0.72
    frame 110  UP32AB1234  0.89
    frame 120  UP32A81234  0.63     <- B misread as 8, low confidence
    frame 130  UP32AB1234  0.94

Trusting any single frame is a coin flip. Trusting frame 120 would be wrong.

Why aggregation works
---------------------
Each frame is a fresh sample of the same underlying string, taken at a different
distance, angle, motion blur and exposure. The *errors* are largely independent
across frames while the *signal* is constant, so errors do not reinforce each
other but agreements do. That is the same argument behind ensembling: averaging
independent noisy estimators reduces variance. Reading the plate from twenty
metres and from five metres are almost different sensors.

How it is implemented
---------------------
1. **Weighted voting over full strings.** Each observation votes with weight

       w = ocr_conf ** conf_power * format_weight

   ``conf_power > 1`` sharpens the influence of confident readings; the format
   weight upgrades strings that match a valid Indian layout and penalises ones
   that match nothing. The string with the greatest total weight wins.

2. **Aggregated confidence.** The weighted mean confidence of the observations
   that agree with the winner, plus a small consensus bonus that grows with the
   number of agreeing observations and is capped. Agreement is evidence, but it
   should not be able to manufacture 99% from four mediocre reads.

3. **Character-level fallback.** When no string has a clear majority (every
   frame disagrees slightly) the aggregator votes *per character position*
   across same-length candidates. This can reconstruct the correct plate even
   when no individual frame got every character right - each position only needs
   a plurality. This is reported explicitly so the result is never silently
   synthetic.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from config.config import AggregationConfig
from src.ocr.text_validator import ValidationStatus

FORMAT_WEIGHTS = {
    ValidationStatus.VALID_FORMAT.value: "valid_format_weight",
    ValidationStatus.SUSPICIOUS_FORMAT.value: "suspicious_format_weight",
}


@dataclass
class Observation:
    """One OCR reading of one tracked vehicle at one point in time."""

    text: str
    confidence: float
    frame_index: int
    status: str = ValidationStatus.SUSPICIOUS_FORMAT.value
    detection_confidence: float = 0.0
    variant: str = ""


@dataclass
class AggregatedPlate:
    """Fused result for one tracked vehicle."""

    track_id: int
    text: str
    confidence: float
    observations: int
    agreement: float                    # share of observations backing the winner
    method: str = "weighted_vote"       # or 'character_vote'
    status: str = ValidationStatus.SUSPICIOUS_FORMAT.value
    vehicle_type: str = "unknown"
    detection_confidence: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    candidates: List[tuple] = field(default_factory=list)  # [(text, weight)]

    @property
    def is_reliable(self) -> bool:
        return (
            self.observations >= 2
            and self.confidence >= 0.5
            and self.status == ValidationStatus.VALID_FORMAT.value
        )

    def as_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "plate": self.text,
            "confidence": round(self.confidence, 4),
            "observations": self.observations,
            "agreement": round(self.agreement, 3),
            "method": self.method,
            "status": self.status,
            "vehicle_type": self.vehicle_type,
        }


class PlateAggregator:
    """Keeps a bounded history of readings per track and fuses them on demand."""

    def __init__(self, config: AggregationConfig, max_observations: int = 24):
        self.cfg = config
        self.max_observations = max_observations
        self._obs: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_observations))
        self._vehicle_type: Dict[int, str] = {}
        self._detection_conf: Dict[int, float] = {}

    # -- ingestion -------------------------------------------------------- #

    def add(
        self,
        track_id: int,
        text: str,
        confidence: float,
        frame_index: int,
        status: str = ValidationStatus.SUSPICIOUS_FORMAT.value,
        detection_confidence: float = 0.0,
        vehicle_type: str = "unknown",
        variant: str = "",
    ) -> None:
        """Record one observation. Empty strings are ignored."""
        if not text:
            return
        self._obs[track_id].append(
            Observation(
                text=text,
                confidence=float(confidence),
                frame_index=int(frame_index),
                status=status,
                detection_confidence=float(detection_confidence),
                variant=variant,
            )
        )
        self._vehicle_type[track_id] = vehicle_type
        self._detection_conf[track_id] = max(
            self._detection_conf.get(track_id, 0.0), float(detection_confidence)
        )

    def observation_count(self, track_id: int) -> int:
        return len(self._obs.get(track_id, ()))

    def observations(self, track_id: int) -> List[Observation]:
        return list(self._obs.get(track_id, ()))

    @property
    def track_ids(self) -> List[int]:
        return list(self._obs.keys())

    # -- weighting -------------------------------------------------------- #

    def _weight(self, obs: Observation) -> float:
        attr = FORMAT_WEIGHTS.get(obs.status)
        fmt_w = getattr(self.cfg, attr) if attr else self.cfg.invalid_format_weight
        return max(obs.confidence, 1e-4) ** self.cfg.conf_power * fmt_w

    # -- character-level fallback ----------------------------------------- #

    @staticmethod
    def _character_vote(observations: Iterable[Observation]) -> Optional[str]:
        """Majority-vote each character position among the modal-length strings."""
        obs = list(observations)
        if not obs:
            return None
        lengths = Counter(len(o.text) for o in obs)
        target_len, count = lengths.most_common(1)[0]
        if count < 2:
            return None
        pool = [o for o in obs if len(o.text) == target_len]

        chars: List[str] = []
        for i in range(target_len):
            votes: Dict[str, float] = defaultdict(float)
            for o in pool:
                votes[o.text[i]] += max(o.confidence, 1e-4)
            chars.append(max(votes.items(), key=lambda kv: kv[1])[0])
        return "".join(chars)

    # -- fusion ----------------------------------------------------------- #

    def aggregate(self, track_id: int) -> Optional[AggregatedPlate]:
        """Fuse all observations for one track into a single best answer."""
        observations = list(self._obs.get(track_id, ()))
        if not observations:
            return None

        weights: Dict[str, float] = defaultdict(float)
        for obs in observations:
            weights[obs.text] += self._weight(obs)

        ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        best_text, best_weight = ranked[0]
        total_weight = sum(weights.values()) or 1.0
        method = "weighted_vote"

        # No clear winner among full strings -> try per-character voting.
        if (
            self.cfg.character_vote_fallback
            and len(ranked) > 1
            and best_weight / total_weight < 0.5
            and len(observations) >= self.cfg.min_observations
        ):
            voted = self._character_vote(observations)
            if voted and voted not in weights:
                best_text = voted
                method = "character_vote"

        supporting = [o for o in observations if o.text == best_text]
        if not supporting:  # character vote produced a novel string
            lengths = Counter(len(o.text) for o in observations).most_common(1)[0][0]
            supporting = [o for o in observations if len(o.text) == lengths] or observations

        conf_weights = [max(o.confidence, 1e-4) for o in supporting]
        mean_conf = sum(c * c for c in conf_weights) / sum(conf_weights)

        bonus = min(
            self.cfg.max_agreement_bonus,
            self.cfg.agreement_bonus * max(0, len(supporting) - 1),
        )
        confidence = min(1.0, mean_conf + bonus)

        status_counts = Counter(o.status for o in supporting)
        status = status_counts.most_common(1)[0][0]
        if method == "character_vote":
            status = ValidationStatus.SUSPICIOUS_FORMAT.value

        frames = [o.frame_index for o in observations]
        return AggregatedPlate(
            track_id=track_id,
            text=best_text,
            confidence=round(confidence, 4),
            observations=len(observations),
            agreement=round(len(supporting) / len(observations), 3),
            method=method,
            status=status,
            vehicle_type=self._vehicle_type.get(track_id, "unknown"),
            detection_confidence=self._detection_conf.get(track_id, 0.0),
            first_frame=min(frames),
            last_frame=max(frames),
            candidates=[(t, round(w, 3)) for t, w in ranked[:5]],
        )

    def aggregate_all(self, min_observations: Optional[int] = None) -> List[AggregatedPlate]:
        """Aggregate every track that has enough observations."""
        floor = self.cfg.min_observations if min_observations is None else min_observations
        out: List[AggregatedPlate] = []
        for track_id in self._obs:
            if len(self._obs[track_id]) < floor:
                continue
            result = self.aggregate(track_id)
            if result:
                out.append(result)
        return sorted(out, key=lambda r: r.confidence, reverse=True)

    def reset(self) -> None:
        self._obs.clear()
        self._vehicle_type.clear()
        self._detection_conf.clear()
