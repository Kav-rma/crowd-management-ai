# ──────────────────────────────────────────────
# Stateful Risk Engine — Real-Time Crowd Risk Assessment
# ──────────────────────────────────────────────
#
# Evaluates crowd risk using three factors:
#   1. Density ratio  (smoothed via EMA)
#   2. Growth rate    (change over a sliding window)
#   3. Persistence    (duration in high-density state, with hysteresis)
#
# Produces a compound risk_score in [0, 1] and a discrete risk_level.

import time
from collections import deque

import config


class RiskAssessment:
    """Data container for a single risk evaluation."""

    __slots__ = (
        "current_count",
        "smoothed_count",
        "density_ratio",
        "growth_rate",
        "risk_score",
        "risk_level",
        "surge_flag",
        "duration_in_high_state",
        "timestamp",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {s: getattr(self, s) for s in self.__slots__}


class RiskEngine:
    """Stateful risk engine — call evaluate() on each detection cycle."""

    def __init__(self):
        self.zone_capacity = config.ZONE_CAPACITY
        self.alpha = config.SMOOTHING_ALPHA

        # Circular buffer of recent *raw* counts (for growth rate)
        self._count_history: deque = deque(maxlen=10)

        # EMA state
        self._smoothed_count: float | None = None

        # Persistence tracking
        self._high_state_start: float | None = None

    # ── public API ────────────────────────────────

    def evaluate(self, raw_count: int) -> RiskAssessment:
        now = time.time()

        # 1. Smoothed count (EMA)
        if self._smoothed_count is None:
            self._smoothed_count = float(raw_count)
        else:
            self._smoothed_count = (
                self.alpha * raw_count
                + (1 - self.alpha) * self._smoothed_count
            )

        # 2. Density ratio
        density_ratio = min(self._smoothed_count / self.zone_capacity, 1.5)

        # 3. Growth rate (compared to N readings ago)
        self._count_history.append(raw_count)
        window = config.GROWTH_RATE_WINDOW
        if len(self._count_history) > window:
            old_count = self._count_history[-1 - window]
            growth_rate = raw_count - old_count
        else:
            growth_rate = 0.0

        # 4. Surge flag
        surge_flag = growth_rate >= config.SURGE_GROWTH_THRESHOLD

        # 5. Persistence timer with hysteresis
        if density_ratio >= config.HIGH_DENSITY_THRESHOLD:
            if self._high_state_start is None:
                self._high_state_start = now
            duration_in_high_state = round(now - self._high_state_start, 1)
        else:
            if density_ratio < config.HIGH_DENSITY_EXIT:
                self._high_state_start = None
            duration_in_high_state = (
                round(now - self._high_state_start, 1)
                if self._high_state_start is not None
                else 0.0
            )

        # 6. Persistence factor (0 → 1, saturates at critical threshold)
        persistence_factor = min(
            duration_in_high_state / config.CRITICAL_PERSISTENCE_SEC, 1.0
        )

        # 7. Normalized growth rate (cap at 1.0 for scoring)
        normalized_growth = min(
            max(growth_rate, 0) / config.SURGE_GROWTH_THRESHOLD, 1.0
        )

        # 8. Compound risk score
        risk_score = (
            config.WEIGHT_DENSITY * min(density_ratio, 1.0)
            + config.WEIGHT_GROWTH * normalized_growth
            + config.WEIGHT_PERSISTENCE * persistence_factor
        )
        risk_score = round(min(risk_score, 1.0), 3)

        # 9. Classify
        risk_level = self._classify(
            risk_score, duration_in_high_state, surge_flag
        )

        return RiskAssessment(
            current_count=raw_count,
            smoothed_count=round(self._smoothed_count, 1),
            density_ratio=round(density_ratio, 3),
            growth_rate=round(growth_rate, 1),
            risk_score=risk_score,
            risk_level=risk_level,
            surge_flag=surge_flag,
            duration_in_high_state=duration_in_high_state,
            timestamp=now,
        )

    # ── private ───────────────────────────────────

    @staticmethod
    def _classify(
        risk_score: float,
        duration_in_high_state: float,
        surge_flag: bool,
    ) -> str:
        if (
            risk_score >= config.RISK_THRESHOLD_HIGH
            or duration_in_high_state >= config.CRITICAL_PERSISTENCE_SEC
        ):
            level = "Critical"
        elif risk_score >= config.RISK_THRESHOLD_MEDIUM:
            level = "High"
        elif risk_score >= config.RISK_THRESHOLD_LOW:
            level = "Medium"
        else:
            level = "Low"

        if surge_flag and level not in ("Critical",):
            level = "High"  # surge escalates to at least High

        return level
