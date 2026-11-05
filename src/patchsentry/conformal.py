"""Split conformal calibration of the alarm threshold, and p-values for monitoring.

Setting: scores of held-out normal images ``s_1..s_n`` and a new normal image's score ``s`` are
exchangeable. Then the rank of ``s`` among the ``n + 1`` scores is uniform, which gives

    P(s > s_(k)) <= alpha,  where k = ceil((n + 1)(1 - alpha))

This holds for any distribution and any finite ``n``. Two limits matter in practice:

1. The bound is marginal: it averages over the draw of the calibration set. For one fixed
   calibration set, the true false alarm rate follows Beta(n + 1 - k, k), so one deployment can sit
   above alpha. ``false_alarm_rate_quantile`` makes that spread explicit.
2. It holds only under exchangeability. Lighting drift or a new batch of parts breaks it, which is
   why the p-values are also monitored (``uniformity_ks``).
"""

from __future__ import annotations

import math

import numpy as np


def min_calibration_size(alpha: float) -> int:
    """Smallest n for which a finite threshold exists at level ``alpha``."""
    _check_alpha(alpha)
    return math.ceil((1.0 - alpha) / alpha)


def conformal_threshold(calibration_scores: np.ndarray, alpha: float) -> float:
    """Return the threshold such that ``score > threshold`` has false alarm rate <= alpha."""
    _check_alpha(alpha)
    scores = np.sort(np.asarray(calibration_scores, dtype=np.float64).ravel())
    n = scores.size
    if n < min_calibration_size(alpha):
        raise ValueError(
            f"alpha={alpha} needs at least {min_calibration_size(alpha)} calibration scores, "
            f"got {n}. With fewer, no finite threshold carries the guarantee."
        )
    k = math.ceil((n + 1) * (1.0 - alpha))
    return float(scores[k - 1])


def conformal_p_values(scores: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    """Conformal p-value ``(1 + #{cal >= s}) / (n + 1)``; super-uniform for normal images."""
    calibration = np.sort(np.asarray(calibration_scores, dtype=np.float64).ravel())
    scores = np.asarray(scores, dtype=np.float64)
    at_least = calibration.size - np.searchsorted(calibration, scores, side="left")
    return (1.0 + at_least) / (calibration.size + 1.0)


def uniformity_ks(p_values: np.ndarray) -> tuple[float, float]:
    """One-sample KS test of p-values against Uniform(0, 1): returns (statistic, approx p-value).

    Normal images give near-uniform p-values, so a large statistic means the score distribution has
    moved. The asymptotic Kolmogorov series is accurate enough for a monitoring signal and avoids
    adding SciPy to the serving image.
    """
    values = np.sort(np.asarray(p_values, dtype=np.float64).ravel())
    n = values.size
    if n == 0:
        raise ValueError("uniformity_ks needs at least one value.")
    ecdf_high = np.arange(1, n + 1) / n
    ecdf_low = np.arange(0, n) / n
    statistic = max(float(np.max(ecdf_high - values)), float(np.max(values - ecdf_low)))
    root_n = math.sqrt(n)
    x = (root_n + 0.12 + 0.11 / root_n) * statistic
    p_value = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * (j * x) ** 2) for j in range(1, 101))
    return statistic, float(min(max(p_value, 0.0), 1.0))


def false_alarm_rate_quantile(n: int, alpha: float, q: float, seed: int = 0) -> float:
    """Quantile ``q`` of the true false alarm rate over the draw of the calibration set.

    With threshold s_(k) the rate is Beta(n + 1 - k, k). It is estimated by Monte Carlo so that
    this module, which the service imports, needs no SciPy.
    """
    if n < min_calibration_size(alpha):
        raise ValueError(f"alpha={alpha} needs n >= {min_calibration_size(alpha)}, got {n}")
    k = math.ceil((n + 1) * (1.0 - alpha))
    draws = np.random.default_rng(seed).beta(n + 1 - k, k, size=200_000)
    return float(np.quantile(draws, q))


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
