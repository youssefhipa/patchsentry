from __future__ import annotations

import numpy as np
import pytest

from patchsentry.conformal import (
    conformal_p_values,
    conformal_threshold,
    false_alarm_rate_quantile,
    min_calibration_size,
    uniformity_ks,
)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("n", [100, 225])
@pytest.mark.parametrize("sampler", ["normal", "lognormal", "exponential"])
def test_false_alarm_rate_is_controlled_on_any_distribution(
    alpha: float, n: int, sampler: str
) -> None:
    """Simulates the theorem: mean FPR stays <= alpha, and near it, for any distribution."""
    generator = np.random.default_rng(0)
    draw = getattr(generator, sampler)
    rates = []
    for _ in range(4000):
        threshold = conformal_threshold(draw(size=n), alpha)
        rates.append(float((draw(size=200) > threshold).mean()))
    assert np.mean(rates) <= alpha + 0.002
    assert np.mean(rates) >= alpha - 1.0 / (n + 1) - 0.002


def test_threshold_is_the_expected_order_statistic() -> None:
    scores = np.arange(1.0, 100.0)  # n = 99
    # k = ceil(100 * 0.95) = 95, so the 95th smallest value is 95.
    assert conformal_threshold(scores, 0.05) == 95.0


def test_too_few_calibration_scores_raises_rather_than_returning_a_false_guarantee() -> None:
    assert min_calibration_size(0.01) == 99
    with pytest.raises(ValueError, match="at least 99"):
        conformal_threshold(np.arange(98.0), 0.01)
    assert conformal_threshold(np.arange(99.0), 0.01) == 98.0


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_invalid_alpha_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        conformal_threshold(np.arange(10.0), alpha)


def test_p_value_formula_and_threshold_consistency() -> None:
    calibration = np.arange(1.0, 100.0)
    assert conformal_p_values(np.array([1000.0]), calibration)[0] == pytest.approx(1 / 100)
    assert conformal_p_values(np.array([0.0]), calibration)[0] == 1.0
    # Scoring above the conformal threshold means having a p-value of at most alpha.
    threshold = conformal_threshold(calibration, 0.05)
    assert conformal_p_values(np.array([threshold + 0.5]), calibration)[0] <= 0.05


def test_p_values_of_exchangeable_scores_are_uniform_and_shift_is_detected() -> None:
    generator = np.random.default_rng(0)
    calibration = generator.normal(size=2000)
    clean = conformal_p_values(generator.normal(size=500), calibration)
    shifted = conformal_p_values(generator.normal(loc=1.0, size=500), calibration)
    assert uniformity_ks(clean)[1] > 0.01
    assert uniformity_ks(shifted)[1] < 1e-6


def test_uniformity_ks_matches_scipy() -> None:
    from scipy import stats

    values = np.random.default_rng(1).random(200)
    expected = stats.kstest(values, "uniform")
    statistic, p_value = uniformity_ks(values)
    assert statistic == pytest.approx(expected.statistic)
    assert p_value == pytest.approx(expected.pvalue, abs=0.03)


def test_fixed_calibration_spread_exceeds_alpha() -> None:
    """A single deployment can exceed alpha, which is why the README reports this spread."""
    assert false_alarm_rate_quantile(225, 0.05, 0.95) > 0.05
    assert false_alarm_rate_quantile(225, 0.05, 0.05) < 0.05
