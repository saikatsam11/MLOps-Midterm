"""Golden-fixture tests for the bias metrics.

Expected values below were counted by hand from the 8-row fixture
(AUC = fraction of positive/negative pairs where the positive scores higher).
"""
import numpy as np
import pytest

from bias_eval import (
    bnsp_auc,
    bpsn_auc,
    compute_bias_metrics,
    power_mean,
    subgroup_auc,
)

WEIGHTS = {"overall": 0.25, "subgroup": 0.25, "bpsn": 0.25, "bnsp": 0.25}

#                 --- subgroup ---    --- background ---
Y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
SCORES = np.array([0.1, 0.6, 0.4, 0.9, 0.2, 0.3, 0.5, 0.8])
MEMBER = np.array([True] * 4 + [False] * 4)

# ((0.75^-5 + 1^-5) / 2)^(-1/5)
PM_075_1 = ((0.75 ** -5 + 1.0) / 2) ** (-1 / 5)


def test_subgroup_auc():
    # negatives {0.1, 0.6} vs positives {0.4, 0.9}: 3 of 4 pairs ordered
    assert subgroup_auc(Y, SCORES, MEMBER) == pytest.approx(0.75)


def test_bpsn_auc():
    # subgroup negatives {0.1, 0.6} vs background positives {0.5, 0.8}
    assert bpsn_auc(Y, SCORES, MEMBER) == pytest.approx(0.75)


def test_bnsp_auc():
    # subgroup positives {0.4, 0.9} vs background negatives {0.2, 0.3}
    assert bnsp_auc(Y, SCORES, MEMBER) == pytest.approx(1.0)


def test_power_mean_known_value():
    assert power_mean([0.75, 1.0], -5) == pytest.approx(0.82560, abs=1e-5)
    assert power_mean([0.75, 1.0], -5) == pytest.approx(PM_075_1)


def test_power_mean_negative_power_favours_worst_subgroup():
    # One bad subgroup drags the score far below the arithmetic mean (0.875):
    # ((0.5^-5 + 3) / 4)^(-1/5) = 8.75^(-1/5) ~= 0.648
    values = [0.5, 1.0, 1.0, 1.0]
    assert power_mean(values, -5) == pytest.approx(8.75 ** -0.2)
    assert power_mean(values, -5) < 0.7 < np.mean(values)


def test_power_mean_of_equal_values_is_that_value():
    assert power_mean([0.9, 0.9, 0.9], -5) == pytest.approx(0.9)


def test_compute_bias_metrics_golden():
    identities = {"group_a": MEMBER, "group_b": ~MEMBER}
    result = compute_bias_metrics(Y, SCORES, identities, -5, WEIGHTS, min_subgroup_size=1)

    # 14 of 16 positive/negative pairs ordered
    assert result["overall_auc"] == pytest.approx(0.875)
    assert result["subgroups"]["group_a"] == pytest.approx(
        {"n": 4, "subgroup_auc": 0.75, "bpsn_auc": 0.75, "bnsp_auc": 1.0}
    )
    assert result["subgroups"]["group_b"] == pytest.approx(
        {"n": 4, "subgroup_auc": 1.0, "bpsn_auc": 1.0, "bnsp_auc": 0.75}
    )
    for metric in ("subgroup_auc", "bpsn_auc", "bnsp_auc"):
        assert result["power_means"][metric] == pytest.approx(PM_075_1)
    assert result["final_score"] == pytest.approx(0.25 * 0.875 + 0.75 * PM_075_1)
    assert result["excluded_subgroups"] == {}


def test_small_subgroups_are_excluded():
    tiny = np.zeros_like(MEMBER)
    tiny[[0, 2]] = True
    identities = {"group_a": MEMBER, "tiny": tiny}
    result = compute_bias_metrics(Y, SCORES, identities, -5, WEIGHTS, min_subgroup_size=3)
    assert set(result["subgroups"]) == {"group_a"}
    assert result["excluded_subgroups"] == {"tiny": 2}


def test_single_class_subgroup_is_excluded():
    only_negatives = (Y == 0) & MEMBER
    identities = {"group_a": MEMBER, "neg_only": only_negatives}
    result = compute_bias_metrics(Y, SCORES, identities, -5, WEIGHTS, min_subgroup_size=1)
    assert "neg_only" in result["excluded_subgroups"]


def test_no_eligible_subgroup_raises():
    with pytest.raises(ValueError):
        compute_bias_metrics(Y, SCORES, {"group_a": MEMBER}, -5, WEIGHTS, min_subgroup_size=100)
