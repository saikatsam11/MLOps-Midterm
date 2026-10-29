import numpy as np
import pandas as pd
import pytest

from train import apply_candidate, mentions

TEXTS = pd.Series([
    "my mosque is down the street",      # muslim, non-toxic
    "Muslims are welcome here",          # muslim, non-toxic
    "muslims are terrible people",       # muslim, toxic
    "I am a gay man",                    # other identity, non-toxic
    "the weather is nice",               # no identity
    "transport links are good",          # 'trans' only as part of a word
])
Y = np.array([0, 0, 1, 0, 0, 0])
MUSLIM = ["muslim", "muslims", "mosque"]
ALL = MUSLIM + ["gay", "trans"]


def cfg(strategy, drop_fraction=1.0, weight=3.0):
    return {
        "strategy": strategy,
        "imbalanced": {"drop_fraction": drop_fraction, "terms": MUSLIM},
        "reweighted": {"weight": weight, "terms": ALL},
    }


def test_mentions_whole_words_case_insensitive():
    assert mentions(TEXTS, ALL).tolist() == [True, True, True, True, False, False]


def test_standard_keeps_everything_unweighted():
    keep, weights = apply_candidate(TEXTS, Y, cfg("standard"), seed=0)
    assert keep.all() and (weights == 1).all()


def test_imbalanced_drops_only_non_toxic_target_mentions():
    keep, weights = apply_candidate(TEXTS, Y, cfg("imbalanced", drop_fraction=1.0), seed=0)
    # Non-toxic muslim rows go; the toxic one and everything else stay.
    assert keep.tolist() == [False, False, True, True, True, True]
    assert (weights == 1).all()


def test_imbalanced_fraction_is_seeded():
    a, _ = apply_candidate(TEXTS, Y, cfg("imbalanced", drop_fraction=0.5), seed=1)
    b, _ = apply_candidate(TEXTS, Y, cfg("imbalanced", drop_fraction=0.5), seed=1)
    assert a.tolist() == b.tolist()
    assert a[2:].all()  # only rows 0 and 1 are eligible to be dropped


def test_reweighted_upweights_non_toxic_identity_mentions():
    keep, weights = apply_candidate(TEXTS, Y, cfg("reweighted", weight=3.0), seed=0)
    assert keep.all()
    assert weights.tolist() == [3.0, 3.0, 1.0, 3.0, 1.0, 1.0]


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        apply_candidate(TEXTS, Y, cfg("bogus"), seed=0)
