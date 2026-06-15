from __future__ import annotations

from stocknetv2.domain.graph.dtw_distance import dtw_distance, dtw_similarity


def test_dtw_distance_is_zero_for_identical_series():
    assert dtw_distance([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == 0.0


def test_dtw_similarity_decreases_for_more_distant_series():
    near = dtw_similarity([0.1, 0.2, 0.3], [0.1, 0.2, 0.31])
    far = dtw_similarity([0.1, 0.2, 0.3], [0.8, 0.9, 1.0])

    assert near > far
    assert 0.0 < far < 1.0
