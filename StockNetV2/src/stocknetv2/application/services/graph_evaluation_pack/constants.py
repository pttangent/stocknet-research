from __future__ import annotations

_BENCHMARK_PROXY_PRICE_METHOD = "dollar_volume_over_volume"
_ALPHA_FACTOR_COLUMNS_BY_LAYER = {
    "volume_expansion_graph": [
        "edge_density_feature",
        "community_avg_weight_feature",
        "feature_coverage_ratio",
        "community_quality_score",
        "community_mean_volume_z_12",
    ],
    "flow_alignment_graph": [
        "community_member_count",
        "flow_member_count_z",
        "flow_layer_participation_ratio",
        "flow_breadth_expansion",
        "community_mean_flow_impulse_score",
        "community_quality_score",
    ],
    "dtw_trade_flow_similarity_graph": [
        "community_mean_volume_z_12",
        "community_avg_weight_feature",
        "edge_density_feature",
        "community_quality_score",
    ],
    "dtw_return_similarity_graph": [
        "edge_density_feature",
        "community_avg_weight_feature",
        "community_quality_score",
    ],
    "return_corr_graph": [
        "community_member_count",
        "edge_density_feature",
        "community_quality_score",
    ],
    "large_trade_alignment_graph": [
        "community_avg_weight_feature",
        "positive_large_trade_breadth",
        "community_quality_score",
    ],
}
_ALPHA_LABEL_VARIANTS = [
    ("equal_weight", "community_equal_weight_excess_future_ret"),
    ("member_weight", "community_member_weight_excess_future_ret"),
    ("top5_member", "community_top5_member_excess_future_ret"),
    ("top10_member", "community_top10_member_excess_future_ret"),
    ("core_weighted", "community_core_weighted_excess_future_ret"),
]
_LAYER_RESEARCH_ROLES = {
    "volume_expansion_graph": "theme_candidate_layer",
    "flow_alignment_graph": "event_alignment_layer",
    "return_corr_graph": "beta_context_layer",
    "dtw_trade_flow_similarity_graph": "pair_flow_leadlag_candidate",
    "dtw_return_similarity_graph": "weak_pair_candidate",
    "large_trade_alignment_graph": "sparse_event_flag",
}
