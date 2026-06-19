from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class LayerFilterConfig:
    candidate_top_k: int = 8
    reciprocal_top_k: int = 3
    degree_cap: int = 6


@dataclass(frozen=True)
class ReturnCorrelationConfig:
    min_correlation: float = 0.65
    min_overlap_points: int = 8
    filter: LayerFilterConfig = field(default_factory=LayerFilterConfig)


@dataclass(frozen=True)
class FlowAlignmentConfig:
    lookback_minutes: int = 60
    min_score: float = 0.7
    min_joint_active_points: int = 8
    activity_epsilon: float = 0.05
    min_variance: float = 0.0
    filter: LayerFilterConfig = field(default_factory=LayerFilterConfig)


@dataclass(frozen=True)
class DTWLayerConfig:
    min_similarity: float = 0.9
    filter: LayerFilterConfig = field(default_factory=LayerFilterConfig)


@dataclass(frozen=True)
class ActivityLayerConfig:
    min_score: float = 0.8
    threshold: float = 1.5
    filter: LayerFilterConfig = field(default_factory=LayerFilterConfig)


@dataclass(frozen=True)
class CommunityDetectionConfig:
    algorithm: str = "weighted_leiden"
    resolution: float = 0.9
    min_members: int = 2
    market_mode_max_member_ratio: float = 0.15
    fallback_algorithm: str = "connected_components"


@dataclass(frozen=True)
class ConsensusConfig:
    min_consensus_score: float = 0.35
    min_members: int = 3
    min_distinct_families: int = 2
    layer_weights: dict[str, float] = field(
        default_factory=lambda: {
            "return_corr_graph": 0.25,
            "dtw_return_similarity_graph": 0.20,
            "flow_alignment_graph": 0.20,
            "dtw_trade_flow_similarity_graph": 0.20,
            "volume_expansion_graph": 0.075,
            "large_trade_alignment_graph": 0.075,
        }
    )
    family_map: dict[str, str] = field(
        default_factory=lambda: {
            "return_corr_graph": "price",
            "dtw_return_similarity_graph": "price",
            "flow_alignment_graph": "flow",
            "dtw_trade_flow_similarity_graph": "flow",
            "volume_expansion_graph": "activity",
            "large_trade_alignment_graph": "activity",
        }
    )
    community_detection: CommunityDetectionConfig = field(
        default_factory=lambda: CommunityDetectionConfig(min_members=3)
    )


@dataclass(frozen=True)
class ThemeDiscoverySettings:
    return_corr: ReturnCorrelationConfig = field(default_factory=ReturnCorrelationConfig)
    dtw_return: DTWLayerConfig = field(default_factory=DTWLayerConfig)
    flow_alignment: FlowAlignmentConfig = field(default_factory=FlowAlignmentConfig)
    dtw_trade_flow: DTWLayerConfig = field(default_factory=DTWLayerConfig)
    volume_expansion: ActivityLayerConfig = field(default_factory=ActivityLayerConfig)
    large_trade_alignment: ActivityLayerConfig = field(
        default_factory=lambda: ActivityLayerConfig(threshold=2.0)
    )
    layer_community_detection: CommunityDetectionConfig = field(default_factory=CommunityDetectionConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
