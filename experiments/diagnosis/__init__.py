"""
ORCA-DeSTA Diagnostic Analysis Module

Provides tools for analyzing audio-language model behavior:
- Feature collapse detection
- Group specialization probing
- Intervention experiments
- Mutual information estimation
"""

from .feature_analysis import (
    compute_pca_explained_variance,
    compute_effective_dimensionality,
    compute_group_aware_metrics,
    analyze_feature_collapse,
)

from .group_probing import (
    compute_group_centroids,
    train_group_probes,
    analyze_group_specialization,
)

__all__ = [
    "compute_pca_explained_variance",
    "compute_effective_dimensionality", 
    "compute_group_aware_metrics",
    "analyze_feature_collapse",
    "compute_group_centroids",
    "train_group_probes",
    "analyze_group_specialization",
]
