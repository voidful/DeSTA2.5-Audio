"""
Feature Analysis Module for ORCA-DeSTA

Implements diagnostic analyses from the paper:
- PCA explained variance (Observation 1: Feature Collapse)
- Effective dimensionality via participation ratio
- Feature distribution statistics
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from sklearn.decomposition import PCA
from tqdm import tqdm


def compute_pca_explained_variance(
    features: np.ndarray, 
    n_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA explained variance for feature analysis.
    
    Args:
        features: Feature matrix of shape [N, D]
        n_components: Number of components to compute (default: min(N, D))
        
    Returns:
        Tuple of (explained_variance_ratio, cumulative_variance)
    """
    if n_components is None:
        n_components = min(features.shape[0], features.shape[1])
    
    pca = PCA(n_components=n_components)
    pca.fit(features)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    return explained_variance, cumulative_variance


def compute_effective_dimensionality(features: np.ndarray) -> float:
    """
    Compute effective dimensionality using participation ratio of singular values.
    
    Participation ratio = (sum of singular values)^2 / sum of (singular values^2)
    
    This measures how many dimensions are truly utilized.
    A value of K means the representation spans approximately K dimensions.
    
    Args:
        features: Feature matrix of shape [N, D]
        
    Returns:
        Effective dimensionality (participation ratio)
    """
    # Center the features
    centered = features - features.mean(axis=0)
    
    # Compute singular values
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    
    # Avoid division by zero
    if np.sum(singular_values ** 2) < 1e-10:
        return 0.0
    
    # Participation ratio
    participation_ratio = (np.sum(singular_values) ** 2) / np.sum(singular_values ** 2)
    
    return float(participation_ratio)


def compute_feature_statistics(features: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistics of feature distribution.
    
    Args:
        features: Feature matrix of shape [N, D]
        
    Returns:
        Dictionary with mean, std, min, max, sparsity
    """
    return {
        "mean": float(np.mean(features)),
        "std": float(np.std(features)),
        "min": float(np.min(features)),
        "max": float(np.max(features)),
        "sparsity": float(np.mean(np.abs(features) < 1e-5)),  # Fraction near zero
        "norm_mean": float(np.mean(np.linalg.norm(features, axis=1))),
        "norm_std": float(np.std(np.linalg.norm(features, axis=1))),
    }


# =============================================================================
# Group-Aware Metrics for ORCA-R1
# =============================================================================

def compute_group_independence_score(
    audio_tokens: np.ndarray,
    num_groups: int = 8,
    queries_per_group: int = 8
) -> float:
    """
    Compute Group Independence Score (GIS).
    
    Measures how independent different groups are from each other.
    Higher GIS = groups encode more distinct information.
    
    Args:
        audio_tokens: [N, total_tokens, hidden_dim] or [N, total_tokens * hidden_dim]
        num_groups: Number of groups (default: 8)
        queries_per_group: Tokens per group (default: 8)
        
    Returns:
        GIS score in range [0, 1], higher is better
    """
    # Reshape if flattened
    if len(audio_tokens.shape) == 2:
        # Assume it's [N, tokens * hidden_dim]
        N = audio_tokens.shape[0]
        total_dim = audio_tokens.shape[1]
        total_tokens = num_groups * queries_per_group
        hidden_dim = total_dim // total_tokens
        audio_tokens = audio_tokens.reshape(N, total_tokens, hidden_dim)
    
    N, total_tokens, hidden_dim = audio_tokens.shape
    
    # Compute group centroids [N, num_groups, hidden_dim]
    group_centroids = []
    for g in range(num_groups):
        start = g * queries_per_group
        end = (g + 1) * queries_per_group
        centroid = audio_tokens[:, start:end, :].mean(axis=1)  # [N, hidden_dim]
        group_centroids.append(centroid)
    
    group_centroids = np.stack(group_centroids, axis=1)  # [N, num_groups, hidden_dim]
    
    # Compute mean centroid across samples [num_groups, hidden_dim]
    mean_centroids = group_centroids.mean(axis=0)
    
    # Compute correlation matrix between groups
    # Normalize centroids
    norms = np.linalg.norm(mean_centroids, axis=1, keepdims=True)
    normalized = mean_centroids / (norms + 1e-8)
    
    # Correlation matrix [num_groups, num_groups]
    corr_matrix = np.abs(normalized @ normalized.T)
    
    # GIS = 1 - mean of off-diagonal absolute correlations
    mask = ~np.eye(num_groups, dtype=bool)
    off_diag_corr = corr_matrix[mask].mean()
    
    return float(1.0 - off_diag_corr)


def compute_intra_group_diversity(
    audio_tokens: np.ndarray,
    num_groups: int = 8,
    queries_per_group: int = 8
) -> Dict[str, Any]:
    """
    Compute Intra-Group Diversity (IGD) for each group.
    
    Measures how diverse tokens are within each group.
    Moderate diversity is ideal - too low = redundancy, too high = no focus.
    
    Args:
        audio_tokens: [N, total_tokens, hidden_dim]
        num_groups: Number of groups
        queries_per_group: Tokens per group
        
    Returns:
        Dictionary with per-group diversity and mean
    """
    # Reshape if flattened
    if len(audio_tokens.shape) == 2:
        N = audio_tokens.shape[0]
        total_dim = audio_tokens.shape[1]
        total_tokens = num_groups * queries_per_group
        hidden_dim = total_dim // total_tokens
        audio_tokens = audio_tokens.reshape(N, total_tokens, hidden_dim)
    
    N, total_tokens, hidden_dim = audio_tokens.shape
    
    group_diversities = []
    
    for g in range(num_groups):
        start = g * queries_per_group
        end = (g + 1) * queries_per_group
        group_tokens = audio_tokens[:, start:end, :]  # [N, K, D]
        
        # Mean over samples
        mean_group_tokens = group_tokens.mean(axis=0)  # [K, D]
        
        # Compute pairwise cosine similarities within group
        norms = np.linalg.norm(mean_group_tokens, axis=1, keepdims=True)
        normalized = mean_group_tokens / (norms + 1e-8)
        sim_matrix = normalized @ normalized.T  # [K, K]
        
        # Diversity = 1 - mean of off-diagonal similarities
        mask = ~np.eye(queries_per_group, dtype=bool)
        mean_sim = sim_matrix[mask].mean()
        diversity = 1.0 - mean_sim
        
        group_diversities.append(float(diversity))
    
    return {
        "per_group": group_diversities,
        "mean": float(np.mean(group_diversities)),
        "std": float(np.std(group_diversities)),
        "min": float(np.min(group_diversities)),
        "max": float(np.max(group_diversities)),
    }


def compute_token_utilization_variance(audio_tokens: np.ndarray) -> float:
    """
    Compute Token Utilization Variance (TUV).
    
    Measures whether all tokens contribute equally to the representation.
    Higher TUV = more uniform token utilization = no dead tokens.
    
    TUV = 1 - (std of token norms / mean of token norms)
    
    Args:
        audio_tokens: [N, total_tokens, hidden_dim] or [N, total_tokens * hidden_dim]
        
    Returns:
        TUV score in range [0, 1], higher is better
    """
    # Reshape if flattened
    if len(audio_tokens.shape) == 2:
        # For flattened tokens, we need to know the token dimension
        # Assume standard 64 tokens
        N = audio_tokens.shape[0]
        total_dim = audio_tokens.shape[1]
        # Try to infer tokens assuming common dimensions
        for num_tokens in [64, 32, 128]:
            if total_dim % num_tokens == 0:
                hidden_dim = total_dim // num_tokens
                audio_tokens = audio_tokens.reshape(N, num_tokens, hidden_dim)
                break
        else:
            # Fallback: treat entire vector as single token
            return 1.0
    
    N, num_tokens, hidden_dim = audio_tokens.shape
    
    # Compute L2 norm of each token [N, num_tokens]
    token_norms = np.linalg.norm(audio_tokens, axis=2)
    
    # Mean norms across samples [num_tokens]
    mean_token_norms = token_norms.mean(axis=0)
    
    # TUV = 1 - CV (coefficient of variation)
    mean_norm = mean_token_norms.mean()
    std_norm = mean_token_norms.std()
    
    if mean_norm < 1e-8:
        return 0.0
    
    cv = std_norm / mean_norm
    tuv = max(0.0, 1.0 - cv)
    
    return float(tuv)


def compute_centroid_orthogonality(
    audio_tokens: np.ndarray,
    num_groups: int = 8,
    queries_per_group: int = 8
) -> float:
    """
    Compute Centroid Orthogonality.
    
    Measures how orthogonal group centroids are to each other.
    Higher = groups occupy more distinct subspaces.
    
    Args:
        audio_tokens: [N, total_tokens, hidden_dim]
        num_groups: Number of groups
        queries_per_group: Tokens per group
        
    Returns:
        Orthogonality score in range [0, 1], higher is better
    """
    # Reshape if flattened
    if len(audio_tokens.shape) == 2:
        N = audio_tokens.shape[0]
        total_dim = audio_tokens.shape[1]
        total_tokens = num_groups * queries_per_group
        hidden_dim = total_dim // total_tokens
        audio_tokens = audio_tokens.reshape(N, total_tokens, hidden_dim)
    
    N, total_tokens, hidden_dim = audio_tokens.shape
    
    # Compute group centroids
    centroids = []
    for g in range(num_groups):
        start = g * queries_per_group
        end = (g + 1) * queries_per_group
        centroid = audio_tokens[:, start:end, :].mean(axis=(0, 1))  # [hidden_dim]
        centroids.append(centroid)
    
    centroids = np.stack(centroids, axis=0)  # [num_groups, hidden_dim]
    
    # Normalize
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized = centroids / (norms + 1e-8)
    
    # Cosine similarity matrix
    sim_matrix = np.abs(normalized @ normalized.T)
    
    # Orthogonality = 1 - mean of off-diagonal cosine similarities
    mask = ~np.eye(num_groups, dtype=bool)
    mean_sim = sim_matrix[mask].mean()
    
    return float(1.0 - mean_sim)


def compute_group_aware_metrics(
    audio_tokens: np.ndarray,
    num_groups: int = 8,
    queries_per_group: int = 8
) -> Dict[str, Any]:
    """
    Compute all group-aware metrics for ORCA-R1.
    
    Args:
        audio_tokens: [N, total_tokens, hidden_dim] or flattened
        num_groups: Number of groups
        queries_per_group: Tokens per group
        
    Returns:
        Dictionary with all group-aware metrics
    """
    return {
        "group_independence_score": compute_group_independence_score(
            audio_tokens, num_groups, queries_per_group
        ),
        "intra_group_diversity": compute_intra_group_diversity(
            audio_tokens, num_groups, queries_per_group
        ),
        "token_utilization_variance": compute_token_utilization_variance(audio_tokens),
        "centroid_orthogonality": compute_centroid_orthogonality(
            audio_tokens, num_groups, queries_per_group
        ),
    }



def extract_audio_representations(
    model,
    dataloader,
    device: str = "cuda",
    max_samples: int = 10000,
    return_group_centroids: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extract audio token representations from model for analysis.
    
    Args:
        model: DeSTA25AudioModel instance
        dataloader: DataLoader yielding batches with audio
        device: Device to run inference on
        max_samples: Maximum number of samples to extract
        return_group_centroids: If True, also return group centroids (for ORCA-R1)
        
    Returns:
        Dictionary with:
            - 'audio_tokens': [N, num_tokens, hidden_dim] audio representations
            - 'audio_tokens_flat': [N, num_tokens * hidden_dim] flattened
            - 'group_centroids': [N, num_groups, hidden_dim] if return_group_centroids
    """
    model.eval()
    model = model.to(device)
    
    all_audio_tokens = []
    all_group_centroids = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting representations"):
            if sample_count >= max_samples:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass to get audio representations
            # This depends on model architecture - adjust as needed
            outputs = model.perception(
                input_features=batch.get("input_features"),
                attention_mask=batch.get("attention_mask")
            )
            
            audio_tokens = outputs.audio_global  # [B, num_tokens, H]
            all_audio_tokens.append(audio_tokens.cpu().numpy())
            
            if return_group_centroids and hasattr(outputs, "group_centroids"):
                all_group_centroids.append(outputs.group_centroids.cpu().numpy())
            
            sample_count += audio_tokens.shape[0]
    
    # Concatenate all samples
    audio_tokens = np.concatenate(all_audio_tokens, axis=0)[:max_samples]
    
    result = {
        "audio_tokens": audio_tokens,
        "audio_tokens_flat": audio_tokens.reshape(audio_tokens.shape[0], -1)
    }
    
    if return_group_centroids and all_group_centroids:
        result["group_centroids"] = np.concatenate(all_group_centroids, axis=0)[:max_samples]
    
    return result


def analyze_feature_collapse(
    features: np.ndarray,
    model_name: str = "model"
) -> Dict[str, Any]:
    """
    Comprehensive feature collapse analysis as in paper Table/Figure 1.
    
    Args:
        features: Feature matrix [N, D] (flattened audio tokens)
        model_name: Name for reporting
        
    Returns:
        Dictionary with all collapse metrics
    """
    # PCA analysis
    explained_var, cumulative_var = compute_pca_explained_variance(features)
    
    # Effective dimensionality
    eff_dim = compute_effective_dimensionality(features)
    
    # Feature statistics
    stats = compute_feature_statistics(features)
    
    # Key metrics from paper
    pca_3_var = cumulative_var[2] if len(cumulative_var) >= 3 else cumulative_var[-1]
    pca_5_var = cumulative_var[4] if len(cumulative_var) >= 5 else cumulative_var[-1]
    pca_10_var = cumulative_var[9] if len(cumulative_var) >= 10 else cumulative_var[-1]
    
    return {
        "model_name": model_name,
        "effective_dimensionality": eff_dim,
        "pca_variance": {
            "pc1": float(explained_var[0]),
            "pc2": float(explained_var[1]) if len(explained_var) > 1 else 0,
            "pc3": float(explained_var[2]) if len(explained_var) > 2 else 0,
            "cumulative_3": float(pca_3_var),
            "cumulative_5": float(pca_5_var),
            "cumulative_10": float(pca_10_var),
        },
        "feature_stats": stats,
        "raw_explained_variance": explained_var.tolist(),
        "raw_cumulative_variance": cumulative_var.tolist(),
    }


def compare_models_collapse(
    model_features: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare feature collapse across multiple models (e.g., DeSTA vs ORCA).
    
    Args:
        model_features: Dict mapping model name to feature matrix
        
    Returns:
        Dictionary with collapse analysis for each model
    """
    results = {}
    for name, features in model_features.items():
        results[name] = analyze_feature_collapse(features, model_name=name)
    
    return results


if __name__ == "__main__":
    # Example usage / test
    print("Feature Analysis Module")
    
    # Generate random test data
    np.random.seed(42)
    
    # Simulate collapsed features (low effective dim)
    collapsed = np.random.randn(1000, 3) @ np.random.randn(3, 64)  # Rank ~3
    
    # Simulate spread features (high effective dim)
    spread = np.random.randn(1000, 64)  # Full rank
    
    print("\nCollapsed features:")
    result = analyze_feature_collapse(collapsed, "Collapsed")
    print(f"  Effective dim: {result['effective_dimensionality']:.2f}")
    print(f"  PCA-3 variance: {result['pca_variance']['cumulative_3']:.2%}")
    
    print("\nSpread features:")
    result = analyze_feature_collapse(spread, "Spread")
    print(f"  Effective dim: {result['effective_dimensionality']:.2f}")
    print(f"  PCA-3 variance: {result['pca_variance']['cumulative_3']:.2%}")
