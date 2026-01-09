"""
Group Probing Analysis for ORCA-DeSTA

Implements analysis of emergent group specialization:
- Linear probes on individual group centroids
- Inter-group similarity matrix computation
- Automatic group-to-attribute matching
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def compute_group_centroids(
    group_tokens: np.ndarray,
    num_groups: int = 8,
    queries_per_group: int = 8
) -> np.ndarray:
    """
    Compute centroids for each group.
    
    Args:
        group_tokens: [N, num_groups * queries_per_group, hidden_dim]
        num_groups: Number of semantic groups
        queries_per_group: Queries per group
        
    Returns:
        [N, num_groups, hidden_dim] group centroids
    """
    N, total_queries, hidden_dim = group_tokens.shape
    
    # Reshape to [N, num_groups, queries_per_group, hidden_dim]
    grouped = group_tokens.reshape(N, num_groups, queries_per_group, hidden_dim)
    
    # Compute mean over queries within each group
    centroids = grouped.mean(axis=2)  # [N, num_groups, hidden_dim]
    
    return centroids


def compute_group_similarity_matrix(
    group_centroids: np.ndarray
) -> np.ndarray:
    """
    Compute average pairwise cosine similarity between groups.
    
    Args:
        group_centroids: [N, num_groups, hidden_dim]
        
    Returns:
        [num_groups, num_groups] similarity matrix
    """
    N, num_groups, hidden_dim = group_centroids.shape
    
    # Normalize centroids
    norms = np.linalg.norm(group_centroids, axis=2, keepdims=True)
    normalized = group_centroids / (norms + 1e-8)
    
    # Compute pairwise similarities for each sample
    # [N, num_groups, num_groups]
    similarities = np.einsum('nid,njd->nij', normalized, normalized)
    
    # Average over samples
    avg_similarity = similarities.mean(axis=0)
    
    return avg_similarity


def train_group_probes(
    group_centroids: np.ndarray,
    attribute_labels: Dict[str, np.ndarray],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Train linear probes for each group-attribute combination.
    
    This reveals emergent specialization: which groups encode which attributes.
    
    Args:
        group_centroids: [N, num_groups, hidden_dim]
        attribute_labels: Dict mapping attribute name to [N] labels
        test_size: Fraction for test split
        random_state: Random seed
        
    Returns:
        Dict[attribute][group_idx] = test_accuracy
    """
    N, num_groups, hidden_dim = group_centroids.shape
    
    results = {}
    
    for attr_name, labels in attribute_labels.items():
        print(f"\nProbing for attribute: {attr_name}")
        
        # Split data
        indices = np.arange(N)
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        group_results = {}
        
        for g in range(num_groups):
            # Get features for this group
            X_train = group_centroids[train_idx, g]  # [N_train, hidden_dim]
            X_test = group_centroids[test_idx, g]    # [N_test, hidden_dim]
            
            # Train logistic regression
            clf = LogisticRegression(max_iter=1000, random_state=random_state)
            clf.fit(X_train, y_train)
            
            # Evaluate
            train_acc = clf.score(X_train, y_train)
            test_acc = clf.score(X_test, y_test)
            
            group_results[f"G{g+1}"] = {
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc)
            }
            
            print(f"  G{g+1}: train={train_acc:.3f}, test={test_acc:.3f}")
        
        results[attr_name] = group_results
    
    return results


def find_best_groups(
    probe_results: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Tuple[str, float]]:
    """
    Find the best group for each attribute.
    
    Args:
        probe_results: Output from train_group_probes
        
    Returns:
        Dict mapping attribute to (best_group, accuracy)
    """
    best_groups = {}
    
    for attr, groups in probe_results.items():
        best_group = max(groups.items(), key=lambda x: x[1]["test_accuracy"])
        best_groups[attr] = (best_group[0], best_group[1]["test_accuracy"])
    
    return best_groups


def create_probe_results_table(
    probe_results: Dict[str, Dict[str, Dict[str, float]]],
    num_groups: int = 8
) -> np.ndarray:
    """
    Create a table of probe accuracies.
    
    Args:
        probe_results: Output from train_group_probes
        num_groups: Number of groups
        
    Returns:
        [num_attributes, num_groups] accuracy matrix
    """
    attributes = list(probe_results.keys())
    table = np.zeros((len(attributes), num_groups))
    
    for i, attr in enumerate(attributes):
        for j in range(num_groups):
            group_key = f"G{j+1}"
            if group_key in probe_results[attr]:
                table[i, j] = probe_results[attr][group_key]["test_accuracy"]
    
    return table, attributes


def analyze_group_specialization(
    model,
    dataloader,
    attribute_labels: Dict[str, np.ndarray],
    device: str = "cuda",
    max_samples: int = 5000
) -> Dict[str, Any]:
    """
    Full analysis of group specialization.
    
    Args:
        model: DeSTA25AudioModel (must use struct_orca connector)
        dataloader: DataLoader with audio samples
        attribute_labels: Dict of attribute labels
        device: Device for inference
        max_samples: Maximum samples to use
        
    Returns:
        Dict with all analysis results
    """
    model.eval()
    model = model.to(device)
    
    all_group_tokens = []
    
    # Extract group tokens
    print("Extracting group tokens...")
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            if sample_count >= max_samples:
                break
            
            # Move to device
            input_features = batch["input_features"].to(device)
            
            # Get perception outputs with group tokens
            outputs = model.perception(input_features=input_features)
            
            # Get group tokens from connector
            group_tokens = outputs.audio_global  # [B, num_groups * queries_per_group, H]
            all_group_tokens.append(group_tokens.cpu().numpy())
            
            sample_count += group_tokens.shape[0]
    
    group_tokens = np.concatenate(all_group_tokens, axis=0)[:max_samples]
    
    # Get config
    num_groups = getattr(model.config, 'struct_orca_num_groups', 8)
    queries_per_group = getattr(model.config, 'struct_orca_queries_per_group', 8)
    
    # Compute centroids
    print("Computing group centroids...")
    centroids = compute_group_centroids(group_tokens, num_groups, queries_per_group)
    
    # Compute inter-group similarity
    print("Computing similarity matrix...")
    similarity = compute_group_similarity_matrix(centroids)
    
    # Train probes
    print("Training linear probes...")
    # Truncate labels to match samples
    truncated_labels = {k: v[:sample_count] for k, v in attribute_labels.items()}
    probe_results = train_group_probes(centroids, truncated_labels)
    
    # Find best groups
    best_groups = find_best_groups(probe_results)
    
    # Create results table
    table, attr_names = create_probe_results_table(probe_results, num_groups)
    
    return {
        "group_tokens": group_tokens,
        "group_centroids": centroids,
        "similarity_matrix": similarity,
        "probe_results": probe_results,
        "best_groups": best_groups,
        "probe_table": table,
        "attribute_names": attr_names,
        "num_groups": num_groups,
        "num_samples": sample_count,
        "average_off_diagonal_similarity": float(
            (similarity.sum() - np.trace(similarity)) / (num_groups * (num_groups - 1))
        )
    }


def print_specialization_summary(results: Dict[str, Any]):
    """
    Print a formatted summary of group specialization.
    
    Args:
        results: Output from analyze_group_specialization
    """
    print("\n" + "=" * 60)
    print("GROUP SPECIALIZATION ANALYSIS")
    print("=" * 60)
    
    print(f"\nNumber of groups: {results['num_groups']}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Average inter-group similarity: {results['average_off_diagonal_similarity']:.4f}")
    
    print("\n--- Best Group for Each Attribute ---")
    for attr, (group, acc) in results['best_groups'].items():
        print(f"  {attr:20s}: {group} (accuracy: {acc:.3f})")
    
    print("\n--- Full Probe Accuracy Table ---")
    table = results['probe_table']
    attrs = results['attribute_names']
    
    # Header
    header = "Attribute".ljust(20) + " ".join([f"G{i+1:1d}".rjust(6) for i in range(table.shape[1])])
    print(header)
    print("-" * len(header))
    
    # Rows
    for i, attr in enumerate(attrs):
        row = attr.ljust(20)
        for j in range(table.shape[1]):
            acc = table[i, j]
            # Highlight best accuracy for this attribute
            best_in_row = table[i].max()
            if acc == best_in_row:
                row += f" *{acc:.3f}"
            else:
                row += f"  {acc:.3f}"
        print(row)
    
    print("\n* = Best group for this attribute")


if __name__ == "__main__":
    print("Group Probing Analysis Module")
    
    # Test with synthetic data
    np.random.seed(42)
    
    N = 1000
    num_groups = 8
    queries_per_group = 8
    hidden_dim = 128
    
    # Create synthetic group tokens with some structure
    group_tokens = np.random.randn(N, num_groups * queries_per_group, hidden_dim).astype(np.float32)
    
    # Add signal for emotion in group 0
    emotion_labels = np.random.randint(0, 4, N)
    for i, label in enumerate(emotion_labels):
        group_tokens[i, :queries_per_group] += label * 0.5
    
    # Add signal for gender in group 1
    gender_labels = np.random.randint(0, 2, N)
    for i, label in enumerate(gender_labels):
        group_tokens[i, queries_per_group:2*queries_per_group] += label * 1.0
    
    # Compute centroids
    centroids = compute_group_centroids(group_tokens, num_groups, queries_per_group)
    print(f"Centroids shape: {centroids.shape}")
    
    # Compute similarity
    similarity = compute_group_similarity_matrix(centroids)
    print(f"\nInter-group similarity (should be near 0 off-diagonal):")
    print(similarity.round(3))
    
    # Train probes
    labels = {
        "Emotion": emotion_labels,
        "Gender": gender_labels
    }
    probe_results = train_group_probes(centroids, labels)
    
    # Find best groups
    best = find_best_groups(probe_results)
    print("\nBest groups:")
    for attr, (group, acc) in best.items():
        print(f"  {attr}: {group} ({acc:.3f})")
