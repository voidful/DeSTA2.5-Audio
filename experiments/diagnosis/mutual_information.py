"""
Mutual Information Estimation Module for ORCA-DeSTA

Implements MINE (Mutual Information Neural Estimation) for measuring
redundancy between audio representations and text embeddings (Observation 2).

Reference: Belghazi et al., 2018 - "Mutual Information Neural Estimation"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from tqdm import tqdm


class MINENetwork(nn.Module):
    """
    Statistics network for MINE estimation.
    
    Estimates T(x, y) such that MI(X; Y) >= E[T(X, Y)] - log(E[exp(T(X', Y))])
    where X' is drawn from marginal P(X).
    """
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute T(x, y) statistic.
        
        Args:
            x: [B, x_dim] first variable
            y: [B, y_dim] second variable
            
        Returns:
            [B, 1] statistic values
        """
        xy = torch.cat([x, y], dim=-1)
        return self.network(xy)


class MINEEstimator:
    """
    Mutual Information Neural Estimation.
    
    Estimates I(X; Y) by training a neural network to distinguish
    joint samples (x, y) from marginal samples (x, y').
    """
    
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 256,
        device: str = "cuda"
    ):
        self.device = device
        self.network = MINENetwork(x_dim, y_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        
    def _mine_loss(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        y_marginal: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute MINE loss and MI estimate.
        
        Args:
            x: [B, x_dim] samples from X
            y: [B, y_dim] samples from Y (joint with X)
            y_marginal: [B, y_dim] samples from marginal P(Y)
            
        Returns:
            (loss, mi_estimate)
        """
        # Joint samples
        t_joint = self.network(x, y)
        
        # Marginal samples (shuffle y to break correlation)
        t_marginal = self.network(x, y_marginal)
        
        # MINE lower bound: E[T(X,Y)] - log(E[exp(T(X, Y'))])
        # Use log-sum-exp trick for stability
        mi_estimate = t_joint.mean() - torch.logsumexp(t_marginal, dim=0) + np.log(t_marginal.size(0))
        
        # Loss is negative MI (we want to maximize MI estimate)
        loss = -mi_estimate
        
        return loss, mi_estimate.item()
    
    def train_step(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> float:
        """
        Single training step.
        
        Args:
            x: [B, x_dim] audio features
            y: [B, y_dim] text features
            
        Returns:
            Current MI estimate
        """
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Create marginal samples by shuffling y
        perm = torch.randperm(y.size(0))
        y_marginal = y[perm]
        
        self.optimizer.zero_grad()
        loss, mi = self._mine_loss(x, y, y_marginal)
        loss.backward()
        self.optimizer.step()
        
        return mi
    
    def estimate(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        batch_size: int = 256,
        num_epochs: int = 100,
        verbose: bool = True
    ) -> float:
        """
        Estimate mutual information between X and Y.
        
        Args:
            x_data: [N, x_dim] audio features
            y_data: [N, y_dim] text features
            batch_size: Training batch size
            num_epochs: Number of training epochs
            verbose: Whether to show progress
            
        Returns:
            Estimated MI in nats
        """
        x_tensor = torch.from_numpy(x_data).float()
        y_tensor = torch.from_numpy(y_data).float()
        
        n_samples = x_tensor.size(0)
        mi_history = []
        
        iterator = range(num_epochs)
        if verbose:
            iterator = tqdm(iterator, desc="MINE training")
        
        for epoch in iterator:
            # Shuffle data each epoch
            perm = torch.randperm(n_samples)
            x_shuffled = x_tensor[perm]
            y_shuffled = y_tensor[perm]
            
            epoch_mi = []
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                if len(x_batch) < 2:
                    continue
                    
                mi = self.train_step(x_batch, y_batch)
                epoch_mi.append(mi)
            
            avg_mi = np.mean(epoch_mi) if epoch_mi else 0
            mi_history.append(avg_mi)
            
            if verbose and (epoch + 1) % 10 == 0:
                iterator.set_postfix({"MI": f"{avg_mi:.4f}"})
        
        # Return average of last 10 epochs for stability
        return float(np.mean(mi_history[-10:]))


def estimate_mutual_information(
    audio_features: np.ndarray,
    text_features: np.ndarray,
    hidden_dim: int = 256,
    num_epochs: int = 100,
    batch_size: int = 256,
    device: str = "cuda",
    verbose: bool = True
) -> float:
    """
    Estimate mutual information between audio and text features using MINE.
    
    Args:
        audio_features: [N, D1] audio representations
        text_features: [N, D2] text embeddings
        hidden_dim: Hidden dimension for MINE network
        num_epochs: Training epochs
        batch_size: Batch size
        device: Device to use
        verbose: Show progress
        
    Returns:
        Estimated MI in nats
    """
    # Flatten if needed
    if len(audio_features.shape) > 2:
        audio_features = audio_features.reshape(audio_features.shape[0], -1)
    if len(text_features.shape) > 2:
        text_features = text_features.reshape(text_features.shape[0], -1)
    
    estimator = MINEEstimator(
        x_dim=audio_features.shape[1],
        y_dim=text_features.shape[1],
        hidden_dim=hidden_dim,
        device=device
    )
    
    return estimator.estimate(
        audio_features,
        text_features,
        batch_size=batch_size,
        num_epochs=num_epochs,
        verbose=verbose
    )


def compute_linear_cka(
    features_x: np.ndarray,
    features_y: np.ndarray
) -> float:
    """
    Compute linear Centered Kernel Alignment (CKA) between two feature sets.
    
    CKA measures similarity between representations in a way that's
    invariant to orthogonal transformations and isotropic scaling.
    
    Args:
        features_x: [N, D1] first feature set
        features_y: [N, D2] second feature set
        
    Returns:
        CKA similarity (0 to 1, higher = more similar)
    """
    # Center features
    x = features_x - features_x.mean(axis=0)
    y = features_y - features_y.mean(axis=0)
    
    # Compute Gram matrices
    k_x = x @ x.T
    k_y = y @ y.T
    
    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_xy = np.sum(k_x * k_y)
    hsic_xx = np.sum(k_x * k_x)
    hsic_yy = np.sum(k_y * k_y)
    
    # CKA
    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)
    
    return float(cka)


if __name__ == "__main__":
    # Test with synthetic data
    print("Mutual Information Estimation Module")
    
    np.random.seed(42)
    n_samples = 2000
    
    # Create correlated data
    z = np.random.randn(n_samples, 10)
    x = z @ np.random.randn(10, 32) + 0.1 * np.random.randn(n_samples, 32)
    y = z @ np.random.randn(10, 32) + 0.1 * np.random.randn(n_samples, 32)
    
    # Create independent data
    x_ind = np.random.randn(n_samples, 32)
    y_ind = np.random.randn(n_samples, 32)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nUsing device: {device}")
    
    print("\nCorrelated data (expect high MI):")
    mi_corr = estimate_mutual_information(x, y, num_epochs=50, device=device)
    print(f"  MI estimate: {mi_corr:.4f} nats")
    
    print("\nIndependent data (expect low MI):")
    mi_ind = estimate_mutual_information(x_ind, y_ind, num_epochs=50, device=device)
    print(f"  MI estimate: {mi_ind:.4f} nats")
    
    print("\nCKA tests:")
    print(f"  CKA (correlated): {compute_linear_cka(x, y):.4f}")
    print(f"  CKA (independent): {compute_linear_cka(x_ind, y_ind):.4f}")
