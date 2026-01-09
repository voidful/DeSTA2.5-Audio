"""
Text Probe Module for ORCA-DeSTA

Implements linear probes for measuring text-predictability of audio representations.
Used to verify redundancy reduction in Observation 2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class LinearProbe(nn.Module):
    """
    Simple linear probe for feature evaluation.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_text_probe(
    audio_features: np.ndarray,
    text_features: np.ndarray,
    hidden_dim: Optional[int] = None,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    test_split: float = 0.2,
    device: str = "cuda",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Train a linear probe to predict text features from audio features.
    
    This measures how much text information is encoded in audio representations.
    High accuracy = high redundancy (bad for disentanglement)
    Low accuracy = low redundancy (good for disentanglement)
    
    Args:
        audio_features: [N, D1] audio representations
        text_features: [N, D2] text embeddings or [N] class labels
        hidden_dim: If set, use MLP instead of linear
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        test_split: Fraction for test set
        device: Device to use
        verbose: Show progress
        
    Returns:
        Dict with train_loss, test_loss, train_acc, test_acc
    """
    # Flatten audio features if needed
    if len(audio_features.shape) > 2:
        audio_features = audio_features.reshape(audio_features.shape[0], -1)
    
    # Check if classification or regression
    is_classification = (len(text_features.shape) == 1 or 
                         (len(text_features.shape) == 2 and text_features.shape[1] == 1))
    
    if is_classification:
        text_features = text_features.flatten()
        num_classes = len(np.unique(text_features))
        output_dim = num_classes
    else:
        output_dim = text_features.shape[1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        audio_features, text_features, test_size=test_split, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    
    if is_classification:
        y_train = torch.from_numpy(y_train).long().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)
    else:
        y_train = torch.from_numpy(y_train).float().to(device)
        y_test = torch.from_numpy(y_test).float().to(device)
    
    # Create model
    input_dim = X_train.shape[1]
    
    if hidden_dim is not None:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
    else:
        model = LinearProbe(input_dim, output_dim).to(device)
    
    # Loss function
    if is_classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training probe")
    
    for epoch in iterator:
        model.train()
        
        # Shuffle training data
        perm = torch.randperm(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            iterator.set_postfix({"loss": f"{epoch_loss/n_batches:.4f}"})
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train)
        test_outputs = model(X_test)
        
        train_loss = criterion(train_outputs, y_train).item()
        test_loss = criterion(test_outputs, y_test).item()
        
        if is_classification:
            train_preds = train_outputs.argmax(dim=1)
            test_preds = test_outputs.argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()
            test_acc = (test_preds == y_test).float().mean().item()
        else:
            # For regression, compute R^2 or cosine similarity as "accuracy"
            train_acc = 1 - (train_loss / (y_train.var().item() + 1e-8))
            test_acc = 1 - (test_loss / (y_test.var().item() + 1e-8))
            train_acc = max(0, train_acc)
            test_acc = max(0, test_acc)
    
    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "is_classification": is_classification,
        "output_dim": output_dim
    }


def train_attribute_probes(
    features: np.ndarray,
    attributes: Dict[str, np.ndarray],
    device: str = "cuda",
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Train probes for multiple attributes.
    
    Args:
        features: [N, D] feature matrix
        attributes: Dict mapping attribute name to labels
        device: Device to use
        **kwargs: Additional arguments for train_text_probe
        
    Returns:
        Dict mapping attribute name to probe results
    """
    results = {}
    for attr_name, labels in attributes.items():
        print(f"\nTraining probe for {attr_name}...")
        results[attr_name] = train_text_probe(
            features, labels, device=device, **kwargs
        )
        print(f"  Test accuracy: {results[attr_name]['test_accuracy']:.4f}")
    
    return results


class BagOfWordsEncoder:
    """
    Simple bag-of-words encoder for text probing.
    """
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.fitted = False
    
    def fit(self, texts: list):
        """Build vocabulary from texts."""
        word_counts = {}
        for text in texts:
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Keep top vocab_size words
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size]):
            self.word2idx[word] = i
            self.idx2word[i] = word
        
        self.fitted = True
    
    def encode(self, texts: list) -> np.ndarray:
        """Encode texts to bag-of-words vectors."""
        if not self.fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        vectors = np.zeros((len(texts), len(self.word2idx)))
        for i, text in enumerate(texts):
            for word in text.lower().split():
                if word in self.word2idx:
                    vectors[i, self.word2idx[word]] = 1
        
        return vectors
    
    def fit_encode(self, texts: list) -> np.ndarray:
        """Fit and encode in one step."""
        self.fit(texts)
        return self.encode(texts)


if __name__ == "__main__":
    # Test with synthetic data
    print("Text Probe Module Test")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create audio features that encode class information
    n_classes = 4
    labels = np.random.randint(0, n_classes, n_samples)
    
    # Create features with high class information
    high_info_features = np.zeros((n_samples, 64))
    for i, label in enumerate(labels):
        high_info_features[i] = np.random.randn(64) + label * 2
    
    # Create features with low class information
    low_info_features = np.random.randn(n_samples, 64)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\nHigh information features (expect high accuracy):")
    result = train_text_probe(high_info_features, labels, device=device, epochs=30)
    print(f"  Test accuracy: {result['test_accuracy']:.4f}")
    
    print("\nLow information features (expect ~random accuracy):")
    result = train_text_probe(low_info_features, labels, device=device, epochs=30)
    print(f"  Test accuracy: {result['test_accuracy']:.4f}")
    print(f"  Random baseline: {1/n_classes:.4f}")
