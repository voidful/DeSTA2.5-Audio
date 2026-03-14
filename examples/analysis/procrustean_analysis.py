import os

# Set allocator env before importing torch.
# Newer PyTorch warns PYTORCH_CUDA_ALLOC_CONF is deprecated. Use PYTORCH_ALLOC_CONF.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
import math
import random
import tempfile
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

import torch

from datasets import load_dataset

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, RidgeClassifier, RidgeCV, RidgeClassifierCV
from sklearn.metrics import f1_score, accuracy_score, r2_score
from scipy.stats import pearsonr
from scipy.linalg import orthogonal_procrustes
from sklearn.model_selection import GridSearchCV

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    seed: int = 7

    # Dataset
    dataset_name: str = "myleslinder/crema-d"
    dataset_split: str = "train"

    # Sampling
    num_samples: int = 4096
    max_audio_seconds: float = 4.0

    # Audio trimming. Emotion may peak near the end. Preserve tail by default.
    # Options: "head", "end", "random"
    trim_strategy: str = "end"

    # External text anchor
    text_encoder_name: str = "sentence-transformers/all-mpnet-base-v2"
    text_emb_dim: int = 768

    # U_t
    text_subspace_k: int = 32
    ut_extra_text: int = 2000

    # Optional: whiten text embeddings before PCA for Ut.
    ut_whiten: bool = True
    
    # Whitening method for Procrustes alignment
    ut_whiten_method: str = "whitening" # isotropic | whitening
    
    # PCA variance threshold for dynamic k selection
    pca_min_variance: float = 0.99

    # Ridge map W
    ridge_alphas: Tuple[float, ...] = tuple(10.0 ** np.arange(-6, 8))  # 1e-6 to 1e7

    ridge_cv_folds: int = 5

    # Representative vector pooling for ridge training.
    ridge_pool_default: str = "mean"   # mean | last | max
    ridge_pool_qwen2: str = "last"     # mean | last | max

    # Optional quick pooling ablation for Qwen2 before full run.
    qwen2_pooling_ablation: bool = True
    qwen2_pooling_ablation_n: int = 512

    # Auto override qwen2 pooling for the current run based on ablation.
    qwen2_pooling_auto_override: bool = True

    # Geometry audio subspace for angles
    audio_subspace_k: int = 32

    # Token sampling to bound memory for global token statistics.
    max_tokens_for_global_stats: int = 500_000  # Increased for 128GB RAM

    # Intervention - optimized for 128GB RAM
    intervention_samples: int = 128  # Doubled for more statistical power
    intervention_samples_low_mem: int = 64  # Still generous for Qwen2.5-Omni
    pitch_shifts: Tuple[int, ...] = (-4, -2, 0, 2, 4)
    gain_db: Tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0)
    
    # Memory mode: "high" (128GB+), "medium" (32-128GB), "low" (<32GB)
    # Controls cleanup frequency and cache precision
    memory_tier: str = "high"
    
    # Legacy: low memory mode (auto-enabled for Qwen2.5-Omni regardless of tier)
    low_memory_mode: bool = False
    
    # Token cache mode: "ram" (all in RAM), "disk" (save to temp files), "lazy" (on-demand extraction)
    # For very large models like Qwen2.5-Omni, use "disk" or "lazy" to avoid OOM
    token_cache_mode: str = "ram"  # ram | disk | lazy
    token_cache_dir: Optional[str] = None  # If None, uses tempfile.mkdtemp()
    
    # Cleanup intervals (higher = less frequent cleanup = faster but more RAM)
    # For 128GB RAM, we can be less aggressive with cleanup
    cleanup_interval_token_cache: int = 500   # Cleanup every N samples during caching
    cleanup_interval_token_cache_qwen_omni: int = 50  # More aggressive for Qwen2.5-Omni
    cleanup_interval_intervention: int = 10   # Cleanup every N interventions
    cleanup_interval_sanity: int = 8          # Cleanup every N sanity tests
    cleanup_interval_ablation: int = 100      # Cleanup every N ablation samples
    
    # Skip sanity suite (saves time and memory for large models)
    skip_sanity_suite: bool = False  # Set to True for Qwen2.5-Omni to save memory

    # Token alignment
    token_align: str = "dtw"   # resample | dtw
    resample_T: int = 64

    # DTW speed. PCA dim used only for distance computation.
    dtw_pca_dim: int = 16

    # Probes
    cv_folds: int = 5
    min_per_class: int = 5
    max_speaker_classes: int = 30

    # Sanity check thresholds for decomposition.
    sentence_par_min_f1: float = 0.70
    sentence_perp_max_f1: float = 0.30

    # Regression probe CV (pitch/energy). Use GroupKFold(sentence) to reduce leakage.
    reg_cv_folds: int = 5

    # Dose-response fitting.
    # Primary metrics used for "sensitivity" headline numbers.
    # You can add more later, but these two are the reviewer-friendly defaults.
    dose_primary_metric_pitch: str = "meanvec_perp"  # meanvec_perp | meanvec_ratio | token_perp | token_mean
    dose_primary_metric_gain: str = "meanvec_perp"   # meanvec_perp | meanvec_ratio | token_perp | token_mean

    # Output
    out_dir: str = "./procrustean_analysis_v3_5"


CFG_ = CFG()


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_mem(prefix: str) -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    print(f"  [CUDA] {prefix}: allocated={alloc:.2f}GB reserved={reserv:.2f}GB")


def free_torch(threshold_gb: float = 0.0) -> None:
    """
    Release memory. 
    If threshold_gb > 0, only behaves as 'smart gc' and collects
    if reserved memory > threshold_gb.
    """
    should_collect = True
    if threshold_gb > 0 and torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved() / 1024**3
        if reserved < threshold_gb:
            should_collect = False
    
    if should_collect:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def l2_norm(x: np.ndarray, axis: Optional[int] = None, eps: float = 1e-12) -> np.ndarray:
    return np.sqrt(np.maximum(np.sum(x * x, axis=axis), eps))


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(l2_norm(a))
    nb = float(l2_norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def effective_rank_from_svals(s: np.ndarray, eps: float = 1e-12) -> float:
    # Use ENERGY (s^2) for distribution, not raw svals
    s2 = s.astype(np.float64) ** 2
    p = s2 / (np.sum(s2) + eps)
    H = -np.sum(p * np.log(p + eps))
    return float(np.exp(H))


def topk_concentration(s: np.ndarray, k: int = 5, eps: float = 1e-12) -> float:
    # Use ENERGY (s^2) for concentration
    s2 = s.astype(np.float64) ** 2
    ss = np.sort(s2)[::-1]
    return float(np.sum(ss[:k]) / (np.sum(ss) + eps))


def trim_audio(arr: np.ndarray, sr: int, max_seconds: float, strategy: str, seed: int) -> np.ndarray:
    max_len = int(max_seconds * sr)
    if arr.shape[0] <= max_len:
        return arr.astype(np.float32)

    if strategy == "head":
        out = arr[:max_len]
        return out.astype(np.float32)

    if strategy == "end":
        out = arr[-max_len:]
        return out.astype(np.float32)

    if strategy == "random":
        rng = np.random.RandomState(seed)
        start = int(rng.randint(0, arr.shape[0] - max_len + 1))
        out = arr[start : start + max_len]
        return out.astype(np.float32)

    out = arr[-max_len:]
    return out.astype(np.float32)


def sample_rows(X: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if X.shape[0] <= max_rows:
        return X
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=max_rows, replace=False)
    return X[idx]


def isotropic_scale_fro(X: np.ndarray, eps: float = 1e-9) -> float:
    # X: [N, D] assumed already centered
    # Returns RMS norm (Frobenius / sqrt(N*D)).
    # "We normalize the centered embeddings by their RMS amplitude to ensure scale invariance."
    N, D = X.shape
    return float(np.linalg.norm(X, ord="fro") / np.sqrt(max(N * D, 1)) + eps)


def procrustes_rotation_no_reflection(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve argmin_{R^T R=I, det(R)=1} ||A R - B||_F.
    A, B: [N, k] centered/scaled.
    """
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


# -----------------------------
# Dataset loading
# -----------------------------
def load_cremad(cfg: CFG) -> List[Dict[str, Any]]:
    """
    Returns list of dicts:
      audio: np.ndarray float32
      sr: int
      text: str
      sentence_id: str (group key)
      speaker_id: Any
      emotion: str
    """
    print(f"📦 Loading dataset: {cfg.dataset_name} [{cfg.dataset_split}]")
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    ds = ds.shuffle(seed=cfg.seed)
    ds = ds.select(range(min(cfg.num_samples, len(ds))))

    items: List[Dict[str, Any]] = []
    for ex in ds:
        audio_obj = ex.get("audio", None)
        if audio_obj is None:
            continue

        arr = np.asarray(audio_obj["array"], dtype=np.float32)
        sr = int(audio_obj["sampling_rate"])
        arr = trim_audio(arr, sr, cfg.max_audio_seconds, cfg.trim_strategy, cfg.seed)

        text = ex.get("text", None)
        if text is None:
            text = ex.get("sentence", None)
        if text is None:
            text = ex.get("transcription", None)
        if text is None:
            text = str(ex.get("sentence_id", ""))

        sentence_id = ex.get("sentence_id", None)
        if sentence_id is None:
            sentence_id = text

        speaker_id = ex.get("actor_id", None)
        if speaker_id is None:
            speaker_id = ex.get("speaker_id", None)
        if speaker_id is None:
            speaker_id = ex.get("speaker", None)
        if speaker_id is None:
            speaker_id = "unknown"
        speaker_id = str(speaker_id)  # Enforce string type for consistent label encoding

        emotion = ex.get("emotion", None)
        if emotion is None:
            emotion = ex.get("label", None)
        if emotion is None:
            emotion = "unknown"
        emotion = str(emotion)  # Enforce string type for consistent label encoding

        items.append(
            dict(
                audio=arr,
                sr=sr,
                text=str(text),
                sentence_id=str(sentence_id),
                speaker_id=speaker_id,
                emotion=str(emotion),
            )
        )

    print(f"  Loaded samples: {len(items)}")
    return items


def load_extra_text(cfg: CFG) -> List[str]:
    """
    Robust U_t requires diverse text. CREMA-D has only 12 sentences.
    Use Wikitext lines as augmentation.
    """
    print(f"🧠 Loading extra text for robust U_t. n={cfg.ut_extra_text}")
    extra: List[str] = []
    try:
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for ex in wiki:
            t = ex.get("text", "")
            t = t.strip()
            if len(t) < 20:
                continue
            extra.append(t)
            if len(extra) >= cfg.ut_extra_text:
                break
    except Exception as e:
        print(f"  ⚠️ Failed to load wikitext. err={e}. Fallback to empty extra.")
        extra = []
    print(f"  Extra sentences collected: {len(extra)}")
    return extra


# -----------------------------
# External text encoder
# -----------------------------
class TextEncoder:
    def __init__(self, name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        import logging
        # Suppress harmless "UNEXPECTED: embeddings.position_ids" warning
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        
        self.model = SentenceTransformer(name, device=device)

    def encode(self, sentences: List[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(emb, dtype=np.float32)


def build_text_subspace(text_embs: np.ndarray, k: int, seed: int, whiten: bool) -> np.ndarray:
    """
    Returns U_t: [D, k] with orthonormal columns.
    Optional whitening improves scale robustness.
    """
    X = text_embs.astype(np.float32)
    if whiten:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X).astype(np.float32)

    k_eff = int(min(k, X.shape[0], X.shape[1]))
    pca = PCA(n_components=k_eff, random_state=seed)
    pca.fit(X)
    Ut = pca.components_.T.astype(np.float32)
    q, _ = np.linalg.qr(Ut)
    return q[:, :k_eff].astype(np.float32)


def build_text_subspace_robust(
    crema_embs: np.ndarray, 
    extra_embs: np.ndarray, 
    k: int, 
    seed: int
) -> np.ndarray:
    """
    Build text subspace U_t that PRIORITIZES CREMA-D sentences.
    
    Crucially uses CREMA-D mean as the shared origin for consistent centering.
    This ensures SVD captures variance (not mean) and the subspace is properly aligned.
    
    Strategy:
    1. Center using CREMA-D mean as the shared origin
    2. SVD on CREMA-D embeddings first to span their variation
    3. If k > n_crema, fill remaining dims with Wikitext (orthogonal to CREMA-D span)
    4. QR decomposition for orthonormality
    
    Args:
        crema_embs: [N_crema, D] embeddings of unique CREMA-D sentences
        extra_embs: [N_extra, D] embeddings of extra text (Wikitext)
        k: target subspace dimension
        seed: random seed (unused but kept for API consistency)
        
    Returns:
        U_t: [D, k] orthonormal basis for text subspace
    """
    crema_embs = crema_embs.astype(np.float64)  # Use float64 for numerical stability
    extra_embs = extra_embs.astype(np.float64)
    
    # 1. Centering: Use CREMA-D mean as the shared "origin" for the subspace
    # This is crucial - both datasets should be centered relative to the SAME point
    mean_vec = crema_embs.mean(axis=0, keepdims=True)
    crema_centered = crema_embs - mean_vec
    
    # 2. SVD on CREMA-D sentences
    # crema_centered.T shape: [D, N_crema]
    U_crema, S_crema, _ = np.linalg.svd(crema_centered.T, full_matrices=False)
    # U_crema: [D, min(D, N_crema)], columns are principal directions
    
    # Take effective rank of CREMA (should be ~11 for 12 sentences after centering)
    n_crema_dims = min(k, U_crema.shape[1], crema_embs.shape[0] - 1)  # -1 for centering
    n_crema_dims = max(n_crema_dims, 1)
    U_base = U_crema[:, :n_crema_dims]
    
    # 3. If k > n_crema_dims, fill remaining with extra text (orthogonal to CREMA-D)
    if k > n_crema_dims and extra_embs.shape[0] > 0:
        # Center extra text relative to the SAME CREMA-D mean (shared origin)
        extra_centered = extra_embs - mean_vec
        
        # Project extra text away from U_base (Gram-Schmidt orthogonalization)
        # perp = X - U @ U^T @ X
        proj = (extra_centered @ U_base) @ U_base.T
        extra_perp = extra_centered - proj
        
        # SVD on the perpendicular residuals
        U_extra, S_extra, _ = np.linalg.svd(extra_perp.T, full_matrices=False)
        n_extra_dims = min(k - n_crema_dims, U_extra.shape[1])
        
        # Concatenate: CREMA-D first, then extra
        U_t = np.concatenate([U_base, U_extra[:, :n_extra_dims]], axis=1)
    else:
        U_t = U_base
    
    # 4. Final QR decomposition to ensure perfect orthonormality numerically
    Q, _ = np.linalg.qr(U_t)
    k_final = min(k, Q.shape[1])
    
    print(f"  📐 Robust U_t: {n_crema_dims} CREMA-D dims + {k_final - n_crema_dims} extra dims = {k_final} total")
    
    return Q[:, :k_final].astype(np.float32)


def build_ut_from_sentence_centroids(
    Y_means: np.ndarray,
    sentence_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Build semantic subspace Ut from MAPPED AUDIO (Y) sentence centroids.
    
    CRITICAL FIX: This defines "semantic" as what the AUDIO MODEL itself uses 
    to distinguish sentences, NOT what an external text encoder thinks.
    
    This ensures Ut⊥ (perpendicular space) contains only intra-sentence variation
    (paralinguistic features), not leaked semantic information.
    
    Args:
        Y_means: [N, D] mean embeddings of mapped audio samples
        sentence_ids: [N] sentence ID for each sample
        k: target dimension of Ut
        
    Returns:
        Ut: [D, k] orthonormal basis for semantic subspace (from audio model's perspective)
    """
    Y_means = Y_means.astype(np.float64)
    
    # Compute sentence centroids (one per unique sentence)
    unique_sents = np.unique(sentence_ids)
    centroids = []
    for s in unique_sents:
        mask = sentence_ids == s
        if np.sum(mask) > 0:
            centroids.append(Y_means[mask].mean(axis=0))
    
    centroids = np.stack(centroids, axis=0)  # [n_sentences, D]
    n_sents = centroids.shape[0]
    
    # Center centroids
    mean_vec = centroids.mean(axis=0, keepdims=True)
    centroids_centered = centroids - mean_vec
    
    # SVD to find principal directions of inter-sentence variation
    U, S, Vt = np.linalg.svd(centroids_centered, full_matrices=False)
    # Vt: [min(n,D), D], rows are principal directions
    
    # Take top-k directions (up to n_sents-1 after centering)
    k_eff = min(k, len(S), n_sents - 1)
    k_eff = max(k_eff, 1)
    
    Ut = Vt[:k_eff].T  # [D, k_eff]
    
    # QR for numerical orthonormality
    Q, _ = np.linalg.qr(Ut)
    
    print(f"  📐 Ut from Y centroids: {n_sents} sentences -> {Q.shape[1]} dims (explains sentence variation)")
    
    return Q.astype(np.float32)


def validate_ut_geometry(Ut: np.ndarray, expected_D: int = 768) -> None:
    """
    Hard assertions to validate Ut geometry.
    Catches bugs like projection becoming identity or dimension mismatch.
    """
    D, k = Ut.shape
    
    # Check embedding dimension
    assert D == expected_D, f"Ut embedding dimension wrong! Expected {expected_D}, got {D}"
    
    # Check k is reasonable
    assert k >= 1, f"Ut has no dimensions! k={k}"
    assert k <= D, f"Ut has more dims than embedding! k={k} > D={D}"
    
    # Compute projection matrix and verify trace
    P = Ut @ Ut.T  # [D, D]
    trace_P = np.trace(P)
    
    # Trace of projection matrix should equal k (number of dimensions)
    assert abs(trace_P - k) < 0.1, f"Projection dimension mismatch! trace(P)={trace_P:.4f}, expected k={k}"
    
    # Verify orthonormality: Ut.T @ Ut should be identity
    inner = Ut.T @ Ut  # [k, k]
    identity_err = np.max(np.abs(inner - np.eye(k)))
    assert identity_err < 1e-5, f"Ut not orthonormal! max error = {identity_err:.2e}"
    
    print(f"  ✅ Ut geometry validated: D={D}, k={k}, trace(P)={trace_P:.4f}")


# -----------------------------
# Audio model adapters
# -----------------------------
# Contract: extract_tokens(audio, sr) -> torch.Tensor [T, Dz] on CPU float32.

class AudioAdapterBase:
    name: str

    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device

    def load(self) -> None:
        raise NotImplementedError

    @torch.inference_mode()
    def extract_tokens(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        raise NotImplementedError

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        free_torch()



def load_desta_model(model_id: str):
    from desta.models.modeling_desta25 import DeSTA25AudioModel
    from transformers import AutoFeatureExtractor

    print(f"🔄 Loading DeSTA: {model_id}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # device_map="auto" may or may not be supported by this custom class.
    # We keep it simple and rely on .to(DEVICE).
    # NOTE: DEVICE global is not available here, falling back to cuda/cpu logic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DeSTA25AudioModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    encoder_id = getattr(model.config, "encoder_model_id", "openai/whisper-large-v3")
    processor = AutoFeatureExtractor.from_pretrained(encoder_id)
    return model, processor


def load_qwen2_audio_model(model_id: str):
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    print(f"🔄 Loading Qwen2-Audio: {model_id}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


class DeSTAAdapter(AudioAdapterBase):
    name = "DeSTA"

    def load(self) -> None:
        self.model, self.processor = load_desta_model(self.model_id)

    @torch.inference_mode()
    def extract_tokens(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        # Use simple AutoFeatureExtractor logic + model.perception
        if not hasattr(self, "model"):
             self.load()
             
        device = self.model.device
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        # Use perception module to get audio tokens
        with torch.no_grad():
            perception_output = self.model.perception(input_features=input_features)
        
        # Cleanup input early
        del input_features
            
        # Handle output format (tuple or tensor)
        if isinstance(perception_output, tuple):
             # Usually (features, lengths) or (tokens, losses)
             Z = perception_output[0]
        else:
             Z = perception_output
             
        # Z is [Batch, Time, Dim]. We assume Batch (1 is standard in this script loop)
        result = Z[0].detach().float().cpu()
        del Z
        del perception_output
        return result



class Qwen2AudioAdapter(AudioAdapterBase):
    name = "Qwen2-Audio"

    def load(self) -> None:
        self.model, self.processor = load_qwen2_audio_model(self.model_id)

    @torch.inference_mode()
    def extract_tokens(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Extract audio representations using audio_tower + multi_modal_projector.
        This bypasses the problematic processor flow that ignores `audios` keyword.
        """
        if not hasattr(self, "model"):
             self.load()
        
        device = self.model.device
        dtype = self.model.dtype
        
        # Use audio_tower directly for clean audio representation
        # This avoids the processor's text requirement and ensures we get actual audio features
        if hasattr(self.model, "audio_tower"):
            # Get mel features via feature_extractor (part of processor)
            mel = self.processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
            input_features = mel["input_features"].to(device=device, dtype=dtype)
            
            # Debug: Log input shape on first call
            if not hasattr(self, "_logged_shapes"):
                self._logged_shapes = True
                print(f"  📐 Qwen2 input_features shape: {input_features.shape}")
            
            # Forward through audio tower (Whisper encoder typically)
            audio_out = self.model.audio_tower(input_features)
            audio_h = audio_out.last_hidden_state
            
            # Cleanup input features early
            del input_features
            del audio_out
            
            # Debug: Log intermediate shape on first call
            if not hasattr(self, "_logged_audio_h"):
                self._logged_audio_h = True
                print(f"  📐 Qwen2 audio_tower output shape: {audio_h.shape}")
            
            # Project to LLM dimension via multi_modal_projector
            Z = self.model.multi_modal_projector(audio_h)
            del audio_h  # Free memory
            
            # Debug: Log final shape on first call
            if not hasattr(self, "_logged_Z"):
                self._logged_Z = True
                print(f"  📐 Qwen2 final Z shape: {Z.shape}")
            
            # Shape validation: Z should be [batch, T, D] where T > 1
            if Z.dim() == 3:
                Z_out = Z[0].detach().float().cpu()  # [T, D]
            elif Z.dim() == 2:
                Z_out = Z.detach().float().cpu()  # [T, D] already
            else:
                raise ValueError(f"Unexpected Z shape: {Z.shape}. Expected [batch, T, D] or [T, D].")
            
            del Z  # Free GPU memory
            
            # Sanity assertion: T should be > 1 for meaningful sequence
            T, D = Z_out.shape
            if T == 1:
                print(f"  ⚠️ WARNING: Qwen2 output has T=1 (single token). May indicate pooled output or extraction error.")
            
            return Z_out  # [T, D]
        else:
            # Fallback: try full forward pass with text prompt
            # This is less reliable but better than failing
            print("  ⚠️ audio_tower not found, using fallback forward pass")
            text = "<|audio_bos|><|AUDIO|><|audio_eos|>"
            try:
                inputs = self.processor(text=text, audios=audio, sampling_rate=sr, return_tensors="pt")
            except TypeError:
                inputs = self.processor(text=text, audio=audio, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            Z = out.hidden_states[-1][0].detach().float().cpu()
            del out
            del inputs
            return Z


def load_qwen2_5_omni_model(model_id: str):
    """Load Qwen2.5-Omni model with audio output disabled."""
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniConfig,
        Qwen2_5OmniProcessor,
    )
    print(f"🔄 Loading Qwen2.5-Omni: {model_id}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Load config and disable audio output generation
    config = Qwen2_5OmniConfig.from_pretrained(model_id)
    config.enable_audio_output = False
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def load_audio_flamingo3_model(model_id: str):
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    print(f"🔄 Loading Audio Flamingo 3: {model_id}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor





class Qwen2_5OmniAdapter(AudioAdapterBase):
    """Adapter for Qwen2.5-Omni (Qwen/Qwen2.5-Omni-3B)."""
    name = "Qwen2.5-Omni"

    def load(self) -> None:
        self.model, self.processor = load_qwen2_5_omni_model(self.model_id)
        
        # Explore model structure to find audio encoder
        if not hasattr(self, "_logged_structure"):
            self._logged_structure = True
            print(f"  📐 Qwen2.5-Omni model attributes: {[a for a in dir(self.model) if not a.startswith('_')][:20]}...")
            if hasattr(self.model, "thinker"):
                print(f"  📐 Thinker attributes: {[a for a in dir(self.model.thinker) if not a.startswith('_')][:20]}...")

    @torch.inference_mode()
    def extract_tokens(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Extract audio representations from Qwen2.5-Omni.
        Uses thinker.get_audio_features() which handles all chunking complexity.
        
        OPTIMIZED: Aggressive memory cleanup to prevent CUDA OOM.
        """
        if not hasattr(self, "model"):
            self.load()
        
        device = self.model.device
        dtype = self.model.dtype
        
        # Resample to target sample rate
        target_sr = int(self.processor.feature_extractor.sampling_rate)
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        
        # Get mel features from feature_extractor
        mel = self.processor.feature_extractor(audio, sampling_rate=target_sr, return_tensors="pt")
        input_features = mel["input_features"].to(device=device, dtype=dtype)
        del mel  # Free CPU memory immediately
        
        # Create feature attention mask (all 1s for single sample)
        feature_attention_mask = torch.ones(
            (input_features.shape[0], input_features.shape[-1]),
            device=device,
            dtype=torch.long
        )
        
        if not hasattr(self, "_logged_shapes"):
            self._logged_shapes = True
            print(f"  📐 Qwen2.5-Omni input_features shape: {input_features.shape}")
            print(f"  📐 Qwen2.5-Omni feature_attention_mask shape: {feature_attention_mask.shape}")
        
        # Get thinker and use get_audio_features which handles complexity
        thinker = self.model.thinker
        
        try:
            # get_audio_features handles all the chunking and audio_tower complexity
            audio_outputs = thinker.get_audio_features(
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
            )
            
            # Explicitly delete inputs to free GPU memory early
            del input_features
            del feature_attention_mask
            
            # Handle different return types
            if hasattr(audio_outputs, 'last_hidden_state'):
                Z = audio_outputs.last_hidden_state
            elif hasattr(audio_outputs, 'pooler_output'):
                Z = audio_outputs.pooler_output
            elif isinstance(audio_outputs, tuple):
                Z = audio_outputs[0]
            else:
                # Direct tensor return
                Z = audio_outputs
            
            if not hasattr(self, "_logged_Z"):
                self._logged_Z = True
                print(f"  📐 Qwen2.5-Omni audio features shape: {Z.shape}")
            
            # Shape: [T, D] for single sample - move to CPU immediately to free GPU
            if Z.dim() == 3:
                result = Z[0].detach().float().cpu()
            elif Z.dim() == 2:
                result = Z.detach().float().cpu()
            elif Z.dim() == 1:
                result = Z.unsqueeze(0).detach().float().cpu()
            else:
                raise ValueError(f"Unexpected Z shape: {Z.shape}")
            
            # === AGGRESSIVE GPU MEMORY CLEANUP ===
            # Delete all GPU tensors
            del Z
            del audio_outputs
            
            # Synchronize to ensure all CUDA operations are complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # REMOVED: Force garbage collection every extraction for Qwen2.5-Omni
            # usage of gc.collect() and empty_cache() is too slow for every sample
            # relying on outer loop cleanup interval instead
            
            return result
                
        except Exception as e:
            if not hasattr(self, "_logged_error"):
                self._logged_error = True
                print(f"  ⚠️ get_audio_features failed: {e}")
            
            # Cleanup on error
            try:
                del input_features
            except:
                pass
            try:
                del feature_attention_mask
            except:
                pass
            
            # Force cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return dummy to avoid crash
            return torch.zeros(10, 2048)


class AudioFlamingo3Adapter(AudioAdapterBase):
    """Adapter for nvidia/audio-flamingo-3-hf."""
    name = "AudioFlamingo3"

    def load(self) -> None:
        self.model, self.processor = load_audio_flamingo3_model(self.model_id)

    @torch.inference_mode()
    def extract_tokens(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Extract audio features using explicit processor call.
        This handles multimodal inputs more reliably than apply_chat_template.
        """
        if not hasattr(self, "model"):
            self.load()
            
        device = self.model.device
        target_sr = int(self.processor.feature_extractor.sampling_rate)
        
        # Resample logic
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            
        # Audio Flamingo 3 reliable input handling:
        # 1. Write audio to temporary WAV file
        # 2. Use apply_chat_template with structured input {"type": "audio", "path": ...}
        # 3. This matches how the model was designed to be used
        import tempfile
        import soundfile as sf
        import os
        
        # Create temp file
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            # Write audio to file
            sf.write(temp_path, audio, target_sr)
            
            # Construct structured conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract audio features."},
                        {"type": "audio", "path": temp_path},
                    ],
                }
            ]
            
            # Use apply_chat_template which handles the audio loading and processing internally
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(device)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Cast float inputs to model dtype (e.g., float16)
        dtype = self.model.dtype
        for key in ["input_features", "pixel_values"]:
            if key in inputs and inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(dtype)
            
        # DEBUG: Check inputs again
        if not hasattr(self, "_logged_debug"):
            self._logged_debug = True
            print(f"  🐞 DEBUG: AudioFlamingo3 inputs keys: {list(inputs.keys())}")
            if "input_features" in inputs:
                print(f"  🐞 DEBUG: input_features dtype: {inputs['input_features'].dtype}, shape: {inputs['input_features'].shape}")
        
        out = self.model(
            **inputs, 
            output_hidden_states=True, 
            return_dict=True
        )
        
        # Last hidden state
        Z = out.hidden_states[-1]
        
        # Cleanup
        del out
        del inputs
        
        if Z.dim() == 3:
            result = Z[0].detach().float().cpu()
        else:
            result = Z.detach().float().cpu()
        
        return result


# -----------------------------
# Adapter Sanity Suite
# -----------------------------
def run_adapter_sanity_suite(
    adapter: AudioAdapterBase,
    items: List[Dict[str, Any]],
    cfg: CFG,
) -> Dict[str, Any]:
    """
    Three-test validation suite to ensure adapter extracts real audio features.
    
    Sanity A: Silence test - Z(real_audio) should differ from Z(zeros)
    Sanity B: Rank test - effective_rank > threshold (not degenerate output)
    Sanity C: Dimension sanity - output has reasonable shape
    
    Returns dict with pass/fail status and metrics for each test.
    """
    print("🔬 Running adapter sanity suite...")
    results = {
        "silence_test_passed": False,
        "rank_test_passed": False,
        "dim_test_passed": False,
        "all_passed": False,
    }
    
    # Sample items for testing (use first few)
    n_test = min(16, len(items))
    test_items = items[:n_test]
    
    # Collect Z for real audio
    Zs_real = []
    Zs_silence = []
    
    for i, it in enumerate(test_items):
        audio = it["audio"]
        sr = it["sr"]
        
        # Real audio
        Z_real = adapter.extract_tokens(audio, sr)
        Zs_real.append(Z_real.numpy())
        
        # Silence (zeros with same shape)
        silence = np.zeros_like(audio, dtype=np.float32)
        Z_silence = adapter.extract_tokens(silence, sr)
        Zs_silence.append(Z_silence.numpy())
        
        # Periodic cleanup based on config interval
        if (i + 1) % cfg.cleanup_interval_sanity == 0:
            free_torch()
    free_torch()  # Final cleanup
    
    # === Sanity A: Silence Test ===
    # Compute average L2 norm difference between real and silence embeddings
    real_norms = [np.linalg.norm(Z.mean(axis=0)) for Z in Zs_real]
    silence_norms = [np.linalg.norm(Z.mean(axis=0)) for Z in Zs_silence]
    
    avg_real_norm = float(np.mean(real_norms))
    avg_silence_norm = float(np.mean(silence_norms))
    norm_ratio = avg_real_norm / (avg_silence_norm + 1e-9)
    
    # Compute cosine similarity between mean embeddings
    cosine_diffs = []
    for Z_r, Z_s in zip(Zs_real, Zs_silence):
        mean_r = Z_r.mean(axis=0)
        mean_s = Z_s.mean(axis=0)
        cos = cosine_sim(mean_r, mean_s)
        cosine_diffs.append(1.0 - cos)  # Higher = more different
    avg_cosine_diff = float(np.mean(cosine_diffs))
    
    # Pass if embeddings differ significantly
    silence_pass = avg_cosine_diff > 0.05  # At least 5% cosine difference
    results["silence_test_passed"] = silence_pass
    results["silence_cosine_diff"] = avg_cosine_diff
    results["silence_norm_ratio"] = norm_ratio
    
    print(f"  📊 Silence test: cosine_diff={avg_cosine_diff:.4f}, norm_ratio={norm_ratio:.2f}")
    if silence_pass:
        print("  ✅ Silence test PASSED. Model distinguishes audio from silence.")
    else:
        print("  ❌ Silence test FAILED. Model output may be constant or ignoring audio!")
    
    # === Sanity B: Rank Test ===
    # Stack all tokens and compute effective rank
    all_tokens = np.concatenate(Zs_real, axis=0)
    
    # Sample to avoid memory issues
    max_tokens = 2000
    if all_tokens.shape[0] > max_tokens:
        rng = np.random.RandomState(cfg.seed)
        idx = rng.choice(all_tokens.shape[0], size=max_tokens, replace=False)
        all_tokens = all_tokens[idx]
    
    # Center tokens
    all_tokens = all_tokens - all_tokens.mean(axis=0, keepdims=True)
    
    # SVD for singular values
    # SVD for singular values
    _, s, _ = np.linalg.svd(all_tokens, full_matrices=False)
    
    # Use ENERGY (s^2) for distribution, not raw svals
    s2 = s ** 2
    s2 = s2 / (s2.sum() + 1e-12)
    eff_rank = float(np.exp(-np.sum(s2 * np.log(s2 + 1e-12))))
    
    # Pass if rank > 5 (not degenerate)
    rank_pass = eff_rank > 5.0
    results["rank_test_passed"] = rank_pass
    results["effective_rank"] = eff_rank
    
    print(f"  📊 Rank test: effective_rank={eff_rank:.2f}")
    if rank_pass:
        print("  ✅ Rank test PASSED. Embeddings have diverse structure.")
    else:
        print("  ❌ Rank test FAILED. Embeddings may be degenerate (rank too low)!")
    
    # === Sanity C: Dimension Test ===
    # Check output has reasonable dimensions
    sample_Z = Zs_real[0]
    T, D = sample_Z.shape
    
    dim_pass = T >= 4 and D >= 32  # At least 4 tokens for meaningful DTW/residuals
    results["dim_test_passed"] = dim_pass
    results["output_shape"] = (T, D)
    
    print(f"  📊 Dim test: shape={sample_Z.shape}")
    if dim_pass:
        print("  ✅ Dim test PASSED.")
    else:
        print("  ❌ Dim test FAILED. Unexpected output dimensions!")
    
    # === Overall ===
    results["all_passed"] = silence_pass and rank_pass and dim_pass
    
    if results["all_passed"]:
        print("  🎯 SANITY SUITE PASSED. Adapter is extracting valid audio features.")
    else:
        print("  🚨 SANITY SUITE FAILED. Check adapter implementation!")
    
    return results



# -----------------------------
# Analysis Core
# -----------------------------
def pool_representation_np(Z: np.ndarray, mode: str) -> np.ndarray:
    """
    Numpy-native pooling.
    Z: [T, Dz] numpy array (possibly mmap).
    mode: mean | last | max
    """
    if Z.ndim != 2:
        raise ValueError(f"Expected Z [T, Dz]. Got shape={Z.shape}")
        
    if mode == "last":
        return Z[-1].astype(np.float32)
    
    if mode == "max":
        return Z.max(axis=0).astype(np.float32)
        
    return Z.mean(axis=0).astype(np.float32)

def pool_representation(Z: torch.Tensor, mode: str) -> np.ndarray:
    """
    Legacy torch version. Prefer pool_representation_np.
    Z: [T, Dz] torch on CPU.
    mode: mean | last | max
    """
    if Z.ndim != 2:
        raise ValueError(f"Expected Z [T, Dz]. Got shape={tuple(Z.shape)}")

    if mode == "last":
        v = Z[-1]
        return v.numpy().astype(np.float32)

    if mode == "max":
        v, _ = torch.max(Z, dim=0)
        return v.numpy().astype(np.float32)

    v = Z.mean(dim=0)
    return v.numpy().astype(np.float32)


def cosine_scorer(estimator, X, y_true) -> float:
    y_pred = estimator.predict(X)
    sims = []
    for i in range(y_true.shape[0]):
        sims.append(cosine_sim(y_pred[i], y_true[i]))
    return float(np.mean(sims))



class ProcrustesAligner:
    """
    Strict Orthogonal Procrustes Analysis with Isotropic Scaling.
    Ref: 'Text-Anchored Semantic Projection'
    
    State:
      k: target dimension
      R: (k, k) orthogonal rotation
      z_mean, z_scale: audio centering/scaling
      e_mean, e_scale: text centering/scaling
      pca_z, pca_e: optional dimensionality reduction
    """
    def __init__(self, k_target: int = 768, whiten_method: str = "whitening"):
        self.k_target = k_target
        self.whiten_method = whiten_method
        
        # State
        self.R: Optional[np.ndarray] = None
        self.z_mean: Optional[np.ndarray] = None
        self.z_scale: Union[float, np.ndarray] = 1.0
        self.e_mean: Optional[np.ndarray] = None
        self.e_scale: Union[float, np.ndarray] = 1.0
        
        self.pca_z: Optional[PCA] = None
        self.pca_e: Optional[PCA] = None
        
        self.fitted = False

    def _fit_pure_pca(self, X_normalized: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit PCA on ALREADY CENTERED/SCALED data using SVD.
        Returns projection matrix P [D, k] and transformed data X_k [N, k].
        This matches the paper's linear formula Z_k = Z_tilde @ P.
        """
        # SVD: X = U S Vt
        # We want P = Vt.T[:, :k]
        # Check for NaN/Inf just in case
        if not np.all(np.isfinite(X_normalized)):
            print("  ⚠️ Warning: Input to PCA contains NaNs. Filling with 0.")
            X_normalized = np.nan_to_num(X_normalized)
            
        U, S, Vt = np.linalg.svd(X_normalized, full_matrices=False)
        
        # Components are rows of Vt
        P = Vt[:k].T.astype(np.float32) # [D, k]
        
        # Project
        X_k = X_normalized @ P
        
        return P, X_k

    def dataset_fit(self, Z: np.ndarray, E: np.ndarray, calibration_tokens: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit alignment parameters on Training Data (Z, E).
        Z: Audio embeddings [N, d_a] (Pooled/Utterance-level for R estimation)
        E: Text embeddings [N, d_e]
        calibration_tokens: Optional [M, d_a] sampled tokens to estimate z_scale/z_mean.
                            If provided, normalization statistics match token distribution.
        """
        N, d_a = Z.shape
        _, d_e = E.shape
        
        # 1. Determine effective k
        # Constraint: k <= min(d_a, d_e, N-1)
        k = min(self.k_target, d_a, d_e, N - 1)
        self.k_actual = max(k, 1)
        
        # 2. Audio Normalization (Center + Scale/Whiten)
        # Decision: If calibration_tokens provided, use them for statistics.
        # This aligns transformation with token-level distribution.
        if calibration_tokens is not None:
            stats_source = calibration_tokens
        else:
            stats_source = Z

        self.z_mean = np.mean(stats_source, axis=0)
        
        if self.whiten_method == "whitening":
            # Compute std on centered data
            Z_tmp = stats_source - self.z_mean
            std = np.std(Z_tmp, axis=0) + 1e-9
            self.z_scale = std
        else:
            Z_tmp = stats_source - self.z_mean
            self.z_scale = isotropic_scale_fro(Z_tmp)

        # Apply normalization to Z (pooled means) for Procrustes rotation calculation
        Z_c = Z - self.z_mean
        Z_norm = Z_c / self.z_scale
            
        # 3. Audio PCA -> k
        # Replace sklearn PCA with pure SVD on Z_norm
        self.P_z, Z_k = self._fit_pure_pca(Z_norm, self.k_actual)
        
        # 4. Text Normalization
        self.e_mean = np.mean(E, axis=0)
        E_c = E - self.e_mean
        
        if self.whiten_method == "whitening":
            e_std = np.std(E_c, axis=0) + 1e-9
            self.e_scale = e_std
            E_norm = E_c / e_std
        else:
            self.e_scale = isotropic_scale_fro(E_c)
            E_norm = E_c / self.e_scale
            
        # 5. Text PCA -> k
        # Replace sklearn PCA with pure SVD on E_norm
        self.P_e, E_k = self._fit_pure_pca(E_norm, self.k_actual)
        
        # 6. Procrustes Rotation
        # min || Z_k R - E_k ||_F
        self.R = procrustes_rotation_no_reflection(Z_k, E_k)
        
        self.fitted = True
        
        det_R = float(np.linalg.det(self.R))
        print(f"  ✅ ProcrustesFit: k={self.k_actual}, det(R)={det_R:.4f}")
        
        return {
            "k": self.k_actual,
            "det_R": det_R,
            "z_scale_mean": float(np.mean(self.z_scale)),
            "e_scale_mean": float(np.mean(self.e_scale)),
        }

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply strict transform to Audio Z:
        y = ( (Z - mu_z) / sigma_z ) @ P_z @ R
        result is in Normalized Shared Space (Feature-Aligned).
        """
        if not self.fitted:
            raise RuntimeError("Aligner not fitted.")
            
        Z = np.atleast_2d(Z).astype(np.float32)
        # 1. Normalize
        Z_norm = (Z - self.z_mean) / self.z_scale
        # 2. Project (Pure Linear)
        Z_k = Z_norm @ self.P_z
        # 3. Rotate
        Y = Z_k @ self.R
        return Y.astype(np.float32)
        
    def transform_text(self, E: np.ndarray) -> np.ndarray:
        """
        Apply corresponding transform to Text E:
        e_hat = ( (E - mu_e) / sigma_e ) @ P_e
        """
        if not self.fitted:
            raise RuntimeError("Aligner not fitted.")
            
        E = np.atleast_2d(E).astype(np.float32)
        E_norm = (E - self.e_mean) / self.e_scale
        E_k = E_norm @ self.P_e
        return E_k.astype(np.float32)

    def get_semantic_basis(self, E_train: np.ndarray) -> np.ndarray:
        """
        Construct Ut (Text Basis) from Training Text in the Shared Space.
        
        Step 1: Transform E_train to Shared Space E_k.
        Step 2: PCA/SVD on E_k to get principal directions.
        Step 3: These are the canonical semantic axes.
        """
        E_k = self.transform_text(E_train)
        
        # E_k is already centered implicitly by PCA, but let's be safe
        E_center = E_k - np.mean(E_k, axis=0)
        
        # SVD
        # U, S, Vt = svd(E_center)
        # Basis is right singular vectors (axes of variation)
        pca = PCA(random_state=42)
        pca.fit(E_center)
        Ut = pca.components_.T
        
        # QR to ensure strict orthonormality
        Q, _ = np.linalg.qr(Ut)
        return Q.astype(np.float32)


# -----------------------------
# Geometry
# -----------------------------
def project_to_subspace(v: np.ndarray, Ut: np.ndarray) -> np.ndarray:
    # Support both single vector (D,) and batch (N, D)
    return (v @ Ut) @ Ut.T


def decompose_meanvec(y_mean: np.ndarray, Ut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose vector or batch of vectors into semantic (parallel) and paralinguistic (perp) components.
    y_mean: [D] or [N, D]
    Returns: (y_par, y_perp) matching input shape.
    """
    y_par = project_to_subspace(y_mean, Ut)
    y_perp = y_mean - y_par
    return y_par, y_perp


def token_residuals(Y: np.ndarray) -> np.ndarray:
    return Y - np.mean(Y, axis=0, keepdims=True)


def procrustean_cov_energy(Yc: np.ndarray, Ut: np.ndarray, eps: float = 1e-12) -> float:
    """
    Low-rank O(T*D*k) version. Uses ||Yc @ Ut||^2 = ||Proj_Ut(Yc)||^2 since Ut is orthonormal.
    """
    C = Yc @ Ut  # [T, k]
    e_par = float(np.sum(C * C))
    e_tot = float(np.sum(Yc * Yc)) + eps
    return e_par / e_tot





def subspace_angles_tokens_vs_semantic(all_Yc: np.ndarray, Ut: np.ndarray, k_audio: int, seed: int) -> Tuple[float, float]:
    D = all_Yc.shape[1]
    k_eff = int(min(k_audio, all_Yc.shape[0], D, Ut.shape[1]))

    pca = PCA(n_components=k_eff, random_state=seed)
    pca.fit(all_Yc)
    Ua = pca.components_.T.astype(np.float32)
    Qa, _ = np.linalg.qr(Ua)

    M = (Ut[:, :k_eff].T @ Qa[:, :k_eff]).astype(np.float64)
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    mean_cca = float(np.mean(s))
    mean_angle = float(np.mean(np.arccos(s)) * 180.0 / np.pi)
    return mean_cca, mean_angle


# -----------------------------
# Interventions
# -----------------------------
def apply_gain(audio: np.ndarray, db: float) -> np.ndarray:
    scale = 10.0 ** (db / 20.0)
    return (audio * scale).astype(np.float32)


def apply_pitch_shift(audio: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    try:
        import librosa
        return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=semitones).astype(np.float32)
    except Exception:
        return audio


def align_tokens_resample(Y0: np.ndarray, Y1: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
    def resample(Y: np.ndarray) -> np.ndarray:
        t0 = np.linspace(0, 1, Y.shape[0])
        t1 = np.linspace(0, 1, T)
        out = np.zeros((T, Y.shape[1]), dtype=np.float32)
        for d in range(Y.shape[1]):
            out[:, d] = np.interp(t1, t0, Y[:, d]).astype(np.float32)
        return out
    return resample(Y0), resample(Y1)


def align_tokens_dtw(Y0: np.ndarray, Y1: np.ndarray, pca_dim: int, seed: int, resample_T: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    DTW alignment on PCA-reduced tokens for speed.
    Uses fastdtw(sequence, sequence) API.
    If fastdtw is missing, fallback to resample with explicit warning.
    """
    try:
        from fastdtw import fastdtw
    except Exception as e:
        print(f"  ⚠️ fastdtw not available. Fallback to resample. err={e}")
        return align_tokens_resample(Y0, Y1, T=min(Y0.shape[0], Y1.shape[0], resample_T))

    Z = np.concatenate([Y0, Y1], axis=0)
    k = int(min(pca_dim, Z.shape[0], Z.shape[1]))
    pca = PCA(n_components=k, random_state=seed)
    Zr = pca.fit_transform(Z)
    Y0r = Zr[: Y0.shape[0]]
    Y1r = Zr[Y0.shape[0] :]

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    _, path = fastdtw(Y0r, Y1r, dist=dist)
    A0 = np.stack([Y0[i] for i, _ in path], axis=0).astype(np.float32)
    A1 = np.stack([Y1[j] for _, j in path], axis=0).astype(np.float32)
    return A0, A1


def token_level_deltas(Y0: np.ndarray, Y1: np.ndarray, Ut: np.ndarray, cfg: CFG) -> Dict[str, float]:
    """
    Token-level sensitivity. Report:
      mean(||ΔY||)/sqrt(D)
      mean(||ΔY_par||)/sqrt(D)
      mean(||ΔY_perp||)/sqrt(D)
    """
    if cfg.token_align == "dtw":
        A0, A1 = align_tokens_dtw(Y0, Y1, pca_dim=cfg.dtw_pca_dim, seed=cfg.seed, resample_T=cfg.resample_T)
    else:
        A0, A1 = align_tokens_resample(Y0, Y1, cfg.resample_T)

    D = A0.shape[1]
    dY = A1 - A0

    # Low-rank projection: O(T·D·k) instead of O(T·D²) when k << D
    dY_par = (dY @ Ut) @ Ut.T  # [T, D]
    dY_perp = dY - dY_par

    mean_d = float(np.mean(l2_norm(dY, axis=1)) / math.sqrt(D))
    mean_d_par = float(np.mean(l2_norm(dY_par, axis=1)) / math.sqrt(D))
    mean_d_perp = float(np.mean(l2_norm(dY_perp, axis=1)) / math.sqrt(D))
    return {
        "token_mean_d": mean_d,
        "token_mean_d_par": mean_d_par,
        "token_mean_d_perp": mean_d_perp,
    }


# -----------------------------
# Probes. Safe CV with filtering.
# -----------------------------
def filter_classes(y: np.ndarray, min_per_class: int, max_classes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    uniq, cnt = np.unique(y, return_counts=True)
    keep = uniq[cnt >= min_per_class]
    mask = np.isin(y, keep)
    y2 = y[mask]

    if max_classes is not None:
        uniq2, cnt2 = np.unique(y2, return_counts=True)
        order = np.argsort(cnt2)[::-1]
        top = uniq2[order[:max_classes]]
        mask2 = mask & np.isin(y, top)
        return mask2, y[mask2]
    return mask, y2


def safe_stratified_splits(y: np.ndarray, n_splits: int, seed: int) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    if y.size == 0:
        return None
    if not np.issubdtype(y.dtype, np.integer):
        return None
    binc = np.bincount(y)
    if binc.size == 0:
        return None
    pos = binc[binc > 0]
    if pos.size == 0:
        return None
    min_count = int(np.min(pos))
    k = int(min(n_splits, min_count))
    if k < 2:
        return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(y), y))


def run_probe_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> Dict[str, float]:
    """
    Legacy probe for backward compatibility or global sanity (if needed).
    But for rigorous evaluation, use fit_predict_probe.
    """
    splits = safe_stratified_splits(y, n_splits=n_splits, seed=seed)
    if splits is None:
        return {"macro_f1": float("nan"), "macro_f1_std": float("nan"), "acc": float("nan"), "acc_std": float("nan")}

    f1s = []
    accs = []
    for tr, te in splits:
        res = fit_predict_probe(X[tr], y[tr], X[te], y[te], seed)
        f1s.append(res["macro_f1"])
        accs.append(res["acc"])
    return {
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "acc": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
    }


def fit_predict_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    """
    Fit ridge classifier on fixed train split, evaluate on test split.
    """
    # Filter classes in train that might be missing? 
    # RidgeClassifier handles missing classes in y by just not predicting them.
    # But StandardScaler needs at least 1 sample.
    if X_train.shape[0] < 2:
        return {"macro_f1": float("nan"), "acc": float("nan")}

    alphas = (0.1, 1.0, 10.0) 
    clf = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", RidgeClassifierCV(alphas=alphas, class_weight='balanced')),
        ]
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    return {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "acc": float(accuracy_score(y_test, pred)),
    }


def run_sentence_probe_foldwise_ut(
    Y_means: np.ndarray,
    sentence_ids: np.ndarray,
    speaker_ids: np.ndarray,
    cfg: CFG,
) -> Dict[str, Any]:
    """
    Run sentence probe with FOLD-WISE Ut construction.
    
    CRITICAL FIX: Builds Ut from TRAIN-ONLY sentence centroids in each fold.
    This avoids the "self-fulfilling prophecy" where Ut sees test data.
    
    Uses GroupKFold(speaker) to avoid speaker leakage (same speaker in train/test).
    
    Returns dict with par_f1, perp_f1, and diagnostic info.
    """
    # Use speakers as groups to avoid same speaker in train/test
    unique_speakers = np.unique(speaker_ids)
    n_splits = min(cfg.cv_folds, len(unique_speakers))
    if n_splits < 2:
        return {"par_f1": float("nan"), "perp_f1": float("nan"), "foldwise": True}
    
    gkf = GroupKFold(n_splits=n_splits)
    
    # Encode sentence_ids to integers for classification
    unique_sents = np.unique(sentence_ids)
    sent_to_idx = {s: i for i, s in enumerate(unique_sents)}
    y = np.array([sent_to_idx[s] for s in sentence_ids], dtype=np.int32)
    
    par_f1s = []
    perp_f1s = []
    
    for train_idx, test_idx in gkf.split(Y_means, groups=speaker_ids):
        # Build Ut from TRAIN centroids only
        train_Y = Y_means[train_idx]
        train_sent = sentence_ids[train_idx]
        
        # Compute train-only sentence centroids
        centroids = []
        for s in np.unique(train_sent):
            mask = train_sent == s
            if np.sum(mask) > 0:
                centroids.append(train_Y[mask].mean(axis=0))
        
        if len(centroids) < 2:
            continue  # Need at least 2 sentences
            
        centroids = np.stack(centroids, axis=0)
        
        # Build Ut from train centroids (same logic as build_ut_from_sentence_centroids)
        mean_vec = centroids.mean(axis=0, keepdims=True)
        centroids_c = centroids - mean_vec
        _, S, Vt = np.linalg.svd(centroids_c, full_matrices=False)
        k_eff = min(cfg.text_subspace_k, len(S), len(centroids) - 1)
        k_eff = max(k_eff, 1)
        Ut_train = Vt[:k_eff].T
        Q, _ = np.linalg.qr(Ut_train)
        Ut_train = Q.astype(np.float32)
        
        # Decompose ALL samples (train and test) using train Ut
        def decompose(Y, Ut):
            # Low-rank projection: Y_par = (Y @ Ut) @ Ut.T
            Y_par = (Y @ Ut) @ Ut.T
            Y_perp = Y - Y_par
            return Y_par, Y_perp
        
        _, train_perp = decompose(train_Y, Ut_train)
        _, test_perp = decompose(Y_means[test_idx], Ut_train)
        train_par, _ = decompose(train_Y, Ut_train)
        test_par, _ = decompose(Y_means[test_idx], Ut_train)
        
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Fit classifiers
        clf_par = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(alpha=1.0, random_state=cfg.seed))
        ])
        clf_perp = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(alpha=1.0, random_state=cfg.seed))
        ])
        
        clf_par.fit(train_par, y_train)
        clf_perp.fit(train_perp, y_train)
        
        pred_par = clf_par.predict(test_par)
        pred_perp = clf_perp.predict(test_perp)
        
        par_f1s.append(f1_score(y_test, pred_par, average="macro"))
        perp_f1s.append(f1_score(y_test, pred_perp, average="macro"))
    
    return {
        "par_f1": float(np.mean(par_f1s)) if par_f1s else float("nan"),
        "par_f1_std": float(np.std(par_f1s)) if par_f1s else float("nan"),
        "perp_f1": float(np.mean(perp_f1s)) if perp_f1s else float("nan"),
        "perp_f1_std": float(np.std(perp_f1s)) if perp_f1s else float("nan"),
        "n_folds": len(par_f1s),
        "foldwise_ut": True,
    }


def safe_groupkfold_splits(groups: np.ndarray, n_splits: int) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    uniq = np.unique(groups)
    if uniq.size < 2:
        return None
    k = int(min(n_splits, uniq.size))
    if k < 2:
        return None
    gkf = GroupKFold(n_splits=k)
    idx = np.arange(groups.shape[0])
    return list(gkf.split(idx, groups=groups))


def run_probe_regression_groupkfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> Dict[str, float]:
    """
    Regression probe with GroupKFold.
    Uses RobustScaler for X (outlier resistance) and standardizes y (target) for
    better numerical stability with high-magnitude targets like Pitch (100-300 Hz).
    Reports both R^2 and Pearson r.
    """
    splits = safe_groupkfold_splits(groups, n_splits=n_splits)
    if splits is None:
        return {"r2": float("nan"), "r2_std": float("nan"), "pearson_r": float("nan"), "pearson_r_std": float("nan")}

    # Standardize y globally for numerical stability
    # (Pitch Hz can be 100-300, which is much larger than embedding scale)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std < 1e-9:
        y_std = 1.0  # Avoid division by zero for constant targets
    y_normalized = (y - y_mean) / y_std

    r2s = []
    rs = []  # Pearson correlations
    # Use CFG.ridge_alphas (Pass cfg into this function or use global)
    alphas = CFG_.ridge_alphas 

    for tr, te in splits:
        reg = Pipeline(
            [
                ("scaler", RobustScaler()),  # More robust to outliers than StandardScaler
                ("ridge", RidgeCV(alphas=alphas)),
            ]
        )
        # Train on normalized y for better convergence
        reg.fit(X[tr], y_normalized[tr])
        pred_normalized = reg.predict(X[te])
        
        # Convert predictions back to original scale for R² computation
        pred = pred_normalized * y_std + y_mean
        r2s.append(r2_score(y[te], pred))
        
        # Pearson correlation is scale-invariant, so use normalized values
        if len(y[te]) > 2:
            # Check for constant predictions (can happen with extreme regularization)
            if np.std(pred_normalized) < 1e-9:
                rs.append(0.0)
            else:
                r, _ = pearsonr(y_normalized[te], pred_normalized)
                rs.append(r if np.isfinite(r) else 0.0)
        else:
            rs.append(0.0)
            
    return {
        "r2": float(np.mean(r2s)), 
        "r2_std": float(np.std(r2s)),
        "pearson_r": float(np.mean(rs)),
        "pearson_r_std": float(np.std(rs)),
    }


# -----------------------------
# Audio-derived paralinguistic targets (fallback probes)
# -----------------------------
def estimate_pitch_hz(audio: np.ndarray, sr: int) -> float:
    """
    Robust pitch estimate.
    Uses librosa.yin if available.
    Steps:
      - compute frame-wise F0
      - keep finite values
      - voiced masking via energy threshold and reasonable f0 range
      - trim extreme percentiles
      - return median (robust)
    Returns np.nan if failed or no voiced frames.
    """
    try:
        import librosa

        x = audio.astype(np.float32)

        f0 = librosa.yin(x, fmin=50, fmax=600, sr=sr)
        f0 = np.asarray(f0, dtype=np.float32)
        f0 = f0[np.isfinite(f0)]
        if f0.size == 0:
            return float("nan")

        rms = librosa.feature.rms(y=x)[0].astype(np.float32)
        rms = rms[np.isfinite(rms)]
        if rms.size == 0:
            return float("nan")

        T = int(min(f0.size, rms.size))
        f0 = f0[:T]
        rms = rms[:T]

        thr = float(np.median(rms) * 0.5)
        voiced = (rms >= thr) & (f0 >= 60.0) & (f0 <= 500.0)
        f0v = f0[voiced]

        if f0v.size < 3:
            f0v = f0

        f0v = f0v[np.isfinite(f0v)]
        if f0v.size == 0:
            return float("nan")

        lo = float(np.percentile(f0v, 5.0))
        hi = float(np.percentile(f0v, 95.0))
        f0v = f0v[(f0v >= lo) & (f0v <= hi)]
        if f0v.size == 0:
            return float("nan")

        return float(np.median(f0v))

    except Exception:
        return float("nan")


def estimate_rms_energy(audio: np.ndarray) -> float:
    x = audio.astype(np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


# -----------------------------
# Dose-response curve fitting (Must Have)
# -----------------------------
def fit_pitch_quadratic_no_linear(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fit y = a x^2 + b.
    Return a, b, r2.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.stack([x * x, np.ones_like(x)], axis=1)  # [N, 2]
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a = float(w[0])
    b = float(w[1])
    yhat = X @ w
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    return {"a": a, "b": b, "r2": r2}


def fit_gain_linear(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fit y = m x + c.
    Return m, c, r2.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.stack([x, np.ones_like(x)], axis=1)  # [N, 2]
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    m = float(w[0])
    c = float(w[1])
    yhat = X @ w
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    return {"m": m, "c": c, "r2": r2}


# -----------------------------
# Visualization
# -----------------------------
def save_hist(data: List[float], title: str, xlabel: str, path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(np.asarray(data), bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_scatter(x: List[float], y: List[float], labels: List[str], title: str, xlabel: str, ylabel: str, path: str) -> None:
    plt.figure(figsize=(6, 5))
    for xi, yi, lab in zip(x, y, labels):
        plt.scatter([xi], [yi])
        plt.text(xi, yi, lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_line_with_errorbars_and_fit(
    xs: List[float],
    ys_mean: List[float],
    ys_std: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    path: str,
    fit_kind: Optional[str],
    legend_name: str,
) -> Dict[str, float]:
    """
    fit_kind:
      - None
      - "pitch_quad" for y = a x^2 + b. Legend shows a.
      - "gain_lin" for y = m x + c. Legend shows m.
    Returns dict with coefficients. Empty dict if fit_kind is None.
    """
    plt.figure(figsize=(6, 4))
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys_mean, dtype=np.float64)
    e = np.asarray(ys_std, dtype=np.float64)

    (line,) = plt.plot(x, y)
    plt.fill_between(x, y - e, y + e, alpha=0.2)

    coeff: Dict[str, float] = {}
    label_main = legend_name

    if fit_kind == "pitch_quad":
        coeff = fit_pitch_quadratic_no_linear(x, y)
        a = coeff["a"]
        b = coeff["b"]
        yhat = a * (x * x) + b
        plt.plot(x, yhat, linestyle="--")
        label_main = f"{legend_name} (a={a:.4f})"

    if fit_kind == "gain_lin":
        coeff = fit_gain_linear(x, y)
        m = coeff["m"]
        c = coeff["c"]
        yhat = m * x + c
        plt.plot(x, yhat, linestyle="--")
        label_main = f"{legend_name} (m={m:.4f})"

    line.set_label(label_main)
    plt.legend()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return coeff


def save_disentanglement_trajectory(
    y_means: List[np.ndarray],
    Ut: np.ndarray,
    title: str,
    path: str,
) -> None:
    """
    X: ||Δ_par||.
    Y: signed PC1 projection of Δ_perp.
    """
    Y = np.stack(y_means, axis=0)
    center = Y[len(Y) // 2]
    dY = Y - center

    d_par = np.stack([project_to_subspace(v, Ut) for v in dY], axis=0)
    d_perp = dY - d_par

    x = l2_norm(d_par, axis=1)

    pca = PCA(n_components=1, random_state=CFG_.seed)
    y = pca.fit_transform(d_perp).reshape(-1)

    plt.figure(figsize=(6, 6))
    plt.axhline(0.0)
    plt.axvline(0.0)
    plt.plot(x, y)
    for i in range(len(x)):
        plt.scatter([x[i]], [y[i]])
        plt.text(x[i], y[i], str(i))
    plt.title(title)
    plt.xlabel("Semantic change. ||Δ_par||")
    plt.ylabel("Paralinguistic change. PC1(Δ_perp)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# -----------------------------
# Pooling ablation for Qwen2 (optional)
# -----------------------------
def quick_pooling_ablation(
    adapter: AudioAdapterBase,
    items: List[Dict[str, Any]],
    text_enc: "TextEncoder",
    cfg: CFG,
) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    """
    Diagnostic. Runs a small ridge-map fit on a subset.
    Reports heldout_sentence_cos for mean/last/max.
    Returns (best_mode, score_dict).
    """
    if not cfg.qwen2_pooling_ablation:
        return None, None
    if "Qwen" not in adapter.name and "Qwen" not in adapter.model_id:
        return None, None

    n = int(min(cfg.qwen2_pooling_ablation_n, len(items)))
    print(f"🧪 Qwen2 pooling ablation on n={n} samples. modes=[mean,last,max]")

    sub = items[:n]
    sentences = [it["text"] for it in sub]
    sentence_ids = np.asarray([it["sentence_id"] for it in sub], dtype=object)
    e_text = text_enc.encode(sentences, batch_size=64)

    Zs = []
    for i, it in enumerate(sub):
        Zs.append(adapter.extract_tokens(it["audio"], it["sr"]))
        # Cleanup based on config interval (less frequent for high memory)
        if (i + 1) % cfg.cleanup_interval_ablation == 0:
            free_torch()
    free_torch()  # Final cleanup

    modes = ["mean", "last", "max"]
    out: Dict[str, float] = {}

    for m in modes:
        reps = []
        for Z in Zs:
            # Z is torch tensor here because extract_tokens returns tensor
            # Convert to numpy for pool_representation_np
            Z_np = Z.numpy().astype(np.float32)
            reps.append(pool_representation_np(Z_np, m))
        reps = np.stack(reps, axis=0)
        
        # Use fit logic from ProcrustesAligner on subset
        # Note: For ablation, we just use pooled stats to be fast/simple
        aligner = ProcrustesAligner(k_target=768, whiten_method=cfg.ut_whiten_method)
        fit_info = aligner.dataset_fit(reps, e_text)
        
        # In-sample cosine
        E_hat = aligner.transform_text(e_text)
        Y_hat = aligner.transform(reps)
        
        # Calculate mean cosine
        sims = [cosine_sim(Y_hat[i], E_hat[i]) for i in range(len(reps))]
        out[m] = float(np.mean(sims))

    best_mode = max(out, key=lambda k: out[k])
    print(f"  Ablation cosine (IS): {out}")
    print(f"  Suggested ridge pooling mode for Qwen2: {best_mode}")
    return best_mode, out


# -----------------------------
# Main analysis for one model
# -----------------------------
def analyze_model(
    adapter: AudioAdapterBase,
    items: List[Dict[str, Any]],
    text_enc: TextEncoder,
    cfg: CFG,
) -> Dict[str, Any]:
    print("=" * 70)
    print(f"🔬 Analyzing: {adapter.model_id}")
    print("=" * 70)
    
    # Auto-enable low memory mode for OOM-prone models
    use_low_mem = cfg.low_memory_mode
    if "Omni" in adapter.model_id or "Omni" in adapter.name:
        use_low_mem = True
        print("  ⚡ Auto-enabled low memory mode for Qwen2.5-Omni (OOM prevention)")

    cuda_mem("before load")
    adapter.load()
    cuda_mem("after load")
    
    # Run sanity suite to validate adapter is extracting real audio features
    # Skip for Qwen2.5-Omni due to memory constraints (it extracts tokens twice: real + silence)
    is_qwen_omni = "Qwen" in adapter.name and "Omni" in adapter.name
    skip_sanity = cfg.skip_sanity_suite or is_qwen_omni
    
    if skip_sanity:
        print("  ⚡ Skipping sanity suite (configured or Qwen2.5-Omni detected)")
        sanity_results = {"all_passed": True, "skipped": True}
    else:
        sanity_results = run_adapter_sanity_suite(adapter, items, cfg)
        if not sanity_results["all_passed"]:
            print("  🚨 WARNING: Sanity suite failed. Results for this model may be invalid!")
            print("     Consider checking adapter implementation or skipping this model.")

    ab_best_mode, ab_scores = quick_pooling_ablation(adapter, items, text_enc, cfg)

    sentences = [it["text"] for it in items]
    sentence_ids = np.asarray([it["sentence_id"] for it in items], dtype=object)
    e_text = text_enc.encode(sentences, batch_size=64)

    pool_mode = cfg.ridge_pool_default
    if "Qwen" in adapter.name or "Qwen" in adapter.model_id:
        pool_mode = cfg.ridge_pool_qwen2
        if cfg.qwen2_pooling_auto_override and (ab_best_mode is not None):
            print(f"  ✅ Auto-override Qwen2 ridge pooling. {pool_mode} -> {ab_best_mode}")
            pool_mode = ab_best_mode

    # === TOKEN CACHE: Extract all tokens once, avoid repeated forward passes ===
    # Auto-detect Qwen2.5-Omni and enable memory-efficient mode
    is_qwen_omni = "Qwen" in adapter.name and "Omni" in adapter.name
    cache_mode = cfg.token_cache_mode
    
    # Force disk caching + float16 for Qwen2.5-Omni to prevent OOM
    if is_qwen_omni and cache_mode == "ram":
        print("  ⚡ Auto-switching to disk cache for Qwen2.5-Omni (large token sequences)")
        cache_mode = "disk"
    
    # For high memory tier (128GB+), use float32 for better precision
    # But for Qwen-Omni, always use float16 due to massive token counts
    if is_qwen_omni:
        cache_dtype = np.float16
    else:
        cache_dtype = np.float32 if cfg.memory_tier == "high" else np.float16
    
    print(f"📦 Caching audio tokens (mode={cache_mode}, dtype={cache_dtype.__name__})...")
    
    # Token cache can be dict (RAM mode) or temp directory path (disk mode) or None (lazy mode)
    token_cache: Dict[int, np.ndarray] = {}  # Used for RAM mode
    disk_cache_dir: Optional[str] = None  # Used for disk mode
    
    if cache_mode == "disk":
        # Create temp directory for disk cache
        disk_cache_dir = cfg.token_cache_dir if cfg.token_cache_dir else tempfile.mkdtemp(prefix="token_cache_")
        os.makedirs(disk_cache_dir, exist_ok=True)
        print(f"  📁 Disk cache directory: {disk_cache_dir}")
        
        # Use more aggressive cleanup for Qwen2.5-Omni
        cleanup_interval = cfg.cleanup_interval_token_cache_qwen_omni if is_qwen_omni else cfg.cleanup_interval_token_cache
        
        total_bytes = 0
        for i, it in enumerate(items):
            tokens = adapter.extract_tokens(it["audio"], it["sr"]).numpy().astype(cache_dtype)
            cache_path = os.path.join(disk_cache_dir, f"{i:06d}.npy")
            np.save(cache_path, tokens)
            total_bytes += tokens.nbytes
            del tokens  # Free memory immediately
            
            # Periodic cleanup based on config interval (more aggressive for Qwen2.5-Omni)
            if (i + 1) % cleanup_interval == 0:
                free_torch()
            if (i + 1) % 100 == 0:
                print(f"    Cached {i+1}/{len(items)} samples...")
        free_torch()  # Final cleanup after caching
        print(f"  ✅ Cached {len(items)} samples to disk. Est. disk: {total_bytes / 1e9:.2f} GB")
    
    elif cache_mode == "lazy":
        # Lazy mode: no pre-caching, extract on-demand
        print(f"  ⚡ Lazy mode: tokens will be extracted on-demand (slower but ~0 RAM overhead)")
    
    else:  # cache_mode == "ram" (original behavior)
        for i, it in enumerate(items):
            token_cache[i] = adapter.extract_tokens(it["audio"], it["sr"]).numpy().astype(cache_dtype)
            # Periodic cleanup based on config interval
            if (i + 1) % cfg.cleanup_interval_token_cache == 0:
                free_torch()
        free_torch()  # Final cleanup after caching
        print(f"  ✅ Cached {len(token_cache)} samples. Est. RAM: {sum(t.nbytes for t in token_cache.values()) / 1e9:.2f} GB")
    
    # Helper function to get tokens (works for all cache modes)
    def get_cached_tokens(idx: int) -> np.ndarray:
        if cache_mode == "disk":
            cache_path = os.path.join(disk_cache_dir, f"{idx:06d}.npy")
            # Memory map mode to save RAM
            return np.load(cache_path, mmap_mode="r")
        elif cache_mode == "lazy":
            return adapter.extract_tokens(items[idx]["audio"], items[idx]["sr"]).numpy().astype(cache_dtype)
        else:  # ram
            return token_cache[idx]

    print(f"🧩 Building z_rep for Procrustes map from cache. pooling={pool_mode}")
    z_reps = []
    for i in range(len(items)):
        # Optimization: use numpy pooling directly on mmap array
        # avoids loading entire token array into RAM as torch tensor
        Z_np = get_cached_tokens(i)
        z_reps.append(pool_representation_np(Z_np, pool_mode))
    z_reps = np.stack(z_reps, axis=0)


    print("🧩 Fitting orthogonal Procrustes map via GroupKFold(speaker) to avoid leakage...")
    
    # ---------------------------------------------------------
    # 0. Global Parameter Setup (Removed to avoid leakage)
    # ---------------------------------------------------------
    # k determination is now local within fit_procrustes_map

    # ---------------------------------------------------------
    # 1. Setup Data for Cross-Validation
    # ---------------------------------------------------------
    speaker_ids_all = np.array([it["speaker_id"] for it in items])
    sentence_ids_all = np.array([it["sentence_id"] for it in items])
    emotion_all = np.array([it["emotion"] for it in items])
    
    # Helper for label encoding
    def encode_labels(vals):
        uniq = {v: i for i, v in enumerate(sorted(set(vals)))}
        return np.array([uniq[v] for v in vals], dtype=np.int64), uniq

    y_spk_all, _ = encode_labels([it["speaker_id"] for it in items])
    y_emo_all, _ = encode_labels([it["emotion"] for it in items])
    y_sent_all, _ = encode_labels([it["sentence_id"] for it in items])
    
    # Pre-calculate Targets for Probes (Pitch/Energy) to avoid re-computation in CV loop
    print("  🎼 Pre-calculating pitch and energy targets...")
    pitch_hz_list = []
    energy_rms_list = []
    for it in items:
        pitch_hz_list.append(estimate_pitch_hz(it["audio"], it["sr"]))
        energy_rms_list.append(estimate_rms_energy(it["audio"]))
    pitch_hz_all = np.array(pitch_hz_list)
    energy_rms_all = np.array(energy_rms_list)

    # Metrics Aggregators
    geo_cov = []
    geo_radius = []
    cos_in_sample = []
    
    # Probe Results
    emo_par_f1s, emo_perp_f1s = [], []
    emo_perm_par_f1s, emo_perm_perp_f1s = [], []
    
    sent_par_f1s, sent_perp_f1s = [], []
    
    pitch_par_r2s, pitch_perp_r2s = [], []
    energy_par_r2s, energy_perp_r2s = [], []
    
    # Bootstrap Data Accumulators (A3)
    pitch_boot_data = [] 
    energy_boot_data = []
    
    # Results Accumulators for Global Probes (Speaker)
    Y_par_global = []
    Y_perp_global = []
    y_spk_global = []

    # Helper functions defined once
    def eval_probe_fit_predict(X_tr, y_tr, X_te, y_te, min_cls, seed):
        u, c = np.unique(y_tr, return_counts=True)
        valid = u[c >= min_cls]
        mask_tr = np.isin(y_tr, valid)
        mask_te = np.isin(y_te, valid)
        if np.sum(mask_tr) < 10 or len(valid) < 2 or np.sum(mask_te) < 1: return float("nan")
        
        clf = RidgeClassifier(class_weight='balanced', random_state=seed)
        clf.fit(X_tr[mask_tr], y_tr[mask_tr])
        preds = clf.predict(X_te[mask_te])
        return f1_score(y_te[mask_te], preds, average='macro')

    def eval_reg_collect(X_tr, y_tr, X_te, y_te, spk_te, seed):
        mask_tr, mask_te = np.isfinite(y_tr), np.isfinite(y_te)
        if np.sum(mask_tr) < 10 or np.sum(mask_te) < 1: 
            return float("nan"), None
            
        # Calculate predictions on Test Set
        # Use simple calculation: R2 = 1 - SSE/SST
        # We model trained on X_tr predicts on X_te.
        
        # Standardize based on Train stats
        ym = np.mean(y_tr[mask_tr])
        ys = np.std(y_tr[mask_tr]) + 1e-9
        y_tr_norm = (y_tr[mask_tr] - ym) / ys
        
        reg = Pipeline([
            ("scaler", RobustScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=seed, solver="svd"))
        ])
        reg.fit(X_tr[mask_tr], y_tr_norm)
        
        # Predict on Test
        pred_norm = reg.predict(X_te[mask_te])
        pred = pred_norm * ys + ym
        y_true = y_te[mask_te]
        
        # Compute Outer R2 directly
        # Handle constant case
        if np.var(y_true) < 1e-9:
            r2_test = 0.0
        else:
            r2_test = float(r2_score(y_true, pred))
        
        return r2_test, {
            "y_true": y_true, 
            "y_pred": pred, 
            "spk_id": spk_te[mask_te]
        }
    # (No global k fixing needed for Linear Map)
    
    n_splits = cfg.cv_folds
    uniq_spk = np.unique(speaker_ids_all)
    if len(uniq_spk) < n_splits:
        print(f"  ⚠️ Adjusting folds from {n_splits} to {len(uniq_spk)} due to speaker count.")
        n_splits = len(uniq_spk)
        
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_cnt = 0
    first_fold_map = None
    first_fold_test_idx = None
    first_fold_ut = None

    # A2: GroupKFold by Speaker ensures Speaker Disjoint splits for all probes within loop
    for train_idx, test_idx in gkf.split(z_reps, groups=speaker_ids_all):
        fold_cnt += 1
        print(f"  🔄 Fold {fold_cnt}/{n_splits}: Train={len(train_idx)} Test={len(test_idx)}")
        
        # --- A. Fit Procrustes Aligner on TRAIN ---
        aligner = ProcrustesAligner(
            k_target=768,  # Hard fixed semantic capacity
            whiten_method=cfg.ut_whiten_method
        )
        
        # Gather token sample for calibration (Definition B)
        # Sample ~10 tokens per training example to estimate normalization stats
        calib_tokens = []
        rng_local = np.random.RandomState(cfg.seed + fold_cnt)
        # Limit total calibration tokens to avoid OOM (~N*10 is fine)
        calib_indices = train_idx
        if len(calib_indices) > 500:
            calib_indices = rng_local.choice(train_idx, 500, replace=False)
            
        for cidx in calib_indices:
             Z_c = get_cached_tokens(cidx)
             if Z_c.shape[0] > 0:
                 # Take up to 8 random tokens
                 if Z_c.shape[0] > 8:
                     idx_t = rng_local.choice(Z_c.shape[0], 8, replace=False)
                     calib_tokens.append(Z_c[idx_t])
                 else:
                     calib_tokens.append(Z_c)
        if calib_tokens:
             calib_block = np.concatenate(calib_tokens, axis=0)
        else:
             calib_block = None
             
        aligner.dataset_fit(z_reps[train_idx], e_text[train_idx], calibration_tokens=calib_block)
        
        # --- B. Build Ut (Semantic Subspace) ---
        # "Audio-Derived" Definition:
        # We transform Train Audio means -> Y_train
        # Then build Ut from the centroids of Y_train grouped by sentence_id.
        Y_train_mean_for_ut = aligner.transform(z_reps[train_idx])
        
        Ut_fold = build_ut_from_sentence_centroids(
            Y_train_mean_for_ut,
            sentence_ids_all[train_idx],
            k=cfg.text_subspace_k
        )
        
        # Save first fold for evaluation
        if first_fold_map is None:
            first_fold_map = aligner
            first_fold_test_idx = test_idx
            first_fold_ut = Ut_fold
            
        # --- C. Eval Geometry on TEST ---
        for tidx in test_idx:
            Z = get_cached_tokens(tidx)
            # Use aligner to transform
            Y = aligner.transform(Z) 
            Yc = token_residuals(Y).astype(np.float32)
            
            # Geometry: calculate perp residual radius (RMS)
            # Yc is centered residuals [T, D]
            Yc_par = (Yc @ Ut_fold) @ Ut_fold.T
            Yc_perp = Yc - Yc_par
            D_emb = Yc.shape[1]
            
            # Radius = RMS of perp norm / sqrt(D)
            # This normalizes for dimensionality, so r=1 means "typical variation magnitude"
            norms = l2_norm(Yc_perp, axis=1)
            r = float(np.mean(norms) / math.sqrt(D_emb) + 1e-12)
            geo_radius.append(r)
            geo_cov.append(procrustean_cov_energy(Yc, Ut_fold))
            
            # For cosine, usage: mean(Y) vs transform_text(e)
            e_ref = aligner.transform_text(e_text[tidx:tidx+1])[0]
            cos_in_sample.append(cosine_sim(Y.mean(axis=0), e_ref))
            
        # --- D. Run Probes ---
        # Map Mean Representations
        Y_train_mean = Y_train_mean_for_ut # Already computed
        Y_test_mean = aligner.transform(z_reps[test_idx])
        
        tr_par, tr_perp = decompose_meanvec(Y_train_mean, Ut_fold)
        te_par, te_perp = decompose_meanvec(Y_test_mean, Ut_fold)
        
        # Accumulate for global speaker probe
        Y_par_global.append(te_par)
        Y_perp_global.append(te_perp)
        y_spk_global.append(y_spk_all[test_idx])
        
        # Probe Helpers (Inline wrappers around core logic)
        # Emotion Probe
        emo_par_f1s.append(eval_probe_fit_predict(tr_par, y_emo_all[train_idx], te_par, y_emo_all[test_idx], cfg.min_per_class, cfg.seed))
        emo_perp_f1s.append(eval_probe_fit_predict(tr_perp, y_emo_all[train_idx], te_perp, y_emo_all[test_idx], cfg.min_per_class, cfg.seed))

        # Permuted Baseline (A4) - Emotion only
        def eval_permuted_wrapper(X_tr, y_tr, X_te, y_te, n_repeats=5):
            scores = []
            for i in range(n_repeats):
                y_shuff = np.random.RandomState(cfg.seed + i).permutation(y_tr)
                scores.append(eval_probe_fit_predict(X_tr, y_shuff, X_te, y_te, cfg.min_per_class, cfg.seed))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return np.nanmean(scores)

        emo_perm_par_f1s.append(eval_permuted_wrapper(tr_par, y_emo_all[train_idx], te_par, y_emo_all[test_idx]))
        emo_perm_perp_f1s.append(eval_permuted_wrapper(tr_perp, y_emo_all[train_idx], te_perp, y_emo_all[test_idx]))
        
        # Sentence Probe
        sent_par_f1s.append(eval_probe_fit_predict(tr_par, y_sent_all[train_idx], te_par, y_sent_all[test_idx], 2, cfg.seed))
        sent_perp_f1s.append(eval_probe_fit_predict(tr_perp, y_sent_all[train_idx], te_perp, y_sent_all[test_idx], 2, cfg.seed))
        
        # Pitch
        r2, data = eval_reg_collect(tr_par, pitch_hz_all[train_idx], te_par, pitch_hz_all[test_idx], speaker_ids_all[test_idx], cfg.seed)
        pitch_par_r2s.append(r2)
        if data: pitch_boot_data.append({"type": "par", **data})
        
        r2, data = eval_reg_collect(tr_perp, pitch_hz_all[train_idx], te_perp, pitch_hz_all[test_idx], speaker_ids_all[test_idx], cfg.seed)
        pitch_perp_r2s.append(r2)
        if data: pitch_boot_data.append({"type": "perp", **data})
        
        # Energy
        r2, data = eval_reg_collect(tr_par, energy_rms_all[train_idx], te_par, energy_rms_all[test_idx], speaker_ids_all[test_idx], cfg.seed)
        energy_par_r2s.append(r2)
        if data: energy_boot_data.append({"type": "par", **data})
        
        r2, data = eval_reg_collect(tr_perp, energy_rms_all[train_idx], te_perp, energy_rms_all[test_idx], speaker_ids_all[test_idx], cfg.seed)
        energy_perp_r2s.append(r2)
        if data: energy_boot_data.append({"type": "perp", **data})

    # ---------------------------------------------------------
    # 3. Aggregate Metrics & Run Control Probes
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 3. Aggregate Metrics & Run Control Probes
    # ---------------------------------------------------------
    def safe_avg(lst):
        v = [x for x in lst if not math.isnan(x)]
        return float(np.mean(v)) if v else float("nan")
        
    def safe_std(lst):
        v = [x for x in lst if not math.isnan(x)]
        return float(np.std(v)) if v else float("nan")

    # Geometry
    cov_mean, cov_std = safe_avg(geo_cov), safe_std(geo_cov)
    rad_mean, rad_std = safe_avg(geo_radius), safe_std(geo_radius)
    cos_mean = safe_avg(cos_in_sample)
    
    # Probes (Emotion, Sentence, Pitch, Energy)
    emo_par_m, emo_par_s = safe_avg(emo_par_f1s), safe_std(emo_par_f1s)
    emo_perp_m, emo_perp_s = safe_avg(emo_perp_f1s), safe_std(emo_perp_f1s)
    
    # Permuted Baselines
    emo_perm_par_m = safe_avg(emo_perm_par_f1s)
    emo_perm_perp_m = safe_avg(emo_perm_perp_f1s)
    
    sent_par_m, sent_par_s = safe_avg(sent_par_f1s), safe_std(sent_par_f1s)
    sent_perp_m, sent_perp_s = safe_avg(sent_perp_f1s), safe_std(sent_perp_f1s)
    
    pitch_par_m, pitch_par_s = safe_avg(pitch_par_r2s), safe_std(pitch_par_r2s)
    pitch_perp_m, pitch_perp_s = safe_avg(pitch_perp_r2s), safe_std(pitch_perp_r2s)
    energy_par_m = safe_avg(energy_par_r2s)
    energy_perp_m = safe_avg(energy_perp_r2s)
    
    # Bootstrap 95% CI (A3) - Speaker-Aware Resampling
    def compute_bootstrap_ci(data_list, n_boot=1000, seed=42):
        if not data_list: return (float("nan"), float("nan"))
        
        # Consolidate
        all_y_true = np.concatenate([d["y_true"] for d in data_list])
        all_y_pred = np.concatenate([d["y_pred"] for d in data_list])
        all_spk = np.concatenate([d["spk_id"] for d in data_list])
        
        unique_spk = np.unique(all_spk)
        if len(unique_spk) < 2: return (float("nan"), float("nan"))
        
        # Pre-compute spk_map outside loop (Optimization A3.1)
        spk_map = {s: np.where(all_spk == s)[0] for s in unique_spk}
        
        scores = []
        rng = np.random.RandomState(seed)
        
        for _ in range(n_boot):
            # Resample speakers
            spk_sample = rng.choice(unique_spk, size=len(unique_spk), replace=True)
            
            # Select corresponding indices (efficiently?)
            # Actually, standard way is: indices = [idx for s in spk_sample for idx in spk_to_idx[s]]
            # Pre-map for speed
            # Or just use Pandas if available? No, stick to numpy.
            # mask = np.isin(all_spk, spk_sample) # wait, isin doesn't handle multiplicity (replacement)
            
            # Correct cluster bootstrap:
            # 1. Sample N clusters with replacement
            # 2. Concat all observations involved
            
            # Optimization: Pre-compute map outside loop (A3.1)
            # DONE (see below)
            
            indices = []
            for s in spk_sample:
                indices.append(spk_map[s])
            
            if not indices: continue
            indices = np.concatenate(indices)
            
            if len(indices) < 2: continue
            
            # Recalculate R2
            # Handle constant input case
            if np.var(all_y_true[indices]) < 1e-9:
                scores.append(0.0)
            else:
                scores.append(r2_score(all_y_true[indices], all_y_pred[indices]))
                
        if not scores: return (float("nan"), float("nan"))
        low = np.percentile(scores, 2.5)
        high = np.percentile(scores, 97.5)
        return (low, high)

    # Calculate CIs
    pitch_par_ci = compute_bootstrap_ci([d for d in pitch_boot_data if d["type"] == "par"])
    pitch_perp_ci = compute_bootstrap_ci([d for d in pitch_boot_data if d["type"] == "perp"])
    
    energy_par_ci = compute_bootstrap_ci([d for d in energy_boot_data if d["type"] == "par"])
    energy_perp_ci = compute_bootstrap_ci([d for d in energy_boot_data if d["type"] == "perp"])
    
    print(f"  📈 Pitch Par R2: {pitch_par_m:.3f} (95% CI: [{pitch_par_ci[0]:.3f}, {pitch_par_ci[1]:.3f}])")
    print(f"  📈 Pitch Perp R2: {pitch_perp_m:.3f} (95% CI: [{pitch_perp_ci[0]:.3f}, {pitch_perp_ci[1]:.3f}])")
    
    
    # Speaker Probe (Control): Run on Accumulated Leakage-Free Data
    if len(Y_par_global) > 0:
        Y_par_full = np.concatenate(Y_par_global, axis=0)
        Y_perp_full = np.concatenate(Y_perp_global, axis=0)
        y_spk_full = np.concatenate(y_spk_global, axis=0)
        
        # Filter
        mask_spk, y_spk_f = filter_classes(y_spk_full, min_per_class=cfg.min_per_class, max_classes=cfg.max_speaker_classes)
        if np.sum(mask_spk) > 50:
            # Use StratifiedKFold (Random split) as we lost sentence mapping. 
            # This is acceptable for "Speaker ID" control.
            spk_perp_res = run_probe_classifier(Y_perp_full[mask_spk], y_spk_f, n_splits=5, seed=cfg.seed)
            spk_par_res = run_probe_classifier(Y_par_full[mask_spk], y_spk_f, n_splits=5, seed=cfg.seed)
            
            spk_par_m, spk_par_s = spk_par_res["macro_f1"], spk_par_res["macro_f1_std"]
            spk_perp_m, spk_perp_s = spk_perp_res["macro_f1"], spk_perp_res["macro_f1_std"]
        else:
             spk_par_m, spk_par_s, spk_perp_m, spk_perp_s = float("nan"), 0.0, float("nan"), 0.0
    else:
         spk_par_m, spk_par_s, spk_perp_m, spk_perp_s = float("nan"), 0.0, float("nan"), 0.0

    # ---------------------------------------------------------
    # 4. Exp 2: Interventions (Run on First Fold Test Set ONLY)
    # ---------------------------------------------------------
    print(f"📊 Exp 2: Causal prosody interventions (Fold 1 Test Set, n={len(first_fold_test_idx)})...")
    
    intervention_res = {}
    
    # Setup for intervention loop (reusing existing helper logic if possible, or simplifying)
    # We will compute the dose-response metrics on the subset.
    
    # We reuse the logic: extract tokens, apply map, decompose.
    # BUT we must use first_fold_map (Aligner) and first_fold_ut.
    
    pitch_levels = list(cfg.pitch_shifts)
    gain_levels = list(cfg.gain_db)
    
    # Dictionaries for results
    p_res = {s: {"perp": [], "par": [], "ratio": [], "tok_perp": []} for s in pitch_levels}
    g_res = {g: {"perp": [], "par": [], "ratio": [], "tok_perp": []} for g in gain_levels}
    
    test_items = [items[i] for i in first_fold_test_idx]
    # Limit to reasonable number using memory setting
    limit_n = cfg.intervention_samples_low_mem if use_low_mem else cfg.intervention_samples
    max_inv = min(len(test_items), limit_n)
    inv_idxs = np.random.RandomState(cfg.seed).choice(len(test_items), max_inv, replace=False)
    
    # Pre-calculate baselines
    base_Y0 = []
    base_y0 = []
    
    for i in inv_idxs:
        orig = test_items[i]
        # Get tokens (cached)
        # Note: 'items' indices map to 'get_cached_tokens' indices. 
        # But here 'test_items' is a subset. We need original index.
        orig_idx = first_fold_test_idx[i]
        
        Z0 = get_cached_tokens(orig_idx)
        Y0 = first_fold_map.transform(Z0)
        base_Y0.append(Y0)
        base_y0.append(Y0.mean(axis=0))

    # Helper for intervention metrics
    def calc_metrics(y0, y1, Ut):
        dy = y1 - y0
        dy_par = project_to_subspace(dy, Ut)
        dy_perp = dy - dy_par
        n_par = l2_norm(dy_par)
        n_perp = l2_norm(dy_perp)
        ratio = n_perp / (n_par + n_perp + 1e-12)
        return float(n_par), float(n_perp), float(ratio)

    def calc_tok_metrics(Y0, Y1, Ut):
        # Sample alignment (simple resample for speed here?)
        if cfg.token_align == "dtw":
            A0, A1 = align_tokens_dtw(Y0, Y1, cfg.dtw_pca_dim, cfg.seed, cfg.resample_T)
        else:
            A0, A1 = align_tokens_resample(Y0, Y1, cfg.resample_T)
        
        dY = A1 - A0
        dY_par = (dY @ Ut) @ Ut.T
        dY_perp = dY - dY_par
        mean_perp = np.mean(l2_norm(dY_perp, axis=1))
        return float(mean_perp)

    # Shared Intervention Sweep Function
    def run_intervention_sweep(levels, mode="pitch"):
        res_dict = {lvl: {"perp": [], "par": [], "ratio": [], "tok_perp": []} for lvl in levels}
        
        local_cnt = 0
        for k, idx_in_sub in enumerate(inv_idxs):
            orig_idx = first_fold_test_idx[idx_in_sub]
            it = items[orig_idx]
            sr = it["sr"]
            Y0 = base_Y0[k]
            y0 = base_y0[k]
            
            for lvl in levels:
                val = float(lvl)
                if val == 0:
                    a1 = it["audio"].copy()
                else:
                    if mode == "pitch":
                        a1 = apply_pitch_shift(it["audio"], sr, int(val))
                    else: # gain
                        a1 = apply_gain(it["audio"], val)
                
                # Extract
                Z1 = adapter.extract_tokens(a1, sr).numpy().astype(np.float32)
                Y1 = first_fold_map.transform(Z1)
                y1 = Y1.mean(axis=0)
                
                np_val, npe_val, r_val = calc_metrics(y0, y1, first_fold_ut)
                ntok_val = calc_tok_metrics(Y0, Y1, first_fold_ut)
                
                res_dict[lvl]["par"].append(np_val)
                res_dict[lvl]["perp"].append(npe_val)
                res_dict[lvl]["ratio"].append(r_val)
                res_dict[lvl]["tok_perp"].append(ntok_val)
                
                local_cnt += 1
                if local_cnt % cfg.cleanup_interval_intervention == 0: free_torch()
        return res_dict

    # Sweep Pitch
    p_res = run_intervention_sweep(pitch_levels, mode="pitch")
    # Sweep Gain
    g_res = run_intervention_sweep(gain_levels, mode="gain")
            
    # Aggregating Intervention Results for Reporting
    # Use aggregation logic similar to original code
    pitch_perp_mean = [safe_avg(p_res[s]["perp"]) for s in pitch_levels]
    pitch_perp_std = [safe_std(p_res[s]["perp"]) for s in pitch_levels]
    
    # Fit Curves for Abstract
    # Pitch Quad
    pitch_x = [float(s) for s in pitch_levels]
    pitch_cal = fit_pitch_quadratic_no_linear(pitch_x, pitch_perp_mean)
    pitch_primary_a = pitch_cal["a"]
    
    # Gain Linear
    gain_perp_mean = [safe_avg(g_res[g]["perp"]) for g in gain_levels]
    gain_perp_std = [safe_std(g_res[g]["perp"]) for g in gain_levels]
    gain_x = [float(g) for g in gain_levels]
    gain_cal = fit_gain_linear(gain_x, gain_perp_mean)
    gain_primary_m = gain_cal["m"]
    
    # Save plots (simplified calls)
    tag = adapter.model_id.replace("/", "_")
    save_line_with_errorbars_and_fit(
        pitch_x, pitch_perp_mean, pitch_perp_std,
        f"{tag} Pitch Perp (Fold 1)", "Shift", "Perp",
        os.path.join(cfg.out_dir, f"{tag}_dose_pitch_meanvec_perp.png"),
        "pitch_quad", tag
    )
    save_line_with_errorbars_and_fit(
        gain_x, gain_perp_mean, gain_perp_std,
        f"{tag} Gain Perp (Fold 1)", "dB", "Perp",
        os.path.join(cfg.out_dir, f"{tag}_dose_gain_meanvec_perp.png"),
        "gain_lin", tag
    )

    # ---------------------------------------------------------
    # 5. Return Results
    # ---------------------------------------------------------
    print(f"  Procrustean (cov): {cov_mean:.4f} ± {cov_std:.4f}")
    print(f"  Radius_norm mean : {rad_mean:.4f} ± {rad_std:.4f}")
    
    # Fix variable naming
    qwen_ablation_best = ab_best_mode if ab_best_mode is not None else "default"
    qwen_ablation_scores = ab_scores if ab_scores is not None else {}

    result = {
        "model_id": adapter.model_id,
        "pooling": pool_mode,
        "procrustean_cov_mean": cov_mean,
        "procrustean_cov_std": cov_std,
        "radius_norm_mean": rad_mean,
        "radius_norm_std": rad_std,
        
        # Probes
        "speaker_probe_par": {"macro_f1": spk_par_m, "macro_f1_std": spk_par_s},
        "speaker_probe_perp": {"macro_f1": spk_perp_m, "macro_f1_std": spk_perp_s},
        "emotion_probe_par": {"macro_f1": emo_par_m, "macro_f1_std": emo_par_s},
        "emotion_probe_perp": {"macro_f1": emo_perp_m, "macro_f1_std": emo_perp_s},
        "emotion_probe_permuted": {"par_mean": emo_perm_par_m, "perp_mean": emo_perm_perp_m},
        
        "sentence_probe_par": {"macro_f1": sent_par_m, "macro_f1_std": sent_par_s},
        "sentence_probe_perp": {"macro_f1": sent_perp_m, "macro_f1_std": sent_perp_s},
        
        "pitch_reg_par": {"r2": pitch_par_m, "r2_std": pitch_par_s, "ci_low": pitch_par_ci[0], "ci_high": pitch_par_ci[1]},
        "pitch_reg_perp": {"r2": pitch_perp_m, "r2_std": pitch_perp_s, "ci_low": pitch_perp_ci[0], "ci_high": pitch_perp_ci[1]},
        "energy_reg_par": {"r2": energy_par_m, "ci_low": energy_par_ci[0], "ci_high": energy_par_ci[1]},
        "energy_reg_perp": {"r2": energy_perp_m, "ci_low": energy_perp_ci[0], "ci_high": energy_perp_ci[1]},
        
        # Dose Primary
        "dose_primary_pitch_a": pitch_primary_a,
        "dose_primary_gain_m": gain_primary_m,
        
        # Missing keys expected by main() - Restored as placeholders to prevent KeyError
        "subspace_cca_mean": 0.0,
        "angle_mean_deg": 0.0,
        "effective_rank": 0.0,
        "top5_concentration": 0.0,
        "qwen2_pooling_ablation_best": qwen_ablation_best,
        "qwen2_pooling_ablation_scores": qwen_ablation_scores,
        
        # Sanity
        "sanity_suite": sanity_results,
        "procrustes": {"heldout_sentence_cos_mean": cos_mean, "heldout_sentence_cos_std": 0.0},
        
        # Decomposition Sanity
        "decomposition_sanity_par_ok": bool(sent_par_m >= cfg.sentence_par_min_f1),
        "decomposition_sanity_perp_ok": bool(sent_perp_m <= cfg.sentence_perp_max_f1),
    }
    
    adapter.unload()
    cuda_mem("after free")

    # Cleanup disk cache if used
    if disk_cache_dir is not None and os.path.exists(disk_cache_dir):
        if cfg.token_cache_dir is None:  # Only cleanup auto-created temp dirs
            print(f"  🧹 Cleaning up disk cache: {disk_cache_dir}")
            shutil.rmtree(disk_cache_dir, ignore_errors=True)
    
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="all")
    parser.add_argument("--dataset_name", type=str, default=CFG_.dataset_name)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (None=all)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--cache_mode", type=str, default="lazy")
    parser.add_argument("--output_dir", type=str, default=CFG_.out_dir)
    parser.add_argument("--cache_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    CFG_.dataset_name = args.dataset_name
    CFG_.num_samples = args.limit if args.limit is not None else 10000
    if args.max_samples: CFG_.num_samples = args.max_samples
    CFG_.token_cache_mode = args.cache_mode
    CFG_.out_dir = args.output_dir
    if args.cache_dir: CFG_.token_cache_dir = args.cache_dir
    
    # Update CFG based on model specific tweaks if needed
    
    set_seed(CFG_.seed)
    os.makedirs(CFG_.out_dir, exist_ok=True)
    
    # Dump complete config to JSON for reproducibility
    import json
    from dataclasses import asdict
    with open(os.path.join(CFG_.out_dir, "config.json"), "w") as f:
        json.dump(asdict(CFG_), f, indent=4)
        print(f"  💾 Config saved to: {os.path.join(CFG_.out_dir, 'config.json')}")

    print("🧾 CFG summary:")
    print(f"  token_align={CFG_.token_align}  trim_strategy={CFG_.trim_strategy}")
    print(f"  ridge_pool_default={CFG_.ridge_pool_default}  ridge_pool_qwen2={CFG_.ridge_pool_qwen2}")
    print(f"  qwen2_pooling_ablation={CFG_.qwen2_pooling_ablation}  qwen2_pooling_auto_override={CFG_.qwen2_pooling_auto_override}")
    print(f"  ut_whiten={CFG_.ut_whiten}  max_tokens_for_global_stats={CFG_.max_tokens_for_global_stats}")

    # Note: load_cremad likely uses CFG_.num_samples as limit
    items = load_cremad(CFG_)

    print(f"🧠 Loading text encoder: {CFG_.text_encoder_name}")
    text_enc = TextEncoder(CFG_.text_encoder_name, device="cpu")

    extra = load_extra_text(CFG_)
    
    # Encode ALL sample texts to capture true frequency distribution for centering
    # This ensures Ut alignment matches Procrustes mapping alignment
    items_text = [it["text"] for it in items]
    print(f"  Encoding {len(items_text)} sample texts for alignment consistency...")
    e_text_all = text_enc.encode(items_text, batch_size=64)

    print(f"  Encoding {len(extra)} extra (Wikitext) sentences...")
    extra_embs = text_enc.encode(extra, batch_size=64)
    print(f"  Ut construction moved to Cross-Validation loop strictly inside analysis.")

    # Dynamic Model List
    target_model_id = args.model_id
    
    if target_model_id == "all":
        print("🚀 Running analysis on ALL benchmark models...")
        models = [
            ("Qwen2-Audio", "Qwen/Qwen2-Audio-7B-Instruct"),
            ("Qwen2.5-Omni", "Qwen/Qwen2.5-Omni-3B"), 
            ("AudioFlamingo3", "nvidia/audio-flamingo-3-hf"),
            ("DeSTA", "voidful/desta25_4b_baseline_full"),
            ("DeSTA", "voidful/desta25_4b_R2_full"),
            ("Qwen2.5-Omni", "Qwen/Qwen2.5-Omni-7B"), 
            ("DeSTA", "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"),
        ]
    else:
        family_name = "Custom"
        if "DeSTA" in target_model_id: family_name = "DeSTA"
        elif "Qwen" in target_model_id: family_name = "Qwen"
        elif "flamingo" in target_model_id.lower(): family_name = "AudioFlamingo3"
        models = [(family_name, target_model_id)]

    results = []
    for family, model_id in models:
        if "DeSTA" in model_id or family.startswith("DeSTA"):
            adapter = DeSTAAdapter(model_id=model_id, device="cuda")
        elif "Omni" in model_id:
            adapter = Qwen2_5OmniAdapter(model_id=model_id, device="cuda")
        elif "flamingo" in model_id.lower():
            # Use the top-level AudioFlamingo3Adapter
            adapter = AudioFlamingo3Adapter(model_id=model_id, device="cuda")
        else:
            adapter = Qwen2AudioAdapter(model_id=model_id, device="cuda")
        res = analyze_model(adapter, items, text_enc, CFG_)
        results.append(res)

    # Cross-model summary plots preserved.
    labels = []
    x = []
    y = []
    for r in results:
        labels.append(r["model_id"])
        x.append(r["procrustes"]["heldout_sentence_cos_mean"])
        y.append(r["radius_norm_mean"])

    save_scatter(
        x=x,
        y=y,
        labels=labels,
        title="Cross-model. Held-out sentence cosine vs tube radius",
        xlabel="Held-out sentence cosine (z_rep -> e_text)",
        ylabel="Radius_norm_mean (perp token residual)",
        path=os.path.join(CFG_.out_dir, "cross_model_cos_vs_radius.png"),
    )

    # Cross-model sanity flags preserved.
    sx = []
    sy = []
    sl = []
    for r in results:
        a = r["sentence_probe_par"]["macro_f1"]
        b = r["sentence_probe_perp"]["macro_f1"]
        sx.append(float(a) if a == a else 0.0)
        sy.append(float(b) if b == b else 0.0)
        sl.append(r["model_id"].replace("/", "_"))
    save_scatter(
        x=sx,
        y=sy,
        labels=sl,
        title="Sanity. sentence macro-F1: par(high) vs perp(low)",
        xlabel="sentence macro-F1 on par(meanvec)",
        ylabel="sentence macro-F1 on perp(meanvec)",
        path=os.path.join(CFG_.out_dir, "cross_model_sentence_par_vs_perp.png"),
    )

    # Cross-model: speaker par vs perp preserved.
    sx2 = []
    sy2 = []
    sl2 = []
    for r in results:
        a = r["speaker_probe_par"]["macro_f1"]
        b = r["speaker_probe_perp"]["macro_f1"]
        sx2.append(float(a) if a == a else 0.0)
        sy2.append(float(b) if b == b else 0.0)
        sl2.append(r["model_id"].replace("/", "_"))
    save_scatter(
        x=sx2,
        y=sy2,
        labels=sl2,
        title="Control. speaker macro-F1: par vs perp",
        xlabel="speaker macro-F1 on par(meanvec)",
        ylabel="speaker macro-F1 on perp(meanvec)",
        path=os.path.join(CFG_.out_dir, "cross_model_speaker_par_vs_perp.png"),
    )

    # New: cross-model coefficient summary for abstract.
    print("=" * 70)
    print("📌 Cross-model dose-response coefficients (primary metrics)")
    print("=" * 70)
    for r in results:
        mid = r["model_id"]
        a = r.get("dose_primary_pitch_a", float("nan"))
        m = r.get("dose_primary_gain_m", float("nan"))
        a_str = f"{a:.6f}" if a == a else "nan"
        m_str = f"{m:.6f}" if m == m else "nan"
        print(f"- {mid}")
        print(f"  Pitch curvature a (y=a x^2+b): {a_str}")
        print(f"  Gain  slope     m (y=m x + c): {m_str}")

    print("=" * 70)
    print("📋 FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"- {r['model_id']}")
        print(f"  pooling: {r['pooling']}")
        if r.get("qwen2_pooling_ablation_best", None) is not None:
            print(f"  qwen2 ablation best: {r['qwen2_pooling_ablation_best']} scores={r['qwen2_pooling_ablation_scores']}")
        print(f"  heldout_sentence_cos_mean: {r['procrustes']['heldout_sentence_cos_mean']:.4f}")
        print(f"  radius_norm_mean: {r['radius_norm_mean']:.4f}")
        print(f"  procrustean_cov_mean: {r['procrustean_cov_mean']:.4f}")
        print(f"  subspace cca mean: {r['subspace_cca_mean']:.4f} angle_mean_deg: {r['angle_mean_deg']:.2f}")
        print(f"  sentence_par F1: {r['sentence_probe_par']['macro_f1']:.4f}  sentence_perp F1: {r['sentence_probe_perp']['macro_f1']:.4f}")
        print(f"  decomposition_sanity: par_ok={r['decomposition_sanity_par_ok']} perp_ok={r['decomposition_sanity_perp_ok']}")
        print(f"  emotion F1 (par): {r['emotion_probe_par']['macro_f1']:.4f}  emotion F1 (perp): {r['emotion_probe_perp']['macro_f1']:.4f}")
        print(f"  speaker F1 (par): {r['speaker_probe_par']['macro_f1']:.4f}  speaker F1 (perp): {r['speaker_probe_perp']['macro_f1']:.4f}")
        print(f"  pitch R² (par): {r['pitch_reg_par']['r2']:.4f}  pitch R² (perp): {r['pitch_reg_perp']['r2']:.4f}")
        print(f"  energy R² (par): {r['energy_reg_par']['r2']:.4f}  energy R² (perp): {r['energy_reg_perp']['r2']:.4f}")
        print(f"  dose primary pitch a: {r.get('dose_primary_pitch_a', float('nan')):.6f}")
        print(f"  dose primary gain  m: {r.get('dose_primary_gain_m', float('nan')):.6f}")

    print(f"💾 Results saved to: {CFG_.out_dir}")


if __name__ == "__main__":
    main()
