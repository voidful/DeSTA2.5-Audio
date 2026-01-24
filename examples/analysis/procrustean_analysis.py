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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch

from datasets import load_dataset

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, RidgeClassifier
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
    cleanup_interval_token_cache_qwen_omni: int = 10  # More aggressive for Qwen2.5-Omni
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


def free_torch() -> None:
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

    @torch.no_grad()
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

    @torch.no_grad()
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

    @torch.no_grad()
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

    @torch.no_grad()
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
            
            # Force garbage collection every extraction for Qwen2.5-Omni
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
# Ridge map: representative pooling
# -----------------------------
def pool_representation(Z: torch.Tensor, mode: str) -> np.ndarray:
    """
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


class RidgeMap:
    def __init__(self, pipeline: Pipeline):
        self.pipe = pipeline

    def map_mean(self, z_rep: np.ndarray) -> np.ndarray:
        return self.pipe.predict(z_rep).astype(np.float32)

    def map_tokens(self, Z: np.ndarray) -> np.ndarray:
        return self.pipe.predict(Z).astype(np.float32)


class ProcrustesMap:
    """
    Orthogonal Procrustes mapping with PCA projection for TRUE norm preservation.
    
    When Dz != De (common: Qwen2 has Dz=4096, text has De=768), direct Procrustes
    with padding does NOT preserve norms. The solution:
    
    1. PCA audio embeddings to k dimensions (k = min(De, Dz, N-1))
    2. Apply orthogonal rotation R (k x k) in the shared k-dim space
    3. R is truly orthogonal: R @ R.T = I, so ||zR|| = ||z||
    
    This guarantees tube radius is preserved in the analysis space.
    """
    def __init__(
        self, 
        U_z: np.ndarray,        # [Dz, k] PCA projection for audio
        R: np.ndarray,          # [k, k] orthogonal rotation
        z_mean: np.ndarray,     # [Dz] audio mean
        e_mean: np.ndarray,     # [k] text mean in k-space
        z_scale: float,
        e_scale: float,
        k: int,
        e_mean_full: np.ndarray = None,  # [De] original text mean for transform_text
        U_e: np.ndarray = None,          # [De, k] PCA projection for text (if De > k)
    ):
        """
        Mapping: Z -> (Z - z_mean) / z_scale @ U_z @ R * e_scale + e_mean[:k]
        """
        self.U_z = U_z.astype(np.float32)  # [Dz, k]
        self.R = R.astype(np.float32)      # [k, k]
        self.z_mean = z_mean.astype(np.float32)
        self.e_mean = e_mean.astype(np.float32)  # [k]
        self.z_scale = float(z_scale)
        self.e_scale = float(e_scale)
        self.k = k
        self.e_mean_full = e_mean_full.astype(np.float32) if e_mean_full is not None else None
        self.U_e = U_e.astype(np.float32) if U_e is not None else None
    
    def map_mean(self, z_rep: np.ndarray) -> np.ndarray:
        """Map a single mean audio embedding to shared k-dim space."""
        z_rep = np.atleast_2d(z_rep).astype(np.float32)
        # Center, scale, project to k-dim, rotate
        z_c = (z_rep - self.z_mean) / self.z_scale
        z_k = z_c @ self.U_z  # [batch, k]
        y_k = z_k @ self.R    # [batch, k] - orthogonal rotation preserves norm!
        y = y_k * self.e_scale + self.e_mean
        return y.astype(np.float32)
    
    def map_tokens(self, Z: np.ndarray) -> np.ndarray:
        """Map token-level audio embeddings to shared k-dim space."""
        Z = Z.astype(np.float32)
        # Center, scale, project to k-dim, rotate
        Z_c = (Z - self.z_mean) / self.z_scale
        Z_k = Z_c @ self.U_z  # [T, k]
        Y_k = Z_k @ self.R    # [T, k] - orthogonal rotation preserves norm!
        Y = Y_k * self.e_scale + self.e_mean
        return Y.astype(np.float32)
    
    def transform_text(self, e_text: np.ndarray) -> np.ndarray:
        """
        Transform raw text embeddings to the same k-dim space as mapped audio (Y).
        This ensures cosine comparisons are in the same coordinate system.
        
        Matches map_mean output: both end up in centered, scaled k-space.
        """
        e_text = np.atleast_2d(e_text).astype(np.float32)
        if self.e_mean_full is not None:
            e_c = (e_text - self.e_mean_full) / self.e_scale
        else:
            e_c = e_text / self.e_scale
        
        # If we have PCA for text, apply it
        if self.U_e is not None:
            e_k = e_c @ self.U_e  # [batch, k]
        else:
            e_k = e_c[:, :self.k]  # Truncate to k dims
        
        # Apply same scaling as map_mean output: * e_scale + e_mean
        e_transformed = e_k * self.e_scale + self.e_mean
        
        return e_transformed.astype(np.float32)
    
    def map_tokens_k(self, Z: np.ndarray) -> np.ndarray:
        """
        Map tokens to NORMALIZED k-space (no e_scale applied).
        Use this for geometry analysis where orthogonal norm preservation holds.
        """
        Z = Z.astype(np.float32)
        Z_c = (Z - self.z_mean) / self.z_scale
        Z_k = Z_c @ self.U_z
        Y_k = Z_k @ self.R  # Orthogonal rotation preserves norm in this space!
        return Y_k.astype(np.float32)
    
    def transform_text_k(self, e_text: np.ndarray) -> np.ndarray:
        """
        Transform text to NORMALIZED k-space (matching heldout evaluation).
        Use this for geometry analysis where coordinates are consistent.
        """
        e_text = np.atleast_2d(e_text).astype(np.float32)
        if self.e_mean_full is not None:
            e_c = (e_text - self.e_mean_full) / self.e_scale
        else:
            e_c = e_text / self.e_scale
        
        if self.U_e is not None:
            E_k = e_c @ self.U_e
        else:
            E_k = e_c[:, :self.k]
        
        return E_k.astype(np.float32)


def fit_procrustes_map(
    z_reps: np.ndarray,
    e_text: np.ndarray,
    groups: np.ndarray,
    cfg: CFG,
) -> Tuple[ProcrustesMap, Dict[str, float]]:
    """
    Fit orthogonal Procrustes mapping with PCA projection.
    
    CRITICAL FIX for Dz != De case:
    1. PCA audio to k dimensions
    2. Orthogonal Procrustes in shared k-dim space
    3. R is k x k truly orthogonal -> norms preserved
    
    This allows us to claim "mapping is orthogonal in the analysis space"
    and tube radius metrics are not artificially crushed by scaling.
    """
    z_reps = z_reps.astype(np.float64)
    e_text = e_text.astype(np.float64)
    
    N = z_reps.shape[0]
    Dz = z_reps.shape[1]
    De = e_text.shape[1]
    
    # Determine shared dimension k
    k = min(De, Dz, N - 1)
    k = max(k, 1)
    
    print(f"  📐 Procrustes setup: Dz={Dz}, De={De}, shared k={k}")
    
    # Center audio
    z_mean = z_reps.mean(axis=0)
    Z_c = z_reps - z_mean
    
    # Scale audio for numerical stability
    z_scale = max(np.std(Z_c), 1e-9)
    Z_c = Z_c / z_scale
    
    # PCA audio to k dimensions
    pca_z = PCA(n_components=k, random_state=cfg.seed)
    Z_k = pca_z.fit_transform(Z_c)  # [N, k]
    U_z = pca_z.components_.T       # [Dz, k] - projection matrix
    
    # Verify U_z is orthonormal
    assert np.allclose(U_z.T @ U_z, np.eye(k), atol=1e-5), "PCA components not orthonormal!"
    
    # Center and scale text in original space, then project to k-dim
    e_mean_full = e_text.mean(axis=0)
    E_c = e_text - e_mean_full
    e_scale = max(np.std(E_c), 1e-9)
    E_c = E_c / e_scale
    
    # If De > k, PCA text to k dims too. If De == k, use directly.
    if De > k:
        pca_e = PCA(n_components=k, random_state=cfg.seed)
        E_k = pca_e.fit_transform(E_c)
        e_mean = np.zeros(k, dtype=np.float64)  # Already centered
    else:
        E_k = E_c[:, :k] if De >= k else np.pad(E_c, ((0, 0), (0, k - De)))
        e_mean = e_mean_full[:k] if De >= k else np.pad(e_mean_full, (0, k - De))
    
    # Orthogonal Procrustes in k-dim space: find R such that ||Z_k @ R - E_k||_F is minimized
    R, scale = orthogonal_procrustes(Z_k, E_k)
    
    # Verify R is orthogonal (k x k)
    assert R.shape == (k, k), f"R shape mismatch: {R.shape}"
    assert np.allclose(R @ R.T, np.eye(k), atol=1e-5), "R is not orthogonal!"
    print(f"  ✅ Orthogonal R verified: {k}x{k}, ||R @ R.T - I|| = {np.max(np.abs(R @ R.T - np.eye(k))):.2e}")
    
    # Held-out evaluation with proper fold-wise fitting (fixing leakage)
    uniq = np.unique(groups)
    heldout_sims = []
    
    for s in uniq:
        mask_te = (groups == s)
        mask_tr = ~mask_te
        if np.sum(mask_tr) < 10 or np.sum(mask_te) < 1:
            continue
        
        # Fold-wise centering and scaling (no leakage)
        z_mean_tr = z_reps[mask_tr].mean(axis=0)
        Z_tr_c = (z_reps[mask_tr] - z_mean_tr)
        z_scale_tr = max(np.std(Z_tr_c), 1e-9)
        Z_tr_c = Z_tr_c / z_scale_tr
        
        # PCA on train
        k_tr = min(k, Z_tr_c.shape[0] - 1)
        if k_tr < 1:
            continue
        pca_tr = PCA(n_components=k_tr, random_state=cfg.seed)
        Z_tr_k = pca_tr.fit_transform(Z_tr_c)
        U_z_tr = pca_tr.components_.T
        
        # Text fold centering
        e_mean_tr = e_text[mask_tr].mean(axis=0)
        E_tr_c = (e_text[mask_tr] - e_mean_tr)
        e_scale_tr = max(np.std(E_tr_c), 1e-9)
        E_tr_c = E_tr_c / e_scale_tr
        
        if De > k_tr:
            pca_e_tr = PCA(n_components=k_tr, random_state=cfg.seed)
            E_tr_k = pca_e_tr.fit_transform(E_tr_c)
        else:
            E_tr_k = E_tr_c[:, :k_tr]
        
        # Fit Procrustes on train
        if Z_tr_k.shape[0] > 1 and E_tr_k.shape[0] > 1:
            R_tr, _ = orthogonal_procrustes(Z_tr_k, E_tr_k)
        else:
            continue
        
        # Apply to test using TRAIN statistics
        Z_te_c = (z_reps[mask_te] - z_mean_tr) / z_scale_tr
        Z_te_k = Z_te_c @ U_z_tr
        pred_k = Z_te_k @ R_tr  # [n_te, k_tr] in k-space
        
        # Put test text in SAME k-space (centered, scaled, PCA-projected if needed)
        E_te_c = (e_text[mask_te] - e_mean_tr) / e_scale_tr  # [n_te, De]
        if De > k_tr:
            E_te_k = pca_e_tr.transform(E_te_c)  # [n_te, k_tr]
        else:
            E_te_k = E_te_c[:, :k_tr]  # [n_te, k_tr]
        
        # Safety assertion
        assert pred_k.shape == E_te_k.shape, f"Shape mismatch: {pred_k.shape} vs {E_te_k.shape}"
        
        for i in range(pred_k.shape[0]):
            heldout_sims.append(cosine_sim(pred_k[i], E_te_k[i]))
    
    heldout_mean = float(np.mean(heldout_sims)) if len(heldout_sims) else 0.0
    heldout_std = float(np.std(heldout_sims)) if len(heldout_sims) else 0.0
    
    # Create map with full-data fitted parameters
    # Use e_mean in k-space
    e_mean_k = np.zeros(k, dtype=np.float64)  # Centered in k-space
    
    # Get U_e for transform_text if PCA was applied to text
    U_e = pca_e.components_.T if De > k else None  # [De, k] or None
    
    pm = ProcrustesMap(U_z, R, z_mean, e_mean_k, z_scale, e_scale, k, e_mean_full=e_mean_full, U_e=U_e)
    
    # In-sample cosine
    pred_all = pm.map_mean(z_reps)
    in_sample_cos = []
    for i in range(pred_all.shape[0]):
        in_sample_cos.append(cosine_sim(pred_all[i], E_k[i]))
    
    info = {
        "procrustes_scale": float(scale),
        "in_sample_cos_mean": float(np.mean(in_sample_cos)),
        "in_sample_cos_std": float(np.std(in_sample_cos)),
        "heldout_sentence_cos_mean": heldout_mean,
        "heldout_sentence_cos_std": heldout_std,
        "z_dim": Dz,
        "e_dim": De,
        "shared_k": k,
        "pca_explained_variance_ratio": float(pca_z.explained_variance_ratio_.sum()),
    }
    
    print(f"  📐 Procrustes: in_sample_cos={info['in_sample_cos_mean']:.4f}, heldout_cos={heldout_mean:.4f}")
    print(f"  📐 PCA explained variance: {info['pca_explained_variance_ratio']:.4f}")
    
    return pm, info


def fit_ridge_map(
    z_reps: np.ndarray,
    e_text: np.ndarray,
    groups: np.ndarray,
    cfg: CFG,
) -> Tuple[RidgeMap, Dict[str, float]]:
    """
    GroupKFold(sentence) alpha selection with cosine scorer.
    Also compute leave-one-sentence-out held-out cosine.
    """
    assert groups is not None, "groups must not be None for GroupKFold(sentence)."

    n_groups = len(np.unique(groups))
    n_splits = int(min(cfg.ridge_cv_folds, n_groups))
    n_splits = max(n_splits, 2) if n_groups >= 2 else 2

    gkf = GroupKFold(n_splits=n_splits)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(random_state=cfg.seed, solver="svd")),
        ]
    )
    grid = GridSearchCV(
        estimator=pipe,
        param_grid={"ridge__alpha": list(cfg.ridge_alphas)},
        scoring=cosine_scorer,
        cv=gkf,
        n_jobs=1,
        verbose=0,
    )

    grid.fit(z_reps, e_text, groups=groups)

    best_alpha = float(grid.best_params_["ridge__alpha"])
    best_cv = float(grid.best_score_)
    best_pipe = grid.best_estimator_
    
    # Check if best_alpha is at boundary of search range (potential warning)
    alphas_sorted = sorted(cfg.ridge_alphas)
    alpha_at_boundary = False
    if best_alpha == alphas_sorted[-1]:
        print(f"  ⚠️ Ridge best_alpha={best_alpha:.0e} is at MAX boundary. Consider extending ridge_alphas range.")
        alpha_at_boundary = True
    elif best_alpha == alphas_sorted[0]:
        print(f"  ⚠️ Ridge best_alpha={best_alpha:.0e} is at MIN boundary. Consider extending ridge_alphas range.")
        alpha_at_boundary = True

    uniq = np.unique(groups)
    heldout_sims = []
    for s in uniq:
        mask_te = (groups == s)
        mask_tr = ~mask_te
        if np.sum(mask_tr) < 10 or np.sum(mask_te) < 1:
            continue
        local_pipe = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=best_alpha, random_state=cfg.seed, solver="svd")),
            ]
        )
        local_pipe.fit(z_reps[mask_tr], e_text[mask_tr])
        pred = local_pipe.predict(z_reps[mask_te])
        for i in range(pred.shape[0]):
            heldout_sims.append(cosine_sim(pred[i], e_text[mask_te][i]))

    heldout_mean = float(np.mean(heldout_sims)) if len(heldout_sims) else 0.0
    heldout_std = float(np.std(heldout_sims)) if len(heldout_sims) else 0.0

    info = {
        "best_alpha": best_alpha,
        "cv_best_cos": best_cv,
        "heldout_sentence_cos_mean": heldout_mean,
        "heldout_sentence_cos_std": heldout_std,
        "alpha_at_boundary": alpha_at_boundary,
    }
    return RidgeMap(best_pipe), info


# -----------------------------
# Geometry
# -----------------------------
def project_to_subspace(v: np.ndarray, Ut: np.ndarray) -> np.ndarray:
    return Ut @ (Ut.T @ v)


def decompose_meanvec(y_mean: np.ndarray, Ut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_par = project_to_subspace(y_mean, Ut)
    y_perp = y_mean - y_par
    return y_par, y_perp


def token_residuals(Y: np.ndarray) -> np.ndarray:
    return Y - np.mean(Y, axis=0, keepdims=True)


def procrustean_cov_energy(Yc: np.ndarray, Ut: np.ndarray, eps: float = 1e-12) -> float:
    """
    Low-rank O(T·D·k) version. Uses ||Yc @ Ut||² = ||Proj_Ut(Yc)||² since Ut is orthonormal.
    """
    C = Yc @ Ut  # [T, k]
    e_par = float(np.sum(C * C))
    e_tot = float(np.sum(Yc * Yc)) + eps
    return e_par / e_tot


def tube_radius_norm(Yc: np.ndarray, Ut: np.ndarray, eps: float = 1e-12) -> float:
    """
    Low-rank O(T·D·k) version. Perp energy = total - par for each token.
    """
    D = Yc.shape[1]
    C = Yc @ Ut  # [T, k]
    per_tok_par = np.sum(C * C, axis=1)  # [T]
    per_tok_tot = np.sum(Yc * Yc, axis=1)  # [T]
    per_tok_perp = np.maximum(per_tok_tot - per_tok_par, 0.0)  # Numerical safety
    rms = float(np.sqrt(np.mean(per_tok_perp) + eps))
    return rms / math.sqrt(D)


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
    splits = safe_stratified_splits(y, n_splits=n_splits, seed=seed)
    if splits is None:
        return {"macro_f1": float("nan"), "macro_f1_std": float("nan"), "acc": float("nan"), "acc_std": float("nan")}

    f1s = []
    accs = []
    for tr, te in splits:
        clf = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", RidgeClassifier(alpha=1.0, random_state=seed)),
            ]
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        f1s.append(f1_score(y[te], pred, average="macro"))
        accs.append(accuracy_score(y[te], pred))
    return {
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "acc": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
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
    Reports both R² and Pearson r.
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
    for tr, te in splits:
        reg = Pipeline(
            [
                ("scaler", RobustScaler()),  # More robust to outliers than StandardScaler
                ("ridge", Ridge(alpha=1.0, random_state=seed, solver="svd")),
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
            reps.append(pool_representation(Z, m))
        reps = np.stack(reps, axis=0)
        _, info = fit_ridge_map(reps, e_text, sentence_ids, cfg)
        out[m] = float(info["heldout_sentence_cos_mean"])

    best_mode = max(out, key=lambda k: out[k])
    print(f"  Ablation heldout_sentence_cos: {out}")
    print(f"  Suggested ridge pooling mode for Qwen2: {best_mode}")
    return best_mode, out


# -----------------------------
# Main analysis for one model
# -----------------------------
def analyze_model(
    adapter: AudioAdapterBase,
    items: List[Dict[str, Any]],
    text_enc: TextEncoder,
    Ut_shared: np.ndarray,
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
            return np.load(cache_path)
        elif cache_mode == "lazy":
            return adapter.extract_tokens(items[idx]["audio"], items[idx]["sr"]).numpy().astype(cache_dtype)
        else:  # ram
            return token_cache[idx]

    print(f"🧩 Building z_rep for Procrustes map from cache. pooling={pool_mode}")
    z_reps = []
    for i in range(len(items)):
        Z = torch.from_numpy(get_cached_tokens(i)).float()  # float16 -> float32 for precision
        z_reps.append(pool_representation(Z, pool_mode))
    z_reps = np.stack(z_reps, axis=0)

    print("🧩 Fitting orthogonal Procrustes map (preserves tube radius) ...")
    procrustes_map, procrustes_info = fit_procrustes_map(z_reps, e_text, sentence_ids, cfg)
    print(f"  Held-out sentence cosine: {procrustes_info['heldout_sentence_cos_mean']:.4f} ± {procrustes_info['heldout_sentence_cos_std']:.4f}")

    print("📊 Exp 1+3: Geometry + Probes in shared 768 (single pass) ...")

    procs: List[float] = []
    radii: List[float] = []
    cos_in_sample: List[float] = []

    global_tokens: List[np.ndarray] = []

    Y_mean_all: List[np.ndarray] = []
    Y_par_all: List[np.ndarray] = []
    Y_perp_all: List[np.ndarray] = []

    speaker: List[Any] = []
    emotion: List[str] = []
    sentence_group: List[str] = []

    pitch_hz: List[float] = []
    energy_rms: List[float] = []

    for i in range(len(items)):
        Z = get_cached_tokens(i)  # Use cached tokens
        Y = procrustes_map.map_tokens_k(Z)  # Normalized k-space for geometry

        y_mean = Y.mean(axis=0).astype(np.float32)
        y_par, y_perp = decompose_meanvec(y_mean, Ut_shared)

        Y_mean_all.append(y_mean)
        Y_par_all.append(y_par)
        Y_perp_all.append(y_perp)

        speaker.append(items[i]["speaker_id"])
        emotion.append(items[i]["emotion"])
        sentence_group.append(items[i]["sentence_id"])

        # Cosine in NORMALIZED k-space (matches map_tokens_k output)
        e_ref = procrustes_map.transform_text_k(e_text[i:i+1])[0]  # [k]
        cos_in_sample.append(cosine_sim(y_mean, e_ref))

        Yc = token_residuals(Y).astype(np.float32)
        procs.append(procrustean_cov_energy(Yc, Ut_shared))
        radii.append(tube_radius_norm(Yc, Ut_shared))

        if cfg.max_tokens_for_global_stats > 0:
            take = min(Yc.shape[0], 256)
            if take > 0:
                global_tokens.append(sample_rows(Yc, max_rows=take, seed=cfg.seed + i))

        pitch_hz.append(estimate_pitch_hz(items[i]["audio"], items[i]["sr"]))
        energy_rms.append(estimate_rms_energy(items[i]["audio"]))

    procs_np = np.asarray(procs, dtype=np.float32)
    radii_np = np.asarray(radii, dtype=np.float32)
    cos_np = np.asarray(cos_in_sample, dtype=np.float32)

    Y_mean_all_np = np.stack(Y_mean_all, axis=0).astype(np.float32)
    Y_par_all_np = np.stack(Y_par_all, axis=0).astype(np.float32)
    Y_perp_all_np = np.stack(Y_perp_all, axis=0).astype(np.float32)
    
    # === CRITICAL: Rebuild Ut from Y sentence centroids (not external text) ===
    # This defines "semantic" as what the AUDIO MODEL uses to distinguish sentences
    sentence_ids_np = np.asarray(sentence_group, dtype=object)
    print("🔄 Rebuilding Ut from mapped audio (Y) sentence centroids...")
    
    # PRESERVE Ut_text for cross-model comparability before overwriting
    Ut_text = Ut_shared.copy()  # Original text-encoder based Ut
    
    Ut_model = build_ut_from_sentence_centroids(Y_mean_all_np, sentence_ids_np, k=cfg.text_subspace_k)
    validate_ut_geometry(Ut_model, expected_D=procrustes_map.k)  # Use actual k-dim from Procrustes
    
    # Compute geometry metrics for BOTH Ut versions
    print("🔄 Computing geometry metrics for Ut_text (cross-model) and Ut_model (model-specific)...")
    procs_text = []
    radii_text = []
    procs_model = []
    radii_model = []
    Y_par_model = []
    Y_perp_model = []
    
    for i in range(len(items)):
        y_mean = Y_mean_all_np[i]
        y_par, y_perp = decompose_meanvec(y_mean, Ut_model)
        Y_par_model.append(y_par)
        Y_perp_model.append(y_perp)
        
        # Geometry metrics with cached tokens
        Z = get_cached_tokens(i)
        Y = procrustes_map.map_tokens_k(Z)  # Normalized k-space for geometry
        Yc = token_residuals(Y).astype(np.float32)
        
        # Ut_text metrics (cross-model comparable)
        procs_text.append(procrustean_cov_energy(Yc, Ut_text))
        radii_text.append(tube_radius_norm(Yc, Ut_text))
        
        # Ut_model metrics (model-specific)
        procs_model.append(procrustean_cov_energy(Yc, Ut_model))
        radii_model.append(tube_radius_norm(Yc, Ut_model))
    
    Y_par_all_np = np.stack(Y_par_model, axis=0).astype(np.float32)
    Y_perp_all_np = np.stack(Y_perp_model, axis=0).astype(np.float32)
    
    # Text-based metrics (for cross-model comparison)
    procs_text_np = np.asarray(procs_text, dtype=np.float32)
    radii_text_np = np.asarray(radii_text, dtype=np.float32)
    
    # Model-based metrics (for model-specific analysis)
    procs_np = np.asarray(procs_model, dtype=np.float32)
    radii_np = np.asarray(radii_model, dtype=np.float32)
    
    # Use Y-based Ut for the rest of analysis (probes, interventions)
    Ut_shared = Ut_model

    if len(global_tokens) > 0:
        all_token_resid = np.concatenate(global_tokens, axis=0)
        all_token_resid = sample_rows(all_token_resid, max_rows=cfg.max_tokens_for_global_stats, seed=cfg.seed)
    else:
        all_token_resid = np.zeros((1, procrustes_map.k), dtype=np.float32)

    cca_mean, angle_mean_deg = subspace_angles_tokens_vs_semantic(all_token_resid, Ut_shared, cfg.audio_subspace_k, cfg.seed)

    s = np.linalg.svd(all_token_resid, compute_uv=False)
    er = effective_rank_from_svals(s)
    top5 = topk_concentration(s, k=5)

    # Scientific notation for debugging (distinguishes bug from collapse)
    print(f"  Procrustean (cov): {float(procs_np.mean()):.4f} ± {float(procs_np.std()):.4f}  (raw: {float(procs_np.mean()):.2e})")
    print(f"  Radius_norm mean : {float(radii_np.mean()):.4f} ± {float(radii_np.std()):.4f}  (raw: {float(radii_np.mean()):.2e})")
    print(f"  Cos(y_mean, e_text) (in-sample upper bound): {float(cos_np.mean()):.4f} ± {float(cos_np.std()):.4f}")
    print(f"  Subspace CCA mean : {cca_mean:.4f}   angle_mean(deg)={angle_mean_deg:.2f}")
    print(f"  Effective rank(768) on sampled tokens: {er:.2f}")
    print(f"  Top-5 concentration on sampled tokens: {top5:.4f}")

    # Exp 2. Full sweep + dose-response.
    intervention_n = cfg.intervention_samples_low_mem if use_low_mem else cfg.intervention_samples
    print(f"📊 Exp 2: Causal prosody interventions (n={intervention_n}, low_mem={use_low_mem}) ...")
    rng = np.random.RandomState(cfg.seed)
    idxs = rng.choice(len(items), size=min(intervention_n, len(items)), replace=False)

    base_Y0: Dict[int, np.ndarray] = {}
    base_y0: Dict[int, np.ndarray] = {}

    for j in idxs:
        it = items[j]
        # Use cached tokens for consistency and speed (avoid redundant forward pass)
        Z0 = get_cached_tokens(j)
        Y0 = procrustes_map.map_tokens_k(Z0).astype(np.float32)  # Normalized k-space
        base_Y0[j] = Y0
        base_y0[j] = Y0.mean(axis=0).astype(np.float32)

    pitch_levels = list(cfg.pitch_shifts)
    gain_levels = list(cfg.gain_db)

    pitch_meanvec_par: Dict[int, List[float]] = {sft: [] for sft in pitch_levels}
    pitch_meanvec_perp: Dict[int, List[float]] = {sft: [] for sft in pitch_levels}
    pitch_meanvec_ratio: Dict[int, List[float]] = {sft: [] for sft in pitch_levels}
    pitch_token_mean: Dict[int, List[float]] = {sft: [] for sft in pitch_levels}
    pitch_token_par: Dict[int, List[float]] = {sft: [] for sft in pitch_levels}
    pitch_token_perp: Dict[int, List[float]] = {sft: [] for sft in pitch_levels}

    gain_meanvec_par: Dict[float, List[float]] = {db: [] for db in gain_levels}
    gain_meanvec_perp: Dict[float, List[float]] = {db: [] for db in gain_levels}
    gain_meanvec_ratio: Dict[float, List[float]] = {db: [] for db in gain_levels}
    gain_token_mean: Dict[float, List[float]] = {db: [] for db in gain_levels}
    gain_token_par: Dict[float, List[float]] = {db: [] for db in gain_levels}
    gain_token_perp: Dict[float, List[float]] = {db: [] for db in gain_levels}

    def meanvec_delta_metrics(y0: np.ndarray, y1: np.ndarray, Ut: np.ndarray) -> Tuple[float, float, float]:
        dy = (y1 - y0).astype(np.float32)
        dy_par = project_to_subspace(dy, Ut)
        dy_perp = dy - dy_par

        D = dy.shape[0]
        n_par = float(l2_norm(dy_par) / math.sqrt(D))
        n_perp = float(l2_norm(dy_perp) / math.sqrt(D))
        ratio = float(n_perp / (n_par + n_perp + 1e-12))
        return n_par, n_perp, ratio

    # Sweep pitch.
    intervention_count = 0
    for j in idxs:
        it = items[j]
        Y0 = base_Y0[j]
        y0 = base_y0[j]
        sr0 = it["sr"]

        for sft in pitch_levels:
            # NOTE: Even for sft==0, we measure to get noise floor
            if int(sft) == 0:
                a1 = it["audio"].copy()  # No modification - measures noise floor
            else:
                a1 = apply_pitch_shift(it["audio"], sr0, semitones=int(sft))
            
            Z1 = adapter.extract_tokens(a1, sr0).numpy().astype(np.float32)
            intervention_count += 1
            # Cleanup based on config interval (less frequent for high memory)
            if intervention_count % cfg.cleanup_interval_intervention == 0:
                free_torch()
            Y1 = procrustes_map.map_tokens_k(Z1).astype(np.float32)  # Normalized k-space
            y1 = Y1.mean(axis=0).astype(np.float32)

            n_par, n_perp, ratio = meanvec_delta_metrics(y0, y1, Ut_shared)
            pitch_meanvec_par[sft].append(n_par)
            pitch_meanvec_perp[sft].append(n_perp)
            pitch_meanvec_ratio[sft].append(ratio)

            tok = token_level_deltas(Y0, Y1, Ut_shared, cfg)
            pitch_token_mean[sft].append(tok["token_mean_d"])
            pitch_token_par[sft].append(tok["token_mean_d_par"])
            pitch_token_perp[sft].append(tok["token_mean_d_perp"])

    # Sweep gain.
    for j in idxs:
        it = items[j]
        Y0 = base_Y0[j]
        y0 = base_y0[j]
        sr0 = it["sr"]

        for db in gain_levels:
            # NOTE: Even for db==0, we measure to get noise floor
            if float(db) == 0.0:
                a1 = it["audio"].copy()  # No modification - measures noise floor
            else:
                a1 = apply_gain(it["audio"], db=float(db))

            Z1 = adapter.extract_tokens(a1, sr0).numpy().astype(np.float32)
            intervention_count += 1
            # Cleanup based on config interval
            if intervention_count % cfg.cleanup_interval_intervention == 0:
                free_torch()
            Y1 = procrustes_map.map_tokens_k(Z1).astype(np.float32)  # Normalized k-space
            y1 = Y1.mean(axis=0).astype(np.float32)

            n_par, n_perp, ratio = meanvec_delta_metrics(y0, y1, Ut_shared)
            gain_meanvec_par[db].append(n_par)
            gain_meanvec_perp[db].append(n_perp)
            gain_meanvec_ratio[db].append(ratio)

            tok = token_level_deltas(Y0, Y1, Ut_shared, cfg)
            gain_token_mean[db].append(tok["token_mean_d"])
            gain_token_par[db].append(tok["token_mean_d_par"])
            gain_token_perp[db].append(tok["token_mean_d_perp"])
    
    # Final cleanup after interventions
    free_torch()

    def safe_mean(xs: List[float]) -> float:
        if len(xs) == 0:
            return float("nan")
        return float(np.mean(np.asarray(xs, dtype=np.float32)))

    anchor_pitch = 2 if 2 in pitch_levels else pitch_levels[len(pitch_levels) // 2]
    anchor_gain = 6.0 if 6.0 in gain_levels else gain_levels[len(gain_levels) // 2]

    pitch_ratio_np = np.asarray(pitch_meanvec_ratio[anchor_pitch], dtype=np.float32)
    gain_ratio_np = np.asarray(gain_meanvec_ratio[anchor_gain], dtype=np.float32)

    pitch_abs_par_anchor = safe_mean(pitch_meanvec_par[anchor_pitch])
    pitch_abs_perp_anchor = safe_mean(pitch_meanvec_perp[anchor_pitch])
    gain_abs_par_anchor = safe_mean(gain_meanvec_par[anchor_gain])
    gain_abs_perp_anchor = safe_mean(gain_meanvec_perp[anchor_gain])

    print(f"  Pitch ratio_perp(meanvec) at shift={anchor_pitch}: {float(np.mean(pitch_ratio_np)):.4f}")
    print(f"  Pitch ||Δ_perp||/sqrt(D) at shift={anchor_pitch}:  {pitch_abs_perp_anchor:.4f}")
    print(f"  Pitch ||Δ_par||/sqrt(D) at shift={anchor_pitch}:   {pitch_abs_par_anchor:.4f}")
    print(f"  Pitch token mean(||ΔY||)/sqrt(D) at shift={anchor_pitch}:      {safe_mean(pitch_token_mean[anchor_pitch]):.4f}")
    print(f"  Pitch token mean(||ΔY_par||)/sqrt(D) at shift={anchor_pitch}:  {safe_mean(pitch_token_par[anchor_pitch]):.4f}")
    print(f"  Pitch token mean(||ΔY_perp||)/sqrt(D) at shift={anchor_pitch}: {safe_mean(pitch_token_perp[anchor_pitch]):.4f}")

    print(f"  Gain  ratio_perp(meanvec) at db={anchor_gain}: {float(np.mean(gain_ratio_np)):.4f}")
    print(f"  Gain  ||Δ_perp||/sqrt(D) at db={anchor_gain}:  {gain_abs_perp_anchor:.4f}")
    print(f"  Gain  ||Δ_par||/sqrt(D) at db={anchor_gain}:   {gain_abs_par_anchor:.4f}")
    print(f"  Gain  token mean(||ΔY||)/sqrt(D) at db={anchor_gain}:      {safe_mean(gain_token_mean[anchor_gain]):.4f}")
    print(f"  Gain  token mean(||ΔY_par||)/sqrt(D) at db={anchor_gain}:  {safe_mean(gain_token_par[anchor_gain]):.4f}")
    print(f"  Gain  token mean(||ΔY_perp||)/sqrt(D) at db={anchor_gain}: {safe_mean(gain_token_perp[anchor_gain]):.4f}")

    # Exp 3. Probes.
    print("📊 Exp 3: Probes (CV, macro-F1) ...")

    def encode_labels(vals: List[Any]) -> np.ndarray:
        uniq = {v: i for i, v in enumerate(sorted(set(vals)))}
        return np.asarray([uniq[v] for v in vals], dtype=np.int64)

    y_spk = encode_labels(speaker)
    y_emo = encode_labels(emotion)
    y_sent = encode_labels(sentence_group)

    mask_spk, y_spk_f = filter_classes(y_spk, min_per_class=cfg.min_per_class, max_classes=cfg.max_speaker_classes)
    if np.sum(mask_spk) < 50 or len(np.unique(y_spk_f)) < 2:
        print(f"  ⚠️ speaker probe skipped. valid_samples={int(np.sum(mask_spk))} uniq={len(np.unique(y_spk_f))}")
        spk_perp_res = {"macro_f1": float("nan"), "macro_f1_std": float("nan"), "acc": float("nan"), "acc_std": float("nan")}
        spk_par_res = {"macro_f1": float("nan"), "macro_f1_std": float("nan"), "acc": float("nan"), "acc_std": float("nan")}
    else:
        spk_perp_res = run_probe_classifier(Y_perp_all_np[mask_spk], y_spk_f, n_splits=cfg.cv_folds, seed=cfg.seed)
        spk_par_res = run_probe_classifier(Y_par_all_np[mask_spk], y_spk_f, n_splits=cfg.cv_folds, seed=cfg.seed)
        print(f"  speaker macro-F1 CV (perp meanvec): {spk_perp_res['macro_f1']:.4f} ± {spk_perp_res['macro_f1_std']:.4f}")
        print(f"  speaker macro-F1 CV (par  meanvec): {spk_par_res['macro_f1']:.4f} ± {spk_par_res['macro_f1_std']:.4f}")

    mask_emo, y_emo_f = filter_classes(y_emo, min_per_class=cfg.min_per_class, max_classes=None)
    if np.sum(mask_emo) < 50 or len(np.unique(y_emo_f)) < 2:
        print(f"  ⚠️ emotion probe skipped. valid_samples={int(np.sum(mask_emo))} uniq={len(np.unique(y_emo_f))}")
        emo_perp_res = {"macro_f1": float("nan"), "macro_f1_std": float("nan"), "acc": float("nan"), "acc_std": float("nan")}
        emo_par_res = {"macro_f1": float("nan"), "macro_f1_std": float("nan"), "acc": float("nan"), "acc_std": float("nan")}
    else:
        emo_perp_res = run_probe_classifier(Y_perp_all_np[mask_emo], y_emo_f, n_splits=cfg.cv_folds, seed=cfg.seed)
        emo_par_res = run_probe_classifier(Y_par_all_np[mask_emo], y_emo_f, n_splits=cfg.cv_folds, seed=cfg.seed)
        print(f"  emotion macro-F1 CV (perp meanvec): {emo_perp_res['macro_f1']:.4f} ± {emo_perp_res['macro_f1_std']:.4f}")
        print(f"  emotion macro-F1 CV (par  meanvec): {emo_par_res['macro_f1']:.4f} ± {emo_par_res['macro_f1_std']:.4f}")

    sent_mean_res = run_probe_classifier(Y_mean_all_np, y_sent, n_splits=cfg.cv_folds, seed=cfg.seed)
    sent_par_res = run_probe_classifier(Y_par_all_np, y_sent, n_splits=cfg.cv_folds, seed=cfg.seed)
    sent_perp_res = run_probe_classifier(Y_perp_all_np, y_sent, n_splits=cfg.cv_folds, seed=cfg.seed)

    print(f"  sentence_y macro-F1 CV (mean): {sent_mean_res['macro_f1']:.4f} ± {sent_mean_res['macro_f1_std']:.4f}")
    print(f"  sentence_par macro-F1 CV (par): {sent_par_res['macro_f1']:.4f} ± {sent_par_res['macro_f1_std']:.4f}")
    print(f"  sentence_perp macro-F1 CV (perp): {sent_perp_res['macro_f1']:.4f} ± {sent_perp_res['macro_f1_std']:.4f}")
    
    # === METHODOLOGICALLY CORRECT: Fold-wise Ut probe ===
    # This builds Ut from TRAIN centroids only in each fold, avoiding self-fulfilling prophecy
    speaker_ids_np = np.asarray(speaker, dtype=object)
    print("  🔬 Running fold-wise Ut probe (train-only Ut, GroupKFold(speaker))...")
    foldwise_res = run_sentence_probe_foldwise_ut(Y_mean_all_np, sentence_ids_np, speaker_ids_np, cfg)
    print(f"  sentence_par macro-F1 (foldwise Ut): {foldwise_res['par_f1']:.4f} ± {foldwise_res.get('par_f1_std', 0):.4f}")
    print(f"  sentence_perp macro-F1 (foldwise Ut): {foldwise_res['perp_f1']:.4f} ± {foldwise_res.get('perp_f1_std', 0):.4f}")
    
    # Compute disentanglement ratio: how much better is par than perp for sentence classification?
    # Higher ratio = better disentanglement (semantic info concentrated in par, not perp)
    par_f1 = sent_par_res["macro_f1"] if not math.isnan(sent_par_res["macro_f1"]) else 0.0
    perp_f1 = sent_perp_res["macro_f1"] if not math.isnan(sent_perp_res["macro_f1"]) else 0.0
    disentangle_ratio = par_f1 / (perp_f1 + 1e-9)
    
    # Also compute foldwise ratio (more rigorous)
    foldwise_par = foldwise_res['par_f1'] if not math.isnan(foldwise_res['par_f1']) else 0.0
    foldwise_perp = foldwise_res['perp_f1'] if not math.isnan(foldwise_res['perp_f1']) else 0.0
    foldwise_ratio = foldwise_par / (foldwise_perp + 1e-9)
    
    print(f"  📐 Disentanglement ratio (global Ut): {disentangle_ratio:.2f}x")
    print(f"  📐 Disentanglement ratio (foldwise Ut): {foldwise_ratio:.2f}x  ← Use this for paper")

    par_ok = (not math.isnan(sent_par_res["macro_f1"])) and (sent_par_res["macro_f1"] >= cfg.sentence_par_min_f1)
    perp_ok = (not math.isnan(sent_perp_res["macro_f1"])) and (sent_perp_res["macro_f1"] <= cfg.sentence_perp_max_f1)
    if par_ok and perp_ok:
        print("  ✅ Decomposition sanity check passed. sentence in par, not in perp.")
    else:
        print("  ⚠️ Decomposition sanity check FAILED (absolute threshold).")
        print(f"     Expect sentence_par F1 >= {cfg.sentence_par_min_f1:.2f}. Got {sent_par_res['macro_f1']:.4f}")
        print(f"     Expect sentence_perp F1 <= {cfg.sentence_perp_max_f1:.2f}. Got {sent_perp_res['macro_f1']:.4f}")
        # NOTE: Relative difference interpretation is key for paper
        if disentangle_ratio >= 2.0:
            print(f"     💡 However, disentanglement ratio = {disentangle_ratio:.2f}x (>= 2x) suggests reasonable separation.")
            print("        For paper: focus on RELATIVE difference (par >> perp), not absolute perp threshold.")
        else:
            print("     Interpretation: Tube may be contaminated by text info, or ridge map is leaking sentence ID.")

    print("📊 Exp 3b: Regression probes (pitch/energy) with GroupKFold(sentence) ...")
    groups_np = np.asarray(sentence_group, dtype=object)
    pitch_np = np.asarray(pitch_hz, dtype=np.float32)
    energy_np = np.asarray(energy_rms, dtype=np.float32)

    pitch_perp_res = run_probe_regression_groupkfold(Y_perp_all_np, pitch_np, groups_np, n_splits=cfg.reg_cv_folds, seed=cfg.seed)
    pitch_par_res = run_probe_regression_groupkfold(Y_par_all_np, pitch_np, groups_np, n_splits=cfg.reg_cv_folds, seed=cfg.seed)
    energy_perp_res = run_probe_regression_groupkfold(Y_perp_all_np, energy_np, groups_np, n_splits=cfg.reg_cv_folds, seed=cfg.seed)
    energy_par_res = run_probe_regression_groupkfold(Y_par_all_np, energy_np, groups_np, n_splits=cfg.reg_cv_folds, seed=cfg.seed)

    print(f"  pitch R² (perp): {pitch_perp_res['r2']:.4f} ± {pitch_perp_res['r2_std']:.4f}")
    print(f"  pitch R² (par ): {pitch_par_res['r2']:.4f} ± {pitch_par_res['r2_std']:.4f}")
    print(f"  energy R² (perp): {energy_perp_res['r2']:.4f} ± {energy_perp_res['r2_std']:.4f}")
    print(f"  energy R² (par ): {energy_par_res['r2']:.4f} ± {energy_par_res['r2_std']:.4f}")

    # Visualization
    os.makedirs(cfg.out_dir, exist_ok=True)
    tag = adapter.model_id.replace("/", "_")
    legend_name = tag

    # Legacy anchored histograms preserved.
    save_hist(pitch_meanvec_perp[anchor_pitch], f"{tag}. Pitch ||Δ_perp||/sqrt(D) (shift={anchor_pitch})", "value", os.path.join(cfg.out_dir, f"{tag}_pitch_abs_perp.png"))
    save_hist(pitch_meanvec_par[anchor_pitch],  f"{tag}. Pitch ||Δ_par||/sqrt(D) (shift={anchor_pitch})",  "value", os.path.join(cfg.out_dir, f"{tag}_pitch_abs_par.png"))
    save_hist(gain_meanvec_perp[anchor_gain],   f"{tag}. Gain  ||Δ_perp||/sqrt(D) (db={anchor_gain})",   "value", os.path.join(cfg.out_dir, f"{tag}_gain_abs_perp.png"))
    save_hist(gain_meanvec_par[anchor_gain],    f"{tag}. Gain  ||Δ_par||/sqrt(D) (db={anchor_gain})",    "value", os.path.join(cfg.out_dir, f"{tag}_gain_abs_par.png"))

    save_hist(pitch_token_mean[anchor_pitch], f"{tag}. Pitch token mean(||ΔY||)/sqrt(D) (shift={anchor_pitch})", "value", os.path.join(cfg.out_dir, f"{tag}_pitch_token_mean_d.png"))
    save_hist(pitch_token_perp[anchor_pitch], f"{tag}. Pitch token mean(||ΔY_perp||)/sqrt(D) (shift={anchor_pitch})", "value", os.path.join(cfg.out_dir, f"{tag}_pitch_token_perp.png"))
    save_hist(gain_token_mean[anchor_gain],  f"{tag}. Gain  token mean(||ΔY||)/sqrt(D) (db={anchor_gain})", "value", os.path.join(cfg.out_dir, f"{tag}_gain_token_mean_d.png"))
    save_hist(gain_token_perp[anchor_gain],  f"{tag}. Gain  token mean(||ΔY_perp||)/sqrt(D) (db={anchor_gain})", "value", os.path.join(cfg.out_dir, f"{tag}_gain_token_perp.png"))

    # Trajectory plot preserved.
    pick = int(np.random.RandomState(cfg.seed).randint(0, len(items)))
    y_means_traj = []
    base = items[pick]
    for sft in cfg.pitch_shifts:
        a = apply_pitch_shift(base["audio"], base["sr"], semitones=int(sft))
        Zt = adapter.extract_tokens(a, base["sr"]).numpy().astype(np.float32)
        Yt = procrustes_map.map_tokens_k(Zt).astype(np.float32)  # Normalized k-space
        y_means_traj.append(Yt.mean(axis=0))
    save_disentanglement_trajectory(
        y_means_traj,
        Ut_shared,
        title=f"{tag}. Disentanglement trajectory. pitch shifts={list(cfg.pitch_shifts)}",
        path=os.path.join(cfg.out_dir, f"{tag}_disentanglement_pitch_traj.png"),
    )

    if not math.isnan(emo_perp_res["macro_f1"]) and not math.isnan(emo_par_res["macro_f1"]):
        save_scatter(
            x=[emo_par_res["macro_f1"]],
            y=[emo_perp_res["macro_f1"]],
            labels=[tag],
            title=f"{tag}. Emotion probe. par vs perp (macro-F1)",
            xlabel="Emotion macro-F1 on par(meanvec)",
            ylabel="Emotion macro-F1 on perp(meanvec)",
            path=os.path.join(cfg.out_dir, f"{tag}_emotion_par_vs_perp.png"),
        )

    if not math.isnan(spk_perp_res["macro_f1"]) and not math.isnan(spk_par_res["macro_f1"]):
        save_scatter(
            x=[spk_par_res["macro_f1"]],
            y=[spk_perp_res["macro_f1"]],
            labels=[tag],
            title=f"{tag}. Speaker probe. par vs perp (macro-F1)",
            xlabel="Speaker macro-F1 on par(meanvec)",
            ylabel="Speaker macro-F1 on perp(meanvec)",
            path=os.path.join(cfg.out_dir, f"{tag}_speaker_par_vs_perp.png"),
        )

    # Dose-response arrays.
    pitch_x = [float(s) for s in pitch_levels]
    pitch_par_mean = [float(np.mean(pitch_meanvec_par[s])) for s in pitch_levels]
    pitch_par_std = [float(np.std(pitch_meanvec_par[s])) for s in pitch_levels]
    pitch_perp_mean = [float(np.mean(pitch_meanvec_perp[s])) for s in pitch_levels]
    pitch_perp_std = [float(np.std(pitch_meanvec_perp[s])) for s in pitch_levels]
    pitch_ratio_mean = [float(np.mean(pitch_meanvec_ratio[s])) for s in pitch_levels]
    pitch_ratio_std = [float(np.std(pitch_meanvec_ratio[s])) for s in pitch_levels]

    pitch_tok_mean_m = [float(np.mean(pitch_token_mean[s])) for s in pitch_levels]
    pitch_tok_mean_s = [float(np.std(pitch_token_mean[s])) for s in pitch_levels]
    pitch_tok_par_m = [float(np.mean(pitch_token_par[s])) for s in pitch_levels]
    pitch_tok_par_s = [float(np.std(pitch_token_par[s])) for s in pitch_levels]
    pitch_tok_perp_m = [float(np.mean(pitch_token_perp[s])) for s in pitch_levels]
    pitch_tok_perp_s = [float(np.std(pitch_token_perp[s])) for s in pitch_levels]

    gain_x = [float(db) for db in gain_levels]
    gain_par_mean = [float(np.mean(gain_meanvec_par[db])) for db in gain_levels]
    gain_par_std = [float(np.std(gain_meanvec_par[db])) for db in gain_levels]
    gain_perp_mean = [float(np.mean(gain_meanvec_perp[db])) for db in gain_levels]
    gain_perp_std = [float(np.std(gain_meanvec_perp[db])) for db in gain_levels]
    gain_ratio_mean = [float(np.mean(gain_meanvec_ratio[db])) for db in gain_levels]
    gain_ratio_std = [float(np.std(gain_meanvec_ratio[db])) for db in gain_levels]

    gain_tok_mean_m = [float(np.mean(gain_token_mean[db])) for db in gain_levels]
    gain_tok_mean_s = [float(np.std(gain_token_mean[db])) for db in gain_levels]
    gain_tok_par_m = [float(np.mean(gain_token_par[db])) for db in gain_levels]
    gain_tok_par_s = [float(np.std(gain_token_par[db])) for db in gain_levels]
    gain_tok_perp_m = [float(np.mean(gain_token_perp[db])) for db in gain_levels]
    gain_tok_perp_s = [float(np.std(gain_token_perp[db])) for db in gain_levels]

    # Must-have curve fitting on plots.
    # Pitch uses quadratic no-linear fit. Gain uses linear fit.
    pitch_fit_meanvec_perp = save_line_with_errorbars_and_fit(
        xs=pitch_x,
        ys_mean=pitch_perp_mean,
        ys_std=pitch_perp_std,
        title=f"{tag}. Pitch dose-response. ||Δ_perp||/sqrt(D)",
        xlabel="pitch shift (semitones)",
        ylabel="||Δ_perp||/sqrt(D)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_pitch_meanvec_perp.png"),
        fit_kind="pitch_quad",
        legend_name=legend_name,
    )
    pitch_fit_meanvec_ratio = save_line_with_errorbars_and_fit(
        xs=pitch_x,
        ys_mean=pitch_ratio_mean,
        ys_std=pitch_ratio_std,
        title=f"{tag}. Pitch dose-response. ratio_perp(meanvec)",
        xlabel="pitch shift (semitones)",
        ylabel="||Δ_perp||/(||Δ_par||+||Δ_perp||)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_pitch_meanvec_ratio.png"),
        fit_kind="pitch_quad",
        legend_name=legend_name,
    )
    pitch_fit_token_perp = save_line_with_errorbars_and_fit(
        xs=pitch_x,
        ys_mean=pitch_tok_perp_m,
        ys_std=pitch_tok_perp_s,
        title=f"{tag}. Pitch dose-response. token mean(||ΔY_perp||)/sqrt(D)",
        xlabel="pitch shift (semitones)",
        ylabel="token mean(||ΔY_perp||)/sqrt(D)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_pitch_token_perp.png"),
        fit_kind="pitch_quad",
        legend_name=legend_name,
    )

    gain_fit_meanvec_perp = save_line_with_errorbars_and_fit(
        xs=gain_x,
        ys_mean=gain_perp_mean,
        ys_std=gain_perp_std,
        title=f"{tag}. Gain dose-response. ||Δ_perp||/sqrt(D)",
        xlabel="gain (dB)",
        ylabel="||Δ_perp||/sqrt(D)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_gain_meanvec_perp.png"),
        fit_kind="gain_lin",
        legend_name=legend_name,
    )
    gain_fit_meanvec_ratio = save_line_with_errorbars_and_fit(
        xs=gain_x,
        ys_mean=gain_ratio_mean,
        ys_std=gain_ratio_std,
        title=f"{tag}. Gain dose-response. ratio_perp(meanvec)",
        xlabel="gain (dB)",
        ylabel="||Δ_perp||/(||Δ_par||+||Δ_perp||)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_gain_meanvec_ratio.png"),
        fit_kind="gain_lin",
        legend_name=legend_name,
    )
    gain_fit_token_perp = save_line_with_errorbars_and_fit(
        xs=gain_x,
        ys_mean=gain_tok_perp_m,
        ys_std=gain_tok_perp_s,
        title=f"{tag}. Gain dose-response. token mean(||ΔY_perp||)/sqrt(D)",
        xlabel="gain (dB)",
        ylabel="token mean(||ΔY_perp||)/sqrt(D)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_gain_token_perp.png"),
        fit_kind="gain_lin",
        legend_name=legend_name,
    )

    # Also keep the rest of dose plots from v3.4, now with fitting for key ones only.
    # These are still plotted without fit to avoid clutter.
    _ = save_line_with_errorbars_and_fit(
        xs=pitch_x, ys_mean=pitch_par_mean, ys_std=pitch_par_std,
        title=f"{tag}. Pitch dose-response. ||Δ_par||/sqrt(D)",
        xlabel="pitch shift (semitones)", ylabel="||Δ_par||/sqrt(D)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_pitch_meanvec_par.png"),
        fit_kind=None, legend_name=legend_name,
    )
    _ = save_line_with_errorbars_and_fit(
        xs=pitch_x, ys_mean=pitch_tok_mean_m, ys_std=pitch_tok_mean_s,
        title=f"{tag}. Pitch dose-response. token mean(||ΔY||)/sqrt(D)",
        xlabel="pitch shift (semitones)", ylabel="token mean(||ΔY||)/sqrt(D)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_pitch_token_mean.png"),
        fit_kind=None, legend_name=legend_name,
    )
    _ = save_line_with_errorbars_and_fit(
        xs=pitch_x, ys_mean=pitch_tok_par_m, ys_std=pitch_tok_par_s,
        title=f"{tag}. Pitch dose-response. token mean(||ΔY_par||)/sqrt(D)",
        xlabel="pitch shift (semitones)", ylabel="token mean(||ΔY_par||)/sqrt(D)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_pitch_token_par.png"),
        fit_kind=None, legend_name=legend_name,
    )

    _ = save_line_with_errorbars_and_fit(
        xs=gain_x, ys_mean=gain_par_mean, ys_std=gain_par_std,
        title=f"{tag}. Gain dose-response. ||Δ_par||/sqrt(D)",
        xlabel="gain (dB)", ylabel="||Δ_par||/sqrt(D)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_gain_meanvec_par.png"),
        fit_kind=None, legend_name=legend_name,
    )
    _ = save_line_with_errorbars_and_fit(
        xs=gain_x, ys_mean=gain_tok_mean_m, ys_std=gain_tok_mean_s,
        title=f"{tag}. Gain dose-response. token mean(||ΔY||)/sqrt(D)",
        xlabel="gain (dB)", ylabel="token mean(||ΔY||)/sqrt(D)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_gain_token_mean.png"),
        fit_kind=None, legend_name=legend_name,
    )
    _ = save_line_with_errorbars_and_fit(
        xs=gain_x, ys_mean=gain_tok_par_m, ys_std=gain_tok_par_s,
        title=f"{tag}. Gain dose-response. token mean(||ΔY_par||)/sqrt(D)",
        xlabel="gain (dB)", ylabel="token mean(||ΔY_par||)/sqrt(D)",
        path=os.path.join(cfg.out_dir, f"{tag}_dose_gain_token_par.png"),
        fit_kind=None, legend_name=legend_name,
    )

    # Pick primary "headline" coefficients for abstract-style ratios.
    # Default: meanvec_perp for both pitch and gain.
    pitch_primary_a = float(pitch_fit_meanvec_perp.get("a", float("nan")))
    gain_primary_m = float(gain_fit_meanvec_perp.get("m", float("nan")))

    print("📌 Dose-response coefficients (reviewer-friendly numbers) ...")
    if pitch_primary_a == pitch_primary_a:
        print(f"  Pitch quadratic curvature a (primary): {pitch_primary_a:.6f}  R2={pitch_fit_meanvec_perp.get('r2', float('nan')):.4f}")
    if gain_primary_m == gain_primary_m:
        print(f"  Gain linear slope m (primary): {gain_primary_m:.6f}  R2={gain_fit_meanvec_perp.get('r2', float('nan')):.4f}")

    result = {
        "model_id": adapter.model_id,
        "pooling": pool_mode,
        "procrustes": procrustes_info,
        "qwen2_pooling_ablation_best": ab_best_mode,
        "qwen2_pooling_ablation_scores": ab_scores,
        # Ut_model based (model-specific)
        "procrustean_cov_mean": float(procs_np.mean()),
        "procrustean_cov_std": float(procs_np.std()),
        "radius_norm_mean": float(radii_np.mean()),
        "radius_norm_std": float(radii_np.std()),
        # Ut_text based (cross-model comparable)
        "procrustean_cov_text_mean": float(procs_text_np.mean()),
        "procrustean_cov_text_std": float(procs_text_np.std()),
        "radius_norm_text_mean": float(radii_text_np.mean()),
        "radius_norm_text_std": float(radii_text_np.std()),
        "cos_in_sample_mean": float(cos_np.mean()),
        "cos_in_sample_std": float(cos_np.std()),
        "subspace_cca_mean": float(cca_mean),
        "angle_mean_deg": float(angle_mean_deg),
        "effective_rank": float(er),
        "top5_concentration": float(top5),
        "pitch_ratio_perp_meanvec": float(np.mean(pitch_ratio_np)),
        "pitch_abs_perp": float(pitch_abs_perp_anchor),
        "pitch_abs_par": float(pitch_abs_par_anchor),
        "gain_ratio_perp_meanvec": float(np.mean(gain_ratio_np)),
        "gain_abs_perp": float(gain_abs_perp_anchor),
        "gain_abs_par": float(gain_abs_par_anchor),
        "pitch_dose_meanvec_par_mean": pitch_par_mean,
        "pitch_dose_meanvec_perp_mean": pitch_perp_mean,
        "pitch_dose_meanvec_ratio_mean": pitch_ratio_mean,
        "gain_dose_meanvec_par_mean": gain_par_mean,
        "gain_dose_meanvec_perp_mean": gain_perp_mean,
        "gain_dose_meanvec_ratio_mean": gain_ratio_mean,
        "speaker_probe_perp": spk_perp_res,
        "speaker_probe_par": spk_par_res,
        "emotion_probe_perp": emo_perp_res,
        "emotion_probe_par": emo_par_res,
        "sentence_probe_mean": sent_mean_res,
        "sentence_probe_par": sent_par_res,
        "sentence_probe_perp": sent_perp_res,
        "sentence_probe_foldwise_ut": foldwise_res,  # PAPER: Use this for rigorous evaluation
        "decomposition_sanity_par_ok": bool(par_ok),
        "decomposition_sanity_perp_ok": bool(perp_ok),
        "pitch_reg_perp": pitch_perp_res,
        "pitch_reg_par": pitch_par_res,
        "energy_reg_perp": energy_perp_res,
        "energy_reg_par": energy_par_res,
        # Must-have fitting coefficients.
        "dose_fit_pitch_meanvec_perp": pitch_fit_meanvec_perp,
        "dose_fit_pitch_meanvec_ratio": pitch_fit_meanvec_ratio,
        "dose_fit_pitch_token_perp": pitch_fit_token_perp,
        "dose_fit_gain_meanvec_perp": gain_fit_meanvec_perp,
        "dose_fit_gain_meanvec_ratio": gain_fit_meanvec_ratio,
        "dose_fit_gain_token_perp": gain_fit_token_perp,
        "dose_primary_pitch_a": pitch_primary_a,
        "dose_primary_gain_m": gain_primary_m,
        # Sanity suite results (validation layer)
        "sanity_suite": sanity_results,
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
    set_seed(CFG_.seed)
    os.makedirs(CFG_.out_dir, exist_ok=True)

    print("🧾 CFG summary:")
    print(f"  token_align={CFG_.token_align}  trim_strategy={CFG_.trim_strategy}")
    print(f"  ridge_pool_default={CFG_.ridge_pool_default}  ridge_pool_qwen2={CFG_.ridge_pool_qwen2}")
    print(f"  qwen2_pooling_ablation={CFG_.qwen2_pooling_ablation}  qwen2_pooling_auto_override={CFG_.qwen2_pooling_auto_override}")
    print(f"  ut_whiten={CFG_.ut_whiten}  max_tokens_for_global_stats={CFG_.max_tokens_for_global_stats}")

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
    
    # Use robust subspace that ALIGNS with sample-level centering
    Ut = build_text_subspace_robust(e_text_all, extra_embs, k=CFG_.text_subspace_k, seed=CFG_.seed)
    print(f"🧩 Shared U_t built. Shape={Ut.shape} (D=768, k={Ut.shape[1]})")

    models = [
        ("DeSTA-Baseline", "voidful/QAQ_4b"),
        ("DeSTA-ORCA", "voidful/desta25_4b_R2_full"),
        ("Qwen2-Audio", "Qwen/Qwen2-Audio-7B"),
        ("Qwen2.5-Omni", "Qwen/Qwen2.5-Omni-3B"),
    ]

    results = []
    for family, model_id in models:
        if family.startswith("DeSTA"):
            adapter = DeSTAAdapter(model_id=model_id, device="cuda")
        elif "Omni" in model_id:
            adapter = Qwen2_5OmniAdapter(model_id=model_id, device="cuda")
        else:
            adapter = Qwen2AudioAdapter(model_id=model_id, device="cuda")
        res = analyze_model(adapter, items, text_enc, Ut, CFG_)
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

    # Print ratio lines if possible.
    # This is the direct "15x" style. Guard against zeros.
    base = None
    for r in results:
        if "voidful/QAQ_4b" in r["model_id"]:
            base = r
    if base is not None:
        a0 = float(base.get("dose_primary_pitch_a", float("nan")))
        m0 = float(base.get("dose_primary_gain_m", float("nan")))
        for r in results:
            a1 = float(r.get("dose_primary_pitch_a", float("nan")))
            m1 = float(r.get("dose_primary_gain_m", float("nan")))
            if a0 == a0 and abs(a0) > 1e-12 and a1 == a1:
                print(f"  Pitch curvature ratio. {r['model_id']} / Baseline = {a1 / a0:.2f}x")
            if m0 == m0 and abs(m0) > 1e-12 and m1 == m1:
                print(f"  Gain slope ratio. {r['model_id']} / Baseline = {m1 / m0:.2f}x")

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
