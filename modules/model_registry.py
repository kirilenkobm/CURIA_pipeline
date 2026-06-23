"""
Model registry for RNA foundation models.

Maps model names to their properties (embedding dim, PCA config, load function).
Used by fit_pca, apply_pca, gpu_executor, and curia.py.

Usage:
    from modules.model_registry import get_model_config, load_model

    cfg = get_model_config("rinalmo")
    model, tokenize_fn, extract_fn = load_model("rinalmo", device)
"""

import sys
from pathlib import Path

import torch

MODULES_DIR = Path(__file__).resolve().parent

MODELS = {
    "rnafm": {
        "module_path": "RNA-FM",
        "pca_file": "rnafm_pca_k16.npz",
        "logreg_file": "logreg_noise_model.json",
        "emb_dim": 640,
        "pca_components": 16,
        # RNA-FM embeddings drift with flanking context, so every sliding
        # window must be re-embedded in its own short context.
        "embed_strategy": "windowed",
        # Island finding (reference/query scanners). window stays 72 to match
        # the logreg classifier input; small stride compensates for context
        # drift.
        "island_scan": {"window_size": 72, "stride": 16, "prob_threshold": 0.25},
        # Island alignment (windowed MMD + Smith-Waterman). Tuned for RNA-FM
        # 16-dim PCA space.
        "island_align": {
            "window_size": 96,
            "stride": 4,
            "sw_tau": 0.15,
            "mean_dist_threshold": 3.0,
            "max_match_mmd": 0.15,
            "min_island_len": 72,
        },
    },
    "rinalmo": {
        "module_path": "RiNALMo",
        # k=16 gives the best signal/noise AUC (0.9733) and best MMD
        # discrimination of all k tried (see rinalmo_signal_noise.ipynb /
        # rinalmo_pca_calibration.ipynb); higher k adds variance but dilutes
        # the discriminative signal.
        "pca_file": "rinalmo_pca_k16.npz",
        "logreg_file": "logreg_noise_model_rinalmo.json",
        "emb_dim": 1280,
        "pca_components": 16,
        # RiNALMo is context-stable (~0 MMD drift under flanking context, see
        # context_dependency.ipynb), so an island can be embedded ONCE and the
        # per-token embeddings sliced into windows locally — no re-embedding.
        "embed_strategy": "embed_once",
        # Larger stride than RNA-FM: context stability removes the need for
        # dense overlap during finding.
        "island_scan": {"window_size": 72, "stride": 32, "prob_threshold": 0.25},
        # Tuned for RiNALMo 16-dim PCA space. mean_dist_threshold is larger
        # because RiNALMo PCA space has a wider scale (see calibration notebook).
        # NOTE: window/stride/thresholds for the wide-window regime are seeded
        # from notebook calibration and may be refined by a dedicated sweep.
        "island_align": {
            "window_size": 96,
            "stride": 16,
            "sw_tau": 0.11,
            "mean_dist_threshold": 9.6,
            "max_match_mmd": 0.12,
            "min_island_len": 72,
        },
    },
}

DEFAULT_MODEL = "rinalmo"


def get_model_config(name=None):
    """Return config dict for the given model name."""
    name = name or DEFAULT_MODEL
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name]


def get_pca_path(model_name=None):
    """Return absolute path to the PCA .npz file for the given model."""
    cfg = get_model_config(model_name)
    return MODULES_DIR / "global_PCA" / cfg["pca_file"]


def get_logreg_path(model_name=None):
    """Return absolute path to the logreg JSON for the given model."""
    cfg = get_model_config(model_name)
    return MODULES_DIR / "logreg_signal_noise" / cfg["logreg_file"]


def get_embed_strategy(model_name=None):
    """Return 'windowed' or 'embed_once' for the given model."""
    return get_model_config(model_name).get("embed_strategy", "windowed")


def get_island_align_params(model_name=None):
    """Return the island-alignment tuning params dict for the given model."""
    return dict(get_model_config(model_name)["island_align"])


def get_island_scan_params(model_name=None):
    """Return the island-finding (scanner) tuning params dict for the model."""
    return dict(get_model_config(model_name)["island_scan"])


def load_model(model_name, device):
    """Load a model and return (model, tokenize_fn, extract_fn).

    tokenize_fn(sequences: list[str]) -> batch_tokens (torch.Tensor on device)
        Accepts a list of RNA strings (ACGU), returns padded token tensor.

    extract_fn(model, batch_tokens) -> representations (torch.Tensor)
        Runs inference and returns per-token embeddings (B, L_max, D).
        Caller is responsible for slicing to actual sequence lengths.
    """
    name = model_name or DEFAULT_MODEL
    if name == "rnafm":
        return _load_rnafm(device)
    elif name == "rinalmo":
        return _load_rinalmo(device)
    else:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")


def _load_rnafm(device):
    rnafm_dir = MODULES_DIR / "RNA-FM"
    if str(rnafm_dir) not in sys.path:
        sys.path.insert(0, str(rnafm_dir))
    import fm

    model, alphabet = fm.pretrained.rna_fm_t12()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    def tokenize(sequences):
        data = [(f"seq_{i}", s) for i, s in enumerate(sequences)]
        _, _, tokens = batch_converter(data)
        return tokens.to(device)

    def extract(mdl, tokens):
        out = mdl(tokens, repr_layers=[12])
        return out["representations"][12].float()

    return model, tokenize, extract


def _load_rinalmo(device):
    rinalmo_dir = MODULES_DIR / "RiNALMo"
    if str(rinalmo_dir) not in sys.path:
        sys.path.insert(0, str(rinalmo_dir))
    from rinalmo.pretrained import get_pretrained_model

    model, alpha = get_pretrained_model(model_name="giga-v1")
    model.eval().to(device)

    def tokenize(sequences):
        tokens = alpha.batch_tokenize(sequences)
        return torch.tensor(tokens, dtype=torch.int64, device=device)

    def extract(mdl, tokens):
        out = mdl(tokens)
        return out["representation"].float()

    return model, tokenize, extract
