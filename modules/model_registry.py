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
        # Orthology classifier (rna_toga): the legacy 3-feature logreg stays on
        # the deprecated RNA-FM path.
        "rna_toga_model": "model.json",
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
        # MATCHING PCA. k=16 gives the best MMD discrimination for the island
        # *matching* step (see calibration_provenance.ipynb).
        # It is tuned for matching, NOT finding: a deployment-faithful scan
        # benchmark (notebooks/finding_benchmark.ipynb) showed k=16 is the
        # bottleneck for the *finding* scanner (~0.06 detection at 10% background-FP).
        # Finding uses a separate, higher-dimensional projection (finding_pca_file).
        "pca_file": "rinalmo_pca_k16.npz",
        # FINDING PCA. Separate k=64 projection fit on mean-pooled window
        # embeddings for the signal/noise scanner (~0.71 detection at 10%
        # background-FP, ~12x over k=16). Applied by the GPU executor on the
        # mean_pool/finding path only; matching keeps pca_file (k=16).
        # Built by modules/logreg_signal_noise/build_rinalmo_finding.py.
        "finding_pca_file": "rinalmo_pca_find_k64.npz",
        "logreg_file": "logreg_noise_model_rinalmo.json",
        # Orthology classifier (rna_toga): the RiNALMo path uses the GBM model
        # (see modules/rna_toga/train_lncrna_gbm.py). Falls back to the legacy
        # logreg model.json automatically if gbm_model.json is not present.
        "rna_toga_model": "gbm_model.json",
        "emb_dim": 1280,
        "pca_components": 16,
        # RiNALMo is context-stable (~0 MMD drift under flanking context, see
        # context_dependency.ipynb), so a matching island can be embedded ONCE
        # and the per-token embeddings sliced into windows locally. NOTE: this
        # governs the *matching* path only; the finding scanner already embeds
        # each window in isolation (embed-once slicing hurts finding — it leaks
        # flank context into the window mean-pool, see finding_benchmark.ipynb).
        "embed_strategy": "embed_once",
        # FINDING scan params. W=128 / stride~W/3 with overlap-labeled training
        # ("window positive if it covers >=20nt of a structured element") maximise
        # recall (see notebooks/finding_benchmark.ipynb). The finding
        # classifier (logreg_noise_model_rinalmo.json, feature_dim=64, lncRNA
        # excluded) MUST be trained at this same window_size. prob_threshold
        # calibrated by build_rinalmo_finding.py on held-out loci: 0.5 -> detection
        # 0.76 (>=50% gene coverage) at 17% background-FP (recall-first, <=20%
        # budget); use 0.575 for a <=10% budget (detection 0.63). See the logreg
        # JSON provenance.calibration.
        "island_scan": {"window_size": 128, "stride": 40, "prob_threshold": 0.5},
        # Island MATCHING via the RiNALMo dotplot matcher (matchers/rinalmo.py):
        # embed each island once -> per-token cosine dotplot (deployed k16
        # matching PCA) -> nucleotide-resolution Smith-Waterman. Beats window-MMD
        # on accuracy and cost in the flank-diluted regime (see
        # notebooks/matching_benchmark.ipynb: AUC 0.993 vs 0.651, 100% core
        # localization, ~0.9 ms/pair). tau_cos/gap are benchmark-validated.
        # Quality = 1/(1+SW_score) (lower=better; the score integrates similarity
        # over band length and is the discriminator). max_match_dist=0.1
        # (i.e. SW score >= ~9) calibrated on Rfam same/cross-family pairs:
        # with min_match_eff_nt=40 it gives TPR 0.98 / FPR 0.01.
        "island_align": {
            "min_island_len": 72,
            "sw_tau_cos": 0.5,
            "sw_gap": 0.3,
            "max_match_dist": 0.1,
            "min_match_eff_nt": 40,
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
    """Return absolute path to the PCA .npz file for the given model.

    This is the MATCHING PCA (used by the island-alignment / per-token path).
    """
    cfg = get_model_config(model_name)
    return MODULES_DIR / "global_PCA" / cfg["pca_file"]


def get_finding_pca_path(model_name=None):
    """Return absolute path to the FINDING PCA .npz for the given model.

    The signal/noise island scanner needs a richer projection than the k=16
    matching PCA. Models may declare a dedicated ``finding_pca_file``; if absent
    we fall back to ``pca_file`` so models without a finding-specific PCA (e.g.
    RNA-FM) behave exactly as before.
    """
    cfg = get_model_config(model_name)
    finding_file = cfg.get("finding_pca_file")
    if finding_file is None:
        return get_pca_path(model_name)
    return MODULES_DIR / "global_PCA" / finding_file


def get_logreg_path(model_name=None):
    """Return absolute path to the logreg JSON for the given model."""
    cfg = get_model_config(model_name)
    return MODULES_DIR / "logreg_signal_noise" / cfg["logreg_file"]


def get_rna_toga_model_path(model_name=None):
    """Return absolute path to the rna_toga orthology model JSON for the given backend.

    rinalmo -> gbm_model.json, rnafm -> model.json. Falls back to the legacy
    model.json if the configured file does not exist yet (keeps the pipeline
    working before the GBM is trained / while the flip is under review).
    """
    cfg = get_model_config(model_name)
    fname = cfg.get("rna_toga_model", "model.json")
    path = MODULES_DIR / "rna_toga" / fname
    if not path.exists():
        return MODULES_DIR / "rna_toga" / "model.json"
    return path


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
