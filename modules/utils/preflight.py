#!/usr/bin/env python3
"""Preflight sanity checks — fail in seconds, not 30 minutes into a full run.

Beyond input_validation (which checks the user's data files), these two checks guard
the two failure modes that otherwise only surface deep inside the async pipeline:

  1. validate_model_artifacts  — every model / PCA / classifier artifact the chosen
     backend needs is present and non-empty on disk. Catches committed artifacts that
     were dropped from the repo (e.g. by a broad ``*.json`` .gitignore rule).

  2. preflight_embedding_check — the RNA foundation model actually produces *finite*
     embeddings on THIS machine. Catches broken weight loads that yield NaN, e.g. the
     CUDA-without-flash-attn trap where fused Wqkv weights are never converted to the
     separate Q/K/V that standard attention expects.

Both raise ValidationError (from input_validation) so curia.py can treat them uniformly.
"""

from pathlib import Path

from modules.model_registry import (
    get_model_config,
    get_pca_path,
    get_finding_pca_path,
    get_logreg_path,
    get_rna_toga_model_path,
    load_model,
)
from modules.utils.input_validation import ValidationError


def validate_model_artifacts(model_name: str) -> None:
    """Assert every on-disk artifact the backend needs exists and is non-empty."""
    cfg = get_model_config(model_name)
    checks = {
        "matching PCA": get_pca_path(model_name),
        "signal/noise classifier": get_logreg_path(model_name),
        "orthology model (rna_toga)": get_rna_toga_model_path(model_name),
    }
    if cfg.get("finding_pca_file"):  # only distinct when the model declares one
        checks["finding PCA"] = get_finding_pca_path(model_name)

    print(f"# Checking model artifacts for backend '{model_name}'...")
    missing = []
    for label, path in checks.items():
        p = Path(path)
        ok = p.is_file() and p.stat().st_size > 0
        print(f"#   {'✓' if ok else '✗'} {label}: {p}")
        if not ok:
            missing.append(f"    {label}: {p}")

    if missing:
        raise ValidationError(
            "Required model artifact(s) missing or empty:\n"
            + "\n".join(missing)
            + "\n  These ship committed under modules/. If you cloned the repo and they are\n"
            "  absent, they were likely dropped by a .gitignore rule — re-fetch via git,\n"
            "  or (re)run ./install.sh."
        )
    print("#   ✓ All model artifacts present\n")


def preflight_embedding_check(
    model_name: str,
    device_pref: str = "auto",
    seqs=("ACGUACGUACGUACGUACGU", "GGGGCCCCAAAAUUUUGCGC"),
) -> None:
    """Load the FM once and embed a couple of dummy RNAs; assert finite + right dim.

    Uses the same load path as the GPU executor, so a broken load (NaN embeddings)
    is caught up front. The model is freed before the executor starts, so no VRAM is
    held. The GPU executor uses a 'spawn' context, so initialising CUDA here is safe.
    """
    import torch
    from modules.GPU_executor.gpu_executor import get_device

    cfg = get_model_config(model_name)
    device = get_device(device_pref)
    print(f"# Preflight embedding self-test ({model_name} on {device})...")

    model = emb = tokens = None
    try:
        try:
            model, tokenize, extract = load_model(model_name, device)
            with torch.no_grad():
                tokens = tokenize(list(seqs))
                emb = extract(model, tokens)
        except Exception as e:
            raise ValidationError(f"Foundation model failed to load/embed a test sequence: {e}")

        emb_cpu = emb.detach().float().cpu()
        exp_dim = cfg.get("emb_dim")
        finite = bool(torch.isfinite(emb_cpu).all())
        nan_frac = float(torch.isnan(emb_cpu).float().mean())
        ok_shape = emb_cpu.ndim == 3 and (exp_dim is None or emb_cpu.shape[-1] == exp_dim)
        print(f"#   output shape={tuple(emb_cpu.shape)} (expected last dim {exp_dim}) | "
              f"finite={finite} | nan_frac={nan_frac:.3g}")

        if not finite:
            raise ValidationError(
                f"Foundation-model embeddings contain NaN/Inf on {device} (nan_frac={nan_frac:.3g}).\n"
                "  This is a broken weight load, not a data problem. Common cause on CUDA: RiNALMo\n"
                "  is built for flash-attention but flash-attn is not installed, so the fused Wqkv\n"
                "  weights are never converted to the separate Q/K/V that standard attention needs.\n"
                "  Fix: install flash-attn, OR ensure pretrained loading ties use_flash to flash-attn\n"
                "  availability (not just torch.cuda.is_available())."
            )
        if not ok_shape:
            raise ValidationError(
                f"Foundation-model output has unexpected shape {tuple(emb_cpu.shape)} "
                f"(expected 3-D with last dim {exp_dim}). Model/config mismatch."
            )
        print("#   ✓ Embeddings are finite and correctly shaped\n")
    finally:
        del model, emb, tokens
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
