from __future__ import annotations

from pathlib import Path
from typing import Dict


def _sizes_from_twobit_obj(obj) -> Dict[str, int] | None:
    if isinstance(obj, dict):
        return {k: int(v) for k, v in obj.items()}

    chroms = getattr(obj, "chroms", None)
    if callable(chroms):
        res = chroms()
        if isinstance(res, dict):
            return {k: int(v) for k, v in res.items()}
        if isinstance(res, (list, tuple)):
            size_fn = getattr(obj, "chrom_size", None)
            if callable(size_fn):
                return {name: int(size_fn(name)) for name in res}

    chrom_sizes = getattr(obj, "chrom_sizes", None)
    if isinstance(chrom_sizes, dict):
        return {k: int(v) for k, v in chrom_sizes.items()}

    return None


def _try_pyrion(twobit_path: str) -> Dict[str, int] | None:
    try:
        import pyrion  # type: ignore
    except Exception:
        return None

    for name in ("read_2bit_file", "read_2bit", "read_twobit", "read_two_bit"):
        fn = getattr(pyrion, name, None)
        if callable(fn):
            obj = fn(twobit_path)
            sizes = _sizes_from_twobit_obj(obj)
            if sizes:
                return sizes

    try:
        from pyrion.core.twobit import TwoBit  # type: ignore

        obj = TwoBit(twobit_path)
        sizes = _sizes_from_twobit_obj(obj)
        if sizes:
            return sizes
    except Exception:
        pass

    return None


def _try_py2bit(twobit_path: str) -> Dict[str, int] | None:
    try:
        import py2bit  # type: ignore
    except Exception:
        return None

    tb = py2bit.open(twobit_path)
    try:
        sizes = tb.chroms()
        return {k: int(v) for k, v in sizes.items()}
    finally:
        tb.close()


def write_chrom_sizes_from_2bit(twobit_path: str, output_path: str) -> Dict[str, int]:
    """Extract chrom sizes from 2bit and write TSV (chrom\tsize)."""
    sizes = _try_pyrion(twobit_path)
    if sizes is None:
        sizes = _try_py2bit(twobit_path)

    if sizes is None:
        raise RuntimeError("Could not read 2bit: install pyrion or py2bit")

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for chrom, size in sizes.items():
            f.write(f"{chrom}\t{size}\n")

    return sizes
