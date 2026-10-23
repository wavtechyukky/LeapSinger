"""Strip a training checkpoint down to what inference, export and --init_from use: the weights and the config
(plus the step, the training recipe and the reflow provenance). The optimizer states and the discriminator go,
so the release file is about a third of the size. Local paths in the metadata are reduced to their basenames.

    python tools/strip_ckpt.py log/<run>/ckpt_020000.pt release/3singer_v2_student.pt

The result loads in infer.load_acoustic, export.cli and train --init_from exactly like the full checkpoint
(it cannot resume training: no optimizer).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KEEP = ("model", "config", "step", "n_speakers", "train_recipe", "code", "reflow")
PATH_KEYS = ("teacher", "ckpt", "path", "init_from")       # metadata values that are file paths


def _basenames(o, key: str = ""):
    """Copy `o` with every path-like string (a value under a PATH_KEYS key, or any string with a separator)
    reduced to its basename, so the checkpoint says which file it came from but not where it lived."""
    if isinstance(o, dict):
        return {k: _basenames(v, str(k)) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_basenames(v, key) for v in o)
    if isinstance(o, str) and ("/" in o or "\\" in o or key in PATH_KEYS):
        return os.path.basename(o.rstrip("/\\"))
    return o


def strip(ck: dict, keep_paths: bool = False) -> dict:
    out = {}
    for k in KEEP:
        if k in ck:
            out[k] = ck[k] if (k == "model" or keep_paths) else _basenames(ck[k])
    return out


def _mb(path) -> float:
    return os.path.getsize(path) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description="学習用 ckpt から optimizer と判別器を落として、推論・書き出し・--init_from 用の軽い ckpt にする")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep-paths", action="store_true", help="メタ情報のパスをそのまま残す（既定はファイル名だけに）")
    ap.add_argument("--no-check", action="store_true", help="書いた後に infer.load_acoustic で読み直して確かめない")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    dropped = sorted(k for k in ck if k not in KEEP)
    out = strip(ck, keep_paths=args.keep_paths)
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dst)
    print(f"{args.src} ({_mb(args.src):.1f} MB) -> {args.dst} ({_mb(args.dst):.1f} MB)")
    print(f"  kept: {', '.join(k for k in KEEP if k in out)}   dropped: {', '.join(dropped) or '-'}   step {out.get('step')}")
    if not args.no_check:
        from infer import load_acoustic
        model, cfg = load_acoustic(args.dst, "cpu")
        n = sum(p.numel() for p in model.parameters())
        print(f"  check: load_acoustic OK ({cfg.get('arch')}, {n / 1e6:.1f} M params, {cfg.get('n_phonemes')} phonemes)")


if __name__ == "__main__":
    main()
