"""Run full preprocessing for one database from a recipe YAML.

    enumerate songs -> read lab -> per-phrase npz -> shard.npz

The recipe describes only what differs per database (paths, lab time unit, speaker id,
F0 range, phoneme normalisation). Everything else is the common pipeline. Phoneme timing
comes straight from the `.lab`; there is no musical score.

    python -m preprocess.run --recipe configs/recipes/ritsu.yaml
"""
from __future__ import annotations

import argparse
import glob
import os

import yaml

from leapsinger.config import MelSpec
from . import recipes
from .lab import process_file
from .shard import combine_db
from .vocab import Vocab


def _enumerate_songs(recipe: dict, db_root: str) -> list[str]:
    layout = recipe.get("layout", "per_song")
    if layout == "per_song":                       # {db_root}/{song}/{song}.*
        return sorted(os.path.basename(d) for d in glob.glob(os.path.join(db_root, "*"))
                      if os.path.isdir(d))
    if layout == "type_split":                     # {db_root}/{wav_dir}/{song}.wav
        wav_dir = os.path.join(db_root, recipe.get("wav_dir", "wav"))
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(wav_dir, "*.wav")))
    raise ValueError(f"unknown layout: {layout}")


def run_recipe(recipe_path: str, out_root: str = "data",
               mel: MelSpec | None = None, limit: int | None = None, vocab=None) -> int:
    mel = mel or MelSpec()
    r = yaml.safe_load(open(recipe_path, encoding="utf-8"))
    db_root = r["db_root"]
    out_dir = os.path.join(out_root, r["name"])

    exclude = set(r.get("exclude", []))
    songs = [s for s in _enumerate_songs(r, db_root) if s not in exclude]
    if limit:
        songs = songs[:limit]
    lab_unit = r.get("lab_unit", "100ns")
    phon_norm = r.get("phon_norm", {}) or {}
    prefix = r.get("name_prefix", "")
    hook = recipes.load(r.get("hook", r["name"]))     # per-singer code module, if any
    has_fix = hook is not None and hasattr(hook, "fix_lab")

    n_ok = 0
    for s in songs:
        paths = {k: recipes.resolve_song_file(db_root, r[k], s) for k in ("wav", "lab")}
        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing:
            print(f"  [MISS {','.join(missing)}] {s}")
            continue
        name = f"{prefix}{s}"
        lab_fix = (lambda iv, _s=s: hook.fix_lab(_s, iv)) if has_fix else None
        # speaker/style identity is NOT baked into the shard — it is assigned per-DB at train
        # time (see LeapSingerDataset spk_map/style_map). The recipe's spk_id/style_id are the
        # suggested default mapping, consumed by the trainer, not by preprocessing.
        try:
            process_file(name, paths["lab"], paths["wav"], out_dir, mel=mel,
                         save_wav=bool(r.get("save_wav", False)),
                         f0_device=r.get("f0_device", "cpu"),
                         lab_unit=lab_unit, phon_norm=phon_norm, lab_fix=lab_fix, vocab=vocab)
        except Exception as e:  # noqa: BLE001 — keep going, report the failing song
            print(f"  [FAIL] {s}: {e}")
            continue
        n_ok += 1
    print(f"{r['name']}: processed {n_ok}/{len(songs)} songs")
    combine_db(out_dir, mel, vocab=vocab)
    return n_ok


def main():
    ap = argparse.ArgumentParser(description="Preprocess one database from a recipe YAML.")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N songs")
    ap.add_argument("--hop", type=int, default=256, help="frame size (default 256)")
    ap.add_argument("--phonemes", default=None,
                    help="phoneme-list file (default: bundled Japanese dict/ja.phonemes)")
    args = ap.parse_args()
    run_recipe(args.recipe, out_root=args.out_root, mel=MelSpec(hop=args.hop),
               limit=args.limit, vocab=Vocab.load(args.phonemes))


if __name__ == "__main__":
    main()
