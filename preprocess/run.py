"""Run full preprocessing for one database from a recipe YAML.

    enumerate songs -> read lab -> per-phrase npz -> shard.npz

The recipe describes only what differs per database (paths, lab time unit, speaker id,
F0 range, phoneme normalisation). Everything else is the common pipeline. Phoneme timing
comes straight from the `.lab`; there is no musical score. A recipe may name its phoneme list
(`phonemes: dict/<list>.phonemes`, relative to the repository); `--phonemes` overrides it, and
without either the bundled Japanese list is used.

    python -m preprocess.run --recipe configs/recipes/ritsu.yaml
    python -m preprocess.run --recipe configs/recipes/aidol.yaml --check   # labels only, writes nothing
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import yaml

from leapsinger.config import MelSpec
from . import recipes
from .lab import process_file
from .shard import combine_db
from .vocab import Vocab

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _enumerate_songs(recipe: dict, db_root: str) -> list[str]:
    layout = recipe.get("layout", "per_song")
    if layout == "per_song":                       # {db_root}/{song}/{song}.*
        return sorted(os.path.basename(d) for d in glob.glob(os.path.join(db_root, "*"))
                      if os.path.isdir(d))
    if layout == "type_split":                     # {db_root}/{wav_dir}/{song}.wav
        wav_dir = os.path.join(db_root, recipe.get("wav_dir", "wav"))
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(wav_dir, "*.wav")))
    if layout == "from_lab":                       # the songs whose label file exists (the `lab` template;
        full = os.path.join(db_root, recipe["lab"])  # it may be absolute, e.g. converted labels elsewhere)
        pre, post = full.split("{song}", 1)
        return sorted(p[len(pre):len(p) - len(post)]
                      for p in glob.glob(glob.escape(pre) + "*" + glob.escape(post)))
    raise ValueError(f"unknown layout: {layout}")


def _check_song(paths: dict, lab_unit: str, phon_norm: dict, lab_fix, vocab) -> dict:
    """--check: read one song's lab the way process_file does, without touching the audio samples."""
    import soundfile as sf
    from .lab import DEFAULT_SIL_MAP, read_lab
    segs = read_lab(paths["lab"], lab_unit)
    if lab_fix is not None:
        segs = lab_fix(segs)
    unknown = sorted({p for _s, _e, p in segs if phon_norm.get(p, DEFAULT_SIL_MAP.get(p, p)) not in vocab})
    end = max((e for _s, e, _p in segs), default=0.0)
    dur = sf.info(paths["wav"]).duration
    return dict(unknown=unknown, lab_sec=sum(e - s for s, e, _p in segs), over_sec=end - dur, wav_sec=dur)


def run_recipe(recipe_path: str, out_root: str = "data",
               mel: MelSpec | None = None, limit: int | None = None, vocab=None, check: bool = False) -> int:
    """check=True only reads the labels and the wav headers: phonemes the vocab lacks (after phon_norm and the
    silence map), missing files, and how far the labels run past the audio. Nothing is written."""
    mel = mel or MelSpec()
    r = yaml.safe_load(open(recipe_path, encoding="utf-8"))
    db_root = r["db_root"]
    out_dir = os.path.join(out_root, r["name"])
    if vocab is None and r.get("phonemes"):            # the recipe's own phoneme list (relative to the repo)
        p = r["phonemes"]
        vocab = Vocab.load(p if os.path.isabs(p) else os.path.join(_REPO, p))
    if check:
        return _check_recipe(r, db_root, limit, vocab or Vocab.load())
    # process_file skips songs whose phrases already exist, so re-running into the same out_dir
    # with other frame settings would silently mix old phrases under new metadata.
    meta_path = os.path.join(out_dir, "metadata.json")
    if os.path.exists(meta_path):
        old = json.load(open(meta_path, encoding="utf-8"))
        prev = (old.get("sample_rate"), old.get("hop_size"), old.get("n_mels"))
        if None not in prev and prev != (mel.sr, mel.hop, mel.n_mels):
            raise SystemExit(f"{out_dir} was built with sr/hop/n_mels={prev}, not "
                             f"{(mel.sr, mel.hop, mel.n_mels)}; use another --out_root or delete it.")

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


def _check_recipe(r: dict, db_root: str, limit, vocab) -> int:
    """The --check pass of run_recipe: one line per problem song, then a summary. Returns the number of songs
    that would be processed without a problem."""
    exclude = set(r.get("exclude", []))
    songs = [s for s in _enumerate_songs(r, db_root) if s not in exclude]
    if limit:
        songs = songs[:limit]
    phon_norm = r.get("phon_norm", {}) or {}
    hook = recipes.load(r.get("hook", r["name"]))
    unknown, n_ok, lab_sec, wav_sec, over = {}, 0, 0.0, 0.0, []
    for s in songs:
        paths = {k: recipes.resolve_song_file(db_root, r[k], s) for k in ("wav", "lab")}
        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing:
            print(f"  [MISS {','.join(missing)}] {s}")
            continue
        lab_fix = (lambda iv, _s=s: hook.fix_lab(_s, iv)) if hook is not None and hasattr(hook, "fix_lab") else None
        c = _check_song(paths, r.get("lab_unit", "100ns"), phon_norm, lab_fix, vocab)
        for p in c["unknown"]:
            unknown.setdefault(p, []).append(s)
        if c["over_sec"] > 0.05:
            over.append((s, c["over_sec"]))
        if c["unknown"]:
            print(f"  [UNKNOWN {' '.join(c['unknown'])}] {s}")
        else:
            n_ok += 1
        lab_sec += c["lab_sec"]
        wav_sec += c["wav_sec"]
    print(f"{r['name']}: {n_ok}/{len(songs)} songs OK  labels {lab_sec / 3600:.3f} h / audio {wav_sec / 3600:.3f} h"
          + (f"  phonemes not in the vocab: {sorted(unknown)}" if unknown else ""))
    if over:
        print(f"  labels run past the audio end (> 50 ms) in {len(over)} songs, at most {max(o for _s, o in over):.2f} s"
              f" ({over[0][0]} ...)")
    return n_ok


def main():
    ap = argparse.ArgumentParser(description="Preprocess one database from a recipe YAML.")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N songs")
    ap.add_argument("--hop", type=int, default=256, help="frame size (default 256)")
    ap.add_argument("--phonemes", default=None,
                    help="phoneme-list file (default: the recipe's `phonemes`, else the bundled Japanese "
                         "dict/ja.phonemes)")
    ap.add_argument("--check", action="store_true",
                    help="only check the labels (phonemes against the vocab, missing files, labels past the audio "
                         "end); no audio processing, nothing written")
    args = ap.parse_args()
    run_recipe(args.recipe, out_root=args.out_root, mel=MelSpec(hop=args.hop),
               limit=args.limit, vocab=Vocab.load(args.phonemes) if args.phonemes else None, check=args.check)


if __name__ == "__main__":
    main()
