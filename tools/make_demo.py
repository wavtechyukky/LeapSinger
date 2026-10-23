"""Build the demo page (demo/index.html): render the demo phrases with a checkpoint, add songs sung from
scores, and lay the versions out side by side.

Each version of the model has a manifest, demo/manifest_<tag>.json: the phrase list with its train/eval
split, seconds and mel-L1, the asset name suffixes, and the songs. `render` writes one for a checkpoint
(using the phrase list of an existing manifest, so every version sings the same phrases), `scores`
adds songs rendered elsewhere, and `html` builds the page from the manifests it is given.

    # 1. copy-synthesis phrases (ogg + mel figures), the same phrases and split as the V1 page
    python tools/make_demo.py render --tag v2 --ckpt release/3singer_v2_student.pt \\
        --config configs/3singer_v2.yaml --data_dirs data/oniku data/natsume data/ritsu \\
        --like demo/manifest_v1.json --label V2 --release v2.0.0 --threads 2
    # 2. songs sung from scores: <wav_dir>/<speaker>/<song>.wav -> demo/audio/svs_<speaker>_<song>_<tag>.ogg
    python tools/make_demo.py scores --tag v2 --wav_dir out/svs --alias kirakiraboshi=kirakira_boshi
    # 3. the page (the first manifest gives the phrase order, the seconds and the song titles; the optional
    #    comparison JSON is a small metrics table shown at the top, e.g. from tools/eval_ab.py)
    python tools/make_demo.py html --manifests demo/manifest_v1.json demo/manifest_v2.json \\
        --comparison demo/comparison_v1_v2.json

Asset names: audio/<spk>_<phrase>_<audio_suffix>.ogg, img/<spk>_<phrase><img_suffix>_{pred,voc}.png,
audio/svs_<spk>_<song><score_suffix>.ogg. The V1 (v0.1.0) assets keep their original names
(_synth.ogg, _pred.png, no suffix on the songs); manifest_v1.json records those suffixes.
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEMO = ROOT / "demo"
MEL_PNG_SIZE = (499, 201)          # the size of the V1 figures (pixels)
DEFAULT_VOCODER = "checkpoints/nhv_v3_2_1.onnx"


# ── shared ────────────────────────────────────────────────────────────────────

def load_manifest(path) -> dict:
    return json.load(open(path, encoding="utf-8"))


def save_manifest(man: dict, path) -> None:
    Path(path).write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def manifest_path(tag: str) -> Path:
    return DEMO / f"manifest_{tag}.json"


def resolve(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() or q.exists() else ROOT / q


def write_ogg(path: Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0), int(sr),
             format="OGG", subtype="VORBIS")


def rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))) + 1e-9))


def read_mono(path: Path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, int(sr)


def match_level(x: np.ndarray, ref: np.ndarray):
    """Scale `x` to the RMS of `ref` (peak kept under 0.99). Returns (scaled, gain_db actually applied)."""
    gain_db = rms_db(ref) - rms_db(x)
    y = x * 10 ** (gain_db / 20)
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak > 0.99:
        y *= 0.99 / peak
        gain_db += 20 * np.log10(0.99 / peak)
    return y.astype(np.float32), float(gain_db)


def save_mel_png(mel: np.ndarray, path: Path, vmin: float, vmax: float, size=MEL_PNG_SIZE) -> None:
    """A bare mel image (no axes) of exactly `size` pixels, magma, the same colour scale as the other figures."""
    from matplotlib import colormaps
    from PIL import Image
    x = (np.asarray(mel, dtype=np.float32) - vmin) / max(vmax - vmin, 1e-6)
    rgba = colormaps["magma"](np.clip(x, 0.0, 1.0), bytes=True)      # [n_mels, T, 4] uint8
    img = Image.fromarray(np.ascontiguousarray(rgba[::-1]), "RGBA")   # low frequencies at the bottom
    path.parent.mkdir(parents=True, exist_ok=True)
    img.resize(size, Image.BILINEAR).save(str(path))


# ── render: the copy-synthesis phrases ────────────────────────────────────────

def build_dataset(cfg: dict, data_dirs, split: str):
    """The dataset the way train.py / tools/eval_ab.py build it (speaker ids from data.spk_map, completed
    from configs/recipes/<db>.yaml), for one split. The split decides the train/eval badge."""
    import yaml
    from dataset import LeapSingerDataset
    dcfg = cfg.get("data", {}) or {}
    spk_map = dict(dcfg.get("spk_map") or {})
    style_map = dict(dcfg.get("style_map") or {})
    for d in data_dirs:
        dbname = os.path.basename(os.path.normpath(d))
        rp = ROOT / "configs" / "recipes" / f"{dbname}.yaml"
        if rp.exists():
            rr = yaml.safe_load(open(rp, encoding="utf-8"))
            spk_map.setdefault(dbname, int(rr.get("spk_id", 0)))
            style_map.setdefault(dbname, int(rr.get("style_id", 0)))
    return LeapSingerDataset(data_dirs, split, eval_songs=dcfg.get("eval_songs", 2),
                             eval_dbs=dcfg.get("eval_dbs"), min_sec=dcfg.get("min_sec", 0.3),
                             silence=dcfg.get("silence", True),
                             silence_fade_sec=dcfg.get("silence_fade_sec", 0.05),
                             spk_map=spk_map, style_map=style_map,
                             loudness_agc=dcfg.get("loudness_agc") or None,
                             max_sec=dcfg.get("max_sec") or None)


def vocoder_label(path: Path) -> str:
    ver = path.stem.replace("nhv_v", "").replace("_", ".")
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        rel = path.name
    return f"NHVSing V{ver} ({rel})"


def cmd_render(args) -> None:
    import torch
    import yaml
    from infer import infer_mel, load_acoustic, load_vocoder, mel_to_wav
    from leapsinger.mel import wav_to_mel_nhv
    torch.set_num_threads(args.threads)

    like = load_manifest(args.like)
    cfg = yaml.safe_load(open(resolve(args.config), encoding="utf-8"))
    want = {(s["id"], p["name"]): p["split"] for s in like["speakers"] for p in s["phrases"]}
    found: dict = {}                                   # (spk, phrase) -> (split, item)
    frame_rate = None
    for split in ("train", "eval"):
        ds = build_dataset(cfg, args.data_dirs, split)
        frame_rate = ds.frame_rate
        for i, (_shard, name, dbdir) in enumerate(ds.files):
            spk = os.path.basename(os.path.normpath(dbdir))
            keys = {(spk, name)}
            if name.startswith(spk) and len(name) > len(spk):   # natsume12_0001 is shown as 12_0001
                keys.add((spk, name[len(spk):].lstrip("_")))
            for key in keys & want.keys():
                found[key] = (split, ds[i])
    missing = [f"{s}:{n}" for (s, n) in want if (s, n) not in found]
    if missing:
        raise SystemExit(f"phrases not in the datasets: {', '.join(missing)}")
    wrong = [f"{s}:{n} is {found[(s, n)][0]} here but {sp} in {args.like}" for (s, n), sp in want.items()
             if found[(s, n)][0] != sp]
    if wrong:                                          # the badge must mean the same thing on every version
        raise SystemExit("train/eval split differs from the manifest:\n  " + "\n  ".join(wrong))

    model, mcfg = load_acoustic(str(resolve(args.ckpt)), "cpu")
    use_uv = bool(mcfg.get("use_uv", True))
    vmin, vmax = float(mcfg.get("mel_vmin", -11.5)), float(mcfg.get("mel_vmax", 2.0))
    sr = int((cfg.get("mel") or {}).get("sr", 44100))
    hop = int((cfg.get("mel") or {}).get("hop", 256))
    voc_path = resolve(args.vocoder)
    voc = load_vocoder(str(voc_path), sr)

    tag = args.tag
    man = {
        "tag": tag, "label": args.label or tag.upper(), "release": args.release, "recipe": args.recipe,
        "vocoder": vocoder_label(voc_path),
        "audio_suffix": tag, "img_suffix": f"_{tag}", "score_suffix": f"_{tag}",
        "ckpt": Path(args.ckpt).name, "steps": args.steps, "seed": args.seed,
        "mel_l1_note": "mel_l1_target is the L1 to the training target (the loudness-corrected GT mel), "
                       "not comparable with V1's mel_l1 against the raw GT",
        "audio_level": args.level,
        "speakers": [], "scores": [],
    }
    if args.note:
        man["audio_note"] = args.note
    for s in like["speakers"]:
        out_s = {"id": s["id"], "name": s["name"], "phrases": []}
        for p in s["phrases"]:
            split, it = found[(s["id"], p["name"])]
            stem = f"{s['id']}_{p['name']}"
            mel = infer_mel(model, it, num_steps=args.steps, device="cpu", seed=args.seed)
            vuv = it["uv"] if use_uv else np.ones_like(it["uv"])      # as train.py's previews
            wav = mel_to_wav(voc, mel, it["f0_logf0"], vuv, normalize=False)   # the level the mel describes
            save_mel_png(mel, DEMO / "img" / f"{stem}{man['img_suffix']}_pred.png", vmin, vmax)
            voc_mel = wav_to_mel_nhv(wav, sr=sr, hop=hop)            # same level as the model mel
            save_mel_png(voc_mel, DEMO / "img" / f"{stem}{man['img_suffix']}_voc.png", vmin, vmax)
            gain_db = 0.0
            gt_path = DEMO / "audio" / f"{stem}_gt.ogg"
            if args.level == "match_gt" and gt_path.exists():        # listen at the recording's level
                gt, _gsr = read_mono(gt_path)
                wav, gain_db = match_level(wav, gt)
            write_ogg(DEMO / "audio" / f"{stem}_{man['audio_suffix']}.ogg", wav, sr)
            T = min(mel.shape[1], it["target_mel"].shape[1])
            l1 = float(np.mean(np.abs(mel[:, :T] - it["target_mel"][:, :T])))
            out_s["phrases"].append({"name": p["name"], "split": split, "sec": round(T / frame_rate, 1),
                                     "mel_l1_target": round(l1, 3), "gain_db": round(gain_db, 1)})
            print(f"[{tag}] {stem}  {split}  {T / frame_rate:.1f}s  L1(target) {l1:.3f}  gain {gain_db:+.1f} dB",
                  flush=True)
        man["speakers"].append(out_s)
    save_manifest(man, manifest_path(tag))
    print(f"wrote {manifest_path(tag)}")


# ── scores: songs sung from scores, rendered elsewhere ─────────────────────────

def _lead_samples(x: np.ndarray, thr_db: float = -45.0) -> int:
    a = np.abs(x) > 10 ** (thr_db / 20)
    return int(np.argmax(a)) if a.any() else 0


def trim_like(x: np.ndarray, sr: int, ref_path: Path, fade_sec: float):
    """Cut `x` to the excerpt the reference file plays: start at the first note (the lead silence of both is
    matched), the reference's length, and a fade-out. Returns (excerpt, start_sec, length_sec)."""
    import soundfile as sf
    r, rsr = sf.read(str(ref_path), dtype="float32")
    if r.ndim > 1:
        r = r.mean(axis=1)
    if int(rsr) != int(sr):
        raise SystemExit(f"{ref_path}: {rsr} Hz, the new render is {sr} Hz")
    start = max(0, _lead_samples(x) - _lead_samples(r))
    y = np.array(x[start:start + len(r)], dtype=np.float32)
    n = min(len(y), int(fade_sec * sr))
    if n > 0:
        y[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return y, start / sr, len(r) / sr


def cmd_scores(args) -> None:
    import soundfile as sf
    man = load_manifest(manifest_path(args.tag))
    titles = {sc["key"]: sc["title"] for sc in load_manifest(args.like)["scores"]} if args.like else {}
    ref = load_manifest(manifest_path(args.trim_like)) if args.trim_like else None
    alias = dict(a.split("=", 1) for a in args.alias)
    keys: dict = {}
    for s in man["speakers"]:
        d = Path(args.wav_dir) / s["id"]
        for wav in sorted(d.glob("*.wav")):
            key = alias.get(wav.stem, wav.stem)
            x, sr = sf.read(str(wav), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            info = ""
            if ref is not None:                        # the same excerpt and level as the reference version
                rp = DEMO / "audio" / f"svs_{s['id']}_{key}{ref['score_suffix']}.ogg"
                if rp.exists():
                    x, t0, dur = trim_like(x, sr, rp, args.trim_fade)
                    x, gain_db = match_level(x, read_mono(rp)[0])
                    man.setdefault("scores_gain_db", {})[f"{s['id']}_{key}"] = round(gain_db, 1)
                    info = f"  (cut to {ref['label']}'s excerpt: from {t0:.2f}s, {dur:.1f}s; gain {gain_db:+.1f} dB)"
            write_ogg(DEMO / "audio" / f"svs_{s['id']}_{key}{man['score_suffix']}.ogg", x, sr)
            keys.setdefault(key, set()).add(s["id"])
            print(f"[{args.tag}] svs_{s['id']}_{key}: {len(x) / sr:.1f}s{info}", flush=True)
    if not keys:
        raise SystemExit(f"no <speaker>/<song>.wav under {args.wav_dir}")
    spk_ids = [s["id"] for s in man["speakers"]]
    short = [k for k, v in keys.items() if v != set(spk_ids)]
    if short:
        raise SystemExit(f"songs missing for some speakers: {short}")
    order = [k for k in titles if k in keys] + sorted(k for k in keys if k not in titles)
    man["scores"] = [{"key": k, "title": titles.get(k, k)} for k in order]
    if args.note:
        man["scores_note"] = args.note
    save_manifest(man, manifest_path(args.tag))
    print(f"wrote {manifest_path(args.tag)}: {len(order)} songs")


# ── html: the page ────────────────────────────────────────────────────────────

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#d9d9d9;--card:#fafafa;--badge-tr:#2b6cb0;--badge-ev:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--fg:#e8e8e8;--muted:#9aa0a6;--line:#333;--card:#1c1f23;--badge-tr:#63b3ed;--badge-ev:#f6ad55}}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:var(--bg);color:var(--fg);line-height:1.55}
.wrap{max-width:1500px;margin:0 auto;padding:2.2em 1.4em 4em}
h1{font-size:1.7em;margin:.2em 0 .1em;letter-spacing:-.01em}
h2{font-size:1.25em;margin:2.2em 0 .6em;border-bottom:2px solid var(--line);padding-bottom:.25em}
h3{font-size:1.05em;color:var(--muted);margin:1.6em 0 .4em;font-weight:600}
p.lead{color:var(--muted);max-width:80ch}
ul.versions{color:var(--muted);max-width:80ch;padding-left:1.2em}
.m{color:var(--muted);font-size:.86em}
table{border-collapse:collapse;width:100%;margin:.4em 0 1em}
td,th{border:1px solid var(--line);padding:9px;vertical-align:top;text-align:left}
th{font-size:.82em;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);font-weight:600;background:var(--card)}
td:first-child{font-family:ui-monospace,Menlo,monospace;font-size:.8em;word-break:break-all;max-width:135px}
table.cmp{max-width:900px}
table.cmp td:first-child{font-family:inherit;font-size:.9em;word-break:normal;max-width:none}
table.cmp td{white-space:nowrap}
audio{width:100%;max-width:240px;display:block}
img.mel{width:300px;max-width:100%;border-radius:3px;display:block}
.stack>*+*{margin-top:.35em}
.lbl{display:block;font-size:.72em;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-top:.1em}
.badge{display:inline-block;font-size:.7em;font-weight:700;padding:.1em .5em;border-radius:999px;color:#fff;margin-top:.35em}
.badge.tr{background:var(--badge-tr)}.badge.ev{background:var(--badge-ev)}
.tablewrap{overflow-x:auto}
"""


def esc(s) -> str:
    return html_mod.escape(str(s), quote=True)


def audio_tag(src: str) -> str:
    return f'<audio controls preload="none" src="{src}"></audio>'


def version_line(m: dict) -> str:
    bits = [f"<b>{esc(m['label'])}</b>"]
    if m.get("release"):
        bits.append(f"({esc(m['release'])})")
    s = " ".join(bits)
    if m.get("recipe"):
        s += f" — {esc(m['recipe'])}"
    if m.get("vocoder"):
        s += f". Vocoder: {esc(m['vocoder'])}"
    return s + "."


def phrase_row(spk: str, p: dict, mans: list) -> str:
    badge = ('<span class="badge tr">train</span>' if p["split"] == "train"
             else '<span class="badge ev">eval</span>')
    stem = f"{spk}_{p['name']}"
    first = f"<td>{esc(p['name'])}<br>{badge}<br><span class=\"m\">{p['sec']}s</span></td>"
    gt_audio = f"<td>{audio_tag(f'audio/{stem}_gt.ogg')}</td>"
    audios, mels = [], []
    for m in mans:
        l1 = ""
        if "mel_l1" in p.get("by_tag", {}).get(m["tag"], {}):
            l1 = f'<span class="m">mel-L1 {p["by_tag"][m["tag"]]["mel_l1"]:.3f}</span>'
        suffix = m["audio_suffix"]
        audios.append(f"<td>{audio_tag(f'audio/{stem}_{suffix}.ogg')}{l1}</td>")
        mels.append('<td><div class="stack">'
                    f'<img class="mel" src="img/{stem}{m["img_suffix"]}_pred.png"><span class="lbl">model</span>'
                    f'<img class="mel" src="img/{stem}{m["img_suffix"]}_voc.png"><span class="lbl">vocoded</span>'
                    '</div></td>')
    gt_mel = f'<td><img class="mel" src="img/{stem}_gt.png"></td>'
    return "<tr>" + first + gt_audio + "".join(audios) + gt_mel + "".join(mels) + "</tr>"


def merge_phrases(mans: list) -> list:
    """Speakers and phrases in the order of the first manifest; each phrase carries the per-version numbers."""
    base = mans[0]
    out = []
    for s in base["speakers"]:
        rows = []
        for p in s["phrases"]:
            row = {"name": p["name"], "split": p["split"], "sec": p.get("sec"), "by_tag": {}}
            for m in mans:
                ms = next((x for x in m["speakers"] if x["id"] == s["id"]), None)
                mp = next((x for x in (ms or {}).get("phrases", []) if x["name"] == p["name"]), None)
                if mp is None:
                    raise SystemExit(f"{m['tag']}: no phrase {s['id']}:{p['name']} (render it first)")
                if mp["split"] != p["split"]:
                    raise SystemExit(f"{m['tag']}: {s['id']}:{p['name']} is {mp['split']}, {base['tag']} says {p['split']}")
                row["by_tag"][m["tag"]] = mp
            rows.append(row)
        out.append({"id": s["id"], "name": s["name"], "phrases": rows})
    return out


def _cell(v) -> str:
    """Escape a table value; *text* marks the better value and is shown bold."""
    return re.sub(r"\*([^*]+)\*", r"<b>\1</b>", esc(v))


def comparison_block(cmp: dict) -> str:
    """A small metrics table (demo/comparison_<a>_<b>.json): title, lead, columns, rows [{metric, values}], notes.
    In the values, *text* is rendered bold (the better value)."""
    h = [f"<h2>{esc(cmp.get('title', 'Comparison'))}</h2>"]
    if cmp.get("lead"):
        h.append(f'<p class="lead">{esc(cmp["lead"])}</p>')
    h.append('<div class="tablewrap"><table class="cmp"><tr><th>metric</th>'
             + "".join(f"<th>{esc(c)}</th>" for c in cmp.get("columns", [])) + "</tr>")
    for r in cmp.get("rows", []):
        h.append(f"<tr><td>{esc(r['metric'])}</td>" + "".join(f"<td>{_cell(v)}</td>" for v in r["values"]) + "</tr>")
    h.append("</table></div>")
    if cmp.get("notes"):
        h.append('<ul class="versions">' + "".join(f"<li>{esc(n)}</li>" for n in cmp["notes"]) + "</ul>")
    return "\n".join(h)


def build_html(mans: list, comparison: dict | None = None) -> str:
    base = mans[0]
    labels = [m["label"] for m in mans]
    speakers = merge_phrases(mans)
    l1_tags = [m["label"] for m in mans
               if any("mel_l1" in p for s in m["speakers"] for p in s["phrases"])]

    h = ['<!doctype html><meta charset="utf-8"><title>LeapSinger Demo</title>',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         f"<style>{CSS}</style>", '<div class="wrap">',
         "<h1>LeapSinger — synthesis demo</h1>",
         '<p class="lead">A lightweight singing-voice synthesizer. The <b>multi-speaker acoustic model</b> '
         "(harmonic-excitation rectified flow, one step) predicts a mel-spectrogram, rendered to audio by the "
         "bundled NHVSing vocoder. This page puts the released versions side by side:</p>",
         '<ul class="versions">' + "".join(f"<li>{version_line(m)}</li>" for m in mans) + "</ul>"]
    if comparison:
        h.append(comparison_block(comparison))

    # 1. copy-synthesis
    h.append("<h2>1. Copy-synthesis — ground truth vs model (3 speakers)</h2>")
    cols = " / ".join(f"<b>{esc(l)}</b>" for l in labels)
    note_l1 = ""
    if l1_tags:
        note_l1 = (f' <span class="m">mel-L1 (shown for {esc(", ".join(l1_tags))}) is the L1 distance to the GT mel '
                   "(lower = closer); it does not track perception, so judge by ear."
                   + (" The other versions train on a loudness-corrected target, so their distance to the raw GT "
                      "is not comparable and is not shown.</span>" if len(l1_tags) < len(mans) else "</span>"))
    h.append('<p class="lead">Per singer, <b>3 phrases from training songs</b> and <b>2 from held-out songs</b> '
             '(<span class="badge ev">eval</span>, never seen in training), driven by the real F0, durations and '
             f"phonemes taken from each recording. Columns: the original audio, the synthesis by {cols}, the "
             "<b>GT mel</b>, and for each version the <b>model's mel</b> (top) with the mel of the "
             f"<b>vocoded output</b> (bottom).{note_l1}</p>")
    notes = [f"<li><b>{esc(m['label'])}</b>: {esc(m['audio_note'])}</li>" for m in mans if m.get("audio_note")]
    if notes:
        h.append('<ul class="versions">' + "".join(notes) + "</ul>")
    for s in speakers:
        h.append(f"<h3>{esc(s['name'])}</h3>")
        head = ("<tr><th>phrase</th><th>GT audio</th>" + "".join(f"<th>{esc(l)} audio</th>" for l in labels)
                + "<th>GT mel</th>" + "".join(f"<th>{esc(l)} mel</th>" for l in labels) + "</tr>")
        h.append('<div class="tablewrap"><table>' + head)
        for p in s["phrases"]:
            h.append(phrase_row(s["id"], p, mans))
        h.append("</table></div>")

    # 2. scores
    songs = base.get("scores") or []
    if songs:
        h.append("<h2>2. Singing unseen scores</h2>")
        h.append('<p class="lead">These are songs the model has <b>never seen</b>, sung from a musical score. The '
                 "score is turned into an F0 curve and phoneme durations by our <b>pitch and duration models</b>, "
                 "then the acoustic model and the NHVSing vocoder render the audio. Three children’s songs, each "
                 "sung by all three voices" + (", every version stacked in each cell." if len(mans) > 1 else ".")
                 + "</p>")
        notes = [f"<li><b>{esc(m['label'])}</b>: {esc(m['scores_note'])}</li>" for m in mans if m.get("scores_note")]
        if notes:
            h.append('<ul class="versions">' + "".join(notes) + "</ul>")
        h.append('<div class="tablewrap"><table><tr><th>song</th>'
                 + "".join(f"<th>{esc(s['name'])}</th>" for s in speakers) + "</tr>")
        for sc in songs:
            cells = []
            for s in speakers:
                parts = []
                for m in mans:
                    if any(x["key"] == sc["key"] for x in m.get("scores") or []):
                        parts.append(f'<span class="lbl">{esc(m["label"])}</span>'
                                     + audio_tag(f"audio/svs_{s['id']}_{sc['key']}{m['score_suffix']}.ogg"))
                cells.append('<td><div class="stack">' + "".join(parts) + "</div></td>")
            h.append(f"<tr><td>{esc(sc['title'])}</td>" + "".join(cells) + "</tr>")
        h.append("</table></div>")
        if (DEMO / "audio" / "svs_momiji.ogg").exists():
            h.append('<p class="lead" style="margin-top:1.4em"><b>Bonus — Momiji (もみじ), a three-part chorus.</b> '
                     f"The three voices arranged into harmony ({esc(base['label'])}).</p>"
                     '<div style="max-width:520px">' + audio_tag("audio/svs_momiji.ogg") + "</div>")

    # 3. style control (the v0.1.0 three-style model; static assets)
    if (DEMO / "audio" / "style_default.ogg").exists():
        h.append("<h2>3. Style control</h2>")
        h.append('<p class="lead">The full model carries Namine Ritsu’s three voice styles — <b>default</b> (kire), '
                 "<b>normal</b>, and <b>soft</b>. One phrase (a real recording, re-rendered by copy-synthesis) is "
                 "sung with the style blended continuously from default toward normal, and toward soft. Only the "
                 "timbre changes — the notes, timing and pitch stay identical. "
                 f'<span class="m">This section uses the {esc(base["label"])} three-style model '
                 f"({esc(base.get('release', ''))}); the later release models have no style input.</span></p>")
        h.append('<p class="m" style="margin-bottom:.3em">Original recording (reference):</p>'
                 '<div style="max-width:340px;margin-bottom:1em">' + audio_tag("audio/style_gt.ogg") + "</div>")
        h.append('<div class="tablewrap"><table><tr><th>blend →</th><th>0%<br><span class="m">default</span></th>'
                 "<th>25%</th><th>50%</th><th>75%</th><th>100%</th></tr>")
        for st in ("normal", "soft"):
            h.append(f"<tr><td>→ {st}</td><td>{audio_tag('audio/style_default.ogg')}</td>"
                     + "".join(f"<td>{audio_tag(f'audio/style_{st}_{k}.ogg')}</td>" for k in (25, 50, 75, 100))
                     + "</tr>")
        h.append("</table></div>")
    h.append("</div>")
    return "\n".join(h) + "\n"


def check_assets(page: str) -> list:
    srcs = sorted(set(re.findall(r'src="([^"]+)"', page)))
    return [s for s in srcs if not (DEMO / s).exists()]


def cmd_html(args) -> None:
    mans = [load_manifest(p) for p in args.manifests]
    cmp = load_manifest(args.comparison) if args.comparison else None
    page = build_html(mans, cmp)
    missing = check_assets(page)
    if missing and not args.allow_missing:
        raise SystemExit("assets missing under demo/:\n  " + "\n  ".join(missing))
    out = Path(args.out) if args.out else DEMO / "index.html"
    out.write_text(page, encoding="utf-8")
    n = len(set(re.findall(r'src="([^"]+)"', page)))
    print(f"wrote {out}: {len(mans)} versions, {n} assets" + (f", {len(missing)} missing" if missing else ""))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="デモページの素材を作って、版を並べた demo/index.html を書く")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="チェックポイントで句を生成して ogg・mel の図・manifest を書く")
    r.add_argument("--tag", required=True, help="版の名前（ファイル名と manifest に使う。例 v2）")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--config", required=True, help="学習に使った config（評価用の曲の分け方と話者の対応）")
    r.add_argument("--data_dirs", nargs="+", required=True)
    r.add_argument("--like", default=str(DEMO / "manifest_v1.json"),
                   help="句の一覧と train/eval を取る manifest（この版も同じ句を歌う）")
    r.add_argument("--label", default=None, help="ページでの表示名（既定は tag の大文字）")
    r.add_argument("--release", default="", help="例 v2.0.0")
    r.add_argument("--recipe", default="", help="版の一言（例: GAN-free reflow teacher → 1-step student）")
    r.add_argument("--vocoder", default=DEFAULT_VOCODER)
    r.add_argument("--steps", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--threads", type=int, default=2, help="torch の CPU スレッド数")
    r.add_argument("--level", choices=["match_gt", "raw"], default="match_gt",
                   help="音声のレベル: match_gt = 句ごとに GT の RMS に合わせる（既定）、raw = ボコーダ出力のまま。"
                        "mel の図はどちらでもボコーダ出力のレベル（モデルの mel と同じ）")
    r.add_argument("--note", default="", help="この版の第 1 節への注記（音声のレベルなど）")
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser("scores", help="楽譜から歌わせた wav を ogg にして manifest に足す")
    s.add_argument("--tag", required=True)
    s.add_argument("--wav_dir", required=True, help="<wav_dir>/<話者>/<曲>.wav")
    s.add_argument("--like", default=str(DEMO / "manifest_v1.json"), help="曲の表示名を取る manifest")
    s.add_argument("--alias", nargs="*", default=[], help="wav の名前 → ページの曲名（例 kirakiraboshi=kirakira_boshi）")
    s.add_argument("--trim_like", default=None, metavar="TAG",
                   help="曲をその版の抜粋と同じ範囲に切り（最初の音から同じ長さ。V1 のデモは曲の頭だけ）、RMS も揃える")
    s.add_argument("--trim_fade", type=float, default=0.2, help="切った末尾のフェードアウト（秒）")
    s.add_argument("--note", default="", help="この版の第 2 節への注記")
    s.set_defaults(fn=cmd_scores)

    p = sub.add_parser("html", help="manifest からページを書く")
    p.add_argument("--manifests", nargs="+", required=True, help="並べる順。先頭が句の順と秒数と曲名の元")
    p.add_argument("--comparison", default=None,
                   help="冒頭に出す指標の表の JSON（例 demo/comparison_v1_v2.json: title, lead, columns, rows, notes）")
    p.add_argument("--out", default=None, help="既定 demo/index.html")
    p.add_argument("--allow_missing", action="store_true", help="無い素材があっても書く")
    p.set_defaults(fn=cmd_html)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
