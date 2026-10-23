"""音量正規化チューナー — 前処理済みのデータセット（shard.npz + metadata.json）を選び、学習目標 mel に掛ける
遅い自動ゲイン（leapsinger/loudness.py、config の data.loudness_agc）のパラメータを、フレーズごとのカード
（音量包絡・ゲイン・mel・原音/正規化音の再生）で確かめながら決める Web アプリ。

    python tools/loudness_tuner/app.py --data data/ritsu [data/oniku ...] [--port 8018] [--host 127.0.0.1]
    → http://127.0.0.1:8018

--data にはデータセットのフォルダ（shard.npz があるフォルダ）か、それらを含む親フォルダを渡す。画面からも
フォルダを打ち込める。正規化の計算は学習と同じ leapsinger/loudness.py。再生音はボコーダを通さず、元の波形に
同じゲイン曲線を掛けたもの。音声があるのは、前処理で `save_wav: true` にして shard_wav.npz を作ったデータセット
（または <親>/phrase/<名前>/<フレーズ>.wav を持つ古いレイアウト）だけ。決めた値は画面の YAML をそのまま
config の data.loudness_agc に貼る。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                                        # リポジトリの頭（tools/loudness_tuner/ の 2 つ上）
sys.path.insert(0, str(ROOT))
from leapsinger import loudness as AGC  # noqa: E402            学習（dataset.py）と同じ実装を使う

AGC_SOURCE = "leapsinger/loudness.py"
AUDIO_SUBDIRS = ("phrase", "phrase_demo", "phrase_demo_cand")   # 古いレイアウト: <親>/<subdir>/<名前>/<フレーズ>.wav、先勝ち
MAX_POINTS = 1600                                                # カードへ送る配列の上限点数
YAML_KEYS = ("sigma_sec", "target_db", "k", "max_gain_db", "max_cut_db")   # config の data.loudness_agc の並び

DATA_DIRS: list[str] = []                                        # 起動時に渡されたフォルダ（画面の初期値）
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "leapsinger" / "loudness_tuner"

app = FastAPI(title="loudness tuner")


# ── パラメータ ────────────────────────────────────────────────────────────────

class Params(BaseModel):
    """agc の引数。既定値は AGC_DEFAULTS だけが持つ（ここにも HTML にも書かない）。"""
    model_config = ConfigDict(extra="forbid")
    sigma_sec: float = Field(default=AGC.AGC_DEFAULTS["sigma_sec"], gt=0.0)
    target_db: float = Field(default=AGC.AGC_DEFAULTS["target_db"])
    k: float = Field(default=AGC.AGC_DEFAULTS["k"], ge=0.0)
    max_gain_db: float = Field(default=AGC.AGC_DEFAULTS["max_gain_db"], ge=0.0)
    max_cut_db: float = Field(default=AGC.AGC_DEFAULTS["max_cut_db"], ge=0.0)


def _phash(p: Params) -> str:
    return hashlib.sha1(json.dumps(p.model_dump(), sort_keys=True).encode()).hexdigest()[:10]


def _params_query(sigma_sec: float = Query(AGC.AGC_DEFAULTS["sigma_sec"], gt=0.0),
                  target_db: float = Query(AGC.AGC_DEFAULTS["target_db"]),
                  k: float = Query(AGC.AGC_DEFAULTS["k"], ge=0.0),
                  max_gain_db: float = Query(AGC.AGC_DEFAULTS["max_gain_db"], ge=0.0),
                  max_cut_db: float = Query(AGC.AGC_DEFAULTS["max_cut_db"], ge=0.0)) -> Params:
    return Params(sigma_sec=sigma_sec, target_db=target_db, k=k, max_gain_db=max_gain_db, max_cut_db=max_cut_db)


# ── キャッシュ ────────────────────────────────────────────────────────────────

class LRU:
    def __init__(self, maxsize: int):
        self.maxsize, self._d, self._lock = maxsize, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)
        return value


_metas: dict = {}
_npz: dict = {}
_npz_locks: dict = {}
_guard = threading.Lock()
_raw = LRU(256)        # (dir, name) -> dict(mel, ph_ids, dur_sec, f0)
_res = LRU(256)        # (dir, name, phash) -> curves + stats
_wav = LRU(64)         # (dir, name) -> float32 [n]
_wavpair = LRU(64)     # (dir, name, phash) -> dict(orig, norm, scale, peak_before, peak_after)
_audio_idx: dict = {}  # dir -> dict(source=..., names={phrase: path | None})


# ── データセットの場所 ────────────────────────────────────────────────────────

def _is_dataset(d: Path) -> bool:
    return (d / "shard.npz").is_file() and (d / "metadata.json").is_file()


def find_datasets(path: Path) -> list[Path]:
    """path 自身がデータセット（shard.npz + metadata.json）ならそれ 1 つ、無ければ直下の子フォルダのうち
    データセットであるものを名前順に。どちらも無ければ 400。"""
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise HTTPException(400, f"フォルダがありません: {path}")
    if _is_dataset(path):
        return [path]
    found = sorted(p for p in path.iterdir() if p.is_dir() and _is_dataset(p))
    if not found:
        raise HTTPException(400, f"{path}: shard.npz と metadata.json のあるフォルダ（またはそれらを含む親フォルダ）を指定してください")
    return found


def _meta(db_dir: Path) -> dict:
    key = str(db_dir)
    if key not in _metas:
        with open(db_dir / "metadata.json", encoding="utf-8") as f:
            _metas[key] = json.load(f)
    return _metas[key]


def _npz_lock(path: Path) -> threading.Lock:
    with _guard:
        return _npz_locks.setdefault(str(path), threading.Lock())


def _open_npz(path: Path):
    """npz を遅延オープン（np.load は zip の目次だけ読み、配列はキーごとに読む）。1 つの npz は 1 本ずつ読む。"""
    key = str(path)
    lock = _npz_lock(path)
    with lock:
        if key not in _npz:
            _npz[key] = np.load(path)
        return _npz[key], lock


def _db_ctx(dir: str) -> dict:
    db_dir = Path(dir).expanduser().resolve()
    if not _is_dataset(db_dir):
        raise HTTPException(400, f"{db_dir} に shard.npz / metadata.json がありません")
    meta = _meta(db_dir)
    fr = float(meta.get("frame_rate") or (meta["sample_rate"] / meta["hop_size"]))
    vocab = meta.get("phoneme_vocab") or {}
    return dict(dir=str(db_dir), name=db_dir.name, db_dir=db_dir, meta=meta, fr=fr,
                hop=int(meta.get("hop_size", 256)), sr=int(meta.get("sample_rate", 44100)),
                pau_id=int(vocab.get("pau", 0)), id2ph={int(v): k for k, v in vocab.items()})


def _raw_phrase(ctx: dict, name: str) -> dict:
    key = (ctx["dir"], name)
    hit = _raw.get(key)
    if hit is not None:
        return hit
    z, lock = _open_npz(ctx["db_dir"] / "shard.npz")
    with lock:
        try:
            d = dict(mel=z[f"{name}|mel"].astype(np.float32),
                     ph_ids=z[f"{name}|phoneme_ids"].astype(np.int64),
                     dur_sec=z[f"{name}|dur_sec"].astype(np.float64),
                     f0=z[f"{name}|f0_interp"].astype(np.float32))
        except KeyError as e:
            raise HTTPException(400, f"シャードにキーがありません: {e}")
    return _raw.put(key, d)


def _curves(ctx: dict, name: str, p: Params) -> dict:
    key = (ctx["dir"], name, _phash(p))
    hit = _res.get(key)
    if hit is not None:
        return hit
    raw = _raw_phrase(ctx, name)
    T = raw["mel"].shape[1]
    sung = ~AGC.pau_mask(raw["ph_ids"], raw["dur_sec"], T, ctx["fr"], ctx["pau_id"])
    cur = AGC.agc_curves(raw["mel"], sung, ctx["fr"], **p.model_dump())
    v = sung if sung.any() else np.ones(T, bool)
    mb, ma = float(cur["E"][v].mean()), float(cur["E_after"][v].mean())
    out = dict(cur, sung=sung, T=T, mean_db=mb, mean_after_db=ma,
               slow_std_before=float((cur["slow_before"] - mb)[v].std()),
               slow_std_after=float((cur["slow_after"] - ma)[v].std()),
               gain_min=float(cur["c_db"][v].min()), gain_max=float(cur["c_db"][v].max()),
               clamp_pct=float(cur["clamped"][v].mean() * 100.0))
    return _res.put(key, out)


# ── 音声 ──────────────────────────────────────────────────────────────────────

def _audio_index(ctx: dict) -> dict:
    """試聴できるフレーズの一覧。① <dir>/shard_wav.npz（前処理の save_wav: true。キー "<フレーズ>|wav"）、
    ② 無ければ古いレイアウト <親>/{phrase,phrase_demo,phrase_demo_cand}/<名前>/<フレーズ>.wav（先に見つかった方）。
    戻り値: dict(source="shard_wav" | "phrase_dir" | None, names={フレーズ: wav のパス | None})"""
    key = ctx["dir"]
    if key in _audio_idx:
        return _audio_idx[key]
    db_dir = ctx["db_dir"]
    if (db_dir / "shard_wav.npz").is_file():
        z, lock = _open_npz(db_dir / "shard_wav.npz")
        with lock:
            names = {k[:-4]: None for k in z.files if k.endswith("|wav")}
        idx = dict(source="shard_wav", names=names)
    else:
        names: dict = {}
        for sub in AUDIO_SUBDIRS:
            d = db_dir.parent / sub / db_dir.name
            if d.is_dir():
                for p in sorted(d.glob("*.wav")):
                    names.setdefault(p.stem, str(p))
        idx = dict(source="phrase_dir" if names else None, names=names)
    _audio_idx[key] = idx
    return idx


def _lead_pad_frames(f0_raw: np.ndarray) -> int:
    """古いシャードの wav は、前処理が付け足した頭の無音（f0 が 0 のまま入っている区間）を含まない。
    その分だけ前に詰めて mel と並べる（dataset.py の _lead_pad_frames と同じ）。"""
    good = np.flatnonzero(f0_raw > 1.0)
    return int(good[0]) if len(good) else 0


def _load_wav(ctx: dict, name: str, T: int) -> np.ndarray:
    key = (ctx["dir"], name)
    hit = _wav.get(key)
    if hit is not None:
        return hit
    idx = _audio_index(ctx)
    n = T * ctx["hop"]
    if idx["source"] == "shard_wav":
        z, lock = _open_npz(ctx["db_dir"] / "shard_wav.npz")
        with lock:
            w = z[f"{name}|wav"].astype(np.float32) / 32767.0
        if len(w) < n:
            w = np.pad(w, (_lead_pad_frames(_raw_phrase(ctx, name)["f0"]) * ctx["hop"], 0))
    else:
        import soundfile as sf
        data, sr = sf.read(idx["names"][name], dtype="float32", always_2d=True)
        if sr != ctx["sr"]:
            raise HTTPException(400, f"{name}.wav: サンプリング周波数 {sr} がシャードの {ctx['sr']} と違います")
        w = data.mean(1).astype(np.float32)
    if len(w) < n:
        w = np.pad(w, (0, n - len(w)))
    elif len(w) > n:
        w = w[:n]
    return _wav.put(key, w)


def _wav_pair(ctx: dict, name: str, p: Params) -> dict:
    """原音と正規化音。両方に同じ scale（正規化音のピークが 0.99 を超えた分だけ）を掛け、相対レベルを保つ。"""
    key = (ctx["dir"], name, _phash(p))
    hit = _wavpair.get(key)
    if hit is not None:
        return hit
    idx = _audio_index(ctx)
    if name not in idx["names"]:
        raise HTTPException(404, f"{name} の音声がありません")
    cur = _curves(ctx, name, p)
    orig = _load_wav(ctx, name, cur["T"])
    norm = AGC.apply_gain_wav(orig, cur["c_db"], ctx["hop"])
    pk_b, pk_a = float(np.abs(orig).max()), float(np.abs(norm).max())
    scale = min(1.0, 0.99 / pk_a) if pk_a > 0 else 1.0
    src = idx["names"][name] or f"shard_wav.npz:{name}|wav"
    return _wavpair.put(key, dict(orig=orig, norm=norm, scale=scale, peak_before=pk_b, peak_after=pk_a,
                                  n_samples=int(len(orig)), src=src))


def _wav_bytes(x: np.ndarray, sr: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _bytes_response(request: Request, data: bytes, media_type: str) -> Response:
    """Range 要求に 206 で応える（Safari の <audio> が要求する）。URL にパラメータが入るので長めにキャッシュ。"""
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "max-age=3600"}
    rng = request.headers.get("range")
    if rng:
        m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
        if m and (m.group(1) or m.group(2)):
            start = int(m.group(1) or 0)
            end = int(m.group(2)) if m.group(2) else len(data) - 1
            end = min(end, len(data) - 1)
            if start > end:
                raise HTTPException(416, "range out of bounds")
            headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
            return Response(data[start:end + 1], status_code=206, media_type=media_type, headers=headers)
    return Response(data, media_type=media_type, headers=headers)


# ── エンドポイント ────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(HERE / "static" / "index.html"))


@app.get("/config")
def config():
    return dict(defaults=AGC.AGC_DEFAULTS, spec=AGC.AGC_SPEC, data_dirs=DATA_DIRS, agc_source=AGC_SOURCE,
                mel_floor=AGC.MEL_FLOOR, max_points=MAX_POINTS, yaml_keys=list(YAML_KEYS))


class ScanReq(BaseModel):
    path: str


@app.post("/scan")
def scan(req: ScanReq):
    dbs = []
    for d in find_datasets(Path(req.path)):
        try:
            meta = _meta(d)
        except Exception as e:                                      # noqa: BLE001
            dbs.append(dict(dir=str(d), name=d.name, error=f"metadata.json が読めません: {e}"))
            continue
        phrases = meta.get("phrases") or {}
        fr = float(meta.get("frame_rate") or (meta["sample_rate"] / meta["hop_size"]))
        ctx = dict(dir=str(d), db_dir=d)
        idx = _audio_index(ctx)
        dbs.append(dict(dir=str(d), name=d.name, n_phrases=len(phrases),
                        hours=round(sum(int(t) for t in phrases.values()) / fr / 3600.0, 2),
                        hop=int(meta.get("hop_size", 0)), frame_rate=round(fr, 3),
                        sample_rate=int(meta.get("sample_rate", 0)),
                        shard_mb=round(os.path.getsize(d / "shard.npz") / 1e6),
                        audio=dict(n=len(idx["names"]), source=idx["source"])))
    return dict(dbs=dbs)


class ComputeReq(BaseModel):
    dir: str
    params: Params = Params()
    sort: str = "name"              # name | shuffle
    seed: int = 0
    query: str = ""
    page: int = 1
    page_size: int = Field(20, ge=1, le=200)
    audio_only: bool = False


def _phrase_names(ctx: dict, query: str, sort: str, seed: int, audio_only: bool) -> list[str]:
    names = sorted(ctx["meta"].get("phrases") or {})
    if audio_only:
        idx = _audio_index(ctx)["names"]
        names = [n for n in names if n in idx]
    if query:
        q = query.lower()
        names = [n for n in names if q in n.lower()]
    if sort == "shuffle":
        random.Random(seed).shuffle(names)
    return names


def _thin(a: np.ndarray, step: int, nd: int = 2) -> list:
    return np.round(np.asarray(a, dtype=np.float64)[::step], nd).tolist()


@app.post("/compute")
def compute(req: ComputeReq):
    ctx = _db_ctx(req.dir)
    names = _phrase_names(ctx, req.query, req.sort, req.seed, req.audio_only)
    total = len(names)
    n_pages = max(1, math.ceil(total / req.page_size))
    page = min(max(1, req.page), n_pages)
    sel = names[(page - 1) * req.page_size: page * req.page_size]
    audio = _audio_index(ctx)
    items, e_lo, e_hi, means = [], [], [], []
    for n in sel:
        raw = _raw_phrase(ctx, n)
        cur = _curves(ctx, n, req.params)
        T = cur["T"]
        step = max(1, math.ceil(T / MAX_POINTS))
        fb = AGC.phoneme_bounds(raw["dur_sec"], T, ctx["fr"])
        bounds = [dict(x=int(fb[k] // step), x2=int(fb[k + 1] // step),
                       ph=ctx["id2ph"].get(int(raw["ph_ids"][k]), str(int(raw["ph_ids"][k]))),
                       pau=bool(int(raw["ph_ids"][k]) == ctx["pau_id"]))
                  for k in range(len(raw["ph_ids"]))]
        v = cur["sung"] if cur["sung"].any() else np.ones(T, bool)
        e_lo.append(float(np.percentile(cur["E"][v], 1)))
        e_hi.append(float(np.percentile(cur["E"][v], 99)))
        means.append(cur["mean_db"])
        items.append(dict(
            name=n, song=re.sub(r"_\d{4,}$", "", n), T=T, dur_sec=round(T / ctx["fr"], 2), step=step,
            E=_thin(cur["E"], step), E_after=_thin(cur["E_after"], step),
            slow_before=_thin(cur["slow_before"], step), slow_after=_thin(cur["slow_after"], step),
            gain=_thin(cur["c_db"], step), sung=cur["sung"][::step].astype(int).tolist(),
            bounds=bounds, mean_db=round(cur["mean_db"], 2), mean_after_db=round(cur["mean_after_db"], 2),
            gain_min=round(cur["gain_min"], 2), gain_max=round(cur["gain_max"], 2),
            slow_std_before=round(cur["slow_std_before"], 2), slow_std_after=round(cur["slow_std_after"], 2),
            clamp_pct=round(cur["clamp_pct"], 1), has_audio=n in audio["names"]))
    return dict(total=total, page=page, n_pages=n_pages, page_size=req.page_size, frame_rate=ctx["fr"],
                hop=ctx["hop"], phash=_phash(req.params), audio_source=audio["source"],
                page_median_db=(round(float(np.median(means)), 2) if means else None),
                y_lo=(round(min(e_lo) - 3, 1) if e_lo else -60), y_hi=(round(max(e_hi) + 3, 1) if e_hi else 0),
                items=items)


@app.get("/audio")
def audio(request: Request, dir: str, name: str, which: str = "norm", p: Params = Depends(_params_query)):
    ctx = _db_ctx(dir)
    pair = _wav_pair(ctx, name, p)
    x = (pair["orig"] if which == "orig" else pair["norm"]) * pair["scale"]
    return _bytes_response(request, _wav_bytes(x, ctx["sr"]), "audio/wav")


@app.get("/audio_info")
def audio_info(dir: str, name: str, p: Params = Depends(_params_query)):
    ctx = _db_ctx(dir)
    pair = _wav_pair(ctx, name, p)
    return dict(src=pair["src"], peak_before=round(pair["peak_before"], 3), peak_after=round(pair["peak_after"], 3),
                scale=round(pair["scale"], 3), n_samples=pair["n_samples"])


@app.get("/mel.png")
def mel_png(request: Request, dir: str, name: str, which: str = "before", p: Params = Depends(_params_query)):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import image as mpimg
    ctx = _db_ctx(dir)
    raw = _raw_phrase(ctx, name)
    cur = _curves(ctx, name, p)
    before = raw["mel"]
    after = AGC.apply_gain_mel(before, cur["c_db"])
    vmax = float(max(before.max(), after.max()))
    step = max(1, math.ceil(before.shape[1] / MAX_POINTS))
    arr = (after if which == "after" else before)[:, ::step]
    buf = io.BytesIO()
    mpimg.imsave(buf, arr, vmin=AGC.MEL_FLOOR, vmax=vmax, cmap="magma", origin="lower", format="png")
    return _bytes_response(request, buf.getvalue(), "image/png")


# ── DB 全体の統計（バックグラウンド 1 ジョブ）────────────────────────────────

_job: dict = {}
_job_lock = threading.Lock()


class StatsReq(BaseModel):
    dir: str
    params: Params = Params()


def _stats_cache_path(ctx: dict, p: Params) -> Path:
    dh = hashlib.sha1(ctx["dir"].encode()).hexdigest()[:6]
    return CACHE_DIR / f"{ctx['name']}_{dh}_{_phash(p)}.json"


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def _run_stats(job: dict, ctx: dict, p: Params):
    t0 = time.time()
    names = sorted(ctx["meta"].get("phrases") or {})
    job["total"] = len(names)
    mb, ma, sb, sa = [], [], [], []
    bands_b = dict(macro=[], meso=[], micro=[]); bands_a = dict(macro=[], meso=[], micro=[])
    gains, clamped, n_frames, n_used = [], 0, 0, 0
    try:
        for i, n in enumerate(names):
            if job.get("cancel"):
                job.update(state="cancelled", elapsed_sec=round(time.time() - t0, 1))
                return
            raw = _raw_phrase(ctx, n)
            T = raw["mel"].shape[1]
            sung = ~AGC.pau_mask(raw["ph_ids"], raw["dur_sec"], T, ctx["fr"], ctx["pau_id"])
            job["done"] = i + 1
            if sung.sum() < ctx["fr"]:                           # 歌唱部 1 秒未満は分析スクリプトと同じく除外
                continue
            cur = AGC.agc_curves(raw["mel"], sung, ctx["fr"], **p.model_dump())
            m_b, m_a = float(cur["E"][sung].mean()), float(cur["E_after"][sung].mean())
            mb.append(m_b); ma.append(m_a)
            sb.append(float((cur["slow_before"] - m_b)[sung].std()))
            sa.append(float((cur["slow_after"] - m_a)[sung].std()))
            db_ = AGC.decompose(cur["E"], sung, ctx["fr"]); da_ = AGC.decompose(cur["E_after"], sung, ctx["fr"])
            for kk in bands_b:
                bands_b[kk].append(db_[f"{kk}_std"]); bands_a[kk].append(da_[f"{kk}_std"])
            gains.append(np.abs(cur["c_db"][sung][::4]))
            clamped += int(cur["clamped"][sung].sum()); n_frames += int(sung.sum()); n_used += 1
        g = np.concatenate(gains) if gains else np.zeros(0)
        mb_, ma_ = np.array(mb), np.array(ma)
        result = dict(
            n_phrases=n_used, n_phrases_total=len(names), n_frames=n_frames,
            between_std=dict(before=round(float(mb_.std()), 2), after=round(float(ma_.std()), 2)),
            phrase_db=dict(median_before=round(float(np.median(mb_)), 2), p10_before=round(_pct(mb_, 10), 2),
                           p90_before=round(_pct(mb_, 90), 2), median_after=round(float(np.median(ma_)), 2)),
            inphrase_slow_std=dict(before=dict(median=round(float(np.median(sb)), 2), p90=round(_pct(sb, 90), 2)),
                                   after=dict(median=round(float(np.median(sa)), 2), p90=round(_pct(sa, 90), 2))),
            bands=dict(before={k: round(float(np.median(v)), 2) for k, v in bands_b.items()},
                       after={k: round(float(np.median(v)), 2) for k, v in bands_a.items()}),
            gain_abs=dict(p50=round(_pct(g, 50), 2), p90=round(_pct(g, 90), 2), p99=round(_pct(g, 99), 2),
                          max=round(float(g.max()), 2) if len(g) else None),
            clamp_pct=round(100.0 * clamped / max(n_frames, 1), 2),
            params=p.model_dump(), phash=_phash(p), elapsed_sec=round(time.time() - t0, 1),
            dir=ctx["dir"], name=ctx["name"])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_stats_cache_path(ctx, p), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        job.update(state="done", result=result, elapsed_sec=result["elapsed_sec"])
    except Exception as e:                                        # noqa: BLE001
        job.update(state="error", error=f"{e.__class__.__name__}: {e}", elapsed_sec=round(time.time() - t0, 1))


@app.post("/stats_start")
def stats_start(req: StatsReq):
    ctx = _db_ctx(req.dir)
    cp = _stats_cache_path(ctx, req.params)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            return dict(job_id=None, cached=True, result=json.load(f))
    with _job_lock:
        if _job.get("state") == "running":
            raise HTTPException(409, "別の統計ジョブが実行中です（stats_cancel で止められます）")
        job_id = hashlib.sha1(f"{ctx['dir']}{_phash(req.params)}{time.time()}".encode()).hexdigest()[:8]
        _job.clear()
        _job.update(id=job_id, state="running", done=0, total=0, t0=time.time(), name=ctx["name"],
                    phash=_phash(req.params), cancel=False)
        threading.Thread(target=_run_stats, args=(_job, ctx, req.params), daemon=True).start()
    return dict(job_id=job_id, cached=False)


@app.get("/stats_progress")
def stats_progress(job_id: str):
    if _job.get("id") != job_id:
        raise HTTPException(404, "そのジョブはありません")
    out = {k: _job.get(k) for k in ("id", "state", "done", "total", "name", "phash", "result", "error")}
    out["elapsed_sec"] = round(time.time() - _job.get("t0", time.time()), 1)
    return out


@app.post("/stats_cancel")
def stats_cancel():
    with _job_lock:
        if _job.get("state") == "running":
            _job["cancel"] = True
    return dict(ok=True)


# ── 起動 ──────────────────────────────────────────────────────────────────────

def main():
    global DATA_DIRS, CACHE_DIR
    ap = argparse.ArgumentParser(description="音量正規化チューナー（data.loudness_agc のパラメータを目と耳で決める）")
    ap.add_argument("--data", nargs="*", default=None, metavar="DIR",
                    help="データセットのフォルダ（shard.npz があるフォルダ）か、それらを含む親フォルダ。複数可。画面でも入力できる")
    ap.add_argument("--port", type=int, default=8018)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cache_dir", default=None, help=f"DB 全体の統計の結果を置く所（既定 {CACHE_DIR}）")
    args = ap.parse_args()
    if args.data:
        DATA_DIRS = [str(Path(d).expanduser().resolve()) for d in args.data]
    if args.cache_dir:
        CACHE_DIR = Path(args.cache_dir).expanduser().resolve()
    import uvicorn
    print(f"loudness tuner  http://{args.host}:{args.port}   agc={AGC_SOURCE}   data={DATA_DIRS or '(画面で入力)'}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
