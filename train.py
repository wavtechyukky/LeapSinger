"""Train the LeapSinger acoustic model.

Two stages in one run: the rectified-flow + reconstruction loss first learns the base voice, then
a lightweight mel-spectrogram discriminator (GAN) is switched on partway through to sharpen the
texture. The schedule — when the GAN turns on, its strength, and so on — is set in the config's
`gan:` section; with `gan.enabled: false` it is plain flow + reconstruction training.

    python -m train --config configs/3speaker_gan2d.yaml \
        --data_dirs data/oniku data/natsume data/ritsu \
        --run_name 3speaker_gan2d --out_root log --device cuda

Re-running the same command resumes from the latest checkpoint.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from leapsinger.config import MelSpec
from dataset import (LeapSingerDataset, acoustic_collate_fn, FrameBasedBatchSampler)
from infer import infer_mel, load_vocoder, mel_to_wav
from leapsinger.modules.discriminators import (
    JCUMelDiscriminator, Mel2DDiscriminator, d_loss_jcu, g_adv_fm_jcu, laplacian_var_ratio)
from preprocess.vocab import Vocab, PAU_ID
from preprocess.phrase_cut import clip_pau_gain
from leapsinger.models.acoustic import HarmonicAcousticModel, HarmonicAcousticModelMultiSpk
from leapsinger.mel import pitch_warp_mel_torch
from leapsinger.modules.harmonic_excitation import harmonic_wave
from leapsinger.loudness import normalize_mel
from leapsinger.phonemes import find_lang
from leapsinger.reflow import config_mismatch, make_reflow_targets, reflow_problems, reflow_settings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mel_fig(mel, title, vmin=-11.5, vmax=2.0):
    """mel [n_mels, T] -> a magma-colormapped matplotlib figure (low freq at the bottom), for
    TensorBoard add_figure. cmap='magma', origin='lower', and a fixed vmin/vmax so GT and
    prediction share one colour scale."""
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(np.asarray(mel), origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap="magma")
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Frame")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    return fig


def _load_eval_ab():
    """tools/eval_ab.py をファイルの場所を指定して読み込む（tools はパッケージではないので、同じ名前の
    別のパッケージと取り違えないように）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_leapsinger_eval_ab", Path(__file__).resolve().parent / "tools" / "eval_ab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gen_metric_specs(ev, band_names) -> list:
    """TensorBoard の eval_gen/ に出す指標 [(タグ, 指標)]。定義は tools/eval_ab.py の build_metrics と同じ。
      err_db              無音以外の全フレームの誤差（dB）
      boundary_sharpness  音素の境界のつなぎの鋭さ（生成/正解、1 未満はぼやけ）
      tex_bN_*            帯域 N の微細テクスチャ（生成/正解、1 未満はのっぺり、1 を超えるとざらつき過多）
      uv_err_db           無声の区間（子音など）の誤差（dB）
      uv_bias_db_bN_*     無声の区間の帯域 N のレベルのずれ（生成 − 正解、dB。低域が + なら倍音の消し残り）
      uv_tex_bN_*         無声の区間の帯域 N の微細テクスチャ（生成/正解）"""
    by_key = {m["key"]: m for m in ev.build_metrics(band_names)}
    lab = [n.replace(" ", "").replace("–", "-") for n in band_names]
    out = [("err_db", by_key["all"]), ("boundary_sharpness", by_key["sharp"])]
    out += [(f"tex_b{b}_{l}", by_key[f"band_tex:{b}"]) for b, l in enumerate(lab)]
    out += [("uv_err_db", by_key["uv_all"])]
    out += [(f"uv_bias_db_b{b}_{l}", by_key[f"uv_band_bias:{b}"]) for b, l in enumerate(lab)]
    out += [(f"uv_tex_b{b}_{l}", by_key[f"uv_band_tex:{b}"]) for b, l in enumerate(lab)]
    return out


def default_nucleus(vocab) -> list:
    """config に model.pos_cond.nucleus が無いときの核音素: 音素の一覧に合う言語パック（dict/<lang>.yaml）の
    nucleus（無ければ vowels）。言語パックが無ければ空。"""
    lang = find_lang(vocab.phonemes)
    return [] if lang is None else sorted(lang.nucleus or lang.vowels)


def pos_cond_config(cfg: dict, vocab) -> dict:
    """model.pos_cond / model.dur_cond を解決して、モデル引数と ckpt に書く形にする。
    nucleus の音素名は vocab で id に引く（無い名前は警告して無視）。"""
    m = cfg["model"]
    pc = dict(m.get("pos_cond") or {})
    names = list(pc.pop("nucleus", None) or default_nucleus(vocab))
    ids, missing = [], []
    for p in names:
        (ids if p in vocab.phoneme2id else missing).append(vocab.phoneme2id.get(p, p))
    if missing and (pc.get("unit") or False):
        print(f"[pos_cond] nucleus に無い音素を無視: {missing}")
    return dict(pos_cond={"phoneme": bool(pc.get("phoneme", False)), "unit": bool(pc.get("unit", False)),
                          "bins": int(pc.get("bins", 8)),
                          # 位置の符号化。キーが無い config / ckpt は旧符号化（2026-10-16 の A/B と同じ）
                          "encoding": str(pc.get("encoding", "raised_cosine"))},
                dur_cond=bool(m.get("dur_cond", False)), nucleus_ids=sorted(ids), pau_id=int(vocab.pau_id))


def build_model(cfg: dict, vocab):
    m, e, mel = cfg["model"], cfg["excitation"], cfg["mel"]
    common = dict(
        n_phonemes=vocab.n_phonemes, hidden=m.get("hidden", 256), mel_bins=mel["n_mels"],
        mel_vmin=m.get("mel_vmin", -11.5), mel_vmax=m.get("mel_vmax", 2.0),
        backbone_ch=m.get("backbone_ch", 256), n_cycles=m.get("n_cycles", 3),
        dilation_schedule=m.get("dilation_schedule", "pow2_15"),
        n_styles=m.get("n_styles", 0),
        flow_loss=m.get("flow_loss", "l2"), use_uv=m.get("use_uv", True),
        **pos_cond_config(cfg, vocab),
    )
    harm = dict(
        n_harm=e.get("n_harm", 50),
        noise_ratio=e.get("noise_ratio", 0.05), exc_scale=e.get("exc_scale", 0.15),
        harm_decay=e.get("harm_decay", 1.0), exc_hop=mel["hop"],
        x0_sigma=e.get("x0_sigma", 0.0),
    )
    spk_dim = int(m.get("spk_dim", 0))
    if spk_dim > 0:
        model = HarmonicAcousticModelMultiSpk(
            **common, **harm, n_speakers=int(m["n_speakers"]), spk_dim=spk_dim)
        arch = "harmonic_multispk"
    else:
        model = HarmonicAcousticModel(**common, **harm, n_speakers=int(m.get("n_speakers", 0)))
        arch = "harmonic"
    return model, arch


def resolve_vocab(phonemes_path, data_dirs):
    """Training phoneme vocab. Explicit --phonemes wins; else the phoneme list stored in the
    shard metadata (so it always matches how the data was preprocessed); else the bundled
    Japanese default. The chosen list is saved into the checkpoint (export/infer reuse it)."""
    if phonemes_path:
        return Vocab.load(phonemes_path)
    for d in data_dirs:
        mp = os.path.join(d, "metadata.json")
        if os.path.exists(mp):
            meta = json.load(open(mp, encoding="utf-8"))
            if meta.get("phonemes"):
                return Vocab(meta["phonemes"])
    return Vocab.load()                              # bundled Japanese


def _pos_arch(pc: dict, dur_cond) -> tuple:
    """モデルの構造を決める位置の設定 (phoneme, unit, encoding, bins, dur_cond)。位置が off なら encoding は
    関係なく、bins は旧符号化（raised_cosine）でだけ効くので、関係ない所は None にそろえる。"""
    ph, un = bool(pc.get("phoneme", False)), bool(pc.get("unit", False))
    enc = str(pc.get("encoding", "raised_cosine")) if (ph or un) else None
    bins = int(pc.get("bins", 8)) if enc == "raised_cosine" else None
    return ph, un, enc, bins, bool(dur_cond)


def align_pos_cond_to_ckpt(cfg: dict, ck_cfg: dict):
    """途中から再開する run の model.pos_cond / model.dur_cond を ckpt の config に合わせる（cfg を書き換える）。
    位置の条件が無かった頃の ckpt は全部 off、encoding の無い ckpt は旧符号化とみなす。返り値: 変えたら
    (変更前, 変更後) の (phoneme, unit, encoding, bins, dur_cond)、同じなら None。"""
    pcfg = dict(cfg["model"].get("pos_cond") or {})
    cur = _pos_arch(pcfg, cfg["model"].get("dur_cond", False))
    new = _pos_arch(dict(ck_cfg.get("pos_cond") or {}), ck_cfg.get("dur_cond", False))
    if cur == new:
        return None
    pcfg.update(phoneme=new[0], unit=new[1])
    if new[2] is not None:
        pcfg["encoding"] = new[2]
    if new[3] is not None:
        pcfg["bins"] = new[3]
    cfg["model"]["pos_cond"] = pcfg
    cfg["model"]["dur_cond"] = new[4]
    return cur, new


def lr_factor(step: int, decay: float, every: int) -> float:
    """学習率の倍率 decay ** (step / every)。毎 step なめらかに下げる（decay = 1 で一定）。
    例: decay 0.8・every 5000 は DiffSinger の StepLR(5000, 0.8) と同じ減り方で、20k で 0.41、35k で 0.21、50k で 0.11 倍。"""
    decay = float(decay)
    return decay ** (step / max(1, int(every))) if decay < 1.0 else 1.0


def cons_weight_settings(tr: dict):
    """子音の損失の重み（train.cons_weight。既定 1 = 付けない）と、子音の直後の音素の頭にも同じ重みを付ける長さ
    （train.cons_weight_next_ms。既定 30 ms）。付けないときは None。"""
    w = tr.get("cons_weight")
    w = 1.0 if w is None else float(w)
    if w == 1.0:
        return None
    nm = tr.get("cons_weight_next_ms")
    nm = 30.0 if nm is None else float(nm)
    if not (w > 0 and math.isfinite(w)) or not (nm >= 0 and math.isfinite(nm)):
        raise SystemExit(f"train.cons_weight={w}（0 より大きく。1 = 付けない）・train.cons_weight_next_ms={nm}"
                         "（0 以上）にしてください")
    return {"weight": w, "next_ms": nm}


def train_recipe(cfg: dict) -> dict:
    """ckpt に残す学習の設定（バッチ・学習率・精度など、学習の進み方を変えるもの）。再開時に比べる。"""
    tr, g, d = (cfg.get(k, {}) or {} for k in ("train", "gan", "data"))
    decay = float(tr.get("lr_decay", 1.0) or 1.0)
    amp = str(tr.get("amp") or "fp32").lower()
    return {"lr": float(tr.get("lr", 1e-4)), "d_lr": float(g.get("d_lr", 2.0e-4)), "lr_decay": decay,
            "lr_decay_steps": int(tr.get("lr_decay_steps", 5000) or 5000) if decay < 1.0 else None,
            "max_batch_frames": int(tr.get("max_batch_frames", 60000)), "max_batch_size": int(tr.get("max_batch_size", 16)),
            "min_batch_items": int(tr.get("min_batch_items", 0) or 0), "max_sec": d.get("max_sec") or None,
            "amp": amp, "amp_fp32_io": bool(tr.get("amp_fp32_io", True)) if amp != "fp32" else None,
            "reflow": reflow_settings(tr),             # reflow の生徒の設定（教師・組の数など）。None = reflow しない
            "cons_weight": cons_weight_settings(tr)}   # 子音の損失の重み。None = 付けない


CODE_FILES = ("train.py", "dataset.py", "infer.py")


def code_fingerprint(root) -> dict:
    """学習に効くコード（train.py・dataset.py・infer.py・leapsinger/ 以下の .py）の指紋と、git の commit・
    未コミットの変更の有無。同じコマンドでもコードが変われば指紋が変わる。ckpt に残し、再開時に比べる。"""
    import hashlib
    import subprocess
    root = Path(root)
    files = sorted([root / f for f in CODE_FILES] + list((root / "leapsinger").rglob("*.py")))
    h = hashlib.sha1()
    for f in files:
        if f.exists():
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    out = {"sha1": h.hexdigest()[:10], "commit": None, "dirty": None}
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}                  # 読むだけ（index を書き換えない）
    try:
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=30, env=env)
        if rev.returncode == 0:
            st = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=60, env=env)
            out["commit"] = rev.stdout.strip() or None
            out["dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def describe_code(c: dict) -> str:
    dirty = {True: "（未コミットの変更あり）", False: "", None: ""}[c.get("dirty")]
    return f"commit {c.get('commit') or '?'}{dirty}・指紋 {c.get('sha1')}"


def recipe_changes(cur: dict, saved):
    """ckpt に記録した学習の設定 saved から変わった項目 {キー: (ckpt の値, 今の値)}。記録の無い古い ckpt は None。"""
    if not saved:
        return None
    return {k: (saved.get(k), v) for k, v in cur.items() if saved.get(k) != v}


EXC_DEFAULTS = {"n_harm": 50, "noise_ratio": 0.05, "exc_scale": 0.15, "harm_decay": 1.0,   # = build_model
                "x0_sigma": 0.0}


def excitation_changes(cfg: dict, ck_cfg: dict) -> dict:
    """再開した run で、ckpt を作ったときから変えた励起の設定 {キー: (ckpt の値, 今の値)}。励起は重みを
    持たないので、変えたまま再開できる（例: 共通の 20k の ckpt から noise_ratio だけ変えて GAN 段階を比べる）。"""
    e = cfg.get("excitation", {}) or {}
    out = {}
    for k, d in EXC_DEFAULTS.items():
        if k in ck_cfg and float(ck_cfg[k]) != float(e.get(k, d)):
            out[k] = (ck_cfg[k], e.get(k, d))
    return out


def ckpt_config(cfg: dict, arch: str, vocab) -> dict:
    m, e, mel, tr = cfg["model"], cfg["excitation"], cfg["mel"], cfg["train"]
    return {
        "arch": arch, "n_phonemes": vocab.n_phonemes, "phonemes": vocab.phonemes,
        **pos_cond_config(cfg, vocab),             # pos_cond / dur_cond / nucleus_ids / pau_id
        "hidden": m.get("hidden", 256), "mel_bins": mel["n_mels"],
        "mel_vmin": m.get("mel_vmin", -11.5), "mel_vmax": m.get("mel_vmax", 2.0),
        "backbone_ch": m.get("backbone_ch", 256), "n_cycles": m.get("n_cycles", 3),
        "dilation_schedule": m.get("dilation_schedule", "pow2_15"),
        "n_speakers": int(m.get("n_speakers", 0)), "n_styles": m.get("n_styles", 0),
        "spk_dim": int(m.get("spk_dim", 0)),
        "flow_loss": m.get("flow_loss", "l2"), "use_uv": m.get("use_uv", True),
        "n_harm": e.get("n_harm", 50), "noise_ratio": e.get("noise_ratio", 0.05),
        "exc_scale": e.get("exc_scale", 0.15), "harm_decay": e.get("harm_decay", 1.0),
        "x0_sigma": e.get("x0_sigma", 0.0),
        "exc_hop": mel["hop"], "hop": mel["hop"], "sample_rate": mel["sr"],
        "recon_weight": tr.get("recon_weight", 1.0), "infer_steps": tr.get("num_steps", 10),
        # 学習目標 mel に掛けた遅い AGC（leapsinger/loudness.py）。None = 掛けていない。推論側が
        # 出力レベルの前提（target_db）を知るために記録する
        "loudness_agc": (cfg.get("data", {}) or {}).get("loudness_agc") or None,
    }


def _forward(model, b, device, recon_weight):
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
    out = model(
        b["ph_ids"], b["ph_durs"], b["f0_logf0"], b["uv"], b["target_mel"],
        padding_mask=b["ph_mask"], spk_id=b["spk_id"], style_id=b["style_id"],
        frame_mask=b["frame_mask"], harm_wave=b.get("harm_wave"))
    loss = out["flow"]
    if recon_weight > 0 and "recon" in out:
        loss = loss + recon_weight * out["recon"]
    return loss, out


def _forward_flow_gan(model, b, flow_loss_type: str, amp_dtype=None, spans=None, reflow_t=None):
    """_forward の GAN 版: flow 損失 + GAN ペア素材を単一 velocity_fn forward で返す。
    数式は mel_dilated_rectified_flow.compute_loss と同一（同じ mask / t 分布）。
    ★x0 は励起（randn ではない）。RNG 順（_encode dropout → 励起 randn → t rand）も compute_loss と一致。
    spans（GpuSpans）: エンコーダ / 励起（擬似 mel）/ 速度場ネット＋損失 の終わりに mark を打つ。
    reflow_t（train.reflow_t。None = reflow しない）: reflow の生徒。目標は正解の mel ではなく、組の出発点（バッチの
    reflow_x0、無ければ reflow_eps で作る）から教師が解いた mel（reflow_x1）。zero = t=0 だけ、uniform = 一様。
    判別器の正例（x1_n）と varL は正解の mel のまま。
    バッチに frame_w [B, T]（train.cons_weight。子音のフレームの重み）があれば、flow 損失と mel_l1 をその重みの
    付いた平均にする（重みの合計で割るので、全体の大きさは変わらない）。gt_l1 は重みを付けない。"""
    mark = spans.mark if spans is not None else (lambda name: None)
    flow = model.flow
    # amp_dtype（config train.amp: bf16）: エンコーダと速度場ネットだけを autocast で回す。励起（STFT）と
    # 損失・x1_pred の計算は fp32 のまま（精度が要る所と、判別器へ渡す値は崩さない）
    amp = (torch.autocast(device_type=b["target_mel"].device.type, dtype=amp_dtype)
           if amp_dtype is not None else contextlib.nullcontext())
    with amp:
        cond = model._encode(
            b["ph_ids"], b["ph_durs"], b["f0_logf0"], b["uv"], b["ph_mask"],
            max_frames=b["target_mel"].shape[2], spk_id=b["spk_id"],
            style_id=(b["style_id"] if getattr(model, "style_emb", None) is not None else None))
    mark("enc")
    gt = flow._norm(b["target_mel"])
    if reflow_t is None:
        x1 = gt
        x0 = model._excitation_x0(b["f0_logf0"], b["uv"], harm_wave=b.get("harm_wave"))   # ★励起（fp32）
    else:                                              # reflow の生徒: 組の出発点と教師の出力
        x1 = b["reflow_x1"].float()                    # fp16 で届く
        if "reflow_x0" in b:                           # 出発点そのもの（波形の白色ノイズがあるとき、fp16 で届く）
            x0 = b["reflow_x0"].float()
        else:
            x0 = model._excitation_x0(b["f0_logf0"], b["uv"], harm_wave=b.get("harm_wave"), eps=b["reflow_eps"])
    mark("exc")
    t = (torch.zeros(x1.shape[0], device=x1.device) if reflow_t == "zero"
         else torch.rand(x1.shape[0], device=x1.device))
    x_t = x0 + t[:, None, None] * (x1 - x0)
    with amp:
        v = flow.velocity_fn(x_t, t, cond)
    v = v.float()
    diff = v - (x1 - x0)
    mask = (~b["frame_mask"]) if "frame_mask" in b and b["frame_mask"] is not None else None
    fw = b.get("frame_w")                              # 子音のフレームの重み [B, T]（無ければ重みを付けない）
    if mask is None and fw is None:                    # compute_loss と同一の縮約
        loss = diff.abs().mean() if flow_loss_type == "l1" else diff.pow(2).mean()
    else:
        e = diff.abs() if flow_loss_type == "l1" else diff.pow(2)
        m = mask[:, None, :].to(e.dtype) if mask is not None else torch.ones_like(e[:, :1, :])
        if fw is not None:
            m = m * fw[:, None, :].to(e.dtype)
        loss = (e * m).sum() / m.sum().clamp(min=1.0) / e.shape[1]
    x1_pred_n = x_t + (1.0 - t[:, None, None]) * v     # 正規化域の 1-step clean 予測
    vmask = mask[:, None, :] if mask is not None else torch.ones_like(x1[:, :1, :], dtype=torch.bool)
    wmask = vmask if fw is None else vmask * fw[:, None, :].to(x1.dtype)
    mel_l1 = ((x1_pred_n - x1).abs() * wmask).sum() / (wmask.sum().clamp(min=1) * x1.shape[1])  # = _recon_loss
    mark("net")
    out = {"flow": loss, "mel_l1": mel_l1, "cond": cond, "t": t, "x_t": x_t,
           "x1_n": gt, "x1_pred_n": x1_pred_n, "x1_pred": flow._denorm(x1_pred_n)}
    if reflow_t is not None:                           # 正解の mel との差（記録用。学習には使わない）
        with torch.no_grad():
            out["gt_l1"] = ((x1_pred_n - gt).abs() * vmask).sum() / (vmask.sum().clamp(min=1) * gt.shape[1])
    return out


def _adaptive_adv_weight(recon_ref, gan_term, last_w, cap: float):
    """VQGAN 式: 生成器最終層重みでの ∥∇recon∥/∥∇GAN∥ を毎 step 計測し、GAN 勾配が
    再構成の錨を上回らないよう自動スケール（cap で上限）。GAN 暴走への構造的対処。"""
    g_r = torch.autograd.grad(recon_ref, last_w, retain_graph=True)[0].norm()
    g_a = torch.autograd.grad(gan_term, last_w, retain_graph=True)[0].norm()
    return (g_r / (g_a + 1e-8)).clamp(max=cap).detach()


_MEL_FLOOR = float(np.log(1e-5))     # dataset._MEL_FLOOR と一致(preprocess mel の clamp 下限)


def maybe_pitch_augment(b, tr, mel_cfg, model, *, silence, fade_frames, loudness_agc=None):
    """pitch_aug: 拡張する item だけ per-item 半音 st でピッチシフト。target_mel(周波数ワープ)・
    f0_logf0(+st/12)・harm_wave(シフト後 f0 から再生成)を同じ連続 r=2^(st/12) で一致更新。周波数軸
    のみワープ=T 不変ゆえ ph_durs/uv/frame_mask は不変。未拡張 item(st=0)はキャッシュ済み harm_wave を
    流用(=高速化)。mel と f0 が同一 st ゆえ f0↔音は構造的に一致(NHVSing の量子化ズレなし)。fp32・
    no_grad。b を in-place 更新して返す。b['wav'] 無し(recipe save_wav 未実施)なら素通し=現行と同一。
    loudness_agc: dataset と同じ遅い AGC を、ワープ後の mel にも掛ける（生 wav 由来＝未処理のため）。"""
    if (not tr.get("pitch_aug", False)) or b.get("wav") is None:
        return b
    lo, hi = tr.get("pitch_aug_semitones", [-3.0, 3.0])
    prob   = float(tr.get("pitch_aug_prob", 0.5))
    fp     = bool(tr.get("pitch_aug_formant_preserve", True))
    dev = b["target_mel"].device
    B, _, T = b["target_mel"].shape
    st = torch.empty(B, device=dev).uniform_(float(lo), float(hi))
    st = st * (torch.rand(B, device=dev) < prob).to(st.dtype)          # prob 未満の item は st=0(素通し)
    idx = torch.nonzero(st != 0, as_tuple=False).flatten()
    if idx.numel() == 0:
        return b
    with torch.no_grad():
        # 1) 周波数軸ワープした mel(フォルマント保存可)。T 不変。
        new_mel = pitch_warp_mel_torch(
            b["wav"][idx].float(), st[idx], sr=mel_cfg["sr"], n_fft=mel_cfg["n_fft"], hop=mel_cfg["hop"],
            win=mel_cfg["win"], n_mels=mel_cfg["n_mels"], fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"],
            formant_preserve=fp)                                        # [k, M, T']
        if new_mel.shape[2] != T:                                     # wav パディング由来の T ずれを整合
            new_mel = (new_mel[:, :, :T] if new_mel.shape[2] > T
                       else torch.nn.functional.pad(new_mel, (0, T - new_mel.shape[2]), value=_MEL_FLOOR))
        # 2) 遅い AGC と無音フェードの再適用(生 wav 由来=未処理 → 未拡張 item と整合。dataset.__getitem__ をミラー)
        if (silence and fade_frames > 0) or loudness_agc:
            phi = b["ph_ids"][idx].cpu().numpy(); dur = b["ph_durs"][idx].cpu().numpy()
            Tv  = (~b["frame_mask"][idx]).sum(1).cpu().numpy()
            frame_rate = float(mel_cfg["sr"]) / float(mel_cfg["hop"])
            for j in range(idx.numel()):
                Ti = int(Tv[j]); fb = np.concatenate([[0], np.cumsum(dur[j])]).clip(max=Ti)
                sil = np.zeros(Ti, dtype=bool)
                for p in range(len(phi[j])):
                    if phi[j][p] == PAU_ID:
                        sil[fb[p]:fb[p + 1]] = True
                if loudness_agc:
                    m = normalize_mel(new_mel[j, :, :Ti].cpu().numpy(), ~sil, frame_rate, **loudness_agc)
                    new_mel[j, :, :Ti] = torch.as_tensor(m, device=dev, dtype=new_mel.dtype)
                if silence and fade_frames > 0 and sil.any():
                    logg = torch.as_tensor(np.log(np.clip(clip_pau_gain(sil, fade_frames), 1e-12, 1.0)),
                                           device=dev, dtype=new_mel.dtype)
                    new_mel[j, :, :Ti] = torch.clamp_min(new_mel[j, :, :Ti] + logg[None, :], _MEL_FLOOR)
        b["target_mel"][idx] = new_mel.to(b["target_mel"].dtype)
        # 3) f0 を同じ r でシフト(全 item・st=0 は +0)。encoder cond と励起の両方に効く。
        b["f0_logf0"] = b["f0_logf0"] + (st / 12.0)[:, None]
        # 4) harm_wave 再生成(拡張 item のみ・シフト後 f0 から。未拡張はキャッシュ流用=高速化の要)
        if "harm_wave" in b:
            f0l  = b["f0_logf0"][idx]                                  # [k, T](シフト済)
            uvsh = torch.ones_like(f0l) if not model.use_uv else b["uv"][idx]
            harm = harmonic_wave(f0l, uvsh, n_harm=model.n_harm, hop=model.exc_hop,
                                 harm_decay=model.harm_decay)          # [k, T*hop]
            N = min(harm.shape[1], b["harm_wave"].shape[1])
            b["harm_wave"][idx, :N] = harm[:, :N].to(b["harm_wave"].dtype)
    return b


def _rand_window(tensors, win: int):
    """時間軸で長さ win の同一ランダム窓を切る（全サンプル共有）。win<=0 or T<=win なら無変更。
    tensors: [B, C, T] のリスト。GAN-TTS 流に「全体を見せない」ための短窓。"""
    T = tensors[0].shape[-1]
    if win <= 0 or T <= win:
        return tensors
    s = int(torch.randint(0, T - win + 1, (1,)))
    return [x[..., s:s + win] for x in tensors]


class StepMonitor:
    """崩壊の原因を後から追うための記録（学習には一切手を入れない）。

    毎 step のクリップ前の勾配ノルム（G、GAN 中は D も）・バッチの本数・損失を貯め、log_interval ごとに
    最大値・中央値を TensorBoard へ出す（50 step に 1 回の瞬間値だけだと、単発の跳ねを見落とすため）。
    G の勾配ノルムが直近 window step の中央値の spike_factor 倍を超えた step は、バッチの中身
    （本数・最長・フレーズ名）ごと <out_dir>/spikes.log とコンソールに残す。
    add_time() で 1 step の時間の内訳（データ待ち・GPU への転送・計算）も貯め、time/* に平均を出す。
    add_gpu() は計算の中の GPU 時間の内訳（GpuSpans: 擬似 mel・速度場ネット・逆伝播など）で、time/gpu_*。"""

    def __init__(self, log_path, window: int = 200, spike_factor: float = 5.0, warmup: int = 200):
        self.log_path = log_path
        self.recent = collections.deque(maxlen=window)
        self.spike_factor = spike_factor
        self.warmup = warmup
        self._reset()

    def _reset(self):
        self.gn, self.gn_d, self.items, self.loss = [], [], [], []
        self.times, self.n_times = collections.defaultdict(float), 0
        self.gpu, self.n_gpu = collections.defaultdict(float), 0

    def add_time(self, **secs):
        """1 step の時間の内訳 [秒]（例: data_wait=, h2d=, step=）。"""
        for k, v in secs.items():
            self.times[k] += float(v)
        self.n_times += 1

    def add_gpu(self, **secs):
        """step の計算部分の GPU 時間の内訳 [秒]（GpuSpans.pop() の値）。steps_per_sec には数えない。"""
        if not secs:
            return
        for k, v in secs.items():
            self.gpu[k] += float(v)
        self.n_gpu += 1

    def update(self, step: int, gn, gn_d, n_items: int, loss: float, describe) -> str | None:
        """1 step 分を記録。跳ねていればその記録行を返す（describe() はバッチの説明文字列）。"""
        gn = float(gn)
        self.gn.append(gn); self.items.append(int(n_items)); self.loss.append(float(loss))
        if gn_d is not None:
            self.gn_d.append(float(gn_d))
        line = None
        if len(self.recent) >= self.warmup:
            med = float(np.median(self.recent))
            if (not np.isfinite(gn)) or gn > self.spike_factor * med:
                line = (f"step {step}: grad_norm {gn:.4g} (median {med:.4g}, x{gn / max(med, 1e-12):.1f})"
                        f"  loss {loss:.4f}  {describe()}")
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        if np.isfinite(gn):
            self.recent.append(gn)
        return line

    def summary(self) -> dict:
        """前回の summary 以降の要約（TensorBoard のタグ名 → 値）。"""
        out = {}
        if self.gn:
            out["train/grad_norm_median"] = float(np.median(self.gn))
            out["train/grad_norm_max"] = float(np.max(self.gn))
        if self.gn_d:
            out["train/d_grad_norm_max"] = float(np.max(self.gn_d))
        if self.items:
            out["train/batch_items_min"] = float(np.min(self.items))
        if self.loss:
            out["train/loss_max"] = float(np.max(self.loss))
        if self.n_times:
            for k, v in self.times.items():
                out[f"time/{k}_ms"] = v / self.n_times * 1000.0
            out["time/steps_per_sec"] = self.n_times / max(sum(self.times.values()), 1e-9)
        if self.n_gpu:
            for k, v in self.gpu.items():
                out[f"time/gpu_{k}_ms"] = v / self.n_gpu * 1000.0
        self._reset()
        return out


class GpuSpans:
    """1 step の中の区間ごとの GPU 時間。CUDA イベントを打つだけで同期は足さない（読むのは step 末の
    .item() で GPU を待った後）。CPU では演算が同期的なので perf_counter の差がそのまま区間時間。
    区間の名前 = その区間の終わりに打った mark の名前（aug / enc / exc / net / gan / bwd / opt）。"""

    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self.marks = []

    def mark(self, name: str):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.marks.append((name, ev))
        else:
            self.marks.append((name, time.perf_counter()))

    def pop(self) -> dict:
        """{区間名: 秒} を返して marks を空に戻す。"""
        marks, self.marks = self.marks, []
        if len(marks) < 2:
            return {}
        if self.cuda:
            marks[-1][1].synchronize()
            return {n: a.elapsed_time(b) / 1000.0 for (_, a), (n, b) in zip(marks, marks[1:])}
        return {n: b - a for (_, a), (n, b) in zip(marks, marks[1:])}


def main():
    ap = argparse.ArgumentParser(description="Train the LeapSinger acoustic model with a light GAN.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_root", default="log")
    ap.add_argument("--max_updates", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--phonemes", default=None,
                    help="phoneme list file (default: shard metadata / bundled Japanese)")
    ap.add_argument("--init_from", default=None,
                    help="finetune 起点の flow ckpt（G 重みのみ読み込み）")
    ap.add_argument("--finetune", action="store_true",
                    help="init_from の G 重みだけ読み、step/optimizer をリセット・D は新規（GAN post-filter）")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="config の点区切り上書き（繰り返し可）。例: --set model.pos_cond.phoneme=true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg.setdefault("mel", {}); cfg.setdefault("model", {})
    cfg.setdefault("excitation", {}); cfg.setdefault("train", {})
    # A/B 用の点区切り上書き: --set model.pos_cond.phoneme=true --set model.dur_cond=true
    for kv in (args.set or []):
        key, _, raw = kv.partition("=")
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[parts[-1]] = yaml.safe_load(raw)
        print(f"config override: {key} = {node[parts[-1]]!r}")
    cfg["mel"] = MelSpec.from_dict(cfg["mel"]).to_dict()
    tr = cfg["train"]
    g = cfg.get("gan", {})
    if args.max_updates:
        tr["max_updates"] = args.max_updates
    device = torch.device(args.device)

    # ── 途中から再開する run は、始めたときの構造（位置の条件）のまま続ける ─────────────────────
    # config の既定が後から変わっても（2026-10-17 に pos_cond.phoneme の既定を true にした）古い run を
    # 再開できるように、最新 ckpt の pos_cond / dur_cond に合わせる。ckpt はここで 1 回だけ読み、下の
    # resume でそのまま使う
    code = code_fingerprint(Path(__file__).resolve().parent)       # どのコードで学習したか（ckpt にも残す）
    print(f"[code] {describe_code(code)}")
    out_dir = Path(args.out_root) / args.run_name
    _ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    _resume_ck = torch.load(_ckpts[-1], map_location="cpu") if _ckpts else None
    if _resume_ck is not None:
        _changed = align_pos_cond_to_ckpt(cfg, _resume_ck.get("config") or {})
        if _changed:
            print(f"[resume] 位置の条件を {_ckpts[-1].name} に合わせます: (phoneme, unit, encoding, bins, dur_cond) "
                  f"{_changed[0]} -> {_changed[1]}。途中から再開する run の構造は変えられません（別の設定で"
                  f"学習するなら --run_name を変える）")
        _exc = excitation_changes(cfg, _resume_ck.get("config") or {})
        if _exc:
            print(f"[resume] 励起の設定を {_ckpts[-1].name} から変えて続けます: "
                  + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in _exc.items()))
        _rec = recipe_changes(train_recipe(cfg), _resume_ck.get("train_recipe"))
        if _rec is None:
            print(f"[resume] {_ckpts[-1].name} には学習の設定（バッチ・学習率など）の記録がありません"
                  "（2026-10-17 より前の run）。始めたときと同じ設定で続けるか確かめてください")
        elif _rec:
            print(f"[resume] 学習の設定が {_ckpts[-1].name} と違います（途中から変えて続けます）: "
                  + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in _rec.items()))
        _old_code = _resume_ck.get("code")
        if _old_code and _old_code.get("sha1") != code["sha1"]:
            print(f"[resume] コードが {_ckpts[-1].name} を作ったときと違います: {describe_code(_old_code)} -> "
                  f"{describe_code(code)}。同じ条件で続けるなら、そのときのコードに戻してください")

    # ── data（データ供給の足場）────────────────────────────
    dcfg = cfg.get("data", {})
    spk_map = dict(dcfg.get("spk_map") or {})
    style_map = dict(dcfg.get("style_map") or {})
    for d in args.data_dirs:
        dbname = os.path.basename(os.path.normpath(d))
        rp = os.path.join("configs", "recipes", f"{dbname}.yaml")
        if os.path.exists(rp):
            rr = yaml.safe_load(open(rp, encoding="utf-8"))
            spk_map.setdefault(dbname, int(rr.get("spk_id", 0)))
            style_map.setdefault(dbname, int(rr.get("style_id", 0)))
    print(f"[spk_map] {spk_map}\n[style_map] {style_map}")
    eval_dbs = dcfg.get("eval_dbs")
    if eval_dbs is not None and not eval_dbs:
        raise ValueError("data.eval_dbs が空です（eval split が空になります）")
    loudness_agc = dcfg.get("loudness_agc") or None
    print(f"[loudness_agc] {loudness_agc if loudness_agc else 'off'}")
    max_sec = dcfg.get("max_sec") or None
    print(f"[max_sec] {max_sec if max_sec else 'off'}")
    train_ds = LeapSingerDataset(args.data_dirs, "train", eval_songs=dcfg.get("eval_songs", 2),
                                eval_dbs=eval_dbs,
                                min_sec=dcfg.get("min_sec", 0.3), pitch_aug=tr.get("pitch_aug", False),
                                silence=dcfg.get("silence", True),
                                silence_fade_sec=dcfg.get("silence_fade_sec", 0.05),
                                spk_map=spk_map, style_map=style_map, loudness_agc=loudness_agc,
                                max_sec=max_sec)
    if "n_speakers" not in cfg["model"]:
        cfg["model"]["n_speakers"] = (max(spk_map.values()) + 1) if spk_map else 1

    vocab = resolve_vocab(args.phonemes, args.data_dirs)
    lang = find_lang(vocab.phonemes)                 # 音素の一覧と中身が同じ言語パック（dict/<lang>.yaml）
    print(f"[vocab] {vocab.n_phonemes} phonemes  言語パック: "
          + (f"dict/{lang.name}.yaml" if lang else "なし（音素の一覧に合う dict/<lang>.yaml がありません）"))
    # 子音の損失の重み（train.cons_weight）: 子音のフレームと直後の音素の頭の flow 損失・recon を重くする。
    # 学習の損失だけに付ける（eval/loss・生成の指標は今までどおり重みなしで、ほかの run と比べられる）
    cons_w = cons_weight_settings(tr)
    if cons_w is not None:
        if lang is None:
            raise SystemExit("[cons_weight] 子音を決める言語パックがありません。音素の一覧（dict/<lang>.phonemes）を"
                             "参照する dict/<lang>.yaml を作ってください（dict/ja.yaml が例）")
        _cids = lang.consonant_ids(vocab.phonemes)
        if not _cids:
            raise SystemExit(f"[cons_weight] dict/{lang.name}.yaml では子音が 1 つもありません")
        _nf = int(round(cons_w["next_ms"] / 1000.0 * train_ds.frame_rate))
        train_ds.set_consonant_weights(_cids, cons_w["weight"], next_frames=_nf)
        _s = train_ds.weighted_frame_share()
        _ws = cons_w["weight"] * _s
        print(f"[cons_weight] 子音 {len(_cids)} 種のフレームと直後の音素の頭 {_nf} フレーム（{cons_w['next_ms']:g} ms）の"
              f"損失を ×{cons_w['weight']:g}（flow 損失と recon。重みの合計で割るので全体の大きさは同じ）。"
              f"該当は全フレームの {100 * _s:.1f}% で、重みの合計に占める割合は {100 * _s:.1f}% -> "
              f"{100 * _ws / max(_ws + 1.0 - _s, 1e-12):.1f}%")
    else:
        print("[cons_weight] off")
    model, arch = build_model(cfg, vocab)
    model = model.to(device)
    _pc = pos_cond_config(cfg, vocab)
    print(f"[pos_cond] {_pc['pos_cond']}  dur_cond={_pc['dur_cond']}  nucleus_ids={_pc['nucleus_ids']}")
    print(f"model {arch}  {sum(p.numel() for p in model.parameters())/1e6:.2f}M params  "
          f"n_speakers={cfg['model']['n_speakers']}")

    # pitch_aug でも harm キャッシュを warm する。未拡張 item はキャッシュ流用、拡張 item のみ
    # maybe_pitch_augment がシフト後 f0 で再生成して上書き(部分キャッシュ=高速化)。
    train_ds.warm_harm_cache(n_harm=model.n_harm, harm_decay=model.harm_decay,
                             exc_hop=model.exc_hop, use_uv=model.use_uv, device=device)

    # ── reflow の生徒（train.reflow_teacher）: 教師で組を作って dataset に持たせ、教師の重みから始める ──────
    # 組は ckpt に保存しない（教師の ckpt と種から同じものが作れる）ので、再開するときも作り直す
    reflow = reflow_settings(tr)
    reflow_meta = None
    if reflow is not None:
        _bad = reflow_problems(cfg)
        if _bad:
            raise SystemExit("[reflow] 設定を直してください:\n  " + "\n  ".join(_bad))
        if args.init_from:
            raise SystemExit("[reflow] --init_from とは一緒に使えません（生徒は教師の重みから始めます）")
        _tck = torch.load(reflow["teacher"], map_location="cpu")
        _mis = config_mismatch(ckpt_config(cfg, arch, vocab), _tck.get("config") or {})
        if _mis:
            raise SystemExit("[reflow] 教師の ckpt と設定が違います（教師の値 -> 今の値）。--set で教師に合わせてください:\n  "
                             + "\n  ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in _mis.items()))
        reflow_meta = {**reflow, "teacher_step": _tck.get("step"), "teacher_code": _tck.get("code")}
        print(f"[reflow] 教師 {reflow['teacher']}（step {_tck.get('step')}・{describe_code(_tck.get('code') or {})}）"
              f"  組 {reflow['draws']} 通り・教師 {reflow['teacher_steps']} step（{reflow['algorithm']}）"
              f"・t={reflow['t']}・seed {reflow['seed']}")
        teacher, _ = build_model(cfg, vocab)
        teacher.load_state_dict(_tck["model"])
        teacher = teacher.to(device)
        _x1, _x0 = make_reflow_targets(teacher, train_ds, draws=reflow["draws"], teacher_steps=reflow["teacher_steps"],
                                       algorithm=reflow["algorithm"], seed=reflow["seed"], device=device)
        train_ds.attach_reflow_targets(_x1, seed=reflow["seed"], x0=_x0)
        _x1 = _x0 = None
        del teacher
        if _resume_ck is None:
            model.load_state_dict(_tck["model"])
            print("[reflow] 生徒を教師の重みから始めます（optimizer と step は新しく始める）")
        _tck = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if bool(g.get("enabled", True)):
            print("[reflow] GAN も on です（判別器の正例は正解の mel）")

    _balance_by = tr.get("balance_by") or ("speaker" if tr.get("balance_speakers", False) else None)
    _weights = None
    if _balance_by:
        _keys = {"speaker": train_ds.speaker_ids, "db": train_ds.db_ids}.get(_balance_by)
        if _keys is None:
            raise SystemExit(f"balance_by='{_balance_by}' は不正（speaker | db）")
        _cnt = collections.Counter(_keys)
        _weights = [1.0 / _cnt[k] for k in _keys]
        print(f"[dataset] balanced sampling by '{_balance_by}'")
    min_batch_items = int(tr.get("min_batch_items", 0) or 0)
    print(f"[sampler] min_batch_items={min_batch_items if min_batch_items else 'off (短い順に詰める従来の方式)'}")
    sampler = FrameBasedBatchSampler(train_ds.frame_counts, tr.get("max_batch_frames", 60000),
                                     tr.get("max_batch_size", 16), shuffle=True, weights=_weights,
                                     min_items=min_batch_items)
    _nw = tr.get("num_workers", 2)
    _dl_kw = dict(pin_memory=True)
    if _nw > 0:
        _dl_kw.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=acoustic_collate_fn,
                        num_workers=_nw, **_dl_kw)

    opt_name = tr.get("optimizer", "radam")
    Opt = torch.optim.RAdam if opt_name == "radam" else torch.optim.AdamW
    opt = Opt(model.parameters(), lr=tr.get("lr", 1e-4))

    # ── 判別器 + D optimizer（軽量 1D JCU、TTUR）───────────────────────────────
    gan_enabled = bool(g.get("enabled", True))
    n_cond = int(cfg["model"]["n_speakers"]) if g.get("cond", "speaker") == "speaker" else 0
    d_type = g.get("d_type", "jcu")                    # jcu(1D, DiffGAN-TTS原典) | mel2d(2D patch)
    _DISC = {"jcu": JCUMelDiscriminator, "mel2d": Mel2DDiscriminator}
    if d_type not in _DISC:
        raise SystemExit(f"gan.d_type='{d_type}' は不正（jcu | mel2d）")
    _dkw = dict(mel_bins=cfg["mel"]["n_mels"], n_cond=n_cond)
    if d_type == "mel2d":
        _dkw["spectral_norm"] = bool(g.get("d_spectral_norm", True))   # Lipschitz拘束で発散抑制
    disc = _DISC[d_type](**_dkw).to(device)
    d_optimizer = torch.optim.Adam(disc.parameters(), lr=float(g.get("d_lr", 2.0e-4)),
                                   betas=tuple(g.get("d_betas", [0.5, 0.9])))
    last_w = model.flow.velocity_fn.out[-1].weight     # VQGAN 適応重み用（生成器の最終層）
    print(f"[gan] enabled={gan_enabled}  d_type={d_type}  "
          f"disc {sum(p.numel() for p in disc.parameters())/1e6:.2f}M  "
          f"d_lr={g.get('d_lr',2e-4)}  window={g.get('window',64)}  cond={g.get('cond','speaker')}  "
          f"fm={'dynamic' if g.get('fm_dynamic',True) else g.get('fm_weight',10.0)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_out = ckpt_config(cfg, arch, vocab)
    recon_weight = tr.get("recon_weight", 1.0)
    max_updates = tr.get("max_updates", 60000)
    save_interval = tr.get("save_interval", 5000)
    log_interval = tr.get("log_interval", 50)
    eval_interval = tr.get("eval_interval", 2000)
    eval_n = int(tr.get("eval_items", 4))
    grad_clip = tr.get("grad_clip", 5.0)

    recipe = train_recipe(cfg)

    def save(step):
        p = out_dir / f"ckpt_{step:06d}.pt"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "disc": disc.state_dict(), "d_optimizer": d_optimizer.state_dict(),
                    "step": step, "config": cfg_out, "train_recipe": recipe, "code": code,
                    "n_speakers": cfg["model"]["n_speakers"], "reflow": reflow_meta}, p)
        tqdm.write(f"  saved {p}")

    # ── TensorBoard eval──────────────────────────────
    writer = SummaryWriter(str(out_dir))
    eval_ds = LeapSingerDataset(args.data_dirs, "eval", eval_songs=dcfg.get("eval_songs", 2),
                                eval_dbs=eval_dbs,
                                min_sec=dcfg.get("min_sec", 0.3),
                                silence=dcfg.get("silence", True),
                                silence_fade_sec=dcfg.get("silence_fade_sec", 0.05),
                                spk_map=spk_map, style_map=style_map, loudness_agc=loudness_agc,
                                max_sec=max_sec)
    _picks: dict = {}
    for _idx, (_sh, _nm, _db) in enumerate(eval_ds.files):
        _s = int(spk_map.get(os.path.basename(os.path.normpath(_db)), 0))
        _picks.setdefault(_s, [])
        if len(_picks[_s]) < eval_n:
            _picks[_s].append(_idx)
    eval_samples = [(s, k, eval_ds[idx]) for s in sorted(_picks) for k, idx in enumerate(_picks[s])]
    # 生成の指標（train.eval_gen_metrics、既定 true）: 評価のたびに評価用の句を全部、推論と同じ step 数
    # （train.num_steps）で生成して正解と比べ、tools/eval_ab.py と同じ定義の値を TensorBoard の eval_gen/ に出す。
    # ランダムな t で測る eval/loss と違い、出発点のノイズ（x0_sigma）や GAN の有無が違う run どうしでも比べられる。
    # 同じ step のチェックポイントを eval_ab（--seed 0）で測った値と同じになる（細かい差の判定は eval_ab で）
    gen_eval = None
    if bool(tr.get("eval_gen_metrics", True)):
        _ev = _load_eval_ab()
        _bands = _ev.band_labels(int(cfg["mel"]["n_mels"]), float(cfg["mel"]["fmin"]), float(cfg["mel"]["fmax"]))
        gen_eval = {"ev": _ev, "items": [eval_ds[i] for i in range(len(eval_ds))],
                    "id2ph": dict(enumerate(vocab.phonemes)), "specs": gen_metric_specs(_ev, _bands)}
        print(f"[eval] 生成の指標: 評価用の {len(gen_eval['items'])} 句を {tr.get('num_steps', 10)} step で生成して"
              f" TensorBoard の eval_gen/ に出す（帯域 b0〜b3 = {', '.join(_bands)}）")
    sr = int(cfg["mel"]["sr"])
    voc, vc = None, tr.get("vocoder")
    if vc:
        try:
            voc = load_vocoder(vc); print(f"[eval] vocoder loaded ({vc})")
        except Exception as e:                        # noqa: BLE001
            print(f"[eval] vocoder load failed ({e}); mel only")
    _gt_done: set = set()
    mvmin = float(cfg["model"].get("mel_vmin", -11.5))
    mvmax = float(cfg["model"].get("mel_vmax", 2.0))

    def log_eval(step):
        # 評価は固定シードで回すが、学習側の乱数は評価の前の状態へ戻す（戻さないと、評価のたびに
        # 学習の乱数列（t・励起ノイズ・dropout）が毎回同じ状態から再開してしまう）
        rng_cpu = torch.random.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            _log_eval(step)
        finally:
            torch.random.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)
            model.train()

    def _log_eval(step):
        model.eval()
        with torch.no_grad():
            torch.manual_seed(1234)
            eb = acoustic_collate_fn([it for _, _, it in eval_samples])
            eloss, eout = _forward(model, eb, device, recon_weight)
            writer.add_scalar("eval/loss", eloss.item(), step)
            writer.add_scalar("eval/flow", eout["flow"].item(), step)
            writer.add_scalar("eval/recon", float(eout.get("recon", 0)), step)
            _vsum = 0.0
            for s, k, it in eval_samples:
                tag = f"spk{s}/s{k}"
                pred = infer_mel(model, it, num_steps=tr.get("num_steps", 10), device=str(device))
                writer.add_figure(f"{tag}_pred_mel",
                                  _mel_fig(pred, f"{tag} pred @ {step}", mvmin, mvmax), step)
                _vsum += laplacian_var_ratio(torch.as_tensor(pred)[None],
                                             torch.as_tensor(it["target_mel"])[None])
                if (s, k) not in _gt_done:
                    writer.add_figure(f"{tag}_gt_mel",
                                      _mel_fig(it["target_mel"], f"{tag} GT", mvmin, mvmax), step)
                if voc is not None:
                    vuv = it["uv"] if getattr(model, "use_uv", True) else np.ones_like(it["uv"])
                    writer.add_audio(f"{tag}_pred", mel_to_wav(voc, pred, it["f0_logf0"], vuv),
                                     step, sample_rate=sr)
                    if (s, k) not in _gt_done:
                        writer.add_audio(f"{tag}_gt",
                                         mel_to_wav(voc, it["target_mel"], it["f0_logf0"], vuv),
                                         step, sample_rate=sr)
                _gt_done.add((s, k))
            writer.add_scalar("eval/varL", _vsum / max(1, len(eval_samples)), step)  # GT比(1.0=同等)
            if gen_eval is not None:                  # 生成の指標（eval_ab と同じ定義・同じ seed 0）
                ev = gen_eval["ev"]
                stats = [ev.phrase_stats(infer_mel(model, it, num_steps=tr.get("num_steps", 10), device=str(device)),
                                         it["target_mel"], it["ph_ids"], it["ph_durs"], gen_eval["id2ph"],
                                         eval_ds.frame_rate, uv=it.get("uv"), lang=lang)
                         for it in gen_eval["items"]]
                for tag, metric in gen_eval["specs"]:
                    v = ev.metric_value(stats, metric)
                    if np.isfinite(v):                # 無声の区間が無いときなどは出さない
                        writer.add_scalar(f"eval_gen/{tag}", v, step)
        model.train()

    # ── resume / finetune ──────────────────────────────────────────────────────
    step = 0
    if _resume_ck is not None:                        # 同 run を継続（G+opt+D+d_opt+step）。上で CPU に読んだもの
        ck, _resume_ck = _resume_ck, None
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            try: opt.load_state_dict(ck["optimizer"])
            except Exception as e: print(f"[resume] optimizer skipped ({e})")  # noqa: BLE001
        if "disc" in ck:                              # arch不一致(D差し替え)なら新規Dにフォールバック
            try: disc.load_state_dict(ck["disc"])
            except Exception as e: print(f"[resume] disc arch mismatch -> fresh D ({e})")  # noqa: BLE001
        if "d_optimizer" in ck:
            try: d_optimizer.load_state_dict(ck["d_optimizer"])
            except Exception as e: print(f"[resume] d_optimizer skipped ({e})")  # noqa: BLE001
        step = int(ck.get("step", 0))
        print(f"[resume] {_ckpts[-1].name} -> step {step}")
    elif args.init_from and args.finetune:            # GAN post-filter: G 重みのみ・D 新規・step 0
        ck = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"[finetune] G weights <- {args.init_from} (D fresh, step 0)")
    ck = None                                         # 読んだ ckpt を学習中ずっと抱えない

    # 学習率: train.lr_decay < 1 なら毎 step なめらかに下げる（lr_factor）。判別器の d_lr も同じ倍率で
    # 下げて、生成器との比（TTUR）を保つ。step から決まるので、再開しても続きの値になる
    lr_g0, lr_d0 = float(tr.get("lr", 1e-4)), float(g.get("d_lr", 2.0e-4))
    lr_decay = float(tr.get("lr_decay", 1.0) or 1.0)
    lr_decay_steps = int(tr.get("lr_decay_steps", 5000) or 5000)

    def set_lr(s):
        f = lr_factor(s, lr_decay, lr_decay_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr_g0 * f
        for pg in d_optimizer.param_groups:
            pg["lr"] = lr_d0 * f

    if lr_decay < 1.0:
        print(f"[lr] {lr_g0:g}（D {lr_d0:g}）× {lr_decay}^(step/{lr_decay_steps}): "
              + ", ".join(f"{s // 1000}k {lr_factor(s, lr_decay, lr_decay_steps):.2f}x" for s in (20000, 35000, 50000)))
    else:
        print(f"[lr] {lr_g0:g}（D {lr_d0:g}）一定")
    set_lr(step)

    d_ema = None
    gan_start = int(g.get("gan_start_step", 0))
    d_warmup = int(g.get("d_warmup_steps", 1000))
    gan_ramp = int(g.get("gan_ramp_steps", 1000))
    d_gate_floor = float(g.get("d_gate_floor", 0.3))
    adv_weight = float(g.get("adv_weight", 1.0))
    fm_dynamic = bool(g.get("fm_dynamic", True))
    fm_weight = float(g.get("fm_weight", 10.0))
    window = int(g.get("window", 64))
    grad_cap = float(g.get("grad_cap", 1.0))
    adaptive_adv = bool(g.get("adaptive_adv", True))
    gan_strength = float(g.get("gan_strength", 1.0))   # 適応重みへの倍率。収束済みfinetuneは∇recon極小
    #  なので 1.0(=パリティ)だとGANが弱すぎる。>1 で「GAN勾配=gan_strength×∇recon」に自己較正
    #  （reconに紐づくので暴走しない）。無制御にしたい時のみ adaptive_adv:false（発散注意）。

    # train.amp: fp32（既定）| bf16。bf16 はエンコーダと速度場ネットだけ（A100 などで速くなる）
    amp_name = str(tr.get("amp") or "fp32").lower()
    if amp_name not in ("fp32", "bf16"):
        raise SystemExit(f"train.amp='{amp_name}' は不正（fp32 | bf16）")
    amp_dtype = torch.bfloat16 if amp_name == "bf16" else None
    # train.amp_fp32_io（既定 true）: bf16 でも速度場ネットの入出力の層・残差と skip の和は fp32 に残す
    model.flow.velocity_fn.fp32_io = bool(tr.get("amp_fp32_io", True))
    print(f"[amp] {amp_name}" + ((" （速度場ネットの入出力と和は fp32）" if model.flow.velocity_fn.fp32_io
                                  else " （速度場ネットも全部 bf16）") if amp_dtype is not None else ""))

    print(f"training -> {out_dir}  (max_updates={max_updates}, start {step}, recon_weight={recon_weight})")
    monitor = StepMonitor(out_dir / "spikes.log")
    spans = GpuSpans(device)
    model.train()
    pbar = tqdm(total=max_updates, initial=step, desc=args.run_name, unit="step", dynamic_ncols=True)
    _done = False
    t_mark = time.perf_counter()                     # 時間の内訳: 前の step の終わり（評価・保存の後）
    while step < max_updates:
        for b in loader:
            t_got = time.perf_counter()              # ここまでがデータ待ち
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            t_moved = time.perf_counter()            # GPU への転送
            set_lr(step)
            spans.mark("start")
            b = maybe_pitch_augment(b, tr, cfg["mel"], model,
                                    silence=train_ds.silence, fade_frames=train_ds.silence_fade_frames,
                                    loudness_agc=train_ds.loudness_agc)
            spans.mark("aug")
            losses = _forward_flow_gan(model, b, tr.get("flow_loss", cfg["model"].get("flow_loss", "l2")),
                                       amp_dtype=amp_dtype, spans=spans,
                                       reflow_t=reflow["t"] if reflow is not None else None)
            g_total = losses["flow"]
            d_loss = None; d_logits = {}; g_adv = g_fm = None; fm_w = 0.0; w_ad = 1.0
            gn_d = None

            gan_on = gan_enabled and step >= gan_start
            _g = step - gan_start - d_warmup
            d_only = gan_on and _g < 0
            ramp = 0.0 if (not gan_on or d_only) else min(1.0, (_g + 1) / max(1, gan_ramp))

            if gan_on:
                cond_id = b["spk_id"] if n_cond > 0 else None
                x_t, t = losses["x_t"], losses["t"]
                real_j = losses["x1_n"]
                fake_j = losses["x1_pred_n"].clamp(-1.0, 1.0)
                x_t_w, real_w, fake_w = _rand_window([x_t, real_j, fake_j], window)
                # D step（fake はヘルパー内で detach）
                d_optimizer.zero_grad()
                d_loss, d_logits = d_loss_jcu(disc, x_t_w, real_w, fake_w, t, cond_id)
                gate_open = (d_gate_floor <= 0.0 or d_ema is None or d_ema >= d_gate_floor)
                if gate_open:
                    d_loss.backward()
                    gn_d = nn.utils.clip_grad_norm_(disc.parameters(), 1.0)   # クリップ前のノルム（記録用）
                    d_optimizer.step()
                d_ema = d_loss.item() if d_ema is None else 0.99 * d_ema + 0.01 * d_loss.item()
                # G step（warmup 後・ランプ付き）
                if not d_only:
                    g_adv, g_fm = g_adv_fm_jcu(disc, x_t_w, real_w, fake_w, t, cond_id)
                    fm_w = (losses["mel_l1"].item() / max(g_fm.item(), 1e-8)) if fm_dynamic else fm_weight
                    gan_term = adv_weight * g_adv + fm_w * g_fm
                    recon_ref = losses["flow"] + recon_weight * losses["mel_l1"]
                    w_ad = gan_strength * (_adaptive_adv_weight(recon_ref, gan_term, last_w, grad_cap)
                                           if adaptive_adv else 1.0)
                    g_total = losses["flow"] + ramp * w_ad * gan_term

            spans.mark("gan")
            g_total = g_total + recon_weight * losses["mel_l1"]    # 土台の recon（常時ON）
            opt.zero_grad()
            g_total.backward()
            spans.mark("bwd")
            gn_g = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # クリップ前のノルム（記録用）
            opt.step()
            spans.mark("opt")
            step += 1
            pbar.update(1)
            _spike = monitor.update(
                step, gn_g, gn_d, len(b["item_name"]),
                (losses["flow"] + recon_weight * losses["mel_l1"]).item(),
                lambda: (f"n={len(b['item_name'])} maxT={b['target_mel'].shape[2] / train_ds.frame_rate:.1f}s  "
                         + ",".join(b["item_name"][:6])))
            if _spike:
                tqdm.write("[spike] " + _spike)
            # monitor.update の .item() で GPU を待つので、ここまでが 1 step の計算時間
            monitor.add_time(data_wait=t_got - t_mark, h2d=t_moved - t_got, step=time.perf_counter() - t_moved)
            monitor.add_gpu(**spans.pop())

            if step % log_interval == 0:
                pbar.set_postfix(flow=f"{losses['flow'].item():.4f}",
                                 recon=f"{losses['mel_l1'].item():.4f}",
                                 d=f"{d_loss.item():.3f}" if d_loss is not None else "-")
                writer.add_scalar("train/flow", losses["flow"].item(), step)
                writer.add_scalar("train/recon", losses["mel_l1"].item(), step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                if "gt_l1" in losses:                # reflow の生徒: 1 step の出力と正解の mel の差（正規化域 L1）
                    writer.add_scalar("train/reflow_gt_l1", losses["gt_l1"].item(), step)
                # パディング（損失をマスクしていて出力が自由なフレーム）を除いて測る
                writer.add_scalar("train/varL_ratio",
                                  laplacian_var_ratio(losses["x1_pred"], model.flow._denorm(losses["x1_n"]),
                                                      valid=~b["frame_mask"]), step)
                for _tag, _v in monitor.summary().items():   # 区間内の最大・中央値（単発の跳ねも残る）
                    writer.add_scalar(_tag, _v, step)
                if d_loss is not None:
                    writer.add_scalar("train/d_loss", d_loss.item(), step)
                    writer.add_scalar("train/d_real_logit", d_logits["d_real_logit"], step)
                    writer.add_scalar("train/d_fake_logit", d_logits["d_fake_logit"], step)
                    writer.add_scalar("train/gan_ramp", ramp, step)
                if g_adv is not None:
                    writer.add_scalar("train/adv", g_adv.item(), step)
                    writer.add_scalar("train/fm", g_fm.item(), step)
                    writer.add_scalar("train/fm_w", fm_w, step)
                    writer.add_scalar("train/adaptive_w", float(w_ad), step)
            if step % eval_interval == 0:
                _t = time.perf_counter()
                log_eval(step)
                writer.add_scalar("time/eval_s", time.perf_counter() - _t, step)
            if step % save_interval == 0:
                _t = time.perf_counter()
                save(step)
                writer.add_scalar("time/save_s", time.perf_counter() - _t, step)
            t_mark = time.perf_counter()
            if step >= max_updates:
                _done = True; break
        if _done:
            break
    pbar.close()
    save(step)
    log_eval(step)
    writer.close()
    print("done.")


if __name__ == "__main__":
    main()
