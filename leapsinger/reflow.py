"""reflow の生徒（1 step）の学習に使う組 — 出発点 x0 と、教師がそこから解いた mel x1。

教師は確率的な出発点・GAN なしで学習した複数 step のモデル。学習データの各フレーズで出発点を D 通り作り、教師を
K step 走らせて、出力 mel を RAM に持つ（正規化域・fp16）。生徒の学習では、同じ x0 から教師の出力へ 1 step で
行くように学ぶ（train.py の train.reflow_teacher）。出発点のランダムさの入れ方で 2 通り:

* 擬似 mel に波形の白色ノイズが無い（excitation.noise_ratio 0）: ランダムさは正規化した mel に足すノイズ
  x0_sigma · eps だけ。eps はフレーズ（"DB名/フレーズ名"）と番号 d から決まる乱数の種で作るので保存せず、学習中は
  dataset が同じ eps を作り直す。eps は CPU の numpy の乱数（PCG64）で作る（torch の CPU の乱数は種の下位 32 bit
  しか使わないので、フレーズ・番号・seed を種に詰めると、別の組が同じノイズになりうる）。
* 波形の白色ノイズがある（noise_ratio > 0。σ = 0 でもよい）: 出発点 x0 そのものも組と一緒に RAM に持つ（RAM は
  2 倍）。波形のノイズはフレーズと seed から決まる種の torch の乱数で作るので、作り直しても同じ組になる。
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

# train.reflow_* の既定（config に書かなければこれ）
REFLOW_DEFAULTS = {"draws": 4, "teacher_steps": 8, "algorithm": "euler", "t": "zero", "seed": 0}
# 教師の ckpt の config と違ってよい項目。ほかは生徒の設定と同じでないと、組や初期値の重みが合わない
FREE_KEYS = ("recon_weight", "infer_steps", "flow_loss")


def reflow_settings(tr: dict) -> dict | None:
    """config の train.reflow_* を読む。train.reflow_teacher（教師の ckpt のパス）が無ければ None（reflow しない）。
      reflow_draws          1 フレーズあたりの組の数 D（出発点の通り数）
      reflow_teacher_steps  組を作るときの教師の step 数 K
      reflow_algorithm      そのときの解き方（euler | rk2 | rk4）
      reflow_t              生徒の学習の t。zero = t=0 だけ（1 step 用の蒸留）| uniform = 一様（2 回目の rectified flow）
      reflow_seed           出発点の種のずらし（同じ教師から別の組を作るとき）"""
    path = tr.get("reflow_teacher")
    if not path:
        return None
    d = REFLOW_DEFAULTS
    s = {"teacher": str(path),
         "draws": int(tr.get("reflow_draws", d["draws"])),
         "teacher_steps": int(tr.get("reflow_teacher_steps", d["teacher_steps"])),
         "algorithm": str(tr.get("reflow_algorithm", d["algorithm"])),
         "t": str(tr.get("reflow_t", d["t"])),
         "seed": int(tr.get("reflow_seed", d["seed"]))}
    if s["draws"] < 1 or s["teacher_steps"] < 1:
        raise SystemExit(f"train.reflow_draws と train.reflow_teacher_steps は 1 以上にしてください: {s}")
    if s["algorithm"] not in ("euler", "rk2", "rk4"):
        raise SystemExit(f"train.reflow_algorithm='{s['algorithm']}' は不正（euler | rk2 | rk4）")
    if s["t"] not in ("zero", "uniform"):
        raise SystemExit(f"train.reflow_t='{s['t']}' は不正（zero | uniform）")
    return s


def stores_x0(noise_ratio: float) -> bool:
    """出発点 x0 そのものを組と一緒に持つか（擬似 mel に波形の白色ノイズがあるとき）。"""
    return float(noise_ratio) > 0.0


def reflow_problems(cfg: dict) -> list:
    """reflow の生徒の学習で組を作れない・再現できなくなる設定（あれば理由の文のリスト）。"""
    e, tr = cfg.get("excitation") or {}, cfg.get("train") or {}
    out = []
    if not float(e.get("x0_sigma", 0.0)) > 0 and not stores_x0(e.get("noise_ratio", 0.05)):
        out.append("excitation.x0_sigma も noise_ratio も 0 です（出発点が 1 通りになり、D 通りの組が全部同じに"
                   "なる）。教師と同じ値を指定してください")
    if tr.get("pitch_aug", False):
        out.append("train.pitch_aug が on です（F0 をずらすと、組の教師の出力と合わなくなる）")
    return out


def config_mismatch(student: dict, teacher: dict) -> dict:
    """生徒の config（student）と教師の ckpt の config（teacher）で違う項目 {キー: (教師の値, 生徒の値)}。
    どちらも train.ckpt_config の形。FREE_KEYS は比べない。教師の ckpt に無い項目は教師の値を None とする。"""
    return {k: (teacher.get(k), v) for k, v in student.items()
            if k not in FREE_KEYS and teacher.get(k) != v}


def reflow_entropy(key: str, d: int, seed: int = 0) -> list:
    """フレーズ key（"DB名/フレーズ名"）の d 番目の組の乱数の種（numpy の SeedSequence に渡す 32 bit × 4 語:
    名前の SHA-1 の 64 bit・d・seed）。語の数を固定して、どの組み合わせも別の種になるようにする。"""
    h = int.from_bytes(hashlib.sha1(key.encode("utf-8")).digest()[:8], "little")
    return [h & 0xFFFFFFFF, h >> 32, int(d), int(seed)]


def reflow_eps(key: str, d: int, mel_bins: int, n_frames: int, seed: int = 0) -> torch.Tensor:
    """d 番目の組の出発点のノイズ eps [mel_bins, n_frames]（float32・CPU）。同じ引数なら毎回同じ値。"""
    rng = np.random.default_rng(reflow_entropy(key, d, seed))
    return torch.from_numpy(rng.standard_normal((mel_bins, n_frames), dtype=np.float32))


def wave_seed(key: str, seed: int = 0) -> int:
    """フレーズ key の波形の白色ノイズの乱数の種（63 bit。D 通りは同じ乱数の続きで作る）。"""
    return int.from_bytes(hashlib.sha1(f"{key}|{int(seed)}|wave".encode("utf-8")).digest()[:8], "little") >> 1


@torch.no_grad()
def make_reflow_targets(teacher, ds, *, draws: int, teacher_steps: int, algorithm: str = "euler",
                        seed: int = 0, device="cpu", log_every: int = 500):
    """ds（dataset.LeapSingerDataset）の全フレーズで組を作る -> (x1, x0)。x1 = {index: np.float16 [draws,
    mel_bins, T]}（教師の出力、正規化域）。x0 = 同じ形の出発点（擬似 mel に波形の白色ノイズがあるとき。無ければ
    None で、学習中は eps を作り直す）。各フレーズは D 通りを 1 つのバッチにして教師に通す（同じ長さなので詰め物が
    無く、推論（infer_mel、batch 1）と同じ条件になる）。倍音波形は dataset の事前計算（warm_harm_cache）があれば使う。"""
    dev = torch.device(device)
    was_training = teacher.training
    teacher.eval()
    mel_bins = int(teacher.flow.mel_bins)
    keep_x0 = stores_x0(teacher.noise_ratio)
    use_spk = hasattr(teacher, "spk_bank") or getattr(teacher, "spk_emb", None) is not None   # = infer_mel
    use_style = getattr(teacher, "style_emb", None) is not None
    out, out0, nbytes, t0, n = {}, ({} if keep_x0 else None), 0, time.time(), len(ds)
    est = sum(ds.frame_counts) * mel_bins * 2 * draws * (2 if keep_x0 else 1) / 1e9
    print(f"[reflow] 組を作ります: {n} フレーズ × {draws} 通り（教師 {teacher_steps} step・{algorithm}）、"
          f"RAM 約 {est:.2f} GB" + ("（波形の白色ノイズがあるので出発点も持つ）" if keep_x0 else ""), flush=True)

    def rep(a, dt):                                   # 同じ item を draws 本並べる [draws, ...]
        x = torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        return x[None].repeat(draws, *([1] * x.dim()))

    for i in range(n):
        it = ds[i]
        T = int(it["target_mel"].shape[1])
        if len(it["f0_logf0"]) != T:
            raise ValueError(f"{ds.phrase_key(i)}: F0 {len(it['f0_logf0'])} フレームと mel {T} フレームが合いません")
        key = ds.phrase_key(i)
        f0, uv = rep(it["f0_logf0"], torch.float32), rep(it["uv"], torch.float32)
        hw = rep(it["harm_wave"], torch.float32) if "harm_wave" in it else None
        eps = (torch.stack([reflow_eps(key, d, mel_bins, T, seed) for d in range(draws)]).to(dev)
               if teacher.x0_sigma > 0 else None)
        gen = torch.Generator(device=dev).manual_seed(wave_seed(key, seed)) if keep_x0 else None
        x0 = teacher._excitation_x0(f0, uv, harm_wave=hw, eps=eps, generator=gen)          # [draws, mel, T]
        kw = dict(phoneme_ids=rep(it["ph_ids"], torch.long), ph_durations=rep(it["ph_durs"], torch.long),
                  f0_logf0=f0, uv=uv, n_frames=T, num_steps=teacher_steps, algorithm=algorithm, x0=x0)
        if use_spk:
            kw["spk_id"] = torch.full((draws,), int(it.get("spk_id", 0)), dtype=torch.long, device=dev)
        if use_style:
            kw["style_id"] = torch.full((draws,), int(it.get("style_id", 0)), dtype=torch.long, device=dev)
        x1 = teacher.flow._norm(teacher.infer(**kw))
        if not torch.isfinite(x1).all():
            raise FloatingPointError(f"{key}: 教師の出力に NaN / inf があります")
        a = x1.to(torch.float16).cpu().numpy()
        out[i] = a
        nbytes += a.nbytes
        if keep_x0:
            a0 = x0.to(torch.float16).cpu().numpy()
            out0[i] = a0
            nbytes += a0.nbytes
        if log_every and (i + 1) % log_every == 0:
            print(f"[reflow]   {i + 1}/{n}（{time.time() - t0:.0f} 秒）", flush=True)
    teacher.train(was_training)
    print(f"[reflow] 組を作りました: {nbytes / 1e9:.2f} GB（fp16）、{time.time() - t0:.0f} 秒", flush=True)
    return out, out0
