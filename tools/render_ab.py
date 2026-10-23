"""試聴用の書き出し: 評価用の句を、いくつかのモデル（チェックポイント × 生成の step 数）で生成して、wav と mel の図にする。

    python tools/render_ab.py --config configs/3speaker_gan2d.yaml \
        --data_dirs /content/data/oniku /content/data/natsume /content/data/ritsu \
        --models log/decay_15/ckpt_035000.pt@1 log/teacher_s015/ckpt_035000.pt@16 \
        --names decay_15 s015_k16 --out log/listen_teachers --device cuda

句は TensorBoard の試聴と同じ選び方（評価用の句の並びで、話者ごとに先頭から --per_speaker 本）なので、
TensorBoard の spk{話者}/s{番号} と同じ句になる。<out>/spk{話者}_{句}/ に
  gt.wav      正解の mel をボコーダに通したもの（ボコーダの癖はどれも同じ）
  {名前}.wav  各モデルの生成
  mel.png     上から正解・各モデルの mel（TensorBoard と同じ色の範囲）
音量は TensorBoard と同じく 1 本ずつピークをそろえる（infer.mel_to_wav）。
ローカルで回す場合は torch 1 プロセス・スレッド 2 本で（--threads）。

ポストフィルタの試聴（leapsinger/postfilter.py。フォルマントの山と谷を広げる）:
    --pf_model decay_15 --pf_summary log/eval_averaging/summary.json --pf_label decay_15@k1 --pf_strength 1 2
で、名前 decay_15 のモデルの出力にポストフィルタをかけた decay_15_pf1.wav・decay_15_pf2.wav も書く。α は
eval_ab の summary.json の、そのモデル（--pf_label）のフォルマントのメリハリの比から (1 / 比) ** 強さ。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from infer import infer_mel, load_acoustic, load_vocoder, mel_to_wav  # noqa: E402
from eval_ab import load_eval_items  # noqa: E402   評価用の句の選び方を eval_ab・train.py とそろえる
from leapsinger.phonemes import find_lang  # noqa: E402
from leapsinger.postfilter import (alphas_from_ratios, contrast_postfilter, frame_classes,  # noqa: E402
                                   ratios_from_summary)


def parse_model(spec: str):
    """「チェックポイント@step 数」→ (パス, step 数)。"""
    path, _, k = spec.rpartition("@")
    if not path or not k.isdigit() or int(k) < 1:
        raise SystemExit(f"--models は「チェックポイント@step 数」で指定してください（例: log/x/ckpt_035000.pt@16）: {spec}")
    return path, int(k)


def pick_items(items, per_speaker: int):
    """話者ごとに先頭から per_speaker 本（train.py の TensorBoard の試聴と同じ選び方）。"""
    count, out = {}, []
    for it in items:
        s = int(it.get("spk_id", 0))
        if count.get(s, 0) < per_speaker:
            count[s] = count.get(s, 0) + 1
            out.append(it)
    return out


def write_wav(path, wav, sr: int):
    from scipy.io import wavfile
    wavfile.write(str(path), sr, (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16))


def plot_mels(path, mels: dict, title: str, vmin: float, vmax: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(mels), 1, figsize=(12, 2.0 * len(mels)), sharex=True, squeeze=False)
    for ax, (nm, mel) in zip(axes[:, 0], mels.items()):
        ax.imshow(np.asarray(mel), origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_ylabel(nm, fontsize=8)
    axes[0, 0].set_title(title, fontsize=9)
    axes[-1, 0].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(str(path), dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="評価用の句をいくつかのモデルで生成して、試聴用の wav と mel の図にする")
    ap.add_argument("--config", required=True, help="学習に使った config（評価用の句の選び方と mel の設定）")
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="チェックポイント@step 数（例: log/teacher_s015/ckpt_035000.pt@16）")
    ap.add_argument("--names", nargs="*", default=None, help="表示名・ファイル名（既定は run のフォルダ名_kN）")
    ap.add_argument("--per_speaker", type=int, default=3, help="話者ごとの句の数（TensorBoard の eval_items と同じ 3）")
    ap.add_argument("--vocoder", default=None,
                    help="ボコーダの ONNX（既定は config の train.vocoder）。none なら mel の図だけ")
    ap.add_argument("--out", default="log/listen")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="出発点のノイズの乱数（eval_ab と同じ既定）")
    ap.add_argument("--threads", type=int, default=2, help="CPU で回すときの torch スレッド数")
    ap.add_argument("--pf_model", default=None, help="ポストフィルタもかけるモデルの名前（--names の 1 つ）")
    ap.add_argument("--pf_summary", default=None, help="α を決める eval_ab の summary.json")
    ap.add_argument("--pf_label", default=None, help="summary.json の中のそのモデルの名前（例: decay_15@k1）")
    ap.add_argument("--pf_strength", nargs="+", type=float, default=[1.0],
                    help="強さ（1 = 正解並みに戻す）。複数渡すとそれぞれ書く")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    specs = [parse_model(m) for m in args.models]
    names = args.names or [f"{Path(p).parent.name}_k{k}" for p, k in specs]
    if len(names) != len(specs):
        raise SystemExit("--names の数が --models と違います")
    if "gt" in names or len(set(names)) != len(names):
        raise SystemExit(f"--names は重ならない名前にしてください（gt は正解に使う）: {names}")
    if args.pf_model is not None and (args.pf_model not in names or not args.pf_summary or not args.pf_label):
        raise SystemExit("--pf_model は --names の 1 つにして、--pf_summary と --pf_label も指定してください")
    items, _fr = load_eval_items(cfg, args.data_dirs)
    items = pick_items(items, args.per_speaker)
    sr = int((cfg.get("mel") or {}).get("sr", 44100))
    voc_path = args.vocoder if args.vocoder is not None else (cfg.get("train") or {}).get("vocoder")
    voc = None
    if voc_path and str(voc_path).lower() != "none":
        p = Path(voc_path)
        voc = load_vocoder(str(p if p.is_absolute() or p.exists() else ROOT / p), sr)
    m = cfg.get("model") or {}
    vmin, vmax = float(m.get("mel_vmin", -11.5)), float(m.get("mel_vmax", 2.0))

    mels = [{"gt": it["target_mel"]} for it in items]
    use_uv = {"gt": bool(m.get("use_uv", True))}
    phonemes = {}
    for (path, k), nm in zip(specs, names):
        model, mcfg = load_acoustic(path, args.device)
        use_uv[nm] = bool(mcfg.get("use_uv", True))
        phonemes[nm] = list(mcfg.get("phonemes") or [])
        for i, it in enumerate(items):
            mels[i][nm] = infer_mel(model, it, num_steps=k, device=args.device, seed=args.seed)
        print(f"[{nm}] {path}（{k} step で生成）: {len(items)} 句", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.pf_model is not None:                     # ポストフィルタ（フォルマントの山と谷を広げる）をかけた版
        import json
        ratios = ratios_from_summary(json.load(open(args.pf_summary, encoding="utf-8")), args.pf_label)
        id2ph = dict(enumerate(phonemes[args.pf_model]))
        lang = find_lang(phonemes[args.pf_model])     # 母音と子音の分け方（dict/<lang>.yaml）
        if lang is None:
            raise SystemExit(f"{args.pf_model} の音素の一覧に合う言語パック（dict/<lang>.yaml）が無いので、"
                             "ポストフィルタで母音と子音を分けられません")
        for s in args.pf_strength:
            alpha = alphas_from_ratios(ratios, s)
            nm = f"{args.pf_model}_pf{s:g}"
            for i, it in enumerate(items):
                src = mels[i][args.pf_model]
                cls = frame_classes(it["ph_ids"], it["ph_durs"], id2ph, src.shape[1], lang.vowels)
                mels[i][nm] = contrast_postfilter(src, cls, alpha)
            use_uv[nm] = use_uv[args.pf_model]
            print(f"[{nm}] α（帯域ごと）母音 " + " ".join(f"{a:.2f}" for a in alpha["vowel"])
                  + "・子音 " + " ".join(f"{a:.2f}" for a in alpha["cons"]), flush=True)

    out = Path(args.out)
    for i, it in enumerate(items):
        d = out / f"spk{int(it.get('spk_id', 0))}_{it['item_name']}"
        d.mkdir(parents=True, exist_ok=True)
        if voc is not None:
            for nm, mel in mels[i].items():
                vuv = it["uv"] if use_uv[nm] else np.ones_like(it["uv"])     # train.py の試聴と同じ
                write_wav(d / f"{nm}.wav", mel_to_wav(voc, mel, it["f0_logf0"], vuv), sr)
        plot_mels(d / "mel.png", mels[i], f"spk{int(it.get('spk_id', 0))} {it['item_name']}", vmin, vmax)
    print(f"wrote {len(items)} phrases to {out}" + ("" if voc is not None else "（ボコーダなし: mel の図だけ）"))


if __name__ == "__main__":
    main()
