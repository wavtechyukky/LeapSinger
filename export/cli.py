"""LeapSinger acoustic-model ONNX exporter.

    python -m export.cli --ckpt log/3speaker_gan2d/ckpt_050000.pt --out export/3speaker \
        --model-name leapsinger_oniku --variant diffsinger --hop 512      # fp32 (default)
    #   add --fp16 for a ~32% smaller file (lossless, but not faster on CPU; helps GPU/mobile)

Variants (chosen to match the trained ckpt; the only weight-level split is use_uv):
    diffsinger (A): use_uv=False, n_styles=0. Inputs [tokens, durations, f0 (+spk_embed)].
                    --hop 512 = pseudo-512 (×2 upsample in / pairwise-average out) | 256 = native.
    full       (B): use_uv=True (and/or styles). Inputs [tokens, durations, f0, uv (+spk_embed)].
                    Native hop. A style-bearing model has one style baked in (--style-id, default 0).

Speaker: none (single-spk) | embed (spk_embed graph input the host feeds) | bake (freeze one
speaker into the graph; acoustic file becomes {model}.{spk}.onnx).

fp32 by default (simple `torch.onnx.export`, the public-release default — none of the fp16
machinery runs). Pass --fp16 for the ~32% smaller partial-fp16 build (see postprocess.py).

The last post-export pass folds the excitation's unrolled harmonic sum (n_harm straight-line
blocks) into one ONNX `Loop` (export/harmonic_loop.py): bit-identical output, ~5x lower peak
memory in every ORT, faster session creation. --no-harmonic-loop keeps the unrolled chain.

Output: {model}.onnx (fp32 unless --fp16), or {model}.{spk}.onnx when a speaker is baked;
a style-bearing model adds .style{N} (e.g. {model}.{spk}.style1.onnx).
"""
from __future__ import annotations

import argparse
import os

import torch

from infer import load_acoustic
from .wrappers import AcousticExportWrapperA, AcousticExportWrapperB
from . import spk_embed as spk
from . import postprocess as post


def _vocab_size(model):
    """Number of phoneme-embedding rows (attn: phoneme_encoder.embed; conv: phoneme_embed)."""
    enc = getattr(model, "phoneme_encoder", None)
    if enc is not None and getattr(enc, "embed", None) is not None:
        return enc.embed.num_embeddings
    return model.phoneme_embed.num_embeddings


def _validate(cfg, model, variant, speaker, style_id=0):
    """Hard checks: variant must match the ckpt's weight-level capabilities."""
    if float(getattr(model, "x0_sigma", 0.0)) > 0:
        raise SystemExit("This ckpt has x0_sigma > 0 (a reflow teacher or student), which cannot be exported yet: "
                         "the Gaussian noise added to the starting point is not in the exported graph.")
    use_uv = bool(cfg.get("use_uv", True))
    has_style = getattr(model, "style_emb", None) is not None
    if has_style and not 0 <= style_id < model.style_emb.num_embeddings:
        raise SystemExit(f"--style-id {style_id} is out of range; this ckpt has "
                         f"{model.style_emb.num_embeddings} styles.")
    if variant == "diffsinger":
        if use_uv:
            raise SystemExit("variant=diffsinger needs a uv-free ckpt (use_uv=False); this ckpt "
                             "has use_uv=True — export it as --variant full, or train uv-free.")
        if has_style:
            raise SystemExit("variant=diffsinger needs n_styles=0; this ckpt has a style bank.")
    elif variant == "full":
        if not use_uv and not has_style:
            print("[export] note: --variant full on a uv-free, style-less ckpt is just the "
                  "native-hop A graph with a v/uv input that is ignored.")
    if speaker in ("embed", "bake") and not spk.has_speakers(model):
        raise SystemExit(f"speaker={speaker} needs a multi-speaker ckpt (spk_bank); this ckpt is "
                         f"single-speaker. Use --speaker none.")
    if speaker == "none" and spk.has_speakers(model):
        print("[export] note: multi-speaker ckpt exported with --speaker none "
              "(speaker 0's bank vector is unused; cond gets no speaker term).")


def _build_wrapper(model, variant, hop, num_steps, speaker, frozen_spk, style_id=0):
    if variant == "diffsinger":
        return AcousticExportWrapperA(model, num_steps=num_steps, hop=hop,
                                      speaker=speaker, frozen_spk=frozen_spk)
    return AcousticExportWrapperB(model, num_steps=num_steps, speaker=speaker,
                                  frozen_spk=frozen_spk, style_id=style_id)


def _example_and_axes(variant, speaker, hidden, vocab_size):
    """Return (input_names, example_tensors, dynamic_axes) for torch.onnx.export.
    Trace tokens are bounded by vocab_size (a ckpt may have n_phonemes < the current 50)."""
    Np, T = 8, 40
    tokens = torch.randint(1, max(2, vocab_size), (1, Np), dtype=torch.long)
    durs = torch.full((1, Np), T // Np, dtype=torch.long)
    T = int(durs.sum())
    f0 = torch.linspace(200, 300, T)[None]
    names = ["tokens", "durations", "f0"]
    args = [tokens, durs, f0]
    axes = {"tokens": {1: "n_tokens"}, "durations": {1: "n_tokens"}, "f0": {1: "n_frames"},
            "mel": {2: "n_frames"}}
    if variant == "full":
        names.append("uv"); args.append(torch.ones(1, T)); axes["uv"] = {1: "n_frames"}
    if speaker == "embed":
        names.append("spk_embed"); args.append(torch.zeros(1, hidden))
    return names, tuple(args), axes


def main():
    ap = argparse.ArgumentParser(description="Export a LeapSinger acoustic ckpt to ONNX.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--variant", choices=["diffsinger", "full"], default="diffsinger")
    ap.add_argument("--hop", type=int, choices=[256, 512], default=512,
                    help="diffsinger only: 512=pseudo-512 | 256=native")
    ap.add_argument("--speaker", choices=["none", "embed", "bake"], default="none")
    ap.add_argument("--spk-name", default=None, help="bake: speaker name; embed: unused")
    ap.add_argument("--spk-id", type=int, default=0, help="bake: which speaker id to freeze")
    ap.add_argument("--style-id", type=int, default=0,
                    help="full variant, style-bearing ckpt: which style to bake in (default 0)")
    ap.add_argument("--num-steps", type=int, default=None, help="flow Euler steps (default ckpt infer_steps)")
    ap.add_argument("--opset", type=int, default=17)
    # fp32 is the public-release default (simple export, faster on CPU, lossless). --fp16 opts into
    # the ~32%-smaller partial-fp16 build (helps GPU/mobile size & speed, no CPU speedup).
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=False)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--simplify", dest="simplify", action="store_true", default=True)
    ap.add_argument("--no-simplify", dest="simplify", action="store_false")
    # Replace the ~17 MB DFT-as-matmul excitation with a native ONNX `DFT` node (lossless, -17 MB,
    # a little faster single-threaded). Default on: the forward `DFT` op is supported by the
    # onnxruntime CPU EP (>= ~1.14). Use --no-native-dft for the matmul build, which runs on any
    # ORT (e.g. a very old runtime without the DFT kernel).
    ap.add_argument("--native-dft", dest="native_dft", action="store_true", default=True)
    ap.add_argument("--no-native-dft", dest="native_dft", action="store_false")
    # Fold the excitation's unrolled harmonic sum into one ONNX `Loop` (export/harmonic_loop.py).
    # Default on: ORT's default execution order runs the unrolled chain breadth-first and keeps
    # ~n_harm sample-rate tensors alive at once (0.28 MB/frame, 1.5 GB for a 30 s phrase); the Loop
    # is bit-identical and ~5x lower in peak memory. `Loop` exists since opset 1 and runs in every
    # ORT. --no-harmonic-loop keeps the unrolled chain (e.g. to bisect a runtime problem).
    ap.add_argument("--harmonic-loop", dest="harmonic_loop", action="store_true", default=True)
    ap.add_argument("--no-harmonic-loop", dest="harmonic_loop", action="store_false")
    ap.add_argument("--deterministic", action="store_true",
                    help="zero the excitation noise (reproducible; for verification, not deployment)")
    ap.add_argument("--verify", action="store_true", help="run ORT parity vs PyTorch after export")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, cfg = load_acoustic(args.ckpt, device="cpu")
    num_steps = args.num_steps or int(cfg.get("infer_steps", 10))
    _validate(cfg, model, args.variant, args.speaker, args.style_id)

    frozen_spk = None
    acoustic_file = f"{args.model_name}.onnx"
    if args.speaker == "bake":
        frozen_spk = spk.speaker_vector(model, args.spk_id)
        spk_name = args.spk_name or f"spk{args.spk_id}"
        acoustic_file = f"{args.model_name}.{spk_name}.onnx"
    if args.variant == "full" and getattr(model, "style_emb", None) is not None:
        acoustic_file = acoustic_file.replace(".onnx", f".style{args.style_id}.onnx")

    wrapper = _build_wrapper(model, args.variant, args.hop, num_steps,
                             args.speaker, frozen_spk, args.style_id).eval()
    wrapper.exc.deterministic = bool(args.deterministic)

    input_names, example, axes = _example_and_axes(
        args.variant, args.speaker, model.hidden, _vocab_size(model))

    fp32_path = os.path.join(args.out, acoustic_file.replace(".onnx", ".fp32.onnx"))
    final_path = os.path.join(args.out, acoustic_file)
    torch.onnx.export(wrapper, example, fp32_path, opset_version=args.opset,
                      input_names=input_names, output_names=["mel"],
                      dynamic_axes=axes, do_constant_folding=True, dynamo=False)
    print(f"[export] fp32 -> {fp32_path} ({os.path.getsize(fp32_path)/1e6:.1f} MB)")

    # --verify also compares the Loop graph with the unrolled one it came from, so keep that copy
    unrolled_path = final_path + ".unrolled.onnx" if (args.verify and args.harmonic_loop) else None
    post.finalize(fp32_path, final_path, fp16=args.fp16, do_simplify=args.simplify,
                  native_dft=args.native_dft, harmonic_loop=args.harmonic_loop,
                  unrolled_out=unrolled_path)
    print(f"[export] final -> {final_path} ({os.path.getsize(final_path)/1e6:.1f} MB) "
          f"fp16={args.fp16} simplify={args.simplify} harmonic_loop={args.harmonic_loop}")
    if final_path != fp32_path and os.path.exists(fp32_path):
        os.remove(fp32_path)

    if args.verify:
        from .verify import verify_full_graph, verify_harmonic_loop
        verify_full_graph(wrapper, final_path, input_names, fp16=args.fp16)
        if unrolled_path:
            verify_harmonic_loop(unrolled_path, final_path, input_names,
                                 vocab=_vocab_size(model), hidden=model.hidden)
            os.remove(unrolled_path)                      # kept only when the check failed


if __name__ == "__main__":
    main()
