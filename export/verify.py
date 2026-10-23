"""Parity checks for the ONNX export.

  verify_excitation(): HarmonicNoiseExcitationONNX(deterministic) vs the training
      harmonic_noise_mel_torch(noise_ratio=0) on a real F0 curve — the excitation is the flow's
      x0, so it must reproduce training numerics (not "improve" on them).
  verify_full_graph(): PyTorch wrapper vs the exported ONNX in ORT, on two different lengths, to
      confirm export fidelity AND that the token/frame axes are dynamic.
  verify_harmonic_loop(): the final graph (harmonic sum folded into one ONNX `Loop`, see
      export/harmonic_loop.py) vs the unrolled graph it was folded from, bit for bit, on the same
      lengths; runs from the ONNX files alone (no PyTorch wrapper needed).

Deterministic (noise=0) is used for parity; the deployment graph draws noise internally and is
only smoke-checked (shape/finite) since it is not reproducible across runtimes.
"""
from __future__ import annotations

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper


def verify_excitation(f0_hz=None, n_harm=50, hop=256, mel_bins=128) -> dict:
    """MAE between the ONNX-friendly excitation and the training excitation (both noise-free)."""
    from leapsinger.modules.harmonic_excitation import harmonic_noise_mel_torch
    from .excitation_onnx import HarmonicNoiseExcitationONNX
    if f0_hz is None:
        t = np.linspace(0, 3, 400)
        f0_hz = (250 + 60 * np.sin(2 * np.pi * 0.8 * t)).astype(np.float32)
    f0l = torch.log2(torch.clamp(torch.as_tensor(f0_hz, dtype=torch.float32), min=1.0))[None]
    T = f0l.shape[1]
    uv = torch.ones(1, T)
    orig = harmonic_noise_mel_torch(f0l, uv, noise_ratio=0.0, n_harm=n_harm, hop=hop,
                                    scale=0.15, harm_decay=1.0)
    exc = HarmonicNoiseExcitationONNX(n_harm=n_harm, exc_hop=hop, n_mels=mel_bins,
                                      deterministic=True).eval()
    with torch.no_grad():
        new = exc(f0l, uv)
    Tc = min(orig.shape[2], new.shape[2])
    d = (orig[:, :, :Tc] - new[:, :, :Tc]).abs()
    return {"mae": float(d.mean()), "max": float(d.max()), "T": int(Tc)}


def _synth_item(Np, seed, vocab=50):
    g = np.random.default_rng(seed)
    tokens = g.integers(1, max(2, vocab), size=Np).astype(np.int64)
    durs = g.integers(4, 14, size=Np).astype(np.int64)
    T = int(durs.sum())
    f0 = (250 + 60 * np.sin(np.linspace(0, 6, T)) + g.normal(0, 3, T)).clip(120, 600).astype(np.float32)
    return tokens, durs, f0


def verify_full_graph(wrapper, onnx_path: str, input_names, *, fp16: bool = False,
                      lengths=((1, 24), (7, 40))) -> list:
    """Run the PyTorch wrapper vs ORT on several lengths. Requires wrapper.exc.deterministic=True
    for a meaningful MAE; otherwise only shape/finite is trustworthy. Returns per-length dicts."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    det = bool(getattr(wrapper.exc, "deterministic", False))
    variant_full = "uv" in input_names
    speaker_embed = "spk_embed" in input_names
    enc = getattr(wrapper.model, "phoneme_encoder", None)
    vocab = enc.embed.num_embeddings if enc is not None and getattr(enc, "embed", None) is not None \
        else wrapper.model.phoneme_embed.num_embeddings
    out = []
    for seed, Np in lengths:
        tokens, durs, f0 = _synth_item(Np, seed, vocab)
        T = int(durs.sum())
        feeds = {"tokens": tokens[None], "durations": durs[None], "f0": f0[None]}
        targs = [torch.as_tensor(tokens)[None], torch.as_tensor(durs)[None], torch.as_tensor(f0)[None]]
        if variant_full:
            uv = np.ones((1, T), np.float32); feeds["uv"] = uv; targs.append(torch.as_tensor(uv))
        if speaker_embed:
            se = np.zeros((1, wrapper.hidden), np.float32); feeds["spk_embed"] = se
            targs.append(torch.as_tensor(se))
        with torch.no_grad():
            tmel = wrapper(*targs)[0].numpy()
        omel = sess.run(None, feeds)[0][0]
        # tmel and omel share the wrapper's layout (WrapperA emits [1,T,mel], WrapperB [1,mel,T]),
        # so compare whole arrays; equal shapes also confirm the ORT frame axis stayed dynamic.
        same_shape = tuple(tmel.shape) == tuple(omel.shape)
        mae = float(np.abs(tmel - omel).mean()) if (det and same_shape) else None
        rec = {"Np": Np, "T": T, "torch_shape": tmel.shape, "ort_shape": omel.shape,
               "mae": mae, "finite": bool(np.isfinite(omel).all()),
               "T_dynamic_ok": same_shape}
        out.append(rec)
        tag = "fp16" if fp16 else "fp32"
        maes = f"MAE={mae:.4f}" if mae is not None else "MAE=n/a(stochastic)"
        print(f"[verify {tag}] Np={Np:2d} T={T:4d} torch{tmel.shape} ort{omel.shape} "
              f"{maes} finite={rec['finite']} dyn_ok={rec['T_dynamic_ok']}")
    return out


def _graph_hints(model):
    """(input names, vocab, hidden) read off an exported graph, for checks that run without the
    PyTorch wrapper: vocab = rows of the phoneme-embedding table that `tokens` gathers from,
    hidden = the static dim of `spk_embed`. Defaults 50 / 256 when the graph has neither."""
    g = model.graph
    dims = {t.name: list(t.dims) for t in g.initializer}
    vocab = next((dims[n.input[0]][0] for n in g.node
                  if n.op_type == "Gather" and len(n.input) == 2 and n.input[1] == "tokens"
                  and n.input[0] in dims), 50)
    hidden = 256
    for i in g.input:
        if i.name == "spk_embed" and len(i.type.tensor_type.shape.dim) == 2:
            hidden = i.type.tensor_type.shape.dim[1].dim_value or hidden
    return [i.name for i in g.input], vocab, hidden


def _ort_feeds(input_names, seed, Np, *, vocab, hidden):
    """The synthetic item of verify_full_graph as ORT feeds. A graph that takes the excitation noise
    as an input (`exc_noise`, host-seeded builds) gets 512 samples per frame: enough for any hop,
    and the graph slices the n it needs."""
    tokens, durs, f0 = _synth_item(Np, seed, vocab)
    T = int(durs.sum())
    feeds = {"tokens": tokens[None], "durations": durs[None], "f0": f0[None]}
    if "uv" in input_names:
        feeds["uv"] = np.ones((1, T), np.float32)
    if "spk_embed" in input_names:
        feeds["spk_embed"] = np.zeros((1, hidden), np.float32)
    if "exc_noise" in input_names:
        feeds["exc_noise"] = np.random.default_rng(seed).standard_normal((1, 512 * T)).astype(np.float32)
    return feeds, T


def verify_harmonic_loop(unrolled_path: str, loop_path: str, input_names=None, *, vocab=None,
                         hidden=None, lengths=((1, 24), (7, 40))) -> list:
    """Bit-for-bit parity of the harmonic-Loop graph (export/harmonic_loop.py) against the unrolled
    form it was folded from, on the same synthetic lengths as verify_full_graph. The harmonic sum
    (the Loop's output, computed before any noise is mixed in) is read out of both graphs as an
    extra output and must agree in every bit. The mel is compared too: it must agree when the graph
    draws no noise of its own (--deterministic, or an `exc_noise` input); a graph with Random* nodes
    draws fresh noise per run, so its mel is reported as n/a. A mismatch raises SystemExit: the
    folded graph must not ship. input_names / vocab / hidden default to what the graph says."""
    import onnxruntime as ort
    from .harmonic_loop import LOOP_NODE_NAME
    um, lm = onnx.load(unrolled_path), onnx.load(loop_path)
    loop = next((n for n in lm.graph.node if n.op_type == "Loop" and n.name == LOOP_NODE_NAME), None)
    if loop is None:
        raise SystemExit(f"[verify loop] {loop_path} has no {LOOP_NODE_NAME} node")
    harm = loop.output[0]
    if harm not in {o for n in um.graph.node for o in n.output}:
        raise SystemExit(f"[verify loop] {unrolled_path} does not produce {harm}: not the unrolled form of {loop_path}")
    mel_name = um.graph.output[0].name
    names, g_vocab, g_hidden = _graph_hints(lm)
    input_names = list(input_names) if input_names is not None else names
    vocab, hidden = vocab or g_vocab, hidden or g_hidden
    stochastic = any(n.op_type.startswith("Random") for m in (um, lm) for n in m.graph.node)
    sessions = []
    for m in (um, lm):
        m.graph.output.append(helper.make_tensor_value_info(harm, TensorProto.FLOAT, None))
        sessions.append(ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]))

    def same_bits(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

    out = []
    for seed, Np in lengths:
        feeds, T = _ort_feeds(input_names, seed, Np, vocab=vocab, hidden=hidden)
        (mel_u, harm_u), (mel_l, harm_l) = (s.run([mel_name, harm], feeds) for s in sessions)
        same_harm, same_mel = same_bits(harm_u, harm_l), same_bits(mel_u, mel_l)
        max_diff = float(np.abs(harm_u.astype(np.float64) - harm_l).max()) if harm_u.shape == harm_l.shape else None
        rec = {"Np": Np, "T": T, "harm_identical": same_harm, "harm_max_diff": max_diff,
               "mel_identical": None if stochastic else same_mel}
        out.append(rec)
        mel_s = "mel=n/a(stochastic)" if stochastic else f"mel identical={same_mel}"
        print(f"[verify loop] Np={Np:2d} T={T:4d} harm identical={same_harm} (max|diff| {max_diff}) {mel_s}")
        if not same_harm or (not stochastic and not same_mel):
            raise SystemExit("[verify loop] the harmonic-Loop graph differs from its unrolled form; do not ship it "
                             "(export with --no-harmonic-loop and report the case)")
    return out


if __name__ == "__main__":
    r = verify_excitation()
    print(f"[verify] excitation  MAE={r['mae']:.6f} MAX={r['max']:.4f} T={r['T']}")
