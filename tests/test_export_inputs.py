"""export: the optional host inputs `exc_noise` (the excitation's white noise from the host) and
`x0_loggain` (per-frame ln(gain) on the pseudo-mel), in the wrappers and in an exported graph."""
import copy
import math
import os
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from export import cli  # noqa: E402
from export import postprocess as post  # noqa: E402
from export.excitation_onnx import HarmonicNoiseExcitationONNX  # noqa: E402
from export.verify import verify_full_graph, verify_harmonic_loop  # noqa: E402
from preprocess.vocab import Vocab  # noqa: E402
from train import build_model  # noqa: E402

ALL = ["tokens", "durations", "f0", "spk_embed", "exc_noise", "x0_loggain"]


def _tiny_v2():
    """The V2 recipe (uv-free, no styles, 3 speakers) at toy size."""
    cfg = copy.deepcopy(yaml.safe_load(open(os.path.join(ROOT, "configs", "3singer_v2.yaml"), encoding="utf-8")))
    cfg["model"].update(hidden=32, backbone_ch=32, n_cycles=1)
    cfg["excitation"]["n_harm"] = 8
    model, _ = build_model(cfg, Vocab.load())
    return model.eval()


def _item(T_tokens=6, seed=0, hop=256):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 20, (1, T_tokens), generator=g)
    durs = torch.randint(4, 9, (1, T_tokens), generator=g)
    T = int(durs.sum())
    f0 = (220 + 40 * torch.sin(torch.linspace(0, 4, T)))[None]
    t256 = 2 * T if hop == 512 else T
    noise = torch.randn(1, 256 * t256, generator=g)
    gain = (math.log(0.5) * 0.5 * (1 - torch.cos(torch.linspace(0, 4 * math.pi, T))))[None]
    return tokens, durs, f0, torch.zeros(1, 32), noise, gain


def test_excitation_noise_input_matches_the_deterministic_and_slices_a_longer_buffer():
    exc = HarmonicNoiseExcitationONNX(n_harm=8, exc_hop=256).eval()
    f0l = torch.log2(220 + 30 * torch.sin(torch.linspace(0, 3, 50)))[None]
    uv = torch.ones_like(f0l)
    exc.deterministic = True
    ref = exc(f0l, uv)
    exc.deterministic = False
    assert torch.equal(exc(f0l, uv, torch.zeros(1, 50 * 256)), ref)        # zeros = the noise-free mel
    z = torch.randn(1, 50 * 256 + 999)
    assert torch.equal(exc(f0l, uv, z), exc(f0l, uv, z[:, :50 * 256]))      # extra samples are ignored
    assert not torch.equal(exc(f0l, uv, z), ref)


@pytest.mark.parametrize("hop", [256, 512])
def test_wrapper_optional_inputs(hop):
    model = _tiny_v2()
    w = cli._build_wrapper(model, "diffsinger", hop, 1, "embed", None, input_names=ALL).eval()
    tokens, durs, f0, spk, noise, gain = _item(hop=hop)
    with torch.no_grad():
        a = w(tokens, durs, f0, spk, noise, gain)
        b = w(tokens, durs, f0, spk, noise, gain)
        plain = w(tokens, durs, f0, spk, noise, torch.zeros_like(gain))
        no_gain = cli.AcousticExportWrapperA.forward(w, tokens, durs, f0, spk, exc_noise=noise)
        other = w(tokens, durs, f0, spk, torch.randn_like(noise), gain)
    assert a.shape == (1, f0.shape[1], 128)
    assert torch.equal(a, b)                       # the same host noise gives the same mel
    assert torch.equal(plain, no_gain)             # ln(gain) = 0 is the graph without the input
    assert not torch.allclose(a, plain)            # the gain reaches the output
    assert not torch.allclose(a, other)            # and so does the noise
    with pytest.raises(TypeError):
        w(tokens, durs, f0)


def test_exported_graph_with_both_inputs(tmp_path):
    pytest.importorskip("onnxruntime")
    model = _tiny_v2()
    names, example, axes = cli._example_and_axes("diffsinger", "embed", model.hidden, cli._vocab_size(model),
                                                 hop=256, exc_noise=True, x0_gain=True)
    assert names == ALL
    w = cli._build_wrapper(model, "diffsinger", 256, 1, "embed", None, input_names=names).eval()
    fp32, final, unrolled = (str(tmp_path / n) for n in ("m.fp32.onnx", "m.onnx", "m.unrolled.onnx"))
    with torch.no_grad():
        torch.onnx.export(w, example, fp32, opset_version=17, input_names=names, output_names=["mel"],
                          dynamic_axes=axes, do_constant_folding=True, dynamo=False)
    post.finalize(fp32, final, fp16=False, do_simplify=True, native_dft=True, harmonic_loop=True,
                  unrolled_out=unrolled)
    import onnx
    assert [i.name for i in onnx.load(final).graph.input] == names
    for rec in verify_full_graph(w, final, names):                      # same noise and gain on both sides
        assert rec["finite"] and rec["T_dynamic_ok"] and rec["mae"] < 1e-3, rec
    for rec in verify_harmonic_loop(unrolled, final, names):
        assert rec["harm_identical"] and rec["mel_identical"], rec
