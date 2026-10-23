"""leapsinger/modules/position.py — relative-position conditions (model.pos_cond / model.dur_cond)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from leapsinger.modules import position as P  # noqa: E402


def test_frame_to_phoneme_matches_onnx_version():
    g = torch.Generator().manual_seed(0)
    dur = torch.randint(1, 9, (3, 12), generator=g)
    dur[1, 9:] = 0                                      # padded phonemes
    dur[2, 5:] = 0
    T = int(dur.sum(1).max())
    a = P.frame_to_phoneme(dur, T)
    b = P.frame_to_phoneme_onnx(dur, T)
    assert torch.equal(a, b)
    # frame f belongs to the phoneme whose [start, end) contains it
    ends = dur.cumsum(1)
    for bi in range(3):
        for f in range(int(dur[bi].sum())):
            k = int(a[bi, f])
            assert int(ends[bi, k] - dur[bi, k]) <= f < int(ends[bi, k])


def test_phoneme_relative_position():
    dur = torch.tensor([[3, 1, 4]])
    ph = P.frame_to_phoneme(dur, 8)
    starts = dur.cumsum(1) - dur
    i = torch.arange(8)[None] - torch.gather(starts, 1, ph)
    p = P.relative_position(i, torch.gather(dur, 1, ph))[0]
    exp = torch.tensor([0, 0.5, 1.0, 0.5, 0, 1 / 3, 2 / 3, 1.0])
    assert torch.allclose(p, exp, atol=1e-6)


def test_unit_bounds_consonants_belong_to_previous_unit():
    # ids: pau k a n o pau    nucleus = {a:2, o:4}, pau = 0
    ids = torch.tensor([[0, 1, 2, 3, 4, 0]])
    dur = torch.tensor([[5, 2, 6, 1, 4, 3]])
    is_nuc = torch.zeros(5, dtype=torch.bool); is_nuc[2] = True; is_nuc[4] = True
    is_b = is_nuc[ids] | (ids == 0)
    us, ul = P.unit_bounds(dur, is_b)
    # units: [pau k] = frames 0..6, [a n] = 7..13, [o] = 14..17, [pau] = 18..20
    assert us[0].tolist() == [0, 0, 7, 7, 14, 18]
    assert ul[0].tolist() == [7, 7, 7, 7, 4, 3]


def test_raised_cosine_is_circular():
    e0 = P.raised_cosine(torch.tensor([0.0]), 8)
    e1 = P.raised_cosine(torch.tensor([1.0]), 8)
    assert torch.allclose(e0, e1, atol=1e-6)
    assert e0.shape == (1, 8)
    assert torch.allclose(e0[0, 0], torch.tensor(1.0))        # bin 0 peaks at p = 0


def test_position_condition_forward_and_export_path():
    torch.manual_seed(0)
    pc = P.PositionCondition(n_phonemes=6, hidden=16, nucleus_ids=[2, 4], pau_id=0,
                             phoneme=True, unit=True, bins=8, dur=True)
    ids = torch.tensor([[0, 1, 2, 3, 4, 0], [0, 2, 1, 0, 0, 0]])
    dur = torch.tensor([[5, 2, 6, 1, 4, 3], [2, 7, 3, 2, 0, 0]])
    T = int(dur.sum(1).max()) + 2                             # +2 padding frames
    out = pc(ids, dur, T)
    assert out.shape == (2, T, 16)
    assert torch.all(out[1, 14:] == 0)                        # item 1 has 14 frames -> padding is zero
    pc.frame_to_phoneme = P.frame_to_phoneme_onnx
    out2 = pc(ids, dur, T)
    assert torch.allclose(out, out2, atol=1e-6)


def test_phoneme_only_position_condition_legacy_code():
    """旧符号化（raised_cosine）で音素内の位置だけ。入力は bins 本、核音素の表は使わない。"""
    torch.manual_seed(0)
    pc = P.PositionCondition(n_phonemes=6, hidden=16, nucleus_ids=[2, 4], pau_id=0,
                             phoneme=True, unit=False, bins=8, dur=False)
    assert pc.encoding == "raised_cosine"                     # encoding を渡さない = 古い ckpt と同じ
    assert pc.proj.in_features == 8 and pc.dur_emb is None and pc.stretch_embed is None
    ids = torch.tensor([[0, 1, 2, 3, 4, 0], [0, 2, 1, 0, 0, 0]])
    dur = torch.tensor([[5, 2, 6, 1, 4, 3], [2, 7, 3, 2, 0, 0]])
    T = int(dur.sum(1).max()) + 2
    out = pc(ids, dur, T)
    assert out.shape == (2, T, 16) and torch.all(out[1, 14:] == 0)
    pc.is_nucleus.zero_()                                     # 単位内の位置を使わないので何も変わらない
    assert torch.equal(out, pc(ids, dur, T))
    pc.frame_to_phoneme = P.frame_to_phoneme_onnx
    assert torch.allclose(out, pc(ids, dur, T), atol=1e-6)


def test_stretch_index_matches_diffsinger():
    # DiffSinger の StretchRegulator の docstring の例: dur = [2, 4, 3] → i/n = [0, .5, 0, .25, .5, .75, 0, 1/3, 2/3]
    dur = torch.tensor([[2, 4, 3]])
    ph = P.frame_to_phoneme(dur, 9)
    starts = dur.cumsum(1) - dur
    i = torch.arange(9)[None] - torch.gather(starts, 1, ph)
    s = P.stretch_index(i, torch.gather(dur, 1, ph))
    assert s[0].tolist() == [0, 500, 0, 250, 500, 750, 0, 333, 667]


def test_sinusoidal_embedding_shape_and_zero():
    e = P.SinusoidalEmbedding(16)(torch.tensor([0, 1, 500]))
    assert e.shape == (3, 16)
    assert torch.allclose(e[0], torch.cat([torch.zeros(8), torch.ones(8)]))   # x = 0: sin 0 / cos 1
    assert not torch.allclose(e[1], e[2])


def _sinusoidal_pc(hidden=16):
    torch.manual_seed(0)
    return P.PositionCondition(n_phonemes=6, hidden=hidden, nucleus_ids=[2, 4], pau_id=0,
                               phoneme=True, unit=False, dur=False, encoding="sinusoidal")


def test_sinusoidal_position_condition():
    """configs の既定（2026-10-17〜）: DiffSinger の stretch 埋め込み。表引き = 各フレームで MLP を直接通した値。"""
    pc = _sinusoidal_pc()
    assert pc.proj is None and pc.stretch_embed is not None
    ids = torch.tensor([[0, 1, 2, 3, 4, 0], [0, 2, 1, 0, 0, 0]])
    dur = torch.tensor([[5, 2, 6, 1, 4, 3], [2, 7, 3, 2, 0, 0]])
    T = int(dur.sum(1).max()) + 2
    out = pc(ids, dur, T)
    assert out.shape == (2, T, 16) and torch.all(out[1, 14:] == 0)
    ph = P.frame_to_phoneme(dur, T)
    starts = dur.cumsum(1) - dur
    s = P.stretch_index(torch.arange(T)[None] - torch.gather(starts, 1, ph), torch.gather(dur, 1, ph))
    direct = pc.stretch_embed(s.float())                            # 表を作らずに各フレームで MLP
    assert torch.allclose(out[0, :21], direct[0, :21], atol=1e-5)   # item 0 は 21 フレーム
    assert not torch.allclose(out[0, 0], out[0, 4])                 # 頭と終わりは別の値（循環しない）
    pc.is_nucleus.zero_()                                           # 単位内の位置は使わない
    assert torch.equal(out, pc(ids, dur, T))
    pc.frame_to_phoneme = P.frame_to_phoneme_onnx                   # export 用のフレーム→音素でも同じ
    assert torch.allclose(out, pc(ids, dur, T), atol=1e-6)
    pc.frame_to_phoneme = P.frame_to_phoneme
    pc(ids, dur, T).sum().backward()                                # 表引きでも MLP まで勾配が届く
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pc.stretch_embed.parameters())


def test_sinusoidal_rejects_unit_position():
    try:
        P.PositionCondition(n_phonemes=6, hidden=16, nucleus_ids=[2], phoneme=True, unit=True, encoding="sinusoidal")
    except ValueError:
        return
    raise AssertionError("unit + sinusoidal should be rejected")


def test_sinusoidal_position_condition_exports_to_onnx():
    """export/cli.py と同じ legacy exporter（opset 17, dynamo=False）で書き出し、onnxruntime と比べる。"""
    import tempfile
    try:
        import onnxruntime as ort
    except ImportError:
        return
    pc = _sinusoidal_pc().eval()
    pc.frame_to_phoneme = P.frame_to_phoneme_onnx

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids, dur):
            return self.m(ids, dur, dur.sum(dim=1).max())

    ids = torch.tensor([[0, 1, 2, 3, 4, 0]])
    dur = torch.tensor([[5, 2, 6, 1, 4, 3]])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "pos.onnx")
        torch.onnx.export(Wrap(pc), (ids, dur), path, opset_version=17, dynamo=False,
                          input_names=["ids", "dur"], output_names=["out"],
                          dynamic_axes={"ids": {1: "L"}, "dur": {1: "L"}, "out": {1: "T"}})
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        names = {x.name for x in sess.get_inputs()}                  # 音素内の位置だけなら ids は使わず消える
        assert "dur" in names
        ids2 = torch.tensor([[0, 2, 1, 3, 0]])                       # 長さの違う入力でも合う（dynamic）
        dur2 = torch.tensor([[3, 9, 2, 5, 4]])
        for a, b in ((ids, dur), (ids2, dur2)):
            feed = {k: v for k, v in (("ids", a.numpy()), ("dur", b.numpy())) if k in names}
            got = sess.run(None, feed)[0]
            ref = Wrap(pc)(a, b).detach().numpy()
            assert got.shape == ref.shape and abs(got - ref).max() < 1e-4
