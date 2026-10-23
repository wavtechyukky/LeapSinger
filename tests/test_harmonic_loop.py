"""export/harmonic_loop.py: 展開された倍音の足し合わせを ONNX の Loop に畳む後処理。本物の書き出し（torch + ckpt）は
重いので、書き出し器が作るのと同じ形の小さな図（zeros_like → 段ごとに Mul・Less・Cast・Sin・Mul・Mul・Add）を
onnx.helper で組み、畳んだ後の出力が onnxruntime でビット単位で同じこと、早期終了の条件、定数の整理、
見つからないときの失敗を確かめる。"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from export import harmonic_loop as hl  # noqa: E402

SR = 44100.0


def _f32(v, name):
    return numpy_helper.from_array(np.asarray(v, np.float32), name)


def _unrolled(n_harm, decay=1.5, constants="initializer"):
    """書き出し器の形: acc = zeros_like(phase); 各 k で acc = acc + a_k·sin(k·phase)·cast(k·f0 < nyq)。
    k = 1 の段は k·phase・k·f0・1·sin の Mul が無い（書き出し時に消える）。k・a_k は initializer か Constant ノード
    （onnxsim を掛けない書き出し）。出力は 和 / rms と、nyq を共有する別の値（共有する定数が残るかを見る）。"""
    inits = [_f32(SR / 2, "nyq"), _f32(0.7, "rms")]
    nodes = [helper.make_node("Shape", ["phase"], ["shape"]),
             helper.make_node("ConstantOfShape", ["shape"], ["acc0"], value=_f32([0.0], "zero"))]

    def const(v, name):
        if constants == "initializer":
            inits.append(_f32(v, name))
        else:
            nodes.append(helper.make_node("Constant", [], [name], value=_f32(v, name + "_v")))
        return name

    acc = "acc0"
    for i in range(n_harm):
        k = i + 1
        if k == 1:
            nodes += [helper.make_node("Less", ["f0", "nyq"], ["lt1"]),
                      helper.make_node("Cast", ["lt1"], ["aa1"], to=TensorProto.FLOAT),
                      helper.make_node("Sin", ["phase"], ["sin1"]),
                      helper.make_node("Mul", ["sin1", "aa1"], ["term1"])]
        else:
            kn, an = const(k, f"k{k}"), const(1.0 / k ** decay, f"a{k}")
            nodes += [helper.make_node("Mul", ["f0", kn], [f"kf{k}"]),
                      helper.make_node("Less", [f"kf{k}", "nyq"], [f"lt{k}"]),
                      helper.make_node("Cast", [f"lt{k}"], [f"aa{k}"], to=TensorProto.FLOAT),
                      helper.make_node("Mul", [kn, "phase"], [f"kp{k}"]),          # 定数が先でも拾える
                      helper.make_node("Sin", [f"kp{k}"], [f"sin{k}"]),
                      helper.make_node("Mul", [f"sin{k}", an], [f"sa{k}"]),
                      helper.make_node("Mul", [f"sa{k}", f"aa{k}"], [f"term{k}"])]
        nodes.append(helper.make_node("Add", [acc, f"term{k}"], [f"acc{k}"]))
        acc = f"acc{k}"
    nodes += [helper.make_node("Div", [acc, "rms"], ["out"]),
              helper.make_node("Mul", ["f0", "nyq"], ["other"])]
    vi = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, "n"])  # noqa: E731
    g = helper.make_graph(nodes, "unrolled", [vi("phase"), vi("f0")], [vi("out"), vi("other")], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def _run(model, feeds):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def _feeds(n, lo, hi):
    f0 = np.linspace(lo, hi, n, dtype=np.float32)[None]
    phase = np.cumsum(2 * np.pi * f0.astype(np.float64) / SR, axis=1).astype(np.float32)
    return {"phase": phase, "f0": f0}


def _body(model):
    loop = next(n for n in model.graph.node if n.op_type == "Loop")
    return next(a.g for a in loop.attribute if a.name == "body")


@pytest.mark.parametrize("constants", ["initializer", "constant_node"])
def test_loop_matches_unrolled_bit_for_bit(constants):
    m = _unrolled(12, constants=constants)
    feeds = _feeds(3000, 100, 4000)                       # 低い F0 では 12 本すべてが通る
    ref = _run(m, feeds)
    info = hl.convert_model(m)
    got = _run(m, feeds)
    assert info["n_harm"] == 12 and info["nodes_after"] < info["nodes_before"] and info["output"] == "acc12"
    loops = [n for n in m.graph.node if n.op_type == "Loop"]
    assert len(loops) == 1 and loops[0].name == hl.LOOP_NODE_NAME and list(loops[0].output) == ["acc12"]
    assert not any(n.op_type in ("Sin", "Constant") for n in m.graph.node)       # 段はすべて Loop の中へ
    assert np.abs(ref[0]).max() > 0.1
    for a, b in zip(ref, got):
        assert a.shape == b.shape and a.tobytes() == b.tobytes()


def test_loop_stops_once_every_harmonic_is_above_nyquist():
    """cond_out = この倍音がどこかのサンプルでナイキスト未満か。f0 が 3000〜4000 Hz なら k = 7（21000 Hz）までは
    続き、k = 8（24000 Hz 以上）で止まる。止めても以降の項は厳密に 0 なので和は変わらない。"""
    m = _unrolled(12)
    feeds = _feeds(500, 3000, 4000)
    ref = _run(m, feeds)
    hl.convert_model(m)
    assert ref[0].tobytes() == _run(m, feeds)[0].tobytes()
    body = onnx.GraphProto()
    body.CopyFrom(_body(m))
    vi = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, "n"])  # noqa: E731
    body.input.extend([vi("phase"), vi("f0")])              # 本体が外側から参照する値を入力にして単体で走らせる
    bm = helper.make_model(body, opset_imports=[helper.make_opsetid("", 17)])
    bm.ir_version = 8
    sess = ort.InferenceSession(bm.SerializeToString(), providers=["CPUExecutionProvider"])

    def cond(i):
        out = sess.run(None, {hl._P + "iter": np.asarray(i, np.int64), hl._P + "cond_in": np.asarray(True),
                              hl._P + "acc_in": np.zeros((1, 500), np.float32), **feeds})
        return bool(out[0])

    assert cond(0) and cond(6) and not cond(7)


def test_constants_move_into_the_loop_and_shared_ones_stay():
    m = _unrolled(6)
    hl.convert_model(m)
    names = {t.name for t in m.graph.initializer}
    assert {"nyq", "rms", hl._P + "trip", hl._P + "true"} <= names
    assert not any(n[0] in "ka" and n[1:].isdigit() for n in names)
    body = _body(m)
    tables = {t.name: numpy_helper.to_array(t) for t in body.initializer}
    assert tables[hl._P + "k_table"].tolist() == [1, 2, 3, 4, 5, 6]
    assert np.array_equal(tables[hl._P + "amp_table"], (1.0 / np.arange(1, 7) ** 1.5).astype(np.float32))
    assert float(tables[hl._P + "nyq"]) == SR / 2 and tables[hl._P + "k_table"].dtype == np.float32


def test_missing_or_broken_pattern_is_a_clear_error():
    vi = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, "n"])  # noqa: E731
    g = helper.make_graph([helper.make_node("Add", ["x", "y"], ["z"])], "plain", [vi("x"), vi("y")], [vi("z")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    with pytest.raises(hl.HarmonicSumNotFound, match="見つからない"):
        hl.convert_model(m)
    m = _unrolled(6)
    hl.convert_model(m)
    with pytest.raises(hl.HarmonicSumNotFound):              # 二度目は畳むものが無い
        hl.convert_model(m)
    m = _unrolled(6)
    m.graph.output.append(vi("acc3"))                        # 途中の和を外から使うと Loop に畳めない
    with pytest.raises(hl.HarmonicSumNotFound, match="acc3"):
        hl.convert_model(m)


def test_file_round_trip(tmp_path):
    src, dst = str(tmp_path / "unrolled.onnx"), str(tmp_path / "loop.onnx")
    onnx.save(_unrolled(8), src)
    info = hl.to_harmonic_loop(src, dst)
    assert info["n_harm"] == 8 and info["bytes_before"] > 0 and info["bytes_after"] > 0
    feeds = _feeds(1000, 200, 300)
    ref, got = _run(onnx.load(src), feeds), _run(onnx.load(dst), feeds)
    assert ref[0].tobytes() == got[0].tobytes()
