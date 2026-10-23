"""書き出した ONNX の「倍音の足し合わせ」を、展開された n_harm 段の直線の列から 1 つの ONNX `Loop` に畳む後処理。

なぜ: `excitation_onnx._harm_sum` は倍音 k = 1..n_harm ごとに「k·phase → Sin → ×a_k → ×[k·f0 < nyq] → acc に足す」を
Python の for で書いてあり、トレースでそれが n_harm 段（既定 256 段、約 2560 ノード）の直線の列に展開される。展開
そのものは [B, n_harm, n] を作らないための設計だが、onnxruntime の既定の実行順はこの列を「全部の Sin を先に作って
から足す」幅優先の順に並べるため、サンプル長（n = フレーム数 × hop）の配列が 256 本同時に生き、ピークメモリは
0.28 MB/フレーム（30 秒の句で約 1.5 GB）に達する。native の onnxruntime でも onnxruntime-web でも同じで、ブラウザ
には実行順を変える設定が無い。おまけに 2560 ノードの図はセッション作成も遅い。

`Loop` にすれば倍音を 1 本ずつ順に計算するので、どの実行順でも同時に生きる配列は数本で済む。実測（native の
onnxruntime CPU、256 倍音、Loop 化の前 → 後）: ピーク +385 → +80 MB（7.8 秒の句）、+1469 → +287 MB（30 秒の句）、
実行は 10〜30 % 速く、セッション作成は 10 倍速い。Loop の本体の式と演算の順番（Mul・Less・Cast・Sin・Mul・Mul・Add）
は展開形と同じなので、出力はビット単位で一致する（export/verify.py の verify_harmonic_loop で確かめる）。ついでに、
倍音の周波数 k·f0 が全サンプルでナイキスト周波数以上になったら止める。以降の項は厳密に 0 なので結果は変わらない
（F0 が低いほど多くの倍音が要る）。

互換性: `Loop` は opset 1 からある演算子で、onnxruntime は最初の公開版から CPU・WASM・GPU のどの実行プロバイダでも
走らせられる。OpenUTAU は Microsoft.ML.OnnxRuntime 1.23.0（Windows の DirectML 版）/ 1.24.4（Linux・macOS）を参照して
いる（2026-10-21 に OpenUtau.Core.csproj で確認）ので、そのまま動くと見込む。Loop の本体は外側の値（phase と f0）を
直接参照するが、これは ONNX の仕様で認められた書き方で、各実行プロバイダが解決する。

形の発見はノード名に頼らない（onnxsim や fp16 変換の後も名前は残るが、残らなくても見つかるように）。zeros_like(phase)
の ConstantOfShape から始まる Add の鎖で、各段が a_k·sin(k·phase)·[k·f0 < nyq] の形のものがちょうど 1 本あることを
確かめ、k が 1, 2, ... と並ぶこと、phase・f0・nyq が全段で同じことを検査する。合わなければ HarmonicSumNotFound で
理由を返す（書き出し器が変わって形が崩れたときに、黙って重い図を出さないため）。

使い方（書き出し済みの ONNX に単体で掛ける。export/cli.py からは既定で掛かり、--no-harmonic-loop で外せる）:
    python -m export.harmonic_loop <in.onnx> <out.onnx> [--check]
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

LOOP_NODE_NAME = "/exc/HarmonicLoop"
_P = "harm_loop_"            # Loop の本体で使う名前の接頭辞（外側の名前と衝突しないことを確かめてから使う）


class HarmonicSumNotFound(RuntimeError):
    """展開された倍音の足し合わせが見つからない、または形が想定と違う。"""


@dataclass
class _Term:
    """1 段ぶん a_k·sin(k·phase)·[k·f0 < nyq] を遡った結果。k・a は定数テンソルの値（k = 1 の段は Mul が無いので None）。"""
    k: np.ndarray | None
    a: np.ndarray | None
    phase: str
    f0: str
    nyq: np.ndarray
    cast_to: int
    nodes: list


@dataclass
class HarmonicSum:
    """図の中で見つけた展開形。ノードは graph.node の位置（int）で持つ。protobuf のラッパーは参照が切れると
    別のノードに id が使い回されるので、id() を後で照合してはいけない。"""
    acc0: str            # zeros_like(phase) = ConstantOfShape の出力。Loop の初期値になる
    adds: list           # Add の鎖の位置（倍音の順）
    remove: set          # 畳む対象のノードの位置
    ks: np.ndarray       # 各段の k（1, 2, ..., n_harm）
    amps: np.ndarray     # 各段の振幅 a_k（k = 1 の段は Mul が無いので 1）
    phase: str           # 位相 [B, n]
    f0: str              # サンプル単位の F0 [B, n]（Hz）
    nyq: np.ndarray      # ナイキスト周波数（スカラー定数）
    cast_to: int         # ゲートの Cast の型（= acc の型）
    final: str           # 最後の Add の出力名。下流はこの名前を使うので、Loop の出力にそのまま付ける

    @property
    def n_harm(self) -> int:
        return len(self.adds)


def _scalar_of(consts: dict, name: str) -> np.ndarray | None:
    """name が要素 1 個の定数（initializer か Constant ノード）ならその値（0 次元）、違えば None。"""
    t = consts.get(name)
    if t is None:
        return None
    v = numpy_helper.to_array(t)
    return v.reshape(()) if v.size == 1 else None


def _scalar_mul(node, consts: dict):
    """Mul(x, 定数スカラー) → (x, 定数)。どちらの入力も定数スカラーでなければ (None, None)。"""
    if node.op_type != "Mul" or len(node.input) != 2:
        return None, None
    for x, c in ((node.input[0], node.input[1]), (node.input[1], node.input[0])):
        v = _scalar_of(consts, c)
        if v is not None:
            return x, v
    return None, None


def _parse_term(name: str, by_out: dict, consts: dict) -> _Term:
    """項 name = Mul(a_k·sin(k·phase), Cast(Less(k·f0, nyq))) を遡る。k = 1 の段は k·phase、k·f0、1·sin の Mul が
    書き出し時に消えているので、その形も受け付ける。"""
    t = by_out.get(name)
    if t is None or t.op_type != "Mul" or len(t.input) != 2:
        raise HarmonicSumNotFound(f"項 {name} が Mul でない")
    x_name, aa_name = t.input
    if by_out.get(aa_name) is None or by_out[aa_name].op_type != "Cast":
        x_name, aa_name = aa_name, x_name
    cast = by_out.get(aa_name)
    if cast is None or cast.op_type != "Cast":
        raise HarmonicSumNotFound(f"項 {name} にゲート Cast(Less(k·f0, nyq)) が無い")
    cast_to = next((a.i for a in cast.attribute if a.name == "to"), None)
    less = by_out.get(cast.input[0])
    if less is None or less.op_type != "Less" or cast_to is None:
        raise HarmonicSumNotFound(f"項 {name} のゲートが Cast(Less) でない")
    nyq = _scalar_of(consts, less.input[1])
    if nyq is None:
        raise HarmonicSumNotFound(f"項 {name} のナイキスト周波数 {less.input[1]} が定数でない")
    nodes = [t, cast, less]
    # k·f0（k = 1 なら f0 そのもの）
    f0 = k = None
    kf = by_out.get(less.input[0])
    if kf is not None:
        f0, k = _scalar_mul(kf, consts)
        if f0 is not None:
            nodes.append(kf)
    if f0 is None:
        f0, k = less.input[0], None
    # a·sin(k·phase)（k = 1 なら a = 1 で Mul が無く、Sin(phase) そのもの）
    x = by_out.get(x_name)
    a = None
    if x is not None and x.op_type == "Mul":
        sin_name, a = _scalar_mul(x, consts)
        if sin_name is not None:
            nodes.append(x)
            x = by_out.get(sin_name)
    if x is None or x.op_type != "Sin":
        raise HarmonicSumNotFound(f"項 {name} に Sin が無い")
    nodes.append(x)
    phase = k2 = None
    kp = by_out.get(x.input[0])
    if kp is not None:
        phase, k2 = _scalar_mul(kp, consts)
        if phase is not None:
            nodes.append(kp)
    if phase is None:
        phase, k2 = x.input[0], None
    # f0 側と phase 側の k は同じでなければならない（片方だけ Mul が無いのは k = 1 の形ではない）
    if (k is None) != (k2 is None) or (k is not None and not np.array_equal(k, k2)):
        raise HarmonicSumNotFound(f"項 {name} の k が f0 側 {k} と phase 側 {k2} で合わない")
    return _Term(k=k, a=a, phase=phase, f0=f0, nyq=nyq, cast_to=cast_to, nodes=nodes)


def _walk_chain(cos, pos: dict, by_out: dict, consumers: dict, consts: dict, outputs: set) -> HarmonicSum:
    """ConstantOfShape cos から Add の鎖をたどり、倍音の項が続く限り集める。pos は id(ノード) → 位置、
    outputs は図の出力名（途中の値が図の出力なら畳めない）。"""
    acc = cos.output[0]
    adds, terms = [], []
    reason = ""
    while True:
        nxt = [n for n in consumers.get(acc, []) if n.op_type == "Add" and len(n.input) == 2]
        if len(nxt) != 1:
            reason = f"{acc} を使う Add が {len(nxt)} 個"
            break
        add = nxt[0]
        other = add.input[1] if add.input[0] == acc else add.input[0]
        try:
            term = _parse_term(other, by_out, consts)
        except HarmonicSumNotFound as e:
            reason = str(e)
            break
        adds.append(add)
        terms.append(term)
        acc = add.output[0]
    if len(adds) < 2:
        raise HarmonicSumNotFound(f"倍音の Add の鎖が {len(adds)} 段で終わる（{reason}）")
    # 途中の和は次の Add だけが使う（他からも使われていたら、Loop に畳むとその値が消える）
    for add in adds[:-1]:
        if len(consumers.get(add.output[0], [])) != 1 or add.output[0] in outputs:
            raise HarmonicSumNotFound(f"途中の和 {add.output[0]} が Add 以外からも使われている")
    ks = [1.0 if t.k is None else float(t.k) for t in terms]
    if ks != [float(i + 1) for i in range(len(terms))]:
        raise HarmonicSumNotFound(f"k が 1, 2, ... の順でない: {ks[:8]}...")
    first = terms[0]
    for t in terms[1:]:
        if t.phase != first.phase or t.f0 != first.f0:
            raise HarmonicSumNotFound(f"phase・f0 が段ごとに違う（{first.phase}/{first.f0} と {t.phase}/{t.f0}）")
        if not np.array_equal(t.nyq, first.nyq) or t.cast_to != first.cast_to:
            raise HarmonicSumNotFound("ナイキスト周波数かゲートの型が段ごとに違う")
    remove = {pos[id(n)] for n in adds}
    for t in terms:
        remove.update(pos[id(n)] for n in t.nodes)
    # 畳むノードの出力は、畳むノードだけが使う（最後の和は別）
    for t in terms:
        for n in t.nodes:
            for o in n.output:
                if o in outputs or any(pos[id(u)] not in remove for u in consumers.get(o, [])):
                    raise HarmonicSumNotFound(f"{o} が倍音の足し合わせの外からも使われている")
    dtype = next((t.k.dtype for t in terms if t.k is not None), None) or helper.tensor_dtype_to_np_dtype(first.cast_to)
    amps = [np.asarray(1.0, dtype) if t.a is None else t.a.astype(dtype, copy=False) for t in terms]
    return HarmonicSum(acc0=cos.output[0], adds=[pos[id(a)] for a in adds], remove=remove,
                       ks=np.asarray(ks, dtype), amps=np.asarray(amps, dtype),
                       phase=first.phase, f0=first.f0, nyq=first.nyq, cast_to=first.cast_to,
                       final=adds[-1].output[0])


def find_harmonic_sum(g: onnx.GraphProto) -> HarmonicSum:
    """展開形を見つける。ちょうど 1 本でなければ HarmonicSumNotFound（候補ごとの理由つき）。"""
    nodes = list(g.node)                     # ラッパーを生かしたまま持ち、id → 位置の対応をここで固める
    pos = {id(n): i for i, n in enumerate(nodes)}
    by_out = {o: n for n in nodes for o in n.output if o}
    consumers: dict[str, list] = {}
    for n in nodes:
        for i in n.input:
            if i:
                consumers.setdefault(i, []).append(n)
    consts = {t.name: t for t in g.initializer}
    for n in nodes:
        if n.op_type == "Constant" and n.output:
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = a.t
    outputs = {o.name for o in g.output}
    found, reasons = [], []
    for cos in nodes:
        if cos.op_type != "ConstantOfShape":
            continue
        try:
            found.append(_walk_chain(cos, pos, by_out, consumers, consts, outputs))
        except HarmonicSumNotFound as e:
            reasons.append(f"{cos.name or cos.output[0]}: {e}")
    if len(found) == 1:
        return found[0]
    if not found:
        raise HarmonicSumNotFound("展開された倍音の足し合わせが見つからない。候補（ConstantOfShape）ごとの理由: "
                                  + ("; ".join(reasons) if reasons else "ConstantOfShape が無い"))
    raise HarmonicSumNotFound(f"倍音の足し合わせの候補が {len(found)} 本ある: " + ", ".join(h.final for h in found))


def _collect_used(nodes, used: set) -> None:
    """ノード（と、その属性の中の部分グラフ）が入力に使う名前を集める。"""
    for n in nodes:
        used.update(i for i in n.input if i)
        for a in n.attribute:
            if a.type == AttributeProto.GRAPH:
                _collect_used(a.g.node, used)
            elif a.type == AttributeProto.GRAPHS:
                for sg in a.graphs:
                    _collect_used(sg.node, used)


def _make_loop(hs: HarmonicSum) -> onnx.NodeProto:
    """展開形と同じ演算を同じ順に 1 周ぶん書いた本体を持つ Loop。cond は「この倍音がどこかのサンプルで
    ナイキスト未満か」（全サンプルで超えたら以降の項は厳密に 0 なので、そこで止めても結果は同じ）。"""
    P = _P
    body_nodes = [
        helper.make_node("Gather", [P + "k_table", P + "iter"], [P + "k"], axis=0),
        helper.make_node("Gather", [P + "amp_table", P + "iter"], [P + "a"], axis=0),
        helper.make_node("Mul", [hs.f0, P + "k"], [P + "kf"]),
        helper.make_node("Less", [P + "kf", P + "nyq"], [P + "lt"]),
        helper.make_node("Cast", [P + "lt"], [P + "aa"], to=hs.cast_to),
        helper.make_node("Mul", [hs.phase, P + "k"], [P + "kp"]),
        helper.make_node("Sin", [P + "kp"], [P + "sin"]),
        helper.make_node("Mul", [P + "sin", P + "a"], [P + "sa"]),
        helper.make_node("Mul", [P + "sa", P + "aa"], [P + "term"]),
        helper.make_node("Add", [P + "acc_in", P + "term"], [P + "acc_out"]),
        helper.make_node("ReduceMax", [P + "aa"], [P + "any"], keepdims=0),
        helper.make_node("Greater", [P + "any", P + "zero"], [P + "cond_out"]),
    ]
    zero = np.zeros((), helper.tensor_dtype_to_np_dtype(hs.cast_to))
    body = helper.make_graph(
        body_nodes, "harmonic_sum_body",
        [helper.make_tensor_value_info(P + "iter", TensorProto.INT64, []),
         helper.make_tensor_value_info(P + "cond_in", TensorProto.BOOL, []),
         helper.make_tensor_value_info(P + "acc_in", hs.cast_to, None)],
        [helper.make_tensor_value_info(P + "cond_out", TensorProto.BOOL, []),
         helper.make_tensor_value_info(P + "acc_out", hs.cast_to, None)],
        initializer=[numpy_helper.from_array(hs.ks, P + "k_table"),
                     numpy_helper.from_array(hs.amps, P + "amp_table"),
                     numpy_helper.from_array(np.asarray(hs.nyq).reshape(()), P + "nyq"),
                     numpy_helper.from_array(zero, P + "zero")])
    return helper.make_node("Loop", [P + "trip", P + "true", hs.acc0], [hs.final], name=LOOP_NODE_NAME, body=body)


def _copy(node: onnx.NodeProto) -> onnx.NodeProto:
    n = onnx.NodeProto()
    n.CopyFrom(node)
    return n


def convert_model(model: onnx.ModelProto) -> dict:
    """model の図をその場で書き換え、ノード数などの統計を返す。見つからなければ HarmonicSumNotFound。"""
    g = model.graph
    nodes_before, inits_before = len(g.node), len(g.initializer)
    hs = find_harmonic_sum(g)
    taken = {o for n in g.node for o in n.output} | {i.name for i in g.input} | {t.name for t in g.initializer}
    clash = sorted(n for n in taken if n.startswith(_P))
    if clash or LOOP_NODE_NAME in {n.name for n in g.node}:
        raise HarmonicSumNotFound(f"Loop 用の名前が既に使われている（もう畳んである？）: {clash[:3] or LOOP_NODE_NAME}")
    loop = _make_loop(hs)
    g.initializer.extend([numpy_helper.from_array(np.asarray(hs.n_harm, np.int64), _P + "trip"),
                          numpy_helper.from_array(np.asarray(True), _P + "true")])
    # 畳むノードを外し、Loop は最後の Add があった位置に入れる（そこなら本体が参照する phase・f0・acc0 は
    # すべて作られた後で、和を使う下流はすべて後ろにある）。残すノードは複製してから入れ直す
    last = hs.adds[-1]
    new_nodes = []
    for i, n in enumerate(g.node):
        if i == last:
            new_nodes.append(loop)
        elif i not in hs.remove:
            new_nodes.append(_copy(n))
    # 使われなくなった定数（k・a_k・nyq）を消す。他のノードや部分グラフと共有している定数は残る
    used = {o.name for o in g.output}
    _collect_used(new_nodes, used)
    new_nodes = [n for n in new_nodes if n.op_type != "Constant" or any(o in used for o in n.output)]
    del g.node[:]
    g.node.extend(new_nodes)
    keep = [t for t in g.initializer if t.name in used]
    dropped = {t.name for t in g.initializer} - {t.name for t in keep}
    del g.initializer[:]
    g.initializer.extend(keep)
    keep_in = [i for i in g.input if i.name not in dropped]      # 古い IR では initializer が input にも並ぶ
    del g.input[:]
    g.input.extend(keep_in)
    del g.value_info[:]                                          # 消したノードの型情報が残らないように
    onnx.checker.check_model(model)
    return {"n_harm": hs.n_harm, "output": hs.final, "removed_nodes": len(hs.remove),
            "nodes_before": nodes_before, "nodes_after": len(g.node),
            "initializers_before": inits_before, "initializers_after": len(g.initializer)}


def to_harmonic_loop(path_in: str, path_out: str) -> dict:
    """ファイル版。path_in と path_out は同じでもよい。"""
    model = onnx.load(path_in)
    info = convert_model(model)
    info["bytes_before"] = os.path.getsize(path_in)
    onnx.save(model, path_out)
    info["bytes_after"] = os.path.getsize(path_out)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="書き出し済みの音響モデル ONNX の倍音の足し合わせを Loop に畳む")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--check", action="store_true",
                    help="前後を onnxruntime で動かし、倍音の和（と mel）がビット単位で一致することを確かめる")
    a = ap.parse_args()
    info = to_harmonic_loop(a.src, a.dst)
    print(json.dumps(info, ensure_ascii=False))
    if a.check:
        from .verify import verify_harmonic_loop
        verify_harmonic_loop(a.src, a.dst)


if __name__ == "__main__":
    main()
