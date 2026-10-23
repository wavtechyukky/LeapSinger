"""ONNX post-processing: (optional) simplify, then a PARTIAL fp16 conversion.

Policy: export fp32 from PyTorch, then convert to fp16 here — never quantize the
PyTorch model before export, and no int8. fp16 halves size and speeds ORT up; IO stays fp32 so
the host keeps passing/receiving float32 (int tokens/durations are untouched).

Why a hand-written converter instead of onnxconverter_common / onnxruntime.transformers.float16:
BOTH of those fail on this graph. The harmonic excitation (`/exc/*`) must stay fp32 — its
cumulative phase reaches thousands of radians and fp16 cannot represent integers past 2048, so
sin(phase) would be destroyed — but the excitation subgraph also carries an `If` (from reflect
padding), mixes int/bool/float, and shares scalar constants with the rest of the graph. The
library converters' node_block_list leaves stale mixed-precision edges (Div/Mul type errors at
`/exc` boundaries) that ORT refuses to load. `to_fp16` below does it precisely: it converts only
the non-excitation region to fp16, keeps `/exc` (and its If subgraph) fp32, duplicates any shared
initializer, and — using onnx shape inference to know each edge's element type — inserts Cast
nodes on exactly the float edges that cross the fp32<->fp16 boundary (and at graph IO). The
result loads and runs in stock ORT.

`simplify()` is best-effort (skipped with a log if onnxsim is absent/fails); `to_fp16()` is what
matters for the deliverable.

The last pass, `to_harmonic_loop` (export/harmonic_loop.py), folds the excitation's unrolled
harmonic sum (n_harm straight-line blocks) into one ONNX `Loop`: bit-identical output, ~5x lower
peak memory in every ORT and a much faster session creation. It only touches the fp32 `/exc`
region, so it runs after fp16 and is independent of it.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .harmonic_loop import HarmonicSumNotFound, to_harmonic_loop


def simplify(path_in: str, path_out: str | None = None) -> str:
    """onnxsim.simplify with dynamic-shape support. Returns the written path (or path_in on skip)."""
    path_out = path_out or path_in
    try:
        from onnxsim import simplify as _simplify
    except Exception as e:                                   # onnxsim optional
        print(f"[postprocess] onnxsim unavailable ({e.__class__.__name__}); skip simplify")
        return path_in
    model = onnx.load(path_in)
    try:
        sm, ok = _simplify(model)
    except Exception as e:
        print(f"[postprocess] simplify failed ({e.__class__.__name__}: {e}); keeping unsimplified")
        return path_in
    if not ok:
        print("[postprocess] simplify check failed; keeping unsimplified")
        return path_in
    onnx.save(sm, path_out)
    return path_out


def _elem_types(model):
    """value name -> elem_type (int enum) for every graph input/output/value_info, via shape
    inference. Values whose type can't be inferred are simply absent (treated as 'not float')."""
    m = onnx.shape_inference.infer_shapes(model)
    types = {}
    for coll in (m.graph.input, m.graph.output, m.graph.value_info):
        for vi in coll:
            if vi.type.HasField("tensor_type"):
                types[vi.name] = vi.type.tensor_type.elem_type
    return types


def to_fp16(path_in: str, path_out: str, *, keep_io_fp32: bool = True,
            fp16_prefixes=("/velocity_fn", "/phoneme_encoder"),
            keep_fp32_prefixes=("/exc",)) -> str:
    """Partial fp16, with Cast nodes on exactly the float edges that cross the fp32<->fp16 boundary
    (and at IO when keep_io_fp32). Two selection modes:

      * fp16_prefixes given (default): convert ONLY those node subtrees to fp16 — the heavy,
        weight-bearing conv backbone (`/velocity_fn*`, all 10 flow steps share its weights) and the
        attention encoder (`/phoneme_encoder`). Everything else — excitation DSP, the flow's Euler
        ACCUMULATION (`x = x + v*dt`), mel (de)norm, up/downsample — stays fp32. This gets almost all
        the size win (the weights) while keeping the sensitive integration exact, so fp16 error
        does not grow with sequence length.
      * fp16_prefixes=None: inverse — convert everything EXCEPT `keep_fp32_prefixes` to fp16."""
    FP32, FP16 = TensorProto.FLOAT, TensorProto.FLOAT16
    model = onnx.load(path_in)
    etypes = _elem_types(model)
    g = model.graph

    if fp16_prefixes is not None:
        def is_keep(name):                                           # keep = NOT in an fp16 subtree
            return not (bool(name) and any(name.startswith(p) for p in fp16_prefixes))
    else:
        def is_keep(name):
            return bool(name) and any(name.startswith(p) for p in keep_fp32_prefixes)

    keep_nodes = [n for n in g.node if is_keep(n.name)]
    conv_nodes = [n for n in g.node if not is_keep(n.name)]
    # region of each value = region of its producing node ('keep' fp32 / 'conv' fp16)
    region = {}
    for n in g.node:
        r = "keep" if is_keep(n.name) else "conv"
        for o in n.output:
            region[o] = r

    init = {t.name: t for t in g.initializer}
    input_names = {i.name for i in g.input}
    output_names = {o.name for o in g.output}

    # ---- initializers: conv-only float32 -> fp16 in place; shared -> keep fp32 + fp16 dup ----
    used_keep, used_conv = set(), set()
    for n in keep_nodes:
        used_keep.update(i for i in n.input if i in init)
    for n in conv_nodes:
        used_conv.update(i for i in n.input if i in init)
    dup_map = {}
    fp16_inits = set()
    new_inits = []
    for name, t in init.items():
        if t.data_type != FP32:
            new_inits.append(t); continue
        in_keep, in_conv = name in used_keep, name in used_conv
        if in_conv and not in_keep:                                   # conv-only -> fp16
            arr = numpy_helper.to_array(t).astype(np.float16)
            new_inits.append(numpy_helper.from_array(arr, name)); fp16_inits.add(name)
        elif in_keep and in_conv:                                     # shared -> fp32 + fp16 dup
            new_inits.append(t)
            dup = name + "__fp16"
            new_inits.append(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float16), dup))
            dup_map[name] = dup; fp16_inits.add(dup)
        else:
            new_inits.append(t)                                       # keep-only or unused -> fp32
    for n in conv_nodes:                                             # rewire conv to fp16 dup
        for i, v in enumerate(n.input):
            if v in dup_map:
                n.input[i] = dup_map[v]
    del g.initializer[:]; g.initializer.extend(new_inits)

    # ---- conv nodes: make their float outputs genuinely fp16 ----
    # (a) Cast(to=FLOAT) -> Cast(to=FLOAT16); (b) Random*/EyeLike dtype FLOAT -> FLOAT16;
    # (c) any float32 constant TENSOR attribute (Constant.value, ConstantOfShape.value) -> fp16.
    _DTYPE_ATTR_OPS = ("RandomNormal", "RandomNormalLike", "RandomUniform",
                       "RandomUniformLike", "EyeLike", "Multinomial")
    for n in conv_nodes:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == FP32:
                    a.i = FP16
        if n.op_type in _DTYPE_ATTR_OPS:
            for a in n.attribute:
                if a.name == "dtype" and a.i == FP32:
                    a.i = FP16
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == FP32:
                arr = numpy_helper.to_array(a.t).astype(np.float16)
                a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
            elif a.name == "value_float" and a.type == onnx.AttributeProto.FLOAT:
                pass  # scalar float Constant attr stays value_float; ORT reads it per output type

    # ---- element type of a value AFTER conversion (float32/float16), else None (int/bool/unknown) ----
    def float_dtype(v):
        et = etypes.get(v)
        if et not in (FP32, FP16):
            return None                                              # not a float edge -> never cast
        if v in input_names:
            return FP32                                              # graph inputs kept fp32
        if v in init:
            return FP16 if v in fp16_inits else FP32
        if v in dup_map.values():
            return FP16
        r = region.get(v)
        if r == "conv":
            return FP16
        if r == "keep":
            return FP32
        return FP32                                                  # unproduced fp32 constant edge

    # ---- insert boundary casts on float edges ----
    cast_cache = {}                                                  # (value, target) -> cast output
    new_nodes = []

    def cast_to(v, target, ctx):
        key = (v, target)
        if key in cast_cache:
            return cast_cache[key]
        out = f"{v}__to{'16' if target == FP16 else '32'}"
        new_nodes.append(helper.make_node("Cast", [v], [out], to=target,
                                          name=f"cast_{ctx}_{len(cast_cache)}"))
        cast_cache[key] = out
        return out

    for n in g.node:
        target = FP16 if not is_keep(n.name) else FP32               # what this node wants its floats in
        for i, v in enumerate(n.input):
            if not v:
                continue
            d = float_dtype(v)
            if d is not None and d != target:
                n.input[i] = cast_to(v, target, n.name.strip("/").replace("/", "_") or "n")

    # graph outputs carry fp32 (keep_io): if produced as fp16, rename the producer's output and
    # add a final Cast -> the original output name (so the output name/contract is preserved).
    if keep_io_fp32:
        producer = {o: n for n in g.node for o in n.output}
        for out in g.output:
            v = out.name
            if float_dtype(v) == FP16:
                p = producer.get(v)
                pre = v + "__pre16"
                if p is not None:
                    for j, o in enumerate(p.output):
                        if o == v:
                            p.output[j] = pre
                    new_nodes.append(helper.make_node("Cast", [pre], [v], to=FP32,
                                                      name=f"cast_gout_{v}"))
            out.type.tensor_type.elem_type = FP32

    g.node.extend(new_nodes)
    # topologically re-sort (casts were appended; ORT needs producers before consumers)
    _toposort(g)
    del g.value_info[:]                                              # let ORT re-infer intermediates
    onnx.save(model, path_out)
    return path_out


def _toposort(graph):
    """Stable topological sort of graph.node (Cast nodes were appended out of order)."""
    produced = set(t.name for t in graph.initializer) | set(i.name for i in graph.input)
    nodes = list(graph.node)
    ordered, remaining = [], nodes
    while remaining:
        progressed = False
        still = []
        for n in remaining:
            if all((not i) or i in produced for i in n.input):
                ordered.append(n); produced.update(n.output); progressed = True
            else:
                still.append(n)
        remaining = still
        if not progressed:                                          # cycle / missing input: append rest
            ordered.extend(remaining); break
    del graph.node[:]; graph.node.extend(ordered)


def to_native_dft(path_in: str, path_out: str, *, n_fft: int = 2048) -> str:
    """Replace the excitation's DFT-as-matmul (two [n_fft, n_fft//2+1] cos/sin basis MatMuls,
    ~17 MB of constants) with a single native ONNX `DFT` operator (a few bytes). This is what
    `torch.fft.fft` would have produced if the legacy exporter still supported it (it doesn't in
    torch>=2.9; the dynamo exporter does, and emits exactly this DFT node). Numerically identical
    to the matmul (verified MAE ~6e-6) and ORT-supported.

    The excitation's `_stft_mag` does: re = frames @ dft_cos ; im = frames @ dft_sin ;
    mag = sqrt(re^2+im^2). Here the two MatMuls become Unsqueeze -> DFT(onesided) -> Gather(re),
    Gather(im), REUSING the original MatMul output names so the downstream magnitude/transpose
    chain is untouched. cos vs sin is told apart by basis[0,0] (cos(0)=1, -sin(0)=0)."""
    model = onnx.load(path_in)
    g = model.graph
    Fb = n_fft // 2 + 1
    init = {t.name: t for t in g.initializer}
    # the two DFT-basis matmuls: MatMul with an initializer input of shape [n_fft, Fb]
    found = []
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        for j, inp in enumerate(n.input):
            t = init.get(inp)
            if t is not None and list(t.dims) == [n_fft, Fb]:
                found.append((n, inp, n.input[1 - j]))
    if len(found) != 2:
        print(f"[postprocess] to_native_dft: expected 2 DFT-basis matmuls, found {len(found)} — skip")
        onnx.save(model, path_out)
        return path_out
    (m0, b0, d0), (m1, b1, d1) = found
    assert d0 == d1, "DFT-basis matmuls do not share the frames input"
    frames = d0
    cos_mm, sin_mm = (m0, m1) if abs(float(numpy_helper.to_array(init[b0])[0, 0]) - 1.0) < 0.5 else (m1, m0)
    cos_basis = b0 if cos_mm is m0 else b1
    sin_basis = b1 if cos_mm is m0 else b0
    re_name, im_name = cos_mm.output[0], sin_mm.output[0]              # keep downstream wiring

    p = "native_dft"
    g.initializer.extend([
        numpy_helper.from_array(np.array([3], dtype=np.int64), p + "_axes"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), p + "_i0"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), p + "_i1"),
    ])
    frames4, spec = p + "_frames4", p + "_spec"
    # frames4 [B,T',n_fft,1] -> DFT along n_fft (axis 2), onesided -> [B,T',Fb,2].
    # opset 17-19 take `axis` as an attribute; opset 20 moved it to an (optional) input.
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 17)
    if opset >= 20:
        g.initializer.append(numpy_helper.from_array(np.array(2, dtype=np.int64), p + "_axis"))
        dft = helper.make_node("DFT", [frames4, "", p + "_axis"], [spec], name=p + "_dft",
                               inverse=0, onesided=1)
    else:
        dft = helper.make_node("DFT", [frames4], [spec], name=p + "_dft", axis=2, inverse=0, onesided=1)
    new = [
        helper.make_node("Unsqueeze", [frames, p + "_axes"], [frames4], name=p + "_unsq"),
        dft,
        helper.make_node("Gather", [spec, p + "_i0"], [re_name], axis=3, name=p + "_re"),  # real
        helper.make_node("Gather", [spec, p + "_i1"], [im_name], axis=3, name=p + "_im"),  # imag
    ]
    keep = [n for n in g.node if n is not cos_mm and n is not sin_mm]
    del g.node[:]; g.node.extend(keep + new)
    remain = [t for t in g.initializer if t.name not in (cos_basis, sin_basis)]
    del g.initializer[:]; g.initializer.extend(remain)
    _toposort(g)
    del g.value_info[:]
    onnx.save(model, path_out)
    return path_out


def finalize(path_fp32: str, path_out: str, *, fp16: bool = True, do_simplify: bool = True,
             native_dft: bool = False, harmonic_loop: bool = True, unrolled_out: str | None = None) -> str:
    """simplify -> native DFT -> fp16 -> harmonic Loop (each optional) -> a single path_out.
    `harmonic_loop` folds the excitation's unrolled harmonic sum into one ONNX `Loop` (see
    export/harmonic_loop.py); it fails loudly if the graph does not have the expected chain, so
    a changed excitation cannot silently ship the memory-hungry unrolled form. `unrolled_out`
    keeps a copy of the graph from just before that pass, for verify.verify_harmonic_loop."""
    src = simplify(path_fp32) if do_simplify else path_fp32
    ndft_tmp = None
    if native_dft:
        ndft_tmp = path_out + ".ndft.onnx"
        src = to_native_dft(src, ndft_tmp)
    if fp16:
        src = to_fp16(src, path_out)
    if harmonic_loop:
        if unrolled_out:
            shutil.copyfile(src, unrolled_out)
        try:
            info = to_harmonic_loop(src, path_out)
        except HarmonicSumNotFound as e:
            raise SystemExit(f"[postprocess] harmonic loop: {e}\n(pass harmonic_loop=False / --no-harmonic-loop "
                             "to keep the unrolled chain)") from e
        print(f"[postprocess] harmonic loop: {info['n_harm']} harmonics -> one Loop; nodes "
              f"{info['nodes_before']} -> {info['nodes_after']}, initializers "
              f"{info['initializers_before']} -> {info['initializers_after']}")
        src = path_out
    if src != path_out:
        onnx.save(onnx.load(src), path_out)
    if ndft_tmp and ndft_tmp != path_out and os.path.exists(ndft_tmp):
        os.remove(ndft_tmp)                                   # drop the native-DFT intermediate
    return path_out
