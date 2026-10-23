"""Speaker-vector helper for multi-speaker models (HarmonicAcousticModelMultiSpk).

The multispk model adds a per-speaker vector `spk_proj(spk_bank.weight[i])` (dim = hidden) to the
condition at every frame (`_spk_add`). For ONNX export we either bake one speaker's vector into
the graph, or expose an H-dim `spk_embed` graph input the host adds itself.

A single-speaker model (HarmonicAcousticModel, no spk_bank) has no speaker embedding — nothing
to export; WrapperA uses speaker='none'.
"""
from __future__ import annotations

import numpy as np
import torch


def has_speakers(model) -> bool:
    return getattr(model, "spk_bank", None) is not None


def speaker_vector(model, spk_id: int) -> np.ndarray:
    """The H-dim vector `_spk_add` would add for `spk_id` (== spk_proj(spk_bank[spk_id]))."""
    assert has_speakers(model), "model has no speaker bank"
    with torch.no_grad():
        idx = torch.tensor([int(spk_id)], device=model.spk_bank.weight.device)
        v = model.spk_proj(model.spk_bank(idx))[0]         # [hidden]
    return v.detach().cpu().numpy().astype(np.float32)


def speakers_table(model, cfg: dict) -> dict:
    """Everything a host needs to drive a `--speaker embed` graph: the H-dim vector of every speaker
    id (what `spk_embed` expects) and the phoneme list behind the `tokens` ids.
    {"dim": H, "speakers": {"0": [H floats], ...}, "phonemes": [...]}"""
    assert has_speakers(model), "model has no speaker bank"
    n = int(model.spk_bank.num_embeddings)
    vecs = {str(i): speaker_vector(model, i).tolist() for i in range(n)}
    return {"dim": int(model.hidden), "speakers": vecs, "phonemes": list(cfg.get("phonemes") or [])}


def write_speakers_json(model, cfg: dict, path: str) -> dict:
    """Write speakers_table() next to the exported ONNX (`{model}.speakers.json`)."""
    import json
    table = speakers_table(model, cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False)
    return table
