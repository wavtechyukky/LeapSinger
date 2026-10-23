"""Phoneme <-> id vocabulary, loaded from a phoneme-list file (default: Japanese).

The phoneme set is NOT hard-coded here: it is read from a plain text file (one phoneme per
line, order = id) so LeapSinger can train on any language by supplying a different list. The
bundled default is ``dict/ja.phonemes``. Pass ``--phonemes <path>`` to ``preprocess.run`` and
``train`` to use another set; the list is also stored in the checkpoint, so export / inference
reconstruct it automatically.

Backward compatibility: the module-level ``CANONICAL_PHONEMES`` / ``PHONEME2ID`` / ``N_PHONEMES``
/ ``PAU_ID`` / ``SILENCE_PHONEMES`` are the *default* (Japanese) vocab, so existing imports keep
working unchanged. Code that must honour a custom set takes a ``Vocab`` (or ``--phonemes``).
"""
from __future__ import annotations

import os

from leapsinger.phonemes import load_phonemes  # noqa: F401  (one reader for preprocess and the language packs)

_DICT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dict")
DEFAULT_PHONEMES_PATH = os.path.join(_DICT_DIR, "ja.phonemes")

# tokens treated as silence for cut/fade (cl is voiced -> not silence). Language-independent.
SILENCE_TOKENS = frozenset({"pau"})


class Vocab:
    """A phoneme set: name<->id maps + the silence/pad id. Build from a list or a file."""

    def __init__(self, phonemes: list[str]):
        self.phonemes: list[str] = list(phonemes)
        self.phoneme2id: dict[str, int] = {ph: i for i, ph in enumerate(self.phonemes)}
        self.id2phoneme: dict[int, str] = {i: ph for i, ph in enumerate(self.phonemes)}
        self.n_phonemes: int = len(self.phonemes)
        self.pau_id: int = self.phoneme2id.get("pau", 0)
        self.silence: frozenset[str] = frozenset(SILENCE_TOKENS & set(self.phonemes))

    @classmethod
    def load(cls, path: str | None = None) -> "Vocab":
        """Load from a phoneme-list file (default: the bundled Japanese ``dict/ja.phonemes``)."""
        return cls(load_phonemes(path or DEFAULT_PHONEMES_PATH))

    def __contains__(self, ph: str) -> bool:
        return ph in self.phoneme2id

    def get(self, ph: str, default=None):
        return self.phoneme2id.get(ph, default)

    def __len__(self) -> int:
        return self.n_phonemes


# ── default (Japanese) vocab + backward-compatible module-level constants ────────────────
DEFAULT_VOCAB = Vocab.load()

CANONICAL_PHONEMES = DEFAULT_VOCAB.phonemes
PHONEME2ID = DEFAULT_VOCAB.phoneme2id
ID2PHONEME = DEFAULT_VOCAB.id2phoneme
N_PHONEMES = DEFAULT_VOCAB.n_phonemes
SILENCE_PHONEMES = DEFAULT_VOCAB.silence
PAU_ID = DEFAULT_VOCAB.pau_id
