"""Token dataset and manifest utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .tokenizer import EOS, MIDI_OFFSET, PAD


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"manifest has no records: {path}")
    return records


def _crop_seed(seed: int, index: int, epoch: int) -> int:
    """Deterministic per-(record, epoch) crop seed with avalanche mixing.

    Crops are derived from the record index and the epoch it appears in, so a
    resumed run reproduces the exact same windows without depending on the
    (per-worker) NumPy RNG state at resume time.
    """

    mask = 0xFFFFFFFFFFFFFFFF
    value = (
        (seed & mask)
        ^ ((index * 0x9E3779B97F4A7C15) & mask)
        ^ ((epoch * 0xD1B54A32D192ED03) & mask)
    )
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return int(value)


class TokenSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        manifest: str | Path,
        sequence_length: int,
        supervised: bool = False,
        random_crop: bool = True,
        seed: int = 1337,
    ) -> None:
        self.records = read_manifest(manifest)
        self.manifest_path = Path(manifest)
        self.sequence_length = sequence_length
        self.supervised = supervised
        self.random_crop = random_crop
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, torch.Tensor]:
        if isinstance(key, tuple):
            index, epoch = key
        else:
            index, epoch = key, 0
        record = self.records[index]
        token_path = Path(record["tokens"])
        if not token_path.is_absolute():
            token_path = Path(self.manifest_root) / token_path
        # Memory-map the source so workers do not repeatedly copy hundreds of MB
        # of token files into Python-managed memory for every batch.
        tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
        if tokens.dtype != np.int64:
            tokens = tokens.astype(np.int64, copy=False)
        if tokens.ndim != 1 or tokens.size < 2:
            raise ValueError(f"token file must contain at least two tokens: {token_path}")

        available = max(0, tokens.size - self.sequence_length - 1)
        if self.random_crop and tokens.size > self.sequence_length + 1:
            rng = np.random.default_rng(_crop_seed(self.seed, index, epoch))
            start = int(rng.integers(0, available + 1))
        else:
            start = 0
        window = tokens[start : start + self.sequence_length + 1]
        input_ids = np.full(self.sequence_length, PAD, dtype=np.int64)
        labels = np.full(self.sequence_length, PAD, dtype=np.int64)
        valid = max(0, min(self.sequence_length, window.size - 1))
        input_ids[:valid] = window[:valid]
        labels[:valid] = window[1 : valid + 1]

        loss_mask = labels != PAD
        if self.supervised:
            loss_mask &= (labels >= MIDI_OFFSET) | (labels == EOS)
        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "loss_mask": torch.from_numpy(loss_mask.astype(np.float32)),
        }

    @property
    def manifest_root(self) -> Path:
        return self.manifest_path.parent


class ResumableShuffleSampler(Sampler[tuple[int, int]]):
    """Deterministic per-epoch shuffling that resumes from the middle.

    Each epoch draws one permutation from a single ``torch.Generator`` seeded
    with ``seed``, so the full sample stream (which record appears in which
    epoch, and in which order) is reproducible from the seed alone.  Passing
    ``samples_seen`` lets a resumed run discard exactly the samples already
    consumed and continue from the same stream position.

    Indices are yielded as ``(record_index, epoch)`` tuples so the dataset can
    derive a deterministic per-epoch crop for each record.
    """

    def __init__(
        self,
        data_source: Any,
        batch_size: int,
        seed: int,
        samples_seen: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.num_samples = len(data_source)
        self.batch_size = batch_size
        self.batches_per_epoch = self.num_samples // batch_size
        if self.batches_per_epoch < 1:
            raise ValueError(
                f"dataset has {self.num_samples} records, which is smaller than "
                f"batch_size={batch_size}"
            )
        self.generator = torch.Generator().manual_seed(seed)
        self._epoch = 0
        self._permutation: torch.Tensor | None = None
        self._offset = 0
        if samples_seen > 0:
            self._advance(samples_seen)

    def _advance(self, samples_seen: int) -> None:
        """Discard already-consumed samples, stopping at a batch boundary."""

        batches_to_skip, remainder = divmod(samples_seen, self.batch_size)
        if remainder:
            print(
                f"warning: resume at sample {samples_seen} is not aligned to "
                f"batch_size={self.batch_size}; replaying {remainder} sample(s)"
            )
        epochs_skipped, batches_in_epoch = divmod(
            batches_to_skip, self.batches_per_epoch
        )
        # Draw (and discard) one permutation per fully-skipped epoch so the
        # generator ends up exactly where the interrupted run left it.
        for _ in range(epochs_skipped):
            torch.randperm(self.num_samples, generator=self.generator)
        self._epoch = epochs_skipped
        if batches_in_epoch:
            self._permutation = torch.randperm(
                self.num_samples, generator=self.generator
            )
            self._offset = batches_in_epoch * self.batch_size
        # else: exactly at an epoch boundary; __iter__ draws the next permutation.

    def __iter__(self) -> Iterator[tuple[int, int]]:
        if self._permutation is None:
            self._permutation = torch.randperm(
                self.num_samples, generator=self.generator
            )
        permutation = self._permutation
        offset = self._offset
        epoch = self._epoch
        for index in permutation[offset:].tolist():
            yield int(index), epoch
        self._epoch = epoch + 1
        self._permutation = None
        self._offset = 0

    def __len__(self) -> int:
        return self.num_samples


def make_dataset(
    manifest: str | Path,
    sequence_length: int,
    supervised: bool = False,
    random_crop: bool = True,
    seed: int = 1337,
) -> TokenSequenceDataset:
    return TokenSequenceDataset(
        manifest, sequence_length, supervised, random_crop, seed
    )
