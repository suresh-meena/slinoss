#!/usr/bin/env python3
"""Streaming FineWeb-Edu input pipeline for LM experiments."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional for local smoke tests.
    load_dataset = None  # type: ignore[assignment]

try:
    from datasets.distributed import split_dataset_by_node
except Exception:  # pragma: no cover - older datasets fallback.
    split_dataset_by_node = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional for byte-tokenizer smoke tests.
    AutoTokenizer = None  # type: ignore[assignment]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True)
class DataSpec:
    vocab_size: int
    eos_token_id: int
    tokenizer_name: str


class ByteTokenizer:
    """Offline byte-level tokenizer for smoke tests."""

    eos_token_id = 256
    pad_token_id = 256
    vocab_size = 257

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> dict[str, list[list[int]]]:
        del add_special_tokens, truncation
        return {"input_ids": [list(text.encode("utf-8")) for text in texts]}


class LocalExampleStream:
    """Small in-memory stream for offline smoke tests."""

    def __init__(self, examples: Iterable[dict[str, Any]]) -> None:
        self.examples = tuple(examples)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.examples

    def shuffle(self, *, seed: int, buffer_size: int):
        del buffer_size
        items = list(self.examples)
        random.Random(seed).shuffle(items)
        return LocalExampleStream(items)

    def take(self, n: int):
        return LocalExampleStream(self.examples[:n])

    def skip(self, n: int):
        return LocalExampleStream(self.examples[n:])


def load_tokenizer(config: dict[str, Any]):
    tokenizer_name = str(config["tokenizer_name"])
    if tokenizer_name in {"byte", "bytes", "byte-level-smoke"}:
        return ByteTokenizer()

    _require(
        AutoTokenizer is not None,
        "transformers is required for Hugging Face tokenizers. "
        "Use data.tokenizer_name=byte for offline smoke tests.",
    )
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=bool(config.get("tokenizer_use_fast", True)),
        token=token,
        cache_dir=str(cache_dir / "tokenizer"),
    )
    _require(tokenizer.eos_token_id is not None, "Tokenizer must expose eos_token_id.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _shard_iterable(
    iterable: Iterable[dict[str, Any]],
    *,
    rank: int,
    world_size: int,
) -> Iterator[dict[str, Any]]:
    if world_size <= 1:
        yield from iterable
        return
    for index, item in enumerate(iterable):
        if index % world_size == rank:
            yield item


def _batched_texts(
    iterable: Iterable[dict[str, Any]],
    *,
    text_field: str,
    batch_size: int,
) -> Iterator[list[str]]:
    batch: list[str] = []
    for example in iterable:
        text = example.get(text_field)
        _require(
            isinstance(text, str),
            f"Expected dataset field '{text_field}' to be a string. Got {type(text)}.",
        )
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class PackedTokenStreamDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Tokenize text on the fly and pack it into contiguous LM blocks."""

    def __init__(
        self,
        *,
        dataset,
        tokenizer,
        seq_len: int,
        text_field: str,
        add_eos_token: bool,
        tokenizer_batch_size: int,
        process_index: int,
        num_processes: int,
        max_blocks: int | None = None,
        pre_sharded_by_process: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.text_field = str(text_field)
        self.add_eos_token = bool(add_eos_token)
        self.tokenizer_batch_size = int(tokenizer_batch_size)
        self.process_index = int(process_index)
        self.num_processes = int(num_processes)
        self.max_blocks = None if max_blocks is None else int(max_blocks)
        self.pre_sharded_by_process = bool(pre_sharded_by_process)

    def _document_stream(self) -> Iterable[dict[str, Any]]:
        stream = self.dataset
        if not self.pre_sharded_by_process and self.num_processes > 1:
            stream = _shard_iterable(
                stream,
                rank=self.process_index,
                world_size=self.num_processes,
            )

        worker = get_worker_info()
        if worker is None or worker.num_workers <= 1:
            yield from stream
            return

        yield from _shard_iterable(
            stream,
            rank=worker.id,
            world_size=worker.num_workers,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        emitted = 0
        buffer: list[int] = []
        eos_token_id = int(self.tokenizer.eos_token_id)

        text_batches = _batched_texts(
            self._document_stream(),
            text_field=self.text_field,
            batch_size=self.tokenizer_batch_size,
        )
        for texts in text_batches:
            encoded = self.tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            for token_ids in encoded:
                if not token_ids:
                    continue
                buffer.extend(int(token_id) for token_id in token_ids)
                if self.add_eos_token:
                    buffer.append(eos_token_id)
                while len(buffer) >= self.seq_len + 1:
                    x = torch.tensor(buffer[: self.seq_len], dtype=torch.long)
                    y = torch.tensor(buffer[1 : self.seq_len + 1], dtype=torch.long)
                    yield {"input_ids": x, "labels": y}
                    emitted += 1
                    del buffer[: self.seq_len]
                    if self.max_blocks is not None and emitted >= self.max_blocks:
                        return


def _resolve_local_data_files(raw: Any, *, split: str) -> list[Path]:
    if isinstance(raw, str):
        return [Path(raw)]
    if isinstance(raw, list):
        return [Path(item) for item in raw]
    if isinstance(raw, dict):
        selected = raw.get(split)
        _require(
            selected is not None,
            f"dataset_kwargs.data_files is missing split key {split!r}.",
        )
        if isinstance(selected, str):
            return [Path(selected)]
        _require(isinstance(selected, list), "Split-specific data_files must be a string or list.")
        return [Path(item) for item in selected]
    raise ValueError("dataset_kwargs.data_files must be a string, list, or split->paths mapping.")


def _load_local_jsonl_stream(config: dict[str, Any]) -> LocalExampleStream:
    dataset_kwargs = config.get("dataset_kwargs", {})
    _require(isinstance(dataset_kwargs, dict), "data.dataset_kwargs must be a mapping when set.")
    paths = _resolve_local_data_files(
        dataset_kwargs.get("data_files"),
        split=str(config["dataset_split"]),
    )
    examples: list[dict[str, Any]] = []
    for path in paths:
        _require(path.exists(), f"Local JSONL dataset file does not exist: {path}.")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                _require(isinstance(record, dict), f"JSONL record must decode to an object. Got {type(record)}.")
                examples.append(record)
    return LocalExampleStream(examples)


def _load_streaming_split(config: dict[str, Any], *, process_index: int, num_processes: int):
    dataset_name = str(config["dataset_name"])
    if dataset_name == "local_jsonl":
        dataset = _load_local_jsonl_stream(config)
        shuffle_buffer_size = int(config.get("shuffle_buffer_size", 0))
        if shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                seed=int(config.get("seed", 0)),
                buffer_size=shuffle_buffer_size,
            )
        validation_docs = int(config["validation_docs"])
        return dataset.skip(validation_docs), dataset.take(validation_docs), False

    _require(
        load_dataset is not None,
        "The `datasets` package is required for Hugging Face dataset loading. "
        "Install experiments/language_modeling/requirements.txt or use data.dataset_name=local_jsonl for smoke tests.",
    )
    token = os.environ.get("HF_TOKEN")
    dataset_kwargs = config.get("dataset_kwargs", {})
    _require(
        isinstance(dataset_kwargs, dict),
        "data.dataset_kwargs must be a mapping when set.",
    )
    dataset = load_dataset(
        dataset_name,
        config.get("dataset_config_name"),
        split=config["dataset_split"],
        streaming=bool(config.get("streaming", True)),
        cache_dir=str(Path(config["cache_dir"]) / "datasets"),
        token=token,
        **dataset_kwargs,
    )
    shuffle_buffer_size = int(config.get("shuffle_buffer_size", 0))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            seed=int(config.get("seed", 0)),
            buffer_size=shuffle_buffer_size,
        )
    validation_docs = int(config["validation_docs"])
    eval_stream = dataset.take(validation_docs)
    train_stream = dataset.skip(validation_docs)

    pre_sharded = False
    if split_dataset_by_node is not None and num_processes > 1:
        train_stream = split_dataset_by_node(
            train_stream,
            rank=process_index,
            world_size=num_processes,
        )
        eval_stream = split_dataset_by_node(
            eval_stream,
            rank=process_index,
            world_size=num_processes,
        )
        pre_sharded = True
    return train_stream, eval_stream, pre_sharded


def create_dataloaders(
    config: dict[str, Any],
    *,
    tokenizer,
    process_index: int,
    num_processes: int,
    device_type: str,
) -> tuple[DataLoader, DataLoader, DataSpec]:
    data = dict(config["data"])
    training = dict(config["training"])

    data["seed"] = int(config["experiment"]["seed"])
    train_stream, eval_stream, pre_sharded = _load_streaming_split(
        data,
        process_index=process_index,
        num_processes=num_processes,
    )

    max_eval_blocks = int(training["max_eval_batches"]) * int(training["eval_batch_size"])
    train_dataset = PackedTokenStreamDataset(
        dataset=train_stream,
        tokenizer=tokenizer,
        seq_len=int(training["seq_len"]),
        text_field=str(data["text_field"]),
        add_eos_token=bool(data.get("add_eos_token", True)),
        tokenizer_batch_size=int(data["tokenizer_batch_size"]),
        process_index=process_index,
        num_processes=num_processes,
        pre_sharded_by_process=pre_sharded,
    )
    eval_dataset = PackedTokenStreamDataset(
        dataset=eval_stream,
        tokenizer=tokenizer,
        seq_len=int(training["seq_len"]),
        text_field=str(data["text_field"]),
        add_eos_token=bool(data.get("add_eos_token", True)),
        tokenizer_batch_size=int(data["tokenizer_batch_size"]),
        process_index=process_index,
        num_processes=num_processes,
        max_blocks=max_eval_blocks,
        pre_sharded_by_process=pre_sharded,
    )

    pin_memory = bool(training.get("pin_memory", True)) and device_type == "cuda"
    num_workers = int(training.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["micro_batch_size"]),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(training["eval_batch_size"]),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    spec = DataSpec(
        vocab_size=vocab_size,
        eos_token_id=int(tokenizer.eos_token_id),
        tokenizer_name=str(data["tokenizer_name"]),
    )
    return train_loader, eval_loader, spec
