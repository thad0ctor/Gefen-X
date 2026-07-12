"""Deterministic byte-level pretraining and instruction-tuning data."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


Block = tuple[torch.Tensor, torch.Tensor]


class PackedPretrainingBlocks(Sequence[Block]):
    """Compact uint8 next-byte blocks; labels alias inputs until batching."""

    def __init__(self, rows: torch.Tensor):
        if rows.dtype != torch.uint8 or rows.ndim != 2:
            raise ValueError("PackedPretrainingBlocks expects [blocks, sequence] uint8")
        self.rows = rows

    def __len__(self) -> int:
        return self.rows.shape[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return PackedPretrainingBlocks(self.rows[index])
        row = self.rows[index]
        return row, row

    def __iter__(self):
        for row in self.rows:
            yield row, row


@dataclass
class DatasetBundle:
    train_blocks: Sequence[Block]
    validation_blocks: Sequence[Block]
    order: list[int]
    order_sha256: str
    data_sha256: str
    source_metadata: dict[str, Any]
    batch_size: int
    training_source_ids: tuple[int | str, ...]
    validation_source_ids: tuple[int | str, ...]
    training_source_hashes: tuple[str, ...]
    validation_source_hashes: tuple[str, ...]

    def training_batch(self, update: int) -> Block:
        start = update * self.batch_size
        indices = self.order[start : start + self.batch_size]
        if len(indices) != self.batch_size:
            raise IndexError(f"training update {update} is outside the prepared deterministic order")
        input_ids = torch.stack([self.train_blocks[index][0] for index in indices]).long()
        labels = torch.stack([self.train_blocks[index][1] for index in indices]).long()
        return input_ids, labels


def encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


def _synthetic_pretraining_records() -> list[str]:
    base = (
        "In a reproducible experiment, every optimizer sees the same tokens in the same order. ",
        "A small decoder predicts the next byte while attention mixes information from earlier positions. ",
        "Muon updates hidden matrices; an auxiliary optimizer updates embeddings, heads, norms, and biases. ",
        "Validation is held out, token weighted, and summarized over the final evaluation windows. ",
        "Checkpoint lineage records which pretraining optimizer produced each fine-tuning initialization. ",
    )
    return [f"document {index:04d}: {base[index % len(base)]}" for index in range(4096)]


def _synthetic_sft_records() -> list[dict[str, str]]:
    operations = (
        ("Reverse the supplied word.", "optimizer", "rezimitpo"),
        ("Return the sum of the two integers.", "17 and 25", "42"),
        ("Name the optimizer used for auxiliary parameters.", "Muon matrix split", "AdamW"),
        ("Complete the sequence.", "2, 4, 6, 8", "10"),
        ("Answer with the held-out metric.", "language-model comparison", "validation loss"),
    )
    records = []
    for index in range(2048):
        instruction, input_text, output = operations[index % len(operations)]
        records.append(
            {
                "instruction": f"{instruction} Example {index}.",
                "input": input_text,
                "output": output,
            }
        )
    return records


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
    return records


def _load_hf(
    name: str,
    split: str,
    cache_only: bool,
    *,
    config: str | None = None,
    revision: str | None = None,
):
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - optional real-data path
        raise RuntimeError("--source hf requires the optional 'datasets' package") from exc
    download_config = datasets.DownloadConfig(local_files_only=True) if cache_only else None
    return datasets.load_dataset(
        name,
        config,
        split=split,
        revision=revision,
        download_config=download_config,
    )


def _pack_ordered_hf_text_rows(
    dataset,
    *,
    split: str,
    text_column: str,
    sequence_length: int,
    max_blocks: int | None,
    max_raw_bytes: int | None,
):
    stream = bytearray()
    source_ids: list[str] = []
    source_hashes: list[str] = []
    raw_bytes = 0
    block_byte_cap = max_blocks * sequence_length if max_blocks is not None else None
    for row_id, record in enumerate(dataset):
        if max_raw_bytes is not None and raw_bytes >= max_raw_bytes:
            break
        if block_byte_cap is not None and len(stream) >= block_byte_cap:
            break
        try:
            text = str(record[text_column])
        except KeyError as exc:
            raise ValueError(f"HF rows lack text column {text_column!r}") from exc
        encoded = text.encode("utf-8", errors="replace")
        if stream:
            stream.extend(b"\n")
            raw_bytes += 1
        stream.extend(encoded)
        raw_bytes += len(encoded)
        source_ids.append(f"{split}:row{row_id}")
        source_hashes.append(hashlib.sha256(encoded).hexdigest())
    if block_byte_cap is not None:
        packed_stream = bytes(stream[:block_byte_cap])
    else:
        packed_stream = bytes(stream)
    blocks = _pretraining_byte_blocks(packed_stream, sequence_length)
    return {
        "blocks": blocks,
        "source_ids": tuple(source_ids),
        "source_hashes": tuple(source_hashes),
        "selected_rows": len(source_ids),
        "raw_bytes": raw_bytes,
        "packed_bytes": len(blocks) * sequence_length,
    }


def _pretraining_blocks(records: Iterable[str], sequence_length: int) -> PackedPretrainingBlocks:
    stream = bytes("\n".join(records), "utf-8", errors="replace")
    return _pretraining_byte_blocks(stream, sequence_length)


def _pretraining_byte_blocks(stream: bytes, sequence_length: int) -> PackedPretrainingBlocks:
    usable = len(stream) - (len(stream) % sequence_length)
    if usable == 0:
        return PackedPretrainingBlocks(torch.empty((0, sequence_length), dtype=torch.uint8))
    rows = torch.frombuffer(bytearray(stream[:usable]), dtype=torch.uint8).view(-1, sequence_length)
    return PackedPretrainingBlocks(rows)


def _render_sft(record: dict[str, Any]) -> tuple[list[int], list[int]]:
    try:
        instruction = str(record["instruction"]).strip()
        output = str(record["output"]).strip()
    except KeyError as exc:
        raise ValueError("SFT records must contain instruction and output fields") from exc
    input_text = str(record.get("input") or "").strip()
    if input_text:
        prompt = ALPACA_WITH_INPUT.format(instruction=instruction, input=input_text)
    else:
        prompt = ALPACA_NO_INPUT.format(instruction=instruction)
    prompt_ids = encode_bytes(prompt)
    response_ids = encode_bytes(output + "\n")
    return prompt_ids + response_ids, [-100] * len(prompt_ids) + response_ids


def _sft_blocks(records: Iterable[dict[str, Any]], sequence_length: int) -> list[Block]:
    buffer_ids: list[int] = []
    buffer_labels: list[int] = []
    blocks: list[Block] = []
    for record in records:
        ids, labels = _render_sft(record)
        buffer_ids.extend(ids)
        buffer_labels.extend(labels)
        while len(buffer_ids) >= sequence_length:
            ids_chunk = buffer_ids[:sequence_length]
            labels_chunk = buffer_labels[:sequence_length]
            buffer_ids = buffer_ids[sequence_length:]
            buffer_labels = buffer_labels[sequence_length:]
            # Prompt-only chunks have no objective and would create a 0/0
            # average. They are omitted deterministically for every cell.
            if any(label != -100 for label in labels_chunk[1:]):
                blocks.append(
                    (
                        torch.tensor(ids_chunk, dtype=torch.long),
                        torch.tensor(labels_chunk, dtype=torch.long),
                    )
                )
    return blocks


def deterministic_order(num_blocks: int, updates: int, batch_size: int, seed: int) -> list[int]:
    if num_blocks < 1:
        raise ValueError("at least one training block is required")
    needed = updates * batch_size
    order: list[int] = []
    epoch = 0
    while len(order) < needed:
        indices = list(range(num_blocks))
        random.Random(seed + epoch).shuffle(indices)
        order.extend(indices)
        epoch += 1
    return order[:needed]


def _update_blocks_hash(digest, blocks: Iterable[Block]) -> None:
    """Append logical ``(input_ids, labels)`` bytes in block order."""

    if isinstance(blocks, PackedPretrainingBlocks):
        # Labels alias inputs for packed next-byte data. Repeat each row next
        # to itself so this bulk path is byte-for-byte equivalent to iterating
        # logical ``(row, row)`` block tuples, while avoiding the Python loop.
        digest.update(blocks.rows.numpy().repeat(2, axis=0).tobytes())
        return
    for input_ids, labels in blocks:
        digest.update(input_ids.numpy().tobytes())
        digest.update(labels.numpy().tobytes())


def _hash_blocks(blocks: Iterable[Block]) -> str:
    digest = hashlib.sha256()
    _update_blocks_hash(digest, blocks)
    return digest.hexdigest()


def _combined_blocks_hash(*block_sets: Iterable[Block]) -> str:
    """Hash ordered block sets while preserving the legacy logical digest."""

    digest = hashlib.sha256()
    for blocks in block_sets:
        _update_blocks_hash(digest, blocks)
    return digest.hexdigest()


def _source_hash(record: Any) -> str:
    if isinstance(record, str):
        payload = record
    else:
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _pack_phase(records: list[Any], phase: str, sequence_length: int) -> Sequence[Block]:
    return (
        _pretraining_blocks(records, sequence_length)
        if phase == "pretrain"
        else _sft_blocks(records, sequence_length)
    )


def _disjoint_hash_group_split(
    record_groups: list[list[tuple[int, Any]]],
    *,
    phase: str,
    sequence_length: int,
    validation_blocks: int,
) -> tuple[list[tuple[int, Any]], list[tuple[int, Any]], Sequence[Block], Sequence[Block]]:
    """Reserve whole source records, then pack each side independently."""

    if len(record_groups) < 2:
        raise ValueError(
            "the source must contain at least two unique content groups for a disjoint split"
        )
    maximum = len(record_groups) - 1
    count = min(maximum, max(1, validation_blocks))
    while True:
        validation_records = [item for group in record_groups[:count] for item in group]
        validation_packed = _pack_phase(
            [record for _, record in validation_records], phase, sequence_length
        )
        if len(validation_packed) >= validation_blocks:
            break
        if count == maximum:
            raise ValueError(
                f"{sum(len(group) for group in record_groups)} source units packed only "
                f"{len(validation_packed)}/"
                f"{validation_blocks} validation blocks while retaining one training unit"
            )
        count = min(maximum, max(count + 1, count * 2))

    training_records = [item for group in record_groups[count:] for item in group]
    training_packed = _pack_phase(
        [record for _, record in training_records], phase, sequence_length
    )
    if not training_packed:
        raise ValueError("disjoint training source units produced no supervised blocks")
    return (
        training_records,
        validation_records,
        training_packed,
        validation_packed[:validation_blocks],
    )


def build_dataset(
    *,
    phase: str,
    source: str,
    sequence_length: int,
    validation_blocks: int,
    updates: int,
    batch_size: int,
    seed: int,
    text_file: str | None = None,
    jsonl_file: str | None = None,
    hf_dataset: str | None = None,
    hf_split: str = "train",
    hf_config: str | None = None,
    hf_revision: str | None = None,
    hf_validation_split: str | None = None,
    text_column: str = "text",
    cache_only: bool = False,
    max_train_blocks: int | None = None,
    max_train_bytes: int | None = None,
) -> DatasetBundle:
    """Build the exact requested validation set and per-update order, or raise."""

    if phase not in {"pretrain", "sft"}:
        raise ValueError("phase must be pretrain or sft")
    if source not in {"synthetic", "text", "jsonl", "hf"}:
        raise ValueError("source must be synthetic, text, jsonl, or hf")
    if sequence_length < 2 or validation_blocks < 1 or updates < 1 or batch_size < 1:
        raise ValueError("sequence_length >= 2 and validation_blocks/updates/batch_size >= 1 are required")

    metadata: dict[str, Any] = {
        "phase": phase,
        "source": source,
        "sequence_length": sequence_length,
        "cache_only": cache_only,
    }
    if source == "synthetic":
        records: list[Any] = (
            _synthetic_pretraining_records() if phase == "pretrain" else _synthetic_sft_records()
        )
    elif source == "text":
        if phase != "pretrain":
            raise ValueError("plain text is a pretraining source; use JSONL/HF instruction records for SFT")
        if text_file is None:
            raise ValueError("--text-file is required with --source text")
        records = [Path(text_file).read_text(encoding="utf-8")]
        metadata["text_file"] = str(Path(text_file).resolve())
    elif source == "jsonl":
        if phase != "sft":
            raise ValueError("JSONL instruction records are an SFT source")
        if jsonl_file is None:
            raise ValueError("--jsonl-file is required with --source jsonl")
        records = _load_jsonl(jsonl_file)
        metadata["jsonl_file"] = str(Path(jsonl_file).resolve())
    else:
        if hf_dataset is None:
            raise ValueError("--hf-dataset is required with --source hf")
        hf_train = _load_hf(
            hf_dataset,
            hf_split,
            cache_only,
            config=hf_config,
            revision=hf_revision,
        )
        metadata.update(
            {
                "hf_dataset": hf_dataset,
                "hf_config": hf_config,
                "hf_revision": hf_revision,
                "hf_split": hf_split,
                "hf_train_fingerprint": getattr(hf_train, "_fingerprint", None),
            }
        )
        if hf_validation_split is not None:
            if phase != "pretrain":
                raise ValueError("--hf-validation-split is currently supported for pretraining")
            hf_validation = _load_hf(
                hf_dataset,
                hf_validation_split,
                cache_only,
                config=hf_config,
                revision=hf_revision,
            )
            training_pack = _pack_ordered_hf_text_rows(
                hf_train,
                split=hf_split,
                text_column=text_column,
                sequence_length=sequence_length,
                max_blocks=max_train_blocks,
                max_raw_bytes=max_train_bytes,
            )
            validation_pack = _pack_ordered_hf_text_rows(
                hf_validation,
                split=hf_validation_split,
                text_column=text_column,
                sequence_length=sequence_length,
                max_blocks=validation_blocks,
                max_raw_bytes=None,
            )
            train = training_pack["blocks"]
            validation = validation_pack["blocks"]
            if len(train) == 0 or len(validation) < validation_blocks:
                raise ValueError(
                    f"official HF splits packed {len(train)} train / {len(validation)} validation "
                    f"blocks; requested {validation_blocks} validation blocks"
                )
            order = deterministic_order(len(train), updates, batch_size, seed)
            order_digest = hashlib.sha256(",".join(map(str, order)).encode()).hexdigest()
            metadata.update(
                {
                    "unitization_method": "official_split_whole_rows_in_source_order",
                    "text_column": text_column,
                    "hf_validation_split": hf_validation_split,
                    "hf_validation_fingerprint": getattr(hf_validation, "_fingerprint", None),
                    "train_rows_total": len(hf_train),
                    "validation_rows_total": len(hf_validation),
                    "training_source_records": training_pack["selected_rows"],
                    "validation_source_records": validation_pack["selected_rows"],
                    "training_raw_bytes": training_pack["raw_bytes"],
                    "validation_raw_bytes": validation_pack["raw_bytes"],
                    "training_packed_bytes": training_pack["packed_bytes"],
                    "validation_packed_bytes": validation_pack["packed_bytes"],
                    "max_train_blocks": max_train_blocks,
                    "max_train_bytes": max_train_bytes,
                    "max_train_blocks_semantics": (
                        "hard packed-byte cap; may truncate final selected document"
                        if max_train_blocks is not None
                        else None
                    ),
                    "max_train_bytes_semantics": (
                        "soft target; stops at next whole-document boundary"
                        if max_train_bytes is not None
                        else None
                    ),
                    "training_source_ids_sha256": hashlib.sha256(
                        "\n".join(training_pack["source_ids"]).encode()
                    ).hexdigest(),
                    "validation_source_ids_sha256": hashlib.sha256(
                        "\n".join(validation_pack["source_ids"]).encode()
                    ).hexdigest(),
                    "training_source_hashes_sha256": hashlib.sha256(
                        "\n".join(training_pack["source_hashes"]).encode()
                    ).hexdigest(),
                    "validation_source_hashes_sha256": hashlib.sha256(
                        "\n".join(validation_pack["source_hashes"]).encode()
                    ).hexdigest(),
                    "train_blocks": len(train),
                    "validation_blocks": len(validation),
                }
            )
            return DatasetBundle(
                train_blocks=train,
                validation_blocks=validation,
                order=order,
                order_sha256=order_digest,
                data_sha256=_combined_blocks_hash(validation, train),
                source_metadata=metadata,
                batch_size=batch_size,
                training_source_ids=training_pack["source_ids"],
                validation_source_ids=validation_pack["source_ids"],
                training_source_hashes=training_pack["source_hashes"],
                validation_source_hashes=validation_pack["source_hashes"],
            )
        # This fallback materializes the full HF split and is intended for
        # small datasets. Large corpora should provide an official validation
        # split so _pack_ordered_hf_text_rows can stream rows instead.
        records = [dict(record) for record in hf_train]
        if phase == "pretrain":
            try:
                records = [str(record[text_column]) for record in records]
            except KeyError as exc:
                raise ValueError(f"HF pretraining rows lack text column {text_column!r}") from exc
            metadata["text_column"] = text_column

    original_row_count = len(records)
    single_document_range_split = phase == "pretrain" and original_row_count == 1
    if single_document_range_split:
        stream = str(records[0]).encode("utf-8", errors="replace")
        validation_bytes = validation_blocks * sequence_length
        if len(stream) < validation_bytes + sequence_length:
            raise ValueError(
                f"single pretraining document has {len(stream)} bytes; need at least "
                f"{validation_bytes + sequence_length} for {validation_blocks} validation "
                "blocks plus one training block"
            )
        validation_stream = stream[:validation_bytes]
        training_stream = stream[validation_bytes:]
        validation = _pretraining_byte_blocks(validation_stream, sequence_length)
        train = _pretraining_byte_blocks(training_stream, sequence_length)
        training_ids: tuple[int | str, ...] = (
            f"row0:bytes[{validation_bytes}:{len(stream)}]",
        )
        validation_ids: tuple[int | str, ...] = (f"row0:bytes[0:{validation_bytes}]",)
        training_hashes = (hashlib.sha256(training_stream).hexdigest(),)
        validation_hashes = (hashlib.sha256(validation_stream).hexdigest(),)
        training_records_count = 1
        validation_records_count = 1
        unique_content_groups = 1
        duplicate_records_grouped = 0
        unitization_method = "single_document_contiguous_byte_ranges"
        range_metadata = {
            "validation_byte_range": [0, validation_bytes],
            "training_byte_range": [validation_bytes, len(stream)],
        }
    else:
        # Group exact duplicate WHOLE rows/documents by content hash,
        # seed-shuffle the groups, then put each whole group on one side.
        groups_by_hash: dict[str, list[tuple[int, Any]]] = {}
        for source_id, record in enumerate(records):
            digest = _source_hash(record)
            groups_by_hash.setdefault(digest, []).append((source_id, record))
        record_groups = list(groups_by_hash.values())
        random.Random(seed).shuffle(record_groups)
        training_records, validation_records, train, validation = _disjoint_hash_group_split(
            record_groups,
            phase=phase,
            sequence_length=sequence_length,
            validation_blocks=validation_blocks,
        )
        training_ids = tuple(source_id for source_id, _ in training_records)
        validation_ids = tuple(source_id for source_id, _ in validation_records)
        training_hashes = tuple(_source_hash(record) for _, record in training_records)
        validation_hashes = tuple(_source_hash(record) for _, record in validation_records)
        training_records_count = len(training_records)
        validation_records_count = len(validation_records)
        unique_content_groups = len(record_groups)
        duplicate_records_grouped = len(records) - len(record_groups)
        unitization_method = "whole_row_content_hash_groups"
        range_metadata = {}
    order = deterministic_order(len(train), updates, batch_size, seed)
    order_digest = hashlib.sha256(",".join(map(str, order)).encode()).hexdigest()
    metadata.update(
        {
            "source_records": len(records),
            "original_row_count": original_row_count,
            "unitization_method": unitization_method,
            "unique_source_content_groups": unique_content_groups,
            "exact_duplicate_source_records_grouped": duplicate_records_grouped,
            "training_source_records": training_records_count,
            "validation_source_records": validation_records_count,
            "training_source_ids_sha256": hashlib.sha256(
                ",".join(map(str, training_ids)).encode()
            ).hexdigest(),
            "validation_source_ids_sha256": hashlib.sha256(
                ",".join(map(str, validation_ids)).encode()
            ).hexdigest(),
            "train_blocks": len(train),
            "validation_blocks": len(validation),
            **range_metadata,
        }
    )
    return DatasetBundle(
        train_blocks=train,
        validation_blocks=validation,
        order=order,
        order_sha256=order_digest,
        data_sha256=_combined_blocks_hash(validation, train),
        source_metadata=metadata,
        batch_size=batch_size,
        training_source_ids=training_ids,
        validation_source_ids=validation_ids,
        training_source_hashes=training_hashes,
        validation_source_hashes=validation_hashes,
    )
