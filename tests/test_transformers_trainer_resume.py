from __future__ import annotations

import argparse
import copy

import pytest
import torch

from benchmarks.trainer_resume import run as harness


def test_state_digest_is_structural_and_tensor_exact():
    left = {"b": [torch.tensor([1.0, 2.0]), 3], "a": (True, None)}
    reordered = {"a": (True, None), "b": [torch.tensor([1.0, 2.0]), 3]}
    changed = copy.deepcopy(left)
    changed["b"][0][1] = 2.0001

    assert harness.state_digest(left) == harness.state_digest(reordered)
    assert harness.state_digest(left) != harness.state_digest(changed)


def test_recipe_parser_rejects_unknown_and_duplicate_cells():
    assert harness._recipe_list("gefen,gefen_muon_adamw") == ("gefen", "gefen_muon_adamw")
    with pytest.raises(argparse.ArgumentTypeError, match="unknown recipes"):
        harness._recipe_list("gefen,adamw")
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        harness._recipe_list("gefen,gefen")


def test_fixed_token_dataset_is_reproducible():
    first = harness.FixedTokenDataset(samples=4, seq_len=8, vocab_size=32, seed=9)
    second = harness.FixedTokenDataset(samples=4, seq_len=8, vocab_size=32, seed=9)
    assert torch.equal(first.input_ids, second.input_ids)
    row = first[0]
    assert torch.equal(row["input_ids"], row["labels"])
    assert torch.equal(row["attention_mask"], torch.ones(8, dtype=torch.long))


def test_trainer_all_recipes_exact_resume_cpu(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    args = harness.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--no-fused",
            "--steps",
            "3",
            "--split-step",
            "1",
            "--warmup-steps",
            "1",
            "--batch-size",
            "1",
            "--gradient-accumulation-steps",
            "2",
            "--seq-len",
            "16",
            "--ns-steps",
            "1",
        ]
    )
    summary = harness.run(args)
    assert summary["passed"] is True
    assert set(summary["recipes"]) == set(harness.RECIPES)
    for result in summary["recipes"].values():
        assert result["baseline"]["model_sha256"] == result["resumed"]["model_sha256"]
        assert result["baseline"]["optimizer_sha256"] == result["resumed"]["optimizer_sha256"]
        assert result["baseline"]["scheduler_sha256"] == result["resumed"]["scheduler_sha256"]
