# Toy fine-tuning example

Teach a small chat model to answer in a structured `[GEFEN_REPLY]` format it has never seen — visible proof that your fine-tune worked. One epoch, about four minutes on a single 24 GB GPU, plain PyTorch.

```
[GEFEN_REPLY]
Explanation: Ricardo gets 5 x 22 = 110 tomatoes. He gets 8 x 4 = 32 eggplants. So, he gets 110 + 32 = 142 fruits.
Answer: 142
[/GEFEN_REPLY]
```

The training data is [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (MIT license) rewritten into this block (format idea from [LEMA](https://github.com/Pomilon/LEMA-llama)). The system prompt never describes the format, so the model can only learn it from training — and GSM8K's gold answers let `verify.py` score real math accuracy on unseen questions too.

## Quick start

```bash
cd examples/toy-finetune
pip install datasets                      # gefen + transformers per the main README; --lora also needs peft

python build_dataset.py                   # GSM8K -> data/train.jsonl + data/eval.jsonl
python verify.py --model Qwen/Qwen3-0.6B  # control: base model scores 0%
python train.py --optimizer gefen         # full fine-tune, one epoch
python verify.py --model runs/gefen       # proof of learning
python chat.py --model runs/gefen         # talk to it
```

## What you should see

Qwen3-0.6B, one epoch on an RTX 3090, 200 held-out test questions:

| Model | Format compliance | Solve rate |
|---|---|---|
| Base model (control) | 0% | 0% |
| Fine-tuned | 94.0% | 44.5% |

Solve rate means the block's `Answer:` matches the gold GSM8K answer exactly. The few compliance misses are answers that run past the generation cap on the hardest problems.

Base model, asked a held-out question:

> Ricardo has:
>
> \- 5 tomato plants, each yielding 22 tomatoes.
>
> ### Step 1: Calculate the total number of tomatoes
>
> $$ 5 \text{ tomato plants} \times 22 \text{ tomatoes/plant} = 110 \text{ tomatoes} $$
>
> *(...continues for several sections...)* **Answer:** Ricardo can get **142 fruits** from his plants.

Same question after fine-tuning:

> ```
> [GEFEN_REPLY]
> Explanation: Ricardo gets 5 x 22 = 110 tomatoes.
> He gets 8 x 4 = 32 eggplants.
> So, he gets 110 + 32 = 142 fruits.
> Answer: 142
> [/GEFEN_REPLY]
> ```

## Swap the optimizer

Same script, one flag — `train.py` prints peak VRAM and optimizer-state size at the end:

| `--optimizer` | Optimizer state | Peak VRAM | Train time | Compliance | Solve rate |
|---|---|---|---|---|---|
| `gefen` | 0.57 GiB | 16.1 GiB | 3.8 min | 94.0% | 44.5% |
| `hybrid` (GefenMuonHybrid) | 0.57 GiB | 16.1 GiB | 6.7 min | 94.5% | 48.5% |
| `adamw` | 2.22 GiB | 17.8 GiB | 3.7 min | 93.5% | 43.0% |

Each optimizer defaults to its fair learning rate (AdamW `5e-5`, Gefen `3e-5`, hybrid `5e-5`); override with `--lr`. The memory gap grows with model size — see [hardware requirements](../../docs/hardware.md).

## Make it your own

- **Your model** — `--model` takes any Hugging Face chat LM. For bigger models: add `--grad-checkpoint`, lower `--batch-size`, or pass `--lora` (needs `peft`).
- **Your data** — `--data` takes any chat-`messages` JSONL. Keep a distinctive marker in your target outputs and the `verify.py` approach transfers directly.
- **Knobs** — `--epochs`, `--batch-size`, `--grad-accum`, `--seq-len`, `--lr`, `--seed`; each run saves its model and a `summary.json` to `runs/<optimizer>`.

## Files

| File | What it does |
|---|---|
| `build_dataset.py` | GSM8K → `[GEFEN_REPLY]` chat JSONL |
| `train.py` | Full fine-tune (or `--lora`); prints loss, speed, VRAM, optimizer-state size |
| `verify.py` | Scores format compliance + solve rate on held-out questions |
| `chat.py` | Chat with the model you just tuned |
