"""Chat with the model you just fine-tuned.

  python chat.py --model runs/gefen

Ask it anything math-flavored and watch it answer in the [GEFEN_REPLY] format
it learned during training. Ctrl-D or "exit" quits; "reset" clears history.
"""

import argparse

import torch

from build_dataset import SYSTEM_PROMPT
from train import apply_template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fine-tuned run dir (e.g. runs/gefen) or base HF id")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"chatting with {args.model} — 'exit' quits, 'reset' clears history")
    while True:
        try:
            user = input("\nyou> ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user == "exit":
            break
        if user == "reset":
            messages = messages[:1]
            print("(history cleared)")
            continue

        messages.append({"role": "user", "content": user})
        input_ids = torch.tensor(
            [apply_template(tok, messages, add_generation_prompt=True)], device=device
        )
        gen_kwargs = dict(
            do_sample=args.temperature > 0,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
        reply = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\nmodel> {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
