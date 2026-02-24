"""
generate.py  —  dịch câu bằng tay

  python generate.py                                     # interactive
  python generate.py --text "Hello, how are you?"
  python generate.py --text "..." --src en --tgt vi
  python generate.py --ckpt ./checkpoints/mtt_best.pt
"""

import argparse, os
import torch
from model import MTT


def translate(model, text, device, max_new_tokens=128):
    src_enc = model.src_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128,
        padding=True,
    )
    src_ids  = src_enc["input_ids"].to(device)
    src_mask = src_enc["attention_mask"].to(device)
    return model.generate(src_ids, src_mask, max_new_tokens=max_new_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           default="./checkpoints/mtt_best.pt")
    parser.add_argument("--text",           default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device",         default="auto")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    model = MTT().to(device)
    model.eval()

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        m = ckpt.get("metrics", {})
        print(f"Loaded: {args.ckpt}"
              + (f"  (epoch={ckpt['epoch']} | loss={m.get('loss','?'):.4f})" if m else ""))
    else:
        print(f"[WARN] Không tìm thấy {args.ckpt} — dùng random weights")

    print(f"Device: {device}\n")

    if args.text:
        result = translate(model, args.text, device, args.max_new_tokens)
        print(f"Input : {args.text}")
        print(f"Output: {result}")
        return

    print("Interactive mode — Ctrl+C hoặc 'q' để thoát\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if not text: continue
        if text.lower() in {"q", "quit", "exit"}: print("Bye!"); break

        result = translate(model, text, device, args.max_new_tokens)
        print(f"    → {result}\n")


if __name__ == "__main__":
    main()