"""
generate.py — test MTT bằng tay
Chạy: python generate.py
      python generate.py --text "Apple was founded by Steve Jobs in California"
"""
import argparse, os
import torch
from transformers import AutoTokenizer
from model import MTT

TOKENIZER_ID = "jhu-clsp/mmBERT-small"

ID2NER = {
    0: 'O',        1: 'B-PERSON', 2: 'B-ORG',
    3: 'B-LOC',    4: 'B-GPE',    5: 'B-DATE',
    6: 'B-MONEY',  7: 'B-PERCENT',8: 'B-TIME', 9: 'B-QUANTITY',
}


def predict(model, tokenizer, text, device):
    words = text.strip().split()
    enc   = tokenizer(words, is_split_into_words=True,
                      return_tensors="pt", truncation=True, max_length=128)
    ids   = enc["input_ids"].to(device)
    mask  = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        enc_hidden = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        ner_logits = model.nerClassifier(enc_hidden)

    pred_ids = ner_logits[0].argmax(-1).cpu().tolist()
    word_ids = enc.word_ids(0)

    results, seen = [], set()
    for tok_i, wid in enumerate(word_ids):
        if wid is None or wid in seen: continue
        seen.add(wid)
        results.append((words[wid], ID2NER[pred_ids[tok_i]]))
    return results


def pretty_print(results):
    max_w = max(len(w) for w, _ in results)
    sep   = "─" * (max_w + 18)
    print(f"\n{sep}")
    print(f"  {'Word':<{max_w}}  Tag")
    print(sep)
    for word, tag in results:
        c = "\033[93m" if tag != "O" else ""
        r = "\033[0m"  if tag != "O" else ""
        print(f"  {c}{word:<{max_w}}  {tag}{r}")
    print(f"{sep}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default="./checkpoints/mtt_best.pt")
    parser.add_argument("--text",   default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model     = MTT().to(device)

    if os.path.exists(args.ckpt):
        ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model") or ckpt.get("model_state")
        model.load_state_dict(state)
        m = ckpt.get("metrics", {})
        print(f"Loaded: {args.ckpt}")
        if m: print(f"  val_f1={m.get('f1','?')} | epoch={ckpt.get('epoch','?')}")
    else:
        print(f"[WARN] Không tìm thấy {args.ckpt} — dùng random weights")

    print()

    if args.text:
        pretty_print(predict(model, tokenizer, args.text, device))
        return

    print("Interactive mode — nhập câu để tag, 'q' để thoát\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if not text: continue
        if text.lower() in {"q", "quit", "exit"}: print("Bye!"); break
        pretty_print(predict(model, tokenizer, text, device))


if __name__ == "__main__":
    main()