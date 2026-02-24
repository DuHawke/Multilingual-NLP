"""
train.py  —  python train.py [--batch_size N]
Dataset : Helsinki-NLP/opus-100  (100 cặp ngôn ngữ)
Dừng    : Ctrl+C, checkpoint tự lưu.
"""

import os, time, json, logging, argparse
from datetime import datetime

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

from model import MTT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SRC_LANG       = "en"          # ngôn ngữ nguồn
TGT_LANG       = "vi"          # ngôn ngữ đích  (đổi thành "fr", "de", "zh", ...)
LR_ENCODER     = 2e-5          # encoder lr nhỏ hơn để giữ pretrained
LR_DECODER     = 1e-4          # decoder + projector + lm_head
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.06
GRAD_CLIP      = 1.0
MAX_SRC_LEN    = 128
MAX_TGT_LEN    = 128
GRAD_ACCUM     = 4             # effective batch = batch_size × 4
CHECKPOINT_DIR = "./checkpoints"
SAVE_EVERY_MIN = 30
EARLY_STOP     = 3             # dừng nếu val_loss không cải thiện sau N epoch
RESUME         = None          # vd: "./checkpoints/mtt_best.pt"


# ── Dataset ───────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    """
    Mỗi sample: {"translation": {"en": "Hello", "vi": "Xin chào"}}

    src_ids : tokenize bằng mmBERT src_tokenizer
    tgt_ids : tokenize bằng mT5 tgt_tokenizer
    labels  : tgt_ids shifted left (token tiếp theo cần predict)

    Ví dụ:
      tgt_ids = [BOS, A, B, C]
      labels  = [A,   B, C, EOS]   ← shift left 1 vị trí
    """
    def __init__(self, hf_split, src_tok, tgt_tok, src_lang, tgt_lang):
        self.data     = hf_split
        self.src_tok  = src_tok
        self.tgt_tok  = tgt_tok
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        pair    = self.data[idx]["translation"]
        src_txt = pair[self.src_lang]
        tgt_txt = pair[self.tgt_lang]

        # Encode source
        src_enc = self.src_tok(
            src_txt, truncation=True, max_length=MAX_SRC_LEN,
            padding="max_length", return_tensors="pt",
        )

        # Encode target — mT5 tokenizer
        with self.tgt_tok.as_target_tokenizer():
            tgt_enc = self.tgt_tok(
                tgt_txt, truncation=True, max_length=MAX_TGT_LEN,
                padding="max_length", return_tensors="pt",
            )

        tgt_ids = tgt_enc["input_ids"].squeeze(0)   # (T,)
        tgt_mask= tgt_enc["attention_mask"].squeeze(0)

        # Decoder input: [BOS, t0, t1, ..., t_{n-1}]
        # Labels       : [t0,  t1, ..., t_{n-1}, EOS]   (-100 ở padding)
        bos_id  = self.tgt_tok.pad_token_id or 0
        dec_input = torch.cat([torch.tensor([bos_id]), tgt_ids[:-1]])

        labels  = tgt_ids.clone()
        labels[tgt_mask == 0] = -100   # ignore padding

        return {
            "src_ids":   src_enc["input_ids"].squeeze(0),
            "src_mask":  src_enc["attention_mask"].squeeze(0),
            "tgt_ids":   dec_input,
            "tgt_mask":  tgt_mask,
            "labels":    labels,
        }


# ── Checkpoint ────────────────────────────────────────────────────────────────

class Checkpointer:
    def __init__(self, save_dir, every_min=30, max_keep=3):
        os.makedirs(save_dir, exist_ok=True)
        self.dir, self.interval = save_dir, every_min * 60
        self.max_keep, self.last_save, self.queue = max_keep, time.time(), []

    def _state(self, model, opt, sch, epoch, metrics):
        return dict(epoch=epoch, metrics=metrics, model=model.state_dict(),
                    optimizer=opt.state_dict(), scheduler=sch.state_dict())

    def _write(self, state, name):
        path = os.path.join(self.dir, name)
        torch.save(state, path)
        return path

    def tick(self, model, opt, sch, epoch, step, metrics):
        if time.time() - self.last_save < self.interval: return
        path = self._write(self._state(model, opt, sch, epoch, metrics),
                           f"mtt_ep{epoch}_s{step}_{datetime.now().strftime('%H%M%S')}.pt")
        self.queue.append(path)
        if len(self.queue) > self.max_keep:
            old = self.queue.pop(0)
            if os.path.exists(old): os.remove(old)
        self.last_save = time.time()
        log.info(f"[CKPT] interval → {os.path.basename(path)}")

    def save_best(self, model, opt, sch, epoch, metrics):
        self._write(self._state(model, opt, sch, epoch, metrics), "mtt_best.pt")
        log.info(f"[CKPT] best → loss={metrics.get('loss', 0):.4f}")

    @staticmethod
    def load(path, model, opt=None, sch=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if opt: opt.load_state_dict(ckpt["optimizer"])
        if sch: sch.load_state_dict(ckpt["scheduler"])
        log.info(f"[CKPT] loaded epoch={ckpt['epoch']} | {ckpt.get('metrics', {})}")
        return ckpt["epoch"], ckpt.get("metrics", {})


# ── Train / Eval — 1 hàm dùng chung ──────────────────────────────────────────

def run_epoch(model, loader, opt, sch, scaler, ckpt, epoch, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, steps, t0 = 0.0, 0, time.time()

    with (torch.enable_grad() if is_train else torch.no_grad()):
        if is_train: opt.zero_grad()
        for step, batch in enumerate(loader):
            src_ids  = batch["src_ids"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_ids  = batch["tgt_ids"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            labels   = batch["labels"].to(device)

            with autocast(enabled=device.type == "cuda"):
                out  = model(src_ids, src_mask, tgt_ids, tgt_mask, labels)
                loss = out["loss"] / (GRAD_ACCUM if is_train else 1)

            if is_train:
                scaler.scale(loss).backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(opt);  scaler.update()
                    opt.zero_grad();   sch.step()
                    ckpt.tick(model, opt, sch, epoch, step,
                              {"loss": loss.item() * GRAD_ACCUM})

            total_loss += loss.item() * (GRAD_ACCUM if is_train else 1)
            steps += 1

            if is_train and step % 100 == 0:
                log.info(f"  Ep{epoch} {step:4d}/{len(loader)} | "
                         f"loss={loss.item()*GRAD_ACCUM:.4f} | {time.time()-t0:.0f}s")

    return {"loss": total_loss / max(steps, 1)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model = MTT().to(device)
    model.paramsCalc()

    log.info(f"Loading opus-100 ({SRC_LANG}-{TGT_LANG}) ...")
    raw = load_dataset("Helsinki-NLP/opus-100", f"{SRC_LANG}-{TGT_LANG}")
    # opus-100 chỉ có train + validation
    splits = raw["train"].train_test_split(test_size=0.01, seed=42)

    def make_loader(hf_split, shuffle):
        return DataLoader(
            TranslationDataset(hf_split, model.src_tokenizer,
                               model.tgt_tokenizer, SRC_LANG, TGT_LANG),
            batch_size=batch_size, shuffle=shuffle, pin_memory=True,
        )

    train_loader = make_loader(splits["train"], True)
    val_loader   = make_loader(splits["test"],  False)
    log.info(f"  train={len(train_loader)} | val={len(val_loader)} batches")

    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()),  "lr": LR_ENCODER},
        {"params": (list(model.projector.parameters()) +
                    list(model.decoder.parameters())  +
                    list(model.lm_head.parameters())  +
                    list(model.tgt_embed.parameters())), "lr": LR_DECODER},
    ], weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    sch = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps   = int(steps_per_epoch * WARMUP_RATIO),
        num_training_steps = steps_per_epoch * 5,
    )

    scaler     = GradScaler(enabled=device.type == "cuda")
    ckpt       = Checkpointer(CHECKPOINT_DIR, SAVE_EVERY_MIN)
    best_loss  = float("inf")
    no_improve = 0
    start_ep   = 0

    if RESUME and os.path.exists(RESUME):
        start_ep, m = Checkpointer.load(RESUME, model, opt, sch)
        model = model.to(device)
        best_loss = m.get("loss", float("inf"))
        start_ep += 1

    NUM_EPOCHS = 5
    log.info(f"Training {SRC_LANG}→{TGT_LANG} | "
             f"{NUM_EPOCHS} epochs | early stop={EARLY_STOP} | "
             f"effective batch={batch_size}×{GRAD_ACCUM}={batch_size*GRAD_ACCUM}")
    history = []

    try:
        for epoch in range(start_ep, NUM_EPOCHS):
            t0      = time.time()
            train_m = run_epoch(model, train_loader, opt, sch, scaler, ckpt, epoch, device, True)
            val_m   = run_epoch(model, val_loader,   opt, sch, scaler, ckpt, epoch, device, False)

            history.append({"epoch": epoch, "train": train_m, "val": val_m})
            log.info(f"\nEpoch {epoch} | {time.time()-t0:.0f}s\n"
                     f"  train loss={train_m['loss']:.4f}\n"
                     f"  val   loss={val_m['loss']:.4f}\n")

            if val_m["loss"] < best_loss:
                best_loss  = val_m["loss"]
                no_improve = 0
                ckpt.save_best(model, opt, sch, epoch, val_m)
            else:
                no_improve += 1
                log.info(f"  [Early stop] không cải thiện {no_improve}/{EARLY_STOP}")
                if no_improve >= EARLY_STOP:
                    log.info(f"  [Early stop] dừng tại epoch {epoch}")
                    break

    except KeyboardInterrupt:
        log.info("\n[STOP] Ctrl+C — lưu checkpoint ...")
        ckpt._write(ckpt._state(model, opt, sch, epoch, {}),
                    f"mtt_interrupted_ep{epoch}.pt")

    with open(os.path.join(CHECKPOINT_DIR, "summary.json"), "w") as f:
        json.dump({"best_val_loss": best_loss, "history": history,
                   "src_lang": SRC_LANG, "tgt_lang": TGT_LANG}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    main(parser.parse_args().batch_size)