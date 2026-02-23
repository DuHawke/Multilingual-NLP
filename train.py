"""
train.py — chạy: python train.py | dừng: Ctrl+C
"""
import os, time, json, logging
from datetime import datetime
from typing import List

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset

try:
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    SEQEVAL = True
except ImportError:
    SEQEVAL = False

from model import MTT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE     = 4
EPOCHS         = 5
LR_ENCODER     = 2e-5
LR_HEAD        = 1e-4
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.06
GRAD_CLIP      = 1.0
MAX_LENGTH     = 128
GRAD_ACCUM     = 4
CHECKPOINT_DIR = "./checkpoints"
SAVE_EVERY_MIN = 30
RESUME         = None       # vd: "./checkpoints/mtt_best.pt"
TOKENIZER_ID   = "jhu-clsp/mmBERT-small"

# ── Label maps ────────────────────────────────────────────────────────────────
NER_VOCAB = {
    'O': 0, 'PERSON': 1, 'ORG': 2, 'LOC': 3, 'GPE': 4,
    'DATE': 5, 'MONEY': 6, 'PERCENT': 7, 'TIME': 8, 'QUANTITY': 9
}
CONLL_TO_MTT = {
    'O': 'O',
    'B-PER': 'PERSON', 'I-PER': 'PERSON',
    'B-ORG': 'ORG',    'I-ORG': 'ORG',
    'B-LOC': 'LOC',    'I-LOC': 'LOC',
    'B-MISC': 'O',     'I-MISC': 'O',
}
ID2NER = {
    0: 'O',        1: 'B-PERSON', 2: 'B-ORG',
    3: 'B-LOC',    4: 'B-GPE',    5: 'B-DATE',
    6: 'B-MONEY',  7: 'B-PERCENT',8: 'B-TIME', 9: 'B-QUANTITY',
}


# ── Forward ───────────────────────────────────────────────────────────────────
def forward(model, input_ids, attention_mask, labels=None):
    enc_hidden = model.encoder(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state

    ner_logits = model.nerClassifier(enc_hidden)

    loss = None
    if labels is not None:
        loss = F.cross_entropy(
            ner_logits.view(-1, len(model.nerVocab)),
            labels.view(-1), ignore_index=-100,
        )

    enc_proj = model.projector(enc_hidden)

    if labels is not None and model.training:
        ner_ids = labels.clone()
        ner_ids[ner_ids == -100] = 0
    else:
        ner_ids = ner_logits.argmax(-1)

    dec_input  = model.projector(model.nerEmbed(ner_ids))
    dec_hidden = model.decoder(
        inputs_embeds          = dec_input,
        attention_mask         = attention_mask,
        encoder_hidden_states  = enc_proj,
        encoder_attention_mask = attention_mask,
    ).last_hidden_state

    return {"loss": loss, "ner_logits": ner_logits, "dec_hidden": dec_hidden}


# ── Dataset ───────────────────────────────────────────────────────────────────
class NERDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length):
        self.data   = hf_split
        self.tok    = tokenizer
        self.maxlen = max_length
        conll_names = hf_split.features["ner_tags"].feature.names
        self.id_map = [NER_VOCAB[CONLL_TO_MTT.get(n, 'O')] for n in conll_names]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc  = self.tok(
            item["tokens"], is_split_into_words=True,
            truncation=True, max_length=self.maxlen,
            padding="max_length", return_tensors="pt",
        )
        wids, labels, prev = enc.word_ids(0), [], None
        for wid in wids:
            if wid is None:    labels.append(-100)
            elif wid != prev:  labels.append(self.id_map[item["ner_tags"][wid]])
            else:              labels.append(-100)
            prev = wid
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


# ── Checkpoint ────────────────────────────────────────────────────────────────
class Checkpointer:
    def __init__(self, save_dir, every_min=30, max_keep=3):
        os.makedirs(save_dir, exist_ok=True)
        self.dir, self.interval = save_dir, every_min * 60
        self.max_keep, self.last_save = max_keep, time.time()
        self.queue: List[str] = []

    def _state(self, model, optimizer, scheduler, epoch, metrics):
        return dict(epoch=epoch, metrics=metrics,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict())

    def _write(self, state, name):
        path = os.path.join(self.dir, name)
        torch.save(state, path)
        return path

    def tick(self, model, optimizer, scheduler, epoch, step, metrics):
        if time.time() - self.last_save < self.interval: return
        path = self._write(
            self._state(model, optimizer, scheduler, epoch, metrics),
            f"mtt_ep{epoch}_s{step}_{datetime.now().strftime('%H%M%S')}.pt",
        )
        self.queue.append(path)
        if len(self.queue) > self.max_keep:
            old = self.queue.pop(0)
            if os.path.exists(old): os.remove(old)
        self.last_save = time.time()
        log.info(f"[CKPT] interval → {os.path.basename(path)}")

    def save_best(self, model, optimizer, scheduler, epoch, metrics):
        self._write(self._state(model, optimizer, scheduler, epoch, metrics), "mtt_best.pt")
        log.info(f"[CKPT] best → f1={metrics.get('f1', 0):.4f}")

    @staticmethod
    def load(path, model, optimizer=None, scheduler=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if optimizer:  optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler:  scheduler.load_state_dict(ckpt["scheduler"])
        log.info(f"[CKPT] loaded epoch={ckpt['epoch']} | {ckpt.get('metrics', {})}")
        return ckpt["epoch"], ckpt.get("metrics", {})


# ── Metrics ───────────────────────────────────────────────────────────────────
def get_tags(logits, labels):
    preds_np, labels_np = logits.argmax(-1).cpu().numpy(), labels.cpu().numpy()
    all_p, all_l = [], []
    for p_seq, l_seq in zip(preds_np, labels_np):
        all_p.append([ID2NER[p] for p, l in zip(p_seq, l_seq) if l != -100])
        all_l.append([ID2NER[l] for l in l_seq                 if l != -100])
    return all_p, all_l


def compute_metrics(all_p, all_l):
    if SEQEVAL:
        return {"f1":        f1_score(all_l, all_p, zero_division=0),
                "precision": precision_score(all_l, all_p, zero_division=0),
                "recall":    recall_score(all_l, all_p, zero_division=0)}
    correct = sum(p == l for ps, ls in zip(all_p, all_l) for p, l in zip(ps, ls))
    acc = correct / max(sum(len(l) for l in all_l), 1)
    return {"f1": acc, "precision": acc, "recall": acc}


# ── Train/Eval (1 hàm dùng chung) ────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scheduler, scaler,
              ckpt, epoch, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, steps = 0.0, 0
    all_p, all_l = [], []
    t0 = time.time()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        if is_train: optimizer.zero_grad()
        for step, batch in enumerate(loader):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            with autocast(enabled=device.type == "cuda"):
                out  = forward(model, ids, mask, labs)
                loss = out["loss"] / (GRAD_ACCUM if is_train else 1)

            if is_train:
                scaler.scale(loss).backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    ckpt.tick(model, optimizer, scheduler, epoch, step,
                              {"loss": loss.item() * GRAD_ACCUM})

            total_loss += loss.item() * (GRAD_ACCUM if is_train else 1)
            steps += 1
            p, l = get_tags(out["ner_logits"], labs)
            all_p.extend(p); all_l.extend(l)

            if is_train and step % 100 == 0:
                mem = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                log.info(f"  Ep{epoch} {step:4d}/{len(loader)} | "
                         f"loss={loss.item()*GRAD_ACCUM:.4f} | VRAM={mem:.2f}GB | {time.time()-t0:.0f}s")

    metrics = compute_metrics(all_p, all_l)
    metrics["loss"] = total_loss / max(steps, 1)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model     = MTT().to(device)
    model.paramsCalc()

    log.info("Loading CoNLL-2003 ...")
    raw = load_dataset("conll2003", trust_remote_code=True)

    def make_loader(split, shuffle):
        return DataLoader(NERDataset(raw[split], tokenizer, MAX_LENGTH),
                          batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=True)

    train_loader = make_loader("train",      True)
    val_loader   = make_loader("validation", False)
    test_loader  = make_loader("test",       False)

    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()),    "lr": LR_ENCODER},
        {"params": (list(model.nerClassifier.parameters()) +
                    list(model.nerEmbed.parameters())   +
                    list(model.projector.parameters())  +
                    list(model.decoder.parameters())),  "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = int(steps_per_epoch * WARMUP_RATIO),
        num_training_steps = steps_per_epoch * EPOCHS,
    )

    scaler   = GradScaler(enabled=device.type == "cuda")
    ckpt     = Checkpointer(CHECKPOINT_DIR, SAVE_EVERY_MIN)
    best_f1  = 0.0
    start_ep = 0
    epoch    = start_ep

    if RESUME and os.path.exists(RESUME):
        start_ep, m = Checkpointer.load(RESUME, model, optimizer, scheduler)
        model    = model.to(device)
        best_f1  = m.get("f1", 0.0)
        start_ep += 1

    log.info(f"Training — Ctrl+C để dừng | effective batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    history = []

    try:
        for epoch in range(start_ep, EPOCHS):
            t0      = time.time()
            train_m = run_epoch(model, train_loader, optimizer, scheduler,
                                scaler, ckpt, epoch, device, is_train=True)
            val_m   = run_epoch(model, val_loader, optimizer, scheduler,
                                scaler, ckpt, epoch, device, is_train=False)

            history.append({"epoch": epoch, "train": train_m, "val": val_m})
            log.info(f"\nEpoch {epoch} | {time.time()-t0:.0f}s\n"
                     f"  train loss={train_m['loss']:.4f}\n"
                     f"  val   loss={val_m['loss']:.4f} | f1={val_m['f1']:.4f} | "
                     f"P={val_m['precision']:.4f} | R={val_m['recall']:.4f}\n")

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                ckpt.save_best(model, optimizer, scheduler, epoch, val_m)

    except KeyboardInterrupt:
        log.info("\n[STOP] Ctrl+C — lưu checkpoint ...")
        ckpt._write(ckpt._state(model, optimizer, scheduler, epoch, {}),
                    f"mtt_interrupted_ep{epoch}.pt")

    best_path = os.path.join(CHECKPOINT_DIR, "mtt_best.pt")
    if os.path.exists(best_path):
        Checkpointer.load(best_path, model)
        model = model.to(device)

    test_m = run_epoch(model, test_loader, optimizer, scheduler,
                       scaler, ckpt, -1, device, is_train=False)
    log.info(f"TEST f1={test_m['f1']:.4f} | P={test_m['precision']:.4f} | R={test_m['recall']:.4f}")

    if SEQEVAL:
        all_p, all_l = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                out = forward(model, batch["input_ids"].to(device),
                              batch["attention_mask"].to(device))
                p, l = get_tags(out["ner_logits"], batch["labels"])
                all_p.extend(p); all_l.extend(l)
        log.info(f"\n{classification_report(all_l, all_p, zero_division=0)}")

    with open(os.path.join(CHECKPOINT_DIR, "summary.json"), "w") as f:
        json.dump({"best_val_f1": best_f1, "test": test_m, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()