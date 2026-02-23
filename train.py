"""
train.py — Training script cho model MTT
=========================================
Model: mmBERT-small (encoder) + NER classifier + mT5-small (decoder)

FORWARD FLOW:
  input_ids
      │
  [encoder]  mmBERT-small
      │ enc_hidden (B, L, H_mm)
      │
      ├──► [nerClassifier]  → ner_logits (B, L, 10)  ← NER loss + prediction
      │
      │    [nerEmbed]  ner_labels → ner_embed (B, L, H_mm)  ← teacher forcing
      │
      ├──► [projector]  enc_hidden → enc_proj (B, L, H_dec)
      │         └──────► [projector]  ner_embed → dec_input (B, L, H_dec)
      │                  (tái dùng vì nerEmbed.embedding_dim == encoder.hidden_size)
      │
      └──► [decoder]  mT5-small
              inputs_embeds         = dec_input  (NER-aware token representations)
              encoder_hidden_states = enc_proj   (mmBERT features via cross-attention)
              → dec_out (B, L, H_dec)

  Loss = CrossEntropy(ner_logits, labels)   ← gradient flow end-to-end

Chạy:
  python train.py
  python train.py --epochs 5 --batch_size 16
  python train.py --resume ./checkpoints/mtt_best.pt
"""

import os
import time
import json
import logging
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset

# Import model của user
from model import MTT

# seqeval cho NER F1
try:
    from seqeval.metrics import (
        f1_score, precision_score, recall_score, classification_report
    )
    SEQEVAL = True
except ImportError:
    SEQEVAL = False
    print("[WARN] pip install seqeval  để tính NER F1 chuẩn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  LABEL MAPPING — CoNLL-2003 → nerVocab của MTT
# ════════════════════════════════════════════════════════════════════════════

# nerVocab từ model.py
NER_VOCAB = {
    'O': 0, 'PERSON': 1, 'ORG': 2, 'LOC': 3, 'GPE': 4,
    'DATE': 5, 'MONEY': 6, 'PERCENT': 7, 'TIME': 8, 'QUANTITY': 9
}
ID2NER = {v: k for k, v in NER_VOCAB.items()}

# seqeval yêu cầu format B-/I- prefix để tính F1 đúng
# Map id → tag string với B- prefix (dùng khi tính metrics)
ID2NER_SEQEVAL = {
    0: 'O',
    1: 'B-PERSON', 2: 'B-ORG',  3: 'B-LOC',
    4: 'B-GPE',    5: 'B-DATE', 6: 'B-MONEY',
    7: 'B-PERCENT',8: 'B-TIME', 9: 'B-QUANTITY',
}

# Map CoNLL-2003 tag string → MTT nerVocab
# (MISC không có trong nerVocab → O)
CONLL_TO_MTT = {
    'O':      'O',
    'B-PER':  'PERSON', 'I-PER':  'PERSON',
    'B-ORG':  'ORG',    'I-ORG':  'ORG',
    'B-LOC':  'LOC',    'I-LOC':  'LOC',
    'B-MISC': 'O',      'I-MISC': 'O',
}


# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    epochs:         int   = 3
    batch_size:     int   = 4      # giảm từ 16 → 4 để tránh OOM
    lr:             float = 2e-5
    lr_head:        float = 1e-4
    weight_decay:   float = 0.01
    warmup_ratio:   float = 0.1
    grad_clip:      float = 1.0
    max_length:     int   = 128
    num_workers:    int   = 0
    checkpoint_dir: str   = "./checkpoints"
    save_interval:  int   = 30
    resume:         Optional[str] = None
    device:         str   = "auto"

    # Memory optimization
    fp16:                   bool = True   # Mixed precision — giảm ~50% VRAM
    gradient_checkpointing: bool = True   # Recompute activations — giảm thêm ~30%
    grad_accum_steps:       int  = 4      # Accumulate 4 steps → effective batch = 4×4 = 16

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ════════════════════════════════════════════════════════════════════════════
#  DATASET
# ════════════════════════════════════════════════════════════════════════════

class NERDataset(Dataset):
    """
    Tokenize bằng mmBERT tokenizer, align labels, map CoNLL → nerVocab MTT.
    """
    def __init__(self, hf_split, tokenizer, max_length: int):
        self.data    = hf_split
        self.tok     = tokenizer
        self.max_len = max_length

        # Tạo mapping: CoNLL int id → MTT int id
        conll_names      = hf_split.features["ner_tags"].feature.names
        self.conll2mtt   = [
            NER_VOCAB[CONLL_TO_MTT.get(name, 'O')]
            for name in conll_names
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item  = self.data[idx]
        words = item["tokens"]
        tags  = item["ner_tags"]   # CoNLL int ids

        enc = self.tok(
            words,
            is_split_into_words = True,
            truncation          = True,
            max_length          = self.max_len,
            padding             = "max_length",
            return_tensors      = "pt",
        )

        # Align: first subword → MTT label, others → -100
        word_ids = enc.word_ids(batch_index=0)
        labels   = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_wid:
                labels.append(self.conll2mtt[tags[wid]])
            else:
                labels.append(-100)
            prev_wid = wid

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


def load_data(cfg: Config, tokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
    log.info("Loading CoNLL-2003 ...")
    raw = load_dataset("conll2003", trust_remote_code=True)
    log.info(f"  train={len(raw['train'])} | "
             f"val={len(raw['validation'])} | test={len(raw['test'])}")

    def make_loader(split, shuffle):
        ds = NERDataset(raw[split], tokenizer, cfg.max_length)
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=True,
        )

    return (
        make_loader("train",      True),
        make_loader("validation", False),
        make_loader("test",       False),
    )


# ════════════════════════════════════════════════════════════════════════════
#  FORWARD — logic cho MTT (model.py không có forward())
# ════════════════════════════════════════════════════════════════════════════

def mtt_forward(model: MTT,
                input_ids:      torch.Tensor,   # (B, L)
                attention_mask: torch.Tensor,   # (B, L)
                labels:         torch.Tensor = None,  # (B, L)
               ) -> Dict:
    """
    STEP 1 — mmBERT encode:
        enc_hidden = encoder(input_ids) → (B, L, H_mm)

    STEP 2 — NER classify (trước decoder):
        ner_logits = nerClassifier(enc_hidden) → (B, L, 10)
        ner_loss   = CrossEntropy(ner_logits, labels)

    STEP 3 — Project mmBERT dim → mT5 decoder dim:
        enc_proj = projector(enc_hidden) → (B, L, H_dec)

    STEP 4 — NER embedding → decoder input:
        Teacher forcing khi train: ner_ids = true labels
        Argmax khi eval           : ner_ids = argmax(ner_logits)
        ner_embed  = nerEmbed(ner_ids) → (B, L, H_mm)
        dec_input  = projector(ner_embed) → (B, L, H_dec)
          ↑ tái dùng projector vì nerEmbed.embedding_dim == encoder.hidden_size

    STEP 5 — mT5 decoder:
        decoder(
            inputs_embeds         = dec_input,   ← NER-aware token input
            encoder_hidden_states = enc_proj,    ← mmBERT features (cross-attention)
        ) → dec_hidden (B, L, H_dec)

    BACKWARD end-to-end:
        ner_loss
          → nerClassifier.grad       ← update nerClassifier weights
          → enc_hidden.grad
          → encoder.grad             ← update mmBERT weights
          → projector.grad           ← update projector weights
          → nerEmbed.grad            ← update nerEmbed weights
          → decoder.grad             ← update mT5 decoder weights
    """

    # ── STEP 1 ──────────────────────────────────────────────────────────
    enc_out    = model.encoder(
        input_ids      = input_ids,
        attention_mask = attention_mask,
    )
    enc_hidden = enc_out.last_hidden_state           # (B, L, H_mm)

    # ── STEP 2: NER classify ─────────────────────────────────────────────
    ner_logits = model.nerClassifier(enc_hidden)     # (B, L, 10)

    ner_loss = None
    if labels is not None:
        ner_loss = F.cross_entropy(
            ner_logits.view(-1, len(model.nerVocab)),
            labels.view(-1),
            ignore_index=-100,
        )

    # ── STEP 3: Project enc_hidden ───────────────────────────────────────
    enc_proj   = model.projector(enc_hidden)         # (B, L, H_dec)

    # ── STEP 4: NER embed → decoder input ───────────────────────────────
    if labels is not None and model.training:
        # Teacher forcing: dùng true labels (thay -100 bằng 0 = 'O')
        ner_ids       = labels.clone()
        ner_ids[ner_ids == -100] = 0
    else:
        # Eval: dùng prediction
        ner_ids       = ner_logits.argmax(dim=-1)    # (B, L)

    ner_embed  = model.nerEmbed(ner_ids)             # (B, L, H_mm)
    dec_input  = model.projector(ner_embed)          # (B, L, H_dec)

    # ── STEP 5: mT5 decoder ─────────────────────────────────────────────
    dec_out    = model.decoder(
        inputs_embeds          = dec_input,
        attention_mask         = attention_mask,
        encoder_hidden_states  = enc_proj,
        encoder_attention_mask = attention_mask,
    )
    dec_hidden = dec_out.last_hidden_state           # (B, L, H_dec)

    return {
        "loss":       ner_loss,
        "ner_logits": ner_logits,
        "dec_hidden": dec_hidden,
    }


# ════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT MANAGER — auto-save mỗi 30 phút
# ════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    def __init__(self, save_dir: str, interval_min: int = 30, max_keep: int = 3):
        self.save_dir  = save_dir
        self.interval  = interval_min * 60
        self.max_keep  = max_keep
        self.last_save = time.time()
        self.ckpt_list: List[str] = []
        os.makedirs(save_dir, exist_ok=True)
        log.info(f"[CKPT] Auto-save mỗi {interval_min} phút → {save_dir}")

    def _write(self, model, optimizer, scheduler, epoch, step, metrics, tag=""):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"mtt_ep{epoch}_step{step}{'_'+tag if tag else ''}_{ts}.pt"
        path = os.path.join(self.save_dir, name)
        torch.save({
            "epoch":           epoch,
            "step":            step,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics":         metrics,
        }, path)
        log.info(f"[CKPT] Saved ({tag or 'interval'}) → {os.path.basename(path)}")
        return path

    def save_interval(self, model, optimizer, scheduler, epoch, step, metrics):
        if (time.time() - self.last_save) >= self.interval:
            path = self._write(model, optimizer, scheduler, epoch, step, metrics)
            self.ckpt_list.append(path)
            while len(self.ckpt_list) > self.max_keep:
                old = self.ckpt_list.pop(0)
                if os.path.exists(old):
                    os.remove(old)
            self.last_save = time.time()

    def save_best(self, model, optimizer, scheduler, epoch, metrics):
        path = os.path.join(self.save_dir, "mtt_best.pt")
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics":         metrics,
        }, path)
        log.info(f"[CKPT] Best → {path}  (f1={metrics.get('val_f1', 0):.4f})")
        return path

    @staticmethod
    def load(path: str, model, optimizer=None, scheduler=None):
        log.info(f"[CKPT] Loading {path} ...")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        log.info(f"[CKPT] Resumed epoch={ckpt['epoch']} | "
                 f"metrics={ckpt.get('metrics', {})}")
        return ckpt["epoch"], ckpt.get("metrics", {})


# ════════════════════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(all_preds: List[List[str]],
                    all_labels: List[List[str]]) -> Dict:
    if SEQEVAL:
        return {
            "f1":        f1_score(all_labels,        all_preds, zero_division=0),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall":    recall_score(all_labels,    all_preds, zero_division=0),
        }
    correct = sum(p == l
                  for ps, ls in zip(all_preds, all_labels)
                  for p, l in zip(ps, ls))
    total   = sum(len(ls) for ls in all_labels)
    acc     = correct / max(total, 1)
    return {"f1": acc, "precision": acc, "recall": acc}


# ════════════════════════════════════════════════════════════════════════════
#  TRAINER
# ════════════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, model: MTT, cfg: Config,
                 train_loader: DataLoader, val_loader: DataLoader):

        self.device = cfg.resolve_device()
        self.model  = model.to(self.device)
        self.cfg    = cfg

        # Tách lr: encoder nhỏ, head + decoder + projector + nerEmbed lớn hơn
        enc_params  = list(model.encoder.parameters())
        head_params = (
            list(model.nerClassifier.parameters()) +
            list(model.nerEmbed.parameters()) +
            list(model.projector.parameters()) +
            list(model.decoder.parameters())
        )

        self.optimizer = torch.optim.AdamW([
            {"params": enc_params,  "lr": cfg.lr,      "name": "encoder"},
            {"params": head_params, "lr": cfg.lr_head, "name": "head+decoder"},
        ], weight_decay=cfg.weight_decay, eps=1e-8)

        total_steps  = len(train_loader) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.ckpt         = CheckpointManager(cfg.checkpoint_dir, cfg.save_interval)
        self.best_f1      = 0.0
        self.start_epoch  = 0
        self.history: Dict = {"train_loss": [], "val_loss": [], "val_f1": []}

        # ── Memory optimizations ──────────────────────────────────────────
        # 1. Mixed precision fp16 — giảm ~50% VRAM
        self.scaler = GradScaler(enabled=cfg.fp16 and self.device.type == "cuda")

        # 2. Gradient checkpointing — recompute activations thay vì lưu hết
        #    Đánh đổi: tốn thêm ~33% compute nhưng giảm ~30-40% VRAM
        if cfg.gradient_checkpointing:
            if hasattr(model.encoder, "gradient_checkpointing_enable"):
                model.encoder.gradient_checkpointing_enable()
                log.info("  ✓ Gradient checkpointing: encoder ON")
            if hasattr(model.decoder, "gradient_checkpointing_enable"):
                model.decoder.gradient_checkpointing_enable()
                log.info("  ✓ Gradient checkpointing: decoder ON")

        # 3. Gradient accumulation — effective_batch = batch_size × grad_accum
        self.grad_accum = cfg.grad_accum_steps
        log.info(f"  ✓ Effective batch size: "
                 f"{cfg.batch_size} × {cfg.grad_accum_steps} = "
                 f"{cfg.batch_size * cfg.grad_accum_steps}")

        # Log thông tin model
        self._log_model_info(model, total_steps, warmup_steps)

        # Resume
        if cfg.resume and os.path.exists(cfg.resume):
            ep, m = CheckpointManager.load(
                cfg.resume, self.model, self.optimizer, self.scheduler)
            self.model       = self.model.to(self.device)
            self.best_f1     = m.get("val_f1", 0.0)
            self.start_epoch = ep + 1
            if self.start_epoch >= cfg.epochs:
                self.start_epoch = 0
                log.info("[RESUME] Epoch cuối → reset về 0 với loaded weights")

    def _log_model_info(self, model, total_steps, warmup_steps):
        log.info(f"\n{'═'*60}")
        log.info(f"  MODEL: MTT")
        log.info(f"  mmBERT-small (encoder) + NER + mT5-small (decoder)")
        log.info(f"{'─'*60}")
        log.info(f"  encoder    : {sum(p.numel() for p in model.encoder.parameters()):>12,}")
        log.info(f"  nerEmbed   : {sum(p.numel() for p in model.nerEmbed.parameters()):>12,}")
        log.info(f"  nerClassif : {sum(p.numel() for p in model.nerClassifier.parameters()):>12,}")
        log.info(f"  projector  : {sum(p.numel() for p in model.projector.parameters()):>12,}")
        log.info(f"  decoder    : {sum(p.numel() for p in model.decoder.parameters()):>12,}")
        log.info(f"{'─'*60}")
        total = sum(p.numel() for p in model.parameters())
        log.info(f"  TOTAL      : {total:>12,}")
        log.info(f"  Device     : {self.device}")
        log.info(f"  Encoder lr : {self.cfg.lr:.1e}")
        log.info(f"  Head/Dec lr: {self.cfg.lr_head:.1e}")
        log.info(f"  Steps      : {total_steps} | Warmup: {warmup_steps}")
        log.info(f"{'═'*60}")

    # ── TRAIN ONE EPOCH ────────────────────────────────────────────────────
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, steps = 0.0, 0
        t0 = time.time()

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            # ── FORWARD với fp16 autocast ─────────────────────────────────
            # float16 → giảm ~50% VRAM, tốc độ nhanh hơn trên CUDA
            use_amp = self.cfg.fp16 and self.device.type == "cuda"
            with autocast(enabled=use_amp):
                out  = mtt_forward(self.model, input_ids, attention_mask, labels)
                # Chia cho grad_accum để scale đúng khi accumulate
                loss = out["loss"] / self.grad_accum

            # ── BACKWARD ─────────────────────────────────────────────────
            # scaler giữ gradient ở fp32 bên trong để tránh underflow
            self.scaler.scale(loss).backward()

            total_loss += loss.item() * self.grad_accum
            steps += 1

            # ── Optimizer step mỗi grad_accum steps ─────────────────────
            # Effective batch = batch_size × grad_accum (không tốn thêm VRAM)
            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)   # optimizer trước
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()              # scheduler sau

            if step % 100 == 0:
                lr_enc = self.optimizer.param_groups[0]["lr"]
                mem_gb = (torch.cuda.memory_allocated() / 1e9
                          if self.device.type == "cuda" else 0.0)
                log.info(
                    f"  Ep{epoch} | step {step:4d}/{len(self.train_loader)} | "
                    f"loss={loss.item()*self.grad_accum:.4f} | "
                    f"lr={lr_enc:.2e} | "
                    f"VRAM={mem_gb:.2f}GB | "
                    f"{time.time()-t0:.0f}s"
                )

            self.ckpt.save_interval(
                self.model, self.optimizer, self.scheduler,
                epoch, step, {"train_loss": loss.item() * self.grad_accum},
            )

        return total_loss / max(steps, 1)


    # ── EVALUATE ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss, n = 0.0, 0
        all_preds, all_labels = [], []

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            out = mtt_forward(self.model, input_ids, attention_mask, labels)
            if out["loss"] is not None:
                total_loss += out["loss"].item()
                n += 1

            preds    = out["ner_logits"].argmax(dim=-1).cpu().numpy()
            label_np = labels.cpu().numpy()

            for p_seq, l_seq in zip(preds, label_np):
                p_tags = [ID2NER_SEQEVAL[p] for p, l in zip(p_seq, l_seq) if l != -100]
                l_tags = [ID2NER_SEQEVAL[l] for l in l_seq                 if l != -100]
                all_preds.append(p_tags)
                all_labels.append(l_tags)

        return total_loss / max(n, 1), compute_metrics(all_preds, all_labels)

    # ── MAIN LOOP ──────────────────────────────────────────────────────────
    def train(self) -> Dict:
        log.info(f"\n{'▓'*60}")
        log.info(f"  START TRAINING — {self.cfg.epochs} epochs")
        log.info(f"{'▓'*60}")

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()

            train_loss       = self.train_epoch(epoch)
            val_loss, val_m  = self.evaluate(self.val_loader)
            elapsed          = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_m["f1"])

            log.info(
                f"\n{'─'*60}\n"
                f"  [Epoch {epoch}/{self.cfg.epochs-1}]  {elapsed:.0f}s\n"
                f"  train_loss : {train_loss:.4f}\n"
                f"  val_loss   : {val_loss:.4f}\n"
                f"  val_f1     : {val_m['f1']:.4f}\n"
                f"  precision  : {val_m['precision']:.4f}\n"
                f"  recall     : {val_m['recall']:.4f}\n"
                f"{'─'*60}\n"
            )

            if val_m["f1"] > self.best_f1:
                self.best_f1 = val_m["f1"]
                self.ckpt.save_best(
                    self.model, self.optimizer, self.scheduler,
                    epoch, {"val_f1": val_m["f1"], "val_loss": val_loss},
                )

        return {"best_val_f1": self.best_f1, "history": self.history}

    # ── TEST ──────────────────────────────────────────────────────────────
    def test(self, test_loader: DataLoader) -> Dict:
        best_path = os.path.join(self.cfg.checkpoint_dir, "mtt_best.pt")
        if os.path.exists(best_path):
            CheckpointManager.load(best_path, self.model)
            self.model = self.model.to(self.device)
            log.info("[TEST] Loaded best checkpoint")
        else:
            log.info("[TEST] Dùng model hiện tại (không tìm thấy best checkpoint)")

        test_loss, test_m = self.evaluate(test_loader)

        log.info(f"\n{'═'*60}")
        log.info(f"  TEST RESULTS — MTT")
        log.info(f"  test_loss : {test_loss:.4f}")
        log.info(f"  test_f1   : {test_m['f1']:.4f}")
        log.info(f"  precision : {test_m['precision']:.4f}")
        log.info(f"  recall    : {test_m['recall']:.4f}")
        log.info(f"{'═'*60}")

        if SEQEVAL:
            # In chi tiết từng label
            all_preds, all_labels = [], []
            self.model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    input_ids      = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels         = batch["labels"].to(self.device)
                    out            = mtt_forward(
                        self.model, input_ids, attention_mask, labels)
                    preds    = out["ner_logits"].argmax(dim=-1).cpu().numpy()
                    label_np = labels.cpu().numpy()
                    for p_seq, l_seq in zip(preds, label_np):
                        p_tags = [ID2NER[p] for p, l in zip(p_seq, l_seq) if l != -100]
                        l_tags = [ID2NER[l] for l in l_seq                 if l != -100]
                        all_preds.append(p_tags)
                        all_labels.append(l_tags)
            report = classification_report(all_labels, all_preds, zero_division=0)
            log.info(f"Classification Report:\n{report}")

        return test_m


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(cfg: Config):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log.info(f"Device: {cfg.resolve_device()}")

    # Tokenizer dùng mmBERT (input đi qua encoder)
    log.info("Loading mmBERT tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")

    # Data
    train_loader, val_loader, test_loader = load_data(cfg, tokenizer)

    # Model
    log.info("Khởi tạo MTT ...")
    model = MTT()
    model.paramsCalc()

    # Train
    trainer = Trainer(model, cfg, train_loader, val_loader)
    result  = trainer.train()
    test_m  = trainer.test(test_loader)

    # Summary
    summary = {
        "model":       "MTT (mmBERT encoder + NER + mT5 decoder)",
        "best_val_f1": result["best_val_f1"],
        "test":        test_m,
        "history":     result["history"],
    }
    out_path = os.path.join(cfg.checkpoint_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"Summary → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MTT: mmBERT encoder + NER + mT5 decoder")

    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--batch_size",     type=int,   default=4,
                        help="Micro batch size (default 4 để tránh OOM 6GB GPU)")
    parser.add_argument("--grad_accum",     type=int,   default=4,
                        help="Gradient accumulation steps (effective_batch = batch×accum)")
    parser.add_argument("--fp16",           action="store_true", default=True,
                        help="Mixed precision fp16 (giảm ~50pct VRAM)")
    parser.add_argument("--no_fp16",        action="store_false", dest="fp16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Gradient checkpointing (giảm thêm ~30pct VRAM)")
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--lr_head",        type=float, default=1e-4,
                        help="NER + decoder + projector + nerEmbed lr")
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--warmup_ratio",   type=float, default=0.1)
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--max_length",     type=int,   default=128)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--save_interval",  type=int,   default=30,
                        help="Auto-save mỗi N phút")
    parser.add_argument("--resume",         default=None,
                        help="Path checkpoint để resume")
    parser.add_argument("--device",         default="auto")

    args = parser.parse_args()

    cfg = Config(
        epochs                 = args.epochs,
        batch_size             = args.batch_size,
        lr                     = args.lr,
        lr_head                = args.lr_head,
        weight_decay           = args.weight_decay,
        warmup_ratio           = args.warmup_ratio,
        grad_clip              = args.grad_clip,
        max_length             = args.max_length,
        checkpoint_dir         = args.checkpoint_dir,
        save_interval          = args.save_interval,
        resume                 = args.resume,
        device                 = args.device,
        fp16                   = args.fp16,
        gradient_checkpointing = args.gradient_checkpointing,
        grad_accum_steps       = args.grad_accum,
    )

    main(cfg)