"""
train.py
========
Training pipeline cho NER: mmBERT-small vs mT5-small
Import model từ models.py

Nội dung:
  ┌─ Config          — dataclass chứa toàn bộ hyperparameters
  ├─ NERDataset      — tokenize + align labels
  ├─ CheckpointManager  — auto-save mỗi 30 phút
  ├─ Metrics         — tính F1, precision, recall (seqeval)
  ├─ NERTrainer      — training loop + eval loop
  └─ main()          — chạy cả hai model và so sánh

Chạy:
  python train.py
  python train.py --freeze_encoder --epochs 5
  python train.py --resume_mmbert ./checkpoints/mmBERT-small/best.pt
"""

import os
import time
import json
import logging
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, DatasetDict

# ── Import models từ models.py ───────────────────────────────────────────────
from models import MMBertNER, MT5NER

# ── Optional: seqeval ────────────────────────────────────────────────────────
try:
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    SEQEVAL = True
except ImportError:
    SEQEVAL = False
    print("[WARN] pip install seqeval  để có NER F1 đúng chuẩn CoNLL")


# ════════════════════════════════════════════════════════════════════════════
#  0.  CONLL-2003 LOADER  — không dùng script, load thẳng từ parquet HF Hub
#      hoặc parse file .conll nếu có sẵn local
# ════════════════════════════════════════════════════════════════════════════

def load_conll2003() -> tuple:
    """Load CoNLL-2003 dùng datasets<3.0 với trust_remote_code=True."""
    log.info("Loading CoNLL-2003 ...")
    raw         = load_dataset("conll2003", trust_remote_code=True)
    label_names = raw["train"].features["ner_tags"].feature.names
    label2id    = {l: i for i, l in enumerate(label_names)}
    id2label    = {i: l for i, l in enumerate(label_names)}
    log.info(f"✓ CoNLL-2003 loaded | train={len(raw['train'])} | "
             f"val={len(raw['validation'])} | test={len(raw['test'])}")
    return raw, label_names, label2id, id2label



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  1.  CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset_name:   str  = "conll2003"
    max_length:     int  = 128
    batch_size:     int  = 16
    num_workers:    int  = 0           # tăng lên 2-4 nếu CPU mạnh

    # ── Model ────────────────────────────────────────────────────────────────
    proj_size:      int  = 256
    dropout:        float = 0.1
    freeze_encoder: bool  = False      # True = chỉ train NER head (nhanh hơn)

    # ── Training ─────────────────────────────────────────────────────────────
    epochs:         int  = 3
    lr_mmbert:      float = 2e-5       # mmBERT thường cần lr nhỏ hơn
    lr_mt5:         float = 3e-5       # mT5 encoder lớn hơn, lr cao hơn chút
    weight_decay:   float = 0.01
    warmup_ratio:   float = 0.1        # 10% đầu dùng warmup
    grad_clip:      float = 1.0        # gradient clipping tránh explode

    # ── Checkpoint ───────────────────────────────────────────────────────────
    checkpoint_dir: str  = "./checkpoints"
    save_interval:  int  = 30          # phút, auto-save mỗi N phút
    max_keep:       int  = 3           # giữ tối đa N checkpoint gần nhất

    # ── Resume ───────────────────────────────────────────────────────────────
    resume_mmbert:  Optional[str] = None
    resume_mt5:     Optional[str] = None

    # ── Device ───────────────────────────────────────────────────────────────
    device:         str  = "auto"      # "auto" | "cuda" | "cpu" | "mps"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


# ════════════════════════════════════════════════════════════════════════════
#  2.  DATASET
# ════════════════════════════════════════════════════════════════════════════

class NERDataset(Dataset):
    """
    Wrap HuggingFace NER dataset, tokenize và align labels với subword tokens.

    Align rule:
      - Special tokens ([CLS], [SEP], [PAD]) → label = -100  (bỏ qua trong loss)
      - First subword của mỗi word          → label thật
      - Các subword tiếp theo               → label = -100
    """

    def __init__(self,
                 hf_split,
                 tokenizer,
                 max_length: int,
                 label2id:   Dict[str, int]):
        self.data     = hf_split
        self.tok      = tokenizer
        self.max_len  = max_length
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item   = self.data[idx]
        words  = item["tokens"]       # list[str]
        tags   = item["ner_tags"]     # list[int]

        enc = self.tok(
            words,
            is_split_into_words = True,
            truncation          = True,
            max_length          = self.max_len,
            padding             = "max_length",
            return_tensors      = "pt",
        )

        # Align labels
        word_ids   = enc.word_ids(batch_index=0)
        labels     = []
        prev_wid   = None
        for wid in word_ids:
            if wid is None:                # special token
                labels.append(-100)
            elif wid != prev_wid:          # first subword → label thật
                labels.append(tags[wid])
            else:                          # subsequent subword → ignore
                labels.append(-100)
            prev_wid = wid

        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # (L,)
            "attention_mask": enc["attention_mask"].squeeze(0),  # (L,)
            "labels":         torch.tensor(labels, dtype=torch.long),  # (L,)
        }


def build_dataloaders(cfg: TrainConfig,
                      tok_mmbert,
                      tok_mt5,
                      raw:      DatasetDict,
                      label2id: Dict[str, int],
                     ) -> Dict[str, Dict[str, DataLoader]]:
    """
    Tạo DataLoader cho mmBERT và mT5, split train/validation/test.
    Hai model dùng tokenizer khác nhau → dataset khác nhau.
    """
    log.info(f"Building DataLoaders ...")

    loaders = {}
    for model_key, tok in [("mmbert", tok_mmbert), ("mt5", tok_mt5)]:
        loaders[model_key] = {}
        for split, shuffle in [("train", True), ("validation", False), ("test", False)]:
            ds = NERDataset(raw[split], tok, cfg.max_length, label2id)
            loaders[model_key][split] = DataLoader(
                ds,
                batch_size  = cfg.batch_size,
                shuffle     = shuffle,
                num_workers = cfg.num_workers,
                pin_memory  = True,
            )

    total_train = len(raw["train"])
    log.info(f"  Train: {total_train} | Val: {len(raw['validation'])} | Test: {len(raw['test'])}")
    return loaders


# ════════════════════════════════════════════════════════════════════════════
#  3.  CHECKPOINT MANAGER  — auto-save mỗi 30 phút
# ════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    """
    Tự động save checkpoint theo 2 trigger:
      1. Mỗi `interval_minutes` phút (tránh crash mất progress)
      2. Khi val_f1 đạt best mới (giữ model tốt nhất)
    Giữ tối đa `max_keep` checkpoint để tiết kiệm disk.
    """

    def __init__(self,
                 save_dir:         str,
                 model_name:       str,
                 interval_minutes: int = 30,
                 max_keep:         int = 3):
        self.save_dir  = save_dir
        self.name      = model_name
        self.interval  = interval_minutes * 60   # convert sang giây
        self.max_keep  = max_keep
        self.last_save = time.time()
        self.ckpt_list: List[str] = []
        os.makedirs(save_dir, exist_ok=True)
        log.info(f"[CKPT] Auto-save mỗi {interval_minutes} phút → {save_dir}")

    def _save(self,
              model:     nn.Module,
              optimizer,
              scheduler,
              epoch:     int,
              step:      int,
              metrics:   Dict,
              tag:       str = "") -> str:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.name}_ep{epoch}_step{step}{('_' + tag) if tag else ''}_{ts}.pt"
        path = os.path.join(self.save_dir, name)

        torch.save({
            "epoch":           epoch,
            "step":            step,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics":         metrics,
            "timestamp":       ts,
        }, path)

        log.info(f"[CKPT] Saved ({tag or 'interval'}) → {os.path.basename(path)}")
        return path

    def _cleanup(self):
        """Xóa checkpoint cũ nếu vượt max_keep."""
        while len(self.ckpt_list) > self.max_keep:
            old = self.ckpt_list.pop(0)
            if os.path.exists(old):
                os.remove(old)
                log.info(f"[CKPT] Removed old → {os.path.basename(old)}")

    def save_interval(self, model, optimizer, scheduler, epoch, step, metrics):
        """Gọi trong training loop — chỉ save khi đến giờ."""
        if (time.time() - self.last_save) >= self.interval:
            path = self._save(model, optimizer, scheduler, epoch, step, metrics)
            self.ckpt_list.append(path)
            self._cleanup()
            self.last_save = time.time()
            return path
        return None

    def save_best(self, model, optimizer, scheduler, epoch, step, metrics):
        """Gọi khi đạt best val_f1 — luôn save, không bị cleanup."""
        best_path = os.path.join(self.save_dir, f"{self.name}_best.pt")
        torch.save({
            "epoch":           epoch,
            "step":            step,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics":         metrics,
        }, best_path)
        log.info(f"[CKPT] Best model → {best_path}  (f1={metrics.get('val_f1', '?'):.4f})")
        return best_path

    @staticmethod
    def load(path: str, model: nn.Module,
             optimizer=None, scheduler=None) -> Tuple[int, int, Dict]:
        log.info(f"[CKPT] Resuming from {path} ...")
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        log.info(f"[CKPT] Resumed: epoch={ckpt['epoch']}, "
                 f"metrics={ckpt.get('metrics', {})}")
        return ckpt["epoch"], ckpt["step"], ckpt.get("metrics", {})


# ════════════════════════════════════════════════════════════════════════════
#  4.  METRICS
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(all_preds:  List[List[str]],
                    all_labels: List[List[str]]) -> Dict[str, float]:
    """
    all_preds / all_labels: list of list of tag strings (không có -100)
    Trả về dict với f1, precision, recall (seqeval) hoặc accuracy (fallback).
    """
    if SEQEVAL:
        return {
            "f1":        f1_score(all_labels,        all_preds, zero_division=0),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall":    recall_score(all_labels,    all_preds, zero_division=0),
        }
    else:
        # Token-level accuracy fallback
        correct = sum(p == l
                      for ps, ls in zip(all_preds, all_labels)
                      for p, l in zip(ps, ls))
        total   = sum(len(ls) for ls in all_labels)
        acc     = correct / max(total, 1)
        return {"f1": acc, "precision": acc, "recall": acc}


# ════════════════════════════════════════════════════════════════════════════
#  5.  NER TRAINER
# ════════════════════════════════════════════════════════════════════════════

class NERTrainer:

    def __init__(self,
                 model:          nn.Module,
                 model_name:     str,
                 train_loader:   DataLoader,
                 val_loader:     DataLoader,
                 id2label:       Dict[int, str],
                 cfg:            TrainConfig,
                 resume_from:    Optional[str] = None,
                 lr:             float = 2e-5):

        self.device     = cfg.resolve_device()
        self.model      = model.to(self.device)
        self.name       = model_name
        self.id2label   = id2label
        self.cfg        = cfg
        self.num_labels = len(id2label)

        log.info(f"\n{'─'*55}")
        log.info(f"  Model    : {model_name}")
        log.info(f"  Device   : {self.device}")
        log.info(f"  Params   : {sum(p.numel() for p in model.parameters()):,}")
        log.info(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        log.info(f"{'─'*55}")

        # ── Optimizer + Scheduler ─────────────────────────────────────────
        # AdamW: gradient descent với adaptive learning rate + weight decay
        # Chỉ update các parameters có requires_grad=True
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr           = lr,
            weight_decay = cfg.weight_decay,
            eps          = 1e-8,
        )

        total_steps  = len(train_loader) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        log.info(f"  Total steps : {total_steps} | Warmup: {warmup_steps}")

        self.train_loader = train_loader
        self.val_loader   = val_loader

        # ── Checkpoint manager ────────────────────────────────────────────
        ckpt_dir = os.path.join(cfg.checkpoint_dir,
                                model_name.replace("/", "_").replace(" ", "_"))
        self.ckpt = CheckpointManager(ckpt_dir, model_name.split("/")[-1],
                                      cfg.save_interval, cfg.max_keep)

        # ── History ───────────────────────────────────────────────────────
        self.history = {
            "train_loss": [],
            "val_loss":   [],
            "val_f1":     [],
            "val_precision": [],
            "val_recall":    [],
            "epoch_time":    [],
        }
        self.best_f1    = 0.0
        self.start_epoch = 0

        # ── Resume từ checkpoint ──────────────────────────────────────────
        if resume_from and os.path.exists(resume_from):
            self.start_epoch, _, prev_metrics = CheckpointManager.load(
                resume_from, self.model, self.optimizer, self.scheduler
            )
            self.model      = self.model.to(self.device)
            self.best_f1    = prev_metrics.get("val_f1", 0.0)
            self.start_epoch += 1   # tiếp tục từ epoch kế tiếp

    # ── TRAIN ONE EPOCH ────────────────────────────────────────────────────
    def train_epoch(self, epoch: int) -> float:
        """
        Một epoch training:
          for each batch:
            1. Forward  → tính loss
            2. Backward → tính gradients (CustomLinear.backward chạy ở đây)
            3. Clip     → giới hạn gradient norm tránh explode
            4. Step     → gradient descent cập nhật W, b
            5. Scheduler → giảm lr theo schedule
            6. [mỗi 30p] → auto-save checkpoint
        """
        self.model.train()
        total_loss  = 0.0
        num_batches = 0
        t_epoch     = time.time()

        for step, batch in enumerate(self.train_loader):
            # Chuyển batch lên device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # ── FORWARD PASS ─────────────────────────────────────────────
            out  = self.model(**batch)        # gọi MMBertNER.forward / MT5NER.forward
            loss = out["loss"]                # scalar cross-entropy

            # ── BACKWARD PASS ────────────────────────────────────────────
            # autograd tự động gọi CustomLinearFunction.backward
            # gradient chảy ngược qua: CE → NERHead → (Projection) → Encoder
            self.optimizer.zero_grad()        # reset gradient từ step trước
            loss.backward()                   # ← đây là nơi backward() được gọi

            # ── GRADIENT CLIPPING ────────────────────────────────────────
            # Giới hạn L2 norm của toàn bộ gradient, tránh gradient explode
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )

            # ── OPTIMIZER STEP (Gradient Descent) ────────────────────────
            # W = W - lr * grad_W   (AdamW biến thể có momentum + adaptive lr)
            self.optimizer.step()
            self.scheduler.step()

            total_loss  += loss.item()
            num_batches += 1

            # ── Logging ──────────────────────────────────────────────────
            if step % 100 == 0:
                elapsed = time.time() - t_epoch
                lr_now  = self.scheduler.get_last_lr()[0]
                log.info(
                    f"  [{self.name}] Ep{epoch} | step {step:4d}/{len(self.train_loader)} "
                    f"| loss={loss.item():.4f} | lr={lr_now:.2e} | {elapsed:.0f}s"
                )

            # ── Auto-save checkpoint mỗi 30 phút ─────────────────────────
            self.ckpt.save_interval(
                self.model, self.optimizer, self.scheduler,
                epoch, step,
                {"train_loss": loss.item(), "step": step},
            )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    # ── EVALUATE ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict]:
        """
        Eval loop: không tính gradient (no_grad để tiết kiệm memory).
        Thu thập predictions và labels, tính NER metrics.
        """
        self.model.eval()
        total_loss  = 0.0
        num_batches = 0
        all_preds   = []   # list of list[str]
        all_labels  = []   # list of list[str]

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out   = self.model(**batch)

            if out["loss"] is not None:
                total_loss  += out["loss"].item()
                num_batches += 1

            # Lấy predicted label id
            preds  = out["logits"].argmax(dim=-1).cpu().numpy()  # (B, L)
            labels = batch["labels"].cpu().numpy()                # (B, L)

            for p_seq, l_seq in zip(preds, labels):
                # Chỉ lấy các vị trí có label thật (l != -100)
                p_tags = [self.id2label[p] for p, l in zip(p_seq, l_seq) if l != -100]
                l_tags = [self.id2label[l] for l in l_seq             if l != -100]
                all_preds.append(p_tags)
                all_labels.append(l_tags)

        avg_loss = total_loss / max(num_batches, 1)
        metrics  = compute_metrics(all_preds, all_labels)
        return avg_loss, metrics

    # ── MAIN TRAIN LOOP ────────────────────────────────────────────────────
    def train(self) -> Dict:
        log.info(f"\n{'═'*55}")
        log.info(f"  START TRAINING: {self.name}")
        log.info(f"  Epochs: {self.cfg.epochs} | Freeze encoder: {self.cfg.freeze_encoder}")
        log.info(f"{'═'*55}")

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()

            # ── Train ──────────────────────────────────────────────────
            train_loss = self.train_epoch(epoch)

            # ── Validate ───────────────────────────────────────────────
            val_loss, val_metrics = self.evaluate(self.val_loader)
            val_f1  = val_metrics["f1"]
            elapsed = time.time() - t0

            # ── Log epoch summary ──────────────────────────────────────
            log.info(
                f"\n[{self.name}] Epoch {epoch} done | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_f1={val_f1:.4f} | "
                f"P={val_metrics['precision']:.4f} | "
                f"R={val_metrics['recall']:.4f} | "
                f"time={elapsed:.0f}s\n"
            )

            # ── Lưu history ────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["epoch_time"].append(elapsed)

            # ── Save best checkpoint ───────────────────────────────────
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.ckpt.save_best(
                    self.model, self.optimizer, self.scheduler,
                    epoch, -1,
                    {"val_f1": val_f1, "val_loss": val_loss,
                     "precision": val_metrics["precision"],
                     "recall": val_metrics["recall"]},
                )

        return {
            "model":         self.name,
            "best_val_f1":   self.best_f1,
            "history":       self.history,
            "epochs_trained": self.cfg.epochs - self.start_epoch,
        }

    # ── TEST (final evaluation) ─────────────────────────────────────────────
    def test(self, test_loader: DataLoader) -> Dict:
        """Load best model, evaluate trên test set."""
        best_path = os.path.join(
            self.cfg.checkpoint_dir,
            self.name.replace("/", "_").replace(" ", "_"),
            f"{self.name.split('/')[-1]}_best.pt"
        )
        if os.path.exists(best_path):
            CheckpointManager.load(best_path, self.model)
            self.model = self.model.to(self.device)
            log.info(f"[TEST] Loaded best model from {best_path}")
        else:
            log.info("[TEST] Best checkpoint không tìm thấy, dùng model hiện tại")

        test_loss, test_metrics = self.evaluate(test_loader)
        log.info(
            f"\n[TEST] {self.name} | "
            f"test_loss={test_loss:.4f} | "
            f"test_f1={test_metrics['f1']:.4f} | "
            f"P={test_metrics['precision']:.4f} | "
            f"R={test_metrics['recall']:.4f}"
        )

        if SEQEVAL:
            all_preds, all_labels = self._collect_predictions(test_loader)
            report = classification_report(all_labels, all_preds, zero_division=0)
            log.info(f"\nClassification Report ({self.name}):\n{report}")

        return {"model": self.name, "test_loss": test_loss, **test_metrics}

    @torch.no_grad()
    def _collect_predictions(self, loader):
        """Thu thập toàn bộ predictions và labels (dưới dạng tag strings)."""
        self.model.eval()
        all_preds, all_labels = [], []
        for batch in loader:
            batch  = {k: v.to(self.device) for k, v in batch.items()}
            out    = self.model(**batch)
            preds  = out["logits"].argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            for p_seq, l_seq in zip(preds, labels):
                p_tags = [self.id2label[p] for p, l in zip(p_seq, l_seq) if l != -100]
                l_tags = [self.id2label[l] for l in l_seq             if l != -100]
                all_preds.append(p_tags)
                all_labels.append(l_tags)
        return all_preds, all_labels


# ════════════════════════════════════════════════════════════════════════════
#  6.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def print_comparison(results: Dict[str, Dict], test_results: Dict[str, Dict]):
    """In bảng so sánh kết quả cuối cùng."""
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  {'KẾT QUẢ SO SÁNH':^66}")
    print(bar)
    print(f"  {'Model':<20} {'Val F1':>8} {'Test F1':>8} "
          f"{'Precision':>10} {'Recall':>8} {'Epochs':>7}")
    print("─" * 70)
    for name, res in results.items():
        tr = test_results.get(name, {})
        print(f"  {name:<20} "
              f"{res['best_val_f1']:>8.4f} "
              f"{tr.get('f1', 0):>8.4f} "
              f"{tr.get('precision', 0):>10.4f} "
              f"{tr.get('recall', 0):>8.4f} "
              f"{res['epochs_trained']:>7}")
    print(bar)

    best_val  = max(results, key=lambda k: results[k]["best_val_f1"])
    best_test = max(test_results, key=lambda k: test_results[k].get("f1", 0)) \
                if test_results else best_val
    print(f"\n  🏆  Best Val  → {best_val}  "
          f"(F1 = {results[best_val]['best_val_f1']:.4f})")
    print(f"  🏆  Best Test → {best_test}  "
          f"(F1 = {test_results.get(best_test, {}).get('f1', 0):.4f})")
    print(bar + "\n")


def main(cfg: TrainConfig):
    device = cfg.resolve_device()
    log.info(f"Device: {device}")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── Load tokenizers ────────────────────────────────────────────────────
    log.info("Loading tokenizers ...")
    tok_mmbert = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
    # mT5 dùng SentencePiece → cần use_fast=False để tránh lỗi protobuf/tiktoken
    tok_mt5    = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)

    # ── Load dataset + label map ───────────────────────────────────────────
    raw, label_names, label2id, id2label = load_conll2003()
    num_labels  = len(label_names)
    log.info(f"Labels ({num_labels}): {label_names}")

    # ── Build DataLoaders ──────────────────────────────────────────────────
    loaders = build_dataloaders(cfg, tok_mmbert, tok_mt5, raw, label2id)

    results      = {}
    test_results = {}

    # ══════════════════════════════════════════════════════════════════════
    #  TRAIN mmBERT-small
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n" + "▓" * 55)
    log.info("  PHASE 1: mmBERT-small")
    log.info("▓" * 55)

    mmbert_model = MMBertNER(
        num_labels     = num_labels,
        proj_size      = cfg.proj_size,
        dropout        = cfg.dropout,
        freeze_encoder = cfg.freeze_encoder,
    )
    mmbert_trainer = NERTrainer(
        model        = mmbert_model,
        model_name   = "mmBERT-small",
        train_loader = loaders["mmbert"]["train"],
        val_loader   = loaders["mmbert"]["validation"],
        id2label     = id2label,
        cfg          = cfg,
        resume_from  = cfg.resume_mmbert,
        lr           = cfg.lr_mmbert,
    )
    results["mmBERT-small"]      = mmbert_trainer.train()
    test_results["mmBERT-small"] = mmbert_trainer.test(loaders["mmbert"]["test"])

    # Giải phóng VRAM trước khi train mT5
    del mmbert_model, mmbert_trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════
    #  TRAIN mT5-small
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n" + "▓" * 55)
    log.info("  PHASE 2: mT5-small")
    log.info("▓" * 55)

    mt5_model = MT5NER(
        num_labels     = num_labels,
        proj_size      = cfg.proj_size,
        dropout        = cfg.dropout,
        freeze_encoder = cfg.freeze_encoder,
    )
    mt5_trainer = NERTrainer(
        model        = mt5_model,
        model_name   = "mT5-small",
        train_loader = loaders["mt5"]["train"],
        val_loader   = loaders["mt5"]["validation"],
        id2label     = id2label,
        cfg          = cfg,
        resume_from  = cfg.resume_mt5,
        lr           = cfg.lr_mt5,
    )
    results["mT5-small"]      = mt5_trainer.train()
    test_results["mT5-small"] = mt5_trainer.test(loaders["mt5"]["test"])

    # ── In kết quả so sánh ─────────────────────────────────────────────────
    print_comparison(results, test_results)

    # ── Lưu summary JSON ───────────────────────────────────────────────────
    summary_path = os.path.join(cfg.checkpoint_dir, "summary.json")
    summary = {
        name: {
            "best_val_f1":   res["best_val_f1"],
            "test_f1":       test_results.get(name, {}).get("f1", 0),
            "test_precision":test_results.get(name, {}).get("precision", 0),
            "test_recall":   test_results.get(name, {}).get("recall", 0),
            "history": {
                "train_loss":  res["history"]["train_loss"],
                "val_loss":    res["history"]["val_loss"],
                "val_f1":      res["history"]["val_f1"],
            },
        }
        for name, res in results.items()
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"Summary saved → {summary_path}")

    return results, test_results


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NER Training: mmBERT-small vs mT5-small"
    )

    # Dataset
    parser.add_argument("--dataset",        default="conll2003")
    parser.add_argument("--max_length",     type=int,   default=128)
    parser.add_argument("--batch_size",     type=int,   default=16)

    # Model
    parser.add_argument("--proj_size",      type=int,   default=256)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Chỉ train NER head, đóng băng encoder")

    # Training
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--lr_mmbert",      type=float, default=2e-5)
    parser.add_argument("--lr_mt5",         type=float, default=3e-5)
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--warmup_ratio",   type=float, default=0.1)
    parser.add_argument("--grad_clip",      type=float, default=1.0)

    # Checkpoint
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--save_interval",  type=int,   default=30,
                        help="Auto-save mỗi N phút (default: 30)")
    parser.add_argument("--max_keep",       type=int,   default=3)
    parser.add_argument("--resume_mmbert",  default=None,
                        help="Path checkpoint mmBERT để resume")
    parser.add_argument("--resume_mt5",     default=None,
                        help="Path checkpoint mT5 để resume")

    # Device
    parser.add_argument("--device",         default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])

    args = parser.parse_args()

    cfg = TrainConfig(
        dataset_name   = args.dataset,
        max_length     = args.max_length,
        batch_size     = args.batch_size,
        proj_size      = args.proj_size,
        dropout        = args.dropout,
        freeze_encoder = args.freeze_encoder,
        epochs         = args.epochs,
        lr_mmbert      = args.lr_mmbert,
        lr_mt5         = args.lr_mt5,
        weight_decay   = args.weight_decay,
        warmup_ratio   = args.warmup_ratio,
        grad_clip      = args.grad_clip,
        checkpoint_dir = args.checkpoint_dir,
        save_interval  = args.save_interval,
        max_keep       = args.max_keep,
        resume_mmbert  = args.resume_mmbert,
        resume_mt5     = args.resume_mt5,
        device         = args.device,
    )

    main(cfg)