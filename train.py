"""
train.py  –  Train MTT 5 epochs, checkpoint mỗi 30 phút.
Tối ưu cho GPU 6GB:
  - bf16 (stable hơn fp16, không bị overflow → NaN)
  - 8-bit AdamW (bitsandbytes) → tiết kiệm ~2.4GB VRAM
  - gradient checkpointing
  - NaN guard: phát hiện NaN loss → skip step, không để lan

pip install bitsandbytes

Chỉnh nếu OOM: giảm MAX_SRC_LEN / MAX_TGT_LEN trong model.py (128 → 64)
"""

import os, time, signal, math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("[WARN] bitsandbytes chưa cài → pip install bitsandbytes\n")

from model import MTT

# ════════════════════════════════════════════════════════════
BATCH_SIZE = 1
# ════════════════════════════════════════════════════════════

GRAD_ACCUM    = 8        # effective batch = 8
LR            = 5e-5     # thấp hơn (3e-4 quá cao → loss diverge)
WEIGHT_DECAY  = 0.01
NUM_EPOCHS    = 50
WARMUP_RATIO  = 0.1      # warmup dài hơn để khởi động ổn định
CKPT_DIR      = "checkpoints"
CKPT_INTERVAL = 30 * 60
LOG_EVERY     = 20
NUM_WORKERS   = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bf16 ổn định hơn fp16 (không overflow), dùng nếu GPU hỗ trợ
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16
DTYPE    = torch.bfloat16 if USE_BF16 else torch.float16

print(f"[INFO] Precision: {'bf16' if USE_BF16 else 'fp16' if USE_FP16 else 'fp32'}")


# ─── Dataset ─────────────────────────────────────────────────────────────────

class TextPairDataset(Dataset):
    def __init__(self):
        # ── THAY DATA THỰC VÀO ĐÂY ──────────────────────────────────────────
        self.pairs = [
            ("The president of France visited Vietnam last week.",
             "Tổng thống Pháp đã thăm Việt Nam tuần trước."),
            ("Apple Inc. reported record revenue in Q4 2024.",
             "Apple Inc. báo cáo doanh thu kỷ lục trong Q4 2024."),
            ("The river flooded three towns in southern Germany.",
             "Dòng sông đã làm ngập lụt ba thị trấn ở miền nam nước Đức."),
        ] * 300
        # ────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate(batch):
    has_ner = len(batch[0]) == 3
    srcs = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    ner  = [b[2] for b in batch] if has_ner else None
    return srcs, tgts, ner


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def save_ckpt(model, optimizer, scheduler, epoch, step, loss):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"ckpt_ep{epoch}_step{step}.pt")
    torch.save({
        "epoch"          : epoch,
        "step"           : step,
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "loss"           : loss,
    }, path)
    print(f"\n[CKPT] ✓ Saved → {path}  (loss={loss:.4f})\n")


def load_latest_ckpt(model, optimizer, scheduler):
    if not os.path.isdir(CKPT_DIR):
        return 0, 0
    pts = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")]
    if not pts:
        return 0, 0
    pts.sort(key=lambda f: int(f.split("step")[1].split(".")[0]) if "step" in f else -1)
    path = os.path.join(CKPT_DIR, pts[-1])
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    model_key = "model_state"     if "model_state"     in ckpt else "model"
    opt_key   = "optimizer_state" if "optimizer_state" in ckpt else "optimizer"
    sched_key = "scheduler_state" if "scheduler_state" in ckpt else "scheduler"

    if model_key not in ckpt:
        print(f"[WARN] Checkpoint {path} cấu trúc không nhận ra, bỏ qua.")
        return 0, 0

    model.load_state_dict(ckpt[model_key])
    if opt_key   in ckpt: optimizer.load_state_dict(ckpt[opt_key])
    if sched_key in ckpt: scheduler.load_state_dict(ckpt[sched_key])

    epoch = ckpt.get("epoch", 0)
    step  = ckpt.get("step",  0)
    loss  = ckpt.get("loss",  float("nan"))
    print(f"[CKPT] Resumed {path}  (epoch={epoch}, step={step}, loss={loss:.4f})")
    return epoch, step


# ─── Optimizer ───────────────────────────────────────────────────────────────

def make_optimizer(model):
    no_decay = {"bias", "LayerNorm.weight"}
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    if HAS_BNB:
        print("[INFO] Dùng 8-bit AdamW (bitsandbytes)")
        return bnb.optim.AdamW8bit(params, lr=LR)
    else:
        return torch.optim.AdamW(params, lr=LR)


# ─── Train ───────────────────────────────────────────────────────────────────

def train():
    stop = False
    def on_sigint(sig, frame):
        nonlocal stop
        print("\n[INFO] Ctrl+C – dừng sau step hiện tại và lưu checkpoint…")
        stop = True
    signal.signal(signal.SIGINT, on_sigint)

    # Model
    print("[INFO] Khởi tạo model…")
    model = MTT().to(DEVICE)
    model.enable_gradient_checkpointing()
    print("[INFO] Param count:")
    model.paramsCalc()

    # GradScaler chỉ dùng cho fp16 (bf16 không cần scaler)
    use_scaler = USE_FP16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler, init_scale=256)

    # Optimizer
    optimizer = make_optimizer(model)

    # Dataset & loader
    dataset = TextPairDataset()
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        collate_fn=collate,
        drop_last=True,
    )

    # Scheduler
    steps_per_epoch = math.ceil(len(dataset) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps     = steps_per_epoch * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # Resume
    start_epoch, global_step = load_latest_ckpt(model, optimizer, scheduler)

    amp_ctx = torch.amp.autocast("cuda", dtype=DTYPE, enabled=(USE_BF16 or USE_FP16))

    print(f"\n[INFO] Training {NUM_EPOCHS} epochs | "
          f"batch={BATCH_SIZE} | accum={GRAD_ACCUM} | effective={BATCH_SIZE*GRAD_ACCUM} | "
          f"lr={LR} | device={DEVICE} | "
          f"{'bf16' if USE_BF16 else 'fp16' if USE_FP16 else 'fp32'} | "
          f"8bit={HAS_BNB}\n")

    last_ckpt_time = time.time()
    last_loss      = float("nan")
    nan_count      = 0       # đếm NaN liên tiếp để cảnh báo
    last_gen_loss  = 0.0
    last_ner_loss  = 0.0

    if start_epoch >= NUM_EPOCHS:
        print(f"[INFO] Checkpoint đã train đủ {NUM_EPOCHS} epochs.")
        print(f"[INFO] Tăng NUM_EPOCHS > {NUM_EPOCHS} để train thêm, hoặc xóa checkpoints/ để train lại.")
        return

    for epoch in range(start_epoch, NUM_EPOCHS):
        if stop:
            break

        model.train()
        epoch_loss, epoch_valid_steps = 0.0, 0
        running_loss, micro           = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for srcs, tgts, ner_labels in loader:
            if stop:
                break

            # NER labels → tensor nếu có
            ner_t = None
            if ner_labels is not None:
                src_len = model.MAX_SRC_LEN
                padded  = [(seq + [-100] * src_len)[:src_len] for seq in ner_labels]
                ner_t   = torch.tensor(padded, dtype=torch.long, device=DEVICE)

            # ── Forward ──────────────────────────────────────────────────────
            with amp_ctx:
                total_loss, gen_loss, ner_loss = model(
                    src_texts=srcs,
                    tgt_texts=tgts,
                    ner_labels=ner_t,
                )

            # ── NaN guard: bỏ qua batch này, không backward ──────────────────
            if not torch.isfinite(total_loss):
                nan_count += 1
                if nan_count % 10 == 1:
                    print(f"  [WARN] NaN/Inf loss tại micro-step, skip "
                          f"(tổng {nan_count} lần). "
                          f"gen={gen_loss.item():.2f}")
                optimizer.zero_grad(set_to_none=True)
                micro = 0   # reset accum để không step với gradient rác
                continue
            nan_count = 0   # reset nếu loss hợp lệ trở lại

            # ── Backward ─────────────────────────────────────────────────────
            loss = total_loss / GRAD_ACCUM
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss  += total_loss.item()
            last_gen_loss  = gen_loss.item()
            last_ner_loss  = ner_loss.item()
            micro         += 1

            # ── Optimizer step ───────────────────────────────────────────────
            if micro % GRAD_ACCUM == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)

                # Clip gradient – quan trọng để tránh diverge
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                # Bỏ qua nếu gradient vẫn NaN sau clip
                if not torch.isfinite(grad_norm):
                    print(f"  [WARN] Gradient NaN/Inf (norm={grad_norm:.2f}), skip optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    if use_scaler:
                        scaler.update()
                    micro        = 0
                    running_loss = 0.0
                    continue

                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step        += 1
                epoch_valid_steps  += 1
                avg                 = running_loss / GRAD_ACCUM
                epoch_loss         += avg
                last_loss           = avg
                running_loss        = 0.0
                micro               = 0

                # Log
                if global_step % LOG_EVERY == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    vram   = torch.cuda.memory_allocated() / 1e9
                    print(
                        f"  ep {epoch+1}/{NUM_EPOCHS}  "
                        f"step {global_step:>6d}  "
                        f"loss={avg:.4f}  "
                        f"gen={last_gen_loss:.4f}  "
                        f"ner={last_ner_loss:.4f}  "
                        f"lr={lr_now:.2e}  "
                        f"vram={vram:.2f}GB"
                    )

                # Checkpoint mỗi 30 phút
                if time.time() - last_ckpt_time >= CKPT_INTERVAL:
                    save_ckpt(model, optimizer, scheduler, epoch, global_step, last_loss)
                    last_ckpt_time = time.time()

        # Cuối epoch
        if epoch_valid_steps > 0:
            avg_ep = epoch_loss / epoch_valid_steps
            print(f"\n{'─'*60}")
            print(f"  ✓ Epoch {epoch+1}/{NUM_EPOCHS} done  |  avg_loss={avg_ep:.4f}  |  valid_steps={epoch_valid_steps}")
            print(f"{'─'*60}\n")
            save_ckpt(model, optimizer, scheduler, epoch + 1, global_step, avg_ep)
        else:
            print(f"  [WARN] Epoch {epoch+1} không có valid step nào (toàn NaN)!")
        last_ckpt_time = time.time()

    print("[INFO] Training hoàn tất.")


if __name__ == "__main__":
    train()