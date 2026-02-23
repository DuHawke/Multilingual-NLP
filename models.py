"""
models.py
=========
Định nghĩa toàn bộ kiến trúc model cho bài NER:
  ┌─ CustomLinearFunction  — forward + backward thủ công
  ├─ CustomLinear          — nn.Module dùng function trên
  ├─ ProjectionLayer       — dành riêng cho mT5 (align dimension)
  ├─ NERHead               — classifier chung cho cả hai model
  ├─ MMBertNER             — mmBERT-small + NERHead
  └─ MT5NER                — mT5-small encoder + ProjectionLayer + NERHead
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, T5EncoderModel


# ════════════════════════════════════════════════════════════════════════════
#  BLOCK 1 — CustomLinearFunction
#  Manual forward + backward, gradient descent dùng các gradient này
# ════════════════════════════════════════════════════════════════════════════

class CustomLinearFunction(torch.autograd.Function):
    """
    Tính y = x @ W.T + b  với forward và backward viết tay.

    Tại sao viết tay thay vì dùng F.linear?
      → Hiểu rõ luồng gradient, dễ debug, dễ custom gradient scaling.

    ┌──────────────── FORWARD ─────────────────┐
    │  Input : x  shape (..., in_features)     │
    │  Weight: W  shape (out_features, in_feat)│
    │  Bias  : b  shape (out_features,)        │
    │                                          │
    │  out = x @ W.T + b                       │
    │  shape: (..., out_features)              │
    └──────────────────────────────────────────┘

    ┌──────────────── BACKWARD ────────────────────────────────────────────┐
    │  Nhận grad_output (dL/dy) từ layer phía sau.                        │
    │                                                                      │
    │  Chain rule:                                                         │
    │    dL/dx = dL/dy  · dy/dx  = grad_output @ W                        │
    │    dL/dW = dL/dy  · dy/dW  = grad_output.T @ x   (batched sum)      │
    │    dL/db = dL/dy  · dy/db  = sum(grad_output, over batch+seq dims)  │
    │                                                                      │
    │  Optimizer (AdamW / SGD) sẽ dùng dL/dW và dL/db để update W và b:  │
    │    W_new = W - lr * dL/dW                                            │
    │    b_new = b - lr * dL/db                                            │
    └──────────────────────────────────────────────────────────────────────┘
    """

    # ── FORWARD ─────────────────────────────────────────────────────────────
    @staticmethod
    def forward(ctx,
                x:      torch.Tensor,   # (..., in_features)
                weight: torch.Tensor,   # (out_features, in_features)
                bias:   torch.Tensor,   # (out_features,)  hoặc None
               ) -> torch.Tensor:

        # Lưu lại để dùng trong backward
        ctx.save_for_backward(x, weight, bias)

        # y = x @ W^T  →  shape (..., out_features)
        out = x.matmul(weight.t())

        # Cộng bias nếu có
        if bias is not None:
            out = out + bias          # broadcast tự động theo dim cuối

        return out                    # shape (..., out_features)

    # ── BACKWARD ────────────────────────────────────────────────────────────
    @staticmethod
    def backward(ctx,
                 grad_output: torch.Tensor,   # dL/dy, shape (..., out_features)
                ) -> tuple:

        x, weight, bias = ctx.saved_tensors

        # ── Flatten batch + seq dims để tính matmul ──────────────────────
        # x           : (batch, seq_len, in_feat)  →  (N, in_feat)
        # grad_output : (batch, seq_len, out_feat) →  (N, out_feat)
        original_shape = x.shape
        x_2d  = x.reshape(-1, original_shape[-1])              # (N, in_feat)
        go_2d = grad_output.reshape(-1, grad_output.shape[-1]) # (N, out_feat)

        # ── dL/dx  =  grad_output @ W ────────────────────────────────────
        # Mỗi token nhận gradient ngược từ tất cả output neurons
        grad_x = go_2d.matmul(weight)           # (N, in_feat)
        grad_x = grad_x.reshape(original_shape) # khôi phục shape gốc

        # ── dL/dW  =  grad_output.T @ x ──────────────────────────────────
        # Sum qua toàn bộ batch và sequence positions
        grad_weight = go_2d.t().matmul(x_2d)   # (out_feat, in_feat)

        # ── dL/db  =  sum(grad_output, dim=0..n-1) ───────────────────────
        # Sum qua tất cả dims trừ dim cuối (feature dim)
        grad_bias = go_2d.sum(dim=0) if bias is not None else None  # (out_feat,)

        # Phải trả về gradient cho TỪNG argument của forward (x, weight, bias)
        return grad_x, grad_weight, grad_bias


# ════════════════════════════════════════════════════════════════════════════
#  BLOCK 2 — CustomLinear  (nn.Module wrapper)
# ════════════════════════════════════════════════════════════════════════════

class CustomLinear(nn.Module):
    """
    Linear layer dùng CustomLinearFunction.
    - W và b là nn.Parameter → autograd track, optimizer update.
    - Gradient descent hoạt động hoàn toàn qua backward ở trên.

    Dùng như nn.Linear bình thường:
        layer = CustomLinear(128, 64)
        out   = layer(x)          # forward
        loss.backward()           # backward tự động gọi CustomLinearFunction.backward
        optimizer.step()          # cập nhật W, b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Kaiming uniform — tốt cho ReLU/GELU, giữ variance ổn định qua layers
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            # Bias khởi tạo gần 0, fan_in dựa theo weight
            bound = 1 / math.sqrt(in_features)
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply() → gọi CustomLinearFunction.forward
        # Khi loss.backward() chạy → CustomLinearFunction.backward được gọi
        return CustomLinearFunction.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}  ← custom backward"
        )


# ════════════════════════════════════════════════════════════════════════════
#  BLOCK 3 — ProjectionLayer  (chỉ dành cho mT5, align hidden size)
# ════════════════════════════════════════════════════════════════════════════

class ProjectionLayer(nn.Module):
    """
    Project hidden_size của mT5 (512) xuống proj_size (256) trước NERHead.

    Tại sao cần layer này?
      mmBERT-small thường có hidden_size nhỏ hơn hoặc bằng proj_size.
      mT5-small có d_model=512 → cần giảm chiều để NERHead hai model
      có cùng kiến trúc, dễ so sánh fair.

    Luồng gradient:
      loss → NERHead.backward → ProjectionLayer.backward (CustomLinear) → encoder
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = CustomLinear(in_dim, out_dim)  # ← custom backward
        self.norm   = nn.LayerNorm(out_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (batch, seq_len, in_dim)
        out : (batch, seq_len, out_dim)
        """
        x = self.linear(x)   # forward của CustomLinear (và backward sẽ flow ngược)
        x = self.norm(x)
        x = self.drop(x)
        return x


# ════════════════════════════════════════════════════════════════════════════
#  BLOCK 4 — NERHead  (Projection + Classifier, dùng chung cho 2 model)
# ════════════════════════════════════════════════════════════════════════════

class NERHead(nn.Module):
    """
    Classifier cho token-level NER.

    Kiến trúc:
        hidden_states (B, L, H)
              ↓  CustomLinear(H → proj_size)   [PROJECTION — giảm chiều]
              ↓  GELU
              ↓  Dropout
              ↓  CustomLinear(proj_size → num_labels)  [CLASSIFIER]
              ↓
          logits (B, L, num_labels)

    Cả hai CustomLinear đều có forward/backward thủ công.
    Gradient từ loss chảy ngược qua:
        Cross-Entropy → classifier.backward → GELU → projection.backward → encoder
    """

    def __init__(self,
                 hidden_size: int,
                 num_labels:  int,
                 proj_size:   int   = 256,
                 dropout:     float = 0.1):
        super().__init__()

        # Lớp 1: Projection  H → proj_size
        self.projection = CustomLinear(hidden_size, proj_size)   # custom backward

        self.act  = nn.GELU()       # smooth activation, tốt hơn ReLU cho BERT
        self.drop = nn.Dropout(dropout)

        # Lớp 2: Classifier  proj_size → num_labels
        self.classifier = CustomLinear(proj_size, num_labels)    # custom backward

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states : (batch, seq_len, hidden_size)
        returns logits: (batch, seq_len, num_labels)

        Backward flow (tự động khi loss.backward()):
          grad từ CE loss
            → classifier.backward()   dL/dW_cls, dL/db_cls, dL/dx
            → dropout (pass-through hoặc zero)
            → GELU.backward()
            → projection.backward()   dL/dW_proj, dL/db_proj, dL/dx
            → encoder hidden states
        """
        x = self.projection(hidden_states)  # (B, L, proj_size)
        x = self.act(x)
        x = self.drop(x)
        x = self.classifier(x)              # (B, L, num_labels)
        return x


# ════════════════════════════════════════════════════════════════════════════
#  BLOCK 5 — MMBertNER
# ════════════════════════════════════════════════════════════════════════════

class MMBertNER(nn.Module):
    """
    mmBERT-small  +  NERHead

    mmBERT-small (jhu-clsp/mmBERT-small):
      - Dựa trên ModernBERT architecture (2024)
      - Hỗ trợ 1800+ ngôn ngữ
      - Nhanh hơn XLM-R ~4x nhờ Flash Attention + Rotary PE
      - Encoder-only (giống BERT, phù hợp NER)

    Forward flow:
      input_ids → mmBERT encoder → last_hidden_state (B, L, H)
                → NERHead → logits (B, L, num_labels)
                → CrossEntropyLoss (nếu có labels)

    Backward flow:
      loss → CE.backward → NERHead.backward (2 CustomLinear) → encoder.backward
    """

    MODEL_ID = "jhu-clsp/mmBERT-small"

    def __init__(self,
                 num_labels:      int,
                 proj_size:       int   = 256,
                 dropout:         float = 0.1,
                 freeze_encoder:  bool  = False):
        super().__init__()

        # Encoder
        self.encoder = AutoModel.from_pretrained(self.MODEL_ID)
        hidden_size  = self.encoder.config.hidden_size

        # NER Head (dùng CustomLinear bên trong)
        self.ner_head   = NERHead(hidden_size, num_labels, proj_size, dropout)
        self.num_labels = num_labels

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Khi freeze: gradient chỉ chạy trong NERHead, không vào encoder

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels:         torch.Tensor = None,
               ) -> dict:
        """
        input_ids      : (B, L)
        attention_mask : (B, L)
        labels         : (B, L)  — -100 cho các token cần ignore

        Returns dict:
          loss   : scalar Tensor (None nếu không truyền labels)
          logits : (B, L, num_labels)
        """
        # ── FORWARD encoder ─────────────────────────────────────────────
        encoder_out     = self.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask)
        hidden_states   = encoder_out.last_hidden_state   # (B, L, H)

        # ── FORWARD NER head ────────────────────────────────────────────
        logits = self.ner_head(hidden_states)              # (B, L, num_labels)

        # ── Loss (Cross-Entropy) ─────────────────────────────────────────
        loss = None
        if labels is not None:
            # Flatten: (B*L, num_labels) vs (B*L,)
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,          # bỏ qua [CLS], [PAD], subword sau first
            )

        return {"loss": loss, "logits": logits}


# ════════════════════════════════════════════════════════════════════════════
#  BLOCK 6 — MT5NER
# ════════════════════════════════════════════════════════════════════════════

class MT5NER(nn.Module):
    """
    mT5-small ENCODER  +  ProjectionLayer  +  NERHead

    mT5-small (google/mt5-small):
      - Encoder-Decoder, nhưng ta chỉ dùng ENCODER cho NER
      - d_model = 512  (lớn hơn mmBERT-small)
      - Multilingual, 101 ngôn ngữ
      - Cần ProjectionLayer để align xuống proj_size trước NERHead

    Forward flow:
      input_ids → T5Encoder → last_hidden_state (B, L, 512)
                → ProjectionLayer (CustomLinear 512→256 + LN) → (B, L, 256)
                → NERHead (2 CustomLinear) → logits (B, L, num_labels)
                → CrossEntropyLoss

    Backward flow:
      loss → CE.backward
           → NERHead.classifier.backward    (dL/dW_cls)
           → NERHead.projection.backward    (dL/dW_proj)
           → ProjectionLayer.linear.backward (dL/dW_proj_mt5)  ← extra layer
           → T5Encoder.backward
    """

    MODEL_ID = "google/mt5-small"

    def __init__(self,
                 num_labels:     int,
                 proj_size:      int   = 256,
                 dropout:        float = 0.1,
                 freeze_encoder: bool  = False):
        super().__init__()

        # Chỉ lấy encoder của mT5
        self.encoder = T5EncoderModel.from_pretrained(self.MODEL_ID)
        d_model      = self.encoder.config.d_model          # 512

        # ProjectionLayer: 512 → proj_size (CustomLinear + LayerNorm)
        self.projection = ProjectionLayer(d_model, proj_size, dropout)

        # NERHead: proj_size → num_labels (2 CustomLinear bên trong)
        # Vì đã project xong, NERHead nhận proj_size làm hidden_size
        self.ner_head   = NERHead(proj_size, num_labels, proj_size, dropout)
        self.num_labels = num_labels

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Gradient chỉ chạy trong ProjectionLayer + NERHead

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels:         torch.Tensor = None,
               ) -> dict:
        """
        Returns dict:
          loss   : scalar Tensor (None nếu không truyền labels)
          logits : (B, L, num_labels)
        """
        # ── FORWARD T5 encoder ──────────────────────────────────────────
        encoder_out   = self.encoder(input_ids=input_ids,
                                     attention_mask=attention_mask)
        hidden_states = encoder_out.last_hidden_state      # (B, L, 512)

        # ── FORWARD ProjectionLayer (CustomLinear: 512 → proj_size) ─────
        projected     = self.projection(hidden_states)     # (B, L, proj_size)

        # ── FORWARD NERHead (2 CustomLinear) ────────────────────────────
        logits        = self.ner_head(projected)           # (B, L, num_labels)

        # ── Loss ─────────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}


# ════════════════════════════════════════════════════════════════════════════
#  SMOKE TEST — chạy python models.py để verify forward + backward
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SMOKE TEST: forward + backward")
    print("=" * 60)

    # ── Test 1: CustomLinear backward ──────────────────────────────────
    print("\n[1] CustomLinear backward ...")
    cl = CustomLinear(32, 16)
    x  = torch.randn(4, 10, 32, requires_grad=True)   # batch=4, seq=10
    y  = cl(x)                                         # forward
    y.sum().backward()                                  # backward
    assert x.grad       is not None, "FAIL: grad_x is None"
    assert cl.weight.grad is not None, "FAIL: grad_W is None"
    assert cl.bias.grad   is not None, "FAIL: grad_b is None"
    print(f"  forward  → output shape : {y.shape}")
    print(f"  backward → grad_x  shape: {x.grad.shape}")
    print(f"  backward → grad_W  shape: {cl.weight.grad.shape}")
    print(f"  backward → grad_b  shape: {cl.bias.grad.shape}")
    print("  ✓ CustomLinear OK")

    # ── Test 2: ProjectionLayer ────────────────────────────────────────
    print("\n[2] ProjectionLayer (512→256) ...")
    proj = ProjectionLayer(512, 256)
    h    = torch.randn(2, 15, 512, requires_grad=True)
    out  = proj(h)
    out.sum().backward()
    assert h.grad is not None, "FAIL: grad không chảy qua ProjectionLayer"
    print(f"  forward  → {h.shape} → {out.shape}")
    print(f"  backward → grad_h shape: {h.grad.shape}")
    print("  ✓ ProjectionLayer OK")

    # ── Test 3: NERHead ────────────────────────────────────────────────
    print("\n[3] NERHead (64 → 9 labels) ...")
    head   = NERHead(hidden_size=64, num_labels=9, proj_size=32)
    hstate = torch.randn(2, 15, 64, requires_grad=True)
    logits = head(hstate)
    logits.sum().backward()
    assert hstate.grad is not None, "FAIL: grad không chảy qua NERHead"
    print(f"  forward  → logits shape: {logits.shape}")
    print(f"  backward → grad_hidden  shape: {hstate.grad.shape}")
    print("  ✓ NERHead OK")

    # ── Test 4: Gradient correctness (numerical check) ─────────────────
    print("\n[4] Numerical gradient check trên CustomLinear ...")
    torch.manual_seed(42)
    small_x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    cl2     = CustomLinear(4, 6)
    cl2     = cl2.double()
    result  = torch.autograd.gradcheck(
        CustomLinearFunction.apply,
        (small_x, cl2.weight, cl2.bias),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )
    print(f"  gradcheck result: {'✓ PASSED' if result else '✗ FAILED'}")

    print("\n" + "=" * 60)
    print("  Tất cả tests passed — forward + backward hoạt động đúng")
    print("=" * 60)