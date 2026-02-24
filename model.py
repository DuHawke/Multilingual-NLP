import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, MT5ForConditionalGeneration


class MTT(nn.Module):
    """
    Multilingual Translation Transformer
    =====================================
    mmBERT-small (encoder)  — encode source language (1800+ ngôn ngữ)
         ↓  projector       — align dimensions
    mT5-small   (decoder)   — generate target language (250k vocab, đa ngôn ngữ)

    Forward (teacher forcing khi train):
      src_ids → encoder → enc_hidden → projector → enc_proj
      tgt_ids → tgt_embed ↘
                           decoder(cross-attn vào enc_proj) → lm_head → logits
      loss = CrossEntropy(logits, labels)  [labels = tgt_ids shifted left]

    Generate (autoregressive khi inference):
      src_ids → encoder → enc_proj
      [BOS] → decoder step 1 → token 1
              decoder step 2 → token 2
              ...
              → [EOS] hoặc max_new_tokens
    """

    ENCODER_ID = "jhu-clsp/mmBERT-small"
    DECODER_ID = "google/mt5-small"

    def __init__(self):
        super().__init__()

        # Hai tokenizer: src dùng mmBERT, tgt dùng mT5 (250k multilingual vocab)
        self.src_tokenizer = AutoTokenizer.from_pretrained(self.ENCODER_ID)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(self.DECODER_ID,
                                                           use_fast=False)

        # Encoder: mmBERT-small
        self.encoder = AutoModel.from_pretrained(self.ENCODER_ID)
        H = self.encoder.config.hidden_size

        # Decoder + LM head + shared embedding: từ mT5-small
        mt5 = MT5ForConditionalGeneration.from_pretrained(self.DECODER_ID)
        self.decoder   = mt5.decoder    # T5Stack
        self.lm_head   = mt5.lm_head   # Linear(d_model → 250112)
        self.tgt_embed = mt5.shared     # Embedding(250112, d_model)
        del mt5

        # Projector: align mmBERT hidden_size → mT5 d_model
        self.projector = nn.Linear(H, self.decoder.config.d_model)

    def forward(self, src_ids, src_mask, tgt_ids, tgt_mask=None, labels=None):
        """
        src_ids  : (B, S)   — source tokens (mmBERT tokenizer)
        tgt_ids  : (B, T)   — target tokens shifted right (mT5 tokenizer)
        labels   : (B, T)   — target tokens shifted left, -100 = ignore
        """
        # Encode source
        enc_hidden = self.encoder(
            input_ids=src_ids, attention_mask=src_mask
        ).last_hidden_state                                  # (B, S, H_mm)

        enc_proj = self.projector(enc_hidden)                # (B, S, d_model)

        # Decode (teacher forcing: feed ground-truth tgt_ids)
        tgt_embeds = self.tgt_embed(tgt_ids)                 # (B, T, d_model)
        dec_out = self.decoder(
            inputs_embeds          = tgt_embeds,
            attention_mask         = tgt_mask,
            encoder_hidden_states  = enc_proj,
            encoder_attention_mask = src_mask,
        ).last_hidden_state                                  # (B, T, d_model)

        logits = self.lm_head(dec_out)                       # (B, T, 250112)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, src_ids, src_mask, max_new_tokens=128, num_beams=1):
        """
        Autoregressive generation — greedy (num_beams=1) hoặc beam search.

        src_ids  : (1, S)  — đã tokenize bằng src_tokenizer
        returns  : str     — translated text
        """
        self.eval()
        device = src_ids.device

        # Encode source một lần
        enc_hidden = self.encoder(
            input_ids=src_ids, attention_mask=src_mask
        ).last_hidden_state
        enc_proj = self.projector(enc_hidden)                # (1, S, d_model)

        # BOS token để bắt đầu decode
        bos_id  = self.tgt_tokenizer.pad_token_id or 0
        eos_id  = self.tgt_tokenizer.eos_token_id
        dec_ids = torch.tensor([[bos_id]], device=device)   # (1, 1)

        for _ in range(max_new_tokens):
            tgt_embeds = self.tgt_embed(dec_ids)
            dec_out = self.decoder(
                inputs_embeds         = tgt_embeds,
                encoder_hidden_states = enc_proj,
                encoder_attention_mask= src_mask,
            ).last_hidden_state

            next_logits  = self.lm_head(dec_out[:, -1, :])  # (1, vocab)
            next_token   = next_logits.argmax(-1, keepdim=True)  # (1, 1)
            dec_ids      = torch.cat([dec_ids, next_token], dim=-1)

            if next_token.item() == eos_id:
                break

        tokens = dec_ids[0].tolist()
        return self.tgt_tokenizer.decode(tokens, skip_special_tokens=True)

    def paramsCalc(self):
        parts = {
            "encoder":    self.encoder,
            "projector":  self.projector,
            "decoder":    self.decoder,
            "lm_head":    self.lm_head,
            "tgt_embed":  self.tgt_embed,
        }
        for name, mod in parts.items():
            print(f"  {name:<12}: {sum(p.numel() for p in mod.parameters()):>12,}")
        print(f"  {'TOTAL':<12}: {sum(p.numel() for p in self.parameters()):>12,}")