import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, MT5ForConditionalGeneration


class MTT(nn.Module):

    ENCODER_ID = "jhu-clsp/mmBERT-small"
    DECODER_ID = "google/mt5-small"

    # Label maps — dùng chung cho train và generate
    nerVocab = {
        'O': 0, 'PERSON': 1, 'ORG': 2, 'LOC': 3, 'GPE': 4,
        'DATE': 5, 'MONEY': 6, 'PERCENT': 7, 'TIME': 8, 'QUANTITY': 9
    }
    id2ner = {        # seqeval cần B- prefix
        0: 'O',
        1: 'B-PERSON', 2: 'B-ORG',      3: 'B-LOC',
        4: 'B-GPE',    5: 'B-DATE',     6: 'B-MONEY',
        7: 'B-PERCENT',8: 'B-TIME',     9: 'B-QUANTITY',
    }
    conll2mtt = {     # CoNLL-2003 → nerVocab
        'O': 'O',
        'B-PER': 'PERSON', 'I-PER': 'PERSON',
        'B-ORG': 'ORG',    'I-ORG': 'ORG',
        'B-LOC': 'LOC',    'I-LOC': 'LOC',
        'B-MISC': 'O',     'I-MISC': 'O',
    }

    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(self.ENCODER_ID)
        self.encoder   = AutoModel.from_pretrained(self.ENCODER_ID)

        H = self.encoder.config.hidden_size
        self.nerEmbed      = nn.Embedding(len(self.nerVocab), H)
        self.nerClassifier = nn.Linear(H, len(self.nerVocab))

        self.decoder   = MT5ForConditionalGeneration.from_pretrained(self.DECODER_ID).decoder
        self.projector = nn.Linear(H, self.decoder.config.hidden_size)

        for p in self.encoder.parameters():  p.requires_grad = True
        for p in self.decoder.parameters():  p.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        """
        STEP 1  encoder         → enc_hidden  (B, L, H_mm)
        STEP 2  nerClassifier   → ner_logits  (B, L, 10)   ← loss here
        STEP 3  projector       → enc_proj    (B, L, H_dec) for cross-attn
        STEP 4  nerEmbed → projector → dec_input  (B, L, H_dec)
                  teacher-forcing (true labels) khi train, argmax khi eval
        STEP 5  decoder         → dec_hidden  (B, L, H_dec)

        Backward end-to-end:
          loss → nerClassifier → enc_hidden → encoder
               → projector → nerEmbed → decoder
        """
        enc_hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                                    # (B, L, H_mm)

        ner_logits = self.nerClassifier(enc_hidden)            # (B, L, 10)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                ner_logits.view(-1, len(self.nerVocab)),
                labels.view(-1), ignore_index=-100,
            )

        enc_proj = self.projector(enc_hidden)                  # (B, L, H_dec)

        if labels is not None and self.training:
            ner_ids = labels.clone()
            ner_ids[ner_ids == -100] = 0                       # pad với O
        else:
            ner_ids = ner_logits.argmax(-1)

        dec_input  = self.projector(self.nerEmbed(ner_ids))    # (B, L, H_dec)
        dec_hidden = self.decoder(
            inputs_embeds          = dec_input,
            attention_mask         = attention_mask,
            encoder_hidden_states  = enc_proj,
            encoder_attention_mask = attention_mask,
        ).last_hidden_state                                    # (B, L, H_dec)

        return {"loss": loss, "ner_logits": ner_logits, "dec_hidden": dec_hidden}

    def paramsCalc(self):
        parts = {
            "encoder":       self.encoder,
            "nerEmbed":      self.nerEmbed,
            "nerClassifier": self.nerClassifier,
            "projector":     self.projector,
            "decoder":       self.decoder,
        }
        for name, mod in parts.items():
            print(f"  {name:<15}: {sum(p.numel() for p in mod.parameters()):>12,}")
        print(f"  {'TOTAL':<15}: {sum(p.numel() for p in self.parameters()):>12,}")