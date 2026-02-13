import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer

class NEREnhancedTranslationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pretrained models
        self.encoder = BertModel.from_pretrained(config.MBERT_MODEL)
        self.encoder_tokenizer = BertTokenizer.from_pretrained(config.MBERT_MODEL)
        
        self.decoder = T5ForConditionalGeneration.from_pretrained(config.MT5_MODEL)
        self.decoder_tokenizer = T5Tokenizer.from_pretrained(config.MT5_MODEL)
        
        # NER embedding
        self.ner_embedding = nn.Embedding(
            len(config.NER_VOCAB),
            self.encoder.config.hidden_size
        )
        
        # Projection layers
        encoder_dim = self.encoder.config.hidden_size
        decoder_dim = self.decoder.config.d_model
        
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.projection = nn.Linear(encoder_dim, decoder_dim)
        
    def forward(self, src_input_ids, src_attention_mask, ner_tags,
                tgt_input_ids=None, tgt_attention_mask=None):
        
        # Encode
        encoder_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask
        )
        encoder_hidden = encoder_outputs.last_hidden_state
        
        # Add NER information
        ner_embeds = self.ner_embedding(ner_tags)
        fused = torch.cat([encoder_hidden, ner_embeds], dim=-1)
        fused = self.fusion(fused)
        
        # Project to decoder dimension
        encoder_hidden_projected = self.projection(fused)
        
        # Decode
        if tgt_input_ids is not None:
            outputs = self.decoder(
                encoder_outputs=(encoder_hidden_projected,),
                attention_mask=src_attention_mask,
                labels=tgt_input_ids,
                decoder_attention_mask=tgt_attention_mask
            )
            return outputs
        else:
            return encoder_hidden_projected
    
    def generate(self, src_input_ids, src_attention_mask, ner_tags, max_length=128):
        encoder_hidden_projected = self.forward(
            src_input_ids, src_attention_mask, ner_tags
        )
        
        generated = self.decoder.generate(
            encoder_outputs=(encoder_hidden_projected,),
            attention_mask=src_attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        return generated