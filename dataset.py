import torch
from torch.utils.data import Dataset
import pandas as pd
import spacy

class NERProcessor:
    def __init__(self, lang='xx'):
        if lang == 'xx':
            self.nlp = spacy.load('xx_ent_wiki_sm')
        elif lang == 'en':
            self.nlp = spacy.load('en_core_web_sm')
        elif lang == 'vi':
            # Fallback to multilingual for Vietnamese
            self.nlp = spacy.load('xx_ent_wiki_sm')
    
    def get_ner_tags(self, text, tokenizer, ner_vocab, max_length=128):
        doc = self.nlp(text)
        
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        
        ner_tags = [0] * max_length
        offsets = encoding['offset_mapping']
        
        for ent in doc.ents:
            ent_start, ent_end = ent.start_char, ent.end_char
            ent_label = ent.label_
            ner_id = ner_vocab.get(ent_label, 0)
            
            for idx, (start, end) in enumerate(offsets):
                if start >= ent_start and end <= ent_end and end > 0:
                    ner_tags[idx] = ner_id
        
        return ner_tags

class TranslationDataset(Dataset):
    def __init__(self, csv_path, config, ner_processor):
        self.config = config
        self.ner_processor = ner_processor
        
        # Load data
        df = pd.read_csv(csv_path)
        self.src_texts = df['source'].tolist()
        self.tgt_texts = df['target'].tolist()
        
        print(f"Loaded {len(self.src_texts)} examples from {csv_path}")
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = str(self.src_texts[idx])
        tgt_text = str(self.tgt_texts[idx])
        
        # Import tokenizers (assuming they're passed via config or global)
        from transformers import BertTokenizer, T5Tokenizer
        src_tokenizer = BertTokenizer.from_pretrained(self.config.MBERT_MODEL)
        tgt_tokenizer = T5Tokenizer.from_pretrained(self.config.MT5_MODEL)
        
        # Tokenize source
        src_encoding = src_tokenizer(
            src_text,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get NER tags
        ner_tags = self.ner_processor.get_ner_tags(
            src_text, src_tokenizer, self.config.NER_VOCAB, self.config.MAX_LENGTH
        )
        
        # Tokenize target
        tgt_encoding = tgt_tokenizer(
            tgt_text,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'src_input_ids': src_encoding['input_ids'].squeeze(),
            'src_attention_mask': src_encoding['attention_mask'].squeeze(),
            'ner_tags': torch.tensor(ner_tags, dtype=torch.long),
            'tgt_input_ids': tgt_encoding['input_ids'].squeeze(),
            'tgt_attention_mask': tgt_encoding['attention_mask'].squeeze()
        }