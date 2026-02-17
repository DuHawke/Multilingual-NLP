"""
File: src/data_utils.py
Mô tả: Data processing utilities
"""

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Any


class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, source_lang='en', target_lang='vi'):
        self.dataset    = hf_dataset
        self.source_lang = source_lang
        self.target_lang = target_lang

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Hỗ trợ cả 2 format
        if 'translation' in item:
            return {
                'source': item['translation'][self.source_lang],
                'target': item['translation'][self.target_lang],
            }
        else:
            return {
                'source': item['source'],
                'target': item['target'],
            }


@dataclass
class NERTranslationDataCollator:
    """Data collator cho NER Translation task"""
    
    model: Any  # MultilingualNERTranslationModel
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        source_texts = [f['source'] for f in features]
        target_texts = [f['target'] for f in features]
        
        return {
            'source_texts': source_texts,
            'target_texts': target_texts
        }


def filter_by_length(example, min_len=3, max_len=150, 
                     source_lang='en', target_lang='vi'):
    """Lọc câu quá ngắn hoặc quá dài — hỗ trợ format translation"""
    # Lấy text theo đúng format của dataset
    if 'translation' in example:
        src_text = example['translation'][source_lang]
        tgt_text = example['translation'][target_lang]
    else:
        src_text = example['source']
        tgt_text = example['target']

    src_words = len(src_text.split())
    tgt_words = len(tgt_text.split())
    return (min_len <= src_words <= max_len) and (min_len <= tgt_words <= max_len)


def prepare_dataset(dataset, min_len=3, max_len=150,
                    source_lang='en', target_lang='vi'):
    """Chuẩn bị dataset với filtering"""
    filter_fn = lambda x: filter_by_length(
        x, min_len, max_len, source_lang, target_lang
    )
    dataset['train']      = dataset['train'].filter(filter_fn)
    dataset['validation'] = dataset['validation'].filter(filter_fn)
    return dataset