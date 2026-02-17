"""
File: src/data_utils.py
Mô tả: Data processing utilities
"""

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Any


class TranslationDataset(Dataset):
    """Wrapper cho Hugging Face dataset"""
    
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
    item = self.dataset[idx]
    # opus-100 dùng key 'translation' thay vì 'source'/'target'
    return {
        'source': item['translation']['en'],
        'target': item['translation']['vi'],
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


def filter_by_length(example, min_len=3, max_len=150):
    """Lọc câu quá ngắn hoặc quá dài"""
    src_words = len(example['source'].split())
    tgt_words = len(example['target'].split())
    return (min_len <= src_words <= max_len) and (min_len <= tgt_words <= max_len)


def prepare_dataset(dataset, min_len=3, max_len=150):
    """Chuẩn bị dataset với filtering"""
    dataset['train'] = dataset['train'].filter(
        lambda x: filter_by_length(x, min_len, max_len)
    )
    dataset['validation'] = dataset['validation'].filter(
        lambda x: filter_by_length(x, min_len, max_len)
    )
    
    return dataset