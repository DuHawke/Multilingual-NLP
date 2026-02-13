import torch

class Config:
    # Model settings
    MBERT_MODEL = 'bert-base-multilingual-cased'
    MT5_MODEL = 'google/mt5-small'  # Hoặc 'google/mt5-base' nếu có đủ RAM
    
    # Training settings
    MAX_LENGTH = 128
    BATCH_SIZE = 8  # Giảm xuống 4 nếu bị OOM
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # NER settings
    NER_VOCAB = {
        'O': 0, 'PERSON': 1, 'ORG': 2, 'LOC': 3,
        'DATE': 4, 'TIME': 5, 'MONEY': 6, 'GPE': 7,
        'PRODUCT': 8, 'EVENT': 9
    }
    
    # Paths
    TRAIN_DATA_PATH = 'data/train.csv'
    VAL_DATA_PATH = 'data/val.csv'
    MODEL_SAVE_PATH = 'saved_models'
    CHECKPOINT_PATH = 'checkpoints'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Logging
    LOG_INTERVAL = 50
    SAVE_INTERVAL = 500