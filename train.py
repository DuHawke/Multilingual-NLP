import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json
from datetime import datetime

from config import Config
from model import NEREnhancedTranslationModel
from dataset import TranslationDataset, NERProcessor
from utils import save_checkpoint, load_checkpoint, calculate_bleu

def train_epoch(model, train_loader, optimizer, scheduler, config, epoch):
    model.train()
    total_loss = 0
    step = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        src_ids = batch['src_input_ids'].to(config.DEVICE)
        src_mask = batch['src_attention_mask'].to(config.DEVICE)
        ner_tags = batch['ner_tags'].to(config.DEVICE)
        tgt_ids = batch['tgt_input_ids'].to(config.DEVICE)
        tgt_mask = batch['tgt_attention_mask'].to(config.DEVICE)
        
        # Forward
        outputs = model(
            src_input_ids=src_ids,
            src_attention_mask=src_mask,
            ner_tags=ner_tags,
            tgt_input_ids=tgt_ids,
            tgt_attention_mask=tgt_mask
        )
        
        loss = outputs.loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        
        if step % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    return total_loss / len(train_loader)

def validate(model, val_loader, config):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            src_ids = batch['src_input_ids'].to(config.DEVICE)
            src_mask = batch['src_attention_mask'].to(config.DEVICE)
            ner_tags = batch['ner_tags'].to(config.DEVICE)
            tgt_ids = batch['tgt_input_ids'].to(config.DEVICE)
            tgt_mask = batch['tgt_attention_mask'].to(config.DEVICE)
            
            outputs = model(
                src_input_ids=src_ids,
                src_attention_mask=src_mask,
                ner_tags=ner_tags,
                tgt_input_ids=tgt_ids,
                tgt_attention_mask=tgt_mask
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(val_loader)

def train(config):
    # Create directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    
    # Initialize NER processor
    print("Loading spaCy models...")
    ner_processor = NERProcessor('en')  # Thay đổi theo ngôn ngữ nguồn
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = TranslationDataset(config.TRAIN_DATA_PATH, config, ner_processor)
    val_dataset = TranslationDataset(config.VAL_DATA_PATH, config, ner_processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model
    print("Initializing model...")
    model = NEREnhancedTranslationModel(config)
    model.to(config.DEVICE)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\n{'='*50}")
    print(f"Starting training on {config.DEVICE}")
    print(f"{'='*50}\n")
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, epoch)
        val_loss = validate(model, val_loader, config)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f'  ✓ Best model saved! (Val Loss: {val_loss:.4f})')
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            config.CHECKPOINT_PATH,
            f'checkpoint_epoch_{epoch+1}.pt'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        print()
    
    # Save training history
    with open(os.path.join(config.MODEL_SAVE_PATH, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("Training completed!")
    return model, history

if __name__ == '__main__':
    config = Config()
    model, history = train(config)