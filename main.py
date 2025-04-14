import torch
import torch.nn as nn
from helper.ParallelDataset import ParallelDataset
import wandb
import nltk
from torch.utils.data import DataLoader, random_split
from model.BengaliEnglishNMT import BengaliEnglishNMT
from torch.optim import Adam
from helper.ParallelDataset import collate_fn
from helper.Evaluate import evaluate_model
import torch.cuda.amp
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import OneCycleLR

nltk.download('punkt')
nltk.download('wordnet')

# Initialize wandb
wandb.init(project="bn-en-nmt")

# Configuration
config = wandb.config
config.batch_size = 32
config.num_epochs = 10
config.learning_rate = 5e-5
config.max_length = 64
config.split_ratios = [0.7, 0.2, 0.1]  # Train, Val, Test
config.gradient_accumulation_steps = 4

def main():
    # cuda optimization
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    full_dataset = ParallelDataset('./dataset/data/corpus.train.en', './dataset/data/corpus.train.bn',
                                   './dataset/vocab/en.model', './dataset/vocab/bn.model')

    # Split dataset
    train_size = int(config.split_ratios[0] * len(full_dataset))
    val_size = int(config.split_ratios[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = BengaliEnglishNMT(
        mbert_model_name='bert-base-multilingual-cased',
        tgt_vocab_size=32000,  # Adjust based on vocab
        bn_spm_path='./dataset/vocab/bn.model'
    ).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=len(train_loader) * config.num_epochs // config.gradient_accumulation_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        train_progress = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{config.num_epochs}",
                            leave=True)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            # Prepare inputs
            src_input = batch['en_tokens'].to(device)
            src_mask = batch['en_mask'].to(device)
            tgt_input = batch['bn_tokens'][:, :-1].to(device)
            tgt_output = batch['bn_tokens'][:, 1:].to(device)


            # mixed prec forward
            with torch.cuda.amp.autocast():
                outputs = model(src_input, src_mask, tgt_input)
                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Step the scheduler
                optimizer.zero_grad(set_to_none=True)

            # Track loss (adjust for gradient accumulation for reporting)
            current_loss = loss.item() * config.gradient_accumulation_steps
            total_loss += current_loss

            # Update progress bar
            train_progress.set_postfix({
                'loss': f"{current_loss:.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.7f}"
            })
            # Log intermediate results
            if batch_idx % 100 == 0:
                wandb.log({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'train_batch_loss': current_loss,
                    'learning_rate': scheduler.get_last_lr()[0]
                })

        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_loader, device)

        # Log metrics
        avg_train_loss = total_loss / len(train_loader)
        wandb.log({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': val_metrics['loss'],
        'val_bleu': val_metrics['bleu'],
        'val_meteor': val_metrics['meteor']
        })

        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val BLEU: {val_metrics['bleu']:.4f}")
        print(f"Val METEOR: {val_metrics['meteor']:.4f}\n")


    # Final evaluation on test set
    test_metrics = evaluate_model(model, test_loader, device)
    wandb.log({
        'test_loss': test_metrics['loss'],
        'test_bleu': test_metrics['bleu'],
        'test_meteor': test_metrics['meteor']
    })

    print("\nFinal Test Results:")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test BLEU: {test_metrics['bleu']:.4f}")
    print(f"Test METEOR: {test_metrics['meteor']:.4f}")

    # Save model
    torch.save(model.state_dict(), 'nmt_model_final.pt')
    wandb.save('nmt_model_final.pt')

if __name__ == '__main__':
    main()
