import torch
import torch.nn as nn
from helper.ParallelDataset import ParallelDataset
import wandb,os
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
from transformers import BertTokenizer

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
config.use_subset = True
config.subset_size = 1000


def main():

    # Print training info
    print(f"Training with config:")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"- Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"- Learning rate: {config.learning_rate}")
    print(f"- Using subset: {config.use_subset}")
    if config.use_subset:
        print(f"- Subset size: {config.subset_size:,}")
    # cuda optimization
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    en_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # Load dataset with the correct parameters
    full_dataset = ParallelDataset(
        en_file='./dataset/data/corpus.train.en',
        bn_file='./dataset/data/corpus.train.bn',
        en_tokenizer=en_tokenizer,  # Pass the tokenizer object
        bn_spm_path='./dataset/vocab/bn.model',
        max_length=config.max_length
    )
    print(f"Full dataset size: {len(full_dataset):,} sentence pairs")
    for i in range(3):
        sample = full_dataset[i]
        print(f"\nSample {i}:")

        # English
        en_tokens = sample['en_tokens'].tolist()
        en_non_pad = [t for t in en_tokens if t != 0]
        print("English tokens:", en_non_pad)
        print("English text:", en_tokenizer.decode(en_non_pad, skip_special_tokens=True))

        # Bengali
        bn_tokens = sample['bn_tokens'].tolist()
        bn_non_pad = [t for t in bn_tokens if t != 0 and t != 1 and t != 2]  # Remove padding and special tokens
        print("Bengali tokens:", bn_non_pad)
        print("Bengali text:", full_dataset.bn_spm.decode(bn_non_pad))

    # Create a subset for faster development if configured
    if config.use_subset:
        subset_size = min(config.subset_size, len(full_dataset))
        print(f"Using a subset of {subset_size:,} samples out of {len(full_dataset):,}")

        # Generate random indices
        indices = torch.randperm(len(full_dataset))[:subset_size].tolist()

        # Create truncated sentence lists
        en_sentences_subset = [full_dataset.en_sentences[i] for i in indices]
        bn_sentences_subset = [full_dataset.bn_sentences[i] for i in indices]

        # Replace with subset
        full_dataset.en_sentences = en_sentences_subset
        full_dataset.bn_sentences = bn_sentences_subset

        working_dataset = full_dataset
        print(f"Created subset with {len(working_dataset.en_sentences):,} samples")
    else:
        working_dataset = full_dataset
        print(f"Using the full dataset with {len(full_dataset):,} samples")

    # Split dataset
    train_size = int(config.split_ratios[0] * len(working_dataset))
    val_size = int(config.split_ratios[1] * len(working_dataset))
    test_size = len(working_dataset) - train_size - val_size

    print(f"Split sizes: Train={train_size:,}, Val={val_size:,}, Test={test_size:,}")

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

    print(f"Number of batches: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    # Initialize model
    model = BengaliEnglishNMT(
        mbert_model_name='bert-base-multilingual-cased',
        tgt_vocab_size=32000,  # Adjust based on vocab
        bn_spm_path='./dataset/vocab/bn.model'
    ).to(device)

    # Check vocabulary size
    bn_spm_vocab_size = model.bn_spm.get_piece_size()
    model_vocab_size = model.output_projection.out_features
    print(f"Bengali SPM vocabulary size: {bn_spm_vocab_size}")
    print(f"Model vocabulary size: {model_vocab_size}")

    if bn_spm_vocab_size > model_vocab_size:
        print(f"WARNING: SPM vocabulary ({bn_spm_vocab_size}) is larger than model vocabulary ({model_vocab_size})!")
        print("This may cause out-of-vocabulary issues during training.")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} total parameters")
    print(f"Training {trainable_params:,} parameters ({trainable_params / total_params:.2%} of total)")

    # Optimizer
    optimizer = Adam(model.parameters(),
                     lr=3e-4,
                     betas=(0.9, 0.98),
                     eps=1e-9)
    # Learning rate scheduler
    total_steps = (len(train_loader) * config.num_epochs) // config.gradient_accumulation_steps + 1
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,  # Match optimizer LR
        total_steps=total_steps,
        pct_start=0.3,  # Longer warmup
        anneal_strategy='linear'
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

            # ===== ADD THIS DEBUG CHECK =====
            if batch_idx == 0:
                print("\n=== Input Verification ===")
                # English
                english_tokens = src_input[0].cpu().numpy()
                english_text = model.encoder_tokenizer.decode(english_tokens[english_tokens != 0])  # Skip padding
                print("English:", english_text)
                print("English token IDs:", english_tokens)

                # Bengali
                bengali_tokens = batch['bn_tokens'][0].cpu().numpy()
                bengali_text = model.bn_spm.decode(bengali_tokens[bengali_tokens != 0].tolist())  # Filter out padding
                print("Bengali:", bengali_text)
                print("Bengali token IDs:", bengali_tokens)

                print("English attention mask:", src_mask[0].cpu().numpy())
            # ===== END DEBUG CHECK =====


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

                # Check if we've reached the total step limit
                current_step = (epoch * len(train_loader) + batch_idx + 1) // config.gradient_accumulation_steps
                if current_step < total_steps:
                    scheduler.step()

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
        print(f"\nEvaluating on validation set...")
        val_metrics = evaluate_model(model, val_loader, device)

        # Log metrics
        avg_train_loss = total_loss / len(train_loader)
        wandb.log({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': val_metrics['loss'],
        'val_bleu': val_metrics['bleu'],
        # 'val_meteor': val_metrics['meteor']
        })

        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val BLEU: {val_metrics['bleu']:.4f}")
        # print(f"Val METEOR: {val_metrics['meteor']:.4f}\n")


    # Final evaluation on test set
    test_metrics = evaluate_model(model, test_loader, device)
    wandb.log({
        'test_loss': test_metrics['loss'],
        'test_bleu': test_metrics['bleu'],
        # 'test_meteor': test_metrics['meteor']
    })

    print("\nFinal Test Results:")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test BLEU: {test_metrics['bleu']:.4f}")
    # print(f"Test METEOR: {test_metrics['meteor']:.4f}")

    # Save model
    torch.save(model.state_dict(), 'nmt_model_final.pt')
    wandb.save('nmt_model_final.pt')

if __name__ == '__main__':
    main()
