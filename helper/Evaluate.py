import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
import numpy as np


def evaluate_model(model, dataloader, device, max_eval_samples=500):
    model.eval()
    total_loss = 0
    all_references = []
    all_hypotheses = []
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    smooth = SmoothingFunction().method1  # For BLEU smoothing

    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src_input = batch['en_tokens'].to(device)
            src_mask = batch['en_mask'].to(device)
            tgt_input = batch['bn_tokens'][:, :-1].to(device)
            tgt_output = batch['bn_tokens'][:, 1:].to(device)

            # Forward pass
            outputs = model(src_input, src_mask, tgt_input)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()

            for i in range(min(src_input.size(0), max_eval_samples - sample_count)):
                if sample_count >= max_eval_samples:
                    break

                # Get and clean reference sentence
                bn_tokens = batch['bn_tokens'][i][batch['bn_mask'][i].bool()].tolist()
                bn_reference = model.bn_spm.decode([
                    t for t in bn_tokens
                    if t not in {0, model.bn_spm.bos_id(), model.bn_spm.eos_id()}
                ])

                # Get and clean source sentence
                en_tokens = src_input[i][src_mask[i].bool()].tolist()
                en_text = model.encoder_tokenizer.decode([
                    t for t in en_tokens
                    if t not in {0, 101, 102}  # Skip [PAD], [CLS], [SEP]
                ])

                # Generate translation
                translated = generate_translation(
                    model,
                    src_input[i].unsqueeze(0),
                    src_mask[i].unsqueeze(0),
                    device,
                    temperature=0.7
                )

                # Clean generated translation
                cleaned_translation = ' '.join([
                    word for word in translated.split()
                    if word not in {'<s>', '</s>', '▁'}
                ])

                # Debug output
                if sample_count < 5:
                    print(f"\nExample {sample_count + 1}:")
                    print(f"Source: {en_text}")
                    print(f"Reference: {bn_reference}")
                    print(f"Translation: {cleaned_translation}")
                    print(f"Source token length: {len(en_tokens)}")
                    print(f"Reference token length: {len(bn_tokens)}")
                    print(f"Translation token length: {len(model.bn_spm.encode(translated))}")

                # Prepare for BLEU calculation
                ref_words = [w for w in bn_reference.split() if w.strip()]
                hyp_words = [w for w in cleaned_translation.split() if w.strip()]

                if hyp_words:  # Only add non-empty hypotheses
                    all_references.append([ref_words])
                    all_hypotheses.append(hyp_words)

                sample_count += 1

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)

    if all_hypotheses:
        bleu = corpus_bleu(
            all_references,
            all_hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),  # Use 1-4 grams
            smoothing_function=smooth
        )

        # Calculate word matches
        matches = sum(
            1 for ref, hyp in zip(all_references, all_hypotheses)
            for word in hyp if word in ref[0]
        )
        total_words = sum(len(h) for h in all_hypotheses)

        print(f"\nEvaluation Results:")
        print(f"- Samples evaluated: {len(all_hypotheses)}")
        print(f"- Word match: {matches}/{total_words} ({matches / max(1, total_words) * 100:.2f}%)")
        print(f"- BLEU score: {bleu:.4f}")
    else:
        bleu = 0.0
        print("Warning: No valid translations for BLEU calculation")

    return {
        'loss': avg_loss,
        'bleu': bleu,
    }


def generate_translation(model, src_tensor, src_mask, device, max_length=64, temperature=0.8):
    model.eval()

    # Encode source
    encoder_output = model.encoder(src_tensor, src_mask)
    memory = model.hierarchical_attention(encoder_output)

    # Initialize with BOS token
    tgt_tokens = [model.bn_spm.bos_id()]

    for _ in range(max_length):
        tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
        tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)

        with torch.no_grad():
            output = model.decode(memory, tgt_tensor, tgt_mask)

        # Temperature sampling
        if temperature != 1.0:
            logits = output[0, -1] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = output.argmax(-1)[0, -1].item()

        tgt_tokens.append(next_token)

        # Stop at EOS or max length
        if next_token == model.bn_spm.eos_id() or len(tgt_tokens) >= max_length:
            break

    # Decode and clean special tokens
    decoded = model.bn_spm.decode(tgt_tokens)
    return ' '.join([word for word in decoded.split()
                     if word not in {'<s>', '</s>'}])