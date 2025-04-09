import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import numpy as np


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_references = []
    all_hypotheses = []
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for batch in dataloader:
            # Prepare inputs
            src_input = batch['en_tokens'].to(device)
            src_mask = batch['en_mask'].to(device)
            tgt_input = batch['bn_tokens'][:, :-1].to(device)
            tgt_output = batch['bn_tokens'][:, 1:].to(device)

            # Forward pass
            outputs = model(src_input, src_mask, tgt_input)

            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()

            # Generate translations
            for i in range(src_input.size(0)):
                # Get English sentence
                en_tokens = src_input[i][src_mask[i].bool()].tolist()

                # Get reference Bengali sentence
                bn_tokens = batch['bn_tokens'][i][batch['bn_mask'][i].bool()].tolist()
                bn_reference = model.bn_spm.decode(bn_tokens[1:-1])  # Remove <s> and </s>

                # Generate translation
                translated = generate_translation(model, src_input[i].unsqueeze(0),
                                                  src_mask[i].unsqueeze(0), device)

                all_references.append([bn_reference.split()])
                all_hypotheses.append(translated.split())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    bleu = corpus_bleu(all_references, all_hypotheses)

    # Calculate METEOR (slower, so we sample)
    sample_size = min(500, len(all_references))
    indices = np.random.choice(len(all_references), sample_size, replace=False)
    meteor_scores = [
        meteor_score(all_references[i], all_hypotheses[i])
        for i in indices
    ]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return {
        'loss': avg_loss,
        'bleu': bleu,
        'meteor': avg_meteor
    }


def generate_translation(model, src_tensor, src_mask, device, max_length=128):
    model.eval()

    # Encode source
    encoder_output = model.encoder(src_tensor, src_mask)
    memory = model.hierarchical_attention(encoder_output)

    # Initialize target with <s> token
    tgt_tokens = [model.bn_spm.bos_id()]

    for _ in range(max_length):
        tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
        tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)

        with torch.no_grad():
            output = model.decode(memory, tgt_tensor, tgt_mask)

        next_token = output.argmax(-1)[0, -1].item()
        tgt_tokens.append(next_token)

        if next_token == model.bn_spm.eos_id():
            break

    return model.bn_spm.decode(tgt_tokens[1:-1])
