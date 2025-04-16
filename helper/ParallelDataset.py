import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence


class ParallelDataset(Dataset):
    def __init__(self, en_file, bn_file, en_tokenizer, bn_spm_path, max_length=64):
        # Read files with utf-8 encoding
        with open(en_file, 'r', encoding='utf-8') as f:
            self.en_sentences = [line.strip() for line in f]
        with open(bn_file, 'r', encoding='utf-8') as f:
            self.bn_sentences = [line.strip() for line in f]

        assert len(self.en_sentences) == len(self.bn_sentences)

        self.en_tokenizer = en_tokenizer
        self.bn_spm = spm.SentencePieceProcessor(model_file=bn_spm_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        # English: use BERT tokenizer
        en_encoded = self.en_tokenizer(
            self.en_sentences[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Bengali: use SPM
        bn_tokens = [self.bn_spm.bos_id()]  # Start with <s>
        bn_tokens += self.bn_spm.encode(self.bn_sentences[idx])[:self.max_length - 2]
        bn_tokens += [self.bn_spm.eos_id()]  # End with </s>

        # Pad to max_length
        if len(bn_tokens) < self.max_length:
            bn_tokens += [0] * (self.max_length - len(bn_tokens))

        return {
            'en_tokens': en_encoded['input_ids'].squeeze(0),
            'en_mask': en_encoded['attention_mask'].squeeze(0),
            'bn_tokens': torch.tensor(bn_tokens, dtype=torch.long)
        }





def collate_fn(batch):
    # Since we already padded in __getitem__, just stack the tensors
    return {
        'en_tokens': torch.stack([item['en_tokens'] for item in batch]),
        'en_mask': torch.stack([item['en_mask'] for item in batch]),
        'bn_tokens': torch.stack([item['bn_tokens'] for item in batch]),
        'bn_mask': (torch.stack([item['bn_tokens'] for item in batch]) != 0).float()
    }





