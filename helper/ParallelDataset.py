import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence





class ParallelDataset(Dataset):
    def __init__(self, en_file, bn_file, en_spm, bn_spm, max_length=128):
        self.en_sentences = open(en_file, 'r', encoding='utf-8').read().splitlines()
        self.bn_sentences = open(bn_file, 'r', encoding='utf-8').read().splitlines()
        assert len(self.en_sentences) == len(self.bn_sentences)

        self.en_spm = spm.SentencePieceProcessor(model_file=en_spm)
        self.bn_spm = spm.SentencePieceProcessor(model_file=bn_spm)
        self.max_length = max_length

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en_tokens = self.en_spm.encode(self.en_sentences[idx])
        bn_tokens = self.bn_spm.encode(self.bn_sentences[idx])

        en_tokens = en_tokens[:self.max_length - 1] + [self.en_spm.eos_id()]
        bn_tokens = bn_tokens[:self.max_length - 1] + [self.bn_spm.eos_id()]

        return {
            'en_tokens': torch.tensor(en_tokens, dtype=torch.long),
            'bn_tokens': torch.tensor(bn_tokens, dtype=torch.long)
        }


def collate_fn(batch):
    en_tokens = [item['en_tokens'] for item in batch]
    bn_tokens = [item['bn_tokens'] for item in batch]

    en_padded = pad_sequence(en_tokens, batch_first=True, padding_value=0)
    bn_padded = pad_sequence(bn_tokens, batch_first=True, padding_value=0)

    en_mask = (en_padded != 0).float()
    bn_mask = (bn_padded != 0).float()

    return {
        'en_tokens': en_padded,
        'en_mask': en_mask,
        'bn_tokens': bn_padded,
        'bn_mask': bn_mask
    }





