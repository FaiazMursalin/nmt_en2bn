import torch.nn as nn
from transformers import BertModel, BertTokenizer


class mBERTEncoder(nn.Module):
    def __init__(self, model_name='bert-base-multilingual-cased'):
        super().__init__()
        # using pretrained bert model
        self.bert = BertModel.from_pretrained(model_name)
        # using pretrained bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # freeze if needed
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

