from torch.utils.data import Dataset
import json
from torch.utils.data import IterableDataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModel
from torch import nn
from huggingface_hub import login

device = torch.device("mps")
login(token="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")


class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample



checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

train_data = AFQMC('data/afqmc_public/train.json')
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)

# print(train_data[0])
# print(len(train_data))

batch_X, batch_y = next(iter(train_dataloader))
# print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
# print('batch_y shape:', batch_y.shape)
# print(batch_X)
# print(batch_y)

class BertForPairwiseCLS(nn.Module):
    def __init__(self):
        super(BertForPairwiseCLS, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits

model = BertForPairwiseCLS().to(device)
print(model)