import os
import numpy as np
import random
from tqdm.auto import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers import AdamW, get_scheduler

learning_rate = 1e-5
batch_size = 4
epoch_num = 3

seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

categories = set()

class PeopleDaily(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, labels = '', []
                for i, item in enumerate(line.split('\n')):
                    char, tag = item.split(' ')
                    sentence += char
                    if tag.startswith('B'):
                        labels.append([i, i, char, tag[2:]]) # Remove the B- or I-
                        categories.add(tag[2:])
                    elif tag.startswith('I'):
                        labels[-1][1] = i
                        labels[-1][2] += char
                Data[idx] = {
                    'sentence': sentence, 
                    'labels': labels
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = PeopleDaily('data/china-people-daily-ner-corpus/example.train')
valid_data = PeopleDaily('data/china-people-daily-ner-corpus/example.dev')
test_data = PeopleDaily('data/china-people-daily-ner-corpus/example.test')

id2label = {0:'O'}
for c in list(sorted(categories)):
    id2label[len(id2label)] = f"B-{c}"
    id2label[len(id2label)] = f"I-{c}"
label2id = {v: k for k, v in id2label.items()}


class BertForNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, len(id2label))
        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

config = AutoConfig.from_pretrained(checkpoint)
model = BertForNER.from_pretrained(checkpoint, config=config).to(device)



sentence = '日本外务省3月18日发布消息称，日本首相岸田文雄将于19至21日访问印度和柬埔寨。'

model.load_state_dict(
    torch.load('models_fune_tuning/BertForNER/epoch_3_valid_macrof1_95.944_microf1_96.071_weights.bin', map_location=torch.device(device))
)
model.eval()
results = []
with torch.no_grad():
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop('offset_mapping').squeeze(0)
    inputs = inputs.to(device)
    pred = model(inputs)
    probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].cpu().numpy().tolist()
    predictions = pred.argmax(dim=-1)[0].cpu().numpy().tolist()

    pred_label = []
    idx = 0
    while idx < len(predictions):
        pred = predictions[idx]
        label = id2label[pred]
        if label != "O":
            label = label[2:] # Remove the B- or I-
            start, end = offsets[idx]
            all_scores = [probabilities[idx][pred]]
            # Grab all the tokens labeled with I-label
            while (
                idx + 1 < len(predictions) and 
                id2label[predictions[idx + 1]] == f"I-{label}"
            ):
                all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = offsets[idx + 1]
                idx += 1

            score = np.mean(all_scores).item()
            start, end = start.item(), end.item()
            word = sentence[start:end]
            pred_label.append(
                {
                    "entity_group": label,
                    "score": score,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )
        idx += 1
    # print(pred_label)
    for item in pred_label:
        print(item)