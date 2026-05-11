from torch.utils.data import Dataset
import json
from torch.utils.data import IterableDataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModel
from torch import nn
from huggingface_hub import login
from pathlib import Path


device = torch.device("mps")
login(token="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")

base_dir = Path(__file__).parent
datafile = base_dir / '..' / 'data' / 'afqmc_public' / 'train.json'
datafile = datafile.resolve()

Data = {}

with open(datafile, 'rt') as f:
    for idx, line in enumerate(f):
        sample = json.loads(line.strip())
        Data[idx] = sample

for i in range(5):
    print(Data[i])