import time
from tqdm import tqdm

pbar = tqdm(range(100), 
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

for epoch in tqdm(range(5), desc="Epochs", position=0):
    for batch in tqdm(range(50), desc="Batches", position=1, leave=False):
        time.sleep(0.01)