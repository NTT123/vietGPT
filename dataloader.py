"""
Load data from disk as an infinite data iterator.
Data is stored in files that match the pattern: tokens_*.npy
"""

from pathlib import Path
from random import Random

import numpy as np


def create_data_iterator(data_dir: str, batch_size: int, seq_len: int) -> np.ndarray:
    """
    Load *.npy files from data_dir and create an infinite iterator.
    """
    token_files = Path(data_dir).glob("tokens_*.npy")
    token_files = sorted(token_files)
    # map path to str
    token_files = map(str, token_files)
    tokens = [np.load(file, allow_pickle=True) for file in token_files]
    rand = Random(42)
    batch = []
    # an infinite data stream
    while True:
        # shuffle the data
        rand.shuffle(tokens)
        # concatenate the data
        all_tokens = np.concatenate(tokens)
        num_iter = len(all_tokens) // seq_len
        for _ in range(num_iter):
            # add a random segment of length seq_len
            start = rand.randint(0, len(all_tokens) - seq_len)
            end = start + seq_len
            batch.append(all_tokens[start:end])
            if len(batch) == batch_size:
                yield np.array(batch, dtype=np.int16)
                batch = []
