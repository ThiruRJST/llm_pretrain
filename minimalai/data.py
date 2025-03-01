import token
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from functools import partial
import numpy as np

data = [x for x in open("data.txt", "r").readlines() if x.strip()]

tokenizer = Tokenizer.from_file("tokenizer.json")
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
)

def create_batches(data, batch_size=16, block_size=256):
    tokens = []
    for x in data:
        tokens.extend(wrapped_tokenizer.encode(x))
    
    ix = torch.randint(0, len(tokens) - block_size, (batch_size,))
    x = torch.from_numpy(np.array([tokens[i: i + block_size] for i in ix]))
    y = torch.from_numpy(np.array([tokens[i + 1: i + block_size + 1] for i in ix]))
    
    return x, y