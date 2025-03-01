import mmap
import numpy as np
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import tiktoken

with open("data.txt", 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)

train_data = data[: int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)) :]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
test_ids = enc.encode_ordinary(test_data)

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


def get_batch(split, block_size=1024, batch_size=16):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap('/home/tensorthiru/llm_pretrain/minimalai/train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('/home/tensorthiru/llm_pretrain/minimalai/val.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    return x, y

# def create_batches(data, batch_size=16, block_size=256):

#     tokens = []
#     for x in data:
#         tokens.extend(wrapped_tokenizer.encode(x))

#     ix = torch.randint(0, len(tokens) - block_size, (batch_size,))
#     x = torch.from_numpy(np.array([tokens[i : i + block_size] for i in ix]))
#     y = torch.from_numpy(np.array([tokens[i + 1 : i + block_size + 1] for i in ix]))

#     return x, y


def create_bins_from_raw_data(tokenizer_path: str, data_dict: dict, bin_filepath: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )

    for mode in ["train", "test"]:
        datapath = data_dict[mode]
        tokens_list = []
        with open(
            datapath,
            "r",
        ) as datafile:
            mmaped_data = mmap.mmap(datafile.fileno(), 0, access=mmap.ACCESS_READ)
            with tqdm(
                total=os.path.getsize(datapath), desc=f"Creating {mode} bin file"
            ) as pbar:
                with open(bin_filepath + f"/{mode}.bin", "wb") as binfile:
                    for line in iter(mmaped_data.readline, b""):
                        sentence = line.decode("utf-8").strip()
                        if sentence:
                            tokens = wrapped_tokenizer.encode(sentence)
                            tokens_list.extend(tokens)

                            pbar.update(len(line))
                    binfile.write(bytes(tokens))
