import mmap
import numpy as np
import os
import re

from ensure import ensure_annotations
from pathlib import Path
from typing import Optional, Union


@ensure_annotations
def create_directory(dir_name: Union[str, Path]):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


@ensure_annotations
def chunk_filereader(fp, block_size: int = 8192):
    """
    A Generator that reads a txt file into chunks

    Arguments
    fp  Path(or)str absolute path of text dataset
    block_size int chunk size in bytes

    Generates blocks of data

    """
    while True:
        data_chunk = fp.read(block_size)
        if not data_chunk:
            break
        yield data_chunk


@ensure_annotations
def split_train_test():
    if np.random.randn() < 0.8:
        return "train"
    else:
        return "test"


@ensure_annotations
def clean_whitespaces(text: str):
    return re.sub(r"\s+", " ", text).strip()


@ensure_annotations
def filter_sentences(file_path, max_len, output_file):
    with open(file_path, "r", encoding="utf-8") as f:

        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        with open(output_file, "w") as out_file:
            for line in iter(mmapped_file.readline, b""):
                sentence = line.decode("utf-8").strip()
                sentence = clean_whitespaces(sentence)
                if sentence:
                    if len(sentence) >= max_len:
                        out_file.write(sentence + "\n")
        mmapped_file.close()
