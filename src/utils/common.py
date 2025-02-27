import numpy as np
import os

from ensure import ensure_annotations
from pathlib import Path
from typing import Optional, Union


@ensure_annotations
def create_directory(dir_name: Union[str, Path]):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

@ensure_annotations
def chunk_filereader(fp, block_size:int = 8192):
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
        return 'train'
    else:
        return 'test'