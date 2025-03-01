import mmap
import numpy as np
import os

from ensure import ensure_annotations
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


@ensure_annotations
def create_bins_from_raw_data(
    tokenizer_path: str, context_length: int, data_dict: dict, bin_filepath: str
):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )

    for mode in ["train", "test"]:
        datapath = data_dict[mode]

        with open(
            datapath,
            "r",
        ) as datafile:
            mmaped_data = mmap.mmap(datafile.fileno(), 0, access=mmap.ACCESS_READ)
            with tqdm(
                total=os.path.getsize(datapath), desc=f"Creating {mode} bin file"
            ) as pbar:
                with open(bin_filepath + f"/{mode}.npy", "wb") as binfile:
                    for line in iter(mmaped_data.readline, b""):
                        sentence = line.decode("utf-8").strip()
                        if sentence:
                            tokens = wrapped_tokenizer.encode(sentence)
                            binfile.write(np.array(tokens))
                            pbar.update(len(line))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path", required=True, type=str, help="Path to the tokenizer"
    )
    parser.add_argument(
        "--train_path", required=True, type=str, help="Path to training data"
    )
    parser.add_argument(
        "--test_path", required=True, type=str, help="Path to testing data"
    )
    parser.add_argument(
        "--bin_filepath", required=True, type=str, help="Path to the bin file"
    )
    parser.add_argument(
        "--context_length", required=True, type=int, help="Context length of the model"
    )

    args = parser.parse_args()

    data_dict = {"train": args.train_path, "test": args.test_path}

    create_bins_from_raw_data(
        args.tokenizer_path, args.context_length, data_dict, args.bin_filepath
    )
