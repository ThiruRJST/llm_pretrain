import argparse
from html import parser
import os
import time


from src import logger
from src.utils.common import (
    create_directory,
    chunk_filereader,
    split_train_test
)
from tqdm import tqdm



def data_splitting(data_path: str):
    #data splitting
    if not os.path.exists(f"dataset/v2/tamil_train.txt"):

        #create a new version of the dataset
        create_directory(dir_name="dataset/v2")

        split_dict = {
            "train": open("dataset/v2/tamil_train.txt", "w", encoding="utf-8"),
            "test": open("dataset/v2/tamil_test.txt", "w", encoding="utf-8")
        }

        context_length = 512

        #reading the files as blocks
        with open(data_path, "r", encoding="utf-8") as datafile:
            with tqdm(total=os.path.getsize(data_path), unit="B", unit_scale=True, desc="Reading..") as pbar:
                for chunk in chunk_filereader(fp=datafile, block_size=8192000):

                    if len(chunk) >= context_length:
                        split = split_train_test()
                        split_dict[split].write(chunk)
                    
                    pbar.update(len(chunk.encode("utf-8")))
            
            #closing the files
            for file in split_dict.values():
                file.close()

        logger.info("Data splitting completed")
    else:
        logger.info("Data already splitted")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tamil Language Model")
    parser.add_argument("--raw_data", required=True, type=str, default="dataset/v2/tamil_train.txt", help="Path to the raw data")

    #parsing the arguments
    args = parser.parse_args()

    #data splitting
    data_splitting(data_path=args.raw_data)