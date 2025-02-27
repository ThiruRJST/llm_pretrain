from dataclasses import dataclass


@dataclass
class Config:
    pretrain_raw_data = "dataset/tamil_pretrain.txt"
    context_length = 512
    filtered_data = "dataset/filtered_cl_tamil_pretrain.txt"
    vocab_size = 30000
    tokenizer_path = "artifacts/tokenizer.json"
