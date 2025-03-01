import argparse
from html import parser
import os
import time


from datasets import load_dataset
from src import logger
from src.configs.configuration import Config
from src.tamil_tokenizer.tamil_tokenizer import train_custom_tokenizer
from src.utils.common import (
    create_directory,
    chunk_filereader,
    split_train_test,
    filter_sentences,
)
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerFast,
    AutoConfig,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def filtering_and_data_splitting(data_path: str):

    # Filtering
    if not os.path.exists(Config.filtered_data):
        logger.info("Filtering data")
        filter_sentences(
            file_path=data_path,
            max_len=Config.context_length,
            output_file=Config.filtered_data,
        )

    # data splitting
    if not os.path.exists(f"dataset/v2/tamil_train.txt"):

        # create a new version of the dataset
        create_directory(dir_name="dataset/v2")

        split_dict = {
            "train": open("dataset/v2/tamil_train.txt", "w", encoding="utf-8"),
            "test": open("dataset/v2/tamil_test.txt", "w", encoding="utf-8"),
        }

        context_length = 512

        # reading the files as blocks
        with open(data_path, "r", encoding="utf-8") as datafile:
            with tqdm(
                total=os.path.getsize(data_path),
                unit="B",
                unit_scale=True,
                desc="Reading..",
            ) as pbar:
                for chunk in chunk_filereader(fp=datafile, block_size=8192000):

                    if len(chunk) >= context_length:
                        split = split_train_test()
                        split_dict[split].write(chunk)

                    pbar.update(len(chunk.encode("utf-8")))

            # closing the files
            for file in split_dict.values():
                file.close()

        logger.info("Data splitting completed")
    else:
        logger.info("Data already splitted")


def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=Config.context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == Config.context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tamil Language Model")
    parser.add_argument(
        "--raw_data", required=True, type=str, help="Path to the raw data"
    )

    # parsing the arguments
    args = parser.parse_args()

    logger.info(f"Starting the training pipeline with : {args}")
    filtering_and_data_splitting(data_path=args.raw_data)

    # Training a custom tokenizer on the filtered data
    logger.info("Training custom tokenizer - ByteLevelBPE")
    logger.info(
        "This may take a while as it trains only on CPU, take a sip of coffee... :)"
    )
    if not os.path.exists(Config.tokenizer_path):
        train_custom_tokenizer(
            filtered_datapath=Config.filtered_data,
            vocab_size=Config.vocab_size,
            save_path=Config.tokenizer_path,
        )
        logger.info("Custom tokenizer trained successfully")

    # Training pipeline setup
    data_files = {
        "train": "dataset/v2/tamil_train.txt",
        "test": "dataset/v2/tamil_test.txt",
    }

    raw_dataset = load_dataset("text", data_files=data_files, streaming=True)

    # loading the custom tokenizer
    logger.info("Loading custom tokenizer")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=Config.tokenizer_path)

    # tokenizing the dataset
    logger.info("Tokenizing the dataset")
    tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])
    logger.info("Dataset tokenized successfully")

    # loading the model
    logger.info("Loading GPT-2 model")
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=Config.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    logger.info("Model loaded successfully")

    # 1. First, add the pad token more explicitly
    if tokenizer.pad_token is None:
        # Method 1: Set pad_token to eos_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # Method 2: If Method 1 doesn't work, try adding a special token
        # This is more reliable as it modifies the tokenizer's vocabulary
        special_tokens_dict = {"pad_token": "[PAD]"}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens: {special_tokens_dict}")

    # If working with a model, resize embeddings to match new vocabulary size
    # model.resize_token_embeddings(len(tokenizer))

    # 2. Verify that pad token is set
    logger.info(f"Pad token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    model.resize_token_embeddings(len(tokenizer))
    model.init_weights()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # training the model
    logger.info("Training the model")

    args = TrainingArguments(
        output_dir="artifacts",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        max_steps=10_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        push_to_hub=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    trainer.train()
    logger.info("Training completed successfully")
    trainer.save_model("artifacts")
    logger.info("Model saved successfully")
    logger.info("Training pipeline completed successfully")
    logger.info("Exiting the program")
