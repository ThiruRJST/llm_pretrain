from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


def train_custom_tokenizer(filtered_datapath: str, vocab_size: int, save_path: str):
    tamil_tokenizer = Tokenizer(models.BPE())
    tamil_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # tamil sample sentence
    sample_text = "வணக்கம் உலகம்"
    tamil_tokenizer.pre_tokenizer.pre_tokenize_str(sample_text)

    # train the tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<|endoftext|>"]
    )
    tamil_tokenizer.model = models.BPE()
    tamil_tokenizer.train([filtered_datapath], trainer=trainer)

    # save the tokenizer
    tamil_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tamil_tokenizer.decoder = decoders.ByteLevel()

    decoded_text = tamil_tokenizer.decode(tamil_tokenizer.encode("வணக்கம் உலகம்").ids)

    assert decoded_text == "வணக்கம் உலகம்", "Tokenizer not trained properly"
    tamil_tokenizer.save(save_path)
