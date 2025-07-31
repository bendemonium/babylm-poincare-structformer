import glob
from transformers import AutoTokenizer
import numpy as np
import pickle

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def read_files(file_pattern):
    files = glob.glob(file_pattern)
    texts = []
    for file in files:
        with open(file, encoding="utf-8") as f:
            texts.extend(line.strip() for line in f if line.strip())
    return texts

def tokenize_and_save(texts, out_file, batch_size=1024, max_length=128):
    # Tokenize all at once for simplicity
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np"
    )
    # Save a dict with input_ids and attention_mask
    with open(out_file, "wb") as f:
        pickle.dump(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"]
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )
    print(f"Saved tokenized data to {out_file}")

if __name__ == "__main__":
    train_texts = read_files("data/train_10M/*.train")
    print(f"Loaded {len(train_texts)} training samples.")

    tokenize_and_save(train_texts, "data/train_tokenized.pkl", batch_size=1024, max_length=128)