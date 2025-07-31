import argparse
import os
import glob
import pickle
from transformers import AutoTokenizer
from tqdm import tqdm

def read_raw_lines(file_pattern):
    """Fast file loader that yields non-empty, stripped lines one at a time."""
    for file_path in glob.glob(file_pattern):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    yield stripped

def batch_iterator(iterable, batch_size):
    """Yield (lists of) items from any iterable in batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def tokenize_and_save(split, max_length=128, batch_size=8192):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_path = os.path.join("data", split, f"*.{split}")
    line_iter = read_raw_lines(raw_path)

    os.makedirs("data/tokens", exist_ok=True)
    out_file = os.path.join("data/tokens", f"{split}_tokenized.pkl")

    input_ids_list = []
    attention_mask_list = []
    total = 0

    for batch in tqdm(batch_iterator(line_iter, batch_size), desc=f"Tokenizing {split}", unit="batch"):
        encoded = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np"  
        )
        input_ids_list.append(encoded["input_ids"])
        attention_mask_list.append(encoded["attention_mask"])
        total += len(batch)

    # Concatenate all batches
    import numpy as np
    input_ids = np.concatenate(input_ids_list, axis=0)
    attention_mask = np.concatenate(attention_mask_list, axis=0)

    # Save to pkl
    with open(out_file, "wb") as f:
        pickle.dump({"input_ids": input_ids, "attention_mask": attention_mask}, f)
    print(f"✅ Tokenized {total} {split} examples and saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAST, batch-optimized tokenization.")
    parser.add_argument("--split", type=str, required=True, help="train, dev, or test")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8192, help="Larger batches = faster. Tune for your RAM.")
    args = parser.parse_args()
    tokenize_and_save(args.split, max_length=args.max_length, batch_size=args.batch_size)
