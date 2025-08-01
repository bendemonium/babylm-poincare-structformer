import argparse
import os
import glob
import pickle
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
HF_TOKENIZED_DATASET_REPO = os.getenv("HF_TOKENIZED_DATASET_REPO", None)

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

def upload_to_huggingface(local_path, hf_repo_id, path_in_repo=None):
    if not hf_repo_id:
        return  # Skip if not configured
    if path_in_repo is None:
        path_in_repo = os.path.basename(local_path)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=hf_repo_id,
        repo_type="dataset",
        commit_message=f"Add {path_in_repo}"
    )
    print(f"☁ Uploaded {path_in_repo} to Hugging Face Hub dataset: {hf_repo_id}")

def tokenize_and_save(split, max_length=128, batch_size=8192):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_path = os.path.join("data", split, f"*.{split}")
    line_iter = read_raw_lines(raw_path)

    os.makedirs("data/tokens", exist_ok=True)
    local_out_path = os.path.join("data/tokens", f"{split}_tokenized.pkl")

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

    if not input_ids_list:
        raise ValueError(f"No input was successfully tokenized! Check raw files in: {raw_path}")

    input_ids = np.concatenate(input_ids_list, axis=0)
    attention_mask = np.concatenate(attention_mask_list, axis=0)

    with open(local_out_path, "wb") as f:
        pickle.dump({"input_ids": input_ids, "attention_mask": attention_mask}, f)
    print(f"✅ Tokenized {total} {split} examples and saved to {local_out_path}")

    # ⬆ Upload to Hugging Face dataset if enabled in .env
    if HF_TOKENIZED_DATASET_REPO:
        upload_to_huggingface(local_out_path, HF_TOKENIZED_DATASET_REPO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAST, batch-optimized tokenization + optional upload to Hugging Face.")
    parser.add_argument("--split", type=str, required=True, help="train, dev, or test")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8192, help="Larger batches = faster. Tune for your RAM.")
    args = parser.parse_args()

    tokenize_and_save(args.split, max_length=args.max_length, batch_size=args.batch_size)
