import argparse
import os
import glob
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import HfApi, HfFolder
from dotenv import load_dotenv

load_dotenv()
HF_DATASET_REPO = os.getenv("HF_TOKENIZED_DATASET_REPO", None)

def read_lines(pattern):
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    yield text

def batch_iter(it, batch_size):
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def tokenize_and_upload(split, raw_glob, hf_dataset_repo, max_length=128, batch_size=8192):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = []
    attention_mask = []
    for batch in tqdm(batch_iter(read_lines(raw_glob), batch_size), desc=f"Tokenizing {split}"):
        encoded = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np"
        )
        input_ids.extend(encoded["input_ids"].tolist())
        attention_mask.extend(encoded["attention_mask"].tolist())

    # Features help ensure int32 for input_ids/attention_mask for best HF compatibility
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
    })
    ds = Dataset.from_dict(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        features=features,
    )

    # Push as split to the same repo
    ds.push_to_hub(hf_dataset_repo, split=split)
    print(f"✅ Pushed {split} tokenized dataset to https://huggingface.co/datasets/{hf_dataset_repo}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--raw_pattern", type=str, default=None)
    args = parser.parse_args()

    hf_repo = HF_DATASET_REPO
    raw_pattern = args.raw_pattern or os.path.join("data", args.split, f"*.{args.split}")

    tokenize_and_upload(
        split=args.split,
        raw_glob=raw_pattern,
        hf_dataset_repo=hf_repo,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

# python scripts/preprocess.py --split train
# python scripts/preprocess.py --split dev
# python scripts/preprocess.py --split test
