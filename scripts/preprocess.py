import argparse
import os
import glob
import pickle
from transformers import AutoTokenizer

def read_raw_lines(file_pattern):
    """Read all lines from matching files."""
    raw_texts = []
    for file_path in glob.glob(file_pattern):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            raw_texts.extend(lines)
    return raw_texts

def tokenize_texts(texts, tokenizer, max_length=128):
    """Batch tokenize all texts into padded input_ids and attention_mask."""
    return tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np"
    )

def save_tokenized_data(encoded, out_path):
    """Save tokenized data to a pickle file."""
    with open(out_path, "wb") as f:
        pickle.dump({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }, f)
    print(f"✅ Saved tokenized data to {out_path}")

def main(args):
    # Load GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

    # Load raw text
    raw_path = os.path.join("data", args.split, f"*.{args.split}")
    texts = read_raw_lines(raw_path)
    print(f"📄 Loaded {len(texts):,} {args.split} samples")

    # Tokenize
    encoded = tokenize_texts(texts, tokenizer, max_length=args.max_length)

    # Ensure data/tokens directory exists
    os.makedirs("data/tokens", exist_ok=True)

    # Save tokenized data
    out_file = os.path.join("data/tokens", f"{args.split}_tokenized.pkl")
    save_tokenized_data(encoded, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, help="Data split: train, dev, test")
    parser.add_argument("--max_length", type=int, default=128, help="Max token length per sequence")
    args = parser.parse_args()
    main(args)
