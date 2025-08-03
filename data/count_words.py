import os
import argparse

def count_words_in_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(len(line.strip().split()) for line in f)
    except Exception as e:
        print(f"❌ Could not read {path}: {e}")
        return 0

def count_words_in_folder(folder_path):
    total_words = 0
    file_counts = []

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(".train"):
                full_path = os.path.join(root, fname)
                word_count = count_words_in_file(full_path)
                total_words += word_count
                file_counts.append((fname, word_count))

    # Print individual files
    for fname, count in sorted(file_counts):
        print(f"📄 {fname}: {count:,} words")

    print(f"\n🧾 Total words across all .train files: {total_words:,}")
    return total_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Folder containing BabyLM .train files")
    args = parser.parse_args()

    count_words_in_folder(args.folder)
