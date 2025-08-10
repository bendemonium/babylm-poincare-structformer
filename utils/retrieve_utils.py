# retrieve_utils.py

import os
import pickle
from datasets import load_dataset


def load_pickle_from_hf(repo_id: str, filename: str, cache_dir: str = None):
    """
    Load a pickle (.pkl) file stored in a Hugging Face Hub repo.

    Args:
        repo_id (str): The HF repo ID where the pickle file is stored.
        filename (str): The name of the pickle file inside the repo.
        cache_dir (str, optional): Path to cache the downloaded file locally.

    Returns:
        object: The deserialized Python object from the pickle file.

    Usage:
        data = load_pickle_from_hf("username/my-pickle-repo", "data.pkl")
    """
    from huggingface_hub import hf_hub_download

    # Download the pickle file from HF repo
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)

    # Load pickle from local file
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_hf_dataset(repo_id: str, split: str = None, cache_dir: str = None, **kwargs):
    """
    Load a dataset directly from the Hugging Face Hub using datasets.load_dataset.

    Args:
        repo_id (str): The HF dataset repo ID or shortcut name (e.g. "glue", "squad").
        split (str, optional): Which dataset split to load ("train", "test", etc.).
        cache_dir (str, optional): Directory to cache the dataset.
        **kwargs: Additional keyword arguments to pass to datasets.load_dataset.

    Returns:
        datasets.Dataset or datasets.DatasetDict: Loaded HF dataset object.

    Usage:
        ds = load_hf_dataset("glue", split="train")
    """
    dataset = load_dataset(repo_id, split=split, cache_dir=cache_dir, **kwargs)
    return dataset


def load_dataset_dynamic(
    repo_id: str,
    method: str = "hf_dataset",
    pickle_filename: str = None,
    split: str = None,
    cache_dir: str = None,
    **kwargs,
):
    """
    Dynamically load data from HF repo either as a pickle file or a regular dataset.

    Args:
        repo_id (str): Repository or dataset ID on Hugging Face Hub.
        method (str): Which loading method: "hf_dataset" or "pickle".
        pickle_filename (str): If method=="pickle", name of the pickle file to load.
        split (str): Dataset split if loading HF dataset.
        cache_dir (str): Optional caching directory.
        kwargs: Additional kwargs passed to load_dataset if applicable.

    Returns:
        Loaded dataset or Python object (pickle).

    Raises:
        ValueError: If loading method is unknown.
        FileNotFoundError: If pickle_filename is not provided when needed.
    """
    if method == "hf_dataset":
        return load_hf_dataset(repo_id, split=split, cache_dir=cache_dir, **kwargs)
    elif method == "pickle":
        if pickle_filename is None:
            raise FileNotFoundError("Must provide pickle_filename when method='pickle'")
        return load_pickle_from_hf(repo_id, pickle_filename, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown loading method: {method}, must be 'hf_dataset' or 'pickle'")
