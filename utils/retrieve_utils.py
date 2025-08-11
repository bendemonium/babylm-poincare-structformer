import os
import pickle
from typing import Dict, List, Optional, Union, Any
import logging

import jax.numpy as jnp
import numpy as np
from huggingface_hub import hf_hub_download, login
from datasets import load_dataset, Dataset
import transformers

logger = logging.getLogger(__name__)

class DatasetConfig:
    """Configuration for dataset loading."""
    def __init__(self, config_dict):
        self.data = config_dict.get('data', {})
        self.dataset_type = self.data.get('dataset_type', 'hf')
        self.train_tokenized_repo = self.data.get('train_tokenized_repo')
        self.train_tokenized_file = self.data.get('train_tokenized_file')
        self.val_tokenized_repo = self.data.get('val_tokenized_repo') 
        self.val_tokenized_file = self.data.get('val_tokenized_file')
        self.train_split = self.data.get('train_split', 'train')
        self.val_split = self.data.get('val_split', 'validation')
        self.vocab_name = self.data.get('vocab_name', 'gpt2')


def load_pickle_from_hub(repo_id: str, filename: str, repo_type: str = "dataset") -> Any:
    """
    Load pickle file from HuggingFace Hub with proper authentication.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "user/repo-name")
        filename: Name of pickle file to load
        repo_type: Type of repository ("dataset" or "model")
    
    Returns:
        Unpickled data object
    """
    try:
        logger.info(f"Downloading {filename} from {repo_id} (type: {repo_type})")
        
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            cache_dir=None  # Use default cache
        )
        
        logger.info(f"Loading pickle file from {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        logger.info(f"Successfully loaded pickle  {type(data)}")
        if hasattr(data, '__len__'):
            logger.info(f"Data length: {len(data)}")
            
        return data
        
    except Exception as e:
        logger.error(f"Failed to load pickle from {repo_id}/{filename}: {str(e)}")
        raise


def load_hf_dataset(
    repo_id: str, 
    split: str = "train",
    streaming: bool = False,
    trust_remote_code: bool = True
) -> Dataset:
    """
    Load dataset from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        split: Dataset split to load
        streaming: Whether to use streaming mode
        trust_remote_code: Whether to trust remote code execution
        
    Returns:
        HuggingFace Dataset object
    """
    try:
        logger.info(f"Loading HF dataset: {repo_id}, split: {split}")
        
        dataset = load_dataset(
            repo_id,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
            cache_dir=None  # Use default cache
        )
        
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        logger.info(f"Dataset features: {dataset.features}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load HF dataset {repo_id}/{split}: {str(e)}")
        raise


def load_dataset_dynamic(
    repo_id: str,
    method: str = "hf",
    split: str = "train", 
    pickle_filename: Optional[str] = None,
    **kwargs
) -> Union[Dataset, Any]:
    """
    Dynamic dataset loader that handles multiple data sources.
    
    Args:
        repo_id: Repository identifier
        method: Loading method ("hf", "pickle", "local")
        split: Dataset split for HF datasets
        pickle_filename: Filename for pickle method
        **kwargs: Additional arguments passed to loaders
        
    Returns:
        Loaded dataset or data object
    """
    if method == "pickle":
        if pickle_filename is None:
            raise ValueError("pickle_filename required for pickle method")
        return load_pickle_from_hub(
            repo_id, 
            pickle_filename, 
            repo_type=kwargs.get('repo_type', 'dataset')
        )
        
    elif method == "hf":
        return load_hf_dataset(
            repo_id, 
            split=split,
            streaming=kwargs.get('streaming', False)
        )
        
    elif method == "local":
        return load_local_dataset(repo_id, **kwargs)
        
    else:
        raise ValueError(f"Unknown loading method: {method}")


def load_local_dataset(path: str, format: str = "auto") -> Dataset:
    """
    Load dataset from local files.
    
    Args:
        path: Path to dataset files
        format: File format ("json", "csv", "parquet", "auto")
        
    Returns:
        HuggingFace Dataset object
    """
    try:
        if format == "auto":
            if path.endswith('.json'):
                format = "json"
            elif path.endswith('.csv'):
                format = "csv" 
            elif path.endswith('.parquet'):
                format = "parquet"
            else:
                raise ValueError(f"Cannot auto-detect format for {path}")
                
        logger.info(f"Loading local dataset from {path} (format: {format})")
        
        dataset = load_dataset(format, data_files=path)['train']
        logger.info(f"Loaded {len(dataset)} examples")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load local dataset {path}: {str(e)}")
        raise


def get_tokenizer(vocab_name: str = "gpt2"):
    """
    Load tokenizer for vocabulary size extraction and text processing.
    
    Args:
        vocab_name: Name of tokenizer to load
        
    Returns:
        HuggingFace tokenizer
    """
    try:
        logger.info(f"Loading tokenizer: {vocab_name}")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(vocab_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        logger.info(f"Special tokens: pad={tokenizer.pad_token_id}, "
                   f"eos={tokenizer.eos_token_id}, bos={getattr(tokenizer, 'bos_token_id', None)}")
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load tokenizer {vocab_name}: {str(e)}")
        raise


def prepare_batch_data(
    examples: List[Dict],
    max_length: int = 128,
    pad_token_id: int = 50256
) -> Dict[str, jnp.ndarray]:
    """
    Prepare batch data for training with proper padding and attention masks.
    
    Args:
        examples: List of examples with 'input_ids' key
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        
    Returns:
        Dictionary with 'input_ids' and 'attention_mask' arrays
    """
    batch_size = len(examples)
    
    # Initialize arrays
    input_ids = np.full((batch_size, max_length), pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((batch_size, max_length), dtype=np.float32)
    
    for i, example in enumerate(examples):
        tokens = example['input_ids']
        seq_len = min(len(tokens), max_length)
        
        # Fill in actual tokens and mask
        input_ids[i, :seq_len] = tokens[:seq_len]
        attention_mask[i, :seq_len] = 1.0
    
    return {
        'input_ids': jnp.array(input_ids),
        'attention_mask': jnp.array(attention_mask)
    }


def create_data_iterator(
    dataset: Union[Dataset, Any],
    batch_size: int = 32,
    max_length: int = 128,
    pad_token_id: int = 50256,
    shuffle: bool = True,
    drop_last: bool = True
):
    """
    Create iterator over dataset batches.
    
    Args:
        dataset: Dataset to iterate over
        batch_size: Batch size
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        shuffle: Whether to shuffle data
        drop_last: Whether to drop incomplete last batch
        
    Yields:
        Prepared batch dictionaries
    """
    # Handle different dataset types
    if isinstance(dataset, Dataset):
        # HuggingFace dataset
        if shuffle:
            dataset = dataset.shuffle()
        data = dataset
    elif isinstance(dataset, (list, tuple)):
        # List/tuple of examples
        if shuffle:
            indices = np.random.permutation(len(dataset))
            data = [dataset[i] for i in indices]
        else:
            data = dataset
    else:
        # Assume it's an iterable
        data = dataset
    
    batch = []
    for example in data:
        # Ensure example has required format
        if isinstance(example, dict) and 'input_ids' in example:
            batch.append(example)
        else:
            # Try to convert if it's raw token list
            batch.append({'input_ids': example})

        if len(batch) == batch_size:
            yield prepare_batch_data(batch, max_length, pad_token_id)
            batch = []
    
    # Handle remaining examples
    if batch and not drop_last:
        yield prepare_batch_data(batch, max_length, pad_token_id)


def load_training_data(config) -> tuple:
    """
    Load training and validation datasets based on configuration.
    
    Args:
        config: Configuration object with data settings
        
    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer)
    """
    try:
        logger.info("Loading training data...")
        
        # Parse config
        if hasattr(config, 'data'):
            data_config = DatasetConfig({'data': config.data})
        else:
            # Fallback to old format
            data_config = DatasetConfig({
                'data': {
                    'dataset_type': getattr(config, 'dataset_type', 'hf'),
                    'train_tokenized_repo': getattr(config, 'train_tokenized_repo', None),
                    'train_tokenized_file': getattr(config, 'train_tokenized_file', None),
                    'val_tokenized_repo': getattr(config, 'val_tokenized_repo', None),
                    'val_tokenized_file': getattr(config, 'val_tokenized_file', None),
                    'vocab_name': getattr(config, 'vocab_name', 'gpt2')
                }
            })
        
        # Load tokenizer
        tokenizer = get_tokenizer(data_config.vocab_name)
        
        # Load training data
        logger.info(f"Loading training data from {data_config.train_tokenized_repo}")
        train_data = load_dataset_dynamic(
            data_config.train_tokenized_repo,
            method=data_config.dataset_type,
            split=data_config.train_split,
            pickle_filename=data_config.train_tokenized_file
        )
        
        # Load validation data  
        logger.info(f"Loading validation data from {data_config.val_tokenized_repo}")
        val_data = load_dataset_dynamic(
            data_config.val_tokenized_repo,
            method=data_config.dataset_type,
            split=data_config.val_split,
            pickle_filename=data_config.val_tokenized_file
        )
        
        logger.info("✅ Successfully loaded all datasets")
        logger.info(f"Training  {len(train_data) if hasattr(train_data, '__len__') else 'streaming'}")
        logger.info(f"Validation  {len(val_data) if hasattr(val_data, '__len__') else 'streaming'}")
        
        return train_data, val_data, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load training  {str(e)}")
        raise


def estimate_dataset_tokens(dataset, tokenizer, sample_size: int = 1000) -> int:
    """
    Estimate total tokens in dataset by sampling.
    
    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer for vocab info
        sample_size: Number of examples to sample
        
    Returns:
        Estimated total tokens
    """
    if not hasattr(dataset, '__len__'):
        logger.warning("Cannot estimate tokens for streaming dataset")
        return -1
        
    try:
        total_examples = len(dataset)
        sample_size = min(sample_size, total_examples)
        
        # Sample random examples
        indices = np.random.choice(total_examples, sample_size, replace=False)
        total_tokens = 0
        
        for idx in indices:
            example = dataset[int(idx)]
            if 'input_ids' in example:
                total_tokens += len(example['input_ids'])
            elif isinstance(example, (list, tuple)):
                total_tokens += len(example)
                
        # Extrapolate to full dataset
        avg_tokens = total_tokens / sample_size
        estimated_total = int(avg_tokens * total_examples)
        
        logger.info(f"Dataset token estimation:")
        logger.info(f"  Sampled {sample_size} examples")
        logger.info(f"  Average tokens per example: {avg_tokens:.1f}")
        logger.info(f"  Estimated total tokens: {estimated_total:,}")
        
        return estimated_total
        
    except Exception as e:
        logger.error(f"Failed to estimate tokens: {str(e)}")
        return -1


# Authentication helpers
def setup_hf_auth(token: Optional[str] = None):
    """
    Set up HuggingFace authentication.
    
    Args:
        token: HF token (if None, tries environment variables)
    """
    try:
        if token:
            login(token)
            logger.info("✅ Authenticated with provided HF token")
        else:
            # Try environment variables
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
            if hf_token:
                login(hf_token)
                logger.info("✅ Authenticated with environment HF token")
            else:
                logger.warning("⚠️  No HF token found - may not be able to access private repos")
                
    except Exception as e:
        logger.error(f"HF authentication failed: {str(e)}")
        raise


# Validation helpers
def validate_dataset_format(dataset, required_keys=['input_ids']) -> bool:
    """
    Validate that dataset has expected format.
    
    Args:
        dataset: Dataset to validate
        required_keys: Keys that must be present in examples
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check first few examples
        for i in range(min(3, len(dataset) if hasattr(dataset, '__len__') else 3)):
            example = dataset[i] if hasattr(dataset, '__getitem__') else next(iter(dataset))
            
            if isinstance(example, dict):
                for key in required_keys:
                    if key not in example:
                        logger.error(f"Missing required key '{key}' in example {i}")
                        return False
            else:
                logger.warning(f"Example {i} is not a dictionary: {type(example)}")
                
        logger.info("✅ Dataset format validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        return False


def get_vocab_size_from_config(config) -> int:
    """
    Extract vocabulary size from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Vocabulary size
    """
    if hasattr(config, 'vocab_size'):
        return config.vocab_size
    elif hasattr(config, 'model') and hasattr(config.model, 'vocab_size'):
        return config.model.vocab_size
    else:
        # Try to infer from tokenizer
        vocab_name = getattr(config, 'vocab_name', 'gpt2')
        if hasattr(config, 'data'):
            vocab_name = config.data.get('vocab_name', vocab_name)
            
        tokenizer = get_tokenizer(vocab_name)
        return tokenizer.vocab_size
