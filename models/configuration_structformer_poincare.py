from typing import Optional
from transformers import PretrainedConfig

class StructformerPoincareConfig(PretrainedConfig):
    model_type = "structformer_poincare"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        max_length: int = 128,
        dropout_rate: float = 0.1,
        c: float = 1.0,
        attention_input: str = "tangent",
        pad_token_id: Optional[int] = 50256,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.c = c
        self.attention_input = attention_input