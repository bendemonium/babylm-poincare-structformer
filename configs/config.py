from transformers import PretrainedConfig

class StructformerConfig(PretrainedConfig):
    model_type = "structformer"

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_length: int = 128,
        vocab_size: int = 50257,
        c: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.c = c