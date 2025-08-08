from flax import linen as nn
import jax.numpy as jnp
from models.hyperbolic_layers import mobius_add, poincare_distance

class PoincareHierarchicalBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    c: float = 1.0

    @nn.compact
    def __call__(self, x, mask):
        # Reshape mask to (batch, 1, 1, seq_len) for attention compatibility
        expanded_mask = jnp.expand_dims(mask, axis=1)  # (batch, 1, seq)
        expanded_mask = jnp.expand_dims(expanded_mask, axis=2)  # (batch, 1, 1, seq)

        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            deterministic=True  # optionally set to False for dropout in training
        )(x, mask=expanded_mask)

        hyp_output = mobius_add(x, attn, c=self.c)
        return hyp_output

class StructformerPoincare(nn.Module):
    vocab_size: int
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_length: int = 128
    c: float = 1.0

    def setup(self):
        self.token_embed = nn.Embed(self.vocab_size, self.hidden_dim)
        self.pos_embed = self.param('pos_embed', nn.initializers.normal(stddev=0.02), (1, self.max_length, self.hidden_dim))
        self.layers = [PoincareHierarchicalBlock(self.hidden_dim, self.num_heads, self.c) for _ in range(self.num_layers)]
        self.ln = nn.LayerNorm()
        self.head = nn.Dense(self.vocab_size)

    def __call__(self, input_ids, attention_mask):
        x = self.token_embed(input_ids) + self.pos_embed[:, :input_ids.shape[1], :]
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln(x)
        logits = self.head(x)
        return logits
