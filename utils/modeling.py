import jax.numpy as jnp
from flax import linen as nn


#------------ helper functions --------------#

def mobius_add(x, y, c=1.0):
    # Möbius addition on the Poincaré ball
    norm_x = jnp.linalg.norm(x, axis=-1, keepdims=True)
    norm_y = jnp.linalg.norm(y, axis=-1, keepdims=True)
    dot = jnp.sum(x * y, axis=-1, keepdims=True)
    numerator = (1 + 2 * c * dot + c * norm_y ** 2) * x + (1 - c * norm_x ** 2) * y
    denominator = 1 + 2 * c * dot + c ** 2 * norm_x ** 2 * norm_y ** 2
    return numerator / jnp.clip(denominator, 1e-5, None)

def poincare_distance(x, y, c=1.0):
    # Compute Poincaré distance between x, y
    sqrt_c = jnp.sqrt(c)
    diff = x - y
    norm_diff = jnp.linalg.norm(diff, axis=-1)
    norm_x = jnp.linalg.norm(x, axis=-1)
    norm_y = jnp.linalg.norm(y, axis=-1)
    num = 2 * sqrt_c * norm_diff
    denom = (1 - c * norm_x ** 2) * (1 - c * norm_y ** 2)
    return jnp.arccosh(1 + num ** 2 / denom)

def project_to_ball(x, c=1.0, eps=0.00001):
    # Project x onto the Poincaré ball if they exceed the radius
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    max_norm = (1.0 - eps) / jnp.sqrt(c)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-15))
    return x * scale

class PoincareHierarchicalBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    c: float = 1.0

    @nn.compact
    def __call__(self, x, mask):
        # Reshape mask to (batch, 1, 1, seq_len) for attention compatibility
        expanded_mask = jnp.expand_dims(mask, axis=1)    # (batch, 1, seq)
        expanded_mask = jnp.expand_dims(expanded_mask, axis=2)  # (batch, 1, 1, seq)

        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            deterministic=True  # Set to False to enable dropout in training if desired
        )(x, mask=expanded_mask)

        # Hyperbolic residual connection
        hyp_output = mobius_add(x, attn, c=self.c)
        return hyp_output

#------------ acual model stuff --------------#

class StructformerPoincare(nn.Module):
    vocab_size: int
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_length: int = 128
    c: float = 1.0

    def setup(self):
        # Token + position embeddings
        self.token_embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_dim
        )
        self.pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, self.max_length, self.hidden_dim)
        )

        # Transformer-like hyperbolic layers
        self.layers = [
            PoincareHierarchicalBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                c=self.c
            )
            for _ in range(self.num_layers)
        ]
        self.ln = nn.LayerNorm()
        self.head = nn.Dense(self.vocab_size)

    def __call__(self, input_ids, attention_mask):
        """
        Standard forward: embeddings → hyperbolic layers → logits.
        This path is used for cross-entropy loss training of non-embedding weights.
        """
        x = self.token_embed(input_ids) + self.pos_embed[:, :input_ids.shape[1], :]
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln(x)
        logits = self.head(x)
        return logits

    def embed_only(self, input_ids, attention_mask=None):
        """
        Return token embeddings + positional embeddings BEFORE any transformer layers.
        Used in training for computing Poincare distance loss on embeddings only.
        We pass through exactly the parameters in 'token_embed' and 'pos_embed'.
        """
        return self.token_embed(input_ids) + self.pos_embed[:, :input_ids.shape[1], :]
