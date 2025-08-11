from typing import Any, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn

from models.hyperbolic_geometry import (
    poincare_proj, log_map_at_basepoint, update_basepoint_ema,
    poincare_distance_capped, hyperbolic_diagnostics,
)

class StructformerPoincare(nn.Module):
    """
    StructFormer with hyperbolic embeddings and geometry-aware attention.
    
    - Hyperbolic embedding table (Poincare ball)
    - Log map to Euclidean for StructFormer
    - Structure-aware attention (soft parsing)
    - Geometry-aware attention bias on selected heads (head splitting)
    - Learned basepoint with EMA updates
    """
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    max_length: int
    c: float = 1.0
    dropout_rate: float = 0.1
    
    def setup(self):
        # Hyperbolic embeddings (small init to stay well inside the ball)
        self.embed_table = self.param(
            'embed_table',
            lambda key, shape: jax.random.normal(key, shape) * 0.01,
            (self.vocab_size, self.hidden_dim)
        )
        # Learned basepoint (kept in batch_stats so we can update during apply)
        self.basepoint = self.variable(
            'batch_stats', 'basepoint',
            lambda: jnp.zeros((self.hidden_dim,), dtype=jnp.float32)
        )
        # Positional embeddings (Euclidean)
        self.position_embeddings = self.param(
            'position_embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.max_length, self.hidden_dim)
        )
        # Layers
        self.layers = [
            StructFormerLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                c=self.c,
                dropout_rate=self.dropout_rate,
                name=f'layer_{i}'
            )
            for i in range(self.num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm()
        self.output_projection = nn.Dense(
            self.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02)
        )
        
    def __call__(self, input_ids, attention_mask=None, training: bool = True):
        B, T = input_ids.shape

        # 1) Hyperbolic lookup + safety projection
        hyperbolic_embeds = self.embed_table[input_ids]                # [B, T, D]
        hyperbolic_embeds = poincare_proj(hyperbolic_embeds, self.c, eps_margin=1e-4)

        # 2) Log-map to Euclidean at learned basepoint
        o = self.basepoint.value                                       # [D]
        euclidean_embeds = jax.vmap(jax.vmap(
            lambda x: log_map_at_basepoint(x, o, self.c), in_axes=0
        ), in_axes=0)(hyperbolic_embeds)                               # [B, T, D]

        # 3) Add positional embeddings
        pos_ids = jnp.arange(T)[None, :]
        pos_emb = self.position_embeddings[pos_ids]                    # [1, T, D]
        hidden_states = euclidean_embeds + pos_emb                     # [B, T, D]

        # 4) Attention mask → large negative bias
        if attention_mask is None:
            attention_mask = jnp.ones((B, T), dtype=jnp.float32)
        attention_bias = (1.0 - attention_mask[:, None, None, :]) * -1e9  # [B,1,1,T]

        # 5) Basepoint EMA update (in hyperbolic space)
        if training:
            flat = hyperbolic_embeds.reshape(-1, self.hidden_dim)
            new_o = update_basepoint_ema(o, flat, self.c, ema=0.99)
            self.basepoint.value = new_o

        # 6) Layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                hyperbolic_embeds,   # raw hyperbolic (used only for bias; stop-grad inside)
                attention_bias,
                training=training
            )

        # 7) Output
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)                 # [B, T, V]

        # 8) Diagnostics
        diagnostics = hyperbolic_diagnostics(hyperbolic_embeds, self.c, dmax=4.0)
        return {
            'logits': logits,
            'hyperbolic_embeds': hyperbolic_embeds,
            'euclidean_embeds': euclidean_embeds,
            'basepoint': self.basepoint.value,
            'diagnostics': diagnostics
        }


class StructFormerLayer(nn.Module):
    """One StructFormer block: soft parsing + geometry/structure-aware attention + FFN."""
    hidden_dim: int
    num_heads: int
    c: float = 1.0
    dropout_rate: float = 0.1
    
    def setup(self):
        # Structure induction (biaffine-ish)
        self.struct_head_proj = nn.Dense(self.hidden_dim, name='struct_head',
                                         kernel_init=nn.initializers.xavier_uniform())
        self.struct_dep_proj  = nn.Dense(self.hidden_dim, name='struct_dep',
                                         kernel_init=nn.initializers.xavier_uniform())
        # Attention
        self.attn = GeometryAwareAttention(
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            c=self.c,
            dropout_rate=self.dropout_rate
        )
        # Norms & FFN
        self.attn_ln = nn.LayerNorm()
        self.ffn_ln  = nn.LayerNorm()
        self.ffn1    = nn.Dense(self.hidden_dim * 4, kernel_init=nn.initializers.xavier_uniform())
        self.ffn2    = nn.Dense(self.hidden_dim,      kernel_init=nn.initializers.xavier_uniform())
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, eu_hidden, hyp_embeds, attn_bias, training: bool = True):
        # 1) Soft parsing scores
        struct_scores = self._structure_scores(eu_hidden)  # [B, T, T]

        # 2) Geo/struct-aware attention
        attn_out = self.attn(
            eu_hidden,
            hyp_embeds,
            struct_scores,
            attn_bias,
            training=training
        )
        attn_out = self.dropout(attn_out, deterministic=not training)
        h = self.attn_ln(eu_hidden + attn_out)

        # 3) FFN
        f = self.ffn2(jax.nn.gelu(self.ffn1(h)))
        f = self.dropout(f, deterministic=not training)
        return self.ffn_ln(h + f)

    def _structure_scores(self, h):
        # Bilinear scores + softmax over possible parents
        head = self.struct_head_proj(h)   # [B, T, D]
        dep  = self.struct_dep_proj(h)    # [B, T, D]
        logits = jnp.einsum('bih,bjh->bij', head, dep) / jnp.sqrt(self.hidden_dim)
        return jax.nn.softmax(logits, axis=-1)  # [B, T, T]


class GeometryAwareAttention(nn.Module):
    """
    Multi-head attention with:
    - Standard Q/K/V projections
    - Structure bias from soft parsing
    - Geometry bias (−beta * min(d_c, Dmax)) for a subset of heads (head-splitting)
    """
    num_heads: int
    hidden_dim: int
    c: float = 1.0
    dropout_rate: float = 0.1

    def setup(self):
        assert self.hidden_dim % self.num_heads == 0
        self.head_dim = self.hidden_dim // self.num_heads

        # Q/K/V and output
        self.q = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=nn.initializers.xavier_uniform())
        self.k = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=nn.initializers.xavier_uniform())
        self.v = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=nn.initializers.xavier_uniform())
        self.o = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())

        # Per-head geometry strength (learned >=0 via softplus)
        self._beta_raw = self.param('geometry_beta_raw', nn.initializers.zeros, (self.num_heads,))

        # Which heads get geometry (50/50 split by default)
        geom_on = self.num_heads // 2
        mask = jnp.concatenate([
            jnp.ones(geom_on, dtype=jnp.float32),
            jnp.zeros(self.num_heads - geom_on, dtype=jnp.float32)
        ])
        self.geometry_mask = mask  # constant buffer

        # Warmup counter (lives in batch_stats)
        self.warmup_counter = self.variable('batch_stats', 'warmup_counter', lambda: jnp.array(0, jnp.int32))

        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, eu_hidden, hyp_embeds, struct_scores, attn_bias, training: bool = True):
        B, T, D = eu_hidden.shape

        # 1) Q/K/V
        q = self.q(eu_hidden).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k(eu_hidden).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v(eu_hidden).reshape(B, T, self.num_heads, self.head_dim)

        # 2) Dot-product attention logits
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)  # [B,H,T,T]

        # 3) Structure bias (safe log)
        alpha_struct = 0.1
        struct_bias = alpha_struct * jnp.log(struct_scores + 1e-8)              # [B,T,T]
        scores = scores + struct_bias[:, None, :, :]                             # [B,H,T,T]

        # 4) Geometry bias (use stop-grad so CE doesn’t flow into embeddings via bias)
        geo_bias = self._geometry_bias(jax.lax.stop_gradient(hyp_embeds), training)  # [B,H,T,T]
        scores = scores + geo_bias

        # 5) Mask → softmax → dropout
        scores = scores + attn_bias                                              # broadcast [B,1,1,T]
        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.dropout(attn, deterministic=not training)

        # 6) Aggregate and project
        ctx = jnp.einsum('bhqk,bkhd->bqhd', attn, v).reshape(B, T, D)
        return self.o(ctx)

    def _geometry_bias(self, hyp_embeds, training: bool):
        """
        Compute −β * min(d_c(x_i, x_j), Dmax) per head.
        """
        B, T, _ = hyp_embeds.shape

        # Pairwise distances per batch item
        def dist_mat(E):  # E: [T, D]
            # For each row xi, compute distances to all xj
            def row(xi):
                return jax.vmap(lambda xj: poincare_distance_capped(xi, xj, self.c, dmax=4.0))(E)  # [T]
            return jax.vmap(row)(E)  # [T, T]
        d = jax.vmap(dist_mat)(hyp_embeds)  # [B, T, T]

        # Warmup: update counter immutably
        if training:
            self.warmup_counter.value = self.warmup_counter.value + 1
        step = self.warmup_counter.value
        warmup_steps = 3000.0
        warm = jnp.minimum(1.0, (step.astype(jnp.float32) / warmup_steps))

        # Per-head β >= 0 via softplus, then warmup, then mask
        beta = jax.nn.softplus(self._beta_raw) * warm                          # [H]
        beta = beta * self.geometry_mask                                       # [H]

        # Broadcast to [B,H,T,T] and apply sign
        return -(beta[None, :, None, None]) * d[:, None, :, :]


# Factories

def create_structformer_poincare(config):
    assert config.hidden_dim % config.num_heads == 0
    assert config.c > 0, "Curvature c must be > 0"
    return StructformerPoincare(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_length=config.max_length,
        c=config.c,
        dropout_rate=getattr(config, 'dropout_rate', 0.1),
    )

def initialize_model_params(model, key, config):
    dummy_input = jnp.ones((1, config.max_length), dtype=jnp.int32)
    dummy_mask  = jnp.ones((1, config.max_length), dtype=jnp.float32)
    variables = model.init(key, dummy_input, attention_mask=dummy_mask, training=False)
    # Safety project embeddings at init
    emb = variables['params']['embed_table']
    variables['params']['embed_table'] = poincare_proj(emb, config.c, eps_margin=1e-4)
    return variables
