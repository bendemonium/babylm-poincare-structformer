# models/modeling_structformer_poincare_flax.py
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers.modeling_flax_utils import (
    FlaxPreTrainedModel,
)
from transformers import FlaxMaskedLMOutput
from .configuration_structformer_poincare import StructformerPoincareConfig
from .structformer_poincare import StructformerPoincare as _InnerModel  # your existing module

def _init_params(rng, config: StructformerPoincareConfig):
    # Build the same inner model and init its params
    model = _InnerModel(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_length=config.max_length,
        c=config.c,
        dropout_rate=config.dropout_rate,
        attention_input=config.attention_input,
    )
    sample = {
        "input_ids": jnp.ones((1, config.max_length), dtype=jnp.int32),
        "attention_mask": jnp.ones((1, config.max_length), dtype=jnp.float32),
    }
    variables = model.init(rng, **sample, training=False)
    return model, variables["params"]

class FlaxStructformerPoincareForMaskedLMModule:
    """
    Lightweight wrapper using your inner Structformer that returns logits only
    (to match FlaxMaskedLMOutput expectations).
    """
    def __init__(self, config: StructformerPoincareConfig):
        self.config = config
        self.inner = _InnerModel(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_length=config.max_length,
            c=config.c,
            dropout_rate=config.dropout_rate,
            attention_input=config.attention_input,
        )

    def __call__(self, params: FrozenDict, input_ids, attention_mask=None, deterministic: bool = True):
        outs = self.inner.apply(
            {"params": params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=not deterministic,
        )
        # outs must contain 'logits' in your inner model forward
        return outs["logits"]

class FlaxStructformerPoincareForMaskedLM(FlaxPreTrainedModel):
    config_class = StructformerPoincareConfig
    module_class = FlaxStructformerPoincareForMaskedLMModule

    def __init__(self, config: StructformerPoincareConfig, dtype: jnp.dtype = jnp.float32, **kwargs):
        super().__init__(config, dtype=dtype, **kwargs)
        self.module = self.module_class(config)

        # Initialize params if not provided
        if self.params is None:
            rng = jax.random.PRNGKey(0)
            inner, params = _init_params(rng, config)
            # Ensure param dict format matches HF expectations (FrozenDict)
            self.params = FrozenDict(unflatten_dict(flatten_dict(params)))

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        params: Optional[FrozenDict] = None,
        dropout_rng: Optional[jax.random.PRNGKey] = None,
    ) -> FlaxMaskedLMOutput:
        if params is None:
            params = self.params
        logits = self.module(params, input_ids=input_ids, attention_mask=attention_mask, deterministic=deterministic)
        return FlaxMaskedLMOutput(logits=logits)