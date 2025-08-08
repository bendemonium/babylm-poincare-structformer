import jax.numpy as jnp

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
