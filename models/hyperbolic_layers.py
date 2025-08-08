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


# # import jax.numpy as jnp
# import torch

# def mobius_add(x, y, c=1.0):
#     # Möbius addition on the Poincaré ball

#     # #### FLAX LOCAL
#     # norm_x = jnp.linalg.norm(x, axis=-1, keepdims=True)
#     # norm_y = jnp.linalg.norm(y, axis=-1, keepdims=True)
#     # dot = jnp.sum(x * y, axis=-1, keepdims=True)
#     # numerator = (1 + 2 * c * dot + c * norm_y ** 2) * x + (1 - c * norm_x ** 2) * y
#     # denominator = 1 + 2 * c * dot + c ** 2 * norm_x ** 2 * norm_y ** 2
#     # return numerator / jnp.clip(denominator, 1e-5, None)

#     x2 = torch.sum(x * x, dim=-1, keepdim=True)
#     y2 = torch.sum(y * y, dim=-1, keepdim=True)
#     xy = torch.sum(x * y, dim=-1, keepdim=True)
#     numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
#     denominator = 1 + 2 * c * xy + c**2 * x2 * y2
#     return numerator / denominator.clamp_min(1e-5)

# def poincare_distance(x, y, c=1.0):
#     # Compute Poincaré distance between x, y

#     sqrt_c = torch.sqrt(torch.tensor(c))
#     diff = x - y
#     norm_diff = torch.linalg.norm(diff, dim=-1)
#     norm_x = torch.linalg.norm(x, dim=-1)
#     norm_y = torch.linalg.norm(y, dim=-1)
#     num = 2 * sqrt_c * norm_diff
#     denom = (1 - c * norm_x ** 2) * (1 - c * norm_y ** 2)
#     # Clamp the argument inside arccosh to >=1 to avoid NaNs
#     arg = 1 + (num ** 2) / denom.clamp(min=1e-5)
#     return torch.arccosh(arg)

#     # #### FLAX LOCAL
#     # sqrt_c = jnp.sqrt(c)
#     # diff = x - y
#     # norm_diff = jnp.linalg.norm(diff, axis=-1)
#     # norm_x = jnp.linalg.norm(x, axis=-1)
#     # norm_y = jnp.linalg.norm(y, axis=-1)
#     # num = 2 * sqrt_c * norm_diff
#     # denom = (1 - c * norm_x ** 2) * (1 - c * norm_y ** 2)
#     # return jnp.arccosh(1 + num ** 2 / denom)
