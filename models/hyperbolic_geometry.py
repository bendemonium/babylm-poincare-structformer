# Stable Poincare-ball utilities for StructFormer + hyperbolic embeddings (JAX 0.6.2)

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax

# ---------------------------------------------------------------------
# Global small constants (tune if needed)
# ---------------------------------------------------------------------
_EPS_DEN   = 1e-12   # for denominators / divisions
_EPS_UNIT  = 1e-12   # for norm floors, unit directions
_EPS_ACOSH = 1e-6    # for arcosh argument clamp: arg >= 1 + EPS_ACOSH
_EPS_BND   = 1e-6    # for boundary margin inside R^2 clamp (distinct from proj margin)

# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def _to_fp32(*xs):
    return tuple(jnp.asarray(x, jnp.float32) for x in xs)

# ---------------------------------------------------------------------
# Möbius addition & negation
# ---------------------------------------------------------------------
@jax.jit  # c is a scalar; passing as positional is fine
def mobius_neg(x, c: float = 1.0):
    x, = _to_fp32(x)
    return -x

@jax.jit
def mobius_addition(x, y, c: float = 1.0):
    """
    Möbius addition in the Poincaré ball: x ⊕ y

    x ⊕ y = [(1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y] /
            [1 + 2c<x,y> + c^2||x||^2||y||^2]
    """
    x, y = _to_fp32(x, y)
    x_norm_sq = jnp.sum(x * x, axis=-1, keepdims=True)
    y_norm_sq = jnp.sum(y * y, axis=-1, keepdims=True)
    xy_dot    = jnp.sum(x * y, axis=-1, keepdims=True)

    coeff_x  = 1.0 + 2.0 * c * xy_dot + c * y_norm_sq
    coeff_y  = 1.0 - c * x_norm_sq
    numer    = coeff_x * x + coeff_y * y

    denom = 1.0 + 2.0 * c * xy_dot + (c ** 2) * x_norm_sq * y_norm_sq
    denom = jnp.maximum(denom, _EPS_DEN)

    return numer / denom

# ---------------------------------------------------------------------
# Poincaré distance (safe & fp32)
# ---------------------------------------------------------------------
@jax.jit
def poincare_distance(x, y, c: float = 1.0):
    """
    Returns distance with shape (...,), fp32.
    Safe near boundary; clamps arcosh argument to >= 1 + _EPS_ACOSH.
    """
    x, y = _to_fp32(x, y)
    R = 1.0 / jnp.sqrt(c)

    x_norm_sq = jnp.sum(x * x, axis=-1, keepdims=True)
    y_norm_sq = jnp.sum(y * y, axis=-1, keepdims=True)

    # Keep squared norms strictly inside the ball with a tiny margin
    max_sq = (R ** 2) * (1.0 - _EPS_BND)
    x_norm_sq = jnp.clip(x_norm_sq, 0.0, max_sq)
    y_norm_sq = jnp.clip(y_norm_sq, 0.0, max_sq)

    diff      = x - y
    diff_norm_sq = jnp.sum(diff * diff, axis=-1, keepdims=True)

    one_minus_cx = 1.0 - c * x_norm_sq
    one_minus_cy = 1.0 - c * y_norm_sq
    one_minus_cx = jnp.maximum(one_minus_cx, _EPS_DEN)
    one_minus_cy = jnp.maximum(one_minus_cy, _EPS_DEN)

    num = 2.0 * c * diff_norm_sq
    den = one_minus_cx * one_minus_cy
    den = jnp.maximum(den, _EPS_DEN)

    arg = 1.0 + (num / den)
    arg = jnp.maximum(arg, 1.0 + _EPS_ACOSH)

    d = jnp.arccosh(arg) / jnp.sqrt(c)  # shape (..., 1)
    return jnp.squeeze(d, axis=-1)      # shape (...,)

@jax.jit
def poincare_distance_capped(x, y, c: float = 1.0, dmax: float = 4.0):
    """
    d_c capped at Dmax for stable attention biasing: min(d_c(x,y), dmax).
    """
    d = poincare_distance(x, y, c)
    return jnp.minimum(d, jnp.asarray(dmax, jnp.float32))

# ---------------------------------------------------------------------
# Log/Exp maps
# ---------------------------------------------------------------------
@jax.jit
def log_map_at_basepoint(x, o, c: float = 1.0):
    """
    log_o(x) using Möbius transport to origin then log@0.
    log_o(x) = (2/√c) * artanh(√c * ||(-o) ⊕ x||) * ((-o) ⊕ x) / ||(-o) ⊕ x||
    """
    x, o = _to_fp32(x, o)
    neg_o = mobius_neg(o, c)
    transported = mobius_addition(neg_o, x, c)  # (-o) ⊕ x

    norm = jnp.linalg.norm(transported, axis=-1, keepdims=True)
    norm = jnp.maximum(norm, _EPS_UNIT)
    direction = transported / norm

    sqrt_c = jnp.sqrt(c)
    arg = jnp.clip(sqrt_c * norm, 0.0, 1.0 - 1e-6)  # artanh domain
    magnitude = (2.0 / sqrt_c) * jnp.arctanh(arg)

    return magnitude * direction  # (..., D), fp32

@jax.jit
def log_map_at_origin(x, c: float = 1.0):
    """
    log_0(x) = (2/√c) * artanh(√c * ||x||) * x/||x||
    """
    x, = _to_fp32(x)
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    safe = jnp.maximum(norm, _EPS_UNIT)
    direction = x / safe

    sqrt_c = jnp.sqrt(c)
    arg = jnp.clip(sqrt_c * norm, 0.0, 1.0 - 1e-6)
    magnitude = (2.0 / sqrt_c) * jnp.arctanh(arg)

    return magnitude * direction

@jax.jit
def exp_map_at_basepoint(v, o, c: float = 1.0):
    """
    exp_o(v) with origin-based exp then Möbius transport to o.
    """
    v, o = _to_fp32(v, o)
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = jnp.maximum(v_norm, _EPS_UNIT)

    sqrt_c = jnp.sqrt(c)
    t = jnp.clip((sqrt_c * v_norm) / 2.0, 0.0, 10.0)  # avoid overflow in tanh
    radius = jnp.tanh(t) / sqrt_c
    direction = v / v_norm
    point_at_origin = radius * direction

    return mobius_addition(o, point_at_origin, c)

# ---------------------------------------------------------------------
# Projection & gradient conversion
# ---------------------------------------------------------------------
@jax.jit
def poincare_proj(x, c: float = 1.0, eps_margin: float = 1e-4):
    """
    Project to inside (1 - eps_margin) * R to avoid boundary.
    """
    x, = _to_fp32(x)
    R = 1.0 / jnp.sqrt(c)
    max_norm = (1.0 - eps_margin) * R
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norm, _EPS_UNIT))
    return x * scale

@jax.jit
def riemannian_gradient_conversion(euclidean_grad, x, c: float = 1.0):
    """
    g_R = ((1 - c||x||^2)/2)^2 * g_E
    """
    euclidean_grad, x = _to_fp32(euclidean_grad, x)
    norm_sq = jnp.sum(x * x, axis=-1, keepdims=True)
    cf = (1.0 - c * norm_sq) / 2.0
    return (cf * cf) * euclidean_grad

# ---------------------------------------------------------------------
# Frechet mean & basepoint updates
# ---------------------------------------------------------------------
def compute_frechet_mean(points, c: float = 1.0, max_iter: int = 50, tol: float = 1e-6):
    """
    Iterative Fréchet mean (not jitted; call infrequently, e.g., per epoch).
    """
    points, = _to_fp32(points)
    # Init from (projected) Euclidean mean
    eu = jnp.mean(points, axis=0)
    current = poincare_proj(eu, c, eps_margin=1e-5)

    for _ in range(max_iter):
        # log maps at current
        def _log(p):
            return log_map_at_basepoint(p, current, c)
        log_pts = jax.vmap(_log)(points)  # (N, D)
        tangent_mean = jnp.mean(log_pts, axis=0)      # (D,)

        if jnp.linalg.norm(tangent_mean) < tol:
            break

        current = exp_map_at_basepoint(tangent_mean, current, c)
        current = poincare_proj(current, c, eps_margin=1e-5)

    return current

@jax.jit
def update_basepoint_ema(prev_o, batch_points, c: float = 1.0, ema: float = 0.9):
    """
    Manifold-correct EMA for basepoint:
    Move from prev_o toward batch Fréchet step in tangent space at prev_o.
    """
    prev_o, batch_points = _to_fp32(prev_o, batch_points)

    # One tangent step toward batch mean:
    # approximate mean in tangent via average of log maps at prev_o
    def _log(p):
        return log_map_at_basepoint(p, prev_o, c)
    log_pts = jax.vmap(_log)(batch_points)
    step = jnp.mean(log_pts, axis=0)  # (D,)

    # EMA: move a (1-ema) fraction of this step
    delta = (1.0 - ema) * step
    new_o = exp_map_at_basepoint(delta, prev_o, c)
    new_o = poincare_proj(new_o, c, eps_margin=1e-5)
    return new_o

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
def hyperbolic_diagnostics(embeddings, c: float = 1.0, dmax: float | None = None):
    """
    Returns simple fp32 scalars useful for logging.
    """
    emb, = _to_fp32(embeddings)
    R = 1.0 / jnp.sqrt(c)
    norms = jnp.linalg.norm(emb, axis=-1)

    max_norm_ratio  = (jnp.max(norms)  / R).astype(jnp.float32)
    mean_norm_ratio = (jnp.mean(norms) / R).astype(jnp.float32)
    near_boundary   = (jnp.mean((norms > (0.8 * R)).astype(jnp.float32)) * 100.0).astype(jnp.float32)
    safe_region     = (jnp.mean((norms < (0.5 * R)).astype(jnp.float32)) * 100.0).astype(jnp.float32)
    spread          = jnp.std(norms).astype(jnp.float32)

    out = {
        "max_norm_ratio":  max_norm_ratio,
        "mean_norm_ratio": mean_norm_ratio,
        "near_boundary_pct": near_boundary,
        "safe_region_pct":   safe_region,
        "embedding_spread":  spread,
    }

    if dmax is not None:
        out["dmax"] = jnp.asarray(dmax, jnp.float32)

    return out

# ---------------------------------------------------------------------
# Quick self-check (run manually; not used in training)
# ---------------------------------------------------------------------
def _self_check():
    x = jnp.array([0.1, 0.2, 0.3], jnp.float32)
    y = jnp.array([0.2, 0.1, 0.1], jnp.float32)
    o = jnp.array([0.05, 0.05, 0.05], jnp.float32)
    c = 1.0

    s = mobius_addition(x, y, c)
    d = poincare_distance(x, y, c)
    logx = log_map_at_basepoint(x, o, c)
    exlogx = exp_map_at_basepoint(logx, o, c)
    rec_err = jnp.linalg.norm(x - exlogx)

    diag = hyperbolic_diagnostics(jnp.stack([x, y], axis=0), c)

    print("x ⊕ y:", s)
    print("d_c(x,y):", d)
    print("‖x - exp_o(log_o(x))‖:", rec_err)
    print("diag:", {k: float(v) for k, v in diag.items()})
