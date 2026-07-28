import jax
import jax.numpy as jnp

from ctrlax.typing import Action


def zeros_mean(low: Action, horizon: int) -> Action:
    """Builds a zero-initialized action sequence, matching low's own per-leaf shape/dtype."""
    return jax.tree_util.tree_map(lambda leaf: jnp.zeros((horizon, *leaf.shape), dtype=leaf.dtype), low)


def validate_matching_bounds(low: Action, high: Action, solver_name: str) -> None:
    """Fails fast if low/high don't describe the same action structure."""
    low_structure = jax.tree_util.tree_structure(low)
    high_structure = jax.tree_util.tree_structure(high)
    if low_structure != high_structure:
        raise ValueError(
            f"{solver_name}'s low and high must have matching PyTree structure, "
            f"got {low_structure} vs {high_structure}."
        )
