import jax
import jax.numpy as jnp

from ctrlax.typing import Action, Key


def zeros_mean(low: Action, horizon: int) -> Action:
    return jax.tree.map(
        lambda leaf: jnp.zeros((horizon, *leaf.shape), dtype=leaf.dtype), low
    )


def validate_matching_bounds(low: Action, high: Action, solver_name: str) -> None:
    low_structure = jax.tree.structure(low)
    high_structure = jax.tree.structure(high)
    assert low_structure == high_structure, (
        f"{solver_name}'s low and high must have matching PyTree structure, "
        f"got {low_structure} vs {high_structure}."
    )
    for low_leaf, high_leaf in zip(
        jax.tree.leaves(low), jax.tree.leaves(high), strict=True
    ):
        assert low_leaf.shape == high_leaf.shape, (
            f"{solver_name}'s low and high leaves must have matching shapes, "
            f"got {low_leaf.shape} vs {high_leaf.shape}."
        )


def sample_gaussian_actions(
    key: Key,
    mean: Action,
    std: Action,
    num_samples: int,
    low: Action,
    high: Action,
) -> Action:
    """Draws (num_samples, horizon, ...) candidates from N(mean, std), clipped
    to [low, high]."""
    mean_leaves, treedef = jax.tree.flatten(mean)
    std_leaves = jax.tree.leaves(std)
    keys = jax.random.split(key, len(mean_leaves))

    sampled_leaves = [
        mean_leaf[None]
        + std_leaf[None] * jax.random.normal(key_leaf, (num_samples, *mean_leaf.shape))
        for key_leaf, mean_leaf, std_leaf in zip(
            keys, mean_leaves, std_leaves, strict=True
        )
    ]
    samples = jax.tree.unflatten(treedef, sampled_leaves)
    return jax.tree.map(jnp.clip, samples, low, high)
