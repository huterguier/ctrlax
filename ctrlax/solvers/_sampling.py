import jax
import jax.numpy as jnp

from ctrlax.typing import Action, Key


def sample_gaussian_actions(
    key: Key,
    mean: Action,
    std: Action,
    num_samples: int,
    low: Action | None = None,
    high: Action | None = None,
) -> Action:
    """Draws num_samples i.i.d. action-sequence candidates from N(mean, std), per leaf.

    If low/high are given (one leaf each, shape matching the space's per-step
    shape — broadcasts against the sampled (num_samples, horizon, *shape)
    leaves), samples are clipped to stay within the action space's bounds.
    """
    mean_leaves, treedef = jax.tree_util.tree_flatten(mean)
    std_leaves = jax.tree_util.tree_leaves(std)
    leaf_keys = jax.random.split(key, len(mean_leaves))

    sampled_leaves = [
        mean_leaf[None]
        + std_leaf[None] * jax.random.normal(k, (num_samples, *mean_leaf.shape))
        for k, mean_leaf, std_leaf in zip(leaf_keys, mean_leaves, std_leaves)
    ]
    samples = jax.tree_util.tree_unflatten(treedef, sampled_leaves)

    if low is not None:
        samples = jax.tree_util.tree_map(jnp.clip, samples, low, high)

    return samples
