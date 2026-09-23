import jax
import jax.numpy as jnp
import pytest

from ctrlax.solvers._actions import (
    sample_gaussian_actions,
    validate_matching_bounds,
    zeros_mean,
)


def test_sample_gaussian_actions_shapes_and_clips():
    mean = {"a": jnp.zeros((4, 2)), "b": jnp.zeros((4,))}
    std = jax.tree.map(lambda leaf: jnp.full_like(leaf, 5.0), mean)
    low = {"a": jnp.full((2,), -1.0), "b": jnp.array(-1.0)}
    high = {"a": jnp.full((2,), 1.0), "b": jnp.array(1.0)}
    samples = sample_gaussian_actions(jax.random.key(0), mean, std, 16, low, high)
    assert samples["a"].shape == (16, 4, 2)
    assert samples["b"].shape == (16, 4)
    assert jnp.all(jnp.abs(samples["a"]) <= 1.0)
    assert jnp.all(jnp.abs(samples["b"]) <= 1.0)


def test_zeros_mean_matches_low():
    low = {"a": jnp.zeros((2,)), "b": jnp.zeros(())}
    mean = zeros_mean(low, 5)
    assert mean["a"].shape == (5, 2)
    assert mean["b"].shape == (5,)


def test_validate_matching_bounds_rejects_structure_and_shape_mismatch():
    with pytest.raises(AssertionError):
        validate_matching_bounds({"a": jnp.zeros(2)}, {"b": jnp.zeros(2)}, "X")
    with pytest.raises(AssertionError):
        validate_matching_bounds(jnp.zeros(2), jnp.zeros(3), "X")
