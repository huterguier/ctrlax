import jax
import jax.numpy as jnp

from ctrlax.rollout import rollout, score_candidates


def dynamics(key, state, action):
    del key
    next_state = state + action
    return next_state, next_state


def test_rollout_returns_observation_per_step():
    actions = jnp.array([1.0, 2.0, 3.0])
    observations = rollout(jax.random.key(0), jnp.array(0.0), actions, dynamics)
    assert jnp.allclose(observations, jnp.array([1.0, 3.0, 6.0]))


def test_score_candidates_scores_each_candidate():
    candidates = jnp.array([[1.0, 1.0], [0.0, 0.0]])
    costs = score_candidates(
        jax.random.key(0),
        jnp.array(0.0),
        candidates,
        dynamics,
        lambda observations, actions: jnp.sum(observations),
    )
    assert jnp.allclose(costs, jnp.array([3.0, 0.0]))
