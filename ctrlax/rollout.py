import jax

from ctrlax.typing import (
    Action,
    Array,
    Dynamics,
    DynamicsState,
    Key,
    Observation,
    TrajectoryCostFn,
)


def rollout(
    key: Key,
    dynamics_state: DynamicsState,
    actions: Action,
    dynamics: Dynamics,
) -> Observation:
    """Returns the observation after each action; the dynamics state is not returned."""
    horizon = jax.tree.leaves(actions)[0].shape[0]
    keys = jax.random.split(key, horizon)

    def step(state, inputs):
        key_step, action = inputs
        return dynamics(key_step, state, action)

    _, observations = jax.lax.scan(step, dynamics_state, (keys, actions))
    return observations


def score_candidates(
    key: Key,
    dynamics_state: DynamicsState,
    candidates: Action,
    dynamics: Dynamics,
    cost_fn: TrajectoryCostFn,
) -> Array:
    """Cost of each (num_candidates, horizon, ...) candidate from dynamics_state."""
    num_candidates = jax.tree.leaves(candidates)[0].shape[0]
    keys = jax.random.split(key, num_candidates)

    def score(key, actions):
        return cost_fn(rollout(key, dynamics_state, actions, dynamics), actions)

    return jax.vmap(score)(keys, candidates)
