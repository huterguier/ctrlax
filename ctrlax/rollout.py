from typing import Tuple

import jax

from ctrlax.typing import Action, Key, ModelState, Observation, TransitionModel


def rollout(
    key: Key,
    model_state: ModelState,
    actions: Action,
    model: TransitionModel,
) -> Observation:
    """Rolls actions forward through model via lax.scan.

    Raw dynamics state is only ever the scan's internal carry (required to keep
    calling model()) — never returned. Returns the *observations* reached after
    each action (length == horizon); model_state itself is not included.
    """
    horizon = jax.tree_util.tree_leaves(actions)[0].shape[0]
    step_keys = jax.random.split(key, horizon)

    def step(state: ModelState, inputs: Tuple[Key, Action]) -> Tuple[ModelState, Observation]:
        step_key, action = inputs
        next_state, obs = model(step_key, state, action)
        return next_state, obs

    _, observations = jax.lax.scan(step, model_state, (step_keys, actions))
    return observations
