import jax

from ctrlax.typing import Action, Dynamics, DynamicsState, Key, Observation


def rollout(
    key: Key,
    dynamics_state: DynamicsState,
    actions: Action,
    dynamics: Dynamics,
) -> Observation:
    """Rolls actions forward through dynamics via lax.scan.

    Raw dynamics state is only ever the scan's internal carry (required to keep
    calling dynamics()) — never returned. Returns the *observations* reached after
    each action (length == horizon); dynamics_state itself is not included.
    """
    horizon = jax.tree_util.tree_leaves(actions)[0].shape[0]
    step_keys = jax.random.split(key, horizon)

    def step(
        state: DynamicsState, inputs: tuple[Key, Action]
    ) -> tuple[DynamicsState, Observation]:
        step_key, action = inputs
        next_state, obs = dynamics(step_key, state, action)
        return next_state, obs

    _, observations = jax.lax.scan(step, dynamics_state, (step_keys, actions))
    return observations
