"""Shared toy problem for the examples: a 1-D point mass driven toward a goal.

Deliberately depends on nothing but jax, so the examples exercise ctrlax's
init/step/rollout contract on their own — including the state (dynamics carry)
vs. observation (what cost_fn sees) split that matches gxm.Dynamics.step's real
(next_state, observation) shape.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

GOAL = 5.0
DT = 0.1
ACTION_BOUND = 8.0


class PointMassState(NamedTuple):
    pos: jax.Array
    vel: jax.Array


def toy_dynamics(
    key: jax.Array, state: PointMassState, action: jax.Array
) -> tuple[PointMassState, PointMassState]:
    del key  # deterministic toy dynamics
    accel = action[0]
    new_vel = state.vel + accel * DT
    new_pos = state.pos + new_vel * DT
    next_state = PointMassState(pos=new_pos, vel=new_vel)
    return next_state, next_state  # fully observed toy problem: obs == state


def cost_fn(observations: PointMassState, actions: jax.Array) -> jax.Array:
    pos_cost = jnp.sum((observations.pos - GOAL) ** 2)
    action_cost = 0.01 * jnp.sum(actions**2)
    return pos_cost + action_cost


def bounds() -> tuple[jax.Array, jax.Array]:
    """The point mass's action-space bounds, as the plain arrays solvers take."""
    return jnp.full((1,), -ACTION_BOUND), jnp.full((1,), ACTION_BOUND)
