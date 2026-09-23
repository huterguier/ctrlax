"""Shared toy problem for the examples: a 1-D point mass driven toward a goal.

Depends only on jax and ctrlax.typing, so the examples exercise ctrlax's
init/step/rollout contract on their own — including the state (dynamics carry)
vs. observation (what cost_fn sees) split that matches gxm.Dynamics.step's real
(next_state, observation) shape.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ctrlax.typing import Array, Key

GOAL = 5.0
DT = 0.1
ACTION_BOUND = 8.0


@jax.tree_util.register_dataclass
@dataclass
class PointMassState:
    pos: Array
    vel: Array


def toy_dynamics(
    key: Key, state: PointMassState, action: Array
) -> tuple[PointMassState, PointMassState]:
    del key  # deterministic toy dynamics
    accel = action[0]
    new_vel = state.vel + accel * DT
    new_pos = state.pos + new_vel * DT
    next_state = PointMassState(pos=new_pos, vel=new_vel)
    return next_state, next_state  # fully observed toy problem: obs == state


def cost_fn(observations: PointMassState, actions: Array) -> Array:
    pos_cost = jnp.sum((observations.pos - GOAL) ** 2)
    action_cost = 0.01 * jnp.sum(actions**2)
    return pos_cost + action_cost


def bounds() -> tuple[Array, Array]:
    """The point mass's action-space bounds, as the plain arrays solvers take."""
    return jnp.full((1,), -ACTION_BOUND), jnp.full((1,), ACTION_BOUND)
