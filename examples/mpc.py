"""Smoke test: a toy point-mass MPC loop driven by RandomShooting.

Proves ctrlax's init/step/rollout contract wires together correctly.

    uv run python examples/mpc.py
"""

import jax
import jax.numpy as jnp

import ctrlax as cx
from point_mass import PointMassState, bounds, cost_fn, toy_dynamics


def main():
    horizon = 20
    low, high = bounds()

    solver = cx.solvers.RandomShooting(toy_dynamics, cost_fn, low, high, horizon, num_samples=500, std=2.0)
    solver_state = solver.init()

    key = jax.random.key(0)
    dynamics_state = PointMassState(pos=jnp.array(0.0), vel=jnp.array(0.0))

    for t in range(30):
        key, plan_key, env_key = jax.random.split(key, 3)
        plan, solver_state, info = solver.step(plan_key, solver_state, dynamics_state)
        action = jax.tree_util.tree_map(lambda leaf: leaf[0], plan)
        dynamics_state, _ = toy_dynamics(env_key, dynamics_state, action)
        print(
            f"t={t:2d}  pos={dynamics_state.pos:6.3f}  vel={dynamics_state.vel:6.3f}  "
            f"action={action[0]:6.3f}  best_cost={info['best_cost']:8.3f}"
        )


if __name__ == "__main__":
    main()
