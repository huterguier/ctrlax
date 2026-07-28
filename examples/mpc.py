"""Smoke test: a toy point-mass MPC loop driven by RandomShooting.

Proves ctrlax's init/step/rollout contract wires together correctly.

    uv run python examples/mpc.py
"""

import jax
import jax.numpy as jnp

import ctrlax as cx
from point_mass import PointMassState, bounds, cost_fn, toy_model


def main():
    horizon = 20
    low, high = bounds()

    solver = cx.solvers.RandomShooting(toy_model, cost_fn, low, high, horizon, num_samples=500, std=2.0)
    solver_state = solver.init()

    key = jax.random.key(0)
    model_state = PointMassState(pos=jnp.array(0.0), vel=jnp.array(0.0))

    for t in range(30):
        key, plan_key, env_key = jax.random.split(key, 3)
        plan, solver_state, info = solver.step(plan_key, solver_state, model_state)
        action = jax.tree_util.tree_map(lambda leaf: leaf[0], plan)
        model_state, _ = toy_model(env_key, model_state, action)
        print(
            f"t={t:2d}  pos={model_state.pos:6.3f}  vel={model_state.vel:6.3f}  "
            f"action={action[0]:6.3f}  best_cost={info['best_cost']:8.3f}"
        )


if __name__ == "__main__":
    main()
