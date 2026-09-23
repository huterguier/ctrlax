import jax
import jax.numpy as jnp
import lox

from ctrlax.solvers import RandomShooting

HORIZON = 3
TARGET = 2.0


def dynamics(key, state, action):
    del key
    return state, action


def cost_fn(observations, actions):
    del actions
    return jnp.sum((observations - TARGET) ** 2)


def make():
    return RandomShooting(
        dynamics=dynamics,
        cost_fn=cost_fn,
        low=jnp.array(-10.0),
        high=jnp.array(10.0),
        horizon=HORIZON,
        num_samples=2000,
        std=3.0,
    )


def test_returns_lowest_cost_sample():
    solver = make()
    (state, plan), logs = lox.spool(solver.step)(
        jax.random.key(0), solver.init(), jnp.array(0.0)
    )
    assert state is None
    assert plan.shape == (HORIZON,)
    assert jnp.allclose(plan, TARGET, atol=0.5)
    assert logs["best_cost"].shape == (1,)
    assert logs["best_cost"][0] <= logs["mean_cost"][0]


def test_step_is_jittable_and_vmappable():
    solver = make()
    keys = jax.random.split(jax.random.key(0), 2)
    step = jax.jit(jax.vmap(solver.step, in_axes=(0, None, 0)))
    _, plans = step(keys, solver.init(), jnp.zeros(2))
    assert plans.shape == (2, HORIZON)
