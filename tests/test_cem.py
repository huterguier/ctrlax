"""CEM's proposals hook: extra first-iteration candidates compete for the elites."""

import jax
import jax.numpy as jnp
import lox

from ctrlax.solvers import CEM

HORIZON = 3
TARGET = 8.0


def dynamics(key, state, action):
    del key
    return state, action


def cost_fn(observations, actions):
    del actions
    return jnp.sum((observations - TARGET) ** 2)


def make(**kwargs):
    return CEM(
        dynamics=dynamics,
        cost_fn=cost_fn,
        low=jnp.array(-10.0),
        high=jnp.array(10.0),
        horizon=HORIZON,
        num_samples=32,
        num_elites=1,
        num_iterations=2,
        init_std=0.1,
        **kwargs,
    )


def test_proposal_wins_when_it_is_best():
    solver = make()
    proposals = jnp.full((4, HORIZON), TARGET)
    (_, plan), logs = lox.spool(solver.step)(
        jax.random.key(0), solver.init(), jnp.array(0.0), proposals
    )
    assert jnp.allclose(plan, TARGET, atol=0.05)
    assert logs["best_cost"].shape == (2,)
    assert logs["std"].shape == (1, *plan.shape)


def test_proposals_are_clipped_to_bounds():
    solver = make()
    proposals = jnp.full((1, HORIZON), 100.0)
    _, plan = solver.step(jax.random.key(0), solver.init(), jnp.array(0.0), proposals)
    assert jnp.allclose(plan, 10.0, atol=0.05)


def test_without_proposals_stays_near_the_warm_start():
    solver = make()
    (state, plan), logs = lox.spool(solver.step)(
        jax.random.key(0), solver.init(), jnp.array(0.0)
    )
    assert jnp.all(jnp.abs(plan) < 1.0)
    assert state.mean.shape == plan.shape
    assert logs["best_cost"].shape == (2,)
    assert logs["std"].shape == (1, *plan.shape)


def test_step_is_jittable_and_vmappable():
    solver = make()
    proposals = jnp.full((2, 5, HORIZON), TARGET)
    keys = jax.random.split(jax.random.key(0), 2)
    states = jax.vmap(lambda _: solver.init())(jnp.arange(2))
    step = jax.jit(jax.vmap(solver.step))
    _, plans = step(keys, states, jnp.zeros(2), proposals)
    assert plans.shape == (2, HORIZON)
    assert jnp.allclose(plans, TARGET, atol=0.05)
