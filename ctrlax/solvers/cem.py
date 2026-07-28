from dataclasses import dataclass
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from ctrlax.rollout import rollout
from ctrlax.typing import (
    Action,
    Array,
    Dynamics,
    DynamicsState,
    InfoDict,
    Key,
    SolverState,
    TrajectoryCostFn,
)
from ctrlax.solvers._sampling import sample_gaussian_actions
from ctrlax.solvers._spaces import validate_matching_bounds, zeros_mean

_MIN_STD = 1e-3


class CEMState(NamedTuple):
    """Only the warm-started search center persists across step() calls — std
    resets to init_std at the start of every call (standard CEM-MPC practice;
    avoids starting the next timestep's search already collapsed narrow)."""

    mean: Action


def _shift_mean(mean: Action) -> Action:
    """Warm-start for the next call: drop the now-executed first action, repeat
    the last planned action to fill the newly-exposed final horizon step."""

    def shift_leaf(leaf):
        return jnp.concatenate([leaf[1:], leaf[-1:]], axis=0)

    return jax.tree_util.tree_map(shift_leaf, mean)


@dataclass(frozen=True)
class CEM:
    """Cross-Entropy Method: repeatedly sample candidate action sequences from a
    Gaussian, keep the lowest-cost elites, refit the Gaussian to them, repeat.
    Runs num_iterations refinement rounds to convergence within a single step()
    call — that inner loop is private, never exposed to the caller.

    Continuous action spaces only, described directly by low/high bounds (real
    array PyTrees, matching the action's own structure) — not a Space object.
    low/high are validated to have matching structure at construction time.

    dynamics/cost_fn are static config, same as every other field — bound once
    at construction, matching the "one solver instance is fully bound to one
    problem" principle already used for low/high/horizon.
    """

    dynamics: Dynamics
    cost_fn: TrajectoryCostFn
    low: Action
    high: Action
    horizon: int
    num_samples: int = 500
    num_elites: int = 50
    num_iterations: int = 5
    init_std: float = 1.0

    def __post_init__(self):
        validate_matching_bounds(self.low, self.high, "CEM")

    def init(self) -> SolverState:
        return CEMState(mean=zeros_mean(self.low, self.horizon))

    def step(
        self,
        key: Key,
        state: SolverState,
        dynamics_state: DynamicsState,
    ) -> Tuple[Action, SolverState, InfoDict]:
        std0 = jax.tree_util.tree_map(lambda leaf: jnp.full_like(leaf, self.init_std), state.mean)
        iteration_keys = jax.random.split(key, self.num_iterations)

        def rollout_and_score(k: Key, actions: Action) -> Array:
            observations = rollout(k, dynamics_state, actions, self.dynamics)
            return self.cost_fn(observations, actions)

        def refine(carry, iter_key):
            mean, std = carry
            sample_key, rollout_key = jax.random.split(iter_key)

            candidates = sample_gaussian_actions(sample_key, mean, std, self.num_samples, self.low, self.high)
            rollout_keys = jax.random.split(rollout_key, self.num_samples)
            costs = jax.vmap(rollout_and_score)(rollout_keys, candidates)

            elite_idx = jnp.argsort(costs)[: self.num_elites]
            elites = jax.tree_util.tree_map(lambda c: c[elite_idx], candidates)

            new_mean = jax.tree_util.tree_map(lambda e: jnp.mean(e, axis=0), elites)
            new_std = jax.tree_util.tree_map(lambda e: jnp.maximum(jnp.std(e, axis=0), _MIN_STD), elites)

            return (new_mean, new_std), costs[elite_idx[0]]

        (final_mean, _), best_cost_per_iteration = jax.lax.scan(
            refine, (state.mean, std0), iteration_keys
        )

        info = {"best_cost": best_cost_per_iteration[-1], "best_cost_per_iteration": best_cost_per_iteration}
        return final_mean, CEMState(mean=_shift_mean(final_mean)), info
