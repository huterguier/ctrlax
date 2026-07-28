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


class RandomShootingState(NamedTuple):
    """Fixed sampling distribution — never adapted between calls."""

    mean: Action
    std: Action


@dataclass(frozen=True)
class RandomShooting:
    """Simplest shooting solver: draw num_samples action sequences from a fixed
    zero-mean Gaussian, roll each out, and return the lowest-cost one. No inner
    refinement loop, no warm-starting.

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
    std: float = 1.0

    def __post_init__(self):
        validate_matching_bounds(self.low, self.high, "RandomShooting")

    def init(self) -> SolverState:
        mean = zeros_mean(self.low, self.horizon)
        std_tree = jax.tree_util.tree_map(lambda leaf: jnp.full_like(leaf, self.std), mean)
        return RandomShootingState(mean=mean, std=std_tree)

    def step(
        self,
        key: Key,
        state: SolverState,
        dynamics_state: DynamicsState,
    ) -> Tuple[Action, SolverState, InfoDict]:
        sample_key, rollout_key = jax.random.split(key)

        candidates = sample_gaussian_actions(
            sample_key, state.mean, state.std, self.num_samples, self.low, self.high
        )

        def rollout_and_score(k: Key, actions: Action) -> Array:
            observations = rollout(k, dynamics_state, actions, self.dynamics)
            return self.cost_fn(observations, actions)

        rollout_keys = jax.random.split(rollout_key, self.num_samples)
        costs = jax.vmap(rollout_and_score)(rollout_keys, candidates)

        best_idx = jnp.argmin(costs)
        best_actions = jax.tree_util.tree_map(lambda leaf: leaf[best_idx], candidates)

        info = {"best_cost": costs[best_idx], "mean_cost": jnp.mean(costs)}
        return best_actions, state, info
