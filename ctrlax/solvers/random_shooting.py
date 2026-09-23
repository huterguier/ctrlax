from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox

from ctrlax.rollout import score_candidates
from ctrlax.solvers._actions import (
    sample_gaussian_actions,
    validate_matching_bounds,
    zeros_mean,
)
from ctrlax.typing import Action, Dynamics, DynamicsState, Key, TrajectoryCostFn


@dataclass(frozen=True)
class RandomShooting:
    """Sample num_samples action sequences from a zero-mean Gaussian and return
    the lowest-cost one. Stateless. Logs "best_cost" and "mean_cost", tagged
    "ctrlax"."""

    dynamics: Dynamics
    cost_fn: TrajectoryCostFn
    low: Action
    high: Action
    horizon: int
    num_samples: int = 500
    std: float = 1.0

    def __post_init__(self):
        validate_matching_bounds(self.low, self.high, type(self).__name__)

    def init(self) -> None:
        return None

    def step(
        self,
        key: Key,
        state: None,
        dynamics_state: DynamicsState,
    ) -> tuple[None, Action]:
        del state
        key_sample, key_rollout = jax.random.split(key)

        mean = zeros_mean(self.low, self.horizon)
        std = jax.tree.map(lambda leaf: jnp.full_like(leaf, self.std), mean)
        candidates = sample_gaussian_actions(
            key_sample, mean, std, self.num_samples, self.low, self.high
        )

        costs = score_candidates(
            key_rollout, dynamics_state, candidates, self.dynamics, self.cost_fn
        )

        best_idx = jnp.argmin(costs)
        best_actions = jax.tree.map(lambda leaf: leaf[best_idx], candidates)

        lox.log(
            {"best_cost": costs[best_idx], "mean_cost": jnp.mean(costs)},
            tags=("ctrlax",),
        )
        return None, best_actions
