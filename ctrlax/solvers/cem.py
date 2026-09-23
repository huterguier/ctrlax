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


@jax.tree_util.register_dataclass
@dataclass
class CEMState:
    mean: Action


def _shift_mean(mean: Action) -> Action:
    def shift_leaf(leaf):
        return jnp.concatenate([leaf[1:], leaf[-1:]], axis=0)

    return jax.tree.map(shift_leaf, mean)


@dataclass(frozen=True)
class CEM:
    """Cross-Entropy Method over action sequences: sample from a Gaussian, refit
    it to the lowest-cost elites, repeat num_iterations times per step().

    The mean is warm-started across calls; std resets to init_std each call.
    Logs "best_cost" per iteration and the final "std", tagged "ctrlax".
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
    min_std: float = 1e-3

    def __post_init__(self):
        validate_matching_bounds(self.low, self.high, type(self).__name__)

    def init(self) -> CEMState:
        return CEMState(mean=zeros_mean(self.low, self.horizon))

    def step(
        self,
        key: Key,
        state: CEMState,
        dynamics_state: DynamicsState,
        proposals: Action | None = None,
    ) -> tuple[CEMState, Action]:
        """proposals: extra (num_proposals, horizon, ...) candidates for the
        first iteration; they compete for elites but never shape the Gaussian
        directly."""
        std0 = jax.tree.map(lambda leaf: jnp.full_like(leaf, self.init_std), state.mean)
        keys = jax.random.split(key, self.num_iterations)

        def refine(key, mean, std, extra):
            key_sample, key_rollout = jax.random.split(key)

            candidates = sample_gaussian_actions(
                key_sample, mean, std, self.num_samples, self.low, self.high
            )
            if extra is not None:
                candidates = jax.tree.map(
                    lambda c, e: jnp.concatenate([c, e], axis=0), candidates, extra
                )
            costs = score_candidates(
                key_rollout, dynamics_state, candidates, self.dynamics, self.cost_fn
            )

            elite_idx = jnp.argsort(costs)[: self.num_elites]
            elites = jax.tree.map(lambda c: c[elite_idx], candidates)

            new_mean = jax.tree.map(lambda e: jnp.mean(e, axis=0), elites)
            new_std = jax.tree.map(
                lambda e: jnp.maximum(jnp.std(e, axis=0), self.min_std), elites
            )
            lox.log({"best_cost": costs[elite_idx[0]]}, tags=("ctrlax",))
            return new_mean, new_std

        if proposals is not None:
            proposals = jax.tree.map(jnp.clip, proposals, self.low, self.high)
        mean, std = refine(keys[0], state.mean, std0, proposals)

        def scan_refine(carry, key_iteration):
            return refine(key_iteration, *carry, None), None

        (mean, std), _ = jax.lax.scan(scan_refine, (mean, std), keys[1:])

        lox.log({"std": std}, tags=("ctrlax",))
        return CEMState(mean=_shift_mean(mean)), mean
