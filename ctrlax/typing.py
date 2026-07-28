from typing import Any, Callable, Dict, Protocol, Tuple

import jax

Key = jax.Array
Array = jax.Array

PyTree = Any
DynamicsState = PyTree
Observation = PyTree
Action = PyTree
InfoDict = Dict[str, Any]
SolverState = PyTree

Dynamics = Callable[[Key, DynamicsState, Action], Tuple[DynamicsState, Observation]]
TrajectoryCostFn = Callable[[Observation, Action], Array]


class Solver(Protocol):
    """Structural contract every solver class satisfies. Purely duck-typed —
    a solver does not need to inherit from this, just expose matching methods.
    All configuration — dynamics, cost_fn, action-space description, horizon,
    hyperparameters — lives as plain instance attributes set at construction,
    and is entirely solver-specific (e.g. CEM/RandomShooting take low/high
    bounds; a future discrete solver like MCTS would take a num_actions
    cardinality instead — there's no shared "Space" vocabulary forcing
    continuous and discrete solvers to look alike). dynamics/cost_fn are static
    just like every other field — passing them per-call instead never had a
    real use case, and one solver instance is already fully bound to one
    problem (see low/high/horizon). init() just builds the initial
    SolverState from self; nothing is passed to init/step directly.

    step()'s own state comes first after key, bare "state" — mirroring how
    Dynamics.step(key, state, action) treats its own state as bare "state" too.
    The foreign, passed-in dynamics state is explicitly qualified as
    dynamics_state, since it belongs to the dynamics, not the solver.
    """

    def init(self) -> SolverState: ...

    def step(
        self,
        key: Key,
        state: SolverState,
        dynamics_state: DynamicsState,
    ) -> Tuple[Action, SolverState, InfoDict]: ...
