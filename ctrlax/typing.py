from typing import Any, Callable, Dict, Protocol, Tuple

import jax

Key = jax.Array
Array = jax.Array

PyTree = Any
ModelState = PyTree
Observation = PyTree
Action = PyTree
InfoDict = Dict[str, Any]
SolverState = PyTree

TransitionModel = Callable[[Key, ModelState, Action], Tuple[ModelState, Observation]]
TrajectoryCostFn = Callable[[Observation, Action], Array]


class Solver(Protocol):
    """Structural contract every solver class satisfies. Purely duck-typed —
    a solver does not need to inherit from this, just expose matching methods.
    All configuration — model, cost_fn, action-space description, horizon,
    hyperparameters — lives as plain instance attributes set at construction,
    and is entirely solver-specific (e.g. CEM/RandomShooting take low/high
    bounds; a future discrete solver like MCTS would take a num_actions
    cardinality instead — there's no shared "Space" vocabulary forcing
    continuous and discrete solvers to look alike). model/cost_fn are static
    just like every other field — passing them per-call instead never had a
    real use case, and one solver instance is already fully bound to one
    problem (see low/high/horizon). init() just builds the initial
    SolverState from self; nothing is passed to init/step directly.

    step()'s own state comes first after key, bare "state" — mirroring how
    Model.step(key, state, action) treats its own state as bare "state" too.
    The foreign, passed-in dynamics state is explicitly qualified as
    model_state, since it belongs to the model, not the solver.
    """

    def init(self) -> SolverState: ...

    def step(
        self,
        key: Key,
        state: SolverState,
        model_state: ModelState,
    ) -> Tuple[Action, SolverState, InfoDict]: ...
