from typing import Protocol

from ctrlax.typing import Action, DynamicsState, Key


class Solver[TState](Protocol):
    """Duck-typed solver contract. All configuration is set at construction;
    diagnostics go through lox.log, never return values."""

    def init(self) -> TState: ...

    def step(
        self,
        key: Key,
        state: TState,
        dynamics_state: DynamicsState,
    ) -> tuple[TState, Action]: ...
