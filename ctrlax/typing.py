from collections.abc import Callable
from typing import Any

import jax

type Key = jax.Array
type Array = jax.Array

type PyTree = Any
type DynamicsState = PyTree
type Observation = PyTree
type Action = PyTree

type Dynamics = Callable[
    [Key, DynamicsState, Action], tuple[DynamicsState, Observation]
]
type TrajectoryCostFn = Callable[[Observation, Action], Array]
