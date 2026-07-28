"""Compares RandomShooting vs CEM on the same toy point-mass MPC problem, at a
matched compute budget (1000 total rollouts/step for both), and plots the
convergence of position and planning cost over the control loop.

    uv run python examples/compare.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from point_mass import GOAL, PointMassState, bounds, cost_fn, toy_dynamics

import ctrlax as cx

OUTPUT_PATH = Path(__file__).parent / "compare.png"

NUM_STEPS = 40
HORIZON = 20
SEED = 0

# Validated categorical palette (references/palette.md), fixed order, light mode.
COLOR_RANDOM_SHOOTING = "#2a78d6"  # slot 1 — blue
COLOR_CEM = "#1baf7a"  # slot 2 — aqua
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def run(solver, num_steps=NUM_STEPS):
    solver_state = solver.init()

    key = jax.random.key(SEED)
    dynamics_state = PointMassState(pos=jnp.array(0.0), vel=jnp.array(0.0))

    positions, best_costs = [], []
    for _ in range(num_steps):
        key, plan_key, env_key = jax.random.split(key, 3)
        plan, solver_state, info = solver.step(plan_key, solver_state, dynamics_state)
        action = jax.tree_util.tree_map(lambda leaf: leaf[0], plan)
        dynamics_state, _ = toy_dynamics(env_key, dynamics_state, action)
        positions.append(float(dynamics_state.pos))
        best_costs.append(float(info["best_cost"]))
    return positions, best_costs


def main():
    low, high = bounds()
    random_shooting = cx.solvers.RandomShooting(
        toy_dynamics, cost_fn, low, high, HORIZON, num_samples=1000, std=2.0
    )
    cem = cx.solvers.CEM(
        toy_dynamics,
        cost_fn,
        low,
        high,
        HORIZON,
        num_samples=200,
        num_elites=20,
        num_iterations=5,
        init_std=2.0,
    )

    rs_pos, rs_cost = run(random_shooting)
    cem_pos, cem_cost = run(cem)
    t = list(range(NUM_STEPS))

    fig, (ax_pos, ax_cost) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    for ax in (ax_pos, ax_cost):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Position vs. goal ---
    ax_pos.axhline(
        GOAL, color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1
    )
    ax_pos.text(NUM_STEPS - 1, GOAL, "  goal", color=INK_MUTED, fontsize=9, va="center")
    ax_pos.plot(
        t,
        rs_pos,
        color=COLOR_RANDOM_SHOOTING,
        linewidth=2,
        label="RandomShooting",
        zorder=3,
    )
    ax_pos.plot(t, cem_pos, color=COLOR_CEM, linewidth=2, label="CEM", zorder=3)
    ax_pos.set_title(
        "Position vs. goal", color=INK_PRIMARY, fontsize=11, loc="left", weight="bold"
    )
    ax_pos.set_xlabel("timestep", color=INK_SECONDARY, fontsize=9)
    ax_pos.set_ylabel("position", color=INK_SECONDARY, fontsize=9)

    # --- Planning cost (log scale: costs span >2 orders of magnitude) ---
    ax_cost.plot(
        t,
        rs_cost,
        color=COLOR_RANDOM_SHOOTING,
        linewidth=2,
        label="RandomShooting",
        zorder=3,
    )
    ax_cost.plot(t, cem_cost, color=COLOR_CEM, linewidth=2, label="CEM", zorder=3)
    ax_cost.set_yscale("log")
    ax_cost.set_title(
        "Best planning cost per step",
        color=INK_PRIMARY,
        fontsize=11,
        loc="left",
        weight="bold",
    )
    ax_cost.set_xlabel("timestep", color=INK_SECONDARY, fontsize=9)
    ax_cost.set_ylabel("best_cost (log scale)", color=INK_SECONDARY, fontsize=9)

    handles, labels = ax_pos.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        labelcolor=INK_PRIMARY,
        fontsize=10,
    )
    fig.suptitle(
        "RandomShooting vs. CEM — matched compute budget (1000 rollouts/step)",
        color=INK_PRIMARY,
        fontsize=12,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {OUTPUT_PATH}")

    print(f"\n{'':14s} {'RandomShooting':>16s} {'CEM':>10s}")
    print(
        f"{'final |pos-goal|':14s} {abs(rs_pos[-1] - GOAL):16.4f} {abs(cem_pos[-1] - GOAL):10.4f}"
    )
    print(f"{'final best_cost':14s} {rs_cost[-1]:16.4f} {cem_cost[-1]:10.4f}")
    print(
        f"{'mean best_cost':14s} {sum(rs_cost) / len(rs_cost):16.4f} {sum(cem_cost) / len(cem_cost):10.4f}"
    )


if __name__ == "__main__":
    main()
