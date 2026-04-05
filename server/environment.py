"""
SpiceRL Environment — Core environment logic.

Implements the OpenEnv Environment interface: reset(), step(), state.
"""

import logging
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.types import (
    Action,
    Observation,
    State,
)

from server.models import SpiceRLAction, SpiceRLObservation, SpiceRLState
from server.reward import compute_reward
from server.simulator import SpiceSimulator
from server.tasks import TASKS, TaskSpec

logger = logging.getLogger(__name__)


class SpiceRLEnvironment:
    """OpenEnv-compatible environment for DC-DC converter design.

    Each episode:
    1. Agent receives task spec + initial observation (default component values)
    2. Agent proposes component values (action)
    3. Environment runs SPICE sim → extracts metrics → computes reward
    4. Repeat until max_steps or success threshold
    """

    def __init__(self):
        self._simulator = SpiceSimulator()
        self._state = SpiceRLState(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[TaskSpec] = None
        self._done = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a new episode.

        Pass task_id in kwargs to select the task. Default: "easy".
        """
        task_id = kwargs.get("task_id", "easy")
        if task_id not in TASKS:
            task_id = "easy"

        self._task = TASKS[task_id]
        self._done = False
        self._state = SpiceRLState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            cumulative_reward=0.0,
            best_reward=0.0,
            done=False,
        )

        # Run initial simulation with default values to give the agent a baseline
        initial_metrics = self._run_with_params(self._task.default_values)

        return self._make_observation(initial_metrics, reward=0.0)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute one design iteration.

        The action contains component values to set in the netlist.
        """
        if self._done:
            return self._make_observation(
                {}, reward=0.0, error="Episode is done. Call reset()."
            )

        if self._task is None:
            return self._make_observation(
                {}, reward=0.0, error="No task set. Call reset() first."
            )

        self._state.step_count += 1

        # Parse action
        if isinstance(action, SpiceRLAction):
            component_values = action.component_values
        elif hasattr(action, "metadata") and "component_values" in (action.metadata or {}):
            component_values = action.metadata["component_values"]
        else:
            # Try to extract from the action dict generically
            try:
                action_dict = action.model_dump()
                component_values = action_dict.get(
                    "component_values", action_dict.get("metadata", {})
                )
            except Exception:
                component_values = self._task.default_values

        # Validate and clamp parameters
        clamped, warnings = self._simulator.validate_params(
            component_values, self._task.param_bounds
        )
        if warnings:
            logger.info(f"Parameter warnings: {warnings}")

        self._state.current_action = clamped

        # Run simulation
        metrics = self._run_with_params(clamped)

        # Compute reward
        reward = compute_reward(metrics, self._task.spec, self._task.difficulty)
        self._state.cumulative_reward += reward
        self._state.best_reward = max(self._state.best_reward, reward)

        # Check termination
        self._done = (
            self._state.step_count >= self._task.max_steps
            or reward >= self._task.success_threshold
        )
        self._state.done = self._done

        return self._make_observation(metrics, reward=reward)

    @property
    def state(self) -> State:
        """Return the current environment state."""
        return self._state

    def _run_with_params(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Run simulation with given parameters and return metrics."""
        if self._task is None:
            return {"sim_error": "No task configured"}

        try:
            metrics = self._simulator.run_simulation(
                template_path=self._task.tran_template,
                params=params,
                run_name=f"step_{self._state.step_count}",
                ac_template_path=self._task.ac_template,
            )
            return metrics
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return {
                "sim_error": str(e),
                "sim_converged": False,
            }

    def _make_observation(
        self,
        metrics: Dict[str, Any],
        reward: float = 0.0,
        error: Optional[str] = None,
    ) -> Observation:
        """Convert simulation metrics into a typed Observation."""
        task = self._task

        return SpiceRLObservation(
            done=self._done,
            reward=reward,
            # Transient results (keys match .MEAS names, lowercased by log parser)
            vout_avg=metrics.get("vout_avg", 0.0),
            vout_ripple=metrics.get("vout_ripple", 0.0),
            il_ripple=metrics.get("il_ripple", 0.0),
            efficiency=metrics.get("efficiency", 0.0),
            # AC results
            phase_margin_deg=metrics.get("phase_margin_deg"),
            gain_margin_dB=metrics.get("gain_margin_db"),
            crossover_freq_kHz=metrics.get("crossover_freq_khz"),
            # Transient response (load step test)
            overshoot_pct=metrics.get("overshoot_pct"),
            undershoot_pct=metrics.get("undershoot_pct"),
            settling_time_us=metrics.get("settling_time_us"),
            load_regulation_pct=metrics.get("load_regulation"),
            # Loss breakdown (matches .MEAS names: P_Ind_DCR, P_C_ESR)
            p_inductor_dcr=metrics.get("p_ind_dcr"),
            p_cap_esr=metrics.get("p_c_esr"),
            # Diagnostics
            sim_error=error or metrics.get("sim_error"),
            sim_converged=metrics.get("sim_converged", True),
            # Context
            task_id=task.task_id if task else "",
            step_number=self._state.step_count,
            max_steps=task.max_steps if task else 10,
            spec=task.spec if task else {},
        )
