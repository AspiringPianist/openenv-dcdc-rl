"""
SpiceRL Models — Pydantic models for Action, Observation, State.

These models define the typed interface between the LLM agent and
the SpiceRL environment. They inherit from OpenEnv's base types.
"""

from typing import Any, Dict, Optional

from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)
from pydantic import Field


class SpiceRLAction(BaseAction):
    """Agent sends component parameter values as a dict.

    Keys are parameter names matching the netlist .PARAM directives.
    Values are floats in the units specified by the parameter name suffix.

    Examples:
        Easy task:   {"L1_uH": 22.0, "Cout_uF": 100.0, "fsw_kHz": 500.0}
        Medium task: {"L1_uH": 47.0, "Cout_uF": 220.0, "fsw_kHz": 300.0,
                      "R_comp_kOhm": 10.0, "C_comp_nF": 2.2, "C_comp2_pF": 100.0}
    """

    component_values: Dict[str, float] = Field(
        ..., description="Component parameter values to set in the netlist"
    )


class SpiceRLObservation(BaseObservation):
    """Observation returned after each simulation step.

    Contains measured circuit performance metrics extracted from
    the SPICE simulation .MEAS results and waveform analysis.
    """

    # ---- Transient analysis results ----
    vout_avg: float = Field(
        default=0.0, description="Average output voltage (V)"
    )
    vout_ripple: float = Field(
        default=0.0, description="Output voltage peak-to-peak ripple (V)"
    )
    il_ripple: float = Field(
        default=0.0, description="Inductor current peak-to-peak ripple (A)"
    )
    efficiency: float = Field(
        default=0.0, description="Power conversion efficiency (0-1)"
    )

    # ---- AC analysis results (Tasks 2, 3) ----
    phase_margin_deg: Optional[float] = Field(
        default=None, description="Phase margin in degrees (from AC sim)"
    )
    gain_margin_dB: Optional[float] = Field(
        default=None, description="Gain margin in dB (from AC sim)"
    )
    crossover_freq_kHz: Optional[float] = Field(
        default=None, description="Unity-gain crossover frequency in kHz"
    )

    # ---- Transient response (load step test) ----
    overshoot_pct: Optional[float] = Field(
        default=None, description="Load step overshoot percentage"
    )
    undershoot_pct: Optional[float] = Field(
        default=None, description="Load step undershoot percentage"
    )
    settling_time_us: Optional[float] = Field(
        default=None, description="Settling time after load step in µs"
    )
    load_regulation_pct: Optional[float] = Field(
        default=None, description="Load regulation: Vout change under load step (%)"
    )

    # ---- Loss breakdown ----
    p_inductor_dcr: Optional[float] = Field(
        default=None, description="Inductor DCR loss (W)"
    )
    p_cap_esr: Optional[float] = Field(
        default=None, description="Capacitor ESR loss (W)"
    )

    # ---- Diagnostics ----
    sim_error: Optional[str] = Field(
        default=None, description="Error message if simulation failed"
    )
    sim_converged: bool = Field(
        default=True, description="Whether the simulation converged"
    )

    # ---- Task context (so the agent knows what it's optimizing) ----
    task_id: str = Field(
        default="", description="Current task identifier"
    )
    step_number: int = Field(
        default=0, description="Current step in the episode"
    )
    max_steps: int = Field(
        default=10, description="Maximum steps for this task"
    )
    spec: Dict[str, Any] = Field(
        default_factory=dict, description="Target specification for this task"
    )

    # ---- Environment-provided LLM coaching ----
    design_guidance: Optional[str] = Field(
        default=None,
        description="Step-specific design guidance synthesized from metrics and specs",
    )
    math_toolbox: Optional[str] = Field(
        default=None,
        description="Allowed equations/functions for controller and power-stage calculations",
    )


class SpiceRLState(BaseState):
    """Full internal environment state.

    Exposed via the state() / GET /state endpoint.
    """

    task_id: str = Field(default="", description="Current task ID")
    current_action: Optional[Dict[str, float]] = Field(
        default=None, description="Last action component values"
    )
    cumulative_reward: float = Field(
        default=0.0, description="Sum of rewards so far"
    )
    best_reward: float = Field(
        default=0.0, description="Best single-step reward achieved"
    )
    done: bool = Field(default=False, description="Episode finished")
