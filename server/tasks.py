"""
SpiceRL Tasks — Task specifications with component bounds (no .net templates).

Each task defines:
- A topology (buck, boost, multiphase)
- The parameters the agent can tune
- The target specification
- The grading criteria
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class TaskSpec:
    """Definition of a single SpiceRL task."""

    task_id: str
    name: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    topology: str    # "buck", "boost", "multiphase"

    # Agent action space
    tunable_params: List[str] = field(default_factory=list)
    param_bounds: Dict[str, tuple] = field(default_factory=dict)  # {name: (min, max)}
    default_values: Dict[str, float] = field(default_factory=dict)

    # Target specification
    spec: Dict[str, Any] = field(default_factory=dict)

    # Episode limits
    max_steps: int = 10
    success_threshold: float = 0.85  # Reward above this = early termination


# Tunable parameters common to all Buck Converter tasks
BUCK_PARAMS = [
    "W_hi_um", "W_lo_um",
    "L1_nH", "C1_nF", "fsw_MHz",
    "R_comp", "C_comp_nF", "C_comp2_pF",
    "V_ramp_h_V", "V_ramp_l_V",
    "R_fb_top_kOhm", "R_fb_bot_kOhm"
]

BUCK_REAL_PARAMS = BUCK_PARAMS + [
    "L1_DCR_mOhm", "C1_ESR_mOhm"
]

# Shared parameter boundaries
BUCK_BOUNDS = {
    "W_hi_um": (5000, 100000),
    "W_lo_um": (2500, 50000),
    "L1_nH": (10, 1000),
    "C1_nF": (10, 4700),
    "fsw_MHz": (2, 100),
    "R_comp": (500, 50000),
    "C_comp_nF": (0.1, 50.0),
    "C_comp2_pF": (0.1, 1000.0),
    "V_ramp_h_V": (0.5, 5.0),
    "V_ramp_l_V": (0.0, 1.0),
    "R_fb_top_kOhm": (1.0, 100.0),
    "R_fb_bot_kOhm": (1.0, 100.0),
    "L1_DCR_mOhm": (0.1, 500.0),
    "C1_ESR_mOhm": (0.1, 100.0)
}

BUCK_DEFAULTS = {
    "W_hi_um": 40000,
    "W_lo_um": 20000,
    "L1_nH": 47,
    "C1_nF": 68,
    "fsw_MHz": 33.3,
    "R_comp": 4000,
    "C_comp_nF": 4.0,
    "C_comp2_pF": 1.0,
    "V_ramp_h_V": 1.8,
    "V_ramp_l_V": 0.0,
    "R_fb_top_kOhm": 10.0,
    "R_fb_bot_kOhm": 10.0,
}

BUCK_DEFAULTS_REAL = {**BUCK_DEFAULTS, "L1_DCR_mOhm": 50.0, "C1_ESR_mOhm": 15.0}

BUCK_SPEC = {
    "Vdd": 1.8,
    "Vout_target": 1.2,
    "Iout": 0.3,
    "vout_tolerance": 0.05,
    "ripple_max_mV": 60.0,
    "efficiency_target": 0.85,
    "load_reg_max_pct": 10.0,
    "pm_min_deg": 45.0,
    "gm_min_dB": 6.0,
}


# ============================================================================
# Task 1: Ideal Buck Converter (Easy)
# ============================================================================
TASK_EASY = TaskSpec(
    task_id="easy",
    name="Ideal Buck Converter",
    description=(
        "Design a 1.8V -> 1.2V synchronous buck converter using ideal components "
        "to meet the requested specifications. The simulation automatically uses "
        "negligible parasitics (e.g. 1mOhm DCR/ESR). Tune the transistor sizes, "
        "passives, compensator, and frequency to hit target Vout efficiently."
    ),
    difficulty="easy",
    topology="buck",
    tunable_params=BUCK_PARAMS,
    param_bounds=BUCK_BOUNDS.copy(),
    default_values=BUCK_DEFAULTS.copy(),
    spec=BUCK_SPEC.copy(),
    max_steps=10,
    success_threshold=0.85,
)

# ============================================================================
# Task 2: Real Component Buck Converter (Medium)
# ============================================================================
TASK_MEDIUM = TaskSpec(
    task_id="medium",
    name="Real Component Buck Converter",
    description=(
        "Design the same 1.8V -> 1.2V buck converter but using real components "
        "with parasitics from Coilcraft inductors and Murata capacitors. "
        "You must explicitly select realistic L1_DCR_mOhm for your L1_nH and "
        "realistic C1_ESR_mOhm for your C1_nF. Ensure the circuit regulates "
        "even with these realistic losses."
    ),
    difficulty="medium",
    topology="buck",
    tunable_params=BUCK_REAL_PARAMS,
    param_bounds=BUCK_BOUNDS.copy(),
    default_values=BUCK_DEFAULTS_REAL.copy(),
    spec=BUCK_SPEC.copy(),
    max_steps=10,
    success_threshold=0.80,
)

# ============================================================================
# Task 3: Cost-Optimized Real Buck Converter (Hard)
# ============================================================================
TASK_HARD = TaskSpec(
    task_id="hard",
    name="Cost-Optimized Real Buck Converter",
    description=(
        "Same real-component buck converter as Task 2, but you must ALSO minimize "
        "the cost of the circuit after meeting circuit specifications. Cost is "
        "reduced by choosing smaller inductor (L) and capacitor (C) values "
        "(which degrades ripple/stability), and minimizing the transistor area "
        "(W_hi_um + W_lo_um) which increases conduction losses."
    ),
    difficulty="hard",
    topology="buck",
    tunable_params=BUCK_REAL_PARAMS,
    param_bounds=BUCK_BOUNDS.copy(),
    default_values=BUCK_DEFAULTS_REAL.copy(),
    spec=BUCK_SPEC.copy(),
    max_steps=10,
    success_threshold=0.75,
)

# Task registry
TASKS: Dict[str, TaskSpec] = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard": TASK_HARD,
}
