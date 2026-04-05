"""
SpiceRL Tasks — Task definitions, specifications, and graders.

Each task defines:
- A topology (template netlist)
- The parameters the agent can tune
- The target specification
- The grading criteria

All tasks use 180nm CMOS process models with integrated power
transistors. The agent tunes transistor W/L ratios, passive component
values from real vendor catalogs, and compensator parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import os

# Resolve template directory relative to this file
_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


@dataclass
class TaskSpec:
    """Definition of a single SpiceRL task."""

    task_id: str
    name: str
    description: str
    difficulty: str  # "easy", "medium", "hard"

    # Netlist templates
    tran_template: str  # Path to transient sim netlist template
    ac_template: str = ""  # Path to AC sim netlist template (empty = no AC)

    # Agent action space
    tunable_params: List[str] = field(default_factory=list)
    param_bounds: Dict[str, tuple] = field(default_factory=dict)  # {name: (min, max)}
    default_values: Dict[str, float] = field(default_factory=dict)

    # Target specification
    spec: Dict[str, Any] = field(default_factory=dict)

    # Episode limits
    max_steps: int = 10
    success_threshold: float = 0.85  # Reward above this = early termination


# ============================================================================
# Task 1: Integrated Buck Converter (Easy)
# ============================================================================
# VLS804 Power Management ICs assignment:
# 1.8V → 1.2V / 300mA, 180nm CMOS, maximize efficiency, L ≤ 50nH
TASK_EASY = TaskSpec(
    task_id="easy",
    name="Miniaturized Buck Converter (180nm CMOS)",
    description=(
        "Design a miniaturized 1.8V → 1.2V synchronous buck converter using "
        "180nm CMOS technology. The converter must operate in CCM at 300mA load. "
        "The inductor value must be ≤50nH to minimize size. "
        "You must tune: power PMOS/NMOS widths (Wp/Wn ratio is a design decision), "
        "inductor and capacitor from vendor catalogs, switching frequency, "
        "and Type II compensator (R4, C2, C3). "
        "Goal: maximize efficiency at 300mA while meeting ripple and "
        "load transient specifications. "
        "Key tradeoffs: PMOS dominates conduction loss (D≈67%), oversizing NMOS "
        "wastes switching loss. Higher fsw → smaller L/C but more switching loss. "
        "Driver sizing affects gate charge losses."
    ),
    difficulty="easy",
    tran_template=os.path.join(_TEMPLATES_DIR, "buck_tran.net"),
    ac_template=os.path.join(_TEMPLATES_DIR, "buck_ac.net"),
    tunable_params=[
        "W_hi_um", "W_lo_um",            # Power transistor sizing
        "L1_nH", "C1_nF", "fsw_MHz",     # Passives + frequency
        "R_comp", "C_comp_nF", "C_comp2_pF",  # Compensator
    ],
    param_bounds={
        "W_hi_um": (5000, 100000),        # PMOS width 5mm–100mm
        "W_lo_um": (2500, 50000),         # NMOS width 2.5mm–50mm
        "L1_nH": (10, 50),               # Max 50nH per assignment spec
        "C1_nF": (10, 470),              # 10nF – 470nF (Würth MLCC range)
        "fsw_MHz": (10, 100),            # 10MHz – 100MHz
        "R_comp": (500, 50000),           # 500Ω – 50kΩ
        "C_comp_nF": (0.1, 50.0),
        "C_comp2_pF": (0.1, 1000.0),
    },
    default_values={
        "W_hi_um": 40000,
        "W_lo_um": 20000,
        "L1_nH": 47,
        "C1_nF": 68,
        "fsw_MHz": 33.3,
        "R_comp": 4000,
        "C_comp_nF": 4.0,
        "C_comp2_pF": 1.0,
    },
    spec={
        "Vdd": 1.8,
        "Vout_target": 1.2,
        "Iout": 0.3,                  # Typical 300mA
        "Iout_min": 0.1,              # 100mA min
        "Iout_max": 0.5,              # 500mA max
        "vout_tolerance": 0.05,       # ±5% steady-state ripple
        "ripple_max_mV": 60.0,        # 5% of 1.2V = 60mV
        "efficiency_target": 0.85,    # Maximize at 300mA
        "load_reg_max_pct": 10.0,     # 10% max load transient Vpp
        "pm_min_deg": 45.0,
        "gm_min_dB": 6.0,
        "L_max_nH": 50,              # Assignment constraint
    },
    max_steps=10,
    success_threshold=0.85,
)

# ============================================================================
# Task 2: Integrated Boost Converter (Medium)
# ============================================================================
# 0.9V → 1.8V / 200mA — battery boost-up for 180nm SoC.
# Agent also tunes compensator for stability (AC analysis).
TASK_MEDIUM = TaskSpec(
    task_id="medium",
    name="Integrated Boost Converter (180nm)",
    description=(
        "Design a 0.9V → 1.8V / 200mA integrated boost converter "
        "using 180nm CMOS. The NMOS main switch charges the inductor, "
        "the PMOS synchronous rectifier transfers energy to the output. "
        "Tune transistor widths, passives, compensator, and frequency. "
        "Requires both transient regulation AND AC stability analysis "
        "(phase margin > 45°, gain margin > 6dB)."
    ),
    difficulty="medium",
    tran_template=os.path.join(_TEMPLATES_DIR, "boost_tran.net"),
    ac_template=os.path.join(_TEMPLATES_DIR, "boost_ac.net"),
    tunable_params=[
        "W_main_um", "W_rect_um",          # Main NMOS + Rectifier PMOS
        "L1_nH", "C1_nF", "fsw_MHz",
        "R_comp_kOhm", "C_comp_nF", "C_comp2_pF",
    ],
    param_bounds={
        "W_main_um": (5000, 80000),
        "W_rect_um": (5000, 100000),
        "L1_nH": (22, 1000),              # 22nH – 1µH
        "C1_nF": (22, 1000),
        "fsw_MHz": (5, 50),
        "R_comp_kOhm": (1.0, 100.0),
        "C_comp_nF": (0.1, 100.0),
        "C_comp2_pF": (10.0, 10000.0),
    },
    default_values={
        "W_main_um": 30000,
        "W_rect_um": 40000,
        "L1_nH": 100,
        "C1_nF": 100,
        "fsw_MHz": 20,
        "R_comp_kOhm": 10.0,
        "C_comp_nF": 2.2,
        "C_comp2_pF": 100.0,
    },
    spec={
        "Vin": 0.9,
        "Vout_target": 1.8,
        "Iout": 0.2,
        "vout_tolerance": 0.04,       # ±4%
        "ripple_max_mV": 50.0,
        "efficiency_target": 0.70,
        "pm_min_deg": 45.0,
        "gm_min_dB": 6.0,
        "load_reg_max_pct": 4.0,
    },
    max_steps=15,
    success_threshold=0.80,
)

# ============================================================================
# Task 3: Multiphase Buck VRM (Hard)
# ============================================================================
# 1.8V → 0.9V / 1A — 2-phase interleaved buck for core voltage rail.
# Agent sizes 4 power FETs + all compensator + droop resistor.
TASK_HARD = TaskSpec(
    task_id="hard",
    name="Multiphase Buck VRM (180nm)",
    description=(
        "Design a 1.8V → 0.9V / 1A 2-phase interleaved synchronous buck "
        "converter using 180nm CMOS for a high-current core voltage rail. "
        "Tune 4 power transistor widths (2 per phase), dual inductors, "
        "output capacitor, compensator, droop resistor, and frequency. "
        "Must meet tight regulation (±1%), ripple (<10mV), efficiency "
        "(>80%), stability (PM>50°), transient response, and load regulation."
    ),
    difficulty="hard",
    tran_template=os.path.join(_TEMPLATES_DIR, "multiphase_buck_tran.net"),
    ac_template=os.path.join(_TEMPLATES_DIR, "multiphase_buck_ac.net"),
    tunable_params=[
        "W_hi1_um", "W_lo1_um",             # Phase 1 FETs
        "W_hi2_um", "W_lo2_um",             # Phase 2 FETs
        "L1_nH", "L2_nH", "C1_nF",         # Passives
        "fsw_MHz",
        "R_comp_kOhm", "C_comp_nF", "C_comp2_pF",  # Compensator
        "R_droop_mOhm",                     # AVP droop
    ],
    param_bounds={
        "W_hi1_um": (10000, 200000),
        "W_lo1_um": (5000, 100000),
        "W_hi2_um": (10000, 200000),
        "W_lo2_um": (5000, 100000),
        "L1_nH": (4.7, 100),
        "L2_nH": (4.7, 100),
        "C1_nF": (100, 4700),             # Up to 4.7µF
        "fsw_MHz": (10, 100),
        "R_comp_kOhm": (0.5, 50.0),
        "C_comp_nF": (0.1, 50.0),
        "C_comp2_pF": (10.0, 5000.0),
        "R_droop_mOhm": (0.1, 10.0),
    },
    default_values={
        "W_hi1_um": 60000,
        "W_lo1_um": 30000,
        "W_hi2_um": 60000,
        "W_lo2_um": 30000,
        "L1_nH": 22,
        "L2_nH": 22,
        "C1_nF": 470,
        "fsw_MHz": 33.3,
        "R_comp_kOhm": 5.0,
        "C_comp_nF": 1.0,
        "C_comp2_pF": 220.0,
        "R_droop_mOhm": 1.0,
    },
    spec={
        "Vdd": 1.8,
        "Vout_target": 0.9,
        "Iout": 1.0,
        "vout_tolerance": 0.01,       # ±1%
        "ripple_max_mV": 10.0,
        "efficiency_target": 0.80,
        "pm_min_deg": 50.0,
        "gm_min_dB": 6.0,
        "overshoot_max_pct": 5.0,
        "settling_max_us": 5.0,
        "load_reg_max_pct": 1.0,
    },
    max_steps=20,
    success_threshold=0.75,
)

# Registry
TASKS: Dict[str, TaskSpec] = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard": TASK_HARD,
}
