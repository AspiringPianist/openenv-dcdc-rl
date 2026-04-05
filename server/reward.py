"""
SpiceRL Reward Function.

Computes a normalized [0.0, 1.0] reward score based on how well the
simulated circuit meets the target specification.

All sub-rewards are normalized:
- 1.0 = spec met or exceeded
- 0.0 = spec completely missed
- Intermediate = proportional closeness

Hard penalties for catastrophic failures (sim crash, dead circuit).
"""

from typing import Any, Dict


def compute_reward(
    metrics: Dict[str, Any],
    spec: Dict[str, Any],
    difficulty: str,
) -> float:
    """Compute reward from simulation metrics vs target spec.

    Args:
        metrics: Dict of measured values from simulation
        spec: Dict of target values from task definition
        difficulty: "easy", "medium", or "hard"

    Returns:
        Reward clamped to [0.0, 1.0]
    """

    # --- Check for catastrophic failures first ---
    if metrics.get("sim_error") or not metrics.get("sim_converged", True):
        return 0.0  # Simulation failed entirely

    if metrics.get("vout_avg", 0) < 0.01:
        return 0.01  # Circuit is dead (Vout ≈ 0), tiny nonzero to signal "try again"

    # --- Sub-reward: Voltage regulation ---
    vout_target = spec["Vout_target"]
    vout_avg = metrics.get("vout_avg", 0)
    vout_error = abs(vout_avg - vout_target) / vout_target
    vout_tolerance = spec.get("vout_tolerance", 0.05)
    r_regulation = max(0.0, 1.0 - vout_error / vout_tolerance)

    # --- Sub-reward: Output ripple ---
    ripple_max = spec.get("ripple_max_mV", 100.0) / 1000.0  # Convert to V
    vout_ripple = metrics.get("vout_ripple", 0)
    if ripple_max > 0:
        r_ripple = max(0.0, 1.0 - vout_ripple / ripple_max)
    else:
        r_ripple = 1.0

    # --- Sub-reward: Efficiency ---
    eff_target = spec.get("efficiency_target", 0.80)
    eff = metrics.get("efficiency", 0)
    if eff_target > 0:
        r_efficiency = min(1.0, eff / eff_target)
    else:
        r_efficiency = 1.0

    # --- Sub-reward: Phase margin (Tasks 2, 3 only) ---
    r_stability = 1.0  # Default for easy task
    pm_min = spec.get("pm_min_deg")
    if pm_min is not None:
        pm = metrics.get("phase_margin_deg")
        if pm is not None:
            pm_margin = (pm - pm_min) / pm_min
            r_stability = max(0.0, min(1.0, 0.5 + pm_margin))
        else:
            r_stability = 0.3  # AC sim didn't run, penalize but don't kill

    # --- Sub-reward: Gain margin (Tasks 2, 3) ---
    r_gain = 1.0
    gm_min = spec.get("gm_min_dB")
    if gm_min is not None:
        gm = metrics.get("gain_margin_dB")
        if gm is not None:
            r_gain = max(0.0, min(1.0, gm / gm_min))
        else:
            r_gain = 0.3

    # --- Sub-reward: Transient response (Task 3 only) ---
    r_transient = 1.0
    os_max = spec.get("overshoot_max_pct")
    if os_max is not None:
        os_pct = metrics.get("overshoot_pct")
        if os_pct is not None:
            r_transient = max(0.0, 1.0 - os_pct / os_max)
        else:
            r_transient = 0.5  # No transient data

    r_settling = 1.0
    settle_max = spec.get("settling_max_us")
    if settle_max is not None:
        settle = metrics.get("settling_time_us")
        if settle is not None:
            r_settling = max(0.0, 1.0 - settle / settle_max)
        else:
            r_settling = 0.5

    # --- Sub-reward: Load regulation (all tasks with load step) ---
    r_load_reg = 1.0
    load_reg_max = spec.get("load_reg_max_pct")
    if load_reg_max is not None:
        load_reg = metrics.get("load_regulation")
        if load_reg is not None:
            r_load_reg = max(0.0, 1.0 - load_reg / load_reg_max)
        else:
            r_load_reg = 0.5

    # --- Weighted combination by difficulty ---
    weights = {
        "easy": {
            "reg": 4.0, "rip": 2.0, "eff": 2.0,
            "stab": 0.0, "gain": 0.0, "trans": 0.0, "settle": 0.0,
            "load_reg": 1.0,
        },
        "medium": {
            "reg": 3.0, "rip": 2.0, "eff": 2.0,
            "stab": 3.0, "gain": 1.0, "trans": 0.0, "settle": 0.0,
            "load_reg": 1.5,
        },
        "hard": {
            "reg": 3.0, "rip": 2.0, "eff": 2.0,
            "stab": 2.0, "gain": 1.0, "trans": 1.5, "settle": 1.5,
            "load_reg": 2.0,
        },
    }

    w = weights.get(difficulty, weights["easy"])
    raw = (
        w["reg"] * r_regulation
        + w["rip"] * r_ripple
        + w["eff"] * r_efficiency
        + w["stab"] * r_stability
        + w["gain"] * r_gain
        + w["trans"] * r_transient
        + w["settle"] * r_settling
        + w["load_reg"] * r_load_reg
    )
    total_weight = sum(w.values())

    if total_weight == 0:
        return 0.0

    normalized = raw / total_weight
    return float(max(0.0, min(1.0, normalized)))

