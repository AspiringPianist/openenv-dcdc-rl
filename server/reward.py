"""
SpiceRL Reward Function (PySpice Native).

Computes a normalized [0.0, 1.0] reward score based on simulated circuit
metrics vs target specification. All sub-rewards are normalized:
- 1.0 = spec met or exceeded
- 0.0 = spec completely missed
- Intermediate = proportional closeness

Hard penalties for catastrophic failures (sim crash, dead circuit).
"""

from typing import Any, Dict
import numpy as np


def compute_physics_losses(
    metrics: Dict[str, Any],
    params: Dict[str, float],
    spec: Dict[str, Any],
) -> Dict[str, float]:
    """Calculate native physics-based losses from component values and waveforms.
    
    Returns:
        Dict with keys like 'p_inductor_dcr', 'p_cap_esr', 'p_switching', etc. (in Watts)
    """
    losses = {
        "p_inductor_dcr": 0.0,
        "p_cap_esr": 0.0,
        "p_switching": 0.0,
        "p_total": 0.0,
    }

    try:
        # --- Inductor DCR loss: I²*R(DCR) ---
        # Typical 180nm switch: PMOS Rds ≈ 10mΩ @ 40µm W, dcr ~50mΩ for 47nH
        il_ripple = metrics.get("il_ripple", 0.1)  # Amps
        il_avg = spec.get("Iout", 0.3)  # Average inductor current ≈ Iout
        il_rms = np.sqrt(il_avg**2 + (il_ripple / np.sqrt(12))**2)  # RMS current
        
        # DCR estimate: ~50mΩ for typical 47nH inductor
        l1_nh = params.get("L1_nH", 47)
        r_dcr = 0.05 * (l1_nh / 47.0)  # Scale with inductance
        losses["p_inductor_dcr"] = float(il_rms**2 * r_dcr)

        # --- Capacitor ESR loss: I²_ripple * R(ESR) ---
        # ESR scales inversely with capacitance (roughly)
        c1_nf = params.get("C1_nF", 68)
        r_esr = 0.02 * (68.0 / c1_nf)  # ~20mΩ base ESR at 68nF
        il_ripple_squared = il_ripple**2 / 12.0  # Mean square of triangle wave
        losses["p_cap_esr"] = float(il_ripple_squared * r_esr)

        # --- Switching loss (frequency-dependent) ---
        # P_sw ≈ C_eff * V² * fsw, where C_eff includes gate charge + parasitic caps
        fsw = params.get("fsw_MHz", 33.3) * 1e6  # Convert to Hz
        vdd = spec.get("Vdd", 1.8)
        
        # Estimate effective switching capacitance from transistor sizes
        w_hi = params.get("W_hi_um", params.get("W_hi1_um", 40000))
        w_lo = params.get("W_lo_um", params.get("W_lo1_um", 20000))
        
        # Larger transistors → more parasitic cap → more switching loss
        c_sw_total = (w_hi + w_lo) * 1e-15  # Very rough: ~1fF per micron width
        losses["p_switching"] = float(c_sw_total * vdd**2 * fsw * 1e-3)  # Rough estimate

        # Total conduction + switching loss
        losses["p_total"] = losses["p_inductor_dcr"] + losses["p_cap_esr"] + losses["p_switching"]
    except Exception:
        pass
        
    return losses


def compute_reward(
    metrics: Dict[str, Any],
    spec: Dict[str, Any],
    difficulty: str,
    params: Dict[str, float] = None,
) -> float:
    if params is None:
        params = {}

    if metrics.get("sim_error") or not metrics.get("sim_converged", True):
        return 0.0

    if metrics.get("vout_avg", 0) < 0.01:
        return 0.01

    # Voltage regulation
    vout_target = spec["Vout_target"]
    vout_avg = metrics.get("vout_avg", 0)
    vout_error = abs(vout_avg - vout_target) / vout_target
    vout_tolerance = spec.get("vout_tolerance", 0.05)
    r_regulation = max(0.0, 1.0 - vout_error / vout_tolerance)

    # Ripple
    ripple_max = spec.get("ripple_max_mV", 100.0) / 1000.0
    vout_ripple = metrics.get("vout_ripple", 0)
    r_ripple = max(0.0, 1.0 - vout_ripple / ripple_max) if ripple_max > 0 else 1.0

    # Efficiency
    eff_target = spec.get("efficiency_target", 0.80)
    eff = metrics.get("efficiency", 0)
    r_efficiency = min(1.0, eff / eff_target) if eff_target > 0 else 1.0

    # Weightings
    weights = {"reg": 4.0, "rip": 2.0, "eff": 2.0}
    
    raw = weights["reg"] * r_regulation + weights["rip"] * r_ripple + weights["eff"] * r_efficiency
    total_weight = sum(weights.values())

    base_reward = float(max(0.0, min(1.0, raw / total_weight)))
    
    base_reward = base_reward ** 0.5

    if difficulty == "hard" and r_regulation > 0.8:
        # Apply cost penalty only if circuit mostly works
        w_total = params.get("W_hi_um", 40000) + params.get("W_lo_um", 20000)
        l1 = params.get("L1_nH", 47)
        c1 = params.get("C1_nF", 68)
        
        # Approximate normalized cost factor [0.0 to ~1.0+]
        cost_score = (w_total / 100000.0) + (l1 / 100.0) + (c1 / 500.0)
        
        # Reduce reward based on cost (up to 30% penalty)
        penalty = max(0.0, min(0.3, cost_score * 0.1))
        return max(0.0, base_reward - penalty)
        
    return base_reward
