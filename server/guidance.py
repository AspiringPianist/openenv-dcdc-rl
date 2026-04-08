"""SpiceRL guidance helpers.

Provides:
- Step-specific design guidance text derived from metrics/spec trends.
- Buck-converter math toolbox text with hardcoded slide equations.
"""

from typing import Any, Dict, List, Optional


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort conversion to float."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_efficiency(value: Optional[float]) -> Optional[float]:
    """Normalize efficiency to 0..1 when simulator reports percentage."""
    if value is None:
        return None
    return value / 100.0 if value > 1.5 else value


def build_math_toolbox(
    spec: Dict[str, Any],
    param_names: List[str],
    last_action: Optional[Dict[str, float]] = None,
) -> str:
    """Build math toolbox text for the LLM.

    This text is meant to be copied directly into prompts so the LLM can
    compute bandwidth, settling behavior, ripple, and loss estimates.
    """
    fsw_mhz = _to_float((last_action or {}).get("fsw_MHz"))
    if fsw_mhz is None:
        fsw_mhz = _to_float(spec.get("fsw_MHz"))
    vout_target = _to_float(spec.get("Vout_target"))

    lines: List[str] = [
        "- Use ONLY buck-converter equations listed below (hardcoded from extracted slide content).",
        "- To avoid math mistakes, DO NOT calculate values mentally. Write out explicit Python code snippets in your thought process to compute precise values for your next action.",
        "- Calculate the exact L, C, R_comp, and C_comp values required using the formulas below.",
        "- Allowed operators/functions: +, -, *, /, ^, sqrt, abs, min, max, log, log10, ln, exp, sin, cos, tan, atan, pi.",
        "- Ideal buck DC relation: Vout = D*Vin, therefore D = Vout/Vin.",
        "- Small-signal buck plant (duty to output): Gvd(s) = Vin / (s^2*L*C + s*(L/Rload) + 1).",
        "- LC resonance: omega_o = 1/sqrt(L*C), f_o = omega_o/(2*pi).",
        "- Damping/quality: Q = Rload*sqrt(C/L). If Q > 0.5, expect peaking from complex poles.",
        "- With capacitor ESR: Gvd_esr(s) = Vin*(1 + s*ESR*C) / (s^2*L*C + s*(L/Rload) + 1).",
        "- ESR zero location: fz_esr = 1/(2*pi*ESR*C).",
        "- PWM gain: Gpwm = Dmax/(Vmax - Vmin) = Dmax/Vsaw.",
        "- Ripple estimate: Delta_iL = (Vin - Vout)*D/(fsw*L), Delta_vout ~= Delta_iL/(8*fsw*C) + Delta_iL*ESR.",
        "- Efficiency relation: eta = Pout/Pin.",
        "- Type-III compensator or Type-II compensator: C_comp calculates the pole, R_comp calculates the zero.",
        "- Type-II placement rule: place compensator zero f_z = 1/(2*pi*R_comp*C_comp) below LC double-pole f_o and place HF pole f_p = 1/(2*pi*R_comp*C_comp2) near fsw/2.",
        "- Bandwidth rule: start with crossover near fsw/10; if phase margin remains strong, push toward 0.2 to 0.3 of fsw.",
        "- Stability targets: phase margin >= 45 deg (prefer 55 to 70 deg), gain margin >= 6 dB.",
        "- Dominant-pole settling estimate: t_settle ~= 3/(2*pi*fc).",
    ]

    if fsw_mhz is not None:
        lines.append(f"- With fsw={fsw_mhz:.4g} MHz, UGB target ~= {fsw_mhz * 100.0:.4g} kHz.")
    if vout_target is not None:
        lines.append(f"- Regulate around Vout_target={vout_target:.4g} V while satisfying ripple/stability constraints.")

    # Only buck-specific equations are intentionally included.
    if "R_comp" not in param_names and "R_comp_kOhm" not in param_names:
        lines.append("- Compensator equations are provided for math reasoning; tune only parameters that exist in this task.")

    return "\n".join(lines)


def build_design_guidance(
    metrics: Dict[str, Any],
    spec: Dict[str, Any],
    param_names: List[str],
    previous_metrics: Optional[Dict[str, Any]] = None,
    reward: Optional[float] = None,
    previous_reward: Optional[float] = None,
    last_action: Optional[Dict[str, float]] = None,
) -> str:
    """Create concise step-specific guidance from current design status."""
    has_param = set(param_names).__contains__

    lines: List[str] = [
        "- Prioritize regulation, ripple, and stability first; then maximize efficiency.",
        "- Prefer incremental parameter changes (typically 5-20% per step).",
    ]

    vout_target = _to_float(spec.get("Vout_target"))
    vout = _to_float(metrics.get("vout_avg"))
    vout_tol = _to_float(spec.get("vout_tolerance"), 0.05) or 0.05
    if vout_target and vout is not None:
        err = (vout - vout_target) / vout_target
        if err < -vout_tol:
            tips = []
            if has_param("W_hi_um") or has_param("W_main_um"):
                tips.append("increase high-side/main transistor width")
            if has_param("R_comp") or has_param("R_comp_kOhm"):
                tips.append("increase compensator R slightly")
            if has_param("C1_nF"):
                tips.append("increase output C")
            if tips:
                lines.append(f"- Vout is low; try: {', '.join(tips[:3])}.")
        elif err > vout_tol:
            tips = []
            if has_param("W_hi_um") or has_param("W_main_um"):
                tips.append("decrease high-side/main transistor width")
            if has_param("R_comp") or has_param("R_comp_kOhm"):
                tips.append("decrease compensator R slightly")
            if has_param("fsw_MHz"):
                tips.append("increase fsw moderately")
            if tips:
                lines.append(f"- Vout is high; try: {', '.join(tips[:3])}.")

    ripple_limit_mv = _to_float(spec.get("ripple_max_mV"))
    ripple_v = _to_float(metrics.get("vout_ripple"))
    if ripple_limit_mv is not None and ripple_v is not None:
        ripple_limit = ripple_limit_mv / 1000.0
        if ripple_v > ripple_limit:
            tips = []
            if has_param("L1_nH") or has_param("L2_nH"):
                tips.append("increase inductor value")
            if has_param("C1_nF"):
                tips.append("increase output capacitor")
            if has_param("fsw_MHz"):
                tips.append("increase fsw only if efficiency budget allows")
            if tips:
                lines.append(f"- Ripple exceeds spec; try: {', '.join(tips[:3])}.")

    eff = _normalize_efficiency(_to_float(metrics.get("efficiency")))
    eff_target = _to_float(spec.get("efficiency_target"))
    if eff is not None and eff_target and eff < eff_target:
        tips = []
        if has_param("fsw_MHz"):
            fsw_now = _to_float((last_action or {}).get("fsw_MHz"))
            if fsw_now is not None:
                tips.append(
                    f"explore fsw locally near {fsw_now:.4g} MHz (+/-10%)"
                )
            else:
                tips.append("perform local fsw exploration (+/-10%)")
        if has_param("W_hi_um") and has_param("W_lo_um"):
            tips.append("keep W_hi:W_lo roughly near 2:1 unless data contradicts")
        if has_param("L1_nH"):
            tips.append("avoid excessive L that increases DCR loss")
        if tips:
            lines.append(f"- Efficiency is below target; try: {', '.join(tips[:3])}.")

    pm = _to_float(metrics.get("phase_margin_deg"))
    pm_min = _to_float(spec.get("pm_min_deg"))
    if pm is not None and pm_min is not None and pm < pm_min:
        tips = []
        if has_param("R_comp") or has_param("R_comp_kOhm"):
            tips.append("decrease R_comp to lower crossover")
        if has_param("C_comp_nF"):
            tips.append("increase C_comp_nF")
        if has_param("C_comp2_pF"):
            tips.append("increase C_comp2_pF to lower HF pole")
        if tips:
            lines.append(f"- Phase margin is low; try: {', '.join(tips[:3])}.")

    fc_khz = _to_float(metrics.get("crossover_freq_kHz"))
    fsw_mhz = _to_float((last_action or {}).get("fsw_MHz"))
    if fsw_mhz is not None:
        ugb_target_khz = fsw_mhz * 100.0
        if fc_khz is not None:
            if fc_khz < 0.5 * ugb_target_khz:
                lines.append("- Crossover is much lower than fsw/10; increase bandwidth carefully.")
            elif fc_khz > 1.5 * ugb_target_khz:
                lines.append("- Crossover is above fsw/10; reduce bandwidth for stability margin.")
        lines.append(
            f"- Bandwidth anchor: target crossover around {ugb_target_khz:.4g} kHz for fsw={fsw_mhz:.4g} MHz."
        )

    if reward is not None and previous_reward is not None:
        if reward < previous_reward - 0.02:
            lines.append("- Reward dropped; partially revert and use smaller perturbations.")
        elif reward > previous_reward + 0.02:
            lines.append("- Reward improved; continue in the same direction with smaller follow-up changes.")

    if previous_metrics:
        prev_ripple = _to_float(previous_metrics.get("vout_ripple"))
        curr_ripple = _to_float(metrics.get("vout_ripple"))
        if prev_ripple is not None and curr_ripple is not None and curr_ripple > prev_ripple * 1.1:
            lines.append("- Ripple worsened versus previous step; back off the last aggressive change.")

    return "\n".join(lines)
