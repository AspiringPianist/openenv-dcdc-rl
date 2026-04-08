"""
SpiceRL Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from server.models import SpiceRLAction
from client import SpiceRLEnv

# ---- Configuration ----
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_NAME = os.getenv("SPICE_RL_TASK", "easy")
BENCHMARK = os.getenv("SPICE_RL_BENCHMARK", "spice_rl")
MAX_STEPS = 10
TEMPERATURE = 0.4
MAX_TOKENS = 500


# ---- Logging helpers (strict stdout format) ----

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---- System prompt for the LLM agent ----

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert integrated circuit designer specializing in
    on-chip DC-DC switch-mode power converters using 180nm CMOS.
    You are iteratively tuning transistor sizes and component values
    to maximize efficiency while meeting specifications.

    IMPORTANT RULES:
    1. You MUST respond with ONLY a valid JSON object containing parameter values.
    2. No explanations, no markdown, no text before or after the JSON.
    3. All values must be numbers (floats).
    4. Parameter name suffixes indicate units:
       - _um = micrometers (transistor width)
       - _nH = nanohenries (inductor)
       - _nF = nanofarads (output capacitor)
       - _MHz = megahertz (switching frequency)
       - R_comp = ohms (compensator resistor, NOT kOhm)
       - _pF = picofarads (compensator capacitor)
     5. You are explicitly allowed to use internal math from the provided
         MATH TOOLBOX and course slide formulas to compute controller bandwidth,
         response time, ripple, and device sizing.
     6. Do not output intermediate math; output only final JSON.

    DESIGN KNOWLEDGE (180nm CMOS Buck Converter):

    POWER STAGE SIZING:
    - PMOS (high-side): Ron ≈ 6000 Ω·µm / W. At W=40mm, Ron ≈ 0.15Ω.
    - NMOS (low-side): Ron ≈ 7400 Ω·µm / W. At W=20mm, Ron ≈ 0.37Ω.
    - Conduction loss = I²·Ron·D (PMOS) or I²·Ron·(1-D) (NMOS).
    - For Vin=1.8V, Vout=1.2V: duty cycle D ≈ 0.67 → PMOS carries 67% of conduction loss.
    - DO NOT oversize NMOS — it increases gate charge (switching loss) without proportional
      conduction benefit since (1-D) ≈ 0.33.
    - Wp/Wn ratio of 2:1 is a reasonable starting point.

    SWITCHING FREQUENCY (fsw):
    - Higher fsw → smaller L and C needed → smaller converter size.
    - Higher fsw → more switching loss (Psw ∝ fsw * Cgg * Vdd²).
    - Typical optimal range: 20-50 MHz for nH inductors.
    - Total loss = P_cond + P_sw + P_gate_drive + P_dcr + P_esr.
    - There is an OPTIMAL fsw where total loss is minimized.

    PASSIVE COMPONENTS:
    - Inductor: ≤50nH constraint. Coilcraft 0805HP series.
      DCR scales as ~0.093Ω at 47nH. Smaller L → more ripple current.
    - Capacitor: Würth WCAP-CSGP MLCC.
      ESR ~0.017Ω for 68nF. Larger C → less voltage ripple.
    - ΔiL = (Vin - Vout) * D / (fsw * L). Must keep ΔiL < 2*Iout for CCM.
    - Vripple ≈ ΔiL / (8 * fsw * Cout) + ΔiL * ESR.

    COMPENSATOR (Type II):
        - This circuit implements ONLY Type-II compensation (R_comp, C_comp_nF, C_comp2_pF).
            Do not assume Type-I/Type-III extra components exist.
    - R_comp sets crossover frequency fc: higher R → higher fc → faster response.
    - C_comp sets the zero: fz = 1/(2π·R_comp·C_comp). Place below fc.
    - C_comp2 sets the HF pole: fp = 1/(2π·R_comp·C_comp2). Place at fsw/2.
    - Phase margin > 45° required for stability.

    Example response:
    {"W_hi_um": 40000, "W_lo_um": 20000, "L1_nH": 47, "C1_nF": 68, "fsw_MHz": 33.3, "R_comp": 4000, "C_comp_nF": 4.0, "C_comp2_pF": 1.0}
""")


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort float conversion helper for robust prompt logic."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_efficiency(value: Optional[float]) -> Optional[float]:
    """Normalize efficiency to 0..1 when simulator reports percentages."""
    if value is None:
        return None
    if value > 1.5:
        return value / 100.0
    return value


def build_domain_guidance(
    step: int,
    obs_data: Dict[str, Any],
    spec: Dict[str, Any],
    param_names: List[str],
    previous_obs: Optional[Dict[str, Any]] = None,
    last_action: Optional[Dict[str, float]] = None,
) -> str:
    """Create actionable design guidance text for the LLM at each step."""
    del step  # Reserved for future scheduling rules.

    has_param = set(param_names).__contains__
    guidance: List[str] = [
        "- Prioritize Vout regulation, ripple, and stability first; maximize efficiency after constraints are near target.",
        "- Use moderate updates (5-20%) and avoid changing too many parameters at once.",
    ]

    vout_target = _to_float(spec.get("Vout_target"))
    vout = _to_float(obs_data.get("vout_avg"))
    vout_tol = _to_float(spec.get("vout_tolerance"), 0.05) or 0.05

    if vout_target and vout is not None:
        vout_err = (vout - vout_target) / vout_target
        if vout_err < -vout_tol:
            moves: List[str] = []
            if has_param("W_hi_um") or has_param("W_main_um"):
                moves.append("increase high-side/main transistor width by about 10-20%")
            if has_param("R_comp") or has_param("R_comp_kOhm"):
                moves.append("increase compensator R slightly to raise loop crossover")
            if has_param("C1_nF"):
                moves.append("increase output capacitance")
            if has_param("fsw_MHz"):
                moves.append("decrease fsw slightly only if efficiency is poor")
            if moves:
                guidance.append(f"- Vout is below target; try: {', '.join(moves[:3])}.")
        elif vout_err > vout_tol:
            moves = []
            if has_param("W_hi_um") or has_param("W_main_um"):
                moves.append("decrease high-side/main transistor width by about 5-15%")
            if has_param("R_comp") or has_param("R_comp_kOhm"):
                moves.append("reduce compensator R slightly")
            if has_param("fsw_MHz"):
                moves.append("increase fsw modestly")
            if moves:
                guidance.append(f"- Vout is above target; try: {', '.join(moves[:3])}.")

    ripple_limit = _to_float(spec.get("ripple_max_mV"))
    ripple_v = _to_float(obs_data.get("vout_ripple"))
    if ripple_limit is not None:
        ripple_limit = ripple_limit / 1000.0
    if ripple_limit and ripple_v is not None and ripple_v > ripple_limit:
        moves = []
        if has_param("L1_nH") or has_param("L2_nH"):
            moves.append("increase inductor value(s) to reduce inductor ripple")
        if has_param("C1_nF"):
            moves.append("increase output capacitor")
        if has_param("fsw_MHz"):
            moves.append("increase fsw only if efficiency budget allows")
        if moves:
            guidance.append(f"- Output ripple is above spec; try: {', '.join(moves[:3])}.")

    eff_raw = _to_float(obs_data.get("efficiency"))
    eff = _normalize_efficiency(eff_raw)
    eff_target = _to_float(spec.get("efficiency_target"))
    if eff_raw is not None and eff_raw > 1.5:
        guidance.append("- Efficiency appears to be reported in percent; compare against target using normalized value (eff/100).")
    if eff_target and eff is not None and eff < eff_target:
        moves = []
        if has_param("fsw_MHz"):
            fsw_now = _to_float((last_action or {}).get("fsw_MHz"))
            if fsw_now:
                moves.append(
                    f"run a local fsw search around current value ({0.9 * fsw_now:.3g}, {fsw_now:.3g}, {1.1 * fsw_now:.3g} MHz across loops)"
                )
            else:
                moves.append("perform local fsw exploration (+/-10%) to locate best efficiency point")
        if has_param("W_hi_um") and has_param("W_lo_um"):
            moves.append("for buck, keep W_hi_um:W_lo_um near ~2:1 unless data strongly suggests otherwise")
        if has_param("L1_nH"):
            moves.append("avoid oversized inductance that raises DCR loss")
        if moves:
            guidance.append(f"- Efficiency is below target; try: {', '.join(moves[:3])}.")

    pm = _to_float(obs_data.get("phase_margin_deg"))
    pm_min = _to_float(spec.get("pm_min_deg"))
    if pm_min and pm is not None and pm < pm_min:
        moves = []
        if has_param("R_comp") or has_param("R_comp_kOhm"):
            moves.append("decrease compensator R to reduce crossover")
        if has_param("C_comp_nF"):
            moves.append("increase C_comp_nF to add low-frequency gain")
        if has_param("C_comp2_pF"):
            moves.append("increase C_comp2_pF to push high-frequency pole lower")
        if moves:
            guidance.append(f"- Phase margin is low; try: {', '.join(moves[:3])}.")

    load_reg = _to_float(obs_data.get("load_regulation_pct"))
    load_reg_max = _to_float(spec.get("load_reg_max_pct"))
    if load_reg is not None and load_reg_max and load_reg > load_reg_max:
        guidance.append("- Load regulation is weak; if phase margin is healthy, increase loop bandwidth gradually.")

    if previous_obs:
        curr_reward = _to_float(obs_data.get("reward"))
        prev_reward = _to_float(previous_obs.get("reward"))
        if curr_reward is not None and prev_reward is not None:
            if curr_reward < (prev_reward - 0.02):
                guidance.append("- Reward dropped vs previous step; partially revert and use smaller changes (<=10%) next step.")
            elif curr_reward > (prev_reward + 0.02):
                guidance.append("- Reward improved vs previous step; continue in the same direction with smaller follow-up changes.")

    if len(guidance) <= 2 and has_param("fsw_MHz"):
        guidance.append("- No dominant violation detected; perform a small fsw perturbation (+/-10%) and keep passives near current best.")

    return "\n".join(guidance)


def build_user_prompt(
    step: int,
    obs_data: Dict[str, Any],
    spec: Dict[str, Any],
    param_names: List[str],
    param_bounds: Dict[str, tuple],
    history: List[str],
    previous_obs: Optional[Dict[str, Any]] = None,
    last_action: Optional[Dict[str, float]] = None,
) -> str:
    """Build a detailed user prompt with observation data and history."""

    history_block = "\n".join(history[-4:]) if history else "None"
    last_action_block = json.dumps(last_action, indent=2) if last_action else "None"
    env_guidance = obs_data.get("design_guidance")
    env_math = obs_data.get("math_toolbox")
    guidance_block = build_domain_guidance(
        step=step,
        obs_data=obs_data,
        spec=spec,
        param_names=param_names,
        previous_obs=previous_obs,
        last_action=last_action,
    )

    if env_guidance:
        guidance_block = f"ENV GUIDANCE:\n{env_guidance}\n\nLOCAL CROSS-CHECK GUIDANCE:\n{guidance_block}"

    if not env_math:
        env_math = (
            "Use equations/functions from slides and control theory as needed. "
            "Allowed functions include sqrt, log, ln, exp, trigonometric functions, "
            "and standard algebra. Keep calculations internal and output only JSON."
        )

    return textwrap.dedent(f"""\
        STEP {step} — Tune the following parameters to meet the specification.

        TARGET SPECIFICATION:
        {json.dumps(spec, indent=2)}

        CURRENT SIMULATION RESULTS:
        - Vout average: {obs_data.get('vout_avg', 'N/A')} V (target: {spec.get('Vout_target', 'N/A')} V)
        - Vout ripple: {obs_data.get('vout_ripple', 'N/A')} V (max: {spec.get('ripple_max_mV', 'N/A')} mV)
        - Inductor ripple: {obs_data.get('il_ripple', 'N/A')} A
        - Efficiency: {obs_data.get('efficiency', 'N/A')}  (target: {spec.get('efficiency_target', 'N/A')})
        - Load regulation: {obs_data.get('load_regulation_pct', 'N/A')}% (max: {spec.get('load_reg_max_pct', 'N/A')}%)
        - Phase margin: {obs_data.get('phase_margin_deg', 'N/A')} deg
        - Overshoot: {obs_data.get('overshoot_pct', 'N/A')}%
        - Reward: {obs_data.get('reward', 'N/A')}

        TUNABLE PARAMETERS AND BOUNDS:
        {json.dumps({k: list(param_bounds[k]) for k in param_names if k in param_bounds}, indent=2)}

        LAST ACTION APPLIED:
        {last_action_block}

        MATH TOOLBOX (ALLOWED FOR THIS STEP):
        {env_math}

        DOMAIN KNOWLEDGE GUIDANCE FOR THIS STEP:
        {guidance_block}

        PREVIOUS STEPS:
        {history_block}

        Respond with ONLY a JSON object mapping parameter names to values.
    """)


def parse_model_response(text: str, param_names: List[str], defaults: Dict[str, float]) -> Dict[str, float]:
    """Parse the model's JSON response into component values."""
    # Strip any markdown formatting
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Only keep keys that are valid parameters
            result = {}
            for key in param_names:
                if key in data:
                    result[key] = float(data[key])
                elif key in defaults:
                    result[key] = defaults[key]
            return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[DEBUG] JSON parse failed: {e}, text={text[:200]}", file=sys.stderr, flush=True)

    return defaults.copy()


def get_model_action(
    client: OpenAI,
    task_name: str,
    step: int,
    obs_data: Dict[str, Any],
    spec: Dict[str, Any],
    param_names: List[str],
    param_bounds: Dict[str, tuple],
    defaults: Dict[str, float],
    history: List[str],
    previous_obs: Optional[Dict[str, Any]] = None,
    last_action: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Query the LLM for component values."""
    user_prompt = build_user_prompt(
        step,
        obs_data,
        spec,
        param_names,
        param_bounds,
        history,
        previous_obs=previous_obs,
        last_action=last_action,
    )

    # Make the prompt contextually independent and specific to this task
    task_prompt = SYSTEM_PROMPT + f"\n    YOU ARE CURRENTLY WORKING ON TASK: {task_name.upper()}\n    "
    if task_name == "easy":
        task_prompt += "Focus purely on ideal sizing and continuous feedback."
    elif task_name == "medium":
        task_prompt += "The environment will snap your L and C values to real scraping catalogs. Focus strictly on stability under these real world ESR and DCR parasitics."
    elif task_name == "hard":
        task_prompt += "The environment uses real components but severely penalizes larger footprint components (W_hi, W_lo, large L/C values). You must strictly balance efficiency and regulation AGAINST cost metrics and footprint size."

    # Use Pydantic JSON enforcement format and implement retry loop per the user request
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": task_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            
            # Verify basic parsing before returning
            result = parse_model_response(text, param_names, defaults)
            if result and result.keys():
                return result
            else:
                user_prompt += f"\n\nERROR on Attempt {attempt+1}: Ensure response is ONLY valid JSON."
                
        except Exception as exc:
            print(f"[DEBUG] Model request failed on attempt {attempt+1}: {exc}", file=sys.stderr, flush=True)
            
    return defaults.copy()


async def run_task(client: OpenAI, env: SpiceRLEnv, task_name: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Deterministic seeds for consistency
    seeds = {"easy": 101, "medium": 202, "hard": 303}
    seed = seeds.get(task_name, 101)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment with specific task and seed
        result = await env.reset(task_id=task_name, seed=seed)
        obs = result.observation
        obs_data = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
        obs_data["reward"] = 0.0
        spec = obs_data.get("spec", {})

        # Get task info from the observation
        from server.tasks import TASKS
        task = TASKS.get(task_name)
        if task is None:
            task = TASKS["easy"]

        param_names = task.tunable_params
        param_bounds = task.param_bounds
        defaults = task.default_values
        previous_obs_data: Optional[Dict[str, Any]] = None
        last_action: Optional[Dict[str, float]] = None
        
        done = False

        for step in range(1, MAX_STEPS + 1):
            obs_before_step = dict(obs_data)

            # Ask the LLM for component values
            component_values = get_model_action(
                client, task_name, step, obs_data, spec,
                param_names, param_bounds, defaults, history,
                previous_obs=previous_obs_data,
                last_action=last_action,
            )

            # Execute action
            action = SpiceRLAction(component_values=component_values)
            result = await env.step(action)
            obs = result.observation
            obs_data = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

            reward = result.reward or 0.0
            done = result.done


            if step == MAX_STEPS:
                done = True

            error = obs_data.get("sim_error")
            obs_data["reward"] = reward

            rewards.append(reward)
            steps_taken = step

            # Format action for logging (compact JSON)
            action_str = json.dumps(component_values, separators=(",", ":"))

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            # Record for history
            history.append(
                f"Step {step}: {action_str} → reward={reward:.3f}, "
                f"Vout={obs_data.get('vout_avg', '?')}V, "
                f"eff={obs_data.get('efficiency', '?')}, "
                f"ripple={obs_data.get('vout_ripple', '?')}V"
            )
            previous_obs_data = obs_before_step
            last_action = component_values

            if done:
                break

        # Score = best reward achieved (clamped to [0, 1])
        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = True

    except Exception as e:
        print(f"[DEBUG] Error during task {task_name}: {e}", file=sys.stderr, flush=True)
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Connect to environment
    if LOCAL_IMAGE_NAME:
        env = await SpiceRLEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = SpiceRLEnv(base_url=os.getenv("ENV_URL", "http://localhost:8000"))

    try:
        # Evaluate across all task levels
        for task in ["easy", "medium", "hard"]:
            await run_task(client, env, task)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
