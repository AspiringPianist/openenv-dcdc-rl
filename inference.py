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
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from server.models import SpiceRLAction
from client import SpiceRLEnv

# ---- Configuration ----
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

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
    - R_comp sets crossover frequency fc: higher R → higher fc → faster response.
    - C_comp sets the zero: fz = 1/(2π·R_comp·C_comp). Place below fc.
    - C_comp2 sets the HF pole: fp = 1/(2π·R_comp·C_comp2). Place at fsw/2.
    - Phase margin > 45° required for stability.

    Example response:
    {"W_hi_um": 40000, "W_lo_um": 20000, "L1_nH": 47, "C1_nF": 68, "fsw_MHz": 33.3, "R_comp": 4000, "C_comp_nF": 4.0, "C_comp2_pF": 1.0}
""")


def build_user_prompt(
    step: int,
    obs_data: Dict[str, Any],
    spec: Dict[str, Any],
    param_names: List[str],
    param_bounds: Dict[str, tuple],
    history: List[str],
) -> str:
    """Build a detailed user prompt with observation data and history."""

    history_block = "\n".join(history[-4:]) if history else "None"

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
        print(f"[DEBUG] JSON parse failed: {e}, text={text[:200]}", flush=True)

    return defaults.copy()


def get_model_action(
    client: OpenAI,
    step: int,
    obs_data: Dict[str, Any],
    spec: Dict[str, Any],
    param_names: List[str],
    param_bounds: Dict[str, tuple],
    defaults: Dict[str, float],
    history: List[str],
) -> Dict[str, float]:
    """Query the LLM for component values."""
    user_prompt = build_user_prompt(step, obs_data, spec, param_names, param_bounds, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_model_response(text, param_names, defaults)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return defaults.copy()


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment
    if IMAGE_NAME:
        env = await SpiceRLEnv.from_docker_image(IMAGE_NAME)
    else:
        env = SpiceRLEnv(base_url=os.getenv("ENV_URL", "http://localhost:8000"))

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        result = await env.reset(task_id=TASK_NAME)
        obs = result.observation
        obs_data = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
        spec = obs_data.get("spec", {})

        # Get task info from the observation
        from server.tasks import TASKS
        task = TASKS.get(TASK_NAME)
        if task is None:
            task = TASKS["easy"]

        param_names = task.tunable_params
        param_bounds = task.param_bounds
        defaults = task.default_values

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Ask the LLM for component values
            component_values = get_model_action(
                client, step, obs_data, spec,
                param_names, param_bounds, defaults, history,
            )

            # Execute action
            action = SpiceRLAction(component_values=component_values)
            result = await env.step(action)
            obs = result.observation
            obs_data = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

            reward = result.reward or 0.0
            done = result.done
            error = obs_data.get("sim_error")

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

            if done:
                break

        # Score = best reward achieved (clamped to [0, 1])
        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5  # Consider success if we got at least 50% of spec met

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
