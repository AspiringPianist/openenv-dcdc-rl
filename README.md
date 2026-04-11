---
title: spiceRL
emoji: "🧪"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# spiceRL: OpenEnv DC-DC Converter Optimization

**spiceRL** is an OpenEnv-compliant Reinforcement Learning (RL) environment designed to automate the design of efficient DC-DC converters using Large Language Models (LLMs) and standard RL agents. It models realistic circuit behaviors, including bond wire inductances, parasitic inductors, resistors, capacitors, and real vendor models.

## The Problem It Solves

Designing power electronics—specifically picking the right passive components (inductors, capacitors) and sizing transistors—is a highly manual, tedious process. Engineers spend hours cross-referencing datasheets from vendors (e.g., Coilcraft, Murata), balancing electrical specifications, physical footprint, and component cost. 

**spiceRL** aims to completely eliminate this manual labor by:
1. **Automating Component Selection:** Mapping simulation parameters directly to real-world vendor components.
2. **Economic Optimization:** Guaranteeing a solution that meets the required efficiency and output voltage ranges while aggressively keeping the component costs down.
3. **Closing the Loop:** Providing an RL-driven design loop where the agent is iteratively penalized for over-engineering (e.g., picking unnecessarily large, expensive components) and rewarded for hitting tight electrical tolerances within a budget.

## Reinforcement Learning (RL) Framework

The environment casts the circuit design process as an RL task seamlessly integrated via the OpenEnv specification.

### Action Space (Agent Actions)
The agent tunes specific components of the converter topography (e.g., Synchronous Buck Converter) iteratively. Tunable parameters include:
* **Transistor Sizings:** High-side `W_hi_um` and low-side `W_lo_um`.
* **Passives:** Inductance `L1_nH`, Capacitance `C1_nF`.
* **Control / Switching:** Frequency `fsw_MHz`, compensator parameters `R_comp`, `C_comp_nF`, etc.
* **Component Parasitics (Real-world):** Inductor DCR `L1_DCR_mOhm` and Capacitor ESR `C1_ESR_mOhm`.

### Observation Space (State)
At each step, the environment spins up an optimized SPICE (LTspice/NGspice) simulation mapping the geometries and parameters. It returns metrics including:
* **Transient Metrics:** `vout_avg`, `vout_ripple`, `il_ripple`, `efficiency`.
* **AC Metrics:** Phase Margin, Gain Margin, Crossover Frequency.
* **Diagnostics:** Simulation convergence status and loss breakdowns.

### Tasks & Difficulties
* **Easy (Ideal Buck Converter):** Design a 1.8V -> 1.2V converter with ideal passives (no dominant parasitics).
* **Medium (Real Component Buck):** Utilize realistic parasitics linking specifically to vendor behaviors (Coilcraft inductors, Murata capacitors) to hit regulation.
* **Hard (Cost-Optimized Buck):** Minimize the financial component cost of the circuit *after* meeting electrical specs. Cost scales with inductor/capacitor sizes and total silicon area.

## Reward Functions in Full Detail

The environment utilizes a highly detailed, physics-informed reward mechanism to guide the RL agent, combining simulated electrical metrics and mathematical loss models.

### 1. Physics-Based Loss Functions (`compute_physics_losses`)
We calculate the dominant losses directly from the waveforms to understand efficiency failures:
* **Inductor DCR Loss:** Computed as $I_{rms}^2 \times R_{DCR}$, where $R_{DCR}$ scales reasonably with inductance tracking average 180nm designs.
* **Capacitor ESR Loss:** Computed as $I_{ripple\_squared} \times R_{ESR}$, recognizing ESR roughly scales inversely with capacitance.
* **Switching Losses:** Approximated as $C_{eff} \times V_{dd}^2 \times f_{sw}$, factoring in the gate charge and parasitic capacitances mapping back to the chosen transistor sizes (`W_hi` and `W_lo`).

### 2. The Comprehensive Reward Formulation (`compute_reward`)
The agent's primary maximization target computes a normalized `[0.0, 1.0]` reward score. A $0.0$ implies total circuit failure (e.g., short circuit or simulation crash), while $1.0$ dictates specs are flawlessly met or exceeded.

**Sub-rewards:**
1. **Voltage Regulation ($R_{reg}$):** Measures how close `vout_avg` is to `Vout_target` (e.g., 1.2V) divided by strict tolerance ranges. Weight: **4.0**.
2. **Ripple Voltage ($R_{rip}$):** Rewards the agent for suppressing $V_{out}$ peak-to-peak switching ripple below `ripple_max_mV`. Weight: **2.0**.
3. **Efficiency ($R_{eff}$):** Rewards for hitting the overall thermal/power `efficiency_target`. Weight: **2.0**.

**Base Calculation:**
The base reward is determined via a weighted sum bounded between `0.0` and `1.0`, taking the square root to encourage gradient slope near higher performances:
$$Base \ Reward = \sqrt{ \frac{4 \cdot R_{reg} + 2 \cdot R_{rip} + 2 \cdot R_{eff}}{8} }$$

### 3. Economic Penalty (The "Hard" Task factor)
When utilizing the `hard` environment, the agent actively minimizes economic cost. If the circuit works (Voltage regulation score > 0.8), a strict cost function is mapped:
$$Cost \ Score \approx \left(\frac{W_{total}}{100,000}\right) + \left(\frac{L_{1, nH}}{100}\right) + \left(\frac{C_{1, nF}}{500}\right)$$
This formula actively penalizes massive die area usage and excessive passive part orders. The reward suffers a penalty of up to **-30%**, forcing the agent to find economical architectures that still fulfill the strict Ripple and Vout specifications.

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run a local simulation smoke test:

```bash
python test_pipeline.py
```

## Hugging Face Spaces Deployment

The repository is configured to deploy directly to Hugging Face Spaces via the Docker SDK.
1. Create a new Space on Hugging Face.
2. Choose **Docker** as the SDK.
3. Push the code to the space.
4. The application will expose port `7860` under a `user` with uid 1000, compliant with Hugging Face requirements.

## Running and Testing via Hugging Face (RL Client)

Once deployed to Hugging Face, the environment serves a compliant OpenEnv HTTP API. You can connect standard RL loops to it remotely using the provided `SpiceRLEnv` client without running the heavy SPICE simulation binaries locally.

Here is an example of an asynchronous RL interaction loop that targets your Hugging Face Space:

```python
import asyncio
from client import SpiceRLEnv
from server.models import SpiceRLAction

async def run_remote_rl_episode():
    # Replace this with your actual Hugging Face Space direct URL
    # Can also be "http://localhost:7860" if running locally via Docker
    SPACE_URL = "https://your-username-spicerl.hf.space"
    
    # 1. Instantiate the remote environment client
    async with SpiceRLEnv(base_url=SPACE_URL) as env:
        
        # 2. Reset the environment and select task difficulty
        # Available tasks: "easy", "medium", "hard"
        result = await env.reset(task_id="hard")
        obs = result.observation
        print(f"Target Specs: {obs.spec}")
        
        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < obs.max_steps:
            steps += 1
            
            # --- LLM / RL Agent Selection Logic Goes Here ---
            # For demonstration, we simply apply a hard-coded heuristic action
            action = SpiceRLAction(
                component_values={
                    "W_hi_um": 40000, 
                    "W_lo_um": 20000,
                    "L1_nH": 47.0, 
                    "C1_nF": 68.0, 
                    "fsw_MHz": 33.3,
                    "L1_DCR_mOhm": 50.0,
                    "C1_ESR_mOhm": 15.0
                }
            )

            # 3. Step the environment on Hugging Face
            step_result = await env.step(action)
            obs = step_result.observation
            done = step_result.done
            total_reward += step_result.reward

            print(f"Step {steps}:")
            print(f"  Reward: {step_result.reward:.4f}")
            print(f"  Vout: {obs.vout_avg:.3f}V | Target: {obs.spec['Vout_target']}V")
            print(f"  Efficiency: {obs.efficiency*100:.1f}%")
            print(f"  Converged: {obs.sim_converged}")
            
            if obs.sim_error:
                print(f"  Sim Error: {obs.sim_error}")

        print(f"Episode Finished! Total Cumulative Reward: {total_reward:.4f}")

if __name__ == "__main__":
    asyncio.run(run_remote_rl_episode())
```

In this setup:
1. `SpiceRLEnv` wraps the network calls logic to standard OpenEnv HTTP boundaries. Let your RL logic propose the dictionary payload inside `SpiceRLAction()`.
2. The agent submits its predicted parameters to Hugging Face.
3. Hugging Face spins up the local NGspice/LTspice daemon, calculates waveforms, assesses penalties, and returns the observation back seamlessly.

