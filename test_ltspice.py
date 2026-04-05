"""Test actual LTspice simulation through SpiceRL pipeline."""
import os
import sys

# Set LTspice backend
os.environ["SPICE_BACKEND"] = "ltspice"

from server.simulator import SpiceSimulator
from server.tasks import TASKS
from server.reward import compute_reward

task = TASKS["easy"]

print("=" * 60)
print("RUNNING LTSPICE TRANSIENT SIMULATION")
print("=" * 60)
print(f"Template: {task.tran_template}")
print(f"Params: {task.default_values}")
print()

sim = SpiceSimulator(output_folder="./sim_output")

# Run with default values
metrics = sim.run_simulation(
    template_path=task.tran_template,
    params=task.default_values,
    run_name="test_buck",
    ac_template_path=task.ac_template,
)

print()
print("=" * 60)
print("SIMULATION RESULTS")
print("=" * 60)
for key, val in sorted(metrics.items()):
    print(f"  {key}: {val}")

print()
print("=" * 60)
print("REWARD CALCULATION")
print("=" * 60)
reward = compute_reward(metrics, task.spec, task.difficulty)
print(f"  Reward: {reward:.4f}")
print(f"  Vout avg: {metrics.get('vout_avg', 'N/A')} (target: {task.spec['Vout_target']})")
print(f"  Efficiency: {metrics.get('efficiency', 'N/A')}%")
print(f"  Ripple: {metrics.get('vout_ripple', 'N/A')} V")
print(f"  Phase margin: {metrics.get('phase_margin_deg', 'N/A')} deg")
print()

if metrics.get("sim_error"):
    print(f"[ERROR] {metrics['sim_error']}")
else:
    print("[PASS] Simulation completed successfully!")
