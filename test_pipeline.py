"""Quick pipeline test for SpiceRL."""
import sys
import os

# Test 1: Imports
print("=" * 50)
print("TEST 1: Module imports")
print("=" * 50)
try:
    from server.models import SpiceRLAction, SpiceRLObservation
    from server.tasks import TASKS
    from server.reward import compute_reward
    from server.simulator import SpiceSimulator
    print("[PASS] All imports successful")
    print(f"  Tasks: {list(TASKS.keys())}")
    task = TASKS["easy"]
    print(f"  Easy task: {task.name}")
    print(f"  Tunable params: {task.tunable_params}")
    print(f"  Default values: {task.default_values}")
    print(f"  Spec: {task.spec}")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Reward function
print()
print("=" * 50)
print("TEST 2: Reward function")
print("=" * 50)
try:
    # Simulate metrics from a good simulation result
    good_metrics = {
        "vout_avg": 1.007,
        "vout_ripple": 0.22,
        "il_ripple": 0.42,
        "efficiency": 86.6,  # Percentage from .MEAS
        "sim_converged": True,
    }
    spec = task.spec
    reward = compute_reward(good_metrics, spec, "easy")
    print(f"[PASS] Reward for good sim: {reward:.4f}")

    # Test with dead circuit
    dead_metrics = {"vout_avg": 0.001, "sim_converged": True}
    reward_dead = compute_reward(dead_metrics, spec, "easy")
    print(f"[PASS] Reward for dead circuit: {reward_dead:.4f}")

    # Test with failed sim
    fail_metrics = {"sim_error": "convergence failed", "sim_converged": False}
    reward_fail = compute_reward(fail_metrics, spec, "easy")
    print(f"[PASS] Reward for failed sim: {reward_fail:.4f}")
except Exception as e:
    print(f"[FAIL] Reward error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: SpiceSimulator instantiation
print()
print("=" * 50)
print("TEST 3: SpiceSimulator")
print("=" * 50)
try:
    sim = SpiceSimulator(output_folder="./sim_output")
    print(f"[PASS] SpiceSimulator created, output: {sim.output_folder}")
except Exception as e:
    print(f"[FAIL] SpiceSimulator error: {e}")

# Test 4: Check topology definitions
print()
print("=" * 50)
print("TEST 4: Topology Support")
print("=" * 50)
for task_id, t in TASKS.items():
    topology = getattr(t, "topology", "buck")
    print(f"  [{task_id}] topology: {topology}")
    if topology == "buck":
        print(f"  [{task_id}] OK - Native PySpice buck supported.")
    else:
        print(f"  [{task_id}] WARN - Topology '{topology}' pending PySpice implementation.")

# Test 5: PySpice Simulator execution
print()
print("=" * 50)
print("TEST 5: Simulator executes PySpice in-memory")
print("=" * 50)

for task_id, task in TASKS.items():
    print(f"\n--- Testing Task: {task.name} ({task_id}) ---")
    try:
        # Validate parameters to test snapping and pricing logic for medium/hard
        difficulty = getattr(task, "difficulty", task_id)
        clamped_params, warnings = sim.validate_params(
            task.default_values,
            task.param_bounds,
            difficulty=difficulty
        )
        if warnings:
            for w in warnings:
                print(f"  [WARN] {w}")

        print(f"  Running simulation with topology '{task.topology}'...")
        result = sim.run_simulation(
            topology=task.topology,
            params=clamped_params,
            run_name=f"dry_run_{task_id}"
        )
        
        print(f"  [PASS] Simulator executed successfully.")
        print(f"    Converged: {result.get('sim_converged')}")
        print(f"    Vout Avg: {result.get('vout_avg', 0):.3f} V")
        print(f"    Vout Ripple: {result.get('vout_ripple', 0):.3f} V")
        
        # Verify pricing was captured for medium/hard
        if difficulty in ["medium", "hard"]:
            l_price = clamped_params.get('L1_Price_USD', 0.0)
            c_price = clamped_params.get('C1_Price_USD', 0.0)
            print(f"    Inductor Cost: ${l_price:.2f}")
            print(f"    Capacitor Cost: ${c_price:.2f}")
            
        print(f"    Raw Reward metric mapped: {result.get('sim_converged')}")
    except Exception as e:
        print(f"  [FAIL] PySpice execution error for {task_id}: {e}")
        import traceback
        traceback.print_exc()

print()
print("=" * 50)
print("ALL TESTS COMPLETE")
print("=" * 50)
