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

# Test 4: Check template files exist
print()
print("=" * 50)
print("TEST 4: Template files")
print("=" * 50)
for task_id, t in TASKS.items():
    tran_ok = os.path.exists(t.tran_template)
    print(f"  [{task_id}] tran: {'OK' if tran_ok else 'MISSING'} - {t.tran_template}")
    if t.ac_template:
        ac_ok = os.path.exists(t.ac_template)
        print(f"  [{task_id}] ac:   {'OK' if ac_ok else 'MISSING'} - {t.ac_template}")

# Test 5: SpiceEditor can load template
print()
print("=" * 50)
print("TEST 5: SpiceEditor loads template")
print("=" * 50)
try:
    from spicelib import SpiceEditor
    net = SpiceEditor(task.tran_template)
    print(f"[PASS] Loaded {task.tran_template}")
    # Try setting a parameter
    net.set_parameters(W_hi_um=50000)
    print("[PASS] set_parameters(W_hi_um=50000) succeeded")
    net.set_parameters(L1_nH=33)
    print("[PASS] set_parameters(L1_nH=33) succeeded")
    net.set_parameters(R_comp=6000)
    print("[PASS] set_parameters(R_comp=6000) succeeded")
except Exception as e:
    print(f"[FAIL] SpiceEditor error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 50)
print("ALL TESTS COMPLETE")
print("=" * 50)
