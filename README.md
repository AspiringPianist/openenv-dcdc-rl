# openenv-dcdc-rl

OpenEnv-compliant RL environment to design efficient DC-DC converters for LLMs, modeling bond wire inductances, parasitic inductors, resistors, capacitors, and real vendor models.

## What this provides

- OpenEnv-compatible environment and task interface for converter optimization.
- LTspice/NGspice simulation pipeline with tunable power-stage and compensation parameters.
- Reward-driven design loop for improving efficiency and electrical performance metrics.

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run a local simulation smoke test:

```bash
python test_ltspice.py
```
