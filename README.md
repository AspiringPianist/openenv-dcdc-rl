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
python test_pipeline.py
```

## Hugging Face Spaces Deployment

The repository is configured to deploy directly to Hugging Face Spaces via the Docker SDK.
1. Create a new Space on Hugging Face.
2. Choose **Docker** as the SDK.
3. Push the code to the space.
4. The application will expose port `7860` under a `user` with uid 1000, compliant with Hugging Face requirements.
