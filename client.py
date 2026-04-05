"""
SpiceRL Client — EnvClient for connecting to the SpiceRL server.

Usage:
    # Connect to a running server
    from spice_rl import SpiceRLEnv, SpiceRLAction

    async with SpiceRLEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="easy")
        result = await env.step(SpiceRLAction(
            component_values={"L1_uH": 22.0, "Cout_uF": 100.0, "fsw_kHz": 500.0}
        ))

    # From Docker image
    env = await SpiceRLEnv.from_docker_image("spice-rl:latest")
"""

from openenv.core.env_client import EnvClient


class SpiceRLEnv(EnvClient):
    """Client for the SpiceRL environment.

    Inherits standard functionality from EnvClient:
    - reset(**kwargs): Reset and get initial observation
    - step(action): Execute an action
    - state: Get current state
    - close(): Clean up
    - from_docker_image(): Start from Docker
    """

    pass
