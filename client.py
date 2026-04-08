"""
SpiceRL Client — EnvClient for connecting to the SpiceRL server.

Usage:
    # Connect to a running server
    from spice_rl import SpiceRLEnv, SpiceRLAction

    async with SpiceRLEnv(base_url="http://localhost:7860") as env:
        result = await env.reset(task_id="easy")
        result = await env.step(SpiceRLAction(
            component_values={"L1_uH": 22.0, "Cout_uF": 100.0, "fsw_kHz": 500.0}
        ))

    # From Docker image
    env = await SpiceRLEnv.from_docker_image("spice-rl:latest")
"""

from typing import Any, Dict
from openenv.core.env_client import EnvClient, StepResult
import warnings
from server.models import SpiceRLAction, SpiceRLObservation, SpiceRLState

class SpiceRLEnv(EnvClient):
    """Client for the SpiceRL environment.

    Inherits standard functionality from EnvClient:
    - reset(**kwargs): Reset and get initial observation
    - step(action): Execute an action
    - state: Get current state
    - close(): Clean up
    - from_docker_image(): Start from Docker
    """
    
    def _step_payload(self, action: SpiceRLAction) -> Dict[str, Any]:
        return action.model_dump()
        
    def _parse_result(self, raw_data: dict) -> StepResult[SpiceRLObservation]:
        obs = SpiceRLObservation(**raw_data.get('observation', {}))
        return StepResult(
            observation=obs,
            reward=raw_data.get('reward', 0.0),
            done=raw_data.get('done', False)
        )
        
    def _parse_state(self, raw_data: dict) -> SpiceRLState:
        return SpiceRLState(**raw_data)
