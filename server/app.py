"""
SpiceRL FastAPI Application.

Creates the HTTP/WebSocket server using OpenEnv's create_app utility.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from server.environment import SpiceRLEnvironment
from server.models import SpiceRLAction, SpiceRLObservation

# Create the app using OpenEnv's standard factory.
# Pass the class (not instance) so each WebSocket session gets its own env.
app = create_app(
    SpiceRLEnvironment,
    SpiceRLAction,
    SpiceRLObservation,
    env_name="spice_rl",
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
