import os
import secrets
from typing import TYPE_CHECKING, Annotated, TypeVar

import httpx_aiohttp
from aviary.core import Message, Messages, Tool, ToolRequestMessage, ToolsAdapter
from pydantic import BaseModel

from ldp.graph import OpResult, get_training_mode

from .agent import Agent
from .simple_agent import SimpleAgentState

if TYPE_CHECKING:
    import httpx._types
    from fastapi import FastAPI

TSerializableAgentState = TypeVar("TSerializableAgentState", bound=BaseModel)


class HTTPAgentClient(Agent[TSerializableAgentState]):
    """Interact with an Agent running in a server via POST requests."""

    def __init__(
        self,
        agent_state_type: type[TSerializableAgentState],
        server_url: str,
        request_headers: "httpx._types.HeaderTypes | None" = None,
        request_timeout: float | None = None,
    ):
        super().__init__()
        self._agent_state_type = agent_state_type
        self._request_url = server_url
        self._request_headers = request_headers
        self._request_timeout = request_timeout

    async def get_asv(
        self,
        agent_state: TSerializableAgentState,
        obs: list[Message],
    ) -> tuple[OpResult[ToolRequestMessage], TSerializableAgentState, float]:
        async with httpx_aiohttp.HttpxAiohttpClient() as client:
            response = await client.post(
                f"{self._request_url}/get_asv",
                json={
                    "agent_state": agent_state.model_dump(),
                    "obs": [m.model_dump() for m in obs],
                    "training": get_training_mode(),
                },
                headers=self._request_headers,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            return (
                OpResult.from_dict(ToolRequestMessage, response_data[0]),
                self._agent_state_type(**response_data[1]),
                response_data[2],
            )

    async def init_state(self, tools: list[Tool]) -> TSerializableAgentState:
        async with httpx_aiohttp.HttpxAiohttpClient() as client:
            response = await client.post(
                f"{self._request_url}/init_state",
                json=ToolsAdapter.dump_python(tools),
                headers=self._request_headers,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            return self._agent_state_type(**response.json())


def make_simple_agent_server(
    agent: Agent[SimpleAgentState], render_docs: bool = False
) -> "FastAPI":
    """
    Make a FastAPI app designed to work with the above HTTPAgentClient.

    Here's how this works:
    1. There is an entity orchestrating an Agent's interactions with an Environment.
       A simple example of this is an integration test that sequentially calls
       Agent.get_asv and Environment.step.
    2. That entity is given the above HTTPAgentClient. Any Agent.init_state or
       Agent.get_asv calls the orchestration entity makes are actually
       POST requests under the hood. The agent's "brains" aren't local.
    3. Remotely, this server code is running, and is where the actual Agent logic lives.
       An example of this is a remote server containing GPU(s).
    """
    try:
        from fastapi import Body, Depends, FastAPI, HTTPException, status
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Please install aviary with the 'server' extra like so:"
            " `pip install aviary[server]`."
        ) from exc

    asgi_app = FastAPI(
        title=f"aviary.Agent {type(agent).__name__}",
        description=(
            "Serve inference endpoints for an aviary.Agent with a SimpleAgentState"
        ),
        # Only render Swagger docs if local since we don't have a login here
        docs_url="/docs" if render_docs else None,
        redoc_url="/redoc" if render_docs else None,
    )
    auth_scheme = HTTPBearer()

    def validate_token(
        token: Annotated[HTTPAuthorizationCredentials, Depends(auth_scheme)],
    ) -> HTTPAuthorizationCredentials:
        # NOTE: don't use os.environ.get() to avoid possible empty string matches, and
        # to have clearer server failures if the AUTH_TOKEN env var isn't present
        if not secrets.compare_digest(token.credentials, os.environ["AUTH_TOKEN"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token

    @asgi_app.get("/info")
    def info(
        _: Annotated[HTTPAuthorizationCredentials, Depends(validate_token)],
    ) -> dict[str, str]:
        """Get agent metadata, useful for debugging."""
        return {"agent_type": type(agent).__name__}

    @asgi_app.post("/get_asv")
    async def get_asv(
        agent_state: SimpleAgentState,
        obs: Messages,
        _: Annotated[HTTPAuthorizationCredentials, Depends(validate_token)],
        training: Annotated[bool, Body()] = True,
    ) -> tuple[dict, SimpleAgentState, float]:
        if training:
            raise NotImplementedError("Training is not yet supported.")
        action, agent_state, vhat = await agent.get_asv(agent_state, obs)
        return action.to_dict(), agent_state, vhat

    @asgi_app.post("/init_state")
    async def init_state(
        tools: list[Tool],
        _: Annotated[HTTPAuthorizationCredentials, Depends(validate_token)],
    ) -> SimpleAgentState:
        return await agent.init_state(tools)

    return asgi_app
