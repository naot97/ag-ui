from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import CopilotKitMiddleware, LangGraphAGUIAgent
from fastapi import FastAPI
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

_LESSON_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LESSON_ROOT.parent
CSV_PATH = _LESSON_ROOT / "db.csv"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from helper import load_api_keys
load_api_keys()


@tool
def query_data(query: str) -> list[dict[str, Any]]:
    """Query the lesson dataset. Always call before showing a chart or graph."""
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _build_graph():
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or os.environ.get("OPENAI_API_VERSION")
    graph = create_agent(
        model=AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            api_version=api_version,
        ),
        tools=[query_data],
        middleware=[CopilotKitMiddleware()],
        checkpointer=MemorySaver(),
        system_prompt=(
            "You are a helpful assistant for a demo app with a few available UI tools. "
            "When a user asks for charts based on the lesson dataset, always call query_data first to fetch all CSV rows. "
            "Prefer using a matching frontend tool when it would present the answer clearly. "
            "Call each frontend UI tool (showMyName, pieChart, flightCard) AT MOST ONCE per user turn — "
            "after a successful UI tool call, reply with a short text confirmation and stop. "
            "Use pieChart for category distributions "
            "and flightCard for a single flight summary when relevant. "
            "Tool arguments must match the provided schema exactly."
        ),
    )
    return graph.with_config(recursion_limit=100)


def build_app() -> FastAPI:
    app = FastAPI()
    agent = LangGraphAGUIAgent(
        name="lesson3_charts_agent",
        description="Lesson 3 controlled generative UI agent",
        graph=_build_graph(),
    )
    add_langgraph_fastapi_endpoint(app=app, agent=agent, path="/")
    return app


def start_backend(port: int = 8003) -> None:
    from helper import start_server

    start_server(build_app(), port=port)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(build_app(), host="0.0.0.0", port=8003, log_level="info")
