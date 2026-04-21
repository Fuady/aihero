"""
search_agent.py
---------------
Initialises the Pydantic AI agent backed by Groq (llama-3.3-70b-versatile).
"""

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

from search_tools import SearchTool

REPO_OWNER = "evidentlyai"
REPO_NAME = "docs"

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions about the Evidently AI documentation.

Use the search tool to find relevant information before answering any question.
Always search first — even for broad questions — so your answers are grounded in the docs.

If you can find specific information, use it to provide accurate, detailed answers.

Always cite your sources by including a Markdown link using the filename:
  Format: [<descriptive title>](https://github.com/{repo_owner}/{repo_name}/blob/main/<filename>)

If the search returns no useful results, say so clearly and offer general guidance.
""".strip()


def init_agent(index, repo_owner: str = REPO_OWNER, repo_name: str = REPO_NAME) -> Agent:
    """
    Create and return a configured Pydantic AI Agent using Groq.

    Args:
        index:      A fitted minsearch Index.
        repo_owner: GitHub owner (used for citation links).
        repo_name:  GitHub repo name (used for citation links).

    Returns:
        A ready-to-use Pydantic AI Agent instance.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        repo_owner=repo_owner, repo_name=repo_name
    )

    search_tool = SearchTool(index=index)

    model = GroqModel("llama-3.3-70b-versatile")

    agent = Agent(
        name="evidently_agent",
        instructions=system_prompt,
        tools=[search_tool.search],
        model=model,
    )

    return agent
