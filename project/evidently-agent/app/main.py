"""
main.py
-------
Command-line interface for the Evidently AI documentation agent.

Usage:
    export GROQ_API_KEY="your-key"
    python main.py
"""

import asyncio
import sys

import ingest
import logs
import search_agent

REPO_OWNER = "evidentlyai"
REPO_NAME = "docs"


def initialize() -> tuple:
    print(f"\n{'='*55}")
    print(f"  Evidently AI Docs Agent")
    print(f"  Repository: {REPO_OWNER}/{REPO_NAME}")
    print(f"{'='*55}\n")

    print("Step 1/2  Downloading & indexing documentation …")
    index, _ = ingest.index_data(
        REPO_OWNER,
        REPO_NAME,
        chunk=True,
        chunking_params={"method": "sliding_window", "size": 2000, "step": 1000},
    )

    print("\nStep 2/2  Initialising agent …")
    agent = search_agent.init_agent(index, REPO_OWNER, REPO_NAME)
    print("  ✓ Agent ready.\n")
    return agent


def main() -> None:
    agent = initialize()

    print("Type your question and press Enter.  Type 'quit' or 'exit' to stop.\n")
    print("-" * 55)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\nAgent: thinking …")
        result = asyncio.run(agent.run(user_prompt=question))
        log_path = logs.log_interaction_to_file(agent, result.new_messages())

        print(f"\nAgent:\n{result.output}")
        print(f"\n  (logged → {log_path.name})")
        print("-" * 55)


if __name__ == "__main__":
    main()
