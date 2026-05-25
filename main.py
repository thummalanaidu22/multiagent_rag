#!/usr/bin/env python3
"""
Interactive chatbot interface for the Multi-Agent RAG system.

Usage:
    python main.py
    python main.py --query "What is the speed limit requirement for AIS-140?"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.orchestrator import OrchestratorAgent


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     Multi-Agent RAG Chatbot — ARAI Automotive Standards      ║
║     Model: Claude claude-sonnet-4-6  |  Type 'exit' to quit       ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_interactive(agent: OrchestratorAgent):
    print(BANNER)
    print("Ask any question about the ingested ARAI AIS documents.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if query.lower() in ("reset", "clear"):
            agent.reset_conversation()
            print("[Conversation cleared]\n")
            continue

        print("\nAssistant: ", end="", flush=True)
        answer, chunks = agent.chat(query)
        print(answer)

        if chunks:
            sources = list({c["source"] for c in chunks})
            print(f"\n  [Sources: {', '.join(sources)}]\n")
        else:
            print()


def run_single(agent: OrchestratorAgent, query: str):
    print(f"Query: {query}\n")
    answer, chunks = agent.chat(query)
    print(f"Answer:\n{answer}")
    if chunks:
        sources = list({c["source"] for c in chunks})
        print(f"\nSources: {', '.join(sources)}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent RAG Chatbot for ARAI Standards.")
    parser.add_argument("--query", "-q", type=str, help="Single query mode (non-interactive)")
    args = parser.parse_args()

    print("Initialising agents...")
    agent = OrchestratorAgent()
    print("Ready.\n")

    if args.query:
        run_single(agent, args.query)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
