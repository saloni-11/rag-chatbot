"""
scripts/test_rag.py
===================
Interactive test for the RAG pipeline with guardrails.

Usage (from project root):
    python scripts/test_rag.py

Commands:
    Type a question  → get an answer
    'sources'        → see source chunks from last answer
    'quit' / 'exit'  → stop
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Configure logging ─────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


def main():
    print("=" * 50)
    print("RAG Chatbot — Interactive Test (Phase 5)")
    print("=" * 50)
    print("Loading pipeline... (first run takes ~10 seconds)\n")

    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()

    print("\nReady! Ask questions about your AI/ML documents.")
    print("Commands: 'sources' (see chunks), 'quit' (exit)\n")
    print("-" * 50)

    last_result = None

    while True:
        try:
            question = input("\n📝 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        if question.lower() == "sources" and last_result:
            print("\n📚 Sources used for last answer:")
            if not last_result["sources"]:
                print("   (no sources — question was rejected by guardrails)")
            for i, src in enumerate(last_result["sources"], 1):
                score_str = f"{src['score']:.4f}" if src["score"] else "N/A"
                print(f"\n--- Source {i} ---")
                print(f"  File:  {src['file_name']}")
                print(f"  Score: {score_str}")
                # Show first 200 chars for readability
                preview = src["text"][:200].replace("\n", " ")
                print(f"  Text:  {preview}...")
            continue

        # Run the query
        result = pipeline.query(question)
        last_result = result

        # Show guardrail action
        action = result["guardrail_action"]
        action_labels = {
            "passed": "✅ Answered from sources",
            "scope_rejected": "🚫 Out of scope (no API call made)",
            "low_confidence": "⚠️  Low confidence (no API call made)",
            "empty_query": "❓ Empty question",
        }
        print(f"\n[{action_labels.get(action, action)}]")

        # Show answer
        print(f"\n💡 Answer:\n{result['answer']}")

        # Show source summary
        if result["sources"]:
            print(f"\n📚 {len(result['sources'])} source(s): ", end="")
            source_summary = ", ".join(
                f"{s['file_name']} ({s['score']:.2f})"
                for s in result["sources"]
                if s["score"]
            )
            print(source_summary)
            print("   (type 'sources' for details)")

    print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()