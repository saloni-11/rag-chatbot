"""
scripts/test_rag.py
===================
Quick interactive test for the RAG pipeline.
Ask questions about your ingested documents and see the answers + sources.

Usage (from project root):
    python scripts/test_rag.py

Type 'quit' or 'exit' to stop.
Type 'sources' after a query to see full source details.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Configure logging (less verbose for interactive use) ──
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


def main():
    print("=" * 50)
    print("RAG Chatbot — Interactive Test (Phase 4)")
    print("=" * 50)
    print("Loading pipeline... (first run takes ~10 seconds)\n")

    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()

    print("\nReady! Ask questions about your AI/ML documents.")
    print("Type 'quit' to exit.\n")
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
            for i, src in enumerate(last_result["sources"], 1):
                print(f"\n--- Source {i} ({src['file_name']}, score: {src['score']}) ---")
                print(src["text"])
            continue

        result = pipeline.query(question)
        last_result = result

        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n📚 Used {len(result['sources'])} source(s): ", end="")
        print(", ".join(s["file_name"] for s in result["sources"]))
        print("   (type 'sources' to see full source text)")


    print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()