"""
src/evaluation/ragas_eval.py — Phase 10: RAG Evaluation
========================================================
Measures your RAG pipeline's quality using RAGAS 0.4.x metrics.

METRICS:
  1. Faithfulness — does the answer stick to the retrieved context?
  2. LLMContextRecall — did we retrieve everything needed?
  3. FactualCorrectness — is the answer factually correct vs ground truth?
  4. LLMContextPrecisionWithoutReference — are retrieved chunks relevant?

Uses Groq as the evaluation LLM and HuggingFace for embeddings,
so no OpenAI API key is needed.

HOW TO RUN:
  python src/evaluation/ragas_eval.py
  python src/evaluation/ragas_eval.py --dataset path/to/dataset.json
"""

import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# langchain_community >= 0.3 removed chat_models.vertexai; stub it so ragas can import
def _stub_vertexai():
    class ChatVertexAI:
        pass

    mod = types.ModuleType("langchain_community.chat_models.vertexai")
    mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mod


_stub_vertexai()

from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

load_dotenv()

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


def load_eval_dataset(dataset_path: str) -> list:
    """Load the evaluation dataset from a JSON file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}\n"
            f"Create it at tests/eval_dataset.json"
        )

    with open(path) as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} evaluation questions from {path.name}")
    return dataset


def run_pipeline_on_dataset(dataset: list) -> list:
    """
    Run each question through the RAG pipeline and collect results
    as RAGAS SingleTurnSample objects.
    """
    from ragas import SingleTurnSample

    from src.rag.pipeline import RAGPipeline

    logger.info("Initialising RAG pipeline for evaluation...")
    pipeline = RAGPipeline()

    samples = []

    logger.info(f"Running {len(dataset)} questions through the pipeline...")

    for i, entry in enumerate(dataset):
        question = entry["question"]
        ground_truth = entry["ground_truth"]

        logger.info(f"  [{i + 1}/{len(dataset)}] {question[:60]}...")

        result = pipeline.query(question)

        answer = result["answer"]
        context_list = [source["text"] for source in result["sources"]]

        if not context_list:
            context_list = ["No relevant context retrieved."]
            logger.warning(f"    Guardrail: {result['guardrail_action']} — no contexts")

        # Create a RAGAS sample for this question
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=context_list,
            reference=ground_truth,
        )
        samples.append(sample)

    logger.info(f"Pipeline run complete — {len(samples)} samples collected")
    return samples


def get_eval_llm_and_embeddings():
    """
    Create Groq LLM and HuggingFace embeddings wrapped for RAGAS.

    RAGAS 0.4.x requires wrappers:
      - LangchainLLMWrapper for the LLM
      - LangchainEmbeddingsWrapper for embeddings

    This keeps everything on free-tier tools — no OpenAI needed.
    """
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    # Groq LLM for evaluation (judges answer quality)
    llm_groq = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.0,
    )
    eval_llm = LangchainLLMWrapper(llm_groq)
    logger.info("Evaluation LLM: Groq (llama-3.1-8b-instant)")

    # HuggingFace embeddings (same model as your pipeline)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    eval_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    logger.info("Evaluation embeddings: HuggingFace (all-MiniLM-L6-v2)")

    return eval_llm, eval_embeddings


def evaluate_with_ragas(samples: list) -> dict:
    """
    Run RAGAS evaluation on the collected samples.

    RAGAS 0.4.x uses:
      - EvaluationDataset instead of HuggingFace Dataset
      - Metric classes instead of metric objects
      - LangchainLLMWrapper for non-OpenAI LLMs
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
    )

    logger.info("Running RAGAS evaluation...")
    logger.info("  (this calls the LLM multiple times — may take a few minutes)")
    logger.info("  (if you hit rate limits, wait and re-run)")

    eval_llm, eval_embeddings = get_eval_llm_and_embeddings()

    eval_dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            LLMContextRecall(),
            FactualCorrectness(),
            LLMContextPrecisionWithoutReference(),
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    return result


def print_results(result, samples: list):
    """Print a formatted summary of the evaluation results."""
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    metric_labels = {
        "faithfulness": "Faithfulness",
        "llm_context_recall": "Context Recall",
        "factual_correctness": "Factual Correctness",
        "llm_context_precision_without_reference": "Context Precision",
    }

    df = result.to_pandas()

    print("\nOverall Metrics:")
    print("-" * 40)
    for key, label in metric_labels.items():
        if key in df.columns:
            value = df[key].mean()
            bar = "\u2588" * int(value * 20) + "\u2591" * (20 - int(value * 20))
            print(f"  {label:25s} {bar} {value:.4f}")
        else:
            print(f"  {label:25s} {'N/A':>6s}")

    print("\nPer-Question Breakdown:")
    print("-" * 60)
    for i, row in df.iterrows():
        q = samples[i].user_input
        print(f"\n  Q{i + 1}: {q[:55]}{'...' if len(q) > 55 else ''}")
        for key, label in metric_labels.items():
            if key in df.columns:
                val = row[key]
                if isinstance(val, (int, float)):
                    print(f"      {label:22s} {val:.4f}")
                else:
                    print(f"      {label:22s} N/A")

    print("\n" + "=" * 60)


def save_results(result, samples: list, output_path: str):
    """Save evaluation results to a JSON file."""
    df = result.to_pandas()

    metric_keys = [
        "faithfulness",
        "llm_context_recall",
        "factual_correctness",
        "llm_context_precision_without_reference",
    ]

    output = {"overall": {}, "per_question": []}

    for key in metric_keys:
        if key in df.columns:
            output["overall"][key] = round(float(df[key].mean()), 4)

    for i, row in df.iterrows():
        entry = {
            "question": samples[i].user_input,
            "answer": samples[i].response,
            "ground_truth": samples[i].reference,
            "scores": {},
        }
        for key in metric_keys:
            if key in df.columns:
                val = row[key]
                if isinstance(val, (int, float)):
                    entry["scores"][key] = round(float(val), 4)
        output["per_question"].append(entry)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the RAG pipeline"
    )
    parser.add_argument(
        "--dataset",
        default="tests/eval_dataset.json",
        help="Path to the evaluation dataset JSON file",
    )
    parser.add_argument(
        "--output",
        default="data/eval_results.json",
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG Evaluation with RAGAS — Phase 10")
    print("=" * 60)

    dataset = load_eval_dataset(args.dataset)
    samples = run_pipeline_on_dataset(dataset)
    result = evaluate_with_ragas(samples)
    print_results(result, samples)
    save_results(result, samples, args.output)

    print(f"\nEvaluation complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
