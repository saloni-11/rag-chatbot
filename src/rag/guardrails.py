"""
Guardrails — Phase 4 Stub (Phase 5 builds this out fully)
==========================================================
For now, this module contains the system prompt that controls how
the LLM behaves when answering questions.

What is a system prompt?
  The system prompt is a set of instructions sent to the LLM BEFORE
  the user's question. It shapes the LLM's behaviour — like giving
  someone a job description before they start working.

  Without a system prompt, the LLM would answer from its general
  knowledge (which might be wrong or outdated). Our system prompt
  tells it: "ONLY answer based on the context chunks I'm giving you."

Why does this matter for RAG?
  The whole point of RAG is to ground answers in your documents.
  If the LLM ignores the retrieved chunks and answers from its own
  training data, you've lost the "retrieval-augmented" part entirely.
  The system prompt enforces this contract.

What Phase 5 will add:
  - Scope guardrail: reject questions outside AI/ML/Data Analytics
  - Confidence threshold: if retrieved chunks aren't relevant enough,
    return "I don't have enough information" instead of guessing
  - Hallucination checks: verify the answer is actually supported
    by the retrieved context
"""

from llama_index.core import PromptTemplate

# ── System Prompt ────────────────────────────────────
# This template has two variables that LlamaIndex fills in automatically:
#   {context_str} → the retrieved chunks (injected by the retriever)
#   {query_str}   → the user's question
#
# The instructions tell the LLM to:
#   1. Only use the provided context to answer
#   2. Cite which parts of the context it used
#   3. Admit when it doesn't know something
#   4. Stay focused on AI/ML/Data Analytics topics

SYSTEM_PROMPT = PromptTemplate(
    """\
You are an AI/ML Interview Preparation Assistant. Your job is to help users \
prepare for interviews on AI, Machine Learning, and Data Analytics topics.

RULES:
1. ONLY answer based on the context provided below. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, \
say: "I don't have enough information in my sources to answer this fully." \
Then share what partial information you do have from the context.
3. When answering, reference which source document the information comes from.
4. Keep answers clear, structured, and interview-appropriate.
5. If the question is not related to AI/ML/Data Analytics, politely redirect: \
"I'm designed to help with AI/ML and Data Analytics interview preparation. \
Could you ask something in that area?"

CONTEXT FROM SOURCE DOCUMENTS:
------------------------------
{context_str}
------------------------------

USER QUESTION: {query_str}

ANSWER:
"""
)