"""Claude API wrapper with streaming for LegacyLens."""

import os
from typing import AsyncGenerator

import anthropic

from config import CLAUDE_MODEL, LLM_MAX_TOKENS, INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK


SYSTEM_PROMPTS = {
    "query": (
        "You are an expert Fortran and legacy systems engineer analyzing the BLAS "
        "(Basic Linear Algebra Subprograms) library. Answer the developer's question "
        "using ONLY the provided code chunks. Cite every factual claim with "
        "[filename:start_line-end_line]. If the context does not contain enough "
        "information to answer, say so explicitly and suggest a more specific query. "
        "Never invent code, function names, or line numbers. Be concise."
    ),
    "explain": (
        "You are an expert Fortran and legacy systems engineer analyzing the BLAS "
        "(Basic Linear Algebra Subprograms) library. Explain the purpose, mathematical "
        "operation, inputs, outputs, and algorithm of the code in the provided chunks. "
        "Use plain English accessible to a developer unfamiliar with Fortran. "
        "Cite every factual claim with [filename:start_line-end_line]. "
        "Never invent code, function names, or line numbers. Be concise."
    ),
    "docgen": (
        "You are an expert Fortran and legacy systems engineer analyzing the BLAS "
        "(Basic Linear Algebra Subprograms) library. Generate a complete Fortran "
        "comment header block in standard BLAS documentation format for the routine "
        "shown in the provided code chunks. Include: Purpose, Arguments (with types, "
        "dimensions, intent), Further Details, and References. "
        "Cite every factual claim with [filename:start_line-end_line]. "
        "Never invent code, function names, or line numbers. Be concise."
    ),
    "translate": (
        "You are an expert Fortran and legacy systems engineer analyzing the BLAS "
        "(Basic Linear Algebra Subprograms) library. Provide a Python/NumPy equivalent "
        "of the Fortran routine shown in the provided code chunks. Show a side-by-side "
        "comparison explaining how each Fortran construct maps to Python. Include the "
        "NumPy/SciPy function call that replaces the BLAS routine if one exists. "
        "Cite every factual claim with [filename:start_line-end_line]. "
        "Never invent code, function names, or line numbers. Be concise."
    ),
    "patterns": (
        "You are an expert Fortran and legacy systems engineer analyzing the BLAS "
        "(Basic Linear Algebra Subprograms) library. Analyze the structural patterns, "
        "coding conventions, and similarities in the provided code chunks. Identify "
        "common patterns such as: argument validation, loop structures, special case "
        "handling, memory access patterns, and optimization techniques. "
        "Cite every factual claim with [filename:start_line-end_line]. "
        "Never invent code, function names, or line numbers. Be concise."
    ),
}

_async_client = None


def _get_async_client():
    """Reuse a single AsyncAnthropic client to avoid connection overhead."""
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _async_client


INPUT_COST_PER_TOKEN = INPUT_COST_PER_MTOK / 1_000_000
OUTPUT_COST_PER_TOKEN = OUTPUT_COST_PER_MTOK / 1_000_000


async def generate_answer(query: str, context: str,
                          mode: str = "query",
                          conversation_history: list[dict] = None) -> AsyncGenerator[str, None]:
    """Stream an answer from Claude Sonnet using the RAG context.

    Yields chunks of text as they arrive, then yields a final JSON metadata line
    prefixed with \\x00 containing token counts and cost.
    """
    client = _get_async_client()

    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["query"])

    if context:
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        user_message = (
            f"Question: {query}\n\n"
            "Note: No relevant code chunks were found in the BLAS codebase for this query. "
            "Please let the user know and suggest how they might refine their question."
        )

    # Build messages list with conversation history
    messages = []
    if conversation_history:
        for turn in conversation_history[-6:]:  # Keep last 3 exchanges
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    input_tokens = 0
    output_tokens = 0

    async with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=system_prompt,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text

        # Get final message for token counts
        final_message = await stream.get_final_message()
        input_tokens = final_message.usage.input_tokens
        output_tokens = final_message.usage.output_tokens

    cost = (input_tokens * INPUT_COST_PER_TOKEN) + (output_tokens * OUTPUT_COST_PER_TOKEN)

    # Yield metadata as a special marker
    import json
    metadata = json.dumps({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6)
    })
    yield f"\x00{metadata}"
