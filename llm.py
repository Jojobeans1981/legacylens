"""Claude API wrapper with streaming for GRIMOIRE."""

import json
import os
from typing import AsyncGenerator

import anthropic

from config import CLAUDE_MODEL, LLM_MAX_TOKENS, INPUT_COST_PER_MTOK, OUTPUT_COST_PER_MTOK


_BASE_PROMPT = (
    "You are an expert Fortran/legacy-systems engineer for BLAS/LAPACK/ScaLAPACK. "
    "Use ONLY the provided code chunks. Cite facts as [file:start-end]. "
    "Never invent code or line numbers. Be concise."
)

_MODE_INSTRUCTIONS = {
    "query": "Answer the developer's question. If context is insufficient, say so and suggest a refined query.",
    "explain": "Explain the purpose, math operation, inputs, outputs, and algorithm in plain English.",
    "docgen": "Generate a Fortran comment header: Purpose, Arguments (types, dims, intent), Details, References.",
    "translate": "Provide a Python/NumPy equivalent. Show side-by-side Fortran→Python mapping. Include the NumPy/SciPy replacement call if one exists.",
    "patterns": "Analyze structural patterns: argument validation, loop structures, special-case handling, memory access, optimization.",
}

SYSTEM_PROMPTS = {mode: f"{_BASE_PROMPT} {instruction}" for mode, instruction in _MODE_INSTRUCTIONS.items()}

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
        for turn in conversation_history[-6:]:  # Last 6 turns (3 exchanges)
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
    metadata = json.dumps({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6)
    })
    yield f"\x00{metadata}"
