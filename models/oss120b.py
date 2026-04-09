"""
GPT-OSS 120B model backend — text-based invoice extraction via Ollama.

Pipeline: PPStructureV3 OCR text → Ollama LLM → structured JSON
"""

import os

import ollama

from utils.invoice import build_prompt, parse_json_response

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")


def chat_with_gpt_oss_with_usage(
    prompt, model_name=DEFAULT_MODEL
):
    """Send prompt to GPT-OSS via Ollama and return content + token usage."""
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        think="high",
        options={"temperature": 0},
    )

    return {
        "content": response.message.content,
        "usage": {
            "prompt_tokens": response.prompt_eval_count,
            "completion_tokens": response.eval_count,
            "total_tokens": (response.prompt_eval_count or 0)
            + (response.eval_count or 0),
        },
    }


def extract_invoice(
    text_content,
    rec_texts,
    model_name=DEFAULT_MODEL,
):
    """Extract invoice data from OCR text using GPT-OSS 120B.

    Args:
        text_content: Structured OCR text (tables + blocks).
        rec_texts: Raw OCR text lines.
        model_name: Ollama model name to use.

    Returns:
        (extracted_json, token_usage, prompt)
    """
    prompt = build_prompt(text_content, rec_texts)
    result = chat_with_gpt_oss_with_usage(prompt, model_name=model_name)
    raw_content = result.get("content", "")
    usage = result.get("usage", {}) or {}

    token_usage = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }

    return parse_json_response(raw_content), token_usage, prompt
