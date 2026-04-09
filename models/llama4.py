"""
Llama 4 Scout model backend — vision-based invoice extraction via Groq.

Pipeline: Image → base64 → Groq API (Llama 4 vision) → structured JSON
"""

import base64
import json
import os
import time
from pathlib import Path

from groq import Groq

from ocr.constants import EXPECTED_JSON_SCHEMA
from utils.invoice import parse_json_response

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"


def _encode_image(image_path):
    """Read an image file and return its base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_invoice(
    image_paths,
    api_key=None,
):
    """Extract invoice data from images using Llama 4 Scout (vision model)."""
    api_key = api_key or GROQ_API_KEY
    if not api_key:
        raise ValueError("Set GROQ_API_KEY env variable.")

    client = Groq(api_key=api_key)

    schema_text = json.dumps(EXPECTED_JSON_SCHEMA, ensure_ascii=False, indent=2)
    content = [
        {
            "type": "text",
            "text": (
                "Extract invoice data from the image(s) below and return ONLY valid JSON.\n"
                f"Schema: {schema_text}\n"
                "Return only valid JSON, no markdown and no explanation. "
                "If a field is not explicitly present, return null. "
                "Never infer or fabricate missing values."
            ),
        }
    ]

    for img_path in image_paths:
        base64_image = _encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        })

    start = time.time()
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": content}],
        model=MODEL_NAME,
        temperature=1,
        max_completion_tokens=1500,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
    )
    response_time = time.time() - start

    response_content = chat_completion.choices[0].message.content
    if not response_content:
        raise ValueError("Empty response from Llama 4 model.")

    extracted_data = parse_json_response(response_content)

    usage = chat_completion.usage
    token_usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }

    return extracted_data, token_usage, response_time
