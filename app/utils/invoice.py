"""
Shared invoice processing utilities.

Functions for normalizing, formatting, cleaning, and converting invoice data.
Used by all model backends and the Streamlit UI.
"""

import io
import json
import re
from typing import Any, Dict, List, Mapping, MutableMapping

import pandas as pd

from ocr.constants import (
    EXPECTED_JSON_SCHEMA,
    LINE_COLUMNS,
    NUMERIC_LINE_FIELDS,
    NUMERIC_TOTAL_FIELDS,
    TOP_FIELDS,
    TOTAL_FIELDS,
)


# ── JSON parsing ─────────────────────────────────────────────────────────────

def parse_json_response(raw_content: str) -> Dict[str, Any]:
    """Parse JSON from an LLM response, handling markdown fences and extra text."""
    content = (raw_content or "").strip()
    if not content:
        raise ValueError("Empty response from LLM API.")

    # Try direct parse
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object from surrounding text
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = content[start_idx : end_idx + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse valid JSON from LLM response.")


# ── Prompt building ──────────────────────────────────────────────────────────

def build_prompt(text_content: str, rec_texts: List[str]) -> str:
    """Build the extraction prompt for text-based LLMs (OSS 120B)."""
    llm_schema = dict(EXPECTED_JSON_SCHEMA)
    schema_text = json.dumps(llm_schema, ensure_ascii=False, indent=2)
    rec_text_block = "\n".join(
        str(text).strip() for text in (rec_texts or []) if str(text).strip()
    )

    return (
        "Extract invoice data from the text below and return ONLY valid JSON.\n\n"
        "OCR Content (structured blocks + tables in HTML):\n"
        f"{text_content}\n\n"
        "RAW OCR rec_texts (line by line):\n"
        f"{rec_text_block}\n\n"
        "Required JSON structure:\n"
        f"{schema_text}\n\n"
        "Return only valid JSON, no markdown and no explanation. "
        "If a field is not explicitly present in OCR text, return null. "
        "Never infer or fabricate missing values. "
        "NB: the facture code starts with letters followed by digits (e.g. RZN210337). "
        "Totals (mnt_ht_total, tva_total, TSE, mnt_ttc) appear in [INVOICE_METADATA] under "
        "labels 'Montant HT:', 'TVA:', 'TSE:', 'Montant TTC:' — extract them exactly as shown. "
        "Livraison dates appear near 'Pour la livraison du ... au ...' in [INVOICE_METADATA]: "
        "extract the start date as date_debut_livraison and the end date as date_fin_livraison. "
        "For prix_unitaire: extract only the price number, ignoring any merged text such as "
        "'Montant HT:' that OCR may have appended to the cell. "
        "The HTML table may contain phantom empty columns caused by OCR column boundary errors — "
        "if two adjacent cells together form a single coherent value (e.g. '80 896' + '000,00'), "
        "merge them."
    )


# ── Number formatting ────────────────────────────────────────────────────────

def format_number_preserve_zeros(value: Any) -> Any:
    """Format numbers WITHOUT stripping leading zeros."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def apply_number_formatting(invoice: Dict[str, Any]) -> Dict[str, Any]:
    """Apply number formatting that preserves zeros to all numeric fields."""
    for field in NUMERIC_TOTAL_FIELDS:
        if field in invoice:
            invoice[field] = format_number_preserve_zeros(invoice.get(field))

    for row in invoice.get("facture_ligne", []) or []:
        if not isinstance(row, dict):
            continue
        for field in NUMERIC_LINE_FIELDS:
            row[field] = format_number_preserve_zeros(row.get(field))

    return invoice


# ── TVA normalization ────────────────────────────────────────────────────────


def _normalize_tva(raw: str) -> str:
    """Map OCR-noisy TVA string to the nearest valid Algerian TVA rate (0, 9, 19)."""
    digits_only = re.sub(r"[^\d,.]", "", str(raw or "").strip()).rstrip(".")
    if not digits_only:
        return str(raw)

    normalized = digits_only.replace(",", ".")
    try:
        val = float(normalized)
    except ValueError:
        return digits_only

    if val < 5:
        val *= 10
    closest = min([0, 9, 19], key=lambda x: abs(x - val))
    return str(closest)


def format_number_preserve_zeros(value: Any) -> Any:
    """Return the string value unchanged (name was misleading)."""
    return value if value is not None else None


def clean_line_fields(invoice: Dict[str, Any]) -> Dict[str, Any]:
    """Strip OCR noise from tva and prix_unitaire fields."""
    for row in invoice.get("facture_ligne", []) or []:
        if not isinstance(row, dict):
            continue
        tva = str(row.get("tva", "") or "").strip()
        if tva:
            row["tva"] = _normalize_tva(tva)
        pu = str(row.get("prix_unitaire", "") or "").strip()
        row["prix_unitaire"] = re.split(r"\s+[A-Za-zÀ-ÿ]", pu)[0].strip()
    return invoice


# ── Invoice normalization ────────────────────────────────────────────────────

def normalize_invoice_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all expected fields exist and line items are properly structured."""
    invoice: Dict[str, Any] = dict(data) if isinstance(data, dict) else {}

    for field in TOP_FIELDS + TOTAL_FIELDS:
        invoice.setdefault(field, None)

    raw_lines = invoice.get("facture_ligne")
    lines: List[Dict[str, Any]] = []
    if isinstance(raw_lines, list):
        for row in raw_lines:
            if isinstance(row, dict):
                lines.append({key: row.get(key, "") for key in LINE_COLUMNS})

    invoice["facture_ligne"] = lines
    return invoice


# ── DataFrame <-> invoice conversions ────────────────────────────────────────

def lines_to_dataframe(lines: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert invoice line items to a Pandas DataFrame."""
    if not lines:
        return pd.DataFrame([{key: "" for key in LINE_COLUMNS}])
    normalized = [{key: row.get(key, "") for key in LINE_COLUMNS} for row in lines]
    return pd.DataFrame(normalized)


def dataframe_to_lines(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a Pandas DataFrame back to invoice line items."""
    if df is None or df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for _, row in df.fillna("").iterrows():
        line = {key: str(row.get(key, "")).strip() for key in LINE_COLUMNS}
        if any(line.values()):
            rows.append(line)
    return rows


def build_current_invoice(edited_lines: pd.DataFrame, session_state: Mapping) -> Dict[str, Any]:
    """Build the current invoice dict from Streamlit session state and edited lines."""
    invoice: Dict[str, Any] = {}
    for field in TOP_FIELDS + TOTAL_FIELDS:
        val = session_state.get(f"edit_{field}")
        invoice[field] = str(val).strip() if val else None
    invoice["facture_ligne"] = dataframe_to_lines(edited_lines)
    return invoice


def seed_editor_state(invoice: Dict[str, Any], session_state: MutableMapping) -> None:
    """Populate Streamlit session state with invoice field values."""
    for field in TOP_FIELDS + TOTAL_FIELDS:
        session_state[f"edit_{field}"] = (
            "" if invoice.get(field) is None else str(invoice.get(field))
        )
    session_state["edit_lines_base"] = lines_to_dataframe(
        invoice.get("facture_ligne", [])
    )
    session_state["editor_revision"] = session_state.get("editor_revision", 0) + 1


# ── Excel export ─────────────────────────────────────────────────────────────

def invoice_to_excel_bytes(invoice: Dict[str, Any]) -> bytes:
    """Convert invoice dict to Excel file bytes (two sheets: invoice + lines)."""
    summary_rows = [{"field": field, "value": invoice.get(field)} for field in TOP_FIELDS + TOTAL_FIELDS]
    summary_df = pd.DataFrame(summary_rows)
    lines_df = pd.DataFrame(invoice.get("facture_ligne", []))

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        summary_df.to_excel(writer, index=False, sheet_name="invoice")
        lines_df.to_excel(writer, index=False, sheet_name="lines")
    buffer.seek(0)
    return buffer.getvalue()
