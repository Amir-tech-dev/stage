"""
PPStructureV3 OCR engine — runs PaddleOCR table/text extraction on images.

No Streamlit dependency. Returns pure data structures.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import warnings
warnings.filterwarnings(
    "ignore",
    message="No ccache found",
    category=UserWarning,
    module="paddle",
)

try:
    from paddleocr import PPStructureV3 as _PPStructureV3
except Exception:
    _PPStructureV3 = None

from ocr.constants import (
    TEXT_DET_BOX_THRESH,
    TEXT_DET_LIMIT_SIDE_LEN,
    TEXT_DET_THRESH,
    TEXT_DET_UNCLIP_RATIO,
    TEXT_REC_SCORE_THRESH,
)


# ── Internal regex patterns ──────────────────────────────────────────────────

_TOTAL_LABEL_RE = re.compile(
    r"^(Montant\s+HT|TVA|TSE|Montant\s+TTC)\s*:?$", re.IGNORECASE
)
_NUMBER_FRAG_RE = re.compile(r"^[\d\s,\.]+$")
_LIVRAISON_RE = re.compile(r"livraison|\d{2}[/,]\d{2}[/,]\d{4}", re.IGNORECASE)
_LIVRE_PAR_RE = re.compile(r"livr[ée]\s*par|camion|pipe|navire|stpe", re.IGNORECASE)


# ── Pipeline initialization (caches expensive model loading) ─────────────────

_cached_pipeline = None


def get_pipeline() -> Any:
    """Return a cached PPStructureV3 pipeline instance."""
    global _cached_pipeline
    if _cached_pipeline is not None:
        return _cached_pipeline

    if _PPStructureV3 is None:
        raise RuntimeError("PaddleOCR is not available. Install paddleocr.")

    _cached_pipeline = _PPStructureV3(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type="max",
        text_det_thresh=TEXT_DET_THRESH,
        text_det_box_thresh=TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=TEXT_DET_UNCLIP_RATIO,
        text_rec_score_thresh=TEXT_REC_SCORE_THRESH,
        use_formula_recognition=False,
        use_seal_recognition=False,
        use_chart_recognition=False,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        use_region_detection=True,
        device="cpu",
        enable_cinn=False,
    )
    return _cached_pipeline


# ── OCR execution ────────────────────────────────────────────────────────────

def run_ppstructure(image_paths: List[Path], json_output_dir: Path) -> List[Dict[str, Any]]:
    """Run PPStructureV3 OCR on a list of images, save JSON results."""
    pipeline = get_pipeline()
    page_results: List[Dict[str, Any]] = []

    for image_path in image_paths:
        output = pipeline.predict(
            str(image_path),
            use_table_recognition=True,
            use_ocr_results_with_table_cells=True,
        )

        for res in output:
            res.save_to_json(save_path=str(json_output_dir))

        json_path = json_output_dir / f"{image_path.stem}_res.json"
        if not json_path.exists():
            candidates = sorted(
                json_output_dir.glob("*_res.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise RuntimeError(f"OCR result JSON not found for {image_path.name}.")
            json_path = candidates[0]

        with json_path.open("r", encoding="utf-8") as f:
            page_results.append(json.load(f))

    return page_results


# ── Text collection ──────────────────────────────────────────────────────────

def _extract_invoice_metadata(rec_texts: List[str]) -> List[str]:
    """Extract invoice metadata (totals, livraison, livre_par) from raw OCR texts.

    Reduces prompt size by filtering only useful metadata lines.
    """
    result: List[str] = []
    i = 0
    while i < len(rec_texts):
        text = (rec_texts[i] or "").strip()

        # Total label: collect and rejoin following number fragments
        if _TOTAL_LABEL_RE.match(text):
            label = text.rstrip(":")
            parts: List[str] = []
            j = i + 1
            while j < len(rec_texts) and _NUMBER_FRAG_RE.match(
                (rec_texts[j] or "").strip()
            ):
                parts.append((rec_texts[j] or "").strip())
                j += 1
            if parts:
                result.append(f"{label}: {' '.join(parts)}")
                i = j
            else:
                i += 1
            continue

        # Livraison line
        if _LIVRAISON_RE.search(text):
            result.append(text)
            if i + 1 < len(rec_texts):
                nxt = (rec_texts[i + 1] or "").strip()
                if re.search(r"\d{2}[/,]\d{2}[/,]\d{4}", nxt) or re.match(
                    r"\)?[\d,/]+\s+au\s+", nxt
                ):
                    result.append(nxt)
                    i += 2
                    continue
            i += 1
            continue

        # Livre par / transport mode
        if _LIVRE_PAR_RE.search(text):
            result.append(text)
            i += 1
            continue

        i += 1
    return result


def collect_ocr_content(page_results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Collect OCR content from PPStructureV3 results.

    Returns:
        (text_content, rec_texts) — structured text blocks and raw OCR lines.
    """
    block_contents: List[str] = []
    rec_texts: List[str] = []
    seen_html: set = set()

    for page_idx, page_data in enumerate(page_results, start=1):
        # Get overall OCR texts
        overall_ocr_res = page_data.get("overall_ocr_res", {}) or {}
        page_rec_texts = overall_ocr_res.get("rec_texts", []) or []
        rec_texts.extend(page_rec_texts)

        # Get tables (already valid HTML)
        for table_res in page_data.get("table_res_list", []) or []:
            pred_html = table_res.get("pred_html", "").strip()
            if pred_html and pred_html not in seen_html:
                seen_html.add(pred_html)
                block_contents.append(f"[TABLE]\n{pred_html}\n[/TABLE]")

        # Get other content blocks
        for block in page_data.get("parsing_res_list", []):
            content = block.get("block_content", "").strip()
            block_type = str(block.get("block_type", "")).strip().lower()
            if not content or block_type == "table":
                continue
            if content.startswith("<html>") and content in seen_html:
                continue
            block_contents.append(content)

        # Append invoice metadata
        metadata = _extract_invoice_metadata(page_rec_texts)
        if metadata:
            flat = "\n".join(t for t in metadata if t)
            block_contents.append(f"[INVOICE_METADATA]\n{flat}\n[/INVOICE_METADATA]")

    return "\n\n".join(block_contents), rec_texts


def livre_par(rec_texts: List[str]) -> Any:
    """Extract delivery method (pipe/camion/navire/stpe) from OCR texts."""
    for text in rec_texts:
        if re.search(r"\b(pipe|camion|navire|stpe)\b", text or "", re.IGNORECASE):
            return text.strip()
    return None
