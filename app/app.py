"""
Unified Streamlit app for Naftal Invoice OCR.

Supports two model backends:
  - OSS 120B: PPStructureV3 OCR → text → Ollama GPT-OSS → JSON
  - Llama 4 Scout: Image → Groq vision API → JSON

Run with:  streamlit run app.py
"""

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import pandas as pd
import streamlit as st

from ocr.constants import ALLOWED_IMAGE_SUFFIXES, LINE_COLUMNS
from utils.invoice import (
    apply_number_formatting,
    build_current_invoice,
    clean_line_fields,
    invoice_to_excel_bytes,
    normalize_invoice_data,
    seed_editor_state,
)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Naftal Invoice OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 980px;
        margin: 0 auto;
    }
    @media (max-width: 1200px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    hr { margin: 1.5rem 0; border-color: #e2e8f0; }

    [data-testid="stBaseButton-primary"] {
        height: 55px;
        font-size: 1.1rem;
        font-weight: 600;
        background-color: #0b2265;
        border-color: #0b2265;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #0d2e87;
        border-color: #0d2e87;
    }

    [data-testid="stFileUploader"],
    [data-testid="stSelectbox"],
    [data-testid="stButton"] {
        max-width: 560px;
        margin: 0 auto;
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed #94a3b8;
        border-radius: 8px;
        background: #f8fafc;
        padding: 1.25rem 1rem;
    }
    [data-testid="stFileUploader"] button {
        border-radius: 6px;
        font-weight: 500;
        min-width: 120px;
    }

    [data-testid="stButton"] {
        text-align: center;
    }
    [data-testid="stBaseButton-primary"] {
        width: 100%;
        min-width: 280px;
    }

    [data-testid="stTextInput"] { margin-bottom: -0.5rem; }

    [data-testid="stDataEditor"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }

    [data-testid="stExpander"] summary {
        font-weight: 500;
        color: #475569;
    }

    [data-testid="stDownloadButton"] button {
        border-radius: 6px;
        font-weight: 500;
    }

    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.75rem;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Header ───────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 6])
with col1:
    st.markdown('<div style="padding-top: 0.8rem;"></div>', unsafe_allow_html=True)
    st.image("naftal_logo.png", width=120)
with col2:
    st.title("Naftal Invoice OCR")


# ── Config & Upload ──────────────────────────────────────────────────────────

MODEL_OPTIONS = {"OSS 120B": "oss120b", "Llama 4 Scout": "llama4"}

model = st.selectbox("Modèle d'IA", list(MODEL_OPTIONS.keys()))
uploaded_file = st.file_uploader("Document (PDF ou Image)")
run_button = uploaded_file and st.button("Lancer l'extraction", type="primary", use_container_width=True)

# ── Sidebar metrics ───────────────────────────────────────────────────────────

st.sidebar.subheader("Détails d'exécution")
if "invoice_data" in st.session_state:
    st.sidebar.metric("Temps total", f"{st.session_state.get('last_extraction_seconds', 0):.2f}s")
    token_usage = st.session_state.get("token_usage", {})
    st.sidebar.metric("Tokens", token_usage.get("total_tokens", "—"))
    st.sidebar.caption(f"Modèle: {st.session_state.get('selected_model_label', '—')}")
    if "timings" in st.session_state:
        st.sidebar.caption("Détail des étapes")
        for step, secs in st.session_state["timings"].items():
            st.sidebar.write(f"{step}: {secs:.1f}s")
else:
    st.sidebar.caption("Les métriques apparaîtront ici après la première extraction.")


# ── Processing pipelines ────────────────────────────────────────────────────

def process_with_oss120b(uploaded_file: Any) -> Dict[str, Any]:
    """Full pipeline: PDF/Image → PPStructureV3 → OSS 120B → structured JSON."""
    from ocr.ppstructure import collect_ocr_content, livre_par, run_ppstructure
    from models.oss120b import extract_invoice
    from utils.pdf import convert_pdf_to_images

    timings = {}
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path("final") / f"{Path(uploaded_file.name).stem}_{run_stamp}"
    images_dir = run_root / "images"
    json_dir = run_root / "ocr_json"
    images_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    upload_path = images_dir / uploaded_file.name
    upload_path.write_bytes(uploaded_file.getvalue())
    suffix = upload_path.suffix.lower()

    t = perf_counter()
    if suffix == ".pdf":
        image_paths = convert_pdf_to_images(upload_path, images_dir)
    elif suffix in ALLOWED_IMAGE_SUFFIXES:
        img_path = images_dir / f"page_1{suffix}"
        img_path.write_bytes(upload_path.read_bytes())
        image_paths = [img_path]
    else:
        raise RuntimeError("Unsupported format.")
    timings["PDF to image"] = perf_counter() - t

    t = perf_counter()
    page_results = run_ppstructure(image_paths, json_dir)
    timings["PPStructure OCR"] = perf_counter() - t

    t = perf_counter()
    text_content, rec_texts = collect_ocr_content(page_results)
    timings["Text collection"] = perf_counter() - t

    if not text_content.strip():
        raise RuntimeError("OCR finished but no readable text was extracted.")

    t = perf_counter()
    extracted_json, token_usage, llm_prompt = extract_invoice(
        text_content, rec_texts
    )
    extracted = normalize_invoice_data(extracted_json)
    extracted["livre_par"] = livre_par(rec_texts) or extracted.get("livre_par")
    extracted = apply_number_formatting(extracted)
    extracted = clean_line_fields(extracted)
    timings["LLM extraction"] = perf_counter() - t

    return {
        "invoice": extracted,
        "page_count": len(image_paths),
        "source_name": uploaded_file.name,
        "run_output_dir": str(run_root),
        "timings": timings,
        "token_usage": token_usage,
        "llm_prompt": llm_prompt,
    }


def process_with_llama4(uploaded_file: Any) -> Dict[str, Any]:
    """Full pipeline: Image/PDF → Groq Llama 4 vision → structured JSON."""
    from models.llama4 import extract_invoice
    from utils.pdf import convert_pdf_to_images

    timings = {}
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path("final") / f"llama4_{Path(uploaded_file.name).stem}_{run_stamp}"
    images_dir = run_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    upload_path = images_dir / uploaded_file.name
    upload_path.write_bytes(uploaded_file.getvalue())
    suffix = upload_path.suffix.lower()

    t = perf_counter()
    if suffix == ".pdf":
        image_paths = convert_pdf_to_images(upload_path, images_dir)
        image_paths_str = [str(p) for p in image_paths]
    elif suffix in ALLOWED_IMAGE_SUFFIXES:
        img_path = images_dir / f"page_1{suffix}"
        img_path.write_bytes(upload_path.read_bytes())
        image_paths_str = [str(img_path)]
    else:
        raise RuntimeError("Unsupported format.")
    timings["PDF to image"] = perf_counter() - t

    t = perf_counter()
    extracted_json, token_usage, response_time = extract_invoice(image_paths_str)
    timings["LLM extraction"] = perf_counter() - t
    timings["LLM response_time"] = response_time
    extracted = normalize_invoice_data(extracted_json)

    return {
        "invoice": extracted,
        "page_count": len(image_paths_str),
        "source_name": uploaded_file.name,
        "run_output_dir": str(run_root),
        "timings": timings,
        "token_usage": token_usage,
        "llm_prompt": None,
    }


# ── Run extraction ───────────────────────────────────────────────────────────

if run_button and uploaded_file is not None:
    extraction_start = perf_counter()
    selected_model = MODEL_OPTIONS[model]
    with st.spinner(f"Extraction avec {model}..."):
        try:
            if selected_model == "oss120b":
                result = process_with_oss120b(uploaded_file)
            else:
                result = process_with_llama4(uploaded_file)

            st.session_state.update({
                "invoice_data": result["invoice"],
                "page_count": result["page_count"],
                "source_name": result["source_name"],
                "timings": result["timings"],
                "token_usage": result["token_usage"],
                "selected_model_label": model,
                "last_extraction_seconds": perf_counter() - extraction_start,
            })

            seed_editor_state(result["invoice"], st.session_state)
            st.success(f"Extraction terminée en {st.session_state['last_extraction_seconds']:.1f}s via {model}.")

        except Exception as exc:
            st.session_state.pop("last_extraction_seconds", None)
            st.session_state.pop("token_usage", None)
            st.error(f"L'extraction a échoué : {exc}")

# ── Stop if no data yet ──────────────────────────────────────────────────────

if "invoice_data" not in st.session_state:
    st.stop()





# ── Invoice editor ────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Invoice header")

c1, c2, c3 = st.columns(3)
with c1:
    st.text_input("Code facture", key="edit_code")
    st.text_input("Date d'établissement", key="edit_date_etablisement")
    st.text_input("Date d'échéance", key="edit_date_echeance")
with c2:
    st.text_input("Début de livraison", key="edit_date_debut_livraison")
    st.text_input("Fin de livraison", key="edit_date_fin_livraison")
    st.text_input("Livré par", key="edit_livre_par")
with c3:
    st.text_input("N° de compte", key="edit_num_compte")
    st.text_input("Type de paiement", key="edit_type_de_paiement")


# ── Invoice lines ────────────────────────────────────────────────────────────

st.subheader("Lignes de facturation")

editor_key = f"line_editor_{st.session_state.get('editor_revision', 0)}"
edited_line_df = st.data_editor(
    st.session_state.get(
        "edit_lines_base",
        pd.DataFrame([{key: "" for key in LINE_COLUMNS}]),
    ),
    num_rows="dynamic",
    use_container_width=True,
    key=editor_key,
    hide_index=True,
    column_config={
        "designation": st.column_config.TextColumn("Designation"),
        "qte": st.column_config.TextColumn("Quantite"),
        "Unite": st.column_config.TextColumn("Unite"),
        "prix_unitaire": st.column_config.TextColumn("Prix Unitaire"),
        "tva": st.column_config.TextColumn("TVA"),
        "mnt_ht": st.column_config.TextColumn("Montant Dinar"),
    },
)


# ── Totals ───────────────────────────────────────────────────────────────────

st.markdown(
    '<p class="section-header">Totaux</p>', unsafe_allow_html=True
)

t1, t2, t3, t4 = st.columns(4)
with t1:
    st.text_input("Montant HT total", key="edit_mnt_ht_total")
with t2:
    st.text_input("TVA totale", key="edit_tva_total")
with t3:
    st.text_input("Montant TTC", key="edit_mnt_ttc")
with t4:
    st.text_input("TSE", key="edit_TSE")


# ── Export ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Export")

stem = Path(st.session_state.get("source_name", "result")).stem
current_invoice = build_current_invoice(edited_line_df, st.session_state)

json_bytes = json.dumps(current_invoice, ensure_ascii=False, indent=2).encode("utf-8")
st.download_button("Télécharger JSON", json_bytes, f"{stem}_result.json", "application/json", use_container_width=True)

excel_bytes = invoice_to_excel_bytes(current_invoice)
st.download_button("Télécharger Excel", excel_bytes, f"{stem}_result.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
