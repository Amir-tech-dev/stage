"""
Shared constants, field definitions, and JSON schema for Naftal invoice OCR.
Single source of truth — imported by all modules.
"""

import os

# ── Invoice field definitions ────────────────────────────────────────────────

LINE_COLUMNS = ["designation", "qte", "Unite", "prix_unitaire", "tva", "mnt_ht"]

NUMERIC_LINE_FIELDS = {"qte", "prix_unitaire", "mnt_ht"}
NUMERIC_TOTAL_FIELDS = {"mnt_ht_total", "tva_total", "mnt_ttc", "TSE"}

TOP_FIELDS = [
    "code",
    "date_etablisement",
    "date_echeance",
    "date_debut_livraison",
    "date_fin_livraison",
    "livre_par",
    "num_compte",
    "type_de_paiement",
    "TSE",
]

TOTAL_FIELDS = ["mnt_ht_total", "tva_total", "mnt_ttc"]

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ── Expected JSON schema (sent to LLMs) ─────────────────────────────────────

EXPECTED_JSON_SCHEMA = {
    "code": None,
    "date_etablisement": None,
    "date_echeance": None,
    "date_debut_livraison": None,
    "date_fin_livraison": None,
    "livre_par": None,
    "num_compte": None,
    "type_de_paiement": None,
    "facture_ligne": [
        {
            "designation": "Designation",
            "qte": "Quantite",
            "Unite": "Unite",
            "prix_unitaire": "Prix Unitaire",
            "tva": "TVA %",
            "mnt_ht": "Montant Dinar",
        }
    ],
    "mnt_ht_total": None,
    "tva_total": None,
    "mnt_ttc": None,
    "TSE": None,
}

# ── OCR tuning (PPStructureV3 server models) ─────────────────────────────────

TEXT_DET_LIMIT_SIDE_LEN = 1536
TEXT_DET_THRESH = 0.22
TEXT_DET_BOX_THRESH = 0.45
TEXT_DET_UNCLIP_RATIO = 2.0
TEXT_REC_SCORE_THRESH = 0.35
CPU_THREADS = min(16, os.cpu_count() or 8)
MKLDNN_CACHE_CAPACITY = 20
