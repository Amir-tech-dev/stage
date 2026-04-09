"""
PDF to images conversion using PyMuPDF.
"""

from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def convert_pdf_to_images(pdf_path, output_dir):
    """Convert PDF pages to PNG images at 2x resolution.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory to save the output PNG images.

    Returns:
        List of paths to the generated PNG images.
    """
    if fitz is None:
        raise RuntimeError("PDF support requires PyMuPDF. Install pymupdf.")

    image_paths = []
    doc = fitz.open(pdf_path)
    try:
        for idx in range(doc.page_count):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            out_path = output_dir / f"page_{idx + 1}.png"
            pix.save(str(out_path))
            image_paths.append(out_path)
    finally:
        doc.close()

    if not image_paths:
        raise RuntimeError("No pages found in uploaded PDF.")

    return image_paths
