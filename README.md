# Naftal Invoice OCR

Naftal Invoice OCR is a Streamlit application that extracts structured invoice data from PDF or image files.

It supports two AI pipelines:
- OSS 120B pipeline: PPStructureV3 OCR -> text extraction -> Ollama (GPT-OSS) -> JSON
- Llama 4 Scout pipeline: image(s) -> Groq vision model -> JSON

## Features

- Upload PDF or image invoices
- Choose extraction backend (OSS 120B or Llama 4 Scout)
- Automatic schema-based JSON extraction
- Editable invoice fields and line items in the UI
- Export results to JSON and Excel
- Sidebar metrics (timings and token usage)

## Project structure

- app.py: Main Streamlit application
- models/: LLM backends
  - models/oss120b.py
  - models/llama4.py
- ocr/: OCR pipeline and constants
  - ocr/ppstructure.py
  - ocr/constants.py
- utils/: Shared business and conversion utilities
  - utils/invoice.py
  - utils/pdf.py
- final/: Generated run outputs (images, OCR JSON, extracted results)

## Requirements

- Python 3.10+
- pip
- Ollama (for OSS 120B backend)
- Groq API key (for Llama 4 backend)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment variables

### For Llama 4 Scout (Groq)

Set your Groq API key before running the app.

PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

### For OSS 120B (Ollama)

Default model name used by the app:

- gpt-oss:120b-cloud

Optional override:

PowerShell:

```powershell
$env:OLLAMA_MODEL="gpt-oss:120b-cloud"
```

Make sure Ollama is installed, running, and the selected model is available.

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## Usage

1. Select the model in the UI.
2. Upload a PDF or image invoice.
3. Click extraction.
4. Review and edit extracted header/line fields.
5. Download JSON or Excel output.

## Output artifacts

Each run creates a timestamped folder in final/:

- images/: Input pages or converted PDF pages
- ocr_json/: OCR raw output (OSS pipeline)

## Troubleshooting

- "Set GROQ_API_KEY env variable": define GROQ_API_KEY before launch.
- OCR errors: verify paddleocr and paddlepaddle installation.
- PDF conversion errors: verify PyMuPDF is installed.
- Ollama extraction issues: ensure Ollama is running and the model is available.
