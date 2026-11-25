# MedTranslate

MedTranslate provides educational explanations of printed prescriptions and laboratory reports. It uses a Flask web UI, Gemini structured extraction, DailyMed medicine-label retrieval, and MedlinePlus lab-test information where a reliable match is available.

## Run on Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your private Google Gemini API key:

```dotenv
GEMINI_API_KEY=your_private_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

Then run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000.

## AI pipeline

1. `backend/document_processor.py` validates PDF/JPG/JPEG/PNG uploads (10 MB maximum), extracts selectable PDF text, and prepares scanned PDFs/images for multimodal processing.
2. `backend/ai_processor.py` uses Gemini to classify the document as `prescription`, `lab_report`, or `unknown` and extract only explicitly written fields.
3. For prescription medicines, `backend/medical_retriever.py` retrieves public medicine-label information from DailyMed.
4. For laboratory tests, the same module attempts a conservative LOINC match and retrieves matching patient information from MedlinePlus Connect. Ambiguous or unavailable matches are not guessed.
5. `backend/explanation_generator.py` rewrites medicine information in plain language and explains lab tests using retrieved source information. Prescription instructions and extracted lab values/ranges remain unchanged.
6. The dashboard shows the report's values and range, a comparison when the numeric format can be safely interpreted, and source links when available.

## API

- `GET /api/health` — reports service health and whether a Gemini key is configured (never returns the key).
- `POST /api/analyze` — multipart form fields: `file` and optional `language`.

Supported uploads: PDF, PNG, JPG, JPEG. Uploads are processed temporarily and deleted after the request.

## Pages

| Route | What |
|---|---|
| `/` | Landing page |
| `/login` | Visual-only sign-in demonstration (no accounts or authentication) |
| `/dashboard` | Document upload and AI analysis |

## Safety and privacy

- Educational information only; not diagnosis, treatment or prescribing.
- AI extracts only text explicitly present in the document. Verify all fields against the original.
- Lab status is only compared against the reference range printed on the report; it is not a diagnosis or a claim that a person is healthy or unwell. Non-numeric or ambiguous ranges are left unclassified.
- Possible meanings are included only when supported by retrieved MedlinePlus information. A lab result alone cannot establish a diagnosis or cause.
- Medicine context is educational and linked to DailyMed when a source is available.
- Never commit `.env`, API keys, or patient documents. Use synthetic documents for development.
- The Flask development server is for local testing, not public production deployment.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```


*Built as an educational project exploring artificial intelligence, document processing, and healthcare information accessibility.*
