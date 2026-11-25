"""Flask backend and integration layer for MedTranslate."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from backend.document_processor import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    DocumentValidationError,
    get_file_extension,
    process_uploaded_file,
    validate_file_extension,
)

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("medtranslate.app")

UPLOAD_FOLDER = "uploads"
DEFAULT_DISCLAIMER = (
    "MedTranslate provides educational explanations of your medical documents "
    "and is not a substitute for professional medical advice or diagnosis."
)


def _normalize_response_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Preserve the AI pipeline metadata while normalizing the UI contract."""
    if not isinstance(payload, dict):
        raise ValueError("AI pipeline must return a dictionary.")

    doc_type = payload.get("document_type", "unknown")
    if doc_type not in {"prescription", "lab_report", "unknown"}:
        doc_type = "unknown"

    return {
        "document_type": doc_type,
        "summary": payload.get("summary") or "",
        "medicines": payload.get("medicines") if isinstance(payload.get("medicines"), list) else [],
        "tests": payload.get("tests") if isinstance(payload.get("tests"), list) else [],
        "questions_for_doctor": (
            payload.get("questions_for_doctor")
            if isinstance(payload.get("questions_for_doctor"), list)
            else []
        ),
        "disclaimer": payload.get("disclaimer") or DEFAULT_DISCLAIMER,
        "stage": payload.get("stage") or "analysis",
        "input_mode": payload.get("input_mode"),
        "language": payload.get("language"),
        "error": payload.get("error"),
        "demo": bool(payload.get("demo", False)),
    }


def _enrich_prescription_medicines(
    medicines: list[dict[str, Any]],
    language: str,
) -> list[dict[str, Any]]:
    """Enrich extracted prescription medicines with DailyMed retrieval and explanations."""
    try:
        from backend.explanation_generator import generate_medicine_explanation
        from backend.medical_retriever import retrieve_medicine_info
    except ImportError:
        logger.info("Retrieval/explanation modules unavailable; keeping Stage 1 medicines.")
        return medicines

    enriched: list[dict[str, Any]] = []
    for med in medicines:
        if not isinstance(med, dict):
            continue

        # Authoritative prescription facts from Stage 1 (never overwritten by Stage 2)
        authoritative_med: dict[str, Any] = {
            "name": med.get("name"),
            "strength": med.get("strength"),
            "dosage": med.get("dosage"),
            "frequency": med.get("frequency"),
            "timing": med.get("timing"),
            "duration": med.get("duration"),
        }

        med_name = (authoritative_med.get("name") or "").strip() if isinstance(authoritative_med.get("name"), str) else ""
        if not med_name:
            enriched.append(
                {
                    **authoritative_med,
                    "purpose": None,
                    "important_information": [],
                    "common_information": [],
                    "source": {"name": "DailyMed", "url": None},
                }
            )
            continue

        try:
            medical_info = retrieve_medicine_info(med_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DailyMed retrieval failed for '%s': %s", med_name, exc)
            medical_info = {
                "medicine": med_name,
                "source": "DailyMed",
                "source_url": None,
                "retrieval_status": "error",
                "purpose_information": None,
                "important_information": [],
            }

        try:
            explanation = generate_medicine_explanation(
                authoritative_med,
                medical_info,
                language=language,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Explanation generation failed for '%s': %s", med_name, exc)
            explanation = {"error": "Explanation generation failed."}

        source = explanation.get("source")
        if not isinstance(source, dict):
            source = {
                "name": medical_info.get("source") or "DailyMed",
                "url": medical_info.get("source_url"),
            }

        if explanation.get("error"):
            enriched.append(
                {
                    **authoritative_med,
                    "purpose": None,
                    "important_information": [],
                    "common_information": [],
                    "source": source,
                    "error": "Patient-friendly explanation temporarily unavailable.",
                }
            )
        else:
            enriched.append(
                {
                    **authoritative_med,
                    "purpose": explanation.get("purpose"),
                    "important_information": (
                        explanation.get("important_information")
                        if isinstance(explanation.get("important_information"), list)
                        else []
                    ),
                    "common_information": (
                        explanation.get("common_information")
                        if isinstance(explanation.get("common_information"), list)
                        else []
                    ),
                    "source": source,
                }
            )

    return enriched


def _enrich_lab_tests(tests: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    """Add source-grounded explanations without changing extracted lab facts."""
    try:
        from backend.explanation_generator import generate_lab_test_explanation
        from backend.medical_retriever import retrieve_lab_test_info
    except ImportError:
        logger.info("Lab retrieval/explanation modules unavailable; keeping Stage 1 tests.")
        return tests

    enriched: list[dict[str, Any]] = []
    for test in tests:
        if not isinstance(test, dict):
            continue

        # Keep all Stage 1 fields exactly as extracted from the uploaded report.
        authoritative_test = dict(test)
        test_name = test.get("name") if isinstance(test.get("name"), str) else ""
        try:
            medical_info = retrieve_lab_test_info(test_name)
        except Exception as exc:  # noqa: BLE001 - explanation must not break extraction
            logger.warning("MedlinePlus lookup failed for lab test '%s': %s", test_name, exc)
            medical_info = None

        try:
            explanation = generate_lab_test_explanation(
                authoritative_test,
                medical_info,
                language=language,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lab explanation failed for '%s': %s", test_name, exc)
            explanation = {
                "status": None,
                "what_it_measures": None,
                "explanation": (
                    "An explanation is unavailable. The extracted result and reference range "
                    "are shown as printed; this result alone cannot establish a diagnosis."
                ),
                "source": None,
            }

        authoritative_test.update(explanation)
        enriched.append(authoritative_test)

    return enriched


def _invoke_ai_pipeline(processed_doc: dict[str, Any], language: str) -> dict[str, Any]:
    """Run Gemini extraction, then add source-grounded prescription or lab context."""
    try:
        from backend.ai_processor import run_medtranslate_pipeline
    except ImportError:
        logger.exception("The AI processor module is missing.")
        return {
            "document_type": "unknown",
            "summary": "AI analysis is unavailable because the AI processor is not installed.",
            "medicines": [],
            "tests": [],
            "questions_for_doctor": [],
            "disclaimer": DEFAULT_DISCLAIMER,
            "stage": "unavailable",
            "error": "AI processor module is unavailable. Restore backend/ai_processor.py.",
            "demo": False,
        }

    try:
        result = run_medtranslate_pipeline(processed_doc, language)
        normalized = _normalize_response_contract(result)
        # Do not attempt enrichment if extraction itself failed.
        if not normalized.get("error") and normalized["document_type"] == "prescription" and normalized["medicines"]:
            normalized["medicines"] = _enrich_prescription_medicines(
                normalized["medicines"],
                language=language,
            )
            normalized["stage"] = "extraction_and_medicine_explanation"
        elif not normalized.get("error") and normalized["document_type"] == "lab_report" and normalized["tests"]:
            normalized["tests"] = _enrich_lab_tests(
                normalized["tests"],
                language=language,
            )
            normalized["stage"] = "extraction_and_lab_explanation"
        return normalized
    except Exception as exc:  # noqa: BLE001
        logger.exception("AI analysis failed: %s", exc)
        return {
            "document_type": "unknown",
            "summary": "The document could not be analyzed. Please try again with a clearer file.",
            "medicines": [],
            "tests": [],
            "questions_for_doctor": [],
            "disclaimer": DEFAULT_DISCLAIMER,
            "stage": "failed",
            "error": "AI analysis failed. Check the server configuration and try again.",
            "demo": False,
        }


def _build_safe_upload_path(upload_folder: str, original_filename: str) -> tuple[str, str]:
    """Create a collision-free temporary path inside uploads/ preserving extension."""
    ext = validate_file_extension(original_filename)
    sanitized = secure_filename(original_filename)
    if not sanitized or get_file_extension(sanitized) != ext:
        sanitized = f"document{ext}"

    unique_name = f"{uuid.uuid4().hex}_{sanitized}"
    return os.path.join(upload_folder, unique_name), sanitized


def create_app() -> Flask:
    """Create and configure the MedTranslate Flask application."""
    app = Flask(__name__)
    CORS(app)

    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_exc: RequestEntityTooLarge):
        return (
            jsonify({"error": f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB."}),
            400,
        )

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def landing():
        return render_template(
            "landing.html",
            github_url=os.getenv("GITHUB_URL", "https://github.com/bharrathkumaran2026-commits/Medtranslate"),
        )

    @app.route("/login", methods=["GET", "POST"])
    def login_page():
        # Lightweight demo entry only: validate that fields are filled, but do not
        # authenticate, store, log, or persist credentials.
        if request.method == "POST":
            email = (request.form.get("email") or "").strip()
            password = request.form.get("password") or ""
            if not email or not password:
                return render_template(
                    "login.html", error="Enter both your email address and password to continue."
                ), 400
            if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
                return render_template(
                    "login.html", error="Enter a valid email address."
                ), 400
            return redirect(url_for("dashboard"))
        return render_template("login.html")

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html", max_file_mb=MAX_FILE_SIZE_MB)

    @app.get("/api/health")
    def health_check():
        return jsonify({
            "status": "ok",
            "service": "MedTranslate Backend",
            "ai_configured": bool((os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()),
            "medicine_lookup": "DailyMed",
        }), 200

    @app.post("/api/analyze")
    def analyze_document():
        if "file" not in request.files:
            return jsonify({"error": "No file part in request. Expected form field 'file'."}), 400

        uploaded_file = request.files["file"]
        original_filename = (uploaded_file.filename or "").strip()
        if not original_filename:
            return jsonify({"error": "No file selected for upload."}), 400

        language = (request.form.get("language") or "English").strip() or "English"
        temp_filepath: str | None = None

        try:
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            temp_filepath, safe_name = _build_safe_upload_path(
                app.config["UPLOAD_FOLDER"], original_filename
            )
            uploaded_file.save(temp_filepath)

            processed_doc = process_uploaded_file(temp_filepath, safe_name)
            result = _invoke_ai_pipeline(processed_doc, language)
            return jsonify(result), 200

        except DocumentValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error analyzing uploaded document: %s", exc)
            return (
                jsonify({"error": "An internal error occurred while analyzing the document."}),
                500,
            )
        finally:
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError as cleanup_exc:
                    logger.warning(
                        "Failed to remove temporary file %s: %s", temp_filepath, cleanup_exc
                    )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
