"""Stage 2 patient-friendly medical explanation generator for MedTranslate.

For prescriptions, it uses authoritative prescription facts and DailyMed label
information. For lab reports, it uses extracted test facts and MedlinePlus data.
Gemini rewrites retrieved information in plain language while preserving
extracted facts and enforcing strict non-diagnostic safety boundaries.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger("medtranslate.explanation_generator")

DEFAULT_GEMINI_MODEL: str = "gemini-3.6-flash"
DEFAULT_LANGUAGE: str = "English"
DEFAULT_EXPLANATION_DISCLAIMER: str = (
    "This explanation is intended to help you understand your medical document "
    "and does not replace advice from a qualified healthcare professional."
)

_PRESCRIPTION_FIELDS: tuple[str, ...] = (
    "strength",
    "dosage",
    "frequency",
    "timing",
    "duration",
)

_SYSTEM_INSTRUCTION: str = """You are a careful patient-education assistant for MedTranslate.

Your role is to explain a prescribed medicine in clear, simple, patient-friendly language using ONLY the supplied trusted medical information (DailyMed).
Strict rules:
1. Prescription facts (medicine name, strength, dosage, frequency, timing, duration) are authoritative. Never alter, infer, or invent prescription details.
2. Medical explanations must be grounded ONLY in the supplied TRUSTED MEDICAL INFORMATION. Never invent medical facts, side effects, or indications that are not present in the supplied text.
3. Preserve the original medical meaning when simplifying information. Do not exaggerate, minimize, or change the severity of warnings.
4. Preserve important qualifications, conditions, exceptions, thresholds, and limitations from the source. Never turn a conditional warning into an absolute prohibition.
5. Do not present every listed contraindication or precaution as something that necessarily applies to the individual patient.
6. Do NOT diagnose any condition.
7. Do NOT recommend starting, stopping, increasing, decreasing, or changing any medication.
8. Do NOT tell the patient to ignore their prescription or provide personalized treatment decisions.
9. If the supplied medical information does not cover a field (such as purpose, warnings, or common information), return null for purpose or an empty list [] for important_information / common_information rather than guessing.
10. Clearly distinguish general medicine information from the patient's actual prescription instructions.
11. Do not infer the patient's medical conditions, allergies, medical history, or suitability for a medicine.
12. If the supplied information is incomplete or ambiguous, preserve that uncertainty rather than filling gaps using assumptions.
13. Include only relevant warnings and precautions. Do not remove medically important context merely to make an explanation shorter.
14. Write patient-friendly explanations in the requested target language without sacrificing medical accuracy.
15. Return valid JSON matching the requested schema.
16. When simplifying medical source text, preserve the scope of the original statement. If the source gives a restriction that depends on a specific condition, measurement, threshold, or clinical circumstance, retain that dependency explicitly in the explanation rather than replacing it with a broader statement.
17. Write for a reader with no medical training: use everyday words, short sentences, and a calm, respectful tone.
18. If a medical term cannot be avoided, explain it briefly in ordinary words the first time it appears.
19. Rewrite source information in your own words. Do not copy sentences or long phrases from DailyMed; preserve their meaning and all safety qualifications.
20. Keep each explanation concise: usually one or two short sentences per item. Do not omit a warning or qualification just to make it shorter.
"""

class ExplanationGeneratorError(RuntimeError):
    """Base exception for explanation-generation failures."""


class ExplanationConfigError(ExplanationGeneratorError):
    """Raised when the Gemini SDK or API key is missing or misconfigured."""


class ExplanationInputError(ExplanationGeneratorError):
    """Raised when input prescription or medical data is invalid."""


class ExplanationAPIError(ExplanationGeneratorError):
    """Raised when the Gemini API call fails or returns unusable output."""


def _clean_optional_str(value: Any) -> str | None:
    """Normalize optional strings, turning blanks into None."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"null", "none", "n/a", "unavailable"}:
        return None
    return cleaned


class PrescribedUsage(BaseModel):
    """Authoritative prescription instructions preserved verbatim from Stage 1."""

    model_config = ConfigDict(extra="ignore")

    strength: str | None = None
    dosage: str | None = None
    frequency: str | None = None
    timing: str | None = None
    duration: str | None = None

    @field_validator("strength", "dosage", "frequency", "timing", "duration", mode="before")
    @classmethod
    def _clean_field(cls, value: Any) -> str | None:
        return _clean_optional_str(value)


class SourceReference(BaseModel):
    """Trusted medical source attribution."""

    model_config = ConfigDict(extra="ignore")

    name: str = "DailyMed"
    url: str | None = None


class GeminiExplanationPayload(BaseModel):
    """Validated JSON structure returned directly by Gemini."""

    model_config = ConfigDict(extra="ignore")

    purpose: str | None = None
    important_information: list[str] = Field(default_factory=list)
    common_information: list[str] = Field(default_factory=list)

    @field_validator("purpose", mode="before")
    @classmethod
    def _clean_purpose(cls, value: Any) -> str | None:
        return _clean_optional_str(value)

    @field_validator("important_information", "common_information", mode="before")
    @classmethod
    def _clean_str_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned_items: list[str] = []
        for item in value:
            text = _clean_optional_str(item)
            if text:
                cleaned_items.append(text)
        return cleaned_items


class MedicineExplanationResponse(BaseModel):
    """Final structured response returned by ``generate_medicine_explanation``."""

    medicine: str | None = None
    purpose: str | None = None
    how_it_was_prescribed: PrescribedUsage = Field(default_factory=PrescribedUsage)
    important_information: list[str] = Field(default_factory=list)
    common_information: list[str] = Field(default_factory=list)
    source: SourceReference = Field(default_factory=SourceReference)
    disclaimer: str = DEFAULT_EXPLANATION_DISCLAIMER
    error: str | None = None


class LabExplanationPayload(BaseModel):
    """Validated, source-grounded explanation fields returned by Gemini."""

    model_config = ConfigDict(extra="ignore")

    what_it_measures: str | None = None
    possible_meaning: str | None = None

    @field_validator("what_it_measures", "possible_meaning", mode="before")
    @classmethod
    def _clean_lab_explanation(cls, value: Any) -> str | None:
        return _clean_optional_str(value)


def _extract_prescribed_usage(medicine: dict[str, Any]) -> tuple[str | None, PrescribedUsage]:
    """Validate and normalize authoritative prescription facts from ``medicine``."""
    if not isinstance(medicine, dict):
        raise ExplanationInputError("medicine must be a dictionary.")

    med_name = _clean_optional_str(medicine.get("name") or medicine.get("medicine"))
    usage = PrescribedUsage(
        strength=medicine.get("strength"),
        dosage=medicine.get("dosage"),
        frequency=medicine.get("frequency"),
        timing=medicine.get("timing"),
        duration=medicine.get("duration"),
    )
    return med_name, usage


def _has_usable_medical_info(medical_information: Any) -> bool:
    """Return True if retrieved medical_information contains usable label content."""
    if not isinstance(medical_information, dict):
        return False
    status = str(medical_information.get("retrieval_status") or "").strip().lower()
    if status and status != "success":
        return False

    purpose_info = _clean_optional_str(medical_information.get("purpose_information"))
    important_info = medical_information.get("important_information")
    has_important = isinstance(important_info, list) and any(
        _clean_optional_str(item) for item in important_info
    )
    return bool(purpose_info or has_important)


def build_explanation_prompt(
    medicine_name: str | None,
    prescribed_usage: PrescribedUsage,
    medical_information: dict[str, Any],
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Build the tightly controlled 5-section Gemini explanation prompt."""
    prescription_lines = [
        f"- Medicine Name: {medicine_name or 'Not specified'}",
        f"- Strength: {prescribed_usage.strength or 'Not specified'}",
        f"- Dosage: {prescribed_usage.dosage or 'Not specified'}",
        f"- Frequency: {prescribed_usage.frequency or 'Not specified'}",
        f"- Timing: {prescribed_usage.timing or 'Not specified'}",
        f"- Duration: {prescribed_usage.duration or 'Not specified'}",
    ]

    source_name = _clean_optional_str(medical_information.get("source")) or "DailyMed"
    purpose_text = _clean_optional_str(medical_information.get("purpose_information")) or "Not available"
    raw_important = medical_information.get("important_information")
    important_items = (
        [str(item).strip() for item in raw_important if _clean_optional_str(item)]
        if isinstance(raw_important, list)
        else []
    )
    important_block = (
        "\n".join(f"- {item}" for item in important_items)
        if important_items
        else "- None available"
    )

    return (
        "PRESCRIPTION FACTS\n"
        "These values come directly from the user's prescription document and are authoritative:\n"
        + "\n".join(prescription_lines)
        + "\n\n"
        "TRUSTED MEDICAL INFORMATION\n"
        f"Source: {source_name}\n"
        f"Purpose / Indications:\n{purpose_text}\n\n"
        f"Warnings / Precautions / Adverse Reactions:\n{important_block}\n\n"
        "TASK\n"
        f"Write a clear, patient-friendly explanation in {language} using ONLY the TRUSTED MEDICAL INFORMATION above.\n"
        "1. `purpose`: Explain what the medicine is generally used for in everyday words, in no more than two short sentences (or null if unavailable).\n"
        "2. `important_information`: Rewrite the important warnings and precautions in everyday language. Briefly explain unavoidable medical terms. Keep each item concise, but preserve every important warning, condition, exception, threshold, and qualification (or [] if unavailable).\n"
        "3. `common_information`: Briefly explain other useful label facts, including common side effects only when they appear in the supplied source (or [] if unavailable).\n"
        "Do not copy source sentences. Explain them in your own words without changing their meaning or severity.\n\n"
        "SAFETY RULES\n"
        "- Prescription facts describe what the user's document says; DailyMed provides general medical information.\n"
        "- NEVER change, infer, or invent prescription facts.\n"
        "- NEVER invent medical facts absent from TRUSTED MEDICAL INFORMATION.\n"
        "- Do NOT diagnose, recommend starting/stopping/changing medication, or give personalized treatment advice.\n"
        "- Do NOT tell the patient to ignore their prescription.\n\n"
        "- Prefer familiar everyday words and short sentences; explain any unavoidable medical term in a few words.\n"
        "- Keep explanations short, but never remove an important safety warning or its conditions.\n"
        "- Do not add facts from memory or general knowledge that are absent from the supplied source.\n\n"
        "OUTPUT FORMAT\n"
        "Return a JSON object with keys: `purpose` (string or null), "
        "`important_information` (array of strings), and `common_information` (array of strings)."
    )


def _build_explanation_schema(types_module: Any) -> Any:
    """Build a Gemini-compatible ``types.Schema`` for Stage 2 explanations."""
    return types_module.Schema(
        type=types_module.Type.OBJECT,
        properties={
            "purpose": types_module.Schema(
                type=types_module.Type.STRING,
                description="Simple patient-friendly explanation of the medicine's general purpose, or null.",
                nullable=True,
            ),
            "important_information": types_module.Schema(
                type=types_module.Type.ARRAY,
                items=types_module.Schema(type=types_module.Type.STRING),
                description="Patient-friendly explanations of important warnings from retrieved text.",
            ),
            "common_information": types_module.Schema(
                type=types_module.Type.ARRAY,
                items=types_module.Schema(type=types_module.Type.STRING),
                description="Patient-friendly explanations of other relevant retrieved facts (e.g., common side effects).",
            ),
        },
        required=["purpose", "important_information", "common_information"],
    )


def _get_gemini_client() -> tuple[Any, Any]:
    """Import ``google-genai`` and initialize an authenticated Gemini client."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ExplanationConfigError(
            "The 'google-genai' package is not installed. Install requirements.txt first."
        ) from exc

    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ExplanationConfigError(
            "Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment."
        )

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        raise ExplanationConfigError(f"Failed to initialize Gemini client: {exc}") from exc

    return client, types


def _parse_explanation_response(response: Any) -> GeminiExplanationPayload:
    """Parse and validate the Gemini explanation JSON response."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, GeminiExplanationPayload):
            return parsed
        if isinstance(parsed, BaseModel):
            return GeminiExplanationPayload.model_validate(parsed.model_dump())
        if isinstance(parsed, dict):
            return GeminiExplanationPayload.model_validate(parsed)

    raw_text = getattr(response, "text", None)
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ExplanationAPIError("Gemini returned an empty response.")

    cleaned_text = raw_text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned_text, flags=re.DOTALL)
    if fence_match:
        cleaned_text = fence_match.group(1).strip()

    try:
        payload = json.loads(cleaned_text)
        return GeminiExplanationPayload.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ExplanationAPIError(f"Gemini returned invalid structured JSON: {exc}") from exc


def _build_source_ref(medical_information: Any) -> SourceReference:
    """Build a safe SourceReference from the retrieved metadata."""
    if not isinstance(medical_information, dict):
        return SourceReference(name="DailyMed", url=None)
    return SourceReference(
        name=_clean_optional_str(medical_information.get("source")) or "DailyMed",
        url=_clean_optional_str(medical_information.get("source_url")),
    )


def generate_medicine_explanation(
    medicine: dict[str, Any],
    medical_information: dict[str, Any],
    language: str = DEFAULT_LANGUAGE,
    *,
    client: Any | None = None,
    types_module: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate a patient-friendly medicine explanation grounded in DailyMed data.

    Args:
        medicine: Extracted prescription dictionary (``name``, ``strength``,
            ``dosage``, ``frequency``, ``timing``, ``duration``).
        medical_information: Output dictionary from
            ``backend.medical_retriever.retrieve_medicine_info``.
        language: Target language for patient-friendly explanations.
        client: Optional pre-configured ``google.genai.Client`` (for testing).
        types_module: Optional ``google.genai.types`` override (for testing).
        model: Optional Gemini model name; defaults to ``GEMINI_MODEL`` env var or
            ``gemini-3.8-flash``.

    Returns:
        Structured dictionary matching the Stage 2 explanation contract.
    """
    normalized_language = (language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
    source_ref = _build_source_ref(medical_information)

    try:
        med_name, prescribed_usage = _extract_prescribed_usage(medicine)
    except ExplanationInputError as exc:
        logger.warning("Invalid medicine input for explanation generator: %s", exc)
        return MedicineExplanationResponse(
            medicine=None,
            purpose=None,
            how_it_was_prescribed=PrescribedUsage(),
            important_information=[],
            common_information=[],
            source=source_ref,
            disclaimer=DEFAULT_EXPLANATION_DISCLAIMER,
            error=str(exc),
        ).model_dump()


    if not med_name:
        return MedicineExplanationResponse(
            medicine=None,
            purpose=None,
            how_it_was_prescribed=prescribed_usage,
            important_information=[],
            common_information=[],
            source=source_ref,
            disclaimer=DEFAULT_EXPLANATION_DISCLAIMER,
            error="Missing medicine name in prescription input.",
        ).model_dump()

    if not _has_usable_medical_info(medical_information):
        retrieval_status = (
            str(medical_information.get("retrieval_status") or "unavailable")
            if isinstance(medical_information, dict)
            else "unavailable"
        )
        return MedicineExplanationResponse(
            medicine=med_name,
            purpose=None,
            how_it_was_prescribed=prescribed_usage,
            important_information=[],
            common_information=[],
            source=source_ref,
            disclaimer=DEFAULT_EXPLANATION_DISCLAIMER,
            error=f"Trusted medical information unavailable (retrieval_status={retrieval_status}).",
        ).model_dump()

    try:
        if client is None or types_module is None:
            resolved_client, resolved_types = _get_gemini_client()
            client = client or resolved_client
            types_module = types_module or resolved_types

        prompt = build_explanation_prompt(
            medicine_name=med_name,
            prescribed_usage=prescribed_usage,
            medical_information=medical_information,
            language=normalized_language,
        )
        schema = _build_explanation_schema(types_module)
        target_model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()

        config = types_module.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=schema,
        )

        try:
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config,
            )
        except ExplanationGeneratorError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExplanationAPIError(f"Gemini API request failed: {exc}") from exc

        generated = _parse_explanation_response(response)
        return MedicineExplanationResponse(
            medicine=med_name,
            purpose=generated.purpose,
            how_it_was_prescribed=prescribed_usage,
            important_information=generated.important_information,
            common_information=generated.common_information,
            source=source_ref,
            disclaimer=DEFAULT_EXPLANATION_DISCLAIMER,
            error=None,
        ).model_dump()

    except ExplanationConfigError as exc:
        logger.error("Explanation generator configuration error: %s", exc)
        return MedicineExplanationResponse(
            medicine=med_name,
            purpose=None,
            how_it_was_prescribed=prescribed_usage,
            important_information=[],
            common_information=[],
            source=source_ref,
            disclaimer=DEFAULT_EXPLANATION_DISCLAIMER,
            error=str(exc),
        ).model_dump()
    except ExplanationAPIError as exc:
        logger.error("Explanation generator Gemini API error: %s", exc)
        return MedicineExplanationResponse(
            medicine=med_name,
            purpose=None,
            how_it_was_prescribed=prescribed_usage,
            important_information=[],
            common_information=[],
            source=source_ref,
            disclaimer=DEFAULT_EXPLANATION_DISCLAIMER,
            error=str(exc),
        ).model_dump()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in generate_medicine_explanation: %s", exc)
        return MedicineExplanationResponse(
            medicine=med_name,
            purpose=None,
            how_it_was_prescribed=prescribed_usage,
            important_information=[],
            common_information=[],
            source=source_ref,
            disclaimer=DEFAULT_EXPLANATION_DISCLAIMER,
            error="An unexpected error occurred while generating the medicine explanation.",
        ).model_dump()


def _classify_lab_result(result: Any, reference_range: Any, unit: Any = None) -> str | None:
    """Compare simple numeric results with a two-number printed reference range.

    Non-numeric, inequality-based, qualitative, or ambiguous ranges are left
    unclassified rather than guessed. The original extracted strings are never
    changed by this helper.
    """
    result_text = _clean_optional_str(result)
    range_text = _clean_optional_str(reference_range)
    if not result_text or not range_text:
        return None

    result_match = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z/%µμ]+(?:\s*/\s*[a-zA-Z]+)?)?\s*",
        result_text,
    )
    range_match = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
        r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z/%µμ]+(?:\s*/\s*[a-zA-Z]+)?)?\s*",
        range_text,
        flags=re.IGNORECASE,
    )
    if not result_match or not range_match:
        return None

    def normalize_unit(value: Any) -> str | None:
        cleaned = _clean_optional_str(value)
        return re.sub(r"\s+", "", cleaned).lower() if cleaned else None

    result_unit = normalize_unit(result_match.group(2))
    report_unit = normalize_unit(unit)
    range_unit = normalize_unit(range_match.group(3))
    if result_unit and report_unit and result_unit != report_unit:
        return None
    effective_result_unit = report_unit or result_unit
    if effective_result_unit and range_unit and effective_result_unit != range_unit:
        return None

    value = float(result_match.group(1))
    lower = float(range_match.group(1))
    upper = float(range_match.group(2))
    if lower > upper:
        return None
    if value < lower:
        return "Below reference range"
    if value > upper:
        return "Above reference range"
    return "Within reference range"


def _parse_lab_explanation_response(response: Any) -> LabExplanationPayload:
    """Parse Gemini's structured lab explanation response."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, LabExplanationPayload):
        return parsed
    if isinstance(parsed, BaseModel):
        return LabExplanationPayload.model_validate(parsed.model_dump())
    if isinstance(parsed, dict):
        return LabExplanationPayload.model_validate(parsed)

    raw_text = getattr(response, "text", None)
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ExplanationAPIError("Gemini returned an empty lab explanation.")
    cleaned = raw_text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        return LabExplanationPayload.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ExplanationAPIError(f"Gemini returned invalid lab explanation JSON: {exc}") from exc


def generate_lab_test_explanation(
    test: dict[str, Any],
    medical_information: dict[str, Any] | None,
    language: str = DEFAULT_LANGUAGE,
    *,
    client: Any | None = None,
    types_module: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Explain a lab test without changing its extracted name, value, or range.

    Interpretation is limited to a deterministic comparison against the
    reference range printed on the report. General test information and
    possible meanings are generated only from retrieved MedlinePlus content.
    """
    if not isinstance(test, dict):
        raise ExplanationInputError("test must be a dictionary.")

    original_result = test.get("result")
    original_range = test.get("reference_range")
    status = _classify_lab_result(original_result, original_range, test.get("unit"))
    source_url = None
    source_text = None
    source_name = None
    loinc_display = None
    if isinstance(medical_information, dict):
        if str(medical_information.get("retrieval_status") or "").lower() == "success":
            source_text = _clean_optional_str(medical_information.get("medical_information"))
            source_url = _clean_optional_str(medical_information.get("source_url"))
            source_name = _clean_optional_str(medical_information.get("source"))
            loinc_display = _clean_optional_str(medical_information.get("loinc_display"))

    generated = LabExplanationPayload()
    if source_text:
        try:
            if client is None or types_module is None:
                resolved_client, resolved_types = _get_gemini_client()
                client = client or resolved_client
                types_module = types_module or resolved_types

            prompt = (
                "You are explaining one laboratory test to a person with no medical training.\n"
                f"Target language: {language}.\n"
                f"Test name from the report: {test.get('name') or 'Not specified'}\n"
                f"Standard test label: {loinc_display or 'Not available'}\n"
                f"Trusted MedlinePlus information: {source_text}\n\n"
                "Return two concise fields. `what_it_measures` must explain in everyday words "
                "what this test measures, defining unavoidable medical terms briefly. "
                "`possible_meaning` may mention general reasons or associations only when "
                "they are explicitly supported by the supplied MedlinePlus text. Phrase these "
                "as possibilities, never as the patient's cause. If the source does not "
                "support an explanation, return null for that field.\n"
                "Do not diagnose, infer patient history, recommend treatment, or invent causes. "
                "Do not repeat or alter the patient's result or reference range. Use simple "
                "language, one or two short sentences per field, and do not copy source wording."
            )
            schema = types_module.Schema(
                type=types_module.Type.OBJECT,
                properties={
                    "what_it_measures": types_module.Schema(
                        type=types_module.Type.STRING,
                        description="Plain-language description of what the test measures, or null.",
                        nullable=True,
                    ),
                    "possible_meaning": types_module.Schema(
                        type=types_module.Type.STRING,
                        description="Source-supported general associations only, or null.",
                        nullable=True,
                    ),
                },
                required=["what_it_measures", "possible_meaning"],
            )
            target_model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
            config = types_module.GenerateContentConfig(
                system_instruction=(
                    "Use only the supplied MedlinePlus information. Explain it simply and briefly. "
                    "Never diagnose or claim that a possible association is the cause for this patient."
                ),
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=schema,
            )
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config,
            )
            generated = _parse_lab_explanation_response(response)
        except Exception as exc:  # noqa: BLE001 - preserve extracted result if explanation fails
            logger.warning("Lab explanation generation failed for '%s': %s", test.get("name"), exc)

    if status == "Within reference range":
        explanation = (
            "This result is within the reference range printed on the report. "
            "A reference range is a guide and does not, by itself, confirm that everything is healthy."
        )
    elif status in {"Below reference range", "Above reference range"}:
        direction = "below" if status.startswith("Below") else "above"
        explanation = f"This result is {direction} the reference range printed on the report."
        if generated.possible_meaning:
            explanation += f" The MedlinePlus information notes that this can sometimes be associated with: {generated.possible_meaning}"
        else:
            explanation += " The available reference information does not establish why it is outside the range."
        explanation += " This result alone cannot identify a cause or establish a diagnosis."
    else:
        explanation = (
            "The result could not be compared reliably with the printed reference range. "
            "This result alone cannot establish a diagnosis."
        )

    return {
        "status": status,
        "what_it_measures": generated.what_it_measures,
        "explanation": explanation,
        "source": (
            {"name": source_name or "MedlinePlus.gov", "url": source_url}
            if source_text
            else None
        ),
    }


def run_structural_tests() -> bool:
    """Run offline structural tests using synthetic Metformin data (no live API needed)."""
    import types as pytypes

    sample_medicine = {
        "name": "Metformin",
        "strength": "500 mg",
        "dosage": "1 tablet",
        "frequency": "twice daily",
        "timing": "after meals",
        "duration": "5 days",
    }
    sample_medical_info = {
        "medicine": "Metformin",
        "source": "DailyMed",
        "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=06ade2e1-5981-4e3d-b7e9-7ff083eb4551",
        "retrieval_status": "success",
        "purpose_information": (
            "Metformin Hydrochloride Tablets USP are indicated as an adjunct to diet and "
            "exercise to improve glycemic control in adults and children with type 2 diabetes mellitus."
        ),
        "important_information": [
            "Warnings: Lactic acidosis is a rare, serious metabolic complication.",
            "Adverse Reactions: Common side effects include diarrhea, nausea/vomiting, and flatulence.",
        ],
    }

    class FakeTypes:
        class Type:
            OBJECT = "OBJECT"
            STRING = "STRING"
            ARRAY = "ARRAY"

        class Schema:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        class GenerateContentConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

    class FakeModels:
        def __init__(self, payload: dict[str, Any] | None = None, raise_exc: Exception | None = None) -> None:
            self.payload = payload
            self.raise_exc = raise_exc

        def generate_content(self, model: str, contents: Any, config: Any) -> Any:
            if self.raise_exc is not None:
                raise self.raise_exc
            return pytypes.SimpleNamespace(text=json.dumps(self.payload))

    class FakeClient:
        def __init__(self, payload: dict[str, Any] | None = None, raise_exc: Exception | None = None) -> None:
            self.models = FakeModels(payload=payload, raise_exc=raise_exc)

    # 1. Prompt construction check
    _, usage = _extract_prescribed_usage(sample_medicine)
    prompt = build_explanation_prompt("Metformin", usage, sample_medical_info, "English")
    for section in ("PRESCRIPTION FACTS", "TRUSTED MEDICAL INFORMATION", "TASK", "SAFETY RULES", "OUTPUT FORMAT"):
        assert section in prompt, f"Missing prompt section: {section}"
    assert "500 mg" in prompt and "after meals" in prompt

    # 2. Successful mocked generation & authoritative prescription preservation
    mock_client = FakeClient(
        payload={
            "purpose": "Metformin is used with diet and exercise to help control blood sugar in type 2 diabetes.",
            "important_information": [
                "A rare but serious condition called lactic acidosis can occur; discuss risk factors with your doctor."
            ],
            "common_information": [
                "Common side effects reported in studies include diarrhea, nausea or vomiting, and gas."
            ],
        }
    )
    res_ok = generate_medicine_explanation(
        sample_medicine,
        sample_medical_info,
        language="English",
        client=mock_client,
        types_module=FakeTypes,
    )
    assert res_ok["medicine"] == "Metformin"
    assert res_ok["how_it_was_prescribed"] == {
        "strength": "500 mg",
        "dosage": "1 tablet",
        "frequency": "twice daily",
        "timing": "after meals",
        "duration": "5 days",
    }
    assert res_ok["source"]["name"] == "DailyMed"
    assert res_ok["error"] is None
    print("Structural test 1 (valid explanation & prescription preservation) -> PASS")

    # 3. Missing prescription fields -> preserved as None
    partial_med = {"name": "Metformin", "strength": "500 mg", "dosage": "", "frequency": None}
    res_partial = generate_medicine_explanation(
        partial_med,
        sample_medical_info,
        client=mock_client,
        types_module=FakeTypes,
    )
    assert res_partial["how_it_was_prescribed"] == {
        "strength": "500 mg",
        "dosage": None,
        "frequency": None,
        "timing": None,
        "duration": None,
    }
    print("Structural test 2 (missing prescription fields -> null)           -> PASS")

    # 4. Retrieval failure handling
    res_not_found = generate_medicine_explanation(
        sample_medicine,
        {"medicine": "Metformin", "source": "DailyMed", "source_url": None, "retrieval_status": "not_found"},
    )
    assert res_not_found["purpose"] is None
    assert res_not_found["important_information"] == []
    assert res_not_found["error"] is not None
    print("Structural test 3 (retrieval failure handled gracefully)         -> PASS")

    # 5. Invalid medicine input handling
    res_bad_input = generate_medicine_explanation("not-a-dict", sample_medical_info)  # type: ignore[arg-type]
    assert res_bad_input["medicine"] is None
    assert res_bad_input["error"] is not None
    print("Structural test 4 (invalid input handled gracefully)             -> PASS")

    # 6. Gemini API error handling
    err_client = FakeClient(raise_exc=RuntimeError("503 UNAVAILABLE"))
    res_api_err = generate_medicine_explanation(
        sample_medicine,
        sample_medical_info,
        client=err_client,
        types_module=FakeTypes,
    )
    assert res_api_err["medicine"] == "Metformin"
    assert res_api_err["how_it_was_prescribed"]["strength"] == "500 mg"
    assert res_api_err["error"] is not None
    print("Structural test 5 (Gemini API error handled gracefully)          -> PASS")

    return True


if __name__ == "__main__":
    raise SystemExit(0 if run_structural_tests() else 1)
