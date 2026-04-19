"""Human Delta API client for structuring raw weather logs into wind parameters.

Converts unformatted text (e.g. field notes, weather station logs) into
validated physics parameters consumed by the OpenDrift drift simulator.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

_STRUCTURE_ENDPOINT = "https://api.humandelta.ai/v1/structure"

_WIND_INSTRUCTIONS = (
    "Extract the following wind parameters from the text. "
    "Return only values that are explicitly stated or can be directly inferred. "
    "wind_speed_ms: wind speed in metres per second (convert from knots if needed: 1 kt = 0.514 m/s). "
    "wind_dir_deg: meteorological FROM-direction in degrees clockwise from North (0-360). "
    "wind_drift_factor: fraction of wind speed applied as leeway to floating debris (0.02-0.04 typical). "
    "confidence_score: your confidence in the extraction (0-1)."
)

_SAFE_FALLBACK: dict[str, float] = {
    "wind_speed_ms": 0.0,
    "wind_dir_deg": 0.0,
    "wind_drift_factor": 0.0,
    "confidence_score": 0.0,
}


class WindDataSchema(BaseModel):
    wind_speed_ms: float = Field(..., ge=0, description="Wind speed in m/s")
    wind_dir_deg: float = Field(..., ge=0, le=360, description="FROM-direction, degrees CW from North")
    wind_drift_factor: float = Field(default=0.03, ge=0, le=0.1, description="Leeway fraction of wind speed")
    confidence_score: float = Field(..., ge=0, le=1, description="Extraction confidence 0-1")


class HumanDeltaClient:
    """Thin async-capable client wrapping the Human Delta structuring API."""

    def __init__(self) -> None:
        api_key = os.getenv("HUMANDELTA_API_KEY")
        if not api_key:
            raise ValueError(
                "HUMANDELTA_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def structure_wind_data(self, unstructured_text: str) -> dict[str, Any]:
        """Call Human Delta to extract wind parameters from free-text weather notes.

        Returns a dict matching WindDataSchema fields, or a zero-valued fallback
        if the request fails or the response fails validation.
        """
        payload = {
            "text": unstructured_text,
            "instructions": _WIND_INSTRUCTIONS,
            "output_schema": WindDataSchema.model_json_schema(),
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(
                    _STRUCTURE_ENDPOINT,
                    headers=self._headers,
                    json=payload,
                )
                response.raise_for_status()
                raw: dict[str, Any] = response.json()
        except httpx.HTTPError as exc:
            print(f"[wind_enrichment] HTTP error contacting Human Delta API: {exc}")
            return dict(_SAFE_FALLBACK)

        try:
            validated = WindDataSchema.model_validate(raw)
        except ValidationError as exc:
            print(f"[wind_enrichment] Response failed schema validation: {exc}")
            return dict(_SAFE_FALLBACK)

        return validated.model_dump()
