"""Offline prediction extractor — extracts city, country, continent from reasoning text.

Replaces the GPT-4o-based ask_gpt() post-processing from inference_example.py.
"""

from .Geocot import GeoPrediction, extract_prediction

__all__ = ["extract_prediction", "GeoPrediction", "format_prediction"]


def format_prediction(prediction: GeoPrediction) -> str:
    """Format prediction including coordinates if available."""
    if prediction.latitude is not None and prediction.longitude is not None:
        return (f"COORDINATES: {prediction.latitude:.4f}, {prediction.longitude:.4f} | "
                f"{prediction.city}, {prediction.country}, {prediction.continent}")
    return f"{prediction.city}, {prediction.country}, {prediction.continent}"
