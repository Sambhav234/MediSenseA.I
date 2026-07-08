"""
MediSense AI v3 — Multi-Modal Fusion Service
=============================================
Combines symptom-based (XGBoost) + image-based (CNN) predictions.
Rule: If CNN finding matches a disease in the text predictions → boost its score.
"""
import logging
log = logging.getLogger("medisense.fusion")

# Mapping: CNN finding → diseases it should boost in text predictions
CNN_DISEASE_BOOST: dict[str, list[str]] = {
    "PNEUMONIA": ["Pneumonia", "Tuberculosis", "Bronchial Asthma"],
    "NORMAL":    [],
    "rash":      ["Fungal infection", "Allergy", "Drug Reaction",
                  "Chicken pox", "Psoriasis", "Impetigo"],
    "acne":      ["Acne"],
    "fungal_infection": ["Fungal infection", "Impetigo"],
    "normal_skin": [],
}

BOOST_FACTOR = 1.35   # multiply confidence by this if CNN confirms


def fuse(
    text_predictions: list[dict],
    image_result: dict | None,
    image_type: str = "xray",   # "xray" | "skin"
) -> dict:
    """
    Merge text predictions with image result.

    Returns:
        {
          "predictions":   list[dict],   — boosted and re-ranked
          "image_result":  dict | None,
          "fusion_applied": bool,
          "fusion_note":   str,
        }
    """
    if not image_result or not image_result.get("loaded"):
        return {
            "predictions":    text_predictions,
            "image_result":   image_result,
            "fusion_applied": False,
            "fusion_note":    "Image result unavailable — using text predictions only.",
        }

    finding = image_result.get("finding")
    if not finding:
        return {
            "predictions":    text_predictions,
            "image_result":   image_result,
            "fusion_applied": False,
            "fusion_note":    "No image finding — using text predictions only.",
        }

    boosted_diseases = CNN_DISEASE_BOOST.get(finding, [])
    if not boosted_diseases:
        return {
            "predictions":    text_predictions,
            "image_result":   image_result,
            "fusion_applied": False,
            "fusion_note":    f"CNN found '{finding}' — no boost applicable.",
        }

    # Apply boost
    preds = []
    boost_applied = False
    for p in text_predictions:
        entry = dict(p)
        if entry["disease"] in boosted_diseases:
            old_prob = entry["probability"]
            new_prob = min(old_prob * BOOST_FACTOR, 99.0)
            entry["probability"]    = round(new_prob, 1)
            entry["cnn_boosted"]    = True
            entry["boost_reason"]   = (
                f"CNN ({image_type}) confirmed '{finding}' "
                f"→ confidence boosted {old_prob}% → {round(new_prob,1)}%"
            )
            boost_applied = True
            log.info(f"Boosted '{entry['disease']}': {old_prob}% → {new_prob}%")
        preds.append(entry)

    # Re-sort by boosted probability
    preds.sort(key=lambda x: x["probability"], reverse=True)

    note = (
        f"CNN ({image_type}) detected '{finding}' — "
        f"{'confidence boosted for matching diseases.' if boost_applied else 'no matching diseases to boost.'}"
    )
    return {
        "predictions":    preds,
        "image_result":   image_result,
        "fusion_applied": boost_applied,
        "fusion_note":    note,
    }
