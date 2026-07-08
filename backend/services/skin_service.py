"""
MediSense AI v3 — Skin / Rash Detection Service
=================================================
Uses MobileNetV2 pretrained features + lightweight heuristic classifier.
No training required — zero-shot skin condition classification.
Classes: rash, acne, fungal_infection, normal_skin

For a production system, fine-tune on a labelled skin dataset
(e.g., HAM10000, ISIC 2019, DermNet).
"""
import io, os, logging
from urllib.parse import quote_plus

import numpy as np

log = logging.getLogger("medisense.skin")

SKIN_CLASSES = ["rash", "acne", "fungal_infection", "normal_skin"]

SKIN_ADVICE = {
    "rash": {
        "description": "Skin rash detected. Could indicate allergy, contact dermatitis, or viral infection.",
        "precautions": [
            "Avoid scratching the affected area",
            "Apply calamine lotion or mild corticosteroid cream",
            "Identify and avoid potential allergens",
            "See a dermatologist if the rash spreads or worsens",
        ],
        "specialization": "Dermatologist",
    },
    "acne": {
        "description": "Acne-like lesions detected. Common inflammatory skin condition.",
        "precautions": [
            "Keep skin clean — wash twice daily with mild cleanser",
            "Avoid touching or popping pimples",
            "Use oil-free, non-comedogenic moisturiser",
            "Consider topical retinoids or benzoyl peroxide",
            "See a dermatologist for persistent or severe acne",
        ],
        "specialization": "Dermatologist",
    },
    "fungal_infection": {
        "description": "Possible fungal skin infection detected (e.g., ringworm, tinea).",
        "precautions": [
            "Apply antifungal cream (clotrimazole, miconazole) as directed",
            "Keep the affected area clean and dry",
            "Avoid sharing towels or clothing",
            "See a dermatologist if no improvement in 2 weeks",
        ],
        "specialization": "Dermatologist",
    },
    "normal_skin": {
        "description": "No significant skin condition detected in this image.",
        "precautions": [
            "Maintain regular skin hygiene",
            "Use sunscreen daily",
            "Stay hydrated",
            "See a dermatologist for annual skin checks",
        ],
        "specialization": "Dermatologist",
    },
}


class SkinService:
    """
    Lightweight skin condition classifier using MobileNetV2 feature extraction.
    Relies on colour and texture statistics from pretrained features as a proxy
    classifier — good enough for demo; replace with fine-tuned model for production.
    """

    def __init__(self):
        self.feature_model = None
        self.ready         = False
        self._try_load()

    def _try_load(self):
        try:
            import tensorflow as tf
            base = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights="imagenet",
                pooling="avg",
            )
            base.trainable = False
            self.feature_model = base
            self.ready = True
            print("[SkinService] MobileNetV2 feature extractor loaded")
        except ImportError:
            print("[SkinService] TensorFlow not installed; skin detection unavailable.")
        except Exception as e:
            print(f"[SkinService] {e}")

    def predict(self, image_bytes: bytes) -> dict:
        if not self.ready:
            return self._not_loaded()

        try:
            img_arr    = self._preprocess(image_bytes)
            features   = self.feature_model.predict(img_arr, verbose=0)[0]
            label, confidence = self._classify(img_arr[0], features)
        except Exception as e:
            log.error(f"Skin prediction error: {e}")
            return self._not_loaded()

        advice = SKIN_ADVICE[label]
        return {
            "loaded":         True,
            "finding":        label,
            "display":        label.replace("_", " ").title(),
            "confidence":     confidence,
            "description":    advice["description"],
            "precautions":    advice["precautions"],
            "specialization": advice["specialization"],
            "primary_specialist": advice["specialization"],
            "secondary_specialists": [],
            "maps_url": f"https://www.google.com/maps/search/{quote_plus(advice['specialization'] + ' near me')}",
            "disclaimer": (
                "⚠️ AI skin analysis is for educational purposes only. "
                "Always consult a qualified dermatologist for diagnosis."
            ),
        }

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        import tensorflow as tf
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        return arr[np.newaxis, ...]

    def _classify(self, img_arr: np.ndarray, features: np.ndarray) -> tuple[str, float]:
        """
        Heuristic classifier based on image colour statistics.
        Replaces with a trained head for production use.
        """
        # Denorm image to [0,255] for colour stats
        img = ((img_arr + 1.0) * 127.5).astype(np.uint8)
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

        r_mean  = float(r.mean())
        g_mean  = float(g.mean())
        b_mean  = float(b.mean())
        r_std   = float(r.std())

        # Redness index — rash and acne tend to be redder
        redness = (r_mean - (g_mean + b_mean) / 2) / 255.0

        # Texture variability — fungal infections have patchy textures
        texture = float(np.std(features))

        # Simple decision tree on colour + texture statistics
        if redness > 0.08 and r_std > 35:
            if r_std > 55:
                label      = "acne"
                confidence = min(55 + redness * 120, 82)
            else:
                label      = "rash"
                confidence = min(50 + redness * 100, 78)
        elif texture > 0.85 and redness > 0.03:
            label      = "fungal_infection"
            confidence = min(45 + texture * 30, 75)
        else:
            label      = "normal_skin"
            confidence = min(60 + (0.15 - abs(redness)) * 100, 85)

        return label, round(float(confidence), 1)

    def _not_loaded(self) -> dict:
        return {
            "loaded":         False,
            "finding":        None,
            "display":        "Unavailable",
            "confidence":     0.0,
            "description":    "Skin detection model not available. Install TensorFlow.",
            "precautions":    [],
            "specialization": "Dermatologist",
            "disclaimer":     "Model unavailable.",
        }
