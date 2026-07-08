import os, io, logging
from urllib.parse import quote_plus

import numpy as np

log = logging.getLogger("medisense.xray")

# Classes
XRAY_CLASSES = ["NORMAL", "PNEUMONIA", "UNCERTAIN"]

# MUST match training size
XRAY_IMG_SIZE = (224, 224)

# Model path
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "ml", "models", "xray_model.hdf5"
)

# Specialist mapping
XRAY_SPEC = {
    "PNEUMONIA": "Pulmonologist",
    "NORMAL": "General Physician",
    "UNCERTAIN": "General Physician",
}

# Advice mapping
XRAY_ADVICE = {
    "PNEUMONIA": {
        "description": "Patterns consistent with pneumonia detected. Seek medical attention.",
        "precautions": [
            "Consult a pulmonologist immediately",
            "Stay hydrated and rest",
            "Follow prescribed medication",
        ],
    },
    "NORMAL": {
        "description": "No major abnormality detected in the X-ray.",
        "precautions": [
            "Maintain healthy lifestyle",
            "Consult doctor if symptoms persist",
        ],
    },
    "UNCERTAIN": {
        "description": "The model is not confident about the result. Further medical evaluation is recommended.",
        "precautions": [
            "Consult a doctor for proper diagnosis",
            "Consider additional tests",
        ],
    },
}


class XRayService:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = os.path.abspath(model_path)
        self.model = None
        self.ready = False
        self._try_load()

    def _try_load(self):
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.ready = True
            print(f"[XRayService] Loaded model: {self.model.input_shape}")
        except Exception as e:
            print(f"[XRayService] Load error: {e}")

    def predict(self, image_bytes: bytes) -> dict:
        if not self.ready:
            return self._not_loaded()

        arr = self._preprocess(image_bytes)

        probs = self.model.predict(arr, verbose=0)[0]
        print("[XRayService] Raw probabilities:", probs)

        # Ensure valid probability distribution
        if abs(probs.sum() - 1.0) > 0.01:
            import tensorflow as tf
            probs = tf.nn.softmax(probs).numpy()

        p_normal = probs[0]
        p_pneumonia = probs[1]

        # 🔥 Improved decision logic
        if p_pneumonia > 0.80:
            finding = "PNEUMONIA"
            confidence = p_pneumonia

        elif p_normal > 0.65:
            finding = "NORMAL"
            confidence = p_normal

        else:
            finding = "UNCERTAIN"
            confidence = max(p_normal, p_pneumonia)

        confidence = round(float(confidence) * 100, 1)

        advice = XRAY_ADVICE[finding]

        return {
            "loaded": True,
            "finding": finding,
            "confidence": confidence,
            "all_scores": [
                {
                    "label": XRAY_CLASSES[i] if i < 2 else "UNCERTAIN",
                    "confidence": round(float(probs[i]) * 100, 1)
                }
                for i in range(len(probs))
            ],
            "description": advice["description"],
            "precautions": advice["precautions"],
            "specialization": XRAY_SPEC[finding],
            "primary_specialist": XRAY_SPEC[finding],
            "secondary_specialists": [],
            "maps_url": f"https://www.google.com/maps/search/{quote_plus(XRAY_SPEC[finding] + ' near me')}",
            "disclaimer": "⚠️ AI-assisted only. Always consult a doctor.",
        }

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(XRAY_IMG_SIZE)  # 🔥 keep full image, no cropping

        arr = np.asarray(img, dtype=np.float32)

        # Normalize
        arr = arr / 255.0

        return arr[np.newaxis, ...]

    def _not_loaded(self) -> dict:
        return {
            "loaded": False,
            "finding": None,
            "confidence": 0.0,
            "all_scores": [],
            "description": "Model not loaded",
            "precautions": [],
            "specialization": "General Physician",
            "disclaimer": "Model unavailable",
        }
