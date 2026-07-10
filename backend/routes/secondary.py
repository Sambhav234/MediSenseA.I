"""
MediSense AI v3 — Image, Doctor & Utility Routes
=================================================
POST /api/predict/xray      — xray CNN inference
POST /api/predict/skin      — skin detection
POST /api/predict/multimodal — text + image combined
GET  /api/doctors           — doctor recommendations
GET  /api/trending          — health tips & alerts
GET  /api/health            — system status
"""
import logging
from urllib.parse import quote_plus
from flask import Blueprint, request, current_app
from backend.models.response import ok, err
from backend.services import fusion_service

log = logging.getLogger("medisense.routes.secondary")
secondary_bp = Blueprint("secondary", __name__)


# ── X-Ray prediction ──────────────────────────────────────────────────────────
@secondary_bp.route("/api/predict/xray", methods=["POST"])
def predict_xray():
    svc = current_app.config["XRAY_SVC"]
    if not svc.ready:
     return err("X-ray model is not loaded.", 503)
    if "image" not in request.files:
        return err("Send image as multipart/form-data field 'image'.", 400)
    result = svc.predict(request.files["image"].read())
    return ok(result, "X-ray analysis complete")


# ── Skin prediction ───────────────────────────────────────────────────────────
@secondary_bp.route("/api/predict/skin", methods=["POST"])
def predict_skin():
    svc = current_app.config["SKIN_SVC"]
    if "image" not in request.files:
        return err("Send image as multipart/form-data field 'image'.", 400)
    result = svc.predict(request.files["image"].read())
    return ok(result, "Skin analysis complete")


# ── Multi-modal: text symptoms + image ───────────────────────────────────────
@secondary_bp.route("/api/predict/multimodal", methods=["POST"])
def predict_multimodal():
    """
    Accepts multipart/form-data:
      - symptoms: JSON string list  OR  text: free text description
      - image:    file upload (optional)
      - image_type: "xray" | "skin"  (default: xray)
    """
    pred_svc = current_app.config["PRED_SVC"]
    xray_svc = current_app.config["XRAY_SVC"]
    skin_svc = current_app.config["SKIN_SVC"]

    if not pred_svc.ready:
        return err("Prediction model not loaded.", 503)

    # Parse symptoms
    import json as _json
    raw_syms = request.form.get("symptoms", "")
    text     = request.form.get("text", "").strip()
    img_type = request.form.get("image_type", "xray").lower()

    symptoms = []
    if raw_syms:
        try:
            symptoms = _json.loads(raw_syms)
        except Exception:
            symptoms = [s.strip() for s in raw_syms.split(",") if s.strip()]
    if not symptoms and text:
        symptoms = pred_svc.extract(text)

    valid, invalid = pred_svc.validate(symptoms)
    text_preds = pred_svc.predict(valid, top_n=3) if valid else []

    # Parse image
    image_result = None
    if "image" in request.files:
        img_bytes = request.files["image"].read()
        if img_type == "skin":
            image_result = skin_svc.predict(img_bytes)
        else:
            image_result = xray_svc.predict(img_bytes)

    # Fuse
    fused = fusion_service.fuse(text_preds, image_result, img_type)

    return ok({
        "matched_symptoms": valid,
        "text_predictions": text_preds,
        "image_result":     image_result,
        "fused_predictions": fused["predictions"],
        "fusion_applied":   fused["fusion_applied"],
        "fusion_note":      fused["fusion_note"],
        "disclaimer": (
            " AI-assisted analysis only. NOT a medical diagnosis. "
            "Always consult a certified healthcare professional."
        ),
    }, "Multi-modal analysis complete")


# ── Doctors ───────────────────────────────────────────────────────────────────
@secondary_bp.route("/api/doctors")
def doctors():
    spec = request.args.get("specialization", "General Physician").strip() or "General Physician"
    return ok({
        "primary_specialist": spec,
        "maps_url": f"https://www.google.com/maps/search/{quote_plus(spec + ' near me')}",
        "mode": "google_maps_search",
    })


# ── Trending tips ─────────────────────────────────────────────────────────────
@secondary_bp.route("/api/trending")
def trending():
    return ok({
        "tips": [
            {"title":"Stay Hydrated",    "body":"Drink 8 glasses of water daily.",         "icon":"water_drop"},
            {"title":"Regular Exercise", "body":"30 min of moderate activity 5×/week.",    "icon":"directions_run"},
            {"title":"Sleep Hygiene",    "body":"7–9 hours of quality sleep nightly.",     "icon":"bedtime"},
            {"title":"Balanced Diet",    "body":"5 servings of fruits & vegetables daily.","icon":"restaurant"},
            {"title":"Mental Wellness",  "body":"Practice mindfulness — stress weakens immunity.","icon":"self_improvement"},
            {"title":"Hand Hygiene",     "body":"20-second handwash prevents 21% of infections.","icon":"clean_hands"},
        ],
        "alerts": [
            {"disease":"Dengue",      "severity":"high",    "cases":"Rising",    "precaution":"Eliminate stagnant water"},
            {"disease":"Seasonal Flu","severity":"moderate","cases":"Elevated",  "precaution":"Get vaccinated"},
            {"disease":"Typhoid",     "severity":"moderate","cases":"Monitoring","precaution":"Drink clean water"},
        ],
    })


# ── System health ─────────────────────────────────────────────────────────────
@secondary_bp.route("/api/health")
def health():
    pred = current_app.config["PRED_SVC"]
    xray = current_app.config["XRAY_SVC"]
    skin = current_app.config["SKIN_SVC"]
    return ok({
        "service":   "MediSense AI v3",
        "status":    "running",
        "models": {
            "xgboost":  pred.ready,
            "xray_cnn": xray.ready,
            "skin_cnn": skin.ready,
        },
        "model_info": pred.info(),
    })
