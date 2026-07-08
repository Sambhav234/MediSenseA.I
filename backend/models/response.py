"""MediSense AI v3 — Standard JSON response envelope."""
from flask import jsonify
from datetime import datetime, timezone

DISCLAIMER = (
    "⚠️ This is an AI-assisted prediction for educational purposes only. "
    "It is NOT a medical diagnosis. Always consult a certified doctor."
)

def _ts():
    return datetime.now(timezone.utc).isoformat()

def ok(data=None, message="OK", code=200):
    return jsonify({"status":"success","message":message,"data":data,"timestamp":_ts()}), code

def err(message="Error", code=400, details=None):
    body = {"status":"error","message":message,"data":None,"timestamp":_ts()}
    if details: body["details"] = details
    return jsonify(body), code

def pred_ok(matched, predictions, extra=None):
    data = {"matched_symptoms":matched,"predictions":predictions,"disclaimer":DISCLAIMER}
    if extra: data.update(extra)
    return ok(data, "Prediction generated")
