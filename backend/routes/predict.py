import os

import google.generativeai as genai
from flask import Blueprint, current_app, request

from backend.models.response import err, ok, pred_ok

predict_bp = Blueprint("predict", __name__)


def _svc():
    return current_app.config["PRED_SVC"]


def _unique(items):
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _gemini_candidates():
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    candidates = [
        env_model,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-pro",
    ]

    try:
        available = []
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                available.append(model.name)

        preferred = sorted(
            available,
            key=lambda name: (
                "flash" not in name.lower(),
                "pro" not in name.lower(),
                name,
            ),
        )
        candidates = [env_model, *preferred, *candidates]
    except Exception:
        pass

    return _unique(candidates)


@predict_bp.route("/api/predict/symptoms", methods=["POST"])
def predict_symptoms():
    svc = _svc()
    body = request.get_json() or {}

    symptoms = body.get("symptoms", [])
    text = body.get("text", "")

    if not symptoms and text:
        symptoms = svc.extract(text)

    valid, invalid = svc.validate(symptoms)

    if not valid:
        return err("No valid symptoms found.", details={"invalid_symptoms": invalid})

    preds = svc.predict(valid, top_n=5)
    specialist_info = svc.specialist_recommendation(preds)
    return pred_ok(valid, preds, {
        "invalid_symptoms": invalid,
        **specialist_info,
    })


@predict_bp.route("/api/explain", methods=["POST"])
def explain():
    body = request.get_json() or {}

    preds = body.get("predictions", [])
    symptoms = body.get("symptoms", [])
    question = str(body.get("question", "")).strip()

    if not preds:
        return err("No predictions provided.")

    prompt = f"""
You are a medical assistant AI for MediSense.

User symptoms:
{', '.join(symptoms)}

Top predicted possible conditions:
{chr(10).join([f"- {p.get('disease', 'Unknown')} ({p.get('probability', 0)}%)" for p in preds])}

User follow-up question:
{question or 'No follow-up question. Give a concise overall explanation.'}

Explain in simple terms:
1. What these possible conditions are
2. Why the symptoms may match
3. What the user should do next
4. When to see a doctor urgently

Rules:
- Do NOT diagnose.
- Say "possible conditions".
- If there is a follow-up question, answer that question directly first.
- Mention low confidence when confidence is low.
- Keep it short.
- Use bullet points.
- Do not use markdown headings.
"""

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return err("GEMINI_API_KEY is not configured.")

        genai.configure(api_key=api_key)
        last_error = None
        for model_name in _gemini_candidates():
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                explanation = getattr(response, "text", "").strip()

                if explanation:
                    return ok({
                        "explanation": explanation,
                        "model": model_name,
                    })

                last_error = "Gemini returned an empty explanation."
            except Exception as model_error:
                last_error = model_error

        return err(f"Gemini error: {last_error}")

    except Exception as e:
        return err(f"Gemini error: {str(e)}")


@predict_bp.route("/api/model/info")
def model_info():
    info = _svc().info()
    return ok({
        **info,
        "model": "XGBoost",
    })


@predict_bp.route("/api/symptoms")
def symptoms():
    svc = _svc()
    return ok({"symptoms": svc.sym_list})
