import os, json, logging
from urllib.parse import quote_plus

import pandas as pd
import joblib

from backend.services.synonyms import SYNONYMS

log = logging.getLogger("medisense.prediction")

MIN_SYMPTOMS = 3
MIN_CONFIDENCE = 2.0

SAFETY_RULES = {
    "Heart attack": ["chest_pain", "breathlessness"],
    "AIDS": ["extra_marital_contacts", "receiving_blood_transfusion"],
    "Paralysis (brain hemorrhage)": [
        "weakness_of_one_body_side",
        "altered_sensorium",
        "loss_of_balance",
        "slurred_speech",
    ],
    "Alcoholic hepatitis": ["history_of_alcohol_consumption"],
    "Dimorphic hemmorhoids(piles)": [
        "pain_in_anal_region",
        "bloody_stool",
        "irritation_in_anus",
    ],
}

SPECIALIST_OVERRIDES = {
    "Pneumonia": "Pulmonologist",
    "Bronchial Asthma": "Pulmonologist",
    "Asthma": "Pulmonologist",
    "Tuberculosis": "Pulmonologist",
    "Common Cold": "General Physician",
    "Migraine": "Neurologist",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Heart attack": "Cardiologist",
    "Hypertension ": "Cardiologist",
    "Diabetes ": "Endocrinologist",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Allergy": "Allergist",
    "Dengue": "Infectious Disease Specialist",
    "Malaria": "Infectious Disease Specialist",
    "Typhoid": "Infectious Disease Specialist",
    "Fungal infection": "Dermatologist",
    "Acne": "Dermatologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist",
}


class PredictionService:
    def __init__(self, ml_dir: str):
        self.ml_dir = ml_dir
        self.ready = False
        self._load()

    def _load(self):
        try:
            self.model = joblib.load(os.path.join(self.ml_dir, "xgb_model.pkl"))
            self.le = joblib.load(os.path.join(self.ml_dir, "label_encoder.pkl"))
            self.sym_list = joblib.load(os.path.join(self.ml_dir, "symptoms_list.pkl"))

            with open(os.path.join(self.ml_dir, "metadata.json")) as f:
                self.meta = json.load(f)

            self._sym_set = set(self.sym_list)
            self._disease_symptoms = self._load_disease_symptoms()
            self.ready = True
            print("[PredictionService] Loaded")

        except Exception as e:
            print("[PredictionService] Model load error:", e)

    def _clean(self, s):
        return s.strip().lower().replace(" ", "_")

    def _load_disease_symptoms(self):
        data_path = os.path.abspath(os.path.join(self.ml_dir, "..", "data", "dataset.csv"))
        if not os.path.exists(data_path):
            return {}

        df = pd.read_csv(data_path)
        sym_cols = [c for c in df.columns if c.startswith("Symptom_")]
        disease_symptoms = {}

        for disease, group in df.groupby("Disease"):
            symptoms = set()
            for _, row in group.iterrows():
                for col in sym_cols:
                    raw = row.get(col, "")
                    if pd.isna(raw):
                        continue
                    symptom = self._clean(str(raw).replace("-", "_"))
                    if symptom:
                        symptoms.add(symptom)
            disease_symptoms[str(disease).strip()] = symptoms

        return disease_symptoms

    def extract(self, text):
        text = text.lower()
        found = []

        for phrase, canonical in SYNONYMS.items():
            if phrase in text and canonical in self._sym_set:
                found.append(canonical)

        return list(dict.fromkeys(found))

    def validate(self, symptoms):
        valid = []
        invalid = []

        for s in symptoms:
            c = self._clean(s)
            if c in self._sym_set and c not in valid:
                valid.append(c)
            else:
                invalid.append(s)

        return valid, invalid

    def _passes_safety(self, disease, syms):
        rules = SAFETY_RULES.get(disease)
        if not rules:
            return True
        return any(r in syms for r in rules)

    def _vec(self, syms):
        severity = self.meta.get("severity_map", {})
        v = {s: 0 for s in self.sym_list}
        for s in syms:
            if s in v:
                v[s] = severity.get(s, 1)
        return pd.DataFrame([[v[s] for s in self.sym_list]], columns=self.sym_list)

    def _result(self, disease, probability, warning=""):
        specialist = self.specialist_for(disease)
        return {
            "disease": disease,
            "probability": probability,
            "description": self.meta["descriptions"].get(disease, ""),
            "precautions": self.meta["precautions"].get(disease, []),
            "specialization": specialist,
            "warning": warning,
        }

    def specialist_for(self, disease):
        return (
            SPECIALIST_OVERRIDES.get(disease)
            or self.meta.get("specializations", {}).get(disease)
            or "General Physician"
        )

    def specialist_recommendation(self, predictions):
        if not predictions:
            primary = "General Physician"
            return {
                "primary_specialist": primary,
                "secondary_specialists": [],
                "maps_url": self.maps_url(primary),
            }

        top = predictions[0]
        top_prob = float(top.get("probability") or 0)
        top_disease = top.get("disease", "")
        top_spec = top.get("specialization") or self.specialist_for(top_disease)

        # ✅ Rule 1: High confidence → trust top disease
        if top_prob >= 80:
            primary = top_spec
            return {
                "primary_specialist": primary,
                "secondary_specialists": [],
                "maps_url": self.maps_url(primary),
            }

        # ✅ Rule 2: Otherwise aggregate specialists
        scores = {}

        for pred in predictions[:5]:
            disease = pred.get("disease", "")
            specialist = pred.get("specialization") or self.specialist_for(disease)
            prob = float(pred.get("probability") or 0)

            if specialist not in scores:
                scores[specialist] = 0

            scores[specialist] += prob

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary = ranked[0][0] if ranked else "General Physician"
        secondary = [ranked[1][0]] if len(ranked) > 1 else []

        return {
            "primary_specialist": primary,
            "secondary_specialists": secondary,
            "maps_url": self.maps_url(primary),
        }

    def maps_url(self, specialist):
        return f"https://www.google.com/maps/search/{quote_plus(specialist + ' near me')}"

    def _symptom_match_results(self, syms, top_n):
        syms = set(syms)
        scored = []

        for disease, disease_syms in self._disease_symptoms.items():
            if not self._passes_safety(disease, syms):
                continue

            overlap = syms & disease_syms
            if not overlap:
                continue

            score = len(overlap) / max(len(syms), 1)
            scored.append((score, len(overlap), disease))

        scored.sort(reverse=True)

        results = []
        for score, _, disease in scored[:top_n]:
            confidence = round(score * 100, 1)
            results.append(self._result(
                disease,
                confidence,
                "Low XGBoost confidence; ranked by symptom match"
            ))

        return results

    def predict(self, syms, top_n=3):
        if len(syms) < MIN_SYMPTOMS:
            return [{
                "disease": "Insufficient data",
                "probability": 0,
                "description": "Add more symptoms",
                "precautions": [],
                "specialization": "General Physician",
                "warning": "Add at least 3 symptoms",
            }]

        probs = self.model.predict_proba(self._vec(syms))[0]
        ranked = probs.argsort()[::-1]
        results = []
        best_conf = round(float(probs[ranked[0]]) * 100, 1)

        if best_conf < 10 and self._disease_symptoms:
            matched = self._symptom_match_results(syms, top_n)
            if matched:
                return matched

        for idx in ranked:
            if len(results) >= top_n:
                break

            disease = self.le.classes_[idx]
            conf = round(float(probs[idx]) * 100, 1)

            if conf < MIN_CONFIDENCE:
                continue

            if not self._passes_safety(disease, syms):
                continue

            warning = "Low confidence" if conf < 10 else ""
            results.append(self._result(disease, conf, warning))

        if not results:
            for idx in ranked:
                disease = self.le.classes_[idx]
                if not self._passes_safety(disease, syms):
                    continue
                conf = round(float(probs[idx]) * 100, 1)
                results.append(self._result(disease, conf, "Low confidence"))
                break

        return results

    def info(self):
        return {
            "ready": self.ready,
            "total_symptoms": len(getattr(self, "sym_list", [])),
            "total_diseases": len(getattr(getattr(self, "le", None), "classes_", [])),
            "metrics": self.meta.get("metrics", {}) if self.ready else {},
        }
