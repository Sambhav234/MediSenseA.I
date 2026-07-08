"""
MediSense AI v3 — XGBoost Training Pipeline
============================================
Replaces Random Forest with XGBoost.
Produces: xgb_pipeline.pkl, label_encoder.pkl, symptoms_list.pkl, metadata.json
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MDL_DIR  = os.path.join(BASE, "models")
os.makedirs(MDL_DIR, exist_ok=True)

print("=" * 60)
print("  MediSense AI v3 — XGBoost Training Pipeline")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\n[1/6] Loading datasets...")
dataset      = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
descriptions = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
precautions  = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
severity_df  = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
print(f"  Rows: {len(dataset)} | Diseases: {dataset['Disease'].nunique()}")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
print("\n[2/6] Preprocessing...")

def clean(s):
    if pd.isna(s) or str(s).strip() == "": return ""
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")

severity_df["Symptom"] = severity_df["Symptom"].apply(clean)
SEV_MAP = dict(zip(severity_df["Symptom"], severity_df["weight"]))

sym_cols    = [c for c in dataset.columns if c.startswith("Symptom_")]
all_symptoms = sorted({clean(v) for col in sym_cols
                        for v in dataset[col].dropna() if clean(v)})
print(f"  Unique symptoms: {len(all_symptoms)}")

def encode_row(row):
    v = {s: 0 for s in all_symptoms}
    for col in sym_cols:
        s = clean(row.get(col, ""))
        if s and s in v:
            v[s] = SEV_MAP.get(s, 1)
    return v

print("  Building feature matrix...")
X = pd.DataFrame([encode_row(row) for _, row in dataset.iterrows()],
                  columns=all_symptoms)
le = LabelEncoder()
y  = le.fit_transform(dataset["Disease"].str.strip())

# ── 3. Train / test split ─────────────────────────────────────────────────────
print("\n[3/6] Train/test split (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. Train XGBoost ──────────────────────────────────────────────────────────
print("\n[4/6] Training XGBoost classifier...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\n[5/6] Evaluating...")
y_pred_train = xgb.predict(X_train)
y_pred_test  = xgb.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc  = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
f1_w      = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
f1_m      = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

print(f"  Train Accuracy  : {train_acc:.4f} ({train_acc*100:.1f}%)")
print(f"  Test  Accuracy  : {test_acc:.4f}  ({test_acc*100:.1f}%)")
print(f"  Precision (W)   : {precision:.4f}")
print(f"  Recall    (W)   : {recall:.4f}")
print(f"  F1 (Weighted)   : {f1_w:.4f}")
print(f"  F1 (Macro)      : {f1_m:.4f}")
print(f"  Overfit gap     : {train_acc - test_acc:.4f}")

report_dict = classification_report(
    y_test, y_pred_test, target_names=le.classes_,
    zero_division=0, output_dict=True)
cm = confusion_matrix(y_test, y_pred_test).tolist()

# ── 6. Save artefacts ─────────────────────────────────────────────────────────
print("\n[6/6] Saving artefacts...")

# Build lookup maps
desc_map = {r["Disease"].strip(): str(r["Description"]).strip()
            for _, r in descriptions.iterrows()}
prec_map = {}
for _, r in precautions.iterrows():
    ps = [str(r[f"Precaution_{i}"]).strip()
          for i in range(1, 5)
          if pd.notna(r.get(f"Precaution_{i}")) and str(r.get(f"Precaution_{i}", "")).strip()]
    prec_map[r["Disease"].strip()] = ps

SPEC_MAP = {
    "Heart attack":                             "Cardiologist",
    "Hypertension ":                            "Cardiologist",
    "Diabetes ":                                "Endocrinologist",
    "Hypothyroidism":                           "Endocrinologist",
    "Hyperthyroidism":                          "Endocrinologist",
    "Hypoglycemia":                             "Endocrinologist",
    "Tuberculosis":                             "Pulmonologist",
    "Pneumonia":                                "Pulmonologist",
    "Bronchial Asthma":                         "Pulmonologist",
    "Common Cold":                              "General Physician",
    "Allergy":                                  "Allergist",
    "Migraine":                                 "Neurologist",
    "(vertigo) Paroymsal  Positional Vertigo":  "Neurologist",
    "Paralysis (brain hemorrhage)":             "Neurologist",
    "Cervical spondylosis":                     "Orthopedist",
    "Arthritis":                                "Rheumatologist",
    "Dengue":                                   "Infectious Disease Specialist",
    "Malaria":                                  "Infectious Disease Specialist",
    "Typhoid":                                  "Infectious Disease Specialist",
    "AIDS":                                     "Infectious Disease Specialist",
    "Hepatitis A":                              "Gastroenterologist",
    "Hepatitis B":                              "Gastroenterologist",
    "Hepatitis C":                              "Gastroenterologist",
    "Hepatitis D":                              "Gastroenterologist",
    "Hepatitis E":                              "Gastroenterologist",
    "Chronic cholestasis":                      "Gastroenterologist",
    "GERD":                                     "Gastroenterologist",
    "Gastroenteritis":                          "Gastroenterologist",
    "Peptic ulcer diseae":                      "Gastroenterologist",
    "Jaundice":                                 "Gastroenterologist",
    "Alcoholic hepatitis":                      "Gastroenterologist",
    "Dimorphic hemmorhoids(piles)":             "Proctologist",
    "Urinary tract infection":                  "Urologist",
    "Drug Reaction":                            "Dermatologist",
    "Acne":                                     "Dermatologist",
    "Fungal infection":                         "Dermatologist",
    "Psoriasis":                                "Dermatologist",
    "Impetigo":                                 "Dermatologist",
    "Varicose veins":                           "Vascular Surgeon",
    "Chicken pox":                              "General Physician",
}

joblib.dump(xgb,           os.path.join(MDL_DIR, "xgb_model.pkl"))
joblib.dump(le,            os.path.join(MDL_DIR, "label_encoder.pkl"))
joblib.dump(all_symptoms,  os.path.join(MDL_DIR, "symptoms_list.pkl"))

metadata = {
    "version":        "3.0",
    "model_type":     "XGBoost",
    "diseases":       list(le.classes_),
    "total_symptoms": len(all_symptoms),
    "descriptions":   desc_map,
    "precautions":    prec_map,
    "specializations": SPEC_MAP,
    "severity_map":   SEV_MAP,
    "metrics": {
        "train_accuracy":       round(train_acc, 4),
        "test_accuracy":        round(test_acc,  4),
        "precision_weighted":   round(precision, 4),
        "recall_weighted":      round(recall,    4),
        "f1_weighted":          round(f1_w,      4),
        "f1_macro":             round(f1_m,      4),
        "overfit_gap":          round(float(train_acc - test_acc), 4),
    },
    "per_class_report": report_dict,
    "confusion_matrix": cm,
    "class_names":      list(le.classes_),
}
with open(os.path.join(MDL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  ✅ xgb_model.pkl, label_encoder.pkl, symptoms_list.pkl, metadata.json")
print(f"\n{'='*60}")
print(f"  ✅ XGBoost training complete!")
print(f"  Test Accuracy : {test_acc*100:.1f}%  |  F1: {f1_w*100:.1f}%")
print(f"{'='*60}\n")
