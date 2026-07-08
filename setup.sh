#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MediSense AI v3 — Setup & Launch
# Usage:
#   chmod +x setup.sh
#   ./setup.sh            full setup + start backend
#   ./setup.sh install    pip install only
#   ./setup.sh train      train XGBoost model only
#   ./setup.sh backend    start Flask server only
# ─────────────────────────────────────────────────────────────────────────────
set -e
G='\033[0;32m'; C='\033[0;36m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'

banner() {
  echo -e "${C}"
  echo "  ╔══════════════════════════════════════════════╗"
  echo "  ║    MediSense AI v3 — XGBoost + CNN + Maps   ║"
  echo "  ╚══════════════════════════════════════════════╝"
  echo -e "${N}"
}

step() { echo -e "\n${Y}▶ $1${N}"; }
ok()   { echo -e "${G}  ✅ $1${N}"; }
fail() { echo -e "${R}  ❌ $1${N}"; exit 1; }

MODE=${1:-all}
banner

# Check Python
command -v python3 &>/dev/null || command -v py &>/dev/null || fail "Python 3 not found."
PY=$(command -v py 2>/dev/null || command -v python3)
ok "Python: $($PY --version)"

# .env setup
if [[ ! -f backend/.env && -f backend/.env.example ]]; then
  cp backend/.env.example backend/.env
  echo -e "  ${C}ℹ  Created backend/.env — add your API keys there.${N}"
fi

# Install
if [[ "$MODE" == "all" || "$MODE" == "install" ]]; then
  step "Installing dependencies…"
  $PY -m pip install -r requirements.txt -q --disable-pip-version-check
  ok "Dependencies installed"
fi

# Train
if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
  step "Training XGBoost model…"
  [[ -f ml/data/dataset.csv ]] || fail "ml/data/dataset.csv not found."
  $PY ml/train_model.py
  ok "XGBoost model saved to ml/models/"
fi

# Start
if [[ "$MODE" == "all" || "$MODE" == "backend" ]]; then
  step "Starting Flask backend…"
  [[ -f ml/models/xgb_model.pkl ]] || { echo "Model not found — training first…"; $PY ml/train_model.py; }
  echo ""
  echo -e "  ${G}🚀 Backend : http://localhost:5000${N}"
  echo -e "  ${C}   Frontend: open frontend/index.html in your browser${N}"
  echo ""
  $PY backend/app.py
fi
