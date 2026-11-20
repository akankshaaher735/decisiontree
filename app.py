import csv
import io
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "decision.pkl"

EXPECTED_FEATURES = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
]

FEATURE_DESCRIPTIONS = {
    "person_age": "Applicant age (numeric)",
    "person_gender": "Encoded gender value (numeric)",
    "person_education": "Encoded education level (numeric)",
    "person_income": "Annual income (numeric)",
    "person_emp_exp": "Employment experience (numeric)",
    "person_home_ownership": "Encoded home ownership (numeric)",
    "loan_amnt": "Requested loan amount (numeric)",
    "loan_intent": "Encoded loan intent (numeric)",
    "loan_int_rate": "Loan interest rate (numeric)",
    "loan_percent_income": "Loan amount as % of income (numeric)",
    "cb_person_cred_hist_length": "Credit history length (numeric)",
    "credit_score": "Credit score (numeric)",
    "previous_loan_defaults_on_file": "Encoded prior default flag (numeric)",
}


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    with open(path, "rb") as file:
        return pickle.load(file)


MODEL = load_model(MODEL_PATH)

FEATURE_ORDER = (
    list(MODEL.feature_names_in_)
    if hasattr(MODEL, "feature_names_in_") and MODEL.feature_names_in_ is not None
    else EXPECTED_FEATURES
)

CLASSES = (
    [int(c) if isinstance(c, (np.integer, int)) else c for c in MODEL.classes_]
    if hasattr(MODEL, "classes_") and MODEL.classes_ is not None
    else [0, 1]
)

SESSION_HISTORY: list[dict[str, Any]] = []
CLASS_COUNTS: dict[str, int] = {str(c): 0 for c in CLASSES}


def build_feature_vector(payload: dict[str, Any]) -> np.ndarray:
    values: list[float] = []
    for feature in FEATURE_ORDER:
        if feature not in payload:
            raise ValueError(f"Missing required feature: '{feature}'.")
        raw_val = payload[feature]
        if raw_val is None or str(raw_val).strip() == "":
            raise ValueError(f"Feature '{feature}' cannot be empty.")
        try:
            num_val = float(raw_val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{feature}' must be a valid number.") from exc
        if not np.isfinite(num_val):
            raise ValueError(f"Feature '{feature}' must be a finite number.")
        values.append(num_val)
    return np.array([values], dtype=np.float64)


HTML_LINES = [
    '<!DOCTYPE html>',
    '<html lang="en" data-theme="dark">',
    '<head>',
    '  <meta charset="UTF-8" />',
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
    '  <title>DecisionTree Inference Studio</title>',
    '  <link rel="preconnect" href="https://fonts.googleapis.com">',
    '  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
    '  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">',
    '  <style>',
    '    :root {',
    '      --bg-1: #050816;',
    '      --bg-2: #120b2e;',
    '      --bg-3: #0b2545;',
    '      --accent-blue: #3b82f6;',
    '      --accent-purple: #8b5cf6;',
    '      --accent-cyan: #22d3ee;',
    '      --glass-bg: rgba(15, 23, 52, 0.58);',
    '      --glass-border: rgba(125, 211, 252, 0.22);',
    '      --glass-highlight: rgba(139, 92, 246, 0.28);',
    '      --text-main: #f1f5f9;',
    '      --text-muted: #94a3b8;',
    '      --input-bg: rgba(8, 13, 32, 0.72);',
    '      --input-border: rgba(148, 163, 184, 0.24);',
    '      --shadow-card: 0 20px 50px rgba(2, 6, 23, 0.65), 0 0 28px rgba(59, 130, 246, 0.14);',
    '      --success-bg: rgba(16, 185, 129, 0.15);',
    '      --success-border: rgba(52, 211, 153, 0.45);',
    '      --success-text: #6ee7b7;',
    '      --error-bg: rgba(244, 63, 94, 0.15);',
    '      --error-border:
