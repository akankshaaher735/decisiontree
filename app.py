import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template_string, request

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

FEATURE_LABELS = {
    "person_age": "Age",
    "person_gender": "Gender",
    "person_education": "Education",
    "person_income": "Annual income",
    "person_emp_exp": "Employment experience",
    "person_home_ownership": "Home ownership",
    "loan_amnt": "Loan amount",
    "loan_intent": "Loan intent",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan % of income",
    "cb_person_cred_hist_length": "Credit history length",
    "credit_score": "Credit score",
    "previous_loan_defaults_on_file": "Prior loan defaults",
}

FEATURE_HINTS = {
    "person_age": "Years",
    "person_income": "Annual amount",
    "person_emp_exp": "Years",
    "loan_amnt": "Requested amount",
    "loan_int_rate": "%",
    "loan_percent_income": "Ratio / %",
    "cb_person_cred_hist_length": "Years",
    "credit_score": "Score",
}

CATEGORICAL_OPTIONS = {
    "person_gender": [
        {"label": "Female", "value": 0},
        {"label": "Male", "value": 1},
    ],
    "person_education": [
        {"label": "High School", "value": 0},
        {"label": "Associate", "value": 1},
        {"label": "Bachelor", "value": 2},
        {"label": "Master", "value": 3},
        {"label": "Doctorate", "value": 4},
    ],
    "person_home_ownership": [
        {"label": "Rent", "value": 0},
        {"label": "Mortgage", "value": 1},
        {"label": "Own", "value": 2},
        {"label": "Other", "value": 3},
    ],
    "loan_intent": [
        {"label": "Personal", "value": 0},
        {"label": "Education", "value": 1},
        {"label": "Medical", "value": 2},
        {"label": "Venture", "value": 3},
        {"label": "Home Improvement", "value": 4},
        {"label": "Debt Consolidation", "value": 5},
    ],
    "previous_loan_defaults_on_file": [
        {"label": "No", "value": 0},
        {"label": "Yes", "value": 1},
    ],
}

SECTIONS = [
    (
        "Applicant profile",
        ["person_age", "person_gender", "person_education", "person_income", "person_emp_exp", "person_home_ownership"],
    ),
    (
        "Loan & credit details",
        ["loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file"],
    ),
]


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


def build_feature_vector(payload: dict[str, Any]) -> np.ndarray:
    values: list[float] = []
    for feature in FEATURE_ORDER:
        if feature not in payload:
            raise ValueError(f"Missing required field: {FEATURE_LABELS.get(feature, feature)}.")
        raw_val = payload[feature]
        if raw_val is None or str(raw_val).strip() == "":
            raise ValueError(f"Please provide {FEATURE_LABELS.get(feature, feature)}.")
        try:
            num_val = float(raw_val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{FEATURE_LABELS.get(feature, feature)} must be a valid number.") from exc
        if not np.isfinite(num_val):
            raise ValueError(f"{FEATURE_LABELS.get(feature, feature)} must be a finite number.")
        values.append(num_val)
    return np.array([values], dtype=np.float64)


HTML_LINES = [
    "<!DOCTYPE html>",
    '<html lang="en">',
    "<head>",
    '  <meta charset="UTF-8" />',
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
    "  <title>Loan Decision Check</title>",
    '  <link rel="preconnect" href="https://fonts.googleapis.com">',
    '  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
    '  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
    "  <style>",
    "    * { box-sizing: border-box; margin: 0; padding: 0; }",
    "    body {",
    '      font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;',
    "      background: #f0f9ff;",
    "      color: #0c4a6e;",
    "      min-height: 100vh;",
    "    }",
    "    .layout {",
    "      display: grid;",
    "      grid-template-columns: 420px 1fr;",
    "      min-height: 100vh;",
    "    }",
    "    @media (max-width: 900px) {",
    "      .layout { grid-template-columns: 1fr; }",
    "      .sidebar { position: static !important; height: auto !important; }",
    "    }",
    "    .sidebar {",
    "      position: sticky;",
    "      top: 0;",
    "      height: 100vh;",
    "      background: linear-gradient(180deg, #0c4a6e 0%, #0369a1 65%, #0284c7 100%);",
    "      color: #f0f9ff;",
    "      padding: 40px 34px;",
    "      display: flex;",
    "      flex-direction: column;",
    "      justify-content: space-between;",
    "    }",
    "    .kicker {",
    "      font-size: 0.92rem;",
    "      font-weight: 600;",
    "      color: #7dd3fc;",
    "      margin-bottom: 14px;",
    "    }",
    "    .hero-title {",
    "      font-size: clamp(2.1rem, 3.4vw, 2.85rem);",
    "      font-weight: 800;",
    "      line-height: 1.08;",
    "      letter-spacing: -0.03em;",
    "      color: #ffffff;",
    "      margin-bottom: 18px;",
    "    }",
    "    .hero-copy {",
    "      font-size: 0.96rem;",
    "      line-height: 1.6;",
    "      color: #bae6fd;",
    "      margin-bottom: 28px;",
    "    }",
    "    .result-card {",
    "      background: rgba(7, 45, 71, 0.62);",
    "      border: 1px solid rgba(125, 211, 252, 0.28);",
    "      border-radius: 18px;",
    "      padding: 22px;",
    "    }",
    "    .result-heading {",
    "      font-size: 1.15rem;",
    "      font-weight: 700;",
    "      color: #ffffff;",
    "      margin-bottom: 6px;",
    "    }",
    "    .result-sub {",
    "      font-size: 0.88rem;",
    "      color: #bae6fd;",
    "      line-height: 1.5;",
    "    }",
    "    .result-badge {",
    "      display: inline-block;",
    "      margin-top: 12px;",
    "      padding: 6px 14px;",
    "      border-radius: 999px;",
    "      background: #38bdf8;",
    "      color: #082f49;",
    "      font-weight: 700;",
    "      font-size: 0.9rem;",
    "    }",
    "    .prob-rows { margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }",
    "    .prob-row { font-size: 0.85rem; color: #e0f2fe; display: flex; justify-content: space-between; }",
    "    .wave-deco {",
    "      height: 54px;",
    "      border-top: 1px solid rgba(125, 211, 252, 0.22);",
    "      display: flex;",
    "      align-items: center;",
    "      font-size: 0.8rem;",
    "      color: #7dd3fc;",
    "    }",
    "    .main-panel {",
    "      padding: 44px 48px 64px;",
    "      background: #f0f9ff;",
    "    }",
    "    @media (max-width: 600px) {",
    "      .main-panel { padding: 28px 20px 48px; }",
    "    }",
    "    .form-wrap { max-width: 780px; }",
    "    .form-title { font-size: 1.85rem; font-weight: 800; color: #0c4a6e; margin-bottom: 6px; }",
    "    .form-subtitle { font-size: 0.92rem; color: #0369a1; margin-bottom: 30px; }",
    "    .section-block { margin-bottom: 32px; }",
    "    .section-title {",
    "      font-size: 1.15rem;",
    "      font-weight: 700;",
    "      color: #0284c7;",
    "      padding-bottom: 10px;",
    "      border-bottom: 1px solid #bae6fd;",
    "      margin-bottom: 20px;",
    "    }",
    "    .fields-grid {",
    "      display: grid;",
    "      grid-template-columns: repeat(2, minmax(0, 1fr));",
    "      gap: 20px 24px;",
    "    }",
    "    @media (max-width: 640px) {",
    "      .fields-grid { grid-template-columns: 1fr; }",
    "    }",
    "    .field { display: flex; flex-direction: column; gap: 7px; }",
    "    .field label { font-size: 0.9rem; font-weight: 600; color: #0c4a6e; }",
    "    .input-box, .select-box {",
    "      width: 100%;",
    "      padding: 12px 14px;",
    "      border-radius: 10px;",
    "      border: 1px solid #7dd3fc;",
    "      background: #ffffff;",
    "      color: #0c4a6e;",
    "      font-size: 0.94rem;",
    "      outline: none;",
    "      transition: border-color 0.2s ease, box-shadow 0.2s ease;",
    "    }",
    "    .input-box:focus, .select-box:focus {",
    "      border-color: #0284c7;",
    "      box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.18);",
    "    }",
    "    .hint { font-size: 0.78rem; color: #0284c7; }",
    "    .pill-group { display: flex; flex-wrap: wrap; gap: 8px; }",
    "    .pill-btn {",
    "      padding: 9px 16px;",
    "      border-radius: 10px;",
    "      border: 1px solid #7dd3fc;",
    "      background: #e0f2fe;",
    "      color: #0369a1;",
    "      font-size: 0.88rem;",
    "      font-weight: 600;",
    "      cursor: pointer;",
    "      transition: all 0.18s ease;",
    "    }",
    "    .pill-btn.active {",
    "      background: #0284c7;",
    "      border-color: #0284c7;",
    "      color: #ffffff;",
    "    }",
    "    .actions { display: flex; gap: 12px; align-items: center; margin-top: 8px; }",
    "    .submit-btn {",
    "      padding: 13px 26px;",
    "      border-radius: 12px;",
    "      border: none;",
    "      background: #0284c7;",
    "      color: #ffffff;",
    "      font-size: 0.96rem;",
    "      font-weight: 700;",
    "      cursor: pointer;",
    "      transition: background 0.2s ease, transform 0.15s ease;",
    "    }",
    "    .submit-btn:hover { background: #0369a1; transform: translateY(-1px); }",
    "    .clear-btn {",
    "      padding: 13px 20px;",
    "      border-radius: 12px;",
    "      border: 1px solid #7dd3fc;",
    "      background: #ffffff;",
    "      color: #0369a1;",
    "      font-size: 0.92rem;",
    "      font-weight: 600;",
    "      cursor: pointer;",
    "    }",
    "    .error-msg { margin-top: 12px; font-size: 0.88rem; font-weight: 600; color: #b91c1c; }",
    "  </style>",
    "</head>",
    "<body>",
    '  <div class="layout">',
    '    <aside class="sidebar">',
    "      <div>",
    '        <div class="kicker">Sky Decision Model</div>',
    '        <h1 class="hero-title">Loan decision check</h1>',
    '        <p class="hero-copy">Enter applicant and loan details to run your decision tree model and view the predicted outcome.</p>',
    '        <div class="result-card">',
    '          <div class="result-heading" id="resultTitle">Your result appears here</div>',
    '          <div class="result-sub" id="resultSub">Fill in the form and select Check outcome.</div>',
    '          <div id="resultBadgeWrap"></div>',
    '          <div class="prob-rows" id="probRows"></div>',
    "        </div>",
    "      </div>",
    '      <div class="wave-deco">DecisionTreeClassifier &bull; decision.pkl</div>',
    "    </aside>",
    '    <main class="main-panel">',
    '      <form class="form-wrap" id="checkForm" novalidate>',
    '        <h2 class="form-title">Application details</h2>',
    '        <p class="form-subtitle">All fields are required. Select categories or enter numeric values.</p>',
    "        {% for sec_title, sec_feats in sections %}",
    '        <div class="section-block">',
    '          <div class="section-title">{{ sec_title }}</div>',
    '          <div class="fields-grid">',
    "            {% for feat in sec_feats %}",
    "            {% set opts = categorical_options.get(feat) %}",
    '            <div class="field">',
    '              <label for="{{ feat }}">{{ labels.get(feat, feat) }}</label>',
    "              {% if opts and opts|length <= 2 %}",
    '              <div class="pill-group" data-pill-for="{{ feat }}">',
    '                <input type="hidden" id="{{ feat }}" name="{{ feat }}" value="" />',
    "                {% for o in opts %}",
    '                <button type="button" class="pill-btn" data-val="{{ o.value }}">{{ o.label }}</button>',
    "                {% endfor %}",
    "              </div>",
    "              {% elif opts %}",
    '              <select class="select-box" id="{{ feat }}" name="{{ feat }}">',
    '                <option value="">Select one</option>',
    "                {% for o in opts %}",
    '                <option value="{{ o.value }}">{{ o.label }}</option>',
    "                {% endfor %}",
    "              </select>",
    "              {% else %}",
    '              <input class="input-box" type="number" step="any" id="{{ feat }}" name="{{ feat }}" placeholder="Enter value" />',
    "              {% endif %}",
    "              {% if hints.get(feat) %}",
    '              <span class="hint">{{ hints.get(feat) }}</span>',
    "              {% endif %}",
    "            </div>",
    "            {% endfor %}",
    "          </div>",
    "        </div>",
    "        {% endfor %}",
    '        <div class="actions">',
    '          <button type="submit" class="submit-btn">Check outcome</button>',
    '          <button type="button" class="clear-btn" id="clearBtn">Reset</button>',
    "        </div>",
    '        <div class="error-msg" id="errorMsg"></div>',
    "      </form>",
    "    </main>",
    "  </div>",
    "  <script>",
    "    const FEATURE_ORDER = {{ features | tojson }};",
    "    const FEATURE_LABELS = {{ labels | tojson }};",
    "    const form = document.getElementById('checkForm');",
    "    const errorMsg = document.getElementById('errorMsg');",
    "    document.querySelectorAll('.pill-group').forEach((group) => {",
    "      const hiddenInput = group.querySelector('input[type=\"hidden\"]');",
    "      const buttons = group.querySelectorAll('.pill-btn');",
    "      buttons.forEach((btn) => {",
    "        btn.addEventListener('click', () => {",
    "          buttons.forEach((b) => b.classList.remove('active'));",
    "          btn.classList.add('active');",
    "          hiddenInput.value = btn.getAttribute('data-val');",
    "        });",
    "      });",
    "    });",
    "    document.getElementById('clearBtn').addEventListener('click', () => {",
    "      form.reset();",
    "      document.querySelectorAll('.pill-group input[type=\"hidden\"]').forEach((inp) => { inp.value = ''; });",
    "      document.querySelectorAll('.pill-btn').forEach((btn) => btn.classList.remove('active'));",
    "      errorMsg.textContent = '';",
    "      document.getElementById('resultTitle').textContent = 'Your result appears here';",
    "      document.getElementById('resultSub').textContent = 'Fill in the form and select Check outcome.';",
    "      document.getElementById('resultBadgeWrap').innerHTML = '';",
    "      document.getElementById('probRows').innerHTML = '';",
    "    });",
    "    form.addEventListener('submit', async (e) => {",
    "      e.preventDefault();",
    "      errorMsg.textContent = '';",
    "      const payload = {};",
    "      for (const feat of FEATURE_ORDER) {",
    "        const el = document.getElementById(feat);",
    "        const raw = el ? el.value.trim() : '';",
    "        if (raw === '' || Number.isNaN(Number(raw)) || !Number.isFinite(Number(raw))) {",
    "          errorMsg.textContent = 'Please provide ' + (FEATURE_LABELS[feat] || feat) + '.';",
    "          return;",
    "        }",
    "        payload[feat] = Number(raw);",
    "      }",
    "      try {",
    "        const resp = await fetch('/predict', {",
    "          method: 'POST',",
    "          headers: { 'Content-Type': 'application/json' },",
    "          body: JSON.stringify(payload),",
    "        });",
    "        const data = await resp.json();",
    "        if (!resp.ok || !data.success) {",
    "          errorMsg.textContent = data.error || 'Could not run prediction.';",
    "          return;",
    "        }",
    "        document.getElementById('resultTitle').textContent = 'Predicted outcome';",
    "        document.getElementById('resultSub').textContent = 'Result from your decision tree model.';",
    "        document.getElementById('resultBadgeWrap').innerHTML = '<span class=\"result-badge\">Class ' + data.prediction + '</span>';",
    "        const probRows = document.getElementById('probRows');",
    "        probRows.innerHTML = '';",
    "        if (Array.isArray(data.probabilities)) {",
    "          data.probabilities.forEach((p) => {",
    "            const row = document.createElement('div');",
    "            row.className = 'prob-row';",
    "            row.innerHTML = '<span>Class ' + p.class + '</span><span>' + (p.probability * 100).toFixed(1) + '%</span>';",
    "            probRows.appendChild(row);",
    "          });",
    "        }",
    "      } catch (err) {",
    "        errorMsg.textContent = 'Network error while running prediction.';",
    "      }",
    "    });",
    "  </script>",
    "</body>",
    "</html>",
]

HTML_TEMPLATE = "\n".join(HTML_LINES)


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template_string(
        HTML_TEMPLATE,
        features=FEATURE_ORDER,
        labels=FEATURE_LABELS,
        hints=FEATURE_HINTS,
        categorical_options=CATEGORICAL_OPTIONS,
        sections=SECTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"success": False, "error": "Invalid request payload."}), 400

        X = build_feature_vector(payload)
        raw_pred = MODEL.predict(X)[0]
        pred_label = int(raw_pred) if isinstance(raw_pred, (np.integer, int)) else raw_pred

        probabilities_data = None
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(X)[0]
            probabilities_data = [
                {
                    "class": int(c) if isinstance(c, (np.integer, int)) else c,
                    "probability": float(p),
                }
                for c, p in zip(CLASSES, probs)
            ]

        return jsonify(
            {
                "success": True,
                "prediction": pred_label,
                "probabilities": probabilities_data,
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Inference error: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
