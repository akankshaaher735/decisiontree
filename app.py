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
        model = pickle.load(file)
    return model


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


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>DecisionTree Inference Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-1: #050816;
      --bg-2: #120b2e;
      --bg-3: #0b2545;
      --accent-blue: #3b82f6;
      --accent-purple: #8b5cf6;
      --accent-cyan: #22d3ee;
      --glass-bg: rgba(15, 23, 52, 0.58);
      --glass-border: rgba(125, 211, 252, 0.22);
      --glass-highlight: rgba(139, 92, 246, 0.28);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --input-bg: rgba(8, 13, 32, 0.72);
      --input-border: rgba(148, 163, 184, 0.24);
      --shadow-card: 0 20px 50px rgba(2, 6, 23, 0.65), 0 0 28px rgba(59, 130, 246, 0.14);
      --success-bg: rgba(16, 185, 129, 0.15);
      --success-border: rgba(52, 211, 153, 0.45);
      --success-text: #6ee7b7;
      --error-bg: rgba(244, 63, 94, 0.15);
      --error-border: rgba(251, 113, 133, 0.45);
      --error-text: #fda4af;
    }

    html[data-theme="light"] {
      --bg-1: #eef2ff;
      --bg-2: #e0e7ff;
      --bg-3: #cffafe;
      --glass-bg: rgba(255, 255, 255, 0.68);
      --glass-border: rgba(59, 130, 246, 0.25);
      --glass-highlight: rgba(124, 58, 237, 0.18);
      --text-main: #0f172a;
      --text-muted: #475569;
      --input-bg: rgba(255, 255, 255, 0.85);
      --input-border: rgba(71, 85, 105, 0.25);
      --shadow-card: 0 18px 40px rgba(30, 41, 59, 0.12), 0 0 24px rgba(59, 130, 246, 0.12);
      --success-bg: rgba(16, 185, 129, 0.14);
      --success-border: rgba(5, 150, 105, 0.4);
      --success-text: #065f46;
      --error-bg: rgba(225, 29, 72, 0.12);
      --error-border: rgba(225, 29, 72, 0.35);
      --error-text: #9f1239;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text-main);
      min-height: 100vh;
      background: linear-gradient(135deg, var(--bg-1), var(--bg-2), var(--bg-3));
      background-size: 220% 220%;
      animation: gradientShift 18s ease infinite;
      overflow-x: hidden;
      transition: color 0.3s ease, background 0.4s ease;
    }

    @keyframes gradientShift {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    #particles-canvas {
      position: fixed;
      inset: 0;
      width: 100%;
      height: 100%;
      z-index: 0;
      pointer-events: none;
      opacity: 0.55;
    }

    .app-shell {
      position: relative;
      z-index: 1;
      max-width: 1280px;
      margin: 0 auto;
      padding: 20px 20px 48px;
      animation: pageEnter 0.65s cubic-bezier(0.22, 1, 0.36, 1) both;
    }

    @keyframes pageEnter {
      from {
        opacity: 0;
        transform: translateY(16px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    /* Navigation Bar */
    .navbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 22px;
      border-radius: 20px;
      background: var(--glass-bg);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border: 1px solid var(--glass-border);
      box-shadow: var(--shadow-card);
      margin-bottom: 26px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-icon {
      width: 40px;
      height: 40px;
      border-radius: 12px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple), var(--accent-cyan));
      box-shadow: 0 0 18px rgba(34, 211, 238, 0.4);
      animation: gentleFloat 4.5s ease-in-out infinite;
    }

    @keyframes gentleFloat {
      0%, 100% { transform: translateY(0px); }
      50% { transform: translateY(-4px); }
    }

    .brand-title {
      font-weight: 800;
      font-size: 1.1rem;
      letter-spacing: -0.02em;
    }

    .brand-sub {
      font-size: 0.78rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }

    .nav-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .pill-badge {
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
      background: rgba(59, 130, 246, 0.14);
      border: 1px solid rgba(34, 211, 238, 0.35);
      color: var(--accent-cyan);
    }

    html[data-theme="light"] .pill-badge {
      color: #1d4ed8;
    }

    /* Glass Cards */
    .glass-card {
      background: var(--glass-bg);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border: 1px solid var(--glass-border);
      border-radius: 22px;
      padding: 24px;
      box-shadow: var(--shadow-card);
      transition: transform 0.28s ease, border-color 0.28s ease, box-shadow 0.28s ease;
    }

    .glass-card:hover {
      transform: translateY(-4px);
      border-color: rgba(34, 211, 238, 0.45);
      box-shadow: 0 24px 56px rgba(2, 6, 23, 0.72), 0 0 32px var(--glass-highlight);
    }

    .hero-banner {
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
    }

    .hero-banner::after {
      content: "";
      position: absolute;
      top: -60px;
      right: -60px;
      width: 200px;
      height: 200px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(34, 211, 238, 0.22), transparent 70%);
      pointer-events: none;
    }

    .hero-banner h1 {
      font-size: clamp(1.45rem, 3vw, 2.1rem);
      font-weight: 800;
      letter-spacing: -0.025em;
      margin-bottom: 8px;
      background: linear-gradient(90deg, var(--text-main), var(--accent-cyan), #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    html[data-theme="light"] .hero-banner h1 {
      background: linear-gradient(90deg, #0f172a, #2563eb, #7c3aed);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .hero-banner p {
      color: var(--text-muted);
      font-size: 0.94rem;
      line-height: 1.55;
      max-width: 820px;
    }

    /* Dashboard Metrics */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }

    .metric-card {
      padding: 20px;
      border-radius: 18px;
      background: var(--glass-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--glass-border);
      box-shadow: var(--shadow-card);
      transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .metric-card:hover {
      transform: translateY(-3px);
      border-color: rgba(139, 92, 246, 0.5);
    }

    .metric-label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      font-weight: 600;
      margin-bottom: 8px;
    }

    .metric-value {
      font-size: 1.9rem;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent-cyan);
    }

    html[data-theme="light"] .metric-value {
      color: #4f46e5;
    }

    /* Main Two-Column Workspace */
    .workspace-grid {
      display: grid;
      grid-template-columns: 1.35fr 0.95fr;
      gap: 22px;
      align-items: start;
      margin-bottom: 24px;
    }

    @media (max-width: 980px) {
      .workspace-grid {
        grid-template-columns: 1fr;
      }
    }

    .section-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 18px;
    }

    .section-title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 1.12rem;
      font-weight: 700;
    }

    .features-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(225px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }

    .field-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .field-label {
      font-size: 0.8rem;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-main);
    }

    .field-hint {
      font-size: 0.72rem;
      color: var(--text-muted);
    }

    .field-input {
      width: 100%;
      padding: 11px 13px;
      border-radius: 12px;
      border: 1px solid var(--input-border);
      background: var(--input-bg);
      color: var(--text-main);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.88rem;
      outline: none;
      transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
    }

    .field-input:focus {
      border-color: var(--accent-cyan);
      box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.22), 0 0 14px rgba(59, 130, 246, 0.25);
      transform: translateY(-1px);
    }

    /* Buttons */
    .btn-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 11px 18px;
      border-radius: 13px;
      border: 1px solid transparent;
      font-family: 'Inter', sans-serif;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.25s ease, background 0.25s ease, border-color 0.25s ease;
    }

    .btn:hover {
      transform: translateY(-2px);
    }

    .btn:active {
      transform: translateY(0px) scale(0.98);
    }

    .btn-primary {
      color: #ffffff;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      box-shadow: 0 8px 22px rgba(79, 70, 229, 0.38), 0 0 14px rgba(34, 211, 238, 0.25);
    }

    .btn-primary:hover {
      box-shadow: 0 12px 28px rgba(79, 70, 229, 0.5), 0 0 20px rgba(34, 211, 238, 0.4);
    }

    .btn-ghost {
      color: var(--text-main);
      background: rgba(148, 163, 184, 0.12);
      border-color: var(--glass-border);
    }

    .btn-ghost:hover {
      border-color: var(--accent-cyan);
      background: rgba(34, 211, 238, 0.12);
    }

    .btn-danger {
      color: #fda4af;
      background: rgba(244, 63, 94, 0.14);
      border-color: rgba(251, 113, 133, 0.35);
    }

    html[data-theme="light"] .btn-danger {
      color: #9f1239;
    }

    /* Result Card & States */
    .result-panel {
      min-height: 360px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: opacity 0.3s ease, transform 0.3s ease;
    }

    .result-placeholder {
      text-align: center;
      padding: 44px 18px;
      color: var(--text-muted);
      animation: fadeState 0.35s ease;
    }

    .result-box {
      display: none;
      animation: fadeState 0.4s cubic-bezier(0.22, 1, 0.36, 1);
    }

    .result-box.active {
      display: block;
    }

    @keyframes fadeState {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .prediction-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 16px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 1rem;
      font-family: 'JetBrains Mono', monospace;
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(34, 211, 238, 0.2));
      border: 1px solid rgba(34, 211, 238, 0.55);
      box-shadow: 0 0 20px rgba(34, 211, 238, 0.25);
      margin-bottom: 14px;
    }

    .confidence-box {
      padding: 14px 16px;
      border-radius: 14px;
      background: var(--input-bg);
      border: 1px solid var(--glass-border);
      margin-bottom: 18px;
    }

    .confidence-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      margin-bottom: 8px;
      font-weight: 600;
    }

    .prob-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .prob-item-head {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      font-family: 'JetBrains Mono', monospace;
      margin-bottom: 5px;
    }

    .progress-track {
      width: 100%;
      height: 10px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.18);
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-cyan));
      transition: width 0.65s cubic-bezier(0.22, 1, 0.36, 1);
      box-shadow: 0 0 12px rgba(34, 211, 238, 0.45);
    }

    /* Toast Notification */
    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 60;
      max-width: 400px;
      padding: 14px 18px;
      border-radius: 14px;
      font-size: 0.86rem;
      font-weight: 500;
      backdrop-filter: blur(16px);
      border: 1px solid transparent;
      box-shadow: 0 14px 32px rgba(2, 6, 23, 0.45);
      opacity: 0;
      transform: translateY(14px);
      pointer-events: none;
      transition: opacity 0.28s ease, transform 0.28s ease;
    }

    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }

    .toast.success {
      background: var(--success-bg);
      border-color: var(--success-border);
      color: var(--success-text);
    }

    .toast.error {
      background: var(--error-bg);
      border-color: var(--error-border);
      color: var(--error-text);
    }

    /* Loading Overlay */
    .loading-overlay {
      position: fixed;
      inset: 0;
      z-index: 50;
      display: none;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 16px;
      background: rgba(5, 8, 22, 0.72);
      backdrop-filter: blur(10px);
    }

    .loading-overlay.active {
      display: flex;
    }

    .ai-orb {
      width: 64px;
      height: 64px;
      border-radius: 50%;
      background: conic-gradient(from 0deg, var(--accent-cyan), var(--accent-purple), var(--accent-blue), var(--accent-cyan));
      padding: 4px;
      animation: spinOrb 1.1s linear infinite;
      box-shadow: 0 0 28px rgba(34, 211, 238, 0.55);
    }

    .ai-orb-inner {
      width: 100%;
      height: 100%;
      border-radius: 50%;
      background: var(--bg-1);
    }

    @keyframes spinOrb {
      to { transform: rotate(360deg); }
    }

    /* History Table */
    .table-wrap {
      overflow-x: auto;
      border-radius: 14px;
      border: 1px solid var(--glass-border);
      background: var(--input-bg);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.83rem;
    }

    th, td {
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid rgba(148, 163, 184, 0.14);
      font-family: 'JetBrains Mono', monospace;
    }

    th {
      font-size: 0.75rem;
      text-transform: uppercase;
      color: var(--text-muted);
      background: rgba(59, 130, 246, 0.08);
    }

    tr:last-child td {
      border-bottom: none;
    }

    .footer {
      margin-top: 28px;
      padding: 18px 22px;
      border-radius: 18px;
      background: var(--glass-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--glass-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 0.82rem;
      color: var(--text-muted);
    }
  </style>
</head>
<body>
  <canvas id="particles-canvas"></canvas>

  <div class="loading-overlay" id="loadingOverlay" aria-hidden="true">
    <div class="ai-orb"><div class="ai-orb-inner"></div></div>
    <div style="font-weight: 600; letter-spacing: 0.03em; color: #e2e8f0;">Running DecisionTree Inference...</div>
  </div>

  <div class="toast" id="toastBox" role="status" aria-live="polite"></div>

  <div class="app-shell">
    <nav class="navbar">
      <div class="brand">
        <div class="brand-icon" aria-hidden="true">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3v6"></path>
            <path d="M12 15v6"></path>
            <path d="M5 12h14"></path>
            <circle cx="12" cy="12" r="3"></circle>
          </svg>
        </div>
        <div>
          <div class="brand-title">DecisionTree AI Studio</div>
          <div class="brand-sub">model: decision.pkl • {{ model_name }}</div>
        </div>
      </div>

      <div class="nav-actions">
        <span class="pill-badge">{{ feature_count }} Features • {{ class_count }} Classes</span>
        <button type="button" class="btn btn-ghost" id="themeToggle" title="Toggle Dark/Light Theme">
          <span id="themeIcon">☀️</span>
          <span id="themeLabel">Light</span>
        </button>
      </div>
    </nav>

    <section class="glass-card hero-banner">
      <h1>Real-Time Decision Tree Inference Engine</h1>
      <p>
        Powered directly by your pre-trained <strong>DecisionTreeClassifier</strong> (<code>decision.pkl</code>) without retraining or altering weights.
        Enter numeric feature values in the exact model order to inspect class predictions and probability distributions.
      </p>
    </section>

    <section class="metrics-grid" id="dashboardMetrics">
      <div class="metric-card">
        <div class="metric-label">Total Session Predictions</div>
        <div class="metric-value" id="totalPredictionsCounter" data-target="{{ total_predictions }}">{{ total_predictions }}</div>
      </div>
      {% for cls, count in class_counts.items() %}
      <div class="metric-card">
        <div class="metric-label">Class {{ cls }} Predictions</div>
        <div class="metric-value class-counter" data-class="{{ cls }}" data-target="{{ count }}">{{ count }}</div>
      </div>
      {% endfor %}
    </section>

    <div class="workspace-grid">
      <section class="glass-card">
        <div class="section-header">
          <div class="section-title">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
