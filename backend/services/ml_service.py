"""
backend/services/ml_service.py

Robust calamity prediction service.
- Tries to load trained ML model first
- Falls back to physics-informed rule engine that ACTUALLY varies with inputs
- Probabilities are computed dynamically — no hardcoded 80/15/5
"""

import os
import math
import logging

logger = logging.getLogger(__name__)

# ── Try loading the trained model ─────────────────────────────────────────────
_model       = None
_label_enc   = None
_scaler      = None
_model_error = None

try:
    import joblib
    _BASE = os.path.dirname(__file__)

    # Common paths — adjust if your model is elsewhere
    _model_candidates = [
        os.path.join(_BASE, "../ml/models/calamity_model.joblib"),
        os.path.join(_BASE, "../ml/models/calamity_model.pkl"),
        os.path.join(_BASE, "calamity_model.joblib"),
        os.path.join(_BASE, "calamity_model.pkl"),
    ]
    _scaler_candidates = [
        os.path.join(_BASE, "../ml/models/scaler.joblib"),
        os.path.join(_BASE, "../ml/models/scaler.pkl"),
    ]

    for path in _model_candidates:
        if os.path.exists(path):
            _model = joblib.load(path)
            logger.info(f"✅ ML model loaded from {path}")
            break

    for path in _scaler_candidates:
        if os.path.exists(path):
            _scaler = joblib.load(path)
            logger.info(f"✅ Scaler loaded")
            break

    if _model is None:
        logger.warning("⚠️  No model file found — using rule-based engine")

except Exception as e:
    _model_error = str(e)
    logger.warning(f"⚠️  Model load failed ({e}) — using rule-based engine")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _softmax(scores: dict) -> dict:
    """Convert raw scores to probabilities that sum to 1."""
    vals = list(scores.values())
    max_v = max(vals)
    exps = {k: math.exp(v - max_v) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: round(v / total, 4) for k, v in exps.items()}


def _severity_label(severity: float) -> str:
    if severity < 0.25:
        return "low"
    elif severity < 0.55:
        return "moderate"
    elif severity < 0.80:
        return "high"
    return "critical"


def _description(calamity: str, severity: float, label: str) -> str:
    pct = int(severity * 100)
    descs = {
        "tsunami": {
            "low":      "Minor coastal disturbance. Monitor alerts.",
            "moderate": f"Tsunami warning! Move inland or to high ground immediately.",
            "high":     f"Severe tsunami threat ({pct}%). Evacuate coastal zones NOW.",
            "critical": f"CRITICAL tsunami ({pct}%). Immediate inland evacuation required.",
        },
        "earthquake": {
            "low":      "Minor earthquake detected. No immediate evacuation required.",
            "moderate": f"Moderate earthquake ({pct}%). Check for structural damage.",
            "high":     f"Strong earthquake ({pct}%). Evacuate damaged buildings.",
            "critical": f"Major earthquake ({pct}%). Full evacuation protocol activated.",
        },
        "typhoon": {
            "low":      "Tropical disturbance. Stay informed.",
            "moderate": f"Typhoon approaching ({pct}%). Secure loose items, prepare to shelter.",
            "high":     f"Strong typhoon ({pct}%). Evacuate to designated shelters immediately.",
            "critical": f"Super typhoon ({pct}%). Evacuate to designated shelters immediately.",
        },
        "none": {
            "low": "No active calamity detected. All clear.",
        },
    }
    return descs.get(calamity, {}).get(label, f"{calamity.title()} detected. Follow local guidance.")


# ── Rule-based engine (dynamic, physics-informed) ─────────────────────────────

def _rule_based_predict(
    magnitude=None,
    depth_km=10.0,
    lat=38.27,
    lng=140.9,
    central_pressure_hpa=None,
    max_wind_knots=None,
    wave_intensity=None,
    dist_to_coast_km=None,
) -> dict:
    """
    Compute calamity probabilities from raw sensor inputs.
    Each feature independently scores each class — no hardcoded outputs.
    """
    mag   = float(magnitude or 0)
    depth = float(depth_km or 10)
    press = float(central_pressure_hpa or 1013)
    wind  = float(max_wind_knots or 0)
    wave  = float(wave_intensity or 0)

    # Raw log-odds scores (start neutral)
    scores = {"earthquake": 0.0, "tsunami": 0.0, "typhoon": 0.0}

    # ── Earthquake signals ────────────────────────────────────────────────────
    if mag > 0:
        # Strong quake boosts earthquake score sharply
        scores["earthquake"] += (mag - 4.0) * 1.8        # 0 at mag4, +10.8 at mag10
        # Shallow quakes are more damaging
        if depth < 30:
            scores["earthquake"] += 1.5
        elif depth < 70:
            scores["earthquake"] += 0.5
        else:
            scores["earthquake"] -= 0.5                   # deep = less surface damage

    # ── Tsunami signals ───────────────────────────────────────────────────────
    if wave > 0:
        scores["tsunami"] += wave * 0.9                   # 0→0, 8.5→7.65, 15→13.5
        scores["earthquake"] += wave * 0.2                # often co-occurs with quake

    if mag >= 7.0 and depth < 50:
        # Shallow high-mag quake near coast → strong tsunami signal
        scores["tsunami"] += (mag - 7.0) * 2.5
        if dist_to_coast_km is not None and dist_to_coast_km < 150:
            scores["tsunami"] += (150 - dist_to_coast_km) / 30

    # ── Typhoon signals ───────────────────────────────────────────────────────
    if wind > 0:
        # Typhoon threshold ~34 knots; super typhoon ~130 knots
        scores["typhoon"] += max(0, (wind - 20) / 15)     # 0 at 20kt, ~7.3 at 130kt

    if press < 1013:
        drop = 1013 - press
        scores["typhoon"] += drop / 20                    # 0 at 1013, 6.15 at 890hPa

    # ── Cross-suppression ─────────────────────────────────────────────────────
    # If it's clearly a typhoon, suppress earthquake
    if wind > 60 or press < 960:
        scores["earthquake"] -= 2.0
        scores["tsunami"]    -= 1.5

    # If strong quake and no typhoon signals, suppress typhoon
    if mag > 6 and wind < 20 and press > 1000:
        scores["typhoon"] -= 3.0

    # ── "None" baseline ───────────────────────────────────────────────────────
    # If all signals are weak, pull everything toward zero/none
    all_weak = mag < 3 and wind < 20 and wave < 1 and press > 1005
    if all_weak:
        scores = {k: v - 4.0 for k, v in scores.items()}

    # Convert to probabilities
    probs = _softmax(scores)

    # Determine winner
    calamity = max(probs, key=probs.get)
    top_prob  = probs[calamity]

    # If nothing is confident enough → "none"
    if top_prob < 0.45 and all_weak:
        calamity  = "none"
        severity  = 0.0
        sev_label = "low"
        probs     = {"earthquake": 0.05, "tsunami": 0.02, "typhoon": 0.03}
    else:
        # Severity = weighted combo of top probability + raw signal strength
        if calamity == "earthquake":
            raw_sev = min(1.0, max(0, (mag - 3) / 6))
        elif calamity == "tsunami":
            raw_sev = min(1.0, max(wave / 12, (mag - 6) / 4 if mag > 6 else 0))
        elif calamity == "typhoon":
            raw_sev = min(1.0, max((wind - 20) / 130, (1013 - press) / 130))
        else:
            raw_sev = 0.0

        severity  = round(top_prob * 0.5 + raw_sev * 0.5, 3)
        sev_label = _severity_label(severity)

    return {
        "calamity":       calamity,
        "severity":       severity,
        "severity_label": sev_label,
        "source":         "rule_based",
        "probabilities":  probs,
        "description":    _description(calamity, severity, sev_label),
    }


# ── ML model inference ────────────────────────────────────────────────────────

def _ml_predict(
    magnitude=None, depth_km=10.0, lat=38.27, lng=140.9,
    central_pressure_hpa=None, max_wind_knots=None,
    wave_intensity=None, dist_to_coast_km=None,
) -> dict:
    """Run the trained sklearn/joblib model."""
    import numpy as np

    features = [
        float(magnitude or 0),
        float(depth_km or 10),
        float(lat),
        float(lng),
        float(central_pressure_hpa or 1013),
        float(max_wind_knots or 0),
        float(wave_intensity or 0),
        float(dist_to_coast_km or 50),
    ]

    X = np.array([features])
    if _scaler is not None:
        X = _scaler.transform(X)

    # Probabilities per class
    proba      = _model.predict_proba(X)[0]
    classes    = list(_model.classes_)
    probs      = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}

    calamity   = str(classes[proba.argmax()])
    top_prob   = float(proba.max())
    severity   = round(top_prob * 0.85, 3)
    sev_label  = _severity_label(severity)

    return {
        "calamity":       calamity,
        "severity":       severity,
        "severity_label": sev_label,
        "source":         "ml_model",
        "probabilities":  probs,
        "description":    _description(calamity, severity, sev_label),
    }


# ── Public API ────────────────────────────────────────────────────────────────

class MLService:
    def predict(
        self,
        magnitude=None,
        depth_km=10.0,
        lat=38.27,
        lng=140.9,
        central_pressure_hpa=None,
        max_wind_knots=None,
        wave_intensity=None,
        dist_to_coast_km=None,
    ) -> dict:
        if _model is not None:
            try:
                result = _ml_predict(
                    magnitude=magnitude, depth_km=depth_km,
                    lat=lat, lng=lng,
                    central_pressure_hpa=central_pressure_hpa,
                    max_wind_knots=max_wind_knots,
                    wave_intensity=wave_intensity,
                    dist_to_coast_km=dist_to_coast_km,
                )
                return result
            except Exception as e:
                logger.warning(f"ML inference failed ({e}), falling back to rules")

        return _rule_based_predict(
            magnitude=magnitude, depth_km=depth_km,
            lat=lat, lng=lng,
            central_pressure_hpa=central_pressure_hpa,
            max_wind_knots=max_wind_knots,
            wave_intensity=wave_intensity,
            dist_to_coast_km=dist_to_coast_km,
        )