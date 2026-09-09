## Evacuation Agent (Sendai)

Dark-mode web UI + FastAPI backend that:

- Detects/classifies calamities (`earthquake`, `tsunami`, `typhoon`, `none`) using an XGBoost classifier.
- Computes severity using a physics-based baseline plus a small learned residual.
- Routes to an appropriate evacuation camp using A*/UCS/BFS/DFS over a prebuilt Sendai graph.

---

## Running the app (Windows)

### Backend (FastAPI)

From the repo root:

```powershell
.\venv\Scripts\uvicorn backend.main:app --reload --port 8000
```

### Frontend (Vite)

From the repo root:

```powershell
cd frontend
npm install
npm run dev
```

The frontend uses a Vite proxy (see `frontend/vite.config.js`) to call the backend.

---

## Why you were seeing “only tsunami”

The classifier was trained with **class-specific NaN patterns** (features that don’t apply to a calamity are `NaN` in training). XGBoost learns those missingness patterns as part of the signal.

At inference time:

- Sending `0` for “not applicable” is **not** the same as sending `NaN`.
- The previous UI defaults (`magnitude=0`, `wind=0`, `wave=0`, `pressure≈1013`) were being converted into an ambiguous all-`NaN` pattern, which can cause a consistent default class.

Fix implemented:

- The frontend now sends **`null` for untouched hazard fields**, and the backend treats “no hazard signal” as `calamity="none"` instead of forcing a class.

---

## UCS vs A* “same cost” (expected)

UCS and A* are both **optimal** shortest-path algorithms (with the heuristic used here, A* is admissible).  
So **path cost should match**; A* is typically faster because it expands fewer nodes.

---

## ML retraining + imbalance notes

Your dataset is highly imbalanced (e.g., typhoon ≫ tsunami). The training script already mitigates this by:

- **Subsampling the majority class** to cap each class at ~\(3 \times\) the minority count
- Using `compute_sample_weight("balanced", ...)` on the remaining imbalance

### Retrain workflow

1) Put raw CSVs in `ml/data/raw/` (see `ml/preprocess.py` docstring for filenames/columns).

2) Build the merged training file:

```powershell
.\venv\Scripts\python ml\preprocess.py
```

3) Train the classifier:

```powershell
.\venv\Scripts\python ml\train.py
```

4) Copy the artifact into the backend:

- From `ml/models/calamity_classifier.pkl`
- To `backend/data/calamity_classifier.pkl`

Similarly, if you retrain severity:

- `ml/models/severity_scaler.pkl` → `backend/data/severity_scaler.pkl`

---

## Key files

- **Backend ML inference**: `backend/services/ml_service.py`
- **Calamity endpoints**: `backend/routers/calamity.py`
- **Routing + blocked road labels**: `backend/services/graph_service.py`
- **Manual input UI**: `frontend/src/components/ManualSelector.jsx`
