"""
preprocess.py
=============
Merges three raw disaster datasets into one training CSV.

Inputs (put in ml/data/raw/):
  - tsunami_events.csv    → NOAA NGDC Tsunami Events dataset
  - earthquake_usgs.csv   → USGS Earthquake Catalog (Japan region, mag ≥ 4.5)
  - typhoon_jma.csv       → JMA Best Track data (Kaggle cleaned version)

Output:
  - ml/data/processed/training_data.csv

Run:
  python preprocess.py

Schema of output CSV:
  magnitude            | float  | seismic magnitude (NaN for typhoon rows)
  depth_km             | float  | focal depth in km (0 for typhoon, NaN filled for tsunami)
  lat                  | float  | event latitude
  lng                  | float  | event longitude
  dist_to_coast_km     | float  | haversine distance to Sendai coast
  central_pressure_hpa | float  | storm central pressure (NaN for quake/tsunami rows)
  max_wind_knots       | float  | max sustained wind (NaN for quake/tsunami rows)
  wave_intensity       | float  | tsunami intensity scale (NaN for quake/typhoon rows)
  label                | str    | "earthquake" | "tsunami" | "typhoon"

WHY NaN is fine here:
  The NaN pattern is part of the signal — in production, you get real
  sensor inputs. If the seismic sensor fires, magnitude is populated;
  if the weather sensor fires, pressure/wind are populated. XGBoost
  handles NaN natively and learns this structure.
"""

import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2

# ── Config ──────────────────────────────────────────────────────────────────
COAST_LAT, COAST_LNG = 38.2688, 141.0251   # Sendai coastline reference point

RAW_DIR       = "ml/data/raw"
PROCESSED_DIR = "ml/data/processed"

SCHEMA = [
    "magnitude", "depth_km", "lat", "lng",
    "dist_to_coast_km", "central_pressure_hpa",
    "max_wind_knots", "wave_intensity", "label"
]

# ── Utility ──────────────────────────────────────────────────────────────────
def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance between two lat/lng points, in km."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── Loader 1: NOAA Tsunami ───────────────────────────────────────────────────
def load_tsunami(path):
    """
    Real column names from the NOAA NGDC tsunami dataset:
      LATITUDE, LONGITUDE, EQ_MAGNITUDE, EQ_DEPTH, TS_INTENSITY
    """
    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["magnitude"]            = df["EQ_MAGNITUDE"]   # 34.7% NaN, filled below
    out["depth_km"]             = df["EQ_DEPTH"]       # 59.8% NaN, filled below
    out["lat"]                  = df["LATITUDE"]
    out["lng"]                  = df["LONGITUDE"]
    out["wave_intensity"]       = df["TS_INTENSITY"]   # 51.3% NaN, filled below
    out["central_pressure_hpa"] = np.nan               # not applicable to tsunamis
    out["max_wind_knots"]       = np.nan               # not applicable to tsunamis
    out["dist_to_coast_km"]     = out.apply(
        lambda r: haversine_km(r["lat"], r["lng"], COAST_LAT, COAST_LNG), axis=1
    )
    out["label"] = "tsunami"
    return out[SCHEMA]


# ── Loader 2: USGS Earthquake ─────────────────────────────────────────────────
def load_earthquake(path):
    """
    Real column names from the USGS earthquake catalog CSV:
      latitude, longitude, depth, mag
    Note: this dataset has no 'sig' or 'tsunami' flag column.
    """
    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["magnitude"]            = df["mag"]
    out["depth_km"]             = df["depth"]
    out["lat"]                  = df["latitude"]
    out["lng"]                  = df["longitude"]
    out["wave_intensity"]       = np.nan   # not applicable to pure earthquakes
    out["central_pressure_hpa"] = np.nan   # not applicable
    out["max_wind_knots"]       = np.nan   # not applicable
    out["dist_to_coast_km"]     = out.apply(
        lambda r: haversine_km(r["lat"], r["lng"], COAST_LAT, COAST_LNG), axis=1
    )
    out["label"] = "earthquake"
    return out[SCHEMA]


# ── Loader 3: JMA Typhoon ────────────────────────────────────────────────────
def load_typhoon(path):
    """
    Real column names from the JMA Best Track CSV (Kaggle version):
      LAT, LON, PRES, WIND
    LAT/LON are already in real degrees (confirmed: LAT range 1.4–69.0).
    """
    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["magnitude"]            = np.nan           # not applicable to typhoons
    out["depth_km"]             = 0.0              # surface phenomenon
    out["lat"]                  = df["LAT"]
    out["lng"]                  = df["LON"]
    out["central_pressure_hpa"] = df["PRES"]       # hPa, range 870–1018
    out["max_wind_knots"]       = df["WIND"]       # knots, range 0–140
    out["wave_intensity"]       = np.nan           # not applicable
    out["dist_to_coast_km"]     = out.apply(
        lambda r: haversine_km(r["lat"], r["lng"], COAST_LAT, COAST_LNG), axis=1
    )
    out["label"] = "typhoon"
    return out[SCHEMA]


# ── Merge ─────────────────────────────────────────────────────────────────────
def build_training_data(
    tsunami_path    = f"{RAW_DIR}/tsunami_events.csv",
    earthquake_path = f"{RAW_DIR}/earthquake_usgs.csv",
    typhoon_path    = f"{RAW_DIR}/typhoon_jma.csv",
    output_path     = f"{PROCESSED_DIR}/training_data.csv",
):
    print("── Loading raw datasets ──────────────────────────────────")
    df_t  = load_tsunami(tsunami_path)
    df_e  = load_earthquake(earthquake_path)
    df_ty = load_typhoon(typhoon_path)

    print(f"  Tsunami rows   : {len(df_t):,}")
    print(f"  Earthquake rows: {len(df_e):,}")
    print(f"  Typhoon rows   : {len(df_ty):,}")

    # Vertical stack — this is the "merge" (UNION, not JOIN)
    combined = pd.concat([df_t, df_e, df_ty], ignore_index=True)
    print(f"\n  Combined (raw) : {len(combined):,} rows")

    # ── Fill NaN with LABEL-WISE median ──────────────────────────────
    # e.g. tsunami magnitude NaN → filled with tsunami magnitude median (7.0)
    #      NOT with the global median which would mix all three event types
    fill_cols = [
        "magnitude", "depth_km", "wave_intensity",
        "central_pressure_hpa", "max_wind_knots"
    ]
    print("\n── Filling NaN with label-wise medians ──────────────────")
    for label in combined["label"].unique():
        mask = combined["label"] == label
        for col in fill_cols:
            n_nan = combined.loc[mask, col].isna().sum()
            if n_nan == 0:
                continue
            med = combined.loc[mask, col].median()
            if pd.isna(med):
                # Column is structurally all-NaN for this label (e.g. pressure for
                # earthquake rows). Leave as NaN — XGBoost handles this natively.
                continue
            combined.loc[mask, col] = combined.loc[mask, col].fillna(med)
            print(f"  [{label}] {col}: {n_nan:,} NaN → filled with median={med:.2f}")

    # Drop rows missing location (can't compute distance without these)
    combined = combined.dropna(subset=["lat", "lng", "dist_to_coast_km"])

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n── Final dataset: {len(combined):,} rows ───────────────────")
    print("\nLabel distribution:")
    vc = combined["label"].value_counts()
    for lbl, cnt in vc.items():
        pct = cnt / len(combined) * 100
        print(f"  {lbl:<12}: {cnt:>6,}  ({pct:.1f}%)")

    ratio = vc.max() / vc.min()
    print(f"\nImbalance ratio (max/min): {ratio:.1f}x")
    if ratio > 5:
        print("  ⚠  Use sample_weight='balanced' in training (already done in train.py)")

    print("\nRemaining NaN per column (structurally expected):")
    nans = combined.isnull().sum()
    nans = nans[nans > 0]
    for col, n in nans.items():
        pct = n / len(combined) * 100
        print(f"  {col:<25}: {n:,}  ({pct:.1f}%)")
    if len(nans) == 0:
        print("  None")

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    return combined


if __name__ == "__main__":
    build_training_data()