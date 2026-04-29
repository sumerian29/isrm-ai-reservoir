# -*- coding: utf-8 -*-
"""
ISRM Advanced Stable – Iraqi Smart Reservoir Manager
نظام عملي مستقر لإدارة المكامن النفطية العراقية باستخدام الذكاء الاصطناعي

Run:
    pip install streamlit pandas numpy scikit-learn plotly openpyxl xlsxwriter reportlab
    streamlit run app.py

Optional advanced packages:
    pip install xgboost deap tensorflow shap

Self-test:
    python app.py --test

Required Excel/CSV columns:
    Date, Well_Name, Oil_Rate, Water_Cut, BHP, THP, Permeability, Net_Pay
Optional columns:
    Intervention, Status
"""

from __future__ import annotations

import io
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import streamlit as st
    STREAMLIT = True
except Exception:
    STREAMLIT = False
    st = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY = True
except Exception:
    PLOTLY = False

try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN = True
except Exception:
    SKLEARN = False

try:
    import xgboost as xgb
    XGB = True
except Exception:
    XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF = True
except Exception:
    TF = False

try:
    import shap
    SHAP = True
except Exception:
    SHAP = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB = True
except Exception:
    REPORTLAB = False


REQUIRED_COLUMNS = [
    "Date", "Well_Name", "Oil_Rate", "Water_Cut", "BHP", "THP", "Permeability", "Net_Pay"
]

FEATURE_COLUMNS = [
    "BHP", "THP", "Water_Cut", "Permeability", "Net_Pay",
    "Oil_7D_MA", "Oil_30D_MA", "Pressure_Decline", "WC_Trend", "PI"
]

DISPLAY_COLUMNS = [
    "Well_Name", "Date", "Oil_Rate", "Water_Cut", "BHP", "THP",
    "Permeability", "Net_Pay", "WII", "AI_Status", "Anomaly", "Recommendation"
]

OPT_COLUMNS = [
    "Well_Name", "Oil_Rate", "Proposed_Rate", "Delta", "Water_Cut", "BHP",
    "WII", "Optimization_Action", "Recommendation"
]


def cache_resource(func):
    if STREAMLIT:
        return st.cache_resource(show_spinner=False)(func)
    return func


# ============================================================================
# 1) Data generation
# ============================================================================
def generate_sample_data(days: int = 730, wells: int = 12, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_date = datetime.today() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    rows: List[Dict[str, object]] = []

    for w in range(1, wells + 1):
        well = f"Ns-{w:02d}"
        base_oil = rng.integers(250, 1800)
        decline = rng.uniform(0.00025, 0.0015)
        base_bhp = rng.integers(2400, 4200)
        water_start = rng.uniform(12, 45)
        perm = int(rng.integers(40, 350))
        net_pay = float(rng.uniform(8, 35))

        for i, d in enumerate(dates):
            seasonal = 75 * np.sin(i / 45.0)
            oil_noise = rng.normal(0, 55)
            oil = max(15, base_oil * np.exp(-decline * i) + seasonal + oil_noise)
            bhp = max(850, base_bhp - i * rng.uniform(0.25, 0.85) + rng.normal(0, 35))
            thp = max(120, bhp * rng.uniform(0.25, 0.43) + rng.normal(0, 15))
            wc = min(97, water_start + i * rng.uniform(0.012, 0.052) + rng.normal(0, 1.8))

            intervention = "None"
            if rng.random() < 0.003:
                intervention = str(rng.choice(["Workover", "Stimulation", "Choke Change"]))

            status = "Active"
            if oil < 120 or wc > 88 or bhp < 1300:
                status = "Critical"
            elif oil < 350 or wc > 70 or bhp < 1900:
                status = "Watch"

            rows.append({
                "Date": d,
                "Well_Name": well,
                "Oil_Rate": round(float(oil), 2),
                "Water_Cut": round(float(wc), 2),
                "BHP": round(float(bhp), 2),
                "THP": round(float(thp), 2),
                "Permeability": perm,
                "Net_Pay": round(net_pay, 2),
                "Intervention": intervention,
                "Status": status,
            })

    return pd.DataFrame(rows)


# ============================================================================
# 2) Data cleaning and features
# ============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("No data provided. Upload Excel/CSV or use sample data.")

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Well_Name"] = out["Well_Name"].astype(str).str.strip()

    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in ["Date", "Well_Name"]]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Date", "Well_Name"])
    out = out.sort_values(["Well_Name", "Date"])

    # Clip outliers per column but safely
    for col in numeric_cols:
        if out[col].notna().sum() > 5:
            low = out[col].quantile(0.01)
            high = out[col].quantile(0.99)
            out[col] = out[col].clip(lower=low, upper=high)

    for col in numeric_cols:
        out[col] = out.groupby("Well_Name")[col].transform(lambda s: s.interpolate().ffill().bfill())

    # Final fill if a whole well column was missing
    for col in numeric_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    out["Intervention"] = out.get("Intervention", "None")
    out["Intervention"] = out["Intervention"].fillna("None").astype(str)

    out["Status"] = out.get("Status", "Unknown")
    out["Status"] = out["Status"].fillna("Unknown").astype(str)

    if out.empty:
        raise ValueError("Data became empty after cleaning. Check Date and Well_Name columns.")

    return out


def normalize_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.isna().all():
        return pd.Series(70.0, index=series.index)
    s = s.fillna(s.median())
    min_v = float(s.min())
    max_v = float(s.max())
    if np.isclose(max_v, min_v):
        return pd.Series(70.0, index=series.index)
    score = 100.0 * (s - min_v) / (max_v - min_v)
    if not higher_is_better:
        score = 100.0 - score
    return score.clip(0, 100)


def recommendation_rule(row: pd.Series) -> str:
    if row["Water_Cut"] >= 85 and row["BHP"] < 1800:
        return "إغلاق مؤقت أو تدخل عاجل"
    if row["Water_Cut"] >= 75:
        return "مراقبة نسبة الماء ودراسة عزل مائي"
    if row["Oil_Rate"] < 200 and row["BHP"] > 2200:
        return "دراسة تحفيز أو Workover"
    if row["WII"] >= 80:
        return "استمرار التشغيل الحالي"
    if row["WII"] < 40:
        return "مراجعة فنية عاجلة"
    return "مراقبة دورية وتحسين تدريجي"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["Well_Name", "Date"])

    out["Oil_7D_MA"] = out.groupby("Well_Name")["Oil_Rate"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["Oil_30D_MA"] = out.groupby("Well_Name")["Oil_Rate"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    out["Pressure_Decline"] = out.groupby("Well_Name")["BHP"].diff().fillna(0)
    out["WC_Trend"] = out.groupby("Well_Name")["Water_Cut"].diff().fillna(0)
    out["PI"] = (out["Oil_Rate"] / out["BHP"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)

    out["Prod_Score"] = normalize_score(out["Oil_Rate"], True)
    out["Pressure_Score"] = normalize_score(out["BHP"], True)
    out["WC_Score"] = normalize_score(out["Water_Cut"], False)
    out["Stability_Score"] = normalize_score(out["Pressure_Decline"].abs(), False)
    out["Intervention_Score"] = np.where(out["Intervention"].str.lower().eq("none"), 100, 60)

    out["WII"] = (
        0.30 * out["Prod_Score"]
        + 0.25 * out["Pressure_Score"]
        + 0.20 * out["WC_Score"]
        + 0.15 * out["Stability_Score"]
        + 0.10 * out["Intervention_Score"]
    ).round(2)

    out["AI_Status"] = pd.cut(
        out["WII"],
        bins=[-0.01, 40, 60, 80, 100.01],
        labels=["Critical", "Watch", "Stable", "Excellent"],
        include_lowest=True,
    ).astype(str)

    out["Recommendation"] = out.apply(recommendation_rule, axis=1)
    return out


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("Date").groupby("Well_Name", as_index=False).tail(1).reset_index(drop=True)


# ============================================================================
# 3) Anomaly Detection
# ============================================================================
def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    if not SKLEARN or len(out) < 30:
        out["Anomaly"] = 0
        return out

    features = ["Oil_Rate", "Water_Cut", "BHP", "THP"]
    data = out[features].replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.median())

    model = IsolationForest(contamination=contamination, random_state=42)
    pred = model.fit_predict(data)
    out["Anomaly"] = np.where(pred == -1, 1, 0)
    return out


# ============================================================================
# 4) Forecasting models
# ============================================================================
def _rmse(y_true, y_pred) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _prepare_ml_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["Oil_Rate"])
    if len(data) < 40:
        raise ValueError("Insufficient valid data for training. Need at least 40 rows.")
    return data[FEATURE_COLUMNS], data["Oil_Rate"]


@cache_resource
def train_forecast_model_cached(data_signature: str, df: pd.DataFrame, model_type: str = "auto"):
    return train_forecast_model(df, model_type)


def train_forecast_model(df: pd.DataFrame, model_type: str = "auto"):
    if not SKLEARN:
        raise RuntimeError("scikit-learn is required. Install: pip install scikit-learn")

    X, y = _prepare_ml_data(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Auto priority: XGBoost if installed, otherwise RandomForest. LSTM only when explicitly selected.
    chosen = model_type.lower().strip()
    if chosen == "auto":
        chosen = "xgb" if XGB else "rf"

    if chosen == "lstm":
        if not TF:
            raise RuntimeError("TensorFlow is not installed. Install it or choose Random Forest/XGBoost.")
        return train_lstm_forecast(X_scaled, y)

    split_idx = int(len(X_scaled) * 0.80)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if chosen == "xgb" and XGB:
        model = xgb.XGBRegressor(
            n_estimators=250,
            learning_rate=0.045,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
        model_label = "XGBoost"
    else:
        model = RandomForestRegressor(
            n_estimators=220,
            max_depth=14,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        model_label = "Random Forest"

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "Model": model_label,
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": _rmse(y_test, pred),
        "R2": float(r2_score(y_test, pred)) if len(y_test) > 1 else 0.0,
    }

    result = pd.DataFrame({"Actual": y_test.values, "Predicted": pred})

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
    else:
        importance = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": np.zeros(len(FEATURE_COLUMNS))})

    return model, scaler, metrics, result, importance


def train_lstm_forecast(X_scaled: np.ndarray, y: pd.Series, seq_length: int = 14):
    if len(X_scaled) < seq_length + 50:
        raise ValueError("Not enough rows for LSTM. Need more historical data.")

    X_seq, y_seq = [], []
    y_values = y.values.astype(float)
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:i + seq_length])
        y_seq.append(y_values[i + seq_length])

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)

    split_idx = int(len(X_seq) * 0.80)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, X_seq.shape[2])),
        Dropout(0.20),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.20,
        callbacks=[early_stop],
        verbose=0,
    )

    pred = model.predict(X_test, verbose=0).flatten()
    metrics = {
        "Model": "LSTM",
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": _rmse(y_test, pred),
        "R2": float(r2_score(y_test, pred)) if len(y_test) > 1 else 0.0,
    }
    result = pd.DataFrame({"Actual": y_test, "Predicted": pred})
    importance = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": np.nan})
    return model, None, metrics, result, importance


# ============================================================================
# 5) Optimization
# ============================================================================
def optimize_rates_rule_based(latest: pd.DataFrame, max_increase_pct: float = 0.15) -> pd.DataFrame:
    out = latest.copy()
    proposed = []

    for _, row in out.iterrows():
        rate = float(row["Oil_Rate"])
        wc = float(row["Water_Cut"])
        bhp = float(row["BHP"])
        wii = float(row.get("WII", 50))

        if wc >= 85 or bhp < 1600 or wii < 40:
            new_rate = rate * 0.35
        elif wii >= 80 and bhp > 2500 and wc < 55:
            new_rate = rate * (1 + max_increase_pct)
        elif wii >= 60 and bhp > 2200 and wc < 70:
            new_rate = rate * (1 + max_increase_pct * 0.60)
        else:
            new_rate = rate * 0.95

        proposed.append(max(0.0, round(new_rate, 2)))

    out["Proposed_Rate"] = proposed
    out["Delta"] = (out["Proposed_Rate"] - out["Oil_Rate"]).round(2)
    out["Optimization_Action"] = np.where(
        out["Delta"] > 50,
        "Increase",
        np.where(out["Delta"] < -50, "Reduce", "Maintain"),
    )
    return out


def optimize_rates_ga(latest: pd.DataFrame, max_increase_pct: float = 0.15,
                      population_size: int = 60, generations: int = 35) -> pd.DataFrame:
    try:
        from deap import base, creator, tools, algorithms
    except Exception:
        return optimize_rates_rule_based(latest, max_increase_pct)

    rates = latest["Oil_Rate"].to_numpy(dtype=float)
    bhp = latest["BHP"].to_numpy(dtype=float)
    wc = latest["Water_Cut"].to_numpy(dtype=float)
    wii = latest["WII"].to_numpy(dtype=float)
    n = len(rates)

    if n == 0:
        return latest.copy()

    # Avoid DEAP repeated creator crash in Streamlit reruns
    if not hasattr(creator, "FitnessMaxISRM"):
        creator.create("FitnessMaxISRM", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualISRM"):
        creator.create("IndividualISRM", list, fitness=creator.FitnessMaxISRM)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0.60, 1.0 + max_increase_pct)
    toolbox.register("individual", tools.initRepeat, creator.IndividualISRM, toolbox.attr_float, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    baseline_total = max(float(np.sum(rates)), 1.0)

    def evaluate(individual):
        mult = np.asarray(individual, dtype=float)
        proposed = np.clip(mult * rates, 0, rates * (1.0 + max_increase_pct))

        oil_gain_norm = (np.sum(proposed) - np.sum(rates)) / baseline_total
        water_penalty = np.sum((proposed / baseline_total) * (wc / 100.0))
        pressure_penalty = np.sum(np.maximum(0, 1500 - bhp) / 1500.0)
        poor_well_penalty = np.sum((wii < 40) * (proposed / baseline_total))

        # Balanced normalized objective
        fitness_value = (
            1.00 * oil_gain_norm
            - 0.45 * water_penalty
            - 0.70 * pressure_penalty
            - 0.60 * poor_well_penalty
        )
        return (float(fitness_value),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.40)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.08, indpb=0.20)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.70,
        mutpb=0.25,
        ngen=generations,
        halloffame=hof,
        verbose=False,
    )

    best = np.asarray(hof[0], dtype=float)
    proposed = np.clip(best * rates, 0, rates * (1.0 + max_increase_pct))

    out = latest.copy()
    out["Proposed_Rate"] = np.round(proposed, 2)
    out["Delta"] = (out["Proposed_Rate"] - out["Oil_Rate"]).round(2)
    out["Optimization_Action"] = np.where(
        out["Delta"] > 50,
        "Increase",
        np.where(out["Delta"] < -50, "Reduce", "Maintain"),
    )
    return out


# ============================================================================
# 6) Reporting and exports
# ============================================================================
def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    engine = "xlsxwriter"
    try:
        with pd.ExcelWriter(buf, engine=engine) as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
    except Exception:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def generate_pdf_report(summary: Dict[str, float], latest: pd.DataFrame, optimization: pd.DataFrame) -> bytes:
    if not REPORTLAB:
        return b""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="ISRM Reservoir Report")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ISRMTitle", parent=styles["Heading1"], fontSize=17, alignment=1)
    story = []

    story.append(Paragraph("Iraqi Smart Reservoir Manager - Advanced Operational Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    summary_data = [
        ["Metric", "Value"],
        ["Total Wells", str(summary["wells"])],
        ["Total Oil Rate (bbl/d)", f"{summary['total_oil']:,.0f}"],
        ["Average BHP (psi)", f"{summary['avg_bhp']:.0f}"],
        ["Average Water Cut (%)", f"{summary['avg_wc']:.1f}"],
        ["Critical Wells", str(summary["critical"])],
    ]
    table = Table(summary_data, colWidths=[180, 180])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6B4E16")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("Lowest 10 Wells by WII", styles["Heading2"]))
    low = latest.nsmallest(10, "WII")[["Well_Name", "Oil_Rate", "Water_Cut", "BHP", "WII", "AI_Status"]]
    table2 = Table([low.columns.tolist()] + low.astype(str).values.tolist(), repeatRows=1)
    table2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table2)
    story.append(Spacer(1, 18))

    story.append(Paragraph("Optimization Summary", styles["Heading2"]))
    opt = optimization[["Well_Name", "Oil_Rate", "Proposed_Rate", "Delta", "Optimization_Action"]].copy()
    opt = opt.sort_values("Delta", ascending=False).head(12)
    table3 = Table([opt.columns.tolist()] + opt.astype(str).values.tolist(), repeatRows=1)
    table3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table3)

    doc.build(story)
    return buf.getvalue()


# ============================================================================
# 7) Pipeline
# ============================================================================
def process_pipeline(raw: pd.DataFrame, use_ga: bool = False, max_increase: float = 0.15):
    cleaned = clean_data(raw)
    featured = add_engineered_features(cleaned)
    featured = detect_anomalies(featured)
    latest = latest_snapshot(featured)

    if use_ga:
        optimized = optimize_rates_ga(latest, max_increase_pct=max_increase)
    else:
        optimized = optimize_rates_rule_based(latest, max_increase_pct=max_increase)

    summary = {
        "wells": int(latest["Well_Name"].nunique()),
        "total_oil": float(latest["Oil_Rate"].sum()),
        "avg_bhp": float(latest["BHP"].mean()),
        "avg_wc": float(latest["Water_Cut"].mean()),
        "critical": int((latest["AI_Status"] == "Critical").sum()),
    }
    return cleaned, featured, latest, optimized, summary


# ============================================================================
# 8) Tests
# ============================================================================
def run_tests() -> None:
    print("Running ISRM Advanced Stable self-tests...")
    raw = generate_sample_data(days=120, wells=4)
    cleaned, featured, latest, optimized, summary = process_pipeline(raw, use_ga=False)

    assert not raw.empty
    assert not cleaned.empty
    assert "WII" in featured.columns
    assert featured["WII"].between(0, 100).all()
    assert "Anomaly" in featured.columns
    assert len(latest) == 4
    assert "Proposed_Rate" in optimized.columns
    assert (optimized["Proposed_Rate"] >= 0).all()
    assert summary["wells"] == 4
    assert summary["total_oil"] > 0

    if SKLEARN:
        model, scaler, metrics, result, importance = train_forecast_model(featured, "rf")
        assert metrics["RMSE"] >= 0
        assert not result.empty
        assert not importance.empty

    excel = to_excel_bytes({"Latest": latest, "Optimization": optimized})
    assert len(excel) > 1000

    if REPORTLAB:
        pdf = generate_pdf_report(summary, latest, optimized)
        assert len(pdf) > 1000

    print("All tests passed successfully.")


# ============================================================================
# 9) Streamlit UI
# ============================================================================
def render_app() -> None:
    if not STREAMLIT:
        print("Streamlit is not installed. Install: pip install streamlit")
        return

    st.set_page_config(page_title="ISRM Advanced Stable", layout="wide", page_icon="🛢️")

    st.markdown(
        """
        <style>
        .title {text-align:center; font-size:34px; font-weight:800; color:#7A4E00;}
        .subtitle {text-align:center; font-size:18px; color:#555; margin-bottom:25px;}
        .note {background:#fff8e1; border-left:6px solid #b8860b; padding:12px; border-radius:8px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="title">🛢️ ISRM Advanced Stable</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Iraqi Smart Reservoir Manager – Practical AI Reservoir Decision Support System</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Configuration")
        uploaded = st.file_uploader("Upload well data Excel/CSV", type=["xlsx", "xls", "csv"])
        use_sample = st.checkbox("Use synthetic Nasiriyah field data", value=uploaded is None)
        model_choice = st.selectbox("Forecast model", ["Auto", "Random Forest", "XGBoost", "LSTM"])
        max_increase = st.slider("Max allowed rate increase (%)", 5, 30, 15) / 100.0
        use_ga = st.checkbox("Use Genetic Algorithm optimization", value=False)
        st.caption("GA requires deap. XGBoost/TensorFlow/SHAP are optional.")

    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)
    elif use_sample:
        raw = generate_sample_data()
    else:
        st.info("Please upload a file or use sample data.")
        return

    try:
        cleaned, featured, latest, optimized, summary = process_pipeline(raw, use_ga=use_ga, max_increase=max_increase)
    except Exception as exc:
        st.error(f"Processing error: {exc}")
        return

    model_map = {"Auto": "auto", "Random Forest": "rf", "XGBoost": "xgb", "LSTM": "lstm"}
    chosen_model = model_map[model_choice]

    forecast_metrics = {}
    forecast_result = pd.DataFrame()
    importance = pd.DataFrame()
    trained_model = None
    scaler = None

    try:
        trained_model, scaler, forecast_metrics, forecast_result, importance = train_forecast_model(
            featured, chosen_model
        )
    except Exception as exc:
        st.warning(f"Forecast model not available: {exc}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", "🔍 Well Analysis", "🤖 Forecast", "⚙️ Optimization", "🧠 Explainability", "📄 Export"
    ])

    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Wells", summary["wells"])
        c2.metric("Total Oil", f"{summary['total_oil']:,.0f} bbl/d")
        c3.metric("Avg BHP", f"{summary['avg_bhp']:,.0f} psi")
        c4.metric("Avg WC", f"{summary['avg_wc']:.1f}%")
        c5.metric("Critical Wells", summary["critical"])

        if PLOTLY:
            prod = featured.groupby("Date", as_index=False)["Oil_Rate"].sum()
            st.plotly_chart(px.line(prod, x="Date", y="Oil_Rate", title="Total Field Production"), use_container_width=True)

            status_count = latest["AI_Status"].value_counts().reset_index()
            status_count.columns = ["AI_Status", "Count"]
            st.plotly_chart(px.pie(status_count, names="AI_Status", values="Count", title="Well Status Distribution"), use_container_width=True)
        else:
            st.line_chart(featured.groupby("Date")["Oil_Rate"].sum())

    with tab2:
        st.subheader("Well Integrity Index and Operational Recommendations")
        st.dataframe(latest[DISPLAY_COLUMNS].sort_values("WII"), use_container_width=True)

        anomalies = featured[featured["Anomaly"] == 1]
        st.subheader("Detected Anomalies")
        if len(anomalies) > 0:
            st.warning(f"Detected {len(anomalies)} anomalous well-day records.")
            st.dataframe(anomalies[["Date", "Well_Name", "Oil_Rate", "Water_Cut", "BHP", "THP"]].head(500), use_container_width=True)
        else:
            st.success("No anomalies detected.")

    with tab3:
        st.subheader("Production Forecasting")
        if forecast_metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", forecast_metrics.get("Model", "N/A"))
            c2.metric("MAE", f"{forecast_metrics.get('MAE', 0):.2f}")
            c3.metric("RMSE", f"{forecast_metrics.get('RMSE', 0):.2f}")
            c4.metric("R²", f"{forecast_metrics.get('R2', 0):.3f}")

            if PLOTLY and not forecast_result.empty:
                temp = forecast_result.tail(250).reset_index(drop=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=temp["Actual"], mode="lines", name="Actual"))
                fig.add_trace(go.Scatter(y=temp["Predicted"], mode="lines", name="Predicted"))
                fig.update_layout(title="Actual vs Predicted Production", xaxis_title="Test Sample", yaxis_title="Oil Rate")
                st.plotly_chart(fig, use_container_width=True)
            elif not forecast_result.empty:
                st.line_chart(forecast_result.tail(250))

            if not importance.empty:
                st.subheader("Feature Importance")
                st.dataframe(importance, use_container_width=True)
        else:
            st.info("No forecast result available.")

    with tab4:
        st.subheader("Rate Optimization")
        st.dataframe(optimized[OPT_COLUMNS].sort_values("Delta", ascending=False), use_container_width=True)

        old_total = float(optimized["Oil_Rate"].sum())
        new_total = float(optimized["Proposed_Rate"].sum())
        delta = new_total - old_total
        pct = (delta / old_total * 100.0) if old_total else 0
        st.metric("Total Field Production Change", f"{new_total:,.0f} bbl/d", f"{delta:+,.0f} bbl/d ({pct:+.2f}%)")

        if PLOTLY:
            plot_df = optimized[["Well_Name", "Oil_Rate", "Proposed_Rate"]].melt(
                id_vars="Well_Name", var_name="Scenario", value_name="Rate"
            )
            st.plotly_chart(px.bar(plot_df, x="Well_Name", y="Rate", color="Scenario", barmode="group", title="Current vs Proposed Rates"), use_container_width=True)
        else:
            st.bar_chart(optimized.set_index("Well_Name")[["Oil_Rate", "Proposed_Rate"]])

    with tab5:
        st.subheader("Explainability")
        if not importance.empty:
            st.write("The current practical explanation is based on model feature importance.")
            st.dataframe(importance, use_container_width=True)
        else:
            st.info("Feature importance is not available for this model.")

        st.markdown(
            """
            **Interpretation guide:**
            - High Water_Cut importance means water production strongly affects forecast and well status.
            - High BHP importance means pressure management is critical.
            - High moving-average importance means recent production trend dominates short-term forecasting.
            """
        )

    with tab6:
        st.subheader("Export Reports")
        excel_bytes = to_excel_bytes({
            "Latest_Well_Analysis": latest,
            "Optimization": optimized,
            "Forecast": forecast_result,
            "Feature_Importance": importance,
            "Full_Data": featured,
        })
        st.download_button(
            "Download Excel Report",
            data=excel_bytes,
            file_name="ISRM_Advanced_Stable_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if REPORTLAB:
            pdf_bytes = generate_pdf_report(summary, latest, optimized)
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="ISRM_Advanced_Stable_Report.pdf",
                mime="application/pdf",
            )
        else:
            st.warning("ReportLab not installed. Install: pip install reportlab")

        st.markdown(
            """
            <div class="note">
            Practical note: use the synthetic dataset only for demonstration. For official technical conclusions, upload real well data from the selected Iraqi field.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# Entry point
# ============================================================================
def main() -> None:
    if "--test" in sys.argv:
        run_tests()
        return

    if not STREAMLIT:
        print("Streamlit is not installed. Install with: pip install streamlit")
        print("Then run: streamlit run app.py")
        print("For tests only: python app.py --test")
        return

    render_app()


if __name__ == "__main__":
    main()
