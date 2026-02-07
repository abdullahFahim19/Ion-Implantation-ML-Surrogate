# ===================== CHAPTER 8.4 — Reverse Model Performance (ALL FIGURES + TABLES) =====================
# Goal (8.4):
#   Evaluate inverse (reverse) model that predicts implant ENERGY from a target depth (Rp*)
#   under material + species + angle conditions, using SRIM-derived dataset as ground truth.
#
# Reverse mapping evaluated (typical in your work):
#   Inputs  : (substrate, ion, Rp_A, angle_deg)   [and optionally thickness constraint used at query time]
#   Output  : energy_keV
#
# This script:
#   1) Builds a "physics-clean" reverse dataset: stopped-only + low-transmission filter
#   2) Runs 10-fold CV for:
#        - Random Forest Regressor (primary)
#        - KNN Regressor with K in {1,3,5,10,20} (comparison)
#   3) Exports:
#        - CSV tables: global + by-substrate + by-ion + model comparison
#        - 300 dpi TIFF figures: predicted vs true energy scatter (global + per substrate),
#          residual histogram, and error vs Rp plot
#
# Dataset needed: full_database.csv
# Required columns: substrate, ion, Rp_A, angle_deg, energy_keV, is_profile, transmitted

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- CONFIG ----------------
DATA_FILE = "sample_database.csv"   # update path if needed

OUT_FIG_DIR = "./Chapter8_Assets/"   # Created a specific folder for organization
OUT_TAB_DIR ="./Chapter8_Assets/"   # Created a specific folder for organization
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_TAB_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 10

# Reverse model input/output (energy prediction)
REV_INPUTS = ["substrate", "ion", "Rp_A", "angle_deg"]
REV_TARGET = "energy_keV"

# Physics-clean filter for inverse:
#  - Use only stopped cases (is_profile==1)
#  - Remove heavy transmission contamination (e.g., transmitted < 50)
# Adjust threshold if you used a different one.
TRANSMIT_THRESHOLD = 50

# KNN variants
K_LIST = [1, 3, 5, 10, 20]

# ---------------- LOAD ----------------
df = pd.read_csv(DATA_FILE)
df["substrate"] = df["substrate"].replace({"Ga50As50": "GaAs", "Si50C50": "SiC"})

need = set(REV_INPUTS + [REV_TARGET, "is_profile", "transmitted"])
missing = sorted(list(need - set(df.columns)))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# numeric coercion
for c in ["Rp_A", "angle_deg", "energy_keV", "transmitted"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["is_profile"] = pd.to_numeric(df["is_profile"], errors="coerce").astype("Int64")

# ---------------- BUILD REVERSE DATASET ----------------
df_rev = df.dropna(subset=REV_INPUTS + [REV_TARGET, "is_profile", "transmitted"]).copy()
df_rev = df_rev[(df_rev["is_profile"] == 1) & (df_rev["transmitted"] < TRANSMIT_THRESHOLD)].copy()

# Sort / unique lists
SUBSTRATES = sorted(df_rev["substrate"].unique().tolist())
IONS = sorted(df_rev["ion"].unique().tolist())

print("Reverse dataset rows:", len(df_rev))
print("Substrates:", SUBSTRATES)
print("Ions:", IONS)

X = df_rev[REV_INPUTS].copy()
y = df_rev[REV_TARGET].astype(float).values

# ---------------- PREPROCESSOR ----------------
cat_features = ["substrate", "ion"]
num_features = ["Rp_A", "angle_deg"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features),
    ]
)

# ---------------- MODELS ----------------
rf_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("reg", RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ))
])

def make_knn(k):
    return Pipeline(steps=[
        ("prep", preprocessor),
        ("reg", KNeighborsRegressor(n_neighbors=k, weights="distance"))
    ])

# ---------------- CV SETUP ----------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ---------------- HELPERS ----------------
def metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2

def save_scatter(y_true, y_pred, title, fig_name_base):
    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    ax.scatter(y_true, y_pred, alpha=0.6)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)
    ax.set_xlabel("SRIM Ground Truth Energy (keV)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted Energy (keV)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, fig_name_base + ".tif"), dpi=300, format="tiff")
    fig.savefig(os.path.join(OUT_FIG_DIR, fig_name_base + ".png"), dpi=300)
    plt.close(fig)

def save_hist(residuals, title, fig_name_base):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.hist(residuals, bins=40)
    ax.set_xlabel("Residual (Predicted − True) Energy (keV)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, fig_name_base + ".tif"), dpi=300, format="tiff")
    fig.savefig(os.path.join(OUT_FIG_DIR, fig_name_base + ".png"), dpi=300)
    plt.close(fig)

def save_error_vs_rp(rp, abs_err, title, fig_name_base):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.scatter(rp, abs_err, alpha=0.6)
    ax.set_xlabel("Target Projected Range $R_p$ (Å)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Absolute Error |ΔE| (keV)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG_DIR, fig_name_base + ".tif"), dpi=300, format="tiff")
    fig.savefig(os.path.join(OUT_FIG_DIR, fig_name_base + ".png"), dpi=300)
    plt.close(fig)

# ===================== (1) RF: OUT-OF-FOLD PREDICTIONS =====================
oof_pred_rf = np.zeros(len(df_rev), dtype=float)

for fold, (tr, te) in enumerate(kf.split(X), start=1):
    rf_model.fit(X.iloc[tr], y[tr])
    oof_pred_rf[te] = rf_model.predict(X.iloc[te])
    print(f"[RF] Fold {fold}/{N_SPLITS} done")

mae_rf, rmse_rf, r2_rf = metrics(y, oof_pred_rf)

# Save global metrics table
pd.DataFrame([{
    "Model": "RandomForest",
    "MAE_keV": mae_rf,
    "RMSE_keV": rmse_rf,
    "R2": r2_rf,
    "N_samples": len(df_rev),
    "Filter": f"is_profile==1 and transmitted<{TRANSMIT_THRESHOLD}"
}]).to_csv(os.path.join(OUT_TAB_DIR, "Table_8_4A_Reverse_Global_RF.csv"), index=False)

# Global scatter + residual plots
save_scatter(
    y, oof_pred_rf,
    title=f"Fig. 8.4A Reverse Model: Predicted vs SRIM Energy (RF, {N_SPLITS}-Fold CV)",
    fig_name_base="Fig_8_4A_Energy_Scatter_Global_RF_300dpi"
)

residuals = oof_pred_rf - y
save_hist(
    residuals,
    title=f"Fig. 8.4B Reverse Model Residual Distribution (RF, {N_SPLITS}-Fold CV)",
    fig_name_base="Fig_8_4B_Energy_Residual_Hist_RF_300dpi"
)

save_error_vs_rp(
    df_rev["Rp_A"].values,
    np.abs(residuals),
    title=f"Fig. 8.4C Absolute Energy Error vs Target $R_p$ (RF, {N_SPLITS}-Fold CV)",
    fig_name_base="Fig_8_4C_AbsError_vs_Rp_RF_300dpi"
)

# ===================== (2) BY-SUBSTRATE METRICS + PER-SUBSTRATE SCATTER =====================
rows_sub = []
for sub in SUBSTRATES:
    idx = (df_rev["substrate"] == sub).values
    yt = y[idx]
    yp = oof_pred_rf[idx]
    if len(yt) < 5:
        continue
    mae, rmse, r2 = metrics(yt, yp)
    rows_sub.append({
        "substrate": sub,
        "N": int(idx.sum()),
        "MAE_keV": mae,
        "RMSE_keV": rmse,
        "R2": r2
    })
    save_scatter(
        yt, yp,
        title=f"Fig. 8.4D Reverse Energy Prediction — {sub} (RF, {N_SPLITS}-Fold CV)",
        fig_name_base=f"Fig_8_4D_Energy_Scatter_{sub}_RF_300dpi"
    )

pd.DataFrame(rows_sub).to_csv(os.path.join(OUT_TAB_DIR, "Table_8_4B_Reverse_By_Substrate_RF.csv"), index=False)

# ===================== (3) BY-ION METRICS =====================
rows_ion = []
for ion in IONS:
    idx = (df_rev["ion"] == ion).values
    yt = y[idx]
    yp = oof_pred_rf[idx]
    if len(yt) < 5:
        continue
    mae, rmse, r2 = metrics(yt, yp)
    rows_ion.append({
        "ion": ion,
        "N": int(idx.sum()),
        "MAE_keV": mae,
        "RMSE_keV": rmse,
        "R2": r2
    })

pd.DataFrame(rows_ion).to_csv(os.path.join(OUT_TAB_DIR, "Table_8_4C_Reverse_By_Ion_RF.csv"), index=False)

# ===================== (4) MODEL COMPARISON: RF vs KNN(K) =====================
# We compute global CV metrics for each model (same folds) and export a comparison table.
comp_rows = []

# RF already computed
comp_rows.append({
    "Model": "RandomForest",
    "MAE_keV": mae_rf,
    "RMSE_keV": rmse_rf,
    "R2": r2_rf
})

# KNN variants
for k in K_LIST:
    model = make_knn(k)
    oof_pred = np.zeros(len(df_rev), dtype=float)
    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        model.fit(X.iloc[tr], y[tr])
        oof_pred[te] = model.predict(X.iloc[te])
    mae, rmse, r2 = metrics(y, oof_pred)
    comp_rows.append({
        "Model": f"KNN(k={k}, weights=distance)",
        "MAE_keV": mae,
        "RMSE_keV": rmse,
        "R2": r2
    })

comp_df = pd.DataFrame(comp_rows).sort_values("R2", ascending=False)
comp_df.to_csv(os.path.join(OUT_TAB_DIR, "Table_8_4D_Reverse_Model_Comparison_RF_vs_KNN.csv"), index=False)

# ---------------- DONE ----------------
print("\n✅ DONE (8.4 Reverse Performance)")
print("Figures saved in:", OUT_FIG_DIR)
print("Tables  saved in:", OUT_TAB_DIR)

print("\nSaved tables:")
for f in sorted(os.listdir(OUT_TAB_DIR)):
    print(" -", f)

print("\nSaved figures (sample):")
for f in sorted(os.listdir(OUT_FIG_DIR))[:12]:
    print(" -", f)
print(" ...")
