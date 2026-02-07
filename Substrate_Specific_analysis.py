# ===================== 8.2 Substrate-Specific Accuracy (ALL FIGURES + TABLES) =====================
# Runs in Google Colab / Jupyter
# Outputs:
#  - CSV tables (classification + regression) saved to ./outputs_tables/
#  - 300 dpi TIFF figures saved to ./outputs_figs/
#
# Assumptions:
#  - You have full_database.csv in the current working directory (or set DATA_PATH).
#  - Column names match your dataset: substrate, ion, energy_keV, angle_deg, thickness_A,
#    is_profile (0/1), and regression targets:
#    Rp_A, dRp_A, lateral_range_A, lateral_straggle_A, radial_range_A, radial_straggle_A,
#    vacancies_per_ion, backscattered, transmitted

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# -------------------- CONFIG --------------------
DATA_PATH = "sample_database.csv"
OUT_FIG_DIR = "./Chapter8_Assets/"   # Created a specific folder for organization "
OUT_TAB_DIR = " ./Chapter8_Assets/"   # Created a specific folder for organization "
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_TAB_DIR, exist_ok=True)

# CV
CLS_FOLDS = 10          # gatekeeper (as you used 10-fold)
REG_FOLDS = 10          # stopped-only regression
RANDOM_STATE = 42

# Inputs
INPUT_FEATURES = ["substrate", "ion", "energy_keV", "angle_deg", "thickness_A"]
CATEGORICAL = ["substrate", "ion"]
NUMERICAL   = ["energy_keV", "angle_deg", "thickness_A"]

# Targets (9 outputs)
TARGETS = [
    "Rp_A", "dRp_A",
    "lateral_range_A", "lateral_straggle_A",
    "radial_range_A", "radial_straggle_A",
    "vacancies_per_ion", "backscattered", "transmitted"
]

# -------------------- LOAD + CLEAN --------------------
df = pd.read_csv(DATA_PATH)

# Standardize substrate labels (as per your code)
df["substrate"] = df["substrate"].replace({"Ga50As50": "GaAs", "Si50C50": "SiC"})

# Basic sanity: keep only rows with required columns
needed = set(INPUT_FEATURES + ["is_profile"] + TARGETS)
missing = sorted(list(needed - set(df.columns)))
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Ensure numeric targets
for c in ["energy_keV", "angle_deg", "thickness_A"] + TARGETS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with NaNs in critical inputs or label
df = df.dropna(subset=INPUT_FEATURES + ["is_profile"]).copy()
df["is_profile"] = df["is_profile"].astype(int)

# Substrate list
SUBSTRATES = sorted(df["substrate"].unique().tolist())

print("Loaded rows:", len(df))
print("Substrates:", SUBSTRATES)

# -------------------- PREPROCESSOR --------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERICAL),
    ]
)

# -------------------- MODELS --------------------
# Gatekeeper: Gradient Boosting (as per your Fig 8.1)
gatekeeper = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
])

# Regressor: Random Forest (stopped only)
regressor = Pipeline(steps=[
    ("prep", preprocessor),
    ("reg", MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    ))
])

# ===================== 8.2(A) CLASSIFICATION BY SUBSTRATE =====================
X = df[INPUT_FEATURES]
y = df["is_profile"].values  # 0 transmission, 1 stopped

skf = StratifiedKFold(n_splits=CLS_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Store out-of-fold predictions
y_pred = np.zeros_like(y)
y_proba = np.zeros(len(y), dtype=float)

for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
    gatekeeper.fit(X.iloc[tr], y[tr])
    y_pred[te] = gatekeeper.predict(X.iloc[te])
    # proba for class 1 if available
    if hasattr(gatekeeper.named_steps["clf"], "predict_proba"):
        y_proba[te] = gatekeeper.predict_proba(X.iloc[te])[:, 1]
    print(f"[CLS] Fold {fold}/{CLS_FOLDS} done")

# Global metrics
cls_global = {
    "Accuracy": accuracy_score(y, y_pred),
    "Precision(Stopped=1)": precision_score(y, y_pred, pos_label=1, zero_division=0),
    "Recall(Stopped=1)": recall_score(y, y_pred, pos_label=1, zero_division=0),
    "F1(Stopped=1)": f1_score(y, y_pred, pos_label=1, zero_division=0),
}
pd.DataFrame([cls_global]).to_csv(os.path.join(OUT_TAB_DIR, "Table_8_2A_Gatekeeper_Global.csv"), index=False)

# Substrate-wise metrics + confusion matrices
rows = []
for sub in SUBSTRATES:
    idx = (df["substrate"] == sub).values
    yt, yp = y[idx], y_pred[idx]

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Row-normalized recall per class (like your plot)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    rows.append({
        "substrate": sub,
        "n_samples": int(idx.sum()),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Transmission Recall (TN/(TN+FP))": (tn / (tn + fp)) if (tn + fp) else np.nan,
        "Stopped Recall (TP/(TP+FN))": (tp / (tp + fn)) if (tp + fn) else np.nan,
        "Accuracy": accuracy_score(yt, yp),
        "Precision(Stopped=1)": precision_score(yt, yp, pos_label=1, zero_division=0),
        "Recall(Stopped=1)": recall_score(yt, yp, pos_label=1, zero_division=0),
        "F1(Stopped=1)": f1_score(yt, yp, pos_label=1, zero_division=0),
    })

    # ---- FIG: Confusion matrix counts + % (normalized by true class) ----
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    im = ax.imshow(cm_norm, vmin=0, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Transmission (0)", "Stopped (1)"])
    ax.set_yticklabels(["Transmission (0)", "Stopped (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Fig. 8.2A Gatekeeper Confusion Matrix — {sub}\n(Gradient Boosting, {CLS_FOLDS}-Fold CV)")

    # annotate with counts and %
    for i in range(2):
        for j in range(2):
            pct = 100.0 * cm_norm[i, j] if not np.isnan(cm_norm[i, j]) else 0.0
            ax.text(j, i, f"{cm[i,j]}\n({pct:.1f}%)", ha="center", va="center", fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Recall (Normalized by True Class)")

    fig.tight_layout()
    base = f"Fig_8_2A_CM_{sub}_300dpi"
    fig.savefig(os.path.join(OUT_FIG_DIR, base + ".tif"), dpi=300, format="tiff")
    fig.savefig(os.path.join(OUT_FIG_DIR, base + ".png"), dpi=300)
    plt.close(fig)

pd.DataFrame(rows).to_csv(os.path.join(OUT_TAB_DIR, "Table_8_2B_Gatekeeper_By_Substrate.csv"), index=False)

# ===================== 8.2(B) REGRESSION BY SUBSTRATE (STOPPED ONLY) =====================
df_stop = df[df["is_profile"] == 1].dropna(subset=TARGETS).copy()
Xr = df_stop[INPUT_FEATURES]
Yr = df_stop[TARGETS].values

kf = KFold(n_splits=REG_FOLDS, shuffle=True, random_state=RANDOM_STATE)

Y_pred = np.zeros_like(Yr, dtype=float)

for fold, (tr, te) in enumerate(kf.split(Xr), start=1):
    regressor.fit(Xr.iloc[tr], Yr[tr])
    Y_pred[te] = regressor.predict(Xr.iloc[te])
    print(f"[REG] Fold {fold}/{REG_FOLDS} done")

# Helper metrics
def rmse(y_true, y_hat):
    return float(np.sqrt(mean_squared_error(y_true, y_hat)))

# ---- Table: overall (all substrates) per-target metrics ----
rows_targets = []
for k, name in enumerate(TARGETS):
    yt = Yr[:, k]
    yp = Y_pred[:, k]
    rows_targets.append({
        "target": name,
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": rmse(yt, yp),
        "R2": float(r2_score(yt, yp))
    })
pd.DataFrame(rows_targets).to_csv(os.path.join(OUT_TAB_DIR, "Table_8_2C_Reg_PerTarget_Global.csv"), index=False)

# ---- Table: substrate-wise average metrics across targets + per-target by substrate ----
rows_sub_avg = []
rows_sub_target = []

for sub in SUBSTRATES:
    idx = (df_stop["substrate"] == sub).values
    if idx.sum() == 0:
        continue

    # per target
    r2s, maes, rmses = [], [], []
    for k, name in enumerate(TARGETS):
        yt = Yr[idx, k]
        yp = Y_pred[idx, k]
        m_mae = float(mean_absolute_error(yt, yp))
        m_rmse = rmse(yt, yp)
        m_r2 = float(r2_score(yt, yp))

        rows_sub_target.append({
            "substrate": sub,
            "target": name,
            "MAE": m_mae,
            "RMSE": m_rmse,
            "R2": m_r2
        })
        r2s.append(m_r2); maes.append(m_mae); rmses.append(m_rmse)

    rows_sub_avg.append({
        "substrate": sub,
        "n_stopped_samples": int(idx.sum()),
        "Avg_MAE": float(np.mean(maes)),
        "Avg_RMSE": float(np.mean(rmses)),
        "Avg_R2": float(np.mean(r2s))
    })

pd.DataFrame(rows_sub_avg).to_csv(os.path.join(OUT_TAB_DIR, "Table_8_2D_Reg_Substrate_Avg.csv"), index=False)
pd.DataFrame(rows_sub_target).to_csv(os.path.join(OUT_TAB_DIR, "Table_8_2E_Reg_Substrate_PerTarget.csv"), index=False)

# ===================== 8.2(C) FIGURES: Predicted vs SRIM PER SUBSTRATE =====================
# Two core plots per substrate: Rp_A and vacancies_per_ion (you can add others by extending this list)
PLOT_TARGETS = ["Rp_A", "vacancies_per_ion"]

for sub in SUBSTRATES:
    idx = (df_stop["substrate"] == sub).values
    if idx.sum() == 0:
        continue

    for tname in PLOT_TARGETS:
        k = TARGETS.index(tname)
        yt = Yr[idx, k]
        yp = Y_pred[idx, k]

        fig = plt.figure(figsize=(7, 7))
        ax = plt.gca()
        ax.scatter(yt, yp, alpha=0.6)

        # y=x guideline (no explicit color set)
        mn = float(np.nanmin([yt.min(), yp.min()]))
        mx = float(np.nanmax([yt.max(), yp.max()]))
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)

        ax.set_xlabel("SRIM Ground Truth")
        ax.set_ylabel("Model Prediction")
        ax.set_title(f"Fig. 8.2F Predicted vs SRIM — {tname} ({sub})\n(Random Forest, stopped-only, {REG_FOLDS}-Fold CV)")
        ax.grid(True)

        fig.tight_layout()
        base = f"Fig_8_2F_{tname}_{sub}_300dpi"
        fig.savefig(os.path.join(OUT_FIG_DIR, base + ".tif"), dpi=300, format="tiff")
        fig.savefig(os.path.join(OUT_FIG_DIR, base + ".png"), dpi=300)
        plt.close(fig)

# ===================== 8.2(D) FIGURE: Bar chart of Avg R2 by substrate =====================
sub_avg = pd.DataFrame(rows_sub_avg).sort_values("substrate")
if len(sub_avg) > 0:
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.bar(sub_avg["substrate"], sub_avg["Avg_R2"])
    ax.set_xlabel("Substrate")
    ax.set_ylabel("Average $R^2$ (across 9 targets)")
    ax.set_title(f"Fig. 8.2G Substrate-wise Average $R^2$\n(Random Forest, stopped-only, {REG_FOLDS}-Fold CV)")
    ax.grid(True, axis="y")
    fig.tight_layout()
    base = "Fig_8_2G_Substrate_AvgR2_300dpi"
    fig.savefig(os.path.join(OUT_FIG_DIR, base + ".tif"), dpi=300, format="tiff")
    fig.savefig(os.path.join(OUT_FIG_DIR, base + ".png"), dpi=300)
    plt.close(fig)

print("\nDONE ✅")
print(f"Figures saved in: {OUT_FIG_DIR}/")
print(f"Tables  saved in: {OUT_TAB_DIR}/")

print("\nGenerated Tables:")
for f in sorted(os.listdir(OUT_TAB_DIR)):
    print(" -", f)

print("\nGenerated Figures (sample list):")
for f in sorted(os.listdir(OUT_FIG_DIR))[:12]:
    print(" -", f)
print(" ...")
