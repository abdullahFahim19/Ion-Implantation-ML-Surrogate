# ===================== CHAPTER 8.3 — Ion-Specific Accuracy Analysis =====================
# Generates (1) per-ion gatekeeper confusion matrices + metrics tables
#          (2) per-ion regression scatter plots (Rp, Vacancies) + per-ion/per-target metrics tables
# Exports: 300-dpi TIFF figures + CSV tables
# Dataset: full_database.csv (must contain columns used below)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------- CONFIG ----------------------
DATA_FILE = "sample_database.csv"   # put in same folder or give full path
OUTDIR   = "/ OUT_DIR = "./Chapter8_Assets/"   # Created a specific folder for organization "
os.makedirs(OUTDIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 10
N_JOBS = -1

# Forward-model definition (must match your thesis I/O)
INPUT_FEATURES = ["substrate", "ion", "energy_keV", "angle_deg", "thickness_A"]
Y_CLASSIFIER   = "is_profile"  # 0=Transmission, 1=Stopped (your gatekeeper label)

# 9 core outputs you kept in thesis (no skewness/kurtosis)
TARGETS_9 = [
    "Rp_A", "dRp_A",
    "lateral_range_A", "lateral_straggle_A",
    "radial_range_A", "radial_straggle_A",
    "vacancies_per_ion",
    "backscattered", "transmitted"
]

# For plots in 8.3 (you can add more targets if you want)
PLOT_TARGETS = ["Rp_A", "vacancies_per_ion"]

# ---------------------- LOAD ----------------------
df = pd.read_csv(DATA_FILE)

# normalize substrate tokens if needed (as you did)
df["substrate"] = df["substrate"].replace({"Ga50As50": "GaAs", "Si50C50": "SiC"})

# basic sanity: keep rows with required columns
need_cols = set(INPUT_FEATURES + [Y_CLASSIFIER] + TARGETS_9)
missing = sorted(list(need_cols - set(df.columns)))
if missing:
    raise ValueError(f"Missing required columns in dataset: {missing}")

# X and y
X_all = df[INPUT_FEATURES].copy()
y_cls = df[Y_CLASSIFIER].astype(int).copy()

ions = sorted(df["ion"].unique())

# ---------------------- PREPROCESSORS ----------------------
cat_features = ["substrate", "ion"]
num_features = ["energy_keV", "angle_deg", "thickness_A"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features),
    ]
)

# ---------------------- MODELS ----------------------
# Gatekeeper (use RF here; if your thesis uses Gradient Boosting for gatekeeper,
# you can swap this estimator and re-run — the rest of the code stays same.)
gatekeeper = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=N_JOBS
    ))
])

# Forward regressor trained on STOPPED ONLY (your thesis logic)
regressor = Pipeline(steps=[
    ("prep", preprocessor),
    ("reg", MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    ))
])

# ---------------------- CROSS-VALIDATION PREDICTIONS ----------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# storage for out-of-fold gatekeeper preds
oof_pred_cls = np.zeros(len(df), dtype=int)

# storage for out-of-fold regression preds on STOPPED test rows only
# We'll fill NaN for rows not evaluated (e.g., transmission rows)
oof_pred_reg = np.full((len(df), len(TARGETS_9)), np.nan, dtype=float)

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_cls), start=1):
    X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
    y_tr, y_te = y_cls.iloc[tr_idx], y_cls.iloc[te_idx]

    # ---- train gatekeeper ----
    gatekeeper.fit(X_tr, y_tr)
    pred_te = gatekeeper.predict(X_te)
    oof_pred_cls[te_idx] = pred_te

    # ---- train regressor on STOPPED cases only (true stopped in train) ----
    tr_stopped_mask = (y_tr.values == 1)
    df_tr_stopped = df.iloc[tr_idx].loc[tr_stopped_mask]
    X_tr_stopped = df_tr_stopped[INPUT_FEATURES]
    y_tr_stopped = df_tr_stopped[TARGETS_9].astype(float)

    regressor.fit(X_tr_stopped, y_tr_stopped)

    # evaluate regression on STOPPED cases only (true stopped in test)
    te_stopped_mask = (y_te.values == 1)
    df_te_stopped = df.iloc[te_idx].loc[te_stopped_mask]
    if len(df_te_stopped) > 0:
        X_te_stopped = df_te_stopped[INPUT_FEATURES]
        pred_reg = regressor.predict(X_te_stopped)
        oof_pred_reg[df_te_stopped.index.values, :] = pred_reg

    print(f"[Fold {fold:02d}/{N_SPLITS}] done | test={len(te_idx)} | stopped_test={te_stopped_mask.sum()}")

# Add predictions to df for easy slicing
df_pred = df.copy()
df_pred["pred_is_profile"] = oof_pred_cls
for j, t in enumerate(TARGETS_9):
    df_pred[f"pred_{t}"] = oof_pred_reg[:, j]

# ---------------------- (A) ION-SPECIFIC GATEKEEPER RESULTS ----------------------
gatekeeper_rows = []
for ion in ions:
    sub = df_pred[df_pred["ion"] == ion]
    y_true = sub[Y_CLASSIFIER].astype(int).values
    y_hat  = sub["pred_is_profile"].astype(int).values

    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_hat)
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec  = recall_score(y_true, y_hat, zero_division=0)
    f1   = f1_score(y_true, y_hat, zero_division=0)

    gatekeeper_rows.append({
        "ion": ion,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "N_total": len(sub),
        "N_true_transmission": int((y_true==0).sum()),
        "N_true_stopped": int((y_true==1).sum())
    })

    # ---- Save confusion matrix figure (counts + row-normalized %) ----
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums!=0)

    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    im = ax.imshow(cm_norm)  # default colormap

    # annotate counts and %
    for r in range(2):
        for c in range(2):
            count = int(confusion_matrix(y_true, y_hat, labels=[0,1])[r, c])
            pct = cm_norm[r, c] * 100.0
            ax.text(c, r, f"{count}\n({pct:.1f}%)", ha="center", va="center", fontsize=16, fontweight="bold")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Transmission (0)", "Stopped (1)"], fontsize=14, fontweight="bold")
    ax.set_yticklabels(["Transmission (0)", "Stopped (1)"], fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=16, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=16, fontweight="bold")
    ax.set_title(f"Ion-Specific Gatekeeper Confusion Matrix — {ion}\n(10-Fold CV, Normalized by True Class)",
                 fontsize=18, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Recall (Row-normalized)", rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    figpath = os.path.join(OUTDIR, f"Fig_8_3_Gatekeeper_CM_{ion}_300dpi.tif")
    plt.savefig(figpath, dpi=300, format="tiff")
    plt.close(fig)

# save gatekeeper table
gatekeeper_df = pd.DataFrame(gatekeeper_rows).sort_values("ion")
gatekeeper_csv = os.path.join(OUTDIR, "Table_8_3_Gatekeeper_Metrics_By_Ion.csv")
gatekeeper_df.to_csv(gatekeeper_csv, index=False)

# ---------------------- (B) ION-SPECIFIC REGRESSION RESULTS (STOPPED ONLY) ----------------------
reg_rows = []
per_target_rows = []

for ion in ions:
    sub = df_pred[(df_pred["ion"] == ion) & (df_pred[Y_CLASSIFIER] == 1)].copy()  # true stopped only
    if len(sub) == 0:
        continue

    # only rows where reg predictions exist
    pred_ok = np.isfinite(sub[[f"pred_{t}" for t in TARGETS_9]].values).all(axis=1)
    sub = sub.loc[pred_ok].copy()
    if len(sub) == 0:
        continue

    # per-target metrics
    r2_list = []
    rmse_list = []
    mae_list = []

    for t in TARGETS_9:
        y_true = sub[t].astype(float).values
        y_hat  = sub[f"pred_{t}"].astype(float).values

        mae = mean_absolute_error(y_true, y_hat)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        r2 = r2_score(y_true, y_hat)

        per_target_rows.append({
            "ion": ion,
            "target": t,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "N": len(sub)
        })

        r2_list.append(r2)
        rmse_list.append(rmse)
        mae_list.append(mae)

    # averaged summary across 9 outputs
    reg_rows.append({
        "ion": ion,
        "N_stopped_evaluated": len(sub),
        "Avg_R2_all_9": float(np.mean(r2_list)),
        "Avg_RMSE_all_9": float(np.mean(rmse_list)),
        "Avg_MAE_all_9": float(np.mean(mae_list)),
    })

    # ---- scatter plots for Rp and vacancies (per ion) ----
    for tgt in PLOT_TARGETS:
        y_true = sub[tgt].astype(float).values
        y_hat  = sub[f"pred_{tgt}"].astype(float).values

        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        ax.scatter(y_true, y_hat, alpha=0.6)

        # identity line
        mn = min(np.min(y_true), np.min(y_hat))
        mx = max(np.max(y_true), np.max(y_hat))
        ax.plot([mn, mx], [mn, mx], linestyle="--")

        ax.set_xlabel("SRIM Ground Truth", fontsize=16, fontweight="bold")
        ax.set_ylabel("Model Prediction", fontsize=16, fontweight="bold")

        r2 = r2_score(y_true, y_hat)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        mae = mean_absolute_error(y_true, y_hat)

        ax.set_title(
            f"Ion-Specific Predicted vs SRIM — {tgt} ({ion})\n"
            f"(Stopped Only, 10-Fold CV)  R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}",
            fontsize=14, fontweight="bold"
        )

        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        safe_tgt = tgt.replace("/", "_")
        figpath = os.path.join(OUTDIR, f"Fig_8_3_Scatter_{safe_tgt}_{ion}_300dpi.tif")
        plt.savefig(figpath, dpi=300, format="tiff")
        plt.close(fig)

# save regression tables
reg_df = pd.DataFrame(reg_rows).sort_values("ion")
reg_csv = os.path.join(OUTDIR, "Table_8_3_Regression_AvgMetrics_By_Ion.csv")
reg_df.to_csv(reg_csv, index=False)

per_target_df = pd.DataFrame(per_target_rows).sort_values(["ion", "target"])
per_target_csv = os.path.join(OUTDIR, "Table_8_3_Regression_PerTargetMetrics_By_Ion.csv")
per_target_df.to_csv(per_target_csv, index=False)

print("\n==================== DONE ====================")
print(f"Output folder: {OUTDIR}")
print(f"Saved tables:\n  - {gatekeeper_csv}\n  - {reg_csv}\n  - {per_target_csv}")
print("Saved figures:")
print("  - Fig_8_3_Gatekeeper_CM_<ION>_300dpi.tif")
print("  - Fig_8_3_Scatter_Rp_A_<ION>_300dpi.tif")
print("  - Fig_8_3_Scatter_vacancies_per_ion_<ION>_300dpi.tif")
