# ============================================================
# CHAPTER 8.1 ASSETS GENERATOR (UPDATED WITH GRADIENT BOOSTING)
# Exports:
#   - Fig 8.1 Confusion Matrix (counts + row %) -> TIF 300dpi
#   - Fig 8.2 Pred vs SRIM (Rp)                  -> TIF 300dpi
#   - Fig 8.3 Pred vs SRIM (Vacancies)           -> TIF 300dpi
#   - Table: Zero-event model comparison        -> CSV
#   - Table: Stopped regression comparison      -> CSV
#   - Table 8.1: Per-target metrics (RF)        -> CSV
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingRegressor, GradientBoostingClassifier # Added Classifier
)
from sklearn.multioutput import MultiOutputRegressor

import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "sample_database.csv"
OUT_DIR = "./Chapter8_Assets/"   # Created a specific folder for organization

# Gatekeeper CV (Classification)
N_SPLITS_CLS = 10 

# Regression CV
N_SPLITS_REG_5 = 5
N_SPLITS_REG_10 = 10

# K choices for KNN
K_LIST = [3, 5, 10] # Reduced list to save time, mostly 3 or 5 is best

# Final regressor settings
RF_REG_N_EST = 300
RF_CLS_N_EST = 300
GB_CLS_N_EST = 200 # Settings for Gradient Boosting

# ----------------------------
# HELPERS
# ----------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_makedirs(path):
    if path and path != "./":
        os.makedirs(path, exist_ok=True)

def save_csv(df_, filename):
    fp = os.path.join(OUT_DIR, filename)
    df_.to_csv(fp, index=False)
    print(f"   -> Saved Table: {fp}")

def scatter_pred_vs_true(y_true, y_pred, title, fname_base):
    plt.figure(figsize=(6.4, 6.4))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=40)
    
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2)  # y=x red dashed line
    plt.xlabel("SRIM Ground Truth", fontsize=12, fontweight='bold')
    plt.ylabel("Model Prediction", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    save_tif = os.path.join(OUT_DIR, f"{fname_base}_300dpi.tif")
    save_png = os.path.join(OUT_DIR, f"{fname_base}_300dpi.png")
    
    plt.savefig(save_tif, dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(save_png, dpi=300)
    plt.show()
    print(f"   -> Saved Figure: {save_tif}")

def plot_confusion_counts_percent(y_true, y_pred, title, fname_base):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    # Normalize by row (True Labels) to get Recall/Sensitivity
    cm_row = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    tn, fp = cm[0,0], cm[0,1]
    fn, tp = cm[1,0], cm[1,1]

    print("\n   [Confusion Matrix Stats]")
    print(f"   Transmission Recall: {tn/(tn+fp):.2%} ({tn}/{tn+fp})")
    print(f"   Stopped Recall:      {tp/(tp+fn):.2%} ({tp}/{tp+fn})")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_row, interpolation="nearest", cmap="Blues")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Recall (Normalized by True Class)', rotation=-90, va="bottom")

    classes = ["Transmission (0)", "Stopped (1)"]
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(classes, fontsize=11, fontweight='bold')
    ax.set_yticklabels(classes, fontsize=11, fontweight='bold')
    
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Annotate with Count and Percentage
    thresh = cm_row.max() / 2.
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_row[i, j] * 100
            color = "white" if cm_row[i, j] > thresh else "black"
            
            text_str = f"{count}\n({pct:.1f}%)"
            ax.text(j, i, text_str, ha="center", va="center", 
                    color=color, fontsize=14, fontweight='bold')

    plt.tight_layout()
    
    save_tif = os.path.join(OUT_DIR, f"{fname_base}_300dpi.tif")
    save_png = os.path.join(OUT_DIR, f"{fname_base}_300dpi.png")
    
    plt.savefig(save_tif, dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(save_png, dpi=300)
    plt.show()
    print(f"   -> Saved Figure: {save_tif}")

# ----------------------------
# MAIN EXECUTION
# ----------------------------
safe_makedirs(OUT_DIR)

# 1) LOAD
print("--- 1. Loading Data ---")
try:
    df = pd.read_csv(DATA_PATH)
    df["substrate"] = df["substrate"].replace({"Ga50As50": "GaAs", "Si50C50": "SiC"})
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    print(f"[Error] {DATA_PATH} not found. Upload it first.")
    exit()

# 2) FEATURES
INPUT_FEATURES = ["substrate", "ion", "energy_keV", "angle_deg", "thickness_A"]
Y_CLASS = "is_profile"  # 0=Transmission, 1=Stopped

categorical_features = ["substrate", "ion"]
numerical_features = ["energy_keV", "angle_deg", "thickness_A"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ]
)

X = df[INPUT_FEATURES].copy()
y = df[Y_CLASS].astype(int).copy()

# ============================================================
# A) CLASSIFICATION COMPARISON (Now with Gradient Boosting)
# ============================================================
print("\n--- 2. Comparing Classifiers (10-Fold CV) ---")
cv_cls = StratifiedKFold(n_splits=N_SPLITS_CLS, shuffle=True, random_state=42)

models_cls = []
# 1. Logistic
models_cls.append((
    "Logistic Regression",
    Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=2000))])
))
# 2. KNN Loop
for k in K_LIST:
    models_cls.append((
        f"KNN (k={k})",
        Pipeline([("prep", preprocessor), ("clf", KNeighborsClassifier(n_neighbors=k))])
    ))
# 3. Random Forest
models_cls.append((
    "Random Forest",
    Pipeline([("prep", preprocessor), 
              ("clf", RandomForestClassifier(n_estimators=RF_CLS_N_EST, random_state=42, n_jobs=-1))])
))
# 4. Gradient Boosting (NEW ADDITION)
models_cls.append((
    "Gradient Boosting",
    Pipeline([("prep", preprocessor), 
              ("clf", GradientBoostingClassifier(n_estimators=GB_CLS_N_EST, random_state=42))])
))

rows = []
pred_store = {}

# Evaluate Models
for name, model in models_cls:
    print(f"   Running {name}...")
    # Using cross_val_predict to get predictions for the whole dataset
    y_pred = cross_val_predict(model, X, y, cv=cv_cls, n_jobs=-1)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    rows.append([name, acc, prec, rec, f1])
    pred_store[name] = y_pred

# Create Table
cls_table = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision (Stopped)", "Recall (Stopped)", "F1-Score (Stopped)"])
cls_table = cls_table.sort_values("F1-Score (Stopped)", ascending=False).reset_index(drop=True)

print("\n[Gatekeeper Leaderboard]")
print(cls_table)
save_csv(cls_table, "Table_Gatekeeper_Comparison.csv")

# --- GENERATE FIG 8.1 (Using the Winner) ---
# Automatically pick the best model from the leaderboard
BEST_GATEKEEPER_NAME = cls_table.iloc[0]["Model"]
print(f"\n>> Generating Fig 8.1 using best model: {BEST_GATEKEEPER_NAME}")

plot_confusion_counts_percent(
    y_true=y.values,
    y_pred=pred_store[BEST_GATEKEEPER_NAME],
    title=f"Fig. 8.1 Gatekeeper Confusion Matrix\n(Model: {BEST_GATEKEEPER_NAME}, Method: 10-Fold CV)",
    fname_base="Fig_8_1_Confusion_Matrix"
)

# ============================================================
# B) REGRESSION COMPARISON & ANALYSIS
# ============================================================
print("\n--- 3. Regression Analysis (Stopped Ions Only) ---")

TARGETS = [
    "Rp_A", "dRp_A",
    "lateral_range_A", "lateral_straggle_A",
    "radial_range_A", "radial_straggle_A",
    "vacancies_per_ion",
    "backscattered", "transmitted"
]

# Filter for stopped ions only
df_stop = df[df[Y_CLASS] == 1].copy()
Xr = df_stop[INPUT_FEATURES].copy()
Yr = df_stop[TARGETS].copy()

cv5  = KFold(n_splits=N_SPLITS_REG_5, shuffle=True, random_state=42)
cv10 = KFold(n_splits=N_SPLITS_REG_10, shuffle=True, random_state=42)

def make_reg_pipeline(reg):
    return Pipeline([("prep", preprocessor), ("reg", reg)])

def eval_reg_cv(model, X, Y, cv):
    # Cross val predict for robustness
    Y_pred = cross_val_predict(model, X, Y, cv=cv, n_jobs=-1)
    r2s = []
    rmses = []
    # Calculate metrics per target then average
    for k in range(Y.shape[1]):
        r2s.append(r2_score(Y.iloc[:,k], Y_pred[:,k]))
        rmses.append(rmse(Y.iloc[:,k].values, Y_pred[:,k]))
    return float(np.mean(r2s)), float(np.mean(rmses))

models_reg = []
models_reg.append(("Linear Regression", make_reg_pipeline(LinearRegression())))
for k in K_LIST:
    models_reg.append((f"KNN (k={k})", make_reg_pipeline(MultiOutputRegressor(KNeighborsRegressor(n_neighbors=k)))))

models_reg.append(("Gradient Boosting", make_reg_pipeline(MultiOutputRegressor(GradientBoostingRegressor(random_state=42)))))
models_reg.append(("Random Forest", make_reg_pipeline(MultiOutputRegressor(RandomForestRegressor(n_estimators=RF_REG_N_EST, random_state=42, n_jobs=-1)))))

rows_reg = []
for name, model in models_reg:
    print(f"   Evaluating {name}...")
    r2_5, rmse_5 = eval_reg_cv(model, Xr, Yr, cv5)
    r2_10, rmse_10 = eval_reg_cv(model, Xr, Yr, cv10)
    rows_reg.append([name, r2_5, rmse_5, r2_10, rmse_10])

reg_table = pd.DataFrame(rows_reg, columns=["Model", "Avg R2 (5-fold)", "Avg RMSE (5-fold)", "Avg R2 (10-fold)", "Avg RMSE (10-fold)"])
reg_table = reg_table.sort_values("Avg R2 (10-fold)", ascending=False).reset_index(drop=True)

print("\n[Regression Leaderboard]")
print(reg_table)
save_csv(reg_table, "Table_Regression_Comparison.csv")

# --- GENERATE TABLE 8.1 & FIGURES (Using RF as standard, or best) ---
# Usually RF is preferred for multi-output physics consistency, even if GB is slightly better
final_model_name = "Random Forest"
print(f"\n>> Generating Final Assets using: {final_model_name}")

final_pipeline = make_reg_pipeline(MultiOutputRegressor(RandomForestRegressor(
    n_estimators=RF_REG_N_EST, random_state=42, n_jobs=-1
)))

# Get 10-Fold CV predictions for final evaluation
Y_pred_final = cross_val_predict(final_pipeline, Xr, Yr, cv=cv10, n_jobs=-1)

# Metric Table per Target
rows_metrics = []
for idx, col in enumerate(TARGETS):
    y_true = Yr[col].values
    y_hat  = Y_pred_final[:, idx]
    mae = mean_absolute_error(y_true, y_hat)
    rmse_val = rmse(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)
    rows_metrics.append([col, mae, rmse_val, r2])

table_8_1 = pd.DataFrame(rows_metrics, columns=["Output Target", "MAE", "RMSE", "R2 Score"])
print("\n[Table 8.1: Per-Target Performance]")
print(table_8_1)
save_csv(table_8_1, "Table_8_1_Per_Target_Metrics.csv")

# Figures 8.2 & 8.3
print("\n>> Saving Scatter Plots...")
rp_idx = TARGETS.index("Rp_A")
vac_idx = TARGETS.index("vacancies_per_ion")

scatter_pred_vs_true(
    Yr["Rp_A"].values, Y_pred_final[:, rp_idx],
    title="Fig. 8.2 Predicted vs SRIM: Projected Range $R_p$",
    fname_base="Fig_8_2_Rp_Scatter"
)

scatter_pred_vs_true(
    Yr["vacancies_per_ion"].values, Y_pred_final[:, vac_idx],
    title="Fig. 8.3 Predicted vs SRIM: Vacancies per Ion",
    fname_base="Fig_8_3_Vacancies_Scatter"
)

print("\n[COMPLETE] All assets saved in:", OUT_DIR)
