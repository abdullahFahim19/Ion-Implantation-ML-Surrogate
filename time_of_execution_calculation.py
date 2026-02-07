# Timing Calculation of forward and inverse model code

import os
import time
import joblib
import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================
TIME_CSV = "time.csv"            # The combination I want to analyse the time of executation
DATA_FILE = "sample_database.csv"  # training database file

# trained model save:
FORWARD_MODEL_JOBLIB = "forward_models.joblib"
INVERSE_MODEL_JOBLIB = "inverse_model.joblib"

# Benchmark controls
WARMUP_RUNS = 10          # first warmup: cache/JIT-ish effects to reduce
REPEAT_PER_ROW = 30       # each row prediction repeat then avg
REPEAT_INVERSE = 20       # inverse solver query numbers repeat then avg

# ==========================================================
# YOUR FORWARD MODEL (same logic as your code, cleaned)
# ==========================================================
INPUT_FEATURES = ["substrate", "ion", "energy_keV", "angle_deg", "thickness_A"]
Y_CLASSIFIER = "is_profile"

TARGETS_RANGE = [
    "Rp_A", "dRp_A", "lateral_range_A", "lateral_straggle_A",
    "radial_range_A", "radial_straggle_A", "backscattered", "transmitted"
]
TARGETS_VACANCY = ["vacancies_per_ion"]
TARGETS_MOMENTS = ["skewness", "kurtosis"]

def load_and_prep_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["substrate"] = df["substrate"].replace({"Ga50As50": "GaAs", "Si50C50": "SiC"})
    return df

def build_transformers():
    categorical_features = ["substrate", "ion"]
    numerical_features = ["energy_keV", "angle_deg", "thickness_A"]

    preprocessor_std = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features),
        ],
        remainder="drop",
    )

    preprocessor_robust = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", RobustScaler(), numerical_features),
        ],
        remainder="drop",
    )
    return preprocessor_std, preprocessor_robust

def train_forward_models(df: pd.DataFrame):
    pre_std, pre_rob = build_transformers()

    X = df[INPUT_FEATURES]
    y_cls = df[Y_CLASSIFIER]

    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", pre_std),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )
    clf_pipeline.fit(X, y_cls)

    df_stopped = df[df[Y_CLASSIFIER] == 1].copy()
    X_stopped = df_stopped[INPUT_FEATURES]

    y_range = df_stopped[TARGETS_RANGE]
    reg_range = Pipeline(
        steps=[
            ("preprocessor", pre_std),
            ("regressor", MultiOutputRegressor(
                RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
            )),
        ]
    )
    reg_range.fit(X_stopped, y_range)

    y_vac = df_stopped[TARGETS_VACANCY].values.ravel()
    reg_vac = Pipeline(
        steps=[
            ("preprocessor", pre_std),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )
    reg_vac.fit(X_stopped, y_vac)

    df_mom = df_stopped.dropna(subset=TARGETS_MOMENTS)
    X_mom = df_mom[INPUT_FEATURES]
    y_mom = df_mom[TARGETS_MOMENTS]
    reg_moments = Pipeline(
        steps=[
            ("preprocessor", pre_rob),
            ("regressor", MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42))),
        ]
    )
    reg_moments.fit(X_mom, y_mom)

    return clf_pipeline, reg_range, reg_vac, reg_moments

def predict_ion_behavior(user_input_df: pd.DataFrame, models):
    clf, r_range, r_vac, r_moments = models

    is_stopped = int(clf.predict(user_input_df)[0])
    results = {}

    if is_stopped == 0:
        # transmission event
        for col in TARGETS_RANGE:
            results[col] = 0.0
        for col in TARGETS_VACANCY:
            results[col] = 0.0
        for col in TARGETS_MOMENTS:
            results[col] = np.nan
        results["transmitted"] = 10000.0  # to match 10k ion convention
        results["backscattered"] = 0.0
    else:
        pred_range = r_range.predict(user_input_df)
        pred_vac = r_vac.predict(user_input_df)
        pred_mom = r_moments.predict(user_input_df)

        for i, col in enumerate(TARGETS_RANGE):
            results[col] = float(pred_range[0][i])
        results["vacancies_per_ion"] = float(pred_vac[0])
        for i, col in enumerate(TARGETS_MOMENTS):
            results[col] = float(pred_mom[0][i])

    return results

# ==========================================================
# YOUR INVERSE MODEL (same structure, cleaned)
# ==========================================================
class SpecificRecommender:
    def __init__(self):
        self.model = None
        self.valid_substrates = []
        self.valid_ions = []

    def load_and_train(self, data_file=DATA_FILE):
        df = pd.read_csv(data_file)
        df["substrate"] = df["substrate"].replace({"Ga50As50": "GaAs", "Si50C50": "SiC"})

        df_clean = df[(df["is_profile"] == 1) & (df["transmitted"] < 50)].copy()

        X = df_clean[["substrate", "ion", "Rp_A", "angle_deg"]]
        y = df_clean["energy_keV"]

        self.valid_substrates = sorted(df_clean["substrate"].unique())
        self.valid_ions = sorted(df_clean["ion"].unique())

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["substrate", "ion"]),
                ("num", StandardScaler(), ["Rp_A", "angle_deg"]),
            ],
            remainder="drop",
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        )

        self.model.fit(X, y)

    def get_solutions(self, substrate, target_rp, thickness_limit):
        standard_angles = [0, 7, 30, 45, 60]
        solutions = []

        for ion in self.valid_ions:
            for angle in standard_angles:
                query = pd.DataFrame({
                    "substrate": [substrate],
                    "ion": [ion],
                    "Rp_A": [target_rp],
                    "angle_deg": [angle],
                })
                pred_energy = float(self.model.predict(query)[0])

                if 10 <= pred_energy <= 10000 and target_rp < thickness_limit:
                    solutions.append({
                        "Ion": ion,
                        "Angle (deg)": angle,
                        "Energy (keV)": round(pred_energy, 2),
                    })

        return pd.DataFrame(solutions)

# ==========================================================
# BENCHMARK UTILITIES
# ==========================================================
def time_function(fn, repeats=50, warmup=10):
    # warmup
    for _ in range(warmup):
        fn()

    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter()

    total = (t1 - t0)
    avg = total / repeats
    return avg, total

def ensure_forward_models():
    if os.path.exists(FORWARD_MODEL_JOBLIB):
        print(f"[INFO] Loading forward models from: {FORWARD_MODEL_JOBLIB}")
        return joblib.load(FORWARD_MODEL_JOBLIB)

    print("[INFO] Forward model joblib not found. Training from full_database.csv ...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"'{DATA_FILE}' not found. Either provide it or place '{FORWARD_MODEL_JOBLIB}' joblib file."
        )
    df = load_and_prep_data(DATA_FILE)
    models = train_forward_models(df)
    joblib.dump(models, FORWARD_MODEL_JOBLIB)
    print(f"[INFO] Saved forward models to: {FORWARD_MODEL_JOBLIB}")
    return models

def ensure_inverse_model():
    if os.path.exists(INVERSE_MODEL_JOBLIB):
        print(f"[INFO] Loading inverse model from: {INVERSE_MODEL_JOBLIB}")
        return joblib.load(INVERSE_MODEL_JOBLIB)

    print("[INFO] Inverse model joblib not found. Training from full_database.csv ...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"'{DATA_FILE}' not found. Either provide it or place '{INVERSE_MODEL_JOBLIB}' joblib file."
        )
    tool = SpecificRecommender()
    tool.load_and_train(DATA_FILE)
    joblib.dump(tool, INVERSE_MODEL_JOBLIB)
    print(f"[INFO] Saved inverse model to: {INVERSE_MODEL_JOBLIB}")
    return tool

# ==========================================================
# MAIN: Forward timing for all combinations in time.csv
# ==========================================================
def benchmark_forward(time_csv=TIME_CSV, out_csv="forward_time_report.csv"):
    df_time = pd.read_csv(time_csv)

    # Filter only 10k ion rows if present
    if "num_ions" in df_time.columns:
        df_time = df_time[df_time["num_ions"] == 10000].copy()

    # Make sure required columns exist
    required = set(INPUT_FEATURES)
    missing = required - set(df_time.columns)
    if missing:
        raise ValueError(f"time.csv missing required columns: {sorted(missing)}")

    models = ensure_forward_models()

    rows = []
    for idx, row in df_time.iterrows():
        query = pd.DataFrame([{
            "substrate": row["substrate"],
            "ion": row["ion"],
            "energy_keV": float(row["energy_keV"]),
            "angle_deg": float(row["angle_deg"]),
            "thickness_A": float(row["thickness_A"]),
        }])

        def _one_call():
            _ = predict_ion_behavior(query, models)

        avg_sec, _ = time_function(_one_call, repeats=REPEAT_PER_ROW, warmup=WARMUP_RUNS)

        rec = row.to_dict()
        rec["forward_pred_avg_time_ms"] = avg_sec * 1000.0
        rows.append(rec)

    out = pd.DataFrame(rows)

    # Speedup columns (if SRIM time columns exist)
    srim_i5 = "execution_time_seconds on core i5"
    srim_i9 = "execution_time_seconds on core i9"

    if srim_i5 in out.columns:
        out["speedup_vs_srim_i5"] = out[srim_i5] / (out["forward_pred_avg_time_ms"] / 1000.0)
    if srim_i9 in out.columns:
        out["speedup_vs_srim_i9"] = out[srim_i9] / (out["forward_pred_avg_time_ms"] / 1000.0)

    out.to_csv(out_csv, index=False)
    print(f"[DONE] Forward timing report saved to: {out_csv}")
    print(f"       Rows benchmarked: {len(out)}")

# ==========================================================
# MAIN: Inverse timing for Rp=500A
# ==========================================================
def benchmark_inverse(target_rp_A=500.0, thickness_limit_A=10000.0, out_csv="inverse_time_report.csv"):
    tool = ensure_inverse_model()

    # We'll benchmark for every valid substrate
    records = []

    for sub in tool.valid_substrates:
        def _one_inverse_query():
            _ = tool.get_solutions(sub, target_rp_A, thickness_limit_A)

        avg_sec, _ = time_function(_one_inverse_query, repeats=REPEAT_INVERSE, warmup=WARMUP_RUNS)

        records.append({
            "substrate": sub,
            "target_Rp_A": target_rp_A,
            "thickness_limit_A": thickness_limit_A,
            "inverse_query_avg_time_ms": avg_sec * 1000.0,
            "ions_tested": len(tool.valid_ions),
            "angles_tested": 5,  # [0,7,30,45,60]
            "total_predictions_per_query": len(tool.valid_ions) * 5,
        })

    out = pd.DataFrame(records)
    out.to_csv(out_csv, index=False)
    print(f"[DONE] Inverse timing report saved to: {out_csv}")
    print(f"       Substrates benchmarked: {len(out)}")

# ==========================================================
# RUN ALL
# ==========================================================
if __name__ == "__main__":
    # Forward: time.csv all combination of forward prediction time estimate
    benchmark_forward(TIME_CSV, out_csv="forward_time_report.csv")

    # Inverse: Rp=500 Angstorm,  inverse solver query time estimate
    benchmark_inverse(target_rp_A=500.0, thickness_limit_A=10000.0, out_csv="inverse_time_report.csv")
