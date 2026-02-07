import pandas as pd
import numpy as np
import warnings

# Scikit-Learn Ensemble Models
from sklearn.ensemble import (
    RandomForestRegressor, 
    RandomForestClassifier, 
    HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputRegressor

# Preprocessing & Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Suppress warnings for clean presentation in demonstration
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
DATA_FILE = 'sample_database.csv' 

# Input Features
INPUT_FEATURES = ['substrate', 'ion', 'energy_keV', 'angle_deg', 'thickness_A']

# Target Variable (Classification)
# 0 = Transmitted (Passed through), 1 = Stopped (Implanted)
Y_CLASSIFIER = 'is_profile' 

# Target Variables (Regression Groups)
TARGETS_RANGE = [
    'Rp_A', 'dRp_A', 'lateral_range_A', 'lateral_straggle_A',
    'radial_range_A', 'radial_straggle_A', 'backscattered', 'transmitted'
]
TARGETS_VACANCY = ['vacancies_per_ion']
TARGETS_MOMENTS = ['skewness', 'kurtosis']

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_prep_data(filepath):
    print("--- Thesis Hybrid Model Initialization ---")
    print("1. Loading Database...")
    try:
        df = pd.read_csv(filepath)
        
        # Standardize Substrate Names (Data Cleaning)
        df['substrate'] = df['substrate'].replace({
            'Ga50As50': 'GaAs', 
            'Si50C50': 'SiC'
        })
        
        print(f"   -> Successfully loaded {len(df)} simulations.")
        return df
    except FileNotFoundError:
        print(f"[ERROR] Could not find '{filepath}'. Exiting.")
        exit()

# ==========================================
# 3. PIPELINE CONSTRUCTION
# ==========================================
def build_transformers():
    print("2. Building Preprocessing Pipelines...")
    
    categorical_features = ['substrate', 'ion']
    numerical_features = ['energy_keV', 'angle_deg', 'thickness_A']

    # Standard Scaler for Geometry/Ranges
    preprocessor_std = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ], remainder='passthrough'
    )

    # Robust Scaler for Statistical Moments (Handling outliers)
    preprocessor_robust = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', RobustScaler(), numerical_features)
        ], remainder='passthrough'
    )
    
    return preprocessor_std, preprocessor_robust

# ==========================================
# 4. MODEL TRAINING
# ==========================================
def train_models(df, pre_std, pre_rob):
    # --- Stage 1: Classifier (The Gatekeeper) ---
    print("3. Training Stage 1: Classifier (Gatekeeper)...")
    
    X = df[INPUT_FEATURES]
    y_cls = df[Y_CLASSIFIER]

    clf_pipeline = Pipeline(steps=[
        ('preprocessor', pre_std),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    clf_pipeline.fit(X, y_cls)

    # --- Stage 2: Regressors (The Physics Engine) ---
    print("4. Training Stage 2: Regressors (Physics Engine)...")

    # Filter Data: Train regressors ONLY on ions that stopped
    df_stopped = df[df[Y_CLASSIFIER] == 1].copy()
    X_stopped = df_stopped[INPUT_FEATURES]

    # Model A: Ranges & Geometry
    y_range = df_stopped[TARGETS_RANGE]
    reg_range = Pipeline(steps=[
        ('preprocessor', pre_std),
        ('regressor', MultiOutputRegressor(
            RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        ))
    ])
    reg_range.fit(X_stopped, y_range)

    # Model B: Vacancy Production
    y_vac = df_stopped[TARGETS_VACANCY].values.ravel()
    reg_vac = Pipeline(steps=[
        ('preprocessor', pre_std),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    reg_vac.fit(X_stopped, y_vac)

    # Model C: Statistical Moments (Handling potential NaNs in training data)
    df_moments = df_stopped.dropna(subset=TARGETS_MOMENTS)
    X_moments = df_moments[INPUT_FEATURES]
    y_moments = df_moments[TARGETS_MOMENTS]

    reg_moments = Pipeline(steps=[
        ('preprocessor', pre_rob),
        ('regressor', MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42)))
    ])
    reg_moments.fit(X_moments, y_moments)

    print("   -> All Models Trained Successfully.")
    print("="*60)
    
    return clf_pipeline, reg_range, reg_vac, reg_moments

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
def predict_ion_behavior(user_input_df, models):
    clf, r_range, r_vac, r_moments = models
    
    # Step 1: Classification
    is_stopped = clf.predict(user_input_df)[0]
    results = {}

    if is_stopped == 0:
        status = "TRANSMISSION EVENT (Class 0)"
        # Default physics for transmission (pass-through)
        for col in TARGETS_RANGE: results[col] = 0.0
        for col in TARGETS_VACANCY: results[col] = 0.0
        for col in TARGETS_MOMENTS: results[col] = np.nan
        results['transmitted'] = 10000.0 # Assuming 10k ion simulation
        results['backscattered'] = 0.0

    else:
        status = "STOPPED EVENT (Class 1)"
        # Step 2: Regression
        pred_range = r_range.predict(user_input_df)
        pred_vac = r_vac.predict(user_input_df)
        pred_moments = r_moments.predict(user_input_df)

        # Map results to column names
        for i, col in enumerate(TARGETS_RANGE): results[col] = pred_range[0][i]
        results['vacancies_per_ion'] = pred_vac[0]
        for i, col in enumerate(TARGETS_MOMENTS): results[col] = pred_moments[0][i]

    return status, pd.DataFrame([results])

# ==========================================
# 6. INTERACTIVE DEMO LOOP
# ==========================================
def run_interactive_mode(df, models):
    valid_subs = sorted(df['substrate'].unique())
    valid_ions = sorted(df['ion'].unique())

    print("\n--- SYSTEM READY FOR QUERY ---")

    while True:
        try:
            print("\nInput Parameters:")
            s = input(f"  Substrate {valid_subs}: ").strip()
            if s not in valid_subs: 
                print("  [!] Invalid Substrate"); continue

            i = input(f"  Ion {valid_ions}: ").strip()
            if i not in valid_ions: 
                print("  [!] Invalid Ion"); continue

            e = float(input("  Energy (keV) [10-10000]: "))
            a = float(input("  Angle (deg) [0-89.9]: "))
            t = float(input("  Thickness (A) [100-10000]: "))

            # Create Query DataFrame
            query = pd.DataFrame({
                'substrate':[s], 'ion':[i], 'energy_keV':[e],
                'angle_deg':[a], 'thickness_A':[t]
            })

            # Predict
            status, res = predict_ion_behavior(query, models)

            # Output Display
            print(f"\n>> PREDICTION: {status}")
            print("-" * 50)

            if status.startswith("STOPPED"):
                def fmt(val): return f"{val:.2f}" # Helper for formatting

                print(f"  [Depth Profile]")
                print(f"  Projected Range (Rp):    {fmt(res['Rp_A'].values[0])} A")
                print(f"  Longitudinal Straggle:   {fmt(res['dRp_A'].values[0])} A")
                print(f"  Skewness:                {fmt(res['skewness'].values[0])}")
                print(f"  Kurtosis:                {fmt(res['kurtosis'].values[0])}")

                print(f"\n  [Lateral/Radial Distribution]")
                print(f"  Lateral Range:           {fmt(res['lateral_range_A'].values[0])} A")
                print(f"  Lateral Straggle:        {fmt(res['lateral_straggle_A'].values[0])} A")
                print(f"  Radial Range:            {fmt(res['radial_range_A'].values[0])} A")
                print(f"  Radial Straggle:         {fmt(res['radial_straggle_A'].values[0])} A")

                print(f"\n  [Defects & Particle Balance]")
                print(f"  Vacancies per Ion:       {fmt(res['vacancies_per_ion'].values[0])}")
                print(f"  Backscattered Ions:      {fmt(res['backscattered'].values[0])}")
                print(f"  Transmitted Ions:        {fmt(res['transmitted'].values[0])}")

            else:
                print("  [Result]")
                print("  The ion had too much energy or the substrate was too thin.")
                print("  -> Transmitted Ions: ~10000 (100%)")
                print("  -> Range: 0.0 A (Passed through)")

            print("-" * 50)

        except ValueError:
            print("  [!] Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    data_df = load_and_prep_data(DATA_FILE)
    
    # 2. Prepare Transformers
    p_std, p_rob = build_transformers()
    
    # 3. Train Models
    trained_models = train_models(data_df, p_std, p_rob)
    
    # 4. Run CLI
    run_interactive_mode(data_df, trained_models)
