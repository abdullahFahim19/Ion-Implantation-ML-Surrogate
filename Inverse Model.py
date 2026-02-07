import pandas as pd
import numpy as np
import warnings

# Scikit-Learn Components
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Suppress warnings for clean CLI output
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_FILE = 'full_database.csv'

# ==========================================
# 2. RECOMMENDER CLASS
# ==========================================
class SpecificRecommender:
    """
    Inverse model that predicts required Energy (y) based on
    Target Depth (X_1) and Material Properties (X_2).
    """
    def __init__(self):
        self.model = None
        self.valid_substrates = []
        self.valid_ions = []

    def load_and_train(self):
        print("--- Initializing Targeted Recommender System ---")
        try:
            df = pd.read_csv(DATA_FILE)
            # Data Cleaning
            df['substrate'] = df['substrate'].replace({
                'Ga50As50': 'GaAs',
                'Si50C50': 'SiC'
            })
        except FileNotFoundError:
            print("[ERROR] Database not found."); return

        # Physics Filter: Train ONLY on naturally stopped ions.
        # Transmission events do not have a defined Rp dependent on stopping physics
        # in the same way, so they are excluded to prevent model confusion.
        df_clean = df[
            (df['is_profile'] == 1) &
            (df['transmitted'] < 50) # Ensure majority stopped
        ].copy()

        # Inverting the problem: Inputs = Substrate, Ion, Target Rp, Angle
        X = df_clean[['substrate', 'ion', 'Rp_A', 'angle_deg']]
        # Target = Energy required
        y = df_clean['energy_keV']

        self.valid_substrates = sorted(df_clean['substrate'].unique())
        self.valid_ions = sorted(df_clean['ion'].unique())

        # Preprocessing Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['substrate', 'ion']),
                ('num', StandardScaler(), ['Rp_A', 'angle_deg'])
            ]
        )

        # Regressor Pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        ])

        print("   -> Training Inverse Model...")
        self.model.fit(X, y)
        print("   -> System Ready.")

    def get_solutions(self, substrate, target_rp, thickness_limit):
        """
        Iterates through valid Ions and Angles for a fixed Substrate
        to find valid Energy configurations.
        """
        # We test standard incidence angles
        standard_angles = [0, 7, 30, 45, 60]
        solutions = []

        print(f"\n   >> Optimizing configuration for {substrate} @ {target_rp} A...")

        for ion in self.valid_ions:
            for angle in standard_angles:
                # Construct query vector
                query = pd.DataFrame({
                    'substrate': [substrate],
                    'ion': [ion],
                    'Rp_A': [target_rp],
                    'angle_deg': [angle]
                })

                # Predict required energy
                pred_energy = self.model.predict(query)[0]

                # Validity Check:
                # 1. Energy must be within SRIM simulator limits (10 - 10,000 keV)
                # 2. Target depth must be less than wafer thickness (physical constraint)
                if 10 <= pred_energy <= 10000 and target_rp < thickness_limit:
                    solutions.append({
                        'Ion': ion,
                        'Angle (deg)': angle,
                        'Energy (keV)': round(pred_energy, 2)
                    })

        return pd.DataFrame(solutions)

# ==========================================
# 3. USER INTERFACE
# ==========================================
def run_tool():
    tool = SpecificRecommender()
    tool.load_and_train()

    while True:
        try:
            print("\n" + "="*50)
            print("   TARGETED ION SOLVER")
            print("="*50)

            # Input 1: Substrate (Fixed choice)
            sub = input(f"1. Select Substrate {tool.valid_substrates}: ").strip()
            if sub not in tool.valid_substrates:
                print("   [!] Invalid Substrate."); continue

            # Input 2: Desired result
            rp = float(input("2. Target Depth (Rp) [Angstrom]: "))

            # Input 3: Constraint
            thick = float(input("3. Wafer Thickness [Angstrom]: "))

            # Compute
            results = tool.get_solutions(sub, rp, thick)

            if results.empty:
                print("\n[!] No valid configurations found within energy limits (10-10,000 keV).")
                print("    Try reducing the target depth or changing the substrate.")
            else:
                # Formatting Output
                results = results.sort_values(by=['Ion', 'Energy (keV)'])
                print(f"\nValid Configurations for {sub} (Depth: {rp} A):")
                print("-" * 65)
                print(f"{'Ion':<10} | {'Angle (deg)':<15} | {'Required Energy':<20}")
                print("-" * 65)

                for _, row in results.iterrows():
                    print(f"{row['Ion']:<10} | {row['Angle (deg)']:<15.0f} | {row['Energy (keV)']:.2f} keV")
                print("-" * 65)

        except ValueError:
            print("   [!] Invalid Input. Please enter numeric values where required.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    run_tool()
