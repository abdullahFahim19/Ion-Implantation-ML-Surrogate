import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FILE = 'sample_database.csv'
OUT_DIR = 'Si_Physics_Plots'    # Same folder as before
DPI = 300                       # High Resolution
SUBSTRATE = 'GaAs'                # Target Substrate (CHANGE HERE FOR DIFFERENT SUBSTRATE)
IONS = ['B', 'P', 'As', 'Ar', 'Mg']

# Custom Colors
CUSTOM_COLORS = {
    0: 'navy',
    7: 'green',
    30: 'blue',
    45: 'black',
    60: 'gold',
    89.9: 'red'
}

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Set Style
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5)

# -----------------------------
# DATA LOADING
# -----------------------------
if not os.path.exists(DATA_FILE):
    print(f"Error: '{DATA_FILE}' not found!")
    exit()

df = pd.read_csv(DATA_FILE)

# -----------------------------
# PLOTTING LOOP (VACANCY)
# -----------------------------
for ion in IONS:
    print(f"Generating Vacancy plot for {ion} in {SUBSTRATE}...")

    # Filter Data
    subdf = df[(df['substrate'] == SUBSTRATE) & (df['ion'] == ion)].copy()

    # Filter valid vacancies (Non-zero)
    subdf = subdf[subdf['vacancies_per_ion'] > 0].sort_values(by='energy_keV')

    if subdf.empty:
        print(f"Skipping {ion}: No valid vacancy data found.")
        continue

    # Initialize Figure
    plt.figure(figsize=(8, 6))

    # Plot Line Chart (Square markers 's' for vacancies to distinguish from Range)
    sns.lineplot(
        data=subdf,
        x='energy_keV',
        y='vacancies_per_ion',
        hue='angle_deg',
        palette=CUSTOM_COLORS,
        marker='s',        # Square marker for Vacancy
        markersize=6,
        linewidth=2,
        legend='full'
    )

    # Axis Scales & Labels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Implantation Energy (keV)", fontweight='bold')
    plt.ylabel("Vacancies / Ion", fontweight='bold')
    plt.title(f"{ion} in Gallium Arsenide (GaAs): Vacancy Production", fontweight='bold', fontsize=14)

    # Grid
    plt.grid(True, which="major", linestyle='-', linewidth=0.8, alpha=0.6)
    plt.grid(True, which="minor", linestyle=':', linewidth=0.5, alpha=0.4)

    # Legend formatting
    plt.legend(title="Tilt Angle ($^\circ$)", title_fontsize=12, fontsize=10, loc='best')

    plt.tight_layout()

    # Save as TIFF
    filename = f"Fig_6.3_{ion}_on_GaAs_Vacancy.tif"
    save_path = os.path.join(OUT_DIR, filename)

    plt.savefig(save_path, dpi=DPI, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    plt.close()

    print(f"Saved: {save_path}")

print("\nAll 5 Vacancy plots generated successfully in folder:", OUT_DIR)
