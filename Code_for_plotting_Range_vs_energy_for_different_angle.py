import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FILE = 'sample_database.csv'
OUT_DIR = 'Si_Physics_Plots'    # Folder name
DPI = 300                       # High Resolution for Thesis
SUBSTRATE = 'GaAs'                # Target Substrate (CHANGE HERE FOR DIFFERENT SUBSTRATE)
IONS = ['B', 'P', 'As', 'Ar', 'Mg'] # List of Ions

# Custom Colors matching your preference
CUSTOM_COLORS = {
    0: 'navy',
    7: 'green',
    30: 'blue',
    45: 'black',
    60: 'gold',
    89.9: 'red'
}

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

# Set Style
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5) # Bigger font for single figures

# -----------------------------
# DATA LOADING
# -----------------------------
if not os.path.exists(DATA_FILE):
    print(f"Error: '{DATA_FILE}' not found!")
    exit()

df = pd.read_csv(DATA_FILE)

# -----------------------------
# PLOTTING LOOP
# -----------------------------
for ion in IONS:
    print(f"Generating plot for {ion} in {SUBSTRATE}...")

    # Filter Data
    subdf = df[(df['substrate'] == SUBSTRATE) & (df['ion'] == ion)].copy()

    # Filter valid ranges (Stopped ions)
    subdf = subdf[subdf['Rp_A'] > 0].sort_values(by='energy_keV')

    if subdf.empty:
        print(f"Skipping {ion}: No valid data found.")
        continue

    # Initialize Figure
    plt.figure(figsize=(8, 6))

    # Plot Line Chart
    sns.lineplot(
        data=subdf,
        x='energy_keV',
        y='Rp_A',
        hue='angle_deg',
        palette=CUSTOM_COLORS,
        marker='o',
        markersize=6,
        linewidth=2,
        legend='full'
    )

    # Axis Scales & Labels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Implantation Energy (keV)", fontweight='bold')
    plt.ylabel("Projected Range ($R_p$) [$\AA$]", fontweight='bold')
    plt.title(f"{ion} in Gallium Arsenide (GaAs): Range vs. Energy", fontweight='bold', fontsize=14)

    # Grid
    plt.grid(True, which="major", linestyle='-', linewidth=0.8, alpha=0.6)
    plt.grid(True, which="minor", linestyle=':', linewidth=0.5, alpha=0.4)

    # Legend formatting
    plt.legend(title="Tilt Angle ($^\circ$)", title_fontsize=12, fontsize=10, loc='best')

    plt.tight_layout()

    # Save as TIFF
    filename = f"Fig_6.2_{ion}_on_GaAs_Range.tif"
    save_path = os.path.join(OUT_DIR, filename)

    plt.savefig(save_path, dpi=DPI, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    plt.close() # Close memory to avoid overload

    print(f"Saved: {save_path}")

print("\nAll 5 plots generated successfully in folder:", OUT_DIR)
