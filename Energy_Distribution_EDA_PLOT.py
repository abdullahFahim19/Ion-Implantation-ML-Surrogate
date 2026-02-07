import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FILE = 'sample_database.csv'
OUT_DIR = 'EDA_Plots_HighRes'   # Folder where images will be saved
DPI = 300                       # High resolution for thesis

# Create output directory if not exists
os.makedirs(OUT_DIR, exist_ok=True)

# Set professional academic style
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.4) # Slightly larger font for readability

# -----------------------------
# LOAD DATA
# -----------------------------
if not os.path.exists(DATA_FILE):
    print(f"[ERROR] '{DATA_FILE}' not found! Please ensure file is in directory.")
    exit()

print("Loading database...")
df = pd.read_csv(DATA_FILE)

# -----------------------------
# PLOT 1: ENERGY DISTRIBUTION (Log-Scale)
# -----------------------------
print("Generating Plot 1: Energy Distribution...")
plt.figure(figsize=(8, 6))

# Log transformation for better visualization of wide energy ranges
log_energy = np.log10(df['energy_keV'])

sns.histplot(log_energy, bins=20, color='skyblue', edgecolor='black', kde=True)

plt.title("Distribution of Implantation Energy (Log-Scale)", fontweight='bold')
plt.xlabel("Log10(Energy [keV])")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save
save_path_1 = os.path.join(OUT_DIR, "Fig_6.8a_Energy_Dist.tif")
plt.savefig(save_path_1, dpi=DPI, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
plt.show()
print(f"   -> Saved: {save_path_1}")

# -----------------------------
# PLOT 2: TILT ANGLE DISTRIBUTION
# -----------------------------
print("Generating Plot 2: Tilt Angle Distribution...")
plt.figure(figsize=(8, 6))

sns.countplot(data=df, x='angle_deg', palette='viridis', edgecolor='black')

plt.title("Distribution of Tilt Angles", fontweight='bold')
plt.xlabel("Tilt Angle (Degrees)")
plt.ylabel("Count")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save
save_path_2 = os.path.join(OUT_DIR, "Fig_6.8b_Angle_Dist.tif")
plt.savefig(save_path_2, dpi=DPI, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
plt.show()
print(f"   -> Saved: {save_path_2}")

# -----------------------------
# PLOT 3: SUBSTRATE THICKNESS DISTRIBUTION
# -----------------------------
print("Generating Plot 3: Thickness Distribution...")
plt.figure(figsize=(8, 6))

sns.histplot(df['thickness_A'], bins=15, color='lightgreen', edgecolor='black', kde=False)

plt.title("Distribution of Substrate Thickness", fontweight='bold')
plt.xlabel("Thickness ($\AA$)")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save
save_path_3 = os.path.join(OUT_DIR, "Fig_6.8c_Thickness_Dist.tif")
plt.savefig(save_path_3, dpi=DPI, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
plt.show()
print(f"   -> Saved: {save_path_3}")

# -----------------------------
# PLOT 4: CLASS BALANCE (Updated Logic)
# -----------------------------
print("Generating Plot 4: Class Balance (Updated Logic)...")

# --- UPDATED LOGIC ---
# Class is 'Stopped (Valid)' if SRIM calculated a valid Range (Rp > 0)
# Class is 'Transmission (Zero)' only if Rp is 0 (Complete pass-through)
df['class'] = df['Rp_A'].apply(lambda x: 'Stopped (Valid)' if x > 0 else 'Transmission (Zero)')

plt.figure(figsize=(8, 6))

class_counts = df['class'].value_counts()
# Ensure consistent color mapping: Green for Stopped (Good), Red for Transmission (Bad/Zero)
colors = {'Stopped (Valid)': '#2ca02c', 'Transmission (Zero)': '#d62728'}

ax = sns.barplot(
    x=class_counts.index, 
    y=class_counts.values, 
    palette=colors, 
    edgecolor='black'
)

# Add counts on top of bars
for i, v in enumerate(class_counts.values):
    # Place text slightly above the bar
    ax.text(i, v + (v * 0.02), str(v), ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.title("Dataset Balance: Useful vs. Zero-Range Simulations", fontweight='bold')
plt.xlabel("Simulation Outcome")
plt.ylabel("Number of Samples")
plt.ylim(0, max(class_counts.values) * 1.15) # Add headroom for labels
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save
save_path_4 = os.path.join(OUT_DIR, "Fig_6.8d_Class_Balance_Updated.tif")
plt.savefig(save_path_4, dpi=DPI, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
plt.show()
print(f"   -> Saved: {save_path_4}")

print("\nAll plots generated and saved successfully in directory:", OUT_DIR)
