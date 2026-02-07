# Code for bell distribution curve
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters (edit as you like) ---
Rp = 100.0          # projected range (e.g., nm)
dRp = 20.0          # straggle (e.g., nm)
Q = 1.0             # dose scaling (normalized)

# Depth axis
z = np.linspace(0, 200, 1000)

# Gaussian concentration profile (normalized)
C = (Q / (np.sqrt(2*np.pi)*dRp)) * np.exp(-0.5*((z - Rp)/dRp)**2)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(z, C, linewidth=2)

# Mark Rp and +/- dRp
plt.axvline(Rp, linestyle='--', linewidth=1)
plt.axvline(Rp - dRp, linestyle=':', linewidth=1)
plt.axvline(Rp + dRp, linestyle=':', linewidth=1)

plt.xlabel("Depth, z (nm)")
plt.ylabel("Normalized concentration, C(z)")
plt.title("Gaussian (Bell-Curve) Approximation of Implantation Profile")
plt.tight_layout()

# Export as TIFF 300 dpi
plt.savefig("gaussian_profile_300dpi.tif", dpi=300, format="tiff")
plt.savefig("gaussian_profile_300dpi.png", dpi=300)  # optional: handy preview
plt.show()

print("Saved: gaussian_profile_300dpi.tif (300 dpi)")
