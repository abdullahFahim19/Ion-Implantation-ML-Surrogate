import numpy as np
import matplotlib.pyplot as plt

# Energy axis (keV, normalized scale)
E = np.linspace(1, 1000, 1000)

# Conceptual stopping curves (shape only, not absolute SRIM data)
Sn = 1.5 * np.exp(-E/200) * (E/50)          # Nuclear stopping: rises then falls
Se = 0.02 * np.log(E + 1) * (E/200)         # Electronic stopping: increases with E
S_total = Sn + Se

plt.figure(figsize=(7, 4))
plt.plot(E, Sn, label="Nuclear Stopping $S_n$")
plt.plot(E, Se, label="Electronic Stopping $S_e$")
plt.plot(E, S_total, '--', label="Total Stopping")

plt.xlabel("Ion Energy (keV)")
plt.ylabel("Stopping Power (arb. units)")
plt.title("Stopping Power vs. Ion Energy (Conceptual)")
plt.legend()
plt.tight_layout()

plt.savefig("stopping_vs_energy_300dpi.tif", dpi=300, format="tiff")
plt.savefig("stopping_vs_energy_300dpi.png", dpi=300)
plt.show()

print("Saved: stopping_vs_energy_300dpi.tif (300 dpi)")
