# Dual-Stage Hybrid Random Forest Framework for Ion Implantation ⚛️

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-green.svg)]()

A physics-informed machine learning surrogate model designed to accelerate ion implantation simulations in semiconductor manufacturing. This framework replaces computationally expensive Monte Carlo simulations (SRIM/TRIM) with a rapid, accurate Random Forest approach, enabling **real-time forward prediction** and **inverse process design**.

---

## 🚀 Key Features

*   **⚡ Massive Speedup:** Achieves **~5,200x faster inference** (0.45s vs 40 mins) compared to traditional SRIM simulations.
*   **🔄 Inverse Design Engine:** Capable of predicting required implantation energy and angle from a target depth ($R_p$), solving the non-unique inverse problem.
*   **🧪 Multi-Material Support:** Validated on **Silicon (Si)**, **Silicon Carbide (4H-SiC)**, and **Gallium Arsenide (GaAs)** substrates.
*   **🎯 High Accuracy:** Maintains **$R^2 > 0.98$** for projected range and backscattering predictions compared to ground-truth physics.
*   **🛠️ Automated Pipeline:** Includes tools for batch SRIM automation, data parsing, and physics-based feature extraction.

---

📊 Methodology
The framework utilizes a Dual-Stage Architecture:
1. Gatekeeper Classifier: Determines if ions are stopped within the substrate or transmitted (using Random Forest Classification).
2. Physics-Informed Regressor: If stopped, a Multi-Output Random Forest Regressor predicts 9 physical quantities:
    ◦ Projected Range (R 
p
​
 ) & Straggles (ΔR 
p
​
 )
    ◦ Lateral & Radial Distribution
    ◦ Vacancy Production & Backscattering
🔧 Installation & Usage
1. Clone the repository:
2. Install dependencies:
3. Run the Forward Model (Interactive Mode):
4. Run the Inverse Solver:
⚠️ Note on Data
This work is based on a dataset of 5,600+ SRIM simulations. To respect ongoing publication processes, only a sample_data.csv (first 30 rows) is provided here to demonstrate code functionality.
📜 Citation & Context
This project was developed as part of an undergraduate thesis at Jashore University of Science and Technology (JUST).
Author: Abdullah Shadek Fahim (fahim.just.19@gmail.com)
