# --- PyYAML 6.x compatibility fix ---
import yaml
_yaml_load = yaml.load
def _load_with_full(stream):
    return _yaml_load(stream, Loader=yaml.FullLoader)
yaml.load = _load_with_full

# --- Imports ---
import os, re, time, subprocess
import pandas as pd
from srim import Ion, Layer, Target, TRIM

# --- Paths ---
COMBO_CSV_PATH = r"C:\SRIM_Auto\combo.csv"
OUTPUT_ROOT    = r"C:\SRIM_Auto\srim_batch_outputs"
SRIM_DIR       = r"C:\Users\ACER\Desktop\SRIM"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- Material parameters ---
ED_DEFAULT = 25.0
E_PARAMS = {
    "Si": {"E_d": 35.0, "lattice": 0.0, "surface": 3.0},
    "O":  {"E_d": 28.0, "lattice": 0.0, "surface": 3.0},
    "C":  {"E_d": 28.0, "lattice": 0.0, "surface": 3.0},
    "Ga": {"E_d": 25.0, "lattice": 0.0, "surface": 3.0},
    "As": {"E_d": 25.0, "lattice": 0.0, "surface": 3.0},
    "N":  {"E_d": 25.0, "lattice": 0.0, "surface": 3.0},
}
DENSITY_FALLBACK = {"Si":2.329,"SiO2":2.20,"GaAs":5.32,"SiC":3.21,"GaN":6.15}

def _norm_sub(s: str) -> str:
    s0 = (str(s) if s is not None else "").strip()
    k = s0.lower().replace(" ", "")
    if k in ["si","silicon"]: return "Si"
    if k in ["sio2","silica","quartz","glass"]: return "SiO2"
    if k in ["gaas","galliumarsenide","gallium arsenide"]: return "GaAs"
    if k in ["sic","siliconcarbide","silicon carbide"]: return "SiC"
    if k in ["gan","galliumnitride","gallium nitride"]: return "GaN"
    return s0

def _norm_ion(x: str) -> str:
    x = (x or "").strip()
    return x[:1].upper() + x[1:].lower()

def parse_formula(formula: str):
    tok = (formula or "").strip()
    if not tok: return {}
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', tok)
    comp = {}
    for sym, num in parts:
        cnt = int(num) if num else 1
        comp[sym] = comp.get(sym, 0) + cnt
    return comp

def make_elements(substrate):
    sub = _norm_sub(substrate)
    if sub == "Si":
        return {"Si": {"stoich": 1.0, **E_PARAMS["Si"]}}
    if sub == "SiO2":
        return {"Si": {"stoich": 1.0/3.0, **E_PARAMS["Si"]}, "O": {"stoich": 2.0/3.0, **E_PARAMS["O"]}}
    if sub == "GaAs":
        return {"Ga": {"stoich": 0.5, **E_PARAMS["Ga"]}, "As": {"stoich": 0.5, **E_PARAMS["As"]}}
    if sub == "SiC":
        return {"Si": {"stoich": 0.5, **E_PARAMS["Si"]}, "C": {"stoich": 0.5, **E_PARAMS["C"]}}
    if sub == "GaN":
        return {"Ga": {"stoich": 0.5, **E_PARAMS["Ga"]}, "N": {"stoich": 0.5, **E_PARAMS["N"]}}
    comp = parse_formula(sub)
    if comp:
        total = float(sum(comp.values()))
        out = {}
        for sym, cnt in comp.items():
            params = E_PARAMS.get(sym, {"E_d": ED_DEFAULT, "lattice": 0.0, "surface": 3.0})
            out[sym] = {"stoich": cnt/total, **params}
        return out
    sym = sub[:1].upper() + sub[1:].lower()
    params = E_PARAMS.get(sym, {"E_d": ED_DEFAULT, "lattice": 0.0, "surface": 3.0})
    return {sym: {"stoich": 1.0, **params}}

def _get_density(row, key_density):
    if key_density in row and pd.notna(row[key_density]):
        return float(row[key_density])
    sub = _norm_sub(row.get("substrate") or row.get("layer1_substrate"))
    return DENSITY_FALLBACK.get(sub, 3.0)

def _get_thickness(row, key_thick):
    if key_thick in row and pd.notna(row[key_thick]):
        return float(row[key_thick])
    return float(row.get("thickness_A", 1000.0))

def build_layers(row):
    layers = []
    sub1 = row["layer1_substrate"] if "layer1_substrate" in row and pd.notna(row["layer1_substrate"]) else row.get("substrate","Si")
    den1 = _get_density(row, "layer1_density_gcc") if "layer1_density_gcc" in row else _get_density(row, "density_gcc")
    w1   = _get_thickness(row, "layer1_thickness_A")
    el1  = make_elements(sub1)
    layers.append(Layer(el1, density=float(den1), width=float(w1)))
    if "layer2_substrate" in row and pd.notna(row["layer2_substrate"]) and str(row["layer2_substrate"]).strip():
        sub2 = row["layer2_substrate"]
        den2 = float(row["layer2_density_gcc"]) if "layer2_density_gcc" in row and pd.notna(row["layer2_density_gcc"]) else DENSITY_FALLBACK.get(_norm_sub(sub2), 3.0)
        w2   = float(row["layer2_thickness_A"]) if "layer2_thickness_A" in row and pd.notna(row["layer2_thickness_A"]) else 0.0
        if den2 > 0.0 and w2 > 0.0:
            el2 = make_elements(sub2)
            layers.append(Layer(el2, density=den2, width=w2))
    return layers

# --- Background launcher ---
CREATE_NO_WINDOW         = 0x08000000
DETACHED_PROCESS         = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

def launch_trim_no_focus(srim_dir: str, exe_name="TRIM.exe", wait=True, poll=0.5):
    exe = os.path.join(srim_dir, exe_name)
    if not os.path.exists(exe):
        raise FileNotFoundError(f"{exe_name} not found in {srim_dir}")
    proc = subprocess.Popen(
        [exe],
        cwd=srim_dir,
        creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    )
    if wait:
        while proc.poll() is None:
            time.sleep(poll)
    return proc

# --- Batch run ---
print(f"Reading input from: {COMBO_CSV_PATH}")
df = pd.read_csv(COMBO_CSV_PATH)

if "execution_time_seconds" not in df.columns:
    df["execution_time_seconds"] = ""

for i, row in df.iterrows():
    try:
        # Check if already run (optional)
        if pd.notna(row.get("execution_time_seconds")) and row["execution_time_seconds"] != "":
            continue 

        ion_name = _norm_ion(str(row["ion"]))
        energy_eV = float(row["energy_keV"]) * 1e3
        ion = Ion(ion_name, energy=energy_eV)
        layers = build_layers(row)
        target = Target(layers)
        angle = float(row["angle_deg"])
        nions = int(row.get("num_ions", row.get("total_ions", 10000)))

        trim = TRIM(target, ion, calculation=1, number_ions=nions, angle_ions=angle)

        laydesc = f"{len(layers)}L"
        run_name = f"run_{i+1}_{ion_name}_{laydesc}_{int(row['energy_keV'])}keV_{int(round(angle))}deg"
        out_dir = os.path.join(OUTPUT_ROOT, run_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Starting {run_name} | layers={len(layers)} | angle={angle} deg | ions={nions}")
        
        # --- TIMER START ---
        start_time = time.time()

        # --- KEY FIX: Monkey Patching subprocess ---
        # আমরা subprocess.check_call কে সাময়িকভাবে অকেজো করে দিচ্ছি
        # যাতে trim.run() ফাইল রাইট করে কিন্তু TRIM.exe রান করতে না পারে।
        original_check_call = subprocess.check_call
        subprocess.check_call = lambda *args, **kwargs: None 
        
        try:
            # এটি এখন শুধু TRIM.IN ফাইল তৈরি করবে, সফটওয়্যার রান হবে না
            trim.run(SRIM_DIR) 
        finally:
            # আবার আগের অবস্থায় ফিরিয়ে আনা হলো
            subprocess.check_call = original_check_call

        # --- NOW WE LAUNCH IT MANUALLY (Just Once) ---
        launch_trim_no_focus(SRIM_DIR, wait=True)

        # 3) Collect results
        try:
            results = trim.results(SRIM_DIR)
        except Exception:
            results = None

        # Copy raw outputs
        try:
            TRIM.copy_output_files(SRIM_DIR, out_dir)
        except Exception as e:
            print(f"Warning: raw file copy failed: {e}")

        # Save parsed tables
        try:
            if results is not None:
                if hasattr(results, "range") and results.range is not None:
                    results.range.to_csv(os.path.join(out_dir, "RANGE_parsed.csv"), index=False)
                if hasattr(results, "ioniz") and results.ioniz is not None:
                    results.ioniz.to_csv(os.path.join(out_dir, "IONIZ_parsed.csv"), index=False)
                if hasattr(results, "vacancy") and results.vacancy is not None:
                    results.vacancy.to_csv(os.path.join(out_dir, "VACANCY_parsed.csv"), index=False)
        except Exception as e:
            print(f"Warning: parsed save failed: {e}")

        # --- TIMER END ---
        end_time = time.time()
        elapsed = end_time - start_time
        
        df.at[i, "execution_time_seconds"] = round(elapsed, 2)
        
        try:
            df.to_csv(COMBO_CSV_PATH, index=False)
            print(f"Finished {run_name} in {elapsed:.2f}s -> CSV Updated")
        except PermissionError:
            print(f"Finished {run_name} in {elapsed:.2f}s -> WARNING: CSV Locked")

    except Exception as ex:
        print(f"ERROR in row {i+1}: {ex}")