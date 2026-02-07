# SRIM batch summarizer with thickness_A and density_gcc
from pathlib import Path
import os
import re
import csv

ROOT_DIR = Path(r"C:\SRIM_Auto\srim_batch_outputs")

def read_text(p):
    p = Path(p)
    return p.read_text(encoding='utf-8', errors='ignore')

# Normalize plain-text material token (e.g., "Silicon (Double Peak)" -> "Silicon")
def _normalize_material(tok: str) -> str:
    tok = (tok or '').strip()
    tok = re.sub(r'\s*\(.*$', '', tok).strip()
    name2sym = {
        'Silicon':'Si','Germanium':'Ge','Diamond':'C','Quartz':'SiO2','Sapphire':'Al2O3',
        'Gallium Arsenide':'GaAs','GalliumArsenide':'GaAs','Silica':'SiO2','Glass':'SiO2'
    }
    return name2sym.get(tok, tok)

def infer_substrate(txt_tdata, txt_range, fallback):
    """
    Priority:
      1) TDATA: 'Layer # 1- <El> = XX Atomic Percent' → dominant element or compact label
      2) RANGE: same logic
      3) TDATA: 'Target energies for target atom = <token>'
      4) RANGE header: '... into <stack/target>' → last segment after '/'
      5) Layer #1 name
      6) Fallback (folder name)
    """
    try:
        def from_composition(txt):
            pairs = re.findall(r'Layer\s*#\s*1-\s*([A-Z][a-z]?)\s*=\s*([0-9.]+)\s*Atomic Percent', txt or '')
            comp = []
            for e, p in pairs:
                try:
                    v = float(p)
                    if v > 0:
                        comp.append((e, v))
                except:
                    pass
            if not comp:
                return None
            comp.sort(key=lambda t: -t[1])  # sort by atomic % descending (fixed index) [was t[13] bug]
            if len(comp) == 1 and abs(comp[0][1] - 100.0) < 1e-3:
                return comp[0][0]
            parts = []
            for e, v in comp:
                parts.append(f"{e}{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{e}{v:g}")
            return "".join(parts) if parts else None

        # 1) TDATA composition
        s = from_composition(txt_tdata)
        if s:
            return s
        # 2) RANGE composition
        s = from_composition(txt_range)
        if s:
            return s
        # 3) TDATA target atom hint
        m = re.search(r'Target energies for target atom\s*=\s*([A-Za-z0-9@/ ]+)', txt_tdata or '')
        if m and m.group(1).strip():
            return _normalize_material(m.group(1).strip())
        # 4) RANGE header “… into <stack> …” → last segment
        m = re.search(r'into\s+([^\n=]+)', txt_range or '', re.I)
        if m:
            last = _normalize_material(m.group(1).strip().split('/')[-1].strip())
            if last:
                return last
        # 5) Layer #1 label
        m = re.search(r'Layer\s*#\s*1\s*-\s*([A-Za-z0-9_ @/]+)', (txt_tdata or '') + '\n' + (txt_range or ''))
        if m and m.group(1).strip():
            return _normalize_material(m.group(1).strip())
        # 6) Final fallback
        return fallback
    except Exception:
        return fallback

def _to_angstrom(val: float, unit: str) -> float:
    u = (unit or '').strip().lower()
    if u in ['a', 'ang', 'angstrom', 'angstroms']: return val
    if u in ['nm', 'nanometer', 'nanometers']:     return val * 10.0
    if u in ['um', 'μm', 'micron', 'microns', 'micrometer', 'micrometers']: return val * 1.0e4
    return val  # default assume Angstroms

def parse_layer1_physical(txt_tdata, txt_range):
    """
    Extract Layer #1 thickness (Å) and density (g/cm3) from TDATA with RANGE fallback.
    SRIM layer width default units are Angstroms; density is g/cm3. [SRIM Ch.8, pysrim docs]
    """
    th_A = None
    # Common TDATA patterns for thickness/width
    for pat in [
        r'Layer\s*#\s*1[^\\n]*?Thickness\s*=\s*([0-9.E+\-]+)\s*([A-Za-zμ]+)',
        r'Layer\s*#\s*1[^\\n]*?Width\s*=\s*([0-9.E+\-]+)\s*([A-Za-zμ]+)',
        r'Thickness\s*=\s*([0-9.E+\-]+)\s*([A-Za-zμ]+)\s*\(Layer\s*#\s*1\)',
        r'Width\s*=\s*([0-9.E+\-]+)\s*([A-Za-zμ]+)\s*\(Layer\s*#\s*1\)',
    ]:
        m = re.search(pat, txt_tdata or '', re.I)
        if m:
            try:
                th_A = _to_angstrom(float(m.group(1).replace(',', '')), m.group(2))
                break
            except:
                pass
    # RANGE fallback
    if th_A is None:
        m = re.search(r'Layer[^\\n]*?Thickness[^=]*=\s*([0-9.E+\-]+)\s*([A-Za-zμ]+)', txt_range or '', re.I)
        if not m:
            m = re.search(r'Layer[^\\n]*?Width[^=]*=\s*([0-9.E+\-]+)\s*([A-Za-zμ]+)', txt_range or '', re.I)
        if m:
            try:
                th_A = _to_angstrom(float(m.group(1).replace(',', '')), m.group(2))
            except:
                th_A = None

    # Density (g/cm3)
    dens = None
    for pat in [
        r'Density\s*=\s*([0-9.]+)\s*g/cm3',
        r'Layer\s*#\s*1[^\\n]*?Density\s*=\s*([0-9.]+)\s*g/cm3',
    ]:
        m = re.search(pat, txt_tdata or '', re.I)
        if m:
            try:
                dens = float(m.group(1))
                break
            except:
                pass
    if dens is None:
        m = re.search(r'Density\s*=\s*([0-9.]+)\s*g/cm3', txt_range or '', re.I)
        if m:
            try:
                dens = float(m.group(1))
            except:
                dens = None

    return th_A, dens

def parse_tdata(txt):
    def ffloat(p):
        m = re.search(p, txt); return float(m.group(1)) if m else None
    def fint(p):
        m = re.search(p, txt)
        if m:
            s = m.group(1).replace(',', '')
            try:
                return int(float(s))
            except:
                return None
        return None
    ion = re.search(r'Ion\s*=\s*([A-Za-z]+)', txt)
    # Prefer totals from TDATA if present
    back = fint(r'Total Backscattered Ions\s*=\s*([0-9.E+\-]+)')
    trans = fint(r'Total Transmitted Ions\s*=\s*([0-9.E+\-]+)')
    total = fint(r'Total Ions calculated\s*=\s*([0-9.E+\-]+)')
    return {
        'ion': ion.group(1) if ion else None,
        'energy_keV': ffloat(r'Energy\s*=\s*([0-9.E+\-]+)\s*keV'),
        'angle_deg': ffloat(r'Ion Angle to Surface\s*=\s*([0-9.+\-]+)\s*degrees'),
        'total_ions': total,
        'backscattered': back,
        'transmitted': trans,
    }

def parse_range_summary_and_stats(txt):
    # Summary ranges, skewness, kurtosis, plus back/trans fallback
    Rp  = re.search(r'Ion Average Range\s*=\s*([0-9.E+\-]+)\s*A', txt)
    dRp = re.search(r'Ion Average Range\s*=.*?Straggling\s*=\s*([0-9.E+\-]+)\s*A', txt, re.S)
    lat = re.search(r'Ion Lateral Range\s*=\s*([0-9.E+\-]+)\s*A\s*.*?Straggling\s*=\s*([0-9.E+\-]+)\s*A', txt, re.S)
    rad = re.search(r'Ion Radial\s*Range\s*=\s*([0-9.E+\-]+)\s*A\s*.*?Straggling\s*=\s*([0-9.E+\-]+)\s*A', txt, re.S)
    skew= re.search(r'Range Skewne-\s*=\s*([0-9.E+\-]+)', txt)
    kurt= re.search(r'Range Kurtosis\s*=\s*([0-9.E+\-]+)', txt)
    back = re.search(r'Backscattered Ions\s*=\s*([0-9.E+\-]+)', txt)
    trans= re.search(r'Transmitted Ions\s*=\s*([0-9.E+\-]+)', txt)
    def fnum(m):
        if not m: return None
        s = m.group(1).replace(',', '')
        try: return float(s)
        except: return None
    return {
        'Rp_A': fnum(Rp),
        'dRp_A': fnum(dRp),
        'lateral_range_A': float(lat.group(1)) if lat else None,
        'lateral_straggle_A': float(lat.group(2)) if lat else None,
        'radial_range_A': float(rad.group(1)) if rad else None,
        'radial_straggle_A': float(rad.group(2)) if rad else None,
        'skewness': float(skew.group(1)) if skew else None,
        'kurtosis': float(kurt.group(1)) if kurt else None,
        'back_fallback': int(fnum(back)) if fnum(back) is not None else None,
        'trans_fallback': int(fnum(trans)) if fnum(trans) is not None else None,
    }

def parse_vacancies(txt):
    m = re.search(r'Total Target Vacancies\s*=\s*([0-9.]+)\s*/Ion', txt or '')
    try:
        return float(m.group(1)) if m else None
    except:
        return None

def process_folder(sim_dir: Path):
    rng = sim_dir / 'RANGE.txt'
    tdt = sim_dir / 'TDATA.txt'
    vac = sim_dir / 'VACANCY.txt'
    if not (rng.exists() and tdt.exists()):
        return None  # must have RANGE & TDATA as per SRIM outputs
    rt = read_text(rng); td = read_text(tdt); vtxt = read_text(vac) if vac.exists() else ''
    tmeta = parse_tdata(td)
    rsum  = parse_range_summary_and_stats(rt)
    # Use TDATA totals; if missing, take from RANGE header fallback
    back = tmeta['backscattered'] if tmeta['backscattered'] is not None else rsum['back_fallback']
    trans= tmeta['transmitted']   if tmeta['transmitted']   is not None else rsum['trans_fallback']
    substrate = infer_substrate(td, rt, fallback=sim_dir.name)
    # NEW: thickness and density
    thickness_A, density_gcc = parse_layer1_physical(td, rt)

    return {
        'sim_dir': sim_dir.name,
        'substrate': substrate,
        'ion': tmeta['ion'],
        'energy_keV': tmeta['energy_keV'],
        'angle_deg': tmeta['angle_deg'],
        'thickness_A': thickness_A,
        'density_gcc': density_gcc,
        'Rp_A': rsum['Rp_A'], 'dRp_A': rsum['dRp_A'],
        'lateral_range_A': rsum['lateral_range_A'], 'lateral_straggle_A': rsum['lateral_straggle_A'],
        'radial_range_A': rsum['radial_range_A'], 'radial_straggle_A': rsum['radial_straggle_A'],
        'skewness': rsum['skewness'], 'kurtosis': rsum['kurtosis'],
        'vacancies_per_ion': parse_vacancies(vtxt),
        'total_ions': tmeta['total_ions'],
        'backscattered': back,
        'transmitted': trans,
    }

def main():
    if not ROOT_DIR.exists() or not ROOT_DIR.is_dir():
        raise FileNotFoundError(f"Folder not found: {ROOT_DIR}")
    rows = []
    # Deterministic order by name
    for p in sorted([d for d in ROOT_DIR.iterdir() if d.is_dir()], key=lambda x: x.name.lower()):
        r = process_folder(p)
        if r:
            rows.append(r)
    cols = [
        'sim_dir','substrate','ion','energy_keV','angle_deg',
        'thickness_A','density_gcc',  # NEW columns
        'Rp_A','dRp_A','lateral_range_A','lateral_straggle_A',
        'radial_range_A','radial_straggle_A','skewness','kurtosis',
        'vacancies_per_ion','total_ions','backscattered','transmitted'
    ]
    out_csv = ROOT_DIR / 'srim_batch_summary_trimmed.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in cols})
    print(f"Wrote {out_csv} with {len(rows)} rows")

if __name__ == "__main__":
    main()
