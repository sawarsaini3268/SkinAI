
from pathlib import Path
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# path stuff
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "skin_ingredients_master_with_diffusion.csv"
OUTDIR = PROJECT_ROOT / "runs" / "phase2"
OUTDIR.mkdir(parents=True, exist_ok=True)

# --- Simulation settings ---
SKIN_THICKNESS_M = 15e-6        # meters (15 µm thick stratum corneum)
T_C = 32.0                      # skin surface temperature in °C
ETA_WATER_25C = 0.00089         # viscosity of water at 25°C in Pa·s (0.89 mPa·s)
CONC_VEHICLE = 1.0              # normalized concentration at vehicle-skin interface
SIM_HOURS = 8.0                 # simulate for 8 hours
DX = 0.5e-6                     # spatial step in meters (0.5 µm)
k_B = 1.380649e-23               # Boltzmann constant in J/K
# ----------------------------

def to_kelvin(T_c):
    """Convert Celsius to Kelvin."""
    return T_c + 273.15

def parse_sci_or_na(s):
    """Turn strings like '1.23e-09' into floats, or return NaN if not possible."""
    try:
        return float(s)
    except:
        return np.nan

def is_modeling_grade(row):
    """Flag 'yes' for single molecules with clear data, 'no' for plant extracts, oils, polymers."""
    bad_tokens = ["mixture", "extract", "polymer", "kda", "oil", "juice", "leaf", "root", "flower", "bark", "seed"]
    combo = f"{row['molecular_weight_g_per_mol']}{row['INCI_name']}{row['ingredient_common_name']}".lower()
    return not any(tok in combo for tok in bad_tokens)

def load_metadata(csv_path):
    """Load CSV and ensure necessary columns exist."""
    df = pd.read_csv(csv_path)
    if "estimated_D_25C_m2_per_s" not in df.columns:
        raise ValueError("CSV missing 'estimated_D_25C_m2_per_s'. Use the prepared file.")
    if "estimated_hydrodynamic_radius_nm" not in df.columns:
        df["estimated_hydrodynamic_radius_nm"] = np.nan
    if "modeling_grade" not in df.columns:
        df["modeling_grade"] = df.apply(is_modeling_grade, axis=1)
    return df

def adjust_D_for_temp_and_viscosity(D25_str, T_c=T_C, eta_vehicle_cP=1.0):
    """Adjust D from 25°C in water to skin temp & different vehicle viscosity using Stokes–Einstein scaling."""
    D25 = parse_sci_or_na(D25_str)
    if np.isnan(D25):
        return np.nan
    T_ratio = to_kelvin(T_c) / to_kelvin(25.0)
    eta_ref = ETA_WATER_25C
    eta_vehicle = max(eta_vehicle_cP * 1e-3, 1e-6)  # convert cP to Pa·s
    return D25 * T_ratio * (eta_ref / eta_vehicle)

def simulate_1d_diffusion(D, L, dx, hours, c0=1.0):
    """
    Explicit finite-difference solution for 1D diffusion in a slab of thickness L.
    x=0: constant concentration c0 (vehicle contact)
    x=L: zero-flux (reflective)
    """
    if np.isnan(D) or D <= 0:
        raise ValueError("Need positive D for simulation.")

    nx = int(round(L / dx)) + 1
    x = np.linspace(0, L, nx)
    dt_max = dx**2 / (2*D)   # stability criterion for explicit method
    dt = 0.5 * dt_max
    steps = int((hours*3600) / dt) + 1
    t = np.linspace(0, steps*dt, steps)

    c = np.zeros(nx)
    c[0] = c0  # boundary stays at c0
    C = [c.copy()]
    for _ in range(steps-1):
        c_new = c.copy()
        for i in range(1, nx-1):
            c_new[i] = c[i] + D*dt/dx**2 * (c[i+1] - 2*c[i] + c[i-1])
        c_new[0] = c0
        c_new[-1] = c_new[-2]
        c = c_new
        C.append(c.copy())
    C = np.array(C)

    frac = C.mean(axis=1) / c0  # average conc in slab normalized
    return t, x, C, frac

def run_demo(ingredients, vehicle_visc_cP=100.0):
    df = load_metadata(DATA_PATH)
    use = df[df["ingredient_common_name"].isin(ingredients)].copy()
    if use.empty:
        raise RuntimeError("No matching ingredients found.")
    results = []
    plt.figure()
    for _, row in use.iterrows():
        D25 = row["estimated_D_25C_m2_per_s"]
        D = adjust_D_for_temp_and_viscosity(D25, T_c=T_C, eta_vehicle_cP=vehicle_visc_cP)
        name = row["ingredient_common_name"]
        if np.isnan(D):
            print(f"[skip] {name}: no valid D")
            continue
        t, x, C, frac = simulate_1d_diffusion(D, SKIN_THICKNESS_M, DX, SIM_HOURS, c0=CONC_VEHICLE)
        plt.plot(t/3600.0, frac, label=name)
        results.append({
            "ingredient": name,
            "D_25C_water_m2_s": parse_sci_or_na(D25),
            "D_adj_m2_s": D,
            "vehicle_cP": vehicle_visc_cP,
            "skin_thickness_um": SKIN_THICKNESS_M*1e6,
            "penetration_metric_8h": float(frac[-1])
        })
    if results:
        plt.xlabel("Time (hours)")
        plt.ylabel("Normalized penetration metric")
        plt.title(f"Predicted Delivery (~{vehicle_visc_cP} cP vehicle, {T_C}°C skin)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / f"penetration_vehicle{int(vehicle_visc_cP)}cP.png", dpi=160)
        pd.DataFrame(results).to_csv(OUTDIR / f"penetration_vehicle{int(vehicle_visc_cP)}cP.csv", index=False)
        print(f"Saved results in {OUTDIR}")
    else:
        print("No valid results.")

if __name__ == "__main__":
    demo_ingredients = [
        "Niacinamide",
        "Azelaic Acid",
        "Sodium Ascorbyl Phosphate (SAP)"
    ]
    run_demo(demo_ingredients, vehicle_visc_cP=100.0)
# This script runs a demo simulation for selected ingredients with a specified vehicle viscosit