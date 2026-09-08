"""Data loading functions for sensitivity analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def prepare_datasets(
    data_dir: Path,
    file_name: str,
    sheet_name: str = None,
    include_normalized: bool = False,
    include_RF: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str], Dict[str, Tuple[float, float]]]:
    """Load data from GPFEM database and prepare for sensitivity analysis.
    
    Parameters:
    -----------
    data_dir : Path
        Directory containing the database file
    file_name : str
        Name of the Excel or CSV file
    sheet_name : str, optional
        Name of the sheet to read (only for Excel files). If None and file is CSV, ignored.
    
    Returns:
    --------
    ed : np.ndarray
        Experimental design matrix (n_samples, n_params)
    out : Dict[str, np.ndarray]
        Dictionary with keys: 'qt', 'fs', 'u2' (each is n_samples,)
    var_names : List[str]
        Parameter names
    param_ranges : Dict[str, Tuple[float, float]]
        Dictionary with parameter ranges (min, max) for each parameter
    """
    file_path = data_dir / file_name
    
    # Check if file is CSV or Excel
    if file_name.lower().endswith('.csv'):
        # Load CSV file
        df = pd.read_csv(file_path)
    else:
        # Load Excel file
        if sheet_name is None:
            raise ValueError("sheet_name must be provided for Excel files")
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=1)
    
    print(f"Loaded {len(df)} rows from {file_name}")
    
    # Find columns for all parameters
    lambda_col = None
    kappa_col = None
    friction_angle_col = None
    unit_weight_col = None
    ocr_col = None
    z_col = None
    qt_col = None
    fs_col = None
    u2_col = None
    initial_sigma_v_eff_col = None
    initial_water_pressure_col = None
    initial_sigma_hor_eff_col = None

    for col in df.columns:
        col_str = str(col).lower().strip()
        if lambda_col is None and ('lambda' in col_str or 'λ' in col_str):
            lambda_col = col
        if kappa_col is None and ('kappa' in col_str or 'κ' in col_str):
            kappa_col = col
        if friction_angle_col is None and ('friction' in col_str or 'phi' in col_str or "φ'" in col_str or "φ" in col_str):
            friction_angle_col = col
        if unit_weight_col is None and ('gamma' in col_str or 'γ' in col_str or 'unit_weight' in col_str):
            unit_weight_col = col
        if ocr_col is None and 'ocr' in col_str:
            ocr_col = col
        if z_col is None and (col_str == 'z [m]' or col_str == 'z' or 'depth' in col_str):
            z_col = col
        if qt_col is None and 'qt' in col_str:
            # Accept if it's exactly 'qt' (CSV) or contains 'kpa'/'avg' (Excel)
            if col_str == 'qt' or 'kpa' in col_str or 'avg' in col_str:
                qt_col = col
        if fs_col is None and 'fs' in col_str:
            # Accept if it's exactly 'fs' (CSV) or contains 'kpa'/'avg' (Excel)
            if col_str == 'fs' or 'kpa' in col_str or 'avg' in col_str:
                fs_col = col
        if u2_col is None and 'u2' in col_str:
            # Accept if it's exactly 'u2' (CSV) or contains 'kpa'/'avg' (Excel)
            if col_str == 'u2' or 'kpa' in col_str or 'avg' in col_str:
                u2_col = col
        col_str_flat = col_str.replace('\n', ' ').replace('_', ' ')
        if initial_sigma_v_eff_col is None and 'initial' in col_str_flat and 'sigma' in col_str_flat and ('v' in col_str_flat or 'eff' in col_str_flat):
            initial_sigma_v_eff_col = col
        if initial_water_pressure_col is None and 'initial' in col_str_flat and ('water' in col_str_flat or 'pressure' in col_str_flat):
            initial_water_pressure_col = col
        # Try to detect initial horizontal effective stress (Initial_Sigma_H_Eff)
        if initial_sigma_hor_eff_col is None and 'initial' in col_str_flat and 'sigma' in col_str_flat and ('h' in col_str_flat or 'hor' in col_str_flat):
            initial_sigma_hor_eff_col = col

    if initial_sigma_v_eff_col is None and 'Initial_Sigma_V_Eff' in df.columns:
        initial_sigma_v_eff_col = 'Initial_Sigma_V_Eff'
    if initial_water_pressure_col is None:
        for c in ['Initial_Water_Pressure', 'Initial Water Pressure']:
            if c in df.columns:
                initial_water_pressure_col = c
                break
    if initial_sigma_hor_eff_col is None and 'Initial_Sigma_H_Eff' in df.columns:
        initial_sigma_hor_eff_col = 'Initial_Sigma_H_Eff'

    # Extract all parameters from database
    if lambda_col is None:
        raise ValueError("Lambda column not found in database")
    lambda_vals = df[lambda_col].values.astype(float)
    
    if kappa_col is None:
        raise ValueError("Kappa column not found in database")
    kappa_vals = df[kappa_col].values.astype(float)
    
    if friction_angle_col is None:
        raise ValueError("Friction angle column not found in database")
    friction_angle = df[friction_angle_col].values.astype(float)
    
    if unit_weight_col is None:
        raise ValueError("Unit weight (gamma) column not found in database")
    unit_weight = df[unit_weight_col].values.astype(float)
    
    if ocr_col is None:
        raise ValueError("OCR column not found in database")
    OCR = df[ocr_col].values.astype(float)
    
    # Extract outputs
    if qt_col is None:
        raise ValueError("qt column not found in database")
    qt = df[qt_col].values.astype(float)
    
    if fs_col is None:
        raise ValueError("fs column not found in database")
    fs = df[fs_col].values.astype(float)
    
    if u2_col is None:
        raise ValueError("u2 column not found in database")
    u2 = df[u2_col].values.astype(float)



    qt_norm = df["Qt_normalized"].values.astype(float)
    fs_norm = df["fs_normalized"].values.astype(float)
    u2_norm = df["U_normalized"].values.astype(float)
    
    # Extract z and compute sigma_v
    if z_col is None:
        raise ValueError("z (depth) column not found in database")
    z_values = df[z_col].values.astype(float)
    sigma_v_values = (unit_weight - 10) * z_values  # gamma_water = 10 kN/m³

    if initial_sigma_v_eff_col is not None and initial_water_pressure_col is not None:
        initial_sigma_v_eff = df[initial_sigma_v_eff_col].values.astype(float)
        initial_water_pressure = df[initial_water_pressure_col].values.astype(float)
        sigma_v_total = initial_sigma_v_eff + initial_water_pressure  # total stress = sigma'_v + u
    else:
        initial_sigma_v_eff = None
        initial_water_pressure = None
        sigma_v_total = None

    # Optional: initial horizontal effective stress (if available)
    if initial_sigma_hor_eff_col is not None:
        initial_sigma_h_eff = df[initial_sigma_hor_eff_col].values.astype(float)
    else:
        initial_sigma_h_eff = None

    # Build experimental design matrix and var_names with all input variables
    # Core: Lambda, Kappa, Friction_Angle, γ, OCR, z, σ'_v
    ed = np.column_stack([
        lambda_vals,
        kappa_vals,
        friction_angle,
        unit_weight,
        OCR,
        z_values,
        sigma_v_values,
    ])
    var_names = ["Lambda_Star", "Kappa_Star", "Friction_Angle", "γ [kN/m3]", "OCR", "z [m]", "σ'_v [kPa]"]

    # Append initial stresses and water pressure to ed and var_names when available
    if initial_sigma_v_eff is not None and initial_water_pressure is not None:
        ed = np.column_stack([ed, initial_sigma_v_eff, initial_water_pressure])
        var_names = var_names + ["Initial_Sigma_V_Eff", "Initial_Water_Pressure"]
    if initial_sigma_h_eff is not None:
        ed = np.column_stack([ed, initial_sigma_h_eff])
        var_names = var_names + ["Initial_Sigma_H_Eff"]
    
    if include_normalized == True:
        out = {
            "qt": qt,
            "fs": fs,
            "u2": u2,
            "qt_norm": qt_norm,
            "fs_norm": fs_norm,
            "u2_norm": u2_norm,
        }
    elif include_RF == True:
        out = {
            "qt": qt,
            "fs": fs,
            "Rf": fs / qt * 100,
            "u2": u2,
        }
    else:
        out = {
            "qt": qt,
            "fs": fs,
            "u2": u2,
        }

    # Attach initial stresses and water pressure to outputs if available
    if initial_sigma_v_eff is not None and initial_water_pressure is not None and sigma_v_total is not None:
        out["Initial_Sigma_V_Eff"] = initial_sigma_v_eff
        out["Initial_Water_Pressure"] = initial_water_pressure
        out["Sigma_V_Total"] = sigma_v_total
    if initial_sigma_h_eff is not None:
        out["Initial_Sigma_H_Eff"] = initial_sigma_h_eff

    # Compute parameter ranges from actual data for PCE margins
    param_ranges = {
        "lambda": (float(np.min(lambda_vals)), float(np.max(lambda_vals))),
        "kappa": (float(np.min(kappa_vals)), float(np.max(kappa_vals))),
        "friction_angle": (float(np.min(friction_angle)), float(np.max(friction_angle))),
        "unit_weight": (float(np.min(unit_weight)), float(np.max(unit_weight))),
        "ocr": (float(np.min(OCR)), float(np.max(OCR))),
        "z": (float(np.min(z_values)), float(np.max(z_values))),
        "sigma_v": (float(np.min(sigma_v_values)), float(np.max(sigma_v_values))),
    }
    if initial_sigma_v_eff is not None and initial_water_pressure is not None and sigma_v_total is not None:
        param_ranges["initial_sigma_v_eff"] = (
            float(np.min(initial_sigma_v_eff)),
            float(np.max(initial_sigma_v_eff)),
        )
        param_ranges["initial_water_pressure"] = (
            float(np.min(initial_water_pressure)),
            float(np.max(initial_water_pressure)),
        )
        param_ranges["sigma_v_total"] = (
            float(np.min(sigma_v_total)),
            float(np.max(sigma_v_total)),
        )
    if initial_sigma_h_eff is not None:
        param_ranges["initial_sigma_h_eff"] = (
            float(np.min(initial_sigma_h_eff)),
            float(np.max(initial_sigma_h_eff)),
        )

    return ed, out, var_names, param_ranges


import numpy as np

def normalize_input_matrix(input_data):
    """
    Normalizes an input matrix (list or array) to the range [0, 1].
    
    Args:
        input_data: The input matrix of shape (n_samples, n_features)
        
    Returns:
        X_norm: The normalized matrix
        X_min: The minimum values used for scaling (for future inverse transform)
        X_max: The maximum values used for scaling (for future inverse transform)
    """
    # Convert to numpy array if it's a list
    X = np.array(input_data)
    
    # Calculate min and max for each column (feature)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    
    # Avoid division by zero if a column is constant
    range_val = X_max - X_min
    range_val[range_val == 0] = 1.0
    
    # Apply Min-Max scaling
    X_norm = (X - X_min) / range_val
    
    return X_norm.tolist(), X_min, X_max


def load_pce_sobol_database(
    inputs_sheet: str | int = "Summary",
    outputs_sheet: str | int = "Result_combine",
    header_search_rows: int = 30,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str], Dict[str, Tuple[float, float]]]:
    """Load the Sobol/PCE database with fixed path + fixed file, using only specific inputs.

    Database location (fixed):
    C:\\Users\\haris\\Documents\\GitHub\\Surrogate_GPFEM_CPT\\data\\02_database\\last_database

    File name (fixed):
    CLV - SCS ML Result R08 - sobol.xlsx

    Inputs returned in `ed` (in order):
    z [m], γ [kN/m3], φ' [deg], OCR, CR, cs/cc

    Outputs in `out` stay the same:
    qt, fs, u2
    """
    data_dir = Path(r"C:\Users\haris\Documents\GitHub\Surrogate_GPFEM_CPT\data\02_database\last_database")
    file_name = "CLV - SCS ML Result R08 - sobol.xlsx"
    file_path = data_dir / file_name

    def _norm_col_name(c: object) -> str:
        s = str(c).lower().strip()
        s = s.replace("\n", " ").replace("_", " ").replace("-", " ")
        s = " ".join(s.split())
        return s

    def _detect_header_and_build_df(raw: pd.DataFrame) -> pd.DataFrame:
        def _row_score(row_vals: List[object]) -> int:
            tokens = [_norm_col_name(v) for v in row_vals]
            joined = " | ".join(tokens)
            score = 0
            # Inputs
            if ("z [m]" in joined) or (" depth" in f" {joined} ") or (" z " in f" {joined} "):
                score += 2
            if ("gamma" in joined) or ("γ" in joined) or ("unit weight" in joined):
                score += 2
            if ("friction" in joined) or ("phi" in joined) or ("φ" in joined) or ("ϕ" in joined):
                score += 2
            if "ocr" in joined:
                score += 2
            if ("cs/cc" in joined) or ("cs cc" in joined) or ("cs\\cc" in joined) or ("cscc" in joined):
                score += 2
            # CR token (avoid OCR)
            if " ocr" not in f" {joined} " and (" cr" in f" {joined} " or f" {joined} ".startswith("cr ")):
                score += 1
            # Outputs
            if "qt" in joined:
                score += 2
            if "fs" in joined:
                score += 2
            if "u2" in joined:
                score += 2
            # Series id
            if "series" in joined:
                score += 1
            return score

        best_idx = None
        best_score = -1
        max_scan = min(header_search_rows, raw.shape[0])
        for i in range(max_scan):
            row_vals = raw.iloc[i].tolist()
            score = _row_score(row_vals)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            raise ValueError("Could not detect header row in sobol Excel sheet.")

        header_row = raw.iloc[best_idx].tolist()
        df = raw.iloc[best_idx + 1 :].copy()
        df.columns = header_row
        df = df.dropna(how="all").reset_index(drop=True)
        return df

    def _read_sheet(sheet: str | int) -> pd.DataFrame:
        raw = pd.read_excel(file_path, sheet_name=sheet, header=None)
        return _detect_header_and_build_df(raw)

    df_in = _read_sheet(inputs_sheet)
    df_out = _read_sheet(outputs_sheet)

    cols_in = {c: _norm_col_name(c) for c in df_in.columns}
    cols_out = {c: _norm_col_name(c) for c in df_out.columns}

    def _find_one(cols_map: Dict[object, str], pred) -> object | None:
        for c, s in cols_map.items():
            if pred(s):
                return c
        return None

    # Inputs sheet columns
    series_in_col = _find_one(cols_in, lambda s: "series" in s)
    z_col = _find_one(cols_in, lambda s: s == "z [m]" or s == "z" or "depth" in s)
    gamma_col = _find_one(cols_in, lambda s: "γ" in s or "gamma" in s or "unit weight" in s or "unit_weight" in s)
    # friction angle / phi': include common latin + greek variants
    phi_col = _find_one(
        cols_in,
        lambda s: ("friction" in s)
        or ("phi" in s)
        or ("φ" in s)
        or ("ϕ" in s)
    )
    ocr_col = _find_one(cols_in, lambda s: "ocr" in s)
    # CR should not match OCR
    cr_col = _find_one(cols_in, lambda s: (s == "cr" or s.endswith(" cr") or s.startswith("cr ")) and "ocr" not in s)
    cscc_col = _find_one(cols_in, lambda s: "cs/cc" in s or "cs cc" in s or "cs\\cc" in s or s == "cscc")

    # Outputs sheet columns
    series_out_col = _find_one(cols_out, lambda s: "series" in s)
    qt_col = _find_one(cols_out, lambda s: "qt" in s and (s == "qt" or "kpa" in s or "avg" in s))
    fs_col = _find_one(cols_out, lambda s: "fs" in s and (s == "fs" or "kpa" in s or "avg" in s))
    u2_col = _find_one(cols_out, lambda s: "u2" in s and (s == "u2" or "kpa" in s or "avg" in s))

    missing = []
    for name, col in [
        ("Series (inputs sheet)", series_in_col),
        ("z [m]", z_col),
        ("γ [kN/m3]", gamma_col),
        ("φ' [deg]", phi_col),
        ("OCR", ocr_col),
        ("CR", cr_col),
        ("cs/cc", cscc_col),
        ("Series (outputs sheet)", series_out_col),
        ("qt", qt_col),
        ("fs", fs_col),
        ("u2", u2_col),
    ]:
        if col is None:
            missing.append(name)
    if missing:
        # Show the available column names to help quick fixes without editing notebooks.
        raise ValueError(
            "Missing required columns in sobol database: "
            + ", ".join(missing)
            + f". Inputs sheet columns: {list(df_in.columns)}"
            + f". Outputs sheet columns: {list(df_out.columns)}"
        )

    # Merge inputs + outputs by series name
    df_in_key = df_in[[series_in_col, z_col, gamma_col, phi_col, ocr_col, cr_col, cscc_col]].copy()
    df_out_key = df_out[[series_out_col, qt_col, fs_col, u2_col]].copy()
    df_in_key["_series_key"] = df_in_key[series_in_col].astype(str).str.strip()
    df_out_key["_series_key"] = df_out_key[series_out_col].astype(str).str.strip()
    merged = df_in_key.merge(df_out_key.drop(columns=[series_out_col]), on="_series_key", how="inner")

    z = merged[z_col].values.astype(float)
    gamma = merged[gamma_col].values.astype(float)
    phi = merged[phi_col].values.astype(float)
    ocr = merged[ocr_col].values.astype(float)
    cr = merged[cr_col].values.astype(float)
    cscc = merged[cscc_col].values.astype(float)

    qt = merged[qt_col].values.astype(float)
    fs = merged[fs_col].values.astype(float)
    u2 = merged[u2_col].values.astype(float)

    ed = np.column_stack([z, gamma, phi, ocr, cr, cscc])
    var_names = ["z [m]", "γ [kN/m3]", "φ' [deg]", "OCR", "CR", "cs/cc"]
    out = {"qt": qt, "fs": fs, "u2": u2}

    param_ranges = {
        "z": (float(np.min(z)), float(np.max(z))),
        "unit_weight": (float(np.min(gamma)), float(np.max(gamma))),
        "friction_angle": (float(np.min(phi)), float(np.max(phi))),
        "ocr": (float(np.min(ocr)), float(np.max(ocr))),
        "cr": (float(np.min(cr)), float(np.max(cr))),
        "cs_cc": (float(np.min(cscc)), float(np.max(cscc))),
    }

    return ed, out, var_names, param_ranges


def PCE_database_total(
    sheet_name: str | int = "sobol only",
    header_search_rows: int = 30,
    cr_cscc: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str], Dict[str, Tuple[float, float]]]:
    """Load Sobol inputs + corresponding qt/fs/u2 from the 'sobol only' sheet.

    Database location (fixed):
    C:\\Users\\haris\\Documents\\GitHub\\Surrogate_GPFEM_CPT\\data\\02_database\\

    File name (fixed):
    CLV - SCS ML Result R08 - sobol.xlsx

    Inputs returned in `ed` (in order):
    z [m], γ [kN/m3], φ' [deg], OCR, CR, cs/cc

    Outputs in `out`:
    qt, fs, u2  (taken from qt_kPa, fs_kPa, u2_kPa columns)
    """
    data_dir = Path(r"C:\Users\hari1996\Documents\GitHub\cpt-pce-surrogate\data\02_database")
    file_name = "GPFEM_database_20260624.xlsx"
    file_path = data_dir / file_name

    def _norm_col_name(c: object) -> str:
        s = str(c).lower().strip()
        s = s.replace("\n", " ").replace("_", " ").replace("-", " ")
        s = " ".join(s.split())
        return s

    def _detect_header_and_build_df(raw: pd.DataFrame) -> pd.DataFrame:
        def _row_score(row_vals: List[object]) -> int:
            tokens = [_norm_col_name(v) for v in row_vals]
            joined = " | ".join(tokens)
            score = 0
            if ("z [m]" in joined) or (" depth" in f" {joined} ") or (" z " in f" {joined} "):
                score += 2
            if ("gamma" in joined) or ("γ" in joined) or ("unit weight" in joined):
                score += 2
            if ("friction" in joined) or ("phi" in joined) or ("φ" in joined) or ("ϕ" in joined):
                score += 2
            if "ocr" in joined:
                score += 2
            if ("cs/cc" in joined) or ("cs cc" in joined) or ("cs\\cc" in joined) or ("cscc" in joined):
                score += 2
            if " cr" in f" {joined} " and " ocr" not in f" {joined} ":
                score += 1
            return score

        best_idx = None
        best_score = -1
        max_scan = min(header_search_rows, raw.shape[0])
        for i in range(max_scan):
            row_vals = raw.iloc[i].tolist()
            score = _row_score(row_vals)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            raise ValueError("Could not detect header row in 'sobol only' sheet.")

        header_row = raw.iloc[best_idx].tolist()
        df = raw.iloc[best_idx + 1 :].copy()
        df.columns = header_row
        df = df.dropna(how="all").reset_index(drop=True)
        return df

    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    df_raw = _detect_header_and_build_df(raw)
    df_raw = pd.DataFrame(df_raw)  # Ensure it's a DataFrame in case of empty sheet
    
    df = df_raw.dropna(how="all").reset_index(drop=True)


    cols = {c: _norm_col_name(c) for c in df.columns}

    def _find_one(pred) -> object | None:
        for c, s in cols.items():
            if pred(s):
                return c
        return None

    z_col = _find_one(lambda s: s == "z [m]" or s == "z" or "depth" in s)
    gamma_col = _find_one(lambda s: "γ" in s or "gamma" in s or "unit weight" in s or "unit_weight" in s)
    phi_col = _find_one(
        lambda s: ("friction" in s)
        or ("phi" in s)
        or ("φ" in s)
        or ("ϕ" in s)
    )
    ocr_col = _find_one(lambda s: "ocr" in s)
    cr_col = _find_one(lambda s: (s == "cr" or s.endswith(" cr") or s.startswith("cr ")) and "ocr" not in s)
    lamda_col = _find_one(lambda s: "lambda" in s or "λ" in s)
    kappa_col = _find_one(lambda s: "kappa" in s or "κ" in s)
    cscc_col = _find_one(lambda s: "cs/cc" in s or "cs cc" in s or "cs\\cc" in s or s == "cscc")

    # Output columns on the same sheet (exactly: qt_kPa, fs_kPa, u2_kPa)
    # After normalization, their names become 'qt kpa', 'fs kpa', 'u2 kpa'
    qt_col = _find_one(lambda s: s == "qt kpa" or s == "qt" or ("qt" in s and "kpa" in s))
    fs_col = _find_one(lambda s: s == "fs kpa" or s == "fs" or ("fs" in s and "kpa" in s))
    u2_col = _find_one(lambda s: s == "u2 kpa" or s == "u2" or ("u2" in s and "kpa" in s))

    missing = []
    for name, col in [
        ("z [m]", z_col),
        ("γ [kN/m3]", gamma_col),
        ("φ' [deg]", phi_col),
        ("OCR", ocr_col),
        ("CR", cr_col),
        ("cs/cc", cscc_col),
        ("λ", lamda_col),
        ("κ", kappa_col),
        ("qt", qt_col),
        ("fs", fs_col),
        ("u2", u2_col),
    ]:
        if col is None:
            missing.append(name)
    if missing:
        raise ValueError(
            "Missing required columns in 'sobol only' sheet: "
            + ", ".join(missing)
            + f". Available columns: {list(df.columns)}"
        )

    if cr_cscc:
        required_cols = [z_col, gamma_col, phi_col, ocr_col, cr_col, cscc_col, qt_col, fs_col, u2_col]
    else:
        required_cols = [z_col, gamma_col, phi_col, ocr_col, lamda_col, kappa_col, qt_col, fs_col, u2_col]
    n_before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"Loaded {n_before} rows from '{sheet_name}' in {file_name}, dropped {n_before - len(df)} rows with NaN values, {len(df)} rows remaining.")

    z = df[z_col].values.astype(float)
    gamma = df[gamma_col].values.astype(float)
    phi = df[phi_col].values.astype(float)
    ocr = df[ocr_col].values.astype(float)
    cr = df[cr_col].values.astype(float) if cr_col is not None else None
    cscc = df[cscc_col].values.astype(float) if cscc_col is not None else None
    lamda = df[lamda_col].values.astype(float)
    kappa = df[kappa_col].values.astype(float)

    qt = df[qt_col].values.astype(float)
    fs = df[fs_col].values.astype(float)
    u2 = df[u2_col].values.astype(float)

    if cr_cscc:
        ed = np.column_stack([z, gamma, phi, ocr, cr, cscc])
        var_names = ["z [m]", "γ [kN/m3]", "φ' [deg]", "OCR", "CR", "cs/cc"]
    else:
        ed = np.column_stack([z, gamma, phi, ocr, lamda, kappa])
        var_names = ["z [m]", "γ [kN/m3]", "φ' [deg]", "OCR", "λ", "κ"]
    
    

    out: Dict[str, np.ndarray] = {
        "qt": qt,
        "fs": fs,
        "u2": u2,
    }

    if cr_cscc:
        param_ranges = {
            "z": (float(np.min(z)), float(np.max(z))),
            "unit_weight": (float(np.min(gamma)), float(np.max(gamma))),
            "friction_angle": (float(np.min(phi)), float(np.max(phi))),
            "ocr": (float(np.min(ocr)), float(np.max(ocr))),
            "cr": (float(np.min(cr)), float(np.max(cr))),
            "cs_cc": (float(np.min(cscc)), float(np.max(cscc))),
        }
    else:
        param_ranges = {
            "z": (float(np.min(z)), float(np.max(z))),
            "unit_weight": (float(np.min(gamma)), float(np.max(gamma))),
            "friction_angle": (float(np.min(phi)), float(np.max(phi))),
            "ocr": (float(np.min(ocr)), float(np.max(ocr))),
            "lambda": (float(np.min(lamda)), float(np.max(lamda))),
            "kappa": (float(np.min(kappa)), float(np.max(kappa))),
        }

    return ed, out, var_names, param_ranges
