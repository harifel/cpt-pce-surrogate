"""Utility functions for sensitivity analysis."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def set_graphics_defaults() -> None:
    """Set default graphics parameters for matplotlib."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "axes.grid": True,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.usetex": False,
            "legend.fontsize": 10,
            "figure.figsize": (10, 6),
        }
    )


def nrmsd(y: np.ndarray, yhat: np.ndarray, norm: str = "range", axis: Optional[int] = None) -> np.ndarray:
    """Compute Normalized Root Mean Squared Deviation."""
    if axis is None:
        axis = next((idx for idx, size in enumerate(y.shape) if size > 1), 0)
    err = y - yhat
    rmse = np.sqrt(np.mean(err**2, axis=axis))
    if norm == "range":
        denom = np.max(y, axis=axis) - np.min(y, axis=axis)
    elif norm == "std":
        denom = np.std(y, axis=axis, ddof=1)
    elif norm == "mean":
        denom = np.abs(np.mean(y, axis=axis))
    elif norm == "rms":
        denom = np.sqrt(np.mean(y**2, axis=axis))
    else:
        raise ValueError(f"Unknown norm mode: {norm}")
    denom = np.where(denom == 0, np.nan, denom)
    return rmse / denom


def sobol_from_pce(pce: Dict, m: int) -> Dict[str, np.ndarray]:
    """Calculate Sobol indices from PCE coefficients.
    
    Returns:
    - first: First-order Sobol indices (S_i)
    - total: Total Sobol indices (S_Ti)
    - second: Second-order Sobol indices (S_ij) as a symmetric matrix
    """
    coeffs = np.asarray(pce["Coefficients"]).flatten()
    alpha = np.asarray(pce["Basis"]["Indices"])
    
    # Use the provided m parameter (number of input variables in experimental design)
    # The PCE basis indices should match, but if there's a mismatch, pad with zeros
    alpha_dim = alpha.shape[1]
    if alpha_dim != m:
        # If basis has fewer dimensions, pad with zeros (parameter has no effect)
        if alpha_dim < m:
            padding = np.zeros((alpha.shape[0], m - alpha_dim), dtype=alpha.dtype)
            alpha = np.hstack([alpha, padding])
        # If basis has more dimensions, truncate (shouldn't happen, but handle it)
        elif alpha_dim > m:
            alpha = alpha[:, :m]
    
    # Total variance: sum of squared coefficients excluding the mean (first coefficient)
    total_variance = np.sum(coeffs[1:] ** 2)
    
    if total_variance < 1e-12:
        return {
            "first": np.zeros(m), 
            "total": np.zeros(m), 
            "second": np.zeros((m, m))
        }
    
    first = np.zeros(m)
    total = np.zeros(m)
    second = np.zeros((m, m))  # Symmetric matrix for second-order indices
    
    for dim in range(m):
        others = np.delete(np.arange(m), dim)
        # First-order: only terms where ONLY this dimension is non-zero
        mask_first = (alpha[:, dim] > 0) & np.all(alpha[:, others] == 0, axis=1)
        # Total: all terms where this dimension is non-zero (includes interactions)
        mask_total = alpha[:, dim] > 0
        
        first[dim] = np.sum(coeffs[mask_first] ** 2) / total_variance
        total[dim] = np.sum(coeffs[mask_total] ** 2) / total_variance
    
    # Calculate second-order indices (interactions between pairs)
    for i in range(m):
        for j in range(i + 1, m):
            # Second-order: terms where ONLY dimensions i and j are non-zero
            mask_ij = (alpha[:, i] > 0) & (alpha[:, j] > 0)
            # Check that all other dimensions are zero
            other_dims = np.delete(np.arange(m), [i, j])
            if len(other_dims) > 0:
                mask_ij = mask_ij & np.all(alpha[:, other_dims] == 0, axis=1)
            
            s_ij = np.sum(coeffs[mask_ij] ** 2) / total_variance
            second[i, j] = s_ij
            second[j, i] = s_ij  # Symmetric matrix
    
    return {"first": first, "total": total, "second": second}


def compute_loo_error(ed: np.ndarray, y: np.ndarray, pce_model: Dict, uq_cli, use_hat_matrix: bool = True) -> Tuple[float, float]:
    """Compute Leave-One-Out (LOO) cross-validation error for a PCE model."""
    model = pce_model
    if "PCE" in model:
        pce = model["PCE"]
    else:
        pce = model
    
    # Check for LOO error in model structure
    loo_error = None
    if isinstance(model, dict) and "Error" in model:
        error_obj = model["Error"]
        if isinstance(error_obj, dict) and "LOO" in error_obj:
            loo_error = error_obj["LOO"]
        elif isinstance(error_obj, (list, tuple)) and len(error_obj) > 0:
            if isinstance(error_obj[0], dict) and "LOO" in error_obj[0]:
                loo_error = error_obj[0]["LOO"]
    
    if loo_error is None and "Error" in pce:
        error_obj = pce["Error"]
        if isinstance(error_obj, dict) and "LOO" in error_obj:
            loo_error = error_obj["LOO"]
    
    if loo_error is None and "LOO" in pce:
        loo_error = pce["LOO"]
    
    # Convert to RMSE if found
    if loo_error is not None:
        if isinstance(loo_error, (int, float)):
            loo_mse = float(loo_error)
            return np.sqrt(loo_mse), loo_mse
        elif isinstance(loo_error, dict):
            if "RMSE" in loo_error:
                return float(loo_error["RMSE"]), float(loo_error.get("MSE", loo_error["RMSE"]**2))
            elif "MSE" in loo_error:
                mse = float(loo_error["MSE"])
                return np.sqrt(mse), mse
        elif isinstance(loo_error, (list, tuple, np.ndarray)):
            loo_mse = np.mean(np.asarray(loo_error))
            return np.sqrt(loo_mse), loo_mse
    
    # Compute manually using hat matrix approximation
    n_samples = len(y)
    if "Basis" in pce and "Degree" in pce["Basis"]:
        degree = pce["Basis"]["Degree"]
    elif "Degree" in pce:
        degree = pce["Degree"]
    else:
        raise ValueError("Cannot determine PCE degree from model")
    
    if use_hat_matrix:
        try:
            y_pred = uq_cli.evalModel(pce_model, ed.tolist()).flatten()
            residuals = y - y_pred
            n_coeffs = len(pce["Coefficients"]) if "Coefficients" in pce else 120
            avg_leverage = n_coeffs / n_samples
            loo_residuals = residuals / (1 - avg_leverage)
            loo_mse = np.mean(loo_residuals ** 2)
            return np.sqrt(loo_mse), loo_mse
        except Exception:
            pass
    
    return np.nan, np.nan


def build_input_structure(include_z_sigma: bool = True, exclude_ocr: bool = False,
                          exclude_gamma: bool = False,
                          include_z: Optional[bool] = None, include_sigma_v: Optional[bool] = None,
                          param_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Dict]:
    """Build input structure for sensitivity analysis.
    
    Parameters:
    -----------
    include_z_sigma : bool
        If True, include z (depth) and σ'_v (effective vertical stress) as parameters.
        If False, only include soil parameters: lambda, kappa, friction_angle, unit_weight, OCR.
        (Used as default if include_z or include_sigma_v are not specified)
    exclude_ocr : bool
        If True, exclude OCR from the input structure (e.g., when filtering for OCR=1, it's constant)
    exclude_gamma : bool
        If True, exclude gamma (unit_weight) from the input structure
    include_z : Optional[bool]
        If specified, overrides include_z_sigma for z parameter specifically
    include_sigma_v : Optional[bool]
        If specified, overrides include_z_sigma for sigma_v parameter specifically
    param_ranges : Optional[Dict[str, Tuple[float, float]]]
        Dictionary with parameter ranges (min, max) for each parameter.
        Keys: 'lambda', 'kappa', 'friction_angle', 'unit_weight', 'ocr', 'z', 'sigma_v'
        If None, uses default hardcoded values.
    
    Returns:
    --------
    List[Dict]
        Input structure for UQPyLab
    """
    # Default ranges (used if param_ranges is None)
    default_ranges = {
        "lambda": (0.108600, 0.1733),
        "kappa": (0.011900, 0.0338),
        "friction_angle": (15.0, 25.0),
        "unit_weight": (14.5, 16.0),
        "ocr": (1.0, 1.0),
        "z": (5.102837, 19.897214),
        "sigma_v": (24.513505, 116.429203),
    }
    
    # Use provided ranges or defaults
    ranges = param_ranges if param_ranges is not None else default_ranges
    
    # Determine which parameters to include
    use_z = include_z if include_z is not None else include_z_sigma
    use_sigma_v = include_sigma_v if include_sigma_v is not None else include_z_sigma
    
    base_params = [
        {"Name": "Lambda_Star", "Type": "Uniform", "Parameters": list(ranges.get("lambda", default_ranges["lambda"]))},
        {"Name": "Kappa_Star", "Type": "Uniform", "Parameters": list(ranges.get("kappa", default_ranges["kappa"]))},
        {"Name": "Friction_Angle", "Type": "Uniform", "Parameters": list(ranges.get("friction_angle", default_ranges["friction_angle"]))},
    ]
    
    # Only include gamma (unit_weight) if not excluded
    if not exclude_gamma:
        base_params.append({"Name": "γ [kN/m3]", "Type": "Uniform", "Parameters": list(ranges.get("unit_weight", default_ranges["unit_weight"]))})
    
    # Only include OCR if not excluded (e.g., when it's constant after filtering)
    if not exclude_ocr:
        ocr_range = ranges.get("ocr", default_ranges["ocr"])
        base_params.append({"Name": "OCR", "Type": "Uniform", "Parameters": list(ocr_range)})
    
    # Add z and/or sigma_v based on flags
    if use_z:
        z_range = ranges.get("z", default_ranges["z"])
        base_params.append({"Name": "z [m]", "Type": "Uniform", "Parameters": list(z_range)})
    
    if use_sigma_v:
        sigma_v_range = ranges.get("sigma_v", default_ranges["sigma_v"])
        base_params.append({"Name": "σ'_v [kPa]", "Type": "Uniform", "Parameters": list(sigma_v_range)})
    
    return base_params



def build_input_structure(include_z_sigma: bool = True, exclude_ocr: bool = False,
                          exclude_gamma: bool = False,
                          include_z: Optional[bool] = None, include_sigma_v: Optional[bool] = None,
                          param_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Dict]:
    """Build input structure for sensitivity analysis.
    
    Parameters:
    -----------
    include_z_sigma : bool
        If True, include z (depth) and σ'_v (effective vertical stress) as parameters.
        If False, only include soil parameters: lambda, kappa, friction_angle, unit_weight, OCR.
        (Used as default if include_z or include_sigma_v are not specified)
    exclude_ocr : bool
        If True, exclude OCR from the input structure (e.g., when filtering for OCR=1, it's constant)
    exclude_gamma : bool
        If True, exclude gamma (unit_weight) from the input structure
    include_z : Optional[bool]
        If specified, overrides include_z_sigma for z parameter specifically
    include_sigma_v : Optional[bool]
        If specified, overrides include_z_sigma for sigma_v parameter specifically
    param_ranges : Optional[Dict[str, Tuple[float, float]]]
        Dictionary with parameter ranges (min, max) for each parameter.
        Keys: 'lambda', 'kappa', 'friction_angle', 'unit_weight', 'ocr', 'z', 'sigma_v'
        If None, uses default hardcoded values.
    
    Returns:
    --------
    List[Dict]
        Input structure for UQPyLab
    """
    # Default ranges (used if param_ranges is None)
    default_ranges = {
        "lambda": (0.108600, 0.1733),
        "kappa": (0.011900, 0.0338),
        "friction_angle": (15.0, 25.0),
        "unit_weight": (14.5, 16.0),
        "ocr": (1.0, 1.0),
    }
    
    # Use provided ranges or defaults
    ranges = param_ranges if param_ranges is not None else default_ranges
    
    
    base_params = [
        {"Name": "Lambda_Star", "Type": "Uniform", "Parameters": list(ranges.get("lambda", default_ranges["lambda"]))},
        {"Name": "Kappa_Star", "Type": "Uniform", "Parameters": list(ranges.get("kappa", default_ranges["kappa"]))},
        {"Name": "Friction_Angle", "Type": "Uniform", "Parameters": list(ranges.get("friction_angle", default_ranges["friction_angle"]))},
        {"Name": "γ [kN/m3]", "Type": "Uniform", "Parameters": list(ranges.get("unit_weight", default_ranges["unit_weight"]))},
        {"Name": "OCE", "Type": "Uniform", "Parameters": list(ranges.get("ocr", default_ranges["sigma_v"]))},
    ]
    
    # Only include gamma (unit_weight) if not excluded
    if not exclude_gamma:
        base_params.append({"Name": "γ [kN/m3]", "Type": "Uniform", "Parameters": list(ranges.get("unit_weight", default_ranges["unit_weight"]))})
    
    # Only include OCR if not excluded (e.g., when it's constant after filtering)
    if not exclude_ocr:
        ocr_range = ranges.get("ocr", default_ranges["ocr"])
        base_params.append({"Name": "OCR", "Type": "Uniform", "Parameters": list(ocr_range)})
    
    
    return base_params
