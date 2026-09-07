"""PCE (Polynomial Chaos Expansion) utility functions."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R² (coefficient of determination).
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual output values
    predicted : np.ndarray
        Predicted output values
        
    Returns:
    --------
    float
        R² score (1.0 = perfect prediction, 0.0 = no better than mean)
    """
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return float(r2)


def build_pce_models(ed: np.ndarray, out: Dict[str, np.ndarray], my_input: Dict, uq) -> tuple:
    """Build PCE models for qt, fs, u2 (scalar outputs).
    
    Tests multiple configurations and selects the best based on R² score.
    
    Parameters:
    -----------
    ed : np.ndarray
        Experimental design matrix (n_samples, n_params)
    out : Dict[str, np.ndarray]
        Dictionary with output arrays for 'qt', 'fs', 'u2'
    my_input : Dict
        UQPyLab input structure
    uq : object
        UQPyLab CLI object
    
    Returns:
    --------
    pce_models : Dict
        Dictionary with PCE models for each output
    prediction_pce : Dict
        Dictionary with predictions for each output
    """
    pce_models = {}
    prediction_pce = {}
    
    # Define configurations to test: (method, max_degree, target_accuracy)
    configurations = [
        ("LARS", 5, 1e-3),      # Original configuration
        ("LARS", 4, 1e-3),      # Lower degree
        ("LARS", 6, 1e-3),      # Higher degree
        ("OLS", 4, 1e-3),       # OLS method with moderate degree
        ("LARS", 5, 1e-4),      # Higher accuracy target
        ("LARS", 7, 1e-3),      # Higher degree
        ("LARS", 8, 1e-3),      # Higher degree
        ("LARS", 12, 1e-3),      # Higher degree
    ]
    
    for out_name in ["qt", "fs", "u2"]:
        actual = out[out_name]
        best_model = None
        best_prediction = None
        best_r2 = float('-inf')
        best_config = None
        
        print(f"\nTesting configurations for {out_name}:")
        
        for method, max_degree, target_acc in configurations:
            try:
                meta_opts = {
                    "Type": "Metamodel",
                    "Input": my_input["Name"],
                    "MetaType": "PCE",
                    "Degree": list(range(1, max_degree + 1)),
                    "Method": method,
                    "TargetAccuracy": target_acc,
                    "Display": "quiet",  # Reduce output during testing
                    "ExpDesign": {
                        "X": ed.tolist(),
                        "Y": actual.tolist()
                    }
                }
                
                my_pce = uq.createModel(meta_opts)
                prediction = uq.evalModel(my_pce, ed.tolist()).flatten()
                r2 = calculate_r2(actual, prediction)
                
                config_str = f"Method={method}, MaxDegree={max_degree}, TargetAcc={target_acc}"
                print(f"  {config_str}: R² = {r2:.6f}")
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = my_pce
                    best_prediction = prediction
                    best_config = config_str
                    
            except Exception as e:
                error_msg = str(e)
                print(f"  Method={method}, MaxDegree={max_degree}, TargetAcc={target_acc}: Failed")
                print(f"    Error: {error_msg}")
                continue
        
        if best_model is None:
            # Try one final time with default settings (more permissive)
            print(f"  All configurations failed. Trying default configuration as fallback...")
            try:
                meta_opts = {
                    "Type": "Metamodel",
                    "Input": my_input["Name"],
                    "MetaType": "PCE",
                    "Degree": list(range(1, 6)),
                    "Method": "LARS",
                    "Display": "standard",
                    "ExpDesign": {
                        "X": ed.tolist(),
                        "Y": actual.tolist()
                    }
                }
                best_model = uq.createModel(meta_opts)
                best_prediction = uq.evalModel(best_model, ed.tolist()).flatten()
                best_r2 = calculate_r2(actual, best_prediction)
                best_config = "Fallback: Method=LARS, MaxDegree=5, No TargetAccuracy"
                print(f"  Fallback succeeded: R² = {best_r2:.6f}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to build PCE model for {out_name} with any configuration.\n"
                    f"Last error: {str(e)}\n"
                    f"Check that 'ed' has shape (n_samples, n_params) matching 'my_input' dimensions."
                ) from e
        
        print(f"  → Selected: {best_config} (R² = {best_r2:.6f})")
        
        pce_models[out_name] = best_model
        prediction_pce[out_name] = best_prediction
    
    return pce_models, prediction_pce

