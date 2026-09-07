"""Plotting functions for sensitivity analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_actual_vs_predicted(
    out: Dict[str, np.ndarray],
    prediction_pce: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    output_names: Optional[List[str]] = None,
) -> None:
    """Plot actual vs predicted values for all outputs.
    
    Parameters:
    -----------
    out : Dict[str, np.ndarray]
        Dictionary with actual output values (e.g. 'depth', 'qt', 'fs', 'u2')
    prediction_pce : Dict[str, np.ndarray]
        Dictionary with predicted output values (same keys as out)
    save_path : Optional[Path]
        If provided, save the figure to this path
    output_names : Optional[List[str]]
        Order of outputs to plot. If None, uses keys from prediction_pce.
    """
    cm = 1 / 2.54
    if output_names is None:
        output_names = list(prediction_pce.keys())
    n_out = len(output_names)
    fig, axes = plt.subplots(1, n_out, figsize=(9*cm , 4*cm * n_out))
    if n_out == 1:
        axes = [axes]
    output_labels = {
        "depth": "depth (m)",
        "qt": "$q_t$ (kPa)",
        "fs": "$f_s$ (kPa)",
        "u2": "$u_2$ (kPa)",
    }
    
    for idx, out_name in enumerate(output_names):
        ax = axes[idx]
        actual = out[out_name]
        predicted = prediction_pce[out_name]
        
        # Calculate R²
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Scatter plot
        ax.scatter(actual, predicted, s=50, color='k')
        
        # 1:1 line (perfect prediction)
        min_val = min(np.min(actual), np.min(predicted))
        max_val = max(np.max(actual), np.max(predicted))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 line')
        
        # Labels and title
        lbl = output_labels.get(out_name, out_name)
        ax.set_xlabel(f'Actual {lbl}')
        ax.set_ylabel(f'Predicted {lbl}')
        ax.grid(True, alpha=0.3)
        
        # Make axes equal for better visualization
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def cusunoro(
    ed: np.ndarray,
    y: np.ndarray,
    plot: bool = False,
    var_names: Optional[List[str]] = None,
    use_actual_values: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CUSUNORO (Cumulative Sum of Normalized Output) for sensitivity analysis."""
    ed = np.asarray(ed)
    y = np.asarray(y)
    ns, m = ed.shape
    if y.ndim == 1:
        y = y[:, None]
    nout = y.shape[1]
    if var_names is None:
        var_names = [f"X_{idx+1}" for idx in range(m)]
    
    # Always compute cumulative normalized output for y-axis
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y_std = np.where(y_std == 0, 1.0, y_std)
    z = (y - y_mean) / y_std
    
    x_vals = np.zeros((ns, m))
    y_vals = np.zeros((ns, m, nout))
    for idx in range(m):
        order = np.lexsort((np.arange(ns), ed[:, idx]))
        if use_actual_values:
            # Use actual input parameter values on x-axis
            x_vals[:, idx] = ed[order, idx]  # Actual input values, sorted
        else:
            # Use normalized rank on x-axis
            x_vals[:, idx] = (np.arange(1, ns + 1)) / ns  # Normalized rank
        # Always use cumulative normalized output on y-axis
        y_vals[:, idx, :] = np.cumsum(z[order, :], axis=0) / ns  # Cumulative normalized
    
    if plot:
        ncols = int(np.ceil(np.sqrt(m)))
        nrows = int(np.ceil(m / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = axes.flatten() if m > 1 else [axes]
        for idx in range(m):
            ax = axes[idx]
            for out_idx in range(nout):
                ax.plot(x_vals[:, idx], y_vals[:, idx, out_idx], linewidth=1.5, label=f"Output {out_idx+1}" if nout > 1 else None)
            ax.axhline(0, linestyle=":", color="gray", linewidth=1)
            if use_actual_values:
                ax.set_xlabel(f"{var_names[idx]}", fontsize=12)
                ax.set_ylabel("Cumulative Normalized Output", fontsize=12)
                ax.set_title(f"CUSUNORO: {var_names[idx]}", fontsize=14)
            else:
                ax.set_xlim(0, 1)
                ax.set_xlabel("Normalized Rank (τ)", fontsize=12)
                ax.set_ylabel("Cumulative Normalized Output", fontsize=12)
                ax.set_title(f"CUSUNORO: {var_names[idx]}", fontsize=14)
            ax.grid(True, alpha=0.3)
            if nout > 1:
                ax.legend(fontsize=10)
        # Hide unused subplots and add formula to the last visible subplot
        last_visible_idx = m - 1
        for idx in range(m, len(axes)):
            axes[idx].set_visible(False)
        
        # Add formula explanation to the last subplot
        if m > 0:
            ax_last = axes[last_visible_idx]
            formula_text = (
                "Formula:\n"
                "z_i = (y_i - ȳ) / σ_y\n"
                "CUSUNORO(k) = (1/n) Σᵢ₌₁ᵏ z_i\n\n"
                "Positive: outputs above average\n"
                "Negative: outputs below average"
            )
            ax_last.text(0.02, 0.98, formula_text, transform=ax_last.transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.tight_layout()
    
    return y_vals, x_vals

