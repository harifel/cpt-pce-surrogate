import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from config import UQCLOUD_TOKEN
myToken = UQCLOUD_TOKEN

# UQpyLab cloud session
from uqpylab import sessions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
TEMPLATE_FILE = str(SCRIPT_DIR / 'GPFEM_template.xlsx')

# Set the output file (uncomment the one you want to use)
OUTPUT_FILE = str(SCRIPT_DIR / 'GPFEM_database_1000_samples_20260527.xlsx')
# Sobol sequences require N to be a power of 2 for optimal space-filling.
# 256 = 2^8  (nearest power of 2 above the original 200)
N_SAMPLES = 1000

# ---------------------------------------------------------------------------
# Parameter definitions — uniform marginals [Min, Max]
# ---------------------------------------------------------------------------
PARAMETERS = {
    'Parameter_Name': ['z', 'gamma', 'phi', 'OCR', 'Rint', 'lambda', 'kappa', 'vur', 'b0', 'h'],

    # Very soft layer
    'Min_Value': [0.5,    14.5, 15.0,  1.0, 0.8, 0.07/2.303,  0.1*0.07/2.303,  0.15, 0.0, 0.0],
    'Max_Value': [20,   19.0, 35.0,  4.0, 0.8, 0.4/2.303, 0.333*0.4/2.303,  0.15, 5.0, 5.0],


    'Units': ['m', 'kN/m³', '°', '-', '-', '-', '-', '-', '-', '-', '-']
}

N_PARAMS = len(PARAMETERS['Parameter_Name'])


# ---------------------------------------------------------------------------
# UQpyLab Sobol sampling
# ---------------------------------------------------------------------------
def generate_sobol_samples_uqpylab(n_samples: int, parameters: dict) -> np.ndarray:
    """
    Draw n_samples Sobol quasi-random samples via UQpyLab.

    Each parameter is treated as an independent Uniform marginal [Min, Max].
    The UQpyLab framework handles the Sobol sequence generation in [0,1]^d
    and maps it to the physical parameter space through the inverse CDF of
    each marginal (i.e. linear rescaling for uniform distributions).

    Parameters
    ----------
    n_samples : int
        Number of samples. Should be a power of 2 (e.g. 128, 256, 512).
    parameters : dict
        Dict with keys 'Parameter_Name', 'Min_Value', 'Max_Value', 'Units'.

    Returns
    -------
    scaled_samples : np.ndarray, shape (n_samples, n_params)
    """
    print("Connecting to UQpyLab cloud session...")

    # Start UQ[PY]LAB session
    UQCloud_instance = os.environ.get("UQCLOUD_HOST", "https://uqcloud.ethz.ch")
    token = os.environ.get("UQCLOUD_TOKEN", myToken)

    mySession = sessions.cloud(host=UQCloud_instance, token=token, strict_ssl=True)
    uq = mySession.cli
    mySession.reset()

    # Build the marginals list — one Uniform distribution per parameter
    marginals = []
    for i in range(N_PARAMS):
        lo = parameters['Min_Value'][i]
        hi = parameters['Max_Value'][i]

        # Parameters with identical bounds (e.g. Rint=0.8, vur=0.15) are
        # treated as constants by using a Constant marginal type.
        if np.isclose(lo, hi):
            marginals.append({
                'Type':       'Constant',
                'Parameters': [lo],
                'Name':       parameters['Parameter_Name'][i],
            })
        else:
            marginals.append({
                'Type':       'Uniform',
                'Parameters': [lo, hi],
                'Name':       parameters['Parameter_Name'][i],
            })

    # Create the UQpyLab INPUT object
    input_opts = {
        'Marginals': marginals,
        'Name':      'GPFEM_InputModel',
    }
    print("Creating UQpyLab INPUT object with Uniform marginals...")
    my_input = uq.createInput(input_opts)

    # Draw Sobol samples — UQpyLab maps from [0,1]^d to physical space
    print(f"Drawing {n_samples} Sobol samples (UQpyLab)...")
    scaled_samples = uq.getSample(my_input, n_samples, 'Sobol')
    # getSample returns an (n_samples x n_params) numpy array

    print("Sobol sampling complete.")
    return np.array(scaled_samples)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_parameters(scaled_samples: np.ndarray, param_names: list, units: list,
                    output_file: str = 'parameter_distributions.png') -> plt.Figure:
    """
    2-row subplot grid: histogram (top) and scatter plot (bottom) per parameter.
    """
    print("Creating parameter plots...")
    fig, axes = plt.subplots(2, N_PARAMS, figsize=(20, 10))
    fig.suptitle(
        f'UQpyLab Sobol Sequence (N={scaled_samples.shape[0]}) — Parameter Distributions',
        fontsize=16, fontweight='bold'
    )

    for i, (name, unit) in enumerate(zip(param_names, units)):
        samples  = scaled_samples[:, i]
        mean_val = np.mean(samples)
        std_val  = np.std(samples)

        # --- Histogram ---
        ax_hist = axes[0, i]
        ax_hist.hist(samples, bins=20, alpha=0.7, color='steelblue',
                     edgecolor='black', linewidth=0.5)
        ax_hist.set_title(f'{name} [{unit}]', fontweight='bold', fontsize=10)
        ax_hist.set_xlabel(f'[{unit}]', fontsize=8)
        ax_hist.set_ylabel('Count', fontsize=8)
        ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.tick_params(labelsize=7)
        stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}'
        ax_hist.text(0.02, 0.97, stats_text, transform=ax_hist.transAxes,
                     fontsize=7, va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # --- Scatter ---
        ax_scat = axes[1, i]
        ax_scat.scatter(np.arange(len(samples)), samples,
                        alpha=0.5, color='navy', s=8)
        ax_scat.set_xlabel('Sample index', fontsize=8)
        ax_scat.set_ylabel(f'[{unit}]', fontsize=8)
        ax_scat.axhline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax_scat.grid(True, alpha=0.3)
        ax_scat.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved → {output_file}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("GPFEM Sampling Plan  —  UQpyLab Sobol Sequence")
    print("=" * 60)

    # 1. Read Excel template
    print(f"\n[1/4] Reading template: {TEMPLATE_FILE}")
    if not Path(TEMPLATE_FILE).exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_FILE}")
    template_df = pd.read_excel(TEMPLATE_FILE, header=None)
    print(f"      Template shape: {template_df.shape}")

    # 2. Generate Sobol samples via UQpyLab
    print(f"\n[2/4] Generating {N_SAMPLES} Sobol samples via UQpyLab...")
    scaled_samples = generate_sobol_samples_uqpylab(N_SAMPLES, PARAMETERS)
    print(f"      Samples shape: {scaled_samples.shape}")

    # 3. Write samples into expanded template
    print("\n[3/4] Writing samples into template...")
    total_cols = 6 + N_SAMPLES          # original 6 header columns + sample columns
    result_df  = template_df.copy()
    for col_idx in range(len(template_df.columns), total_cols):
        result_df[f'Col_{col_idx}'] = ''

    for i in range(N_PARAMS):
        for j in range(N_SAMPLES):
            result_df.iloc[i, 6 + j] = scaled_samples[j, i]

    result_df.to_excel(OUTPUT_FILE, index=False, header=False)
    print(f"      Saved → {OUTPUT_FILE}")
    print(f"      Shape: {result_df.shape[0]} rows × {result_df.shape[1]} cols")

    # 4. Visualise
    print("\n[4/4] Visualising distributions...")
    plot_file = OUTPUT_FILE.replace('.xlsx', '_parameter_distributions.png')
    plot_parameters(
        scaled_samples,
        PARAMETERS['Parameter_Name'],
        PARAMETERS['Units'],
        plot_file
    )

    print("\nDone.")


if __name__ == "__main__":
    main()