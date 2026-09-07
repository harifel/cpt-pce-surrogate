import pandas as pd
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from pathlib import Path
from openpyxl.utils import get_column_letter

# # Configuration - Get the directory where this script is located
# SCRIPT_DIR = Path(__file__).parent
# TEMPLATE_FILE = str(SCRIPT_DIR / 'GPFEM_template.xlsx')
# # Set the output file (uncomment the one you want to use)
# OUTPUT_FILE = str(SCRIPT_DIR / 'GPFEM_map_raw_originparameters_20260424.xlsx')
# N_SAMPLES = 500
# SEED = 42

# # Parameters
# PARAMETERS = {
#     'Parameter_Name': ['z', 'gamma', 'phi', 'OCR', 'Rint', 'lambda', 'kappa', 'vur'],

#     'Min_Value': [5, 14.5, 15.0, 1.0, 0.8, 0.4/2.303, 0.4/2.303*0.333, 0.15],
#     'Max_Value': [20, 19.0, 35.0, 4.0, 0.8, 0.07/2.303, 0.07/2.303*0.10, 0.15],

#     'Units': ['m', 'kN/m³', '°', '-', '-', '-', '-', '-']
# }



# Configuration - Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
TEMPLATE_FILE = str(SCRIPT_DIR / 'GPFEM_template.xlsx')
# Set the output file (uncomment the one you want to use)
OUTPUT_FILE = str(SCRIPT_DIR / 'GPFEM_map_raw_originparameters_20260424_depth_1_5m_sigma.xlsx')
N_SAMPLES = 500
SEED = 42

# Parameters
PARAMETERS = {
    'Parameter_Name': ['z', 'gamma', 'phi', 'OCR', 'Rint', 'lambda', 'kappa', 'vur', 'sigma'],

    'Min_Value': [1, 0, 15.0, 1.0, 0.8, 0.4/2.303, 0.4/2.303*0.333, 0.15, 4.5],
    'Max_Value': [5, 0, 35.0, 4.0, 0.8, 0.07/2.303, 0.07/2.303*0.10, 0.15, 45],

    'Units': ['m', 'kN/m³', '°', '-', '-', '-', '-', '-', 'kPa']
}



def plot_parameters(scaled_samples, param_names, units, output_file='parameter_distributions.png'):
    """
    Create subplots for all parameters showing both histogram and scatter plot distributions.
    """
    print("Creating parameter plots...")
    
    # Create figure with 2 rows x num_params columns (2 plots per parameter)
    num_params = len(param_names)
    fig, axes = plt.subplots(2, num_params, figsize=(2.5 * num_params, 10))
    fig.suptitle('Latin Hypercube Sampling - Parameter Distributions', fontsize=16, fontweight='bold')
    
    for i, (name, unit) in enumerate(zip(param_names, units)):
        # Get samples for this parameter
        samples = scaled_samples[:, i]
        
        # Add statistics
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        min_val = np.min(samples)
        max_val = np.max(samples)
        
        # Top row: Histogram
        ax_hist = axes[0, i]
        ax_hist.hist(samples, bins=20, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
        ax_hist.set_title(f'{name} [{unit}] - Histogram', fontweight='bold', fontsize=11)
        ax_hist.set_xlabel(f'Value [{unit}]', fontsize=9)
        ax_hist.set_ylabel('Frequency', fontsize=9)
        ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.tick_params(axis='both', which='major', labelsize=8)
        
        # Bottom row: Scatter plot
        ax_scatter = axes[1, i]
        x_values = np.arange(len(samples))
        ax_scatter.scatter(x_values, samples, alpha=0.6, color='darkblue', s=20)
        ax_scatter.set_title(f'{name} [{unit}] - Scatter Plot', fontweight='bold', fontsize=11)
        ax_scatter.set_xlabel('Sample Index', fontsize=9)
        ax_scatter.set_ylabel(f'Value [{unit}]', fontsize=9)
        ax_scatter.axhline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.tick_params(axis='both', which='major', labelsize=8)
        
        # Add statistics text to histogram
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nRange: {min_val:.3f}-{max_val:.3f}'
        ax_hist.text(0.02, 0.98, stats_text, transform=ax_hist.transAxes, fontsize=7,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot in the GPFEM folder
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Parameter plots saved as: {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig

def main():
    print("Starting sampling plan generation...")

    num_params = len(PARAMETERS['Parameter_Name'])

    # Read template
    print(f"Reading template: {TEMPLATE_FILE}")
    if not Path(TEMPLATE_FILE).exists():
        raise FileNotFoundError(f"Template file not found: {TEMPLATE_FILE}")
    template_df = pd.read_excel(TEMPLATE_FILE, header=None)
    print(f"Template shape: {template_df.shape}")

    # Generate LHS samples
    print(f"Generating {N_SAMPLES} LHS samples...")
    sampler = qmc.LatinHypercube(d=num_params, seed=SEED, optimization="random-cd")
    samples = sampler.random(n=N_SAMPLES)

    # Scale samples to parameter ranges
    scaled_samples = np.zeros_like(samples)
    for i in range(num_params):
        min_val = PARAMETERS['Min_Value'][i]
        max_val = PARAMETERS['Max_Value'][i]
        scaled_samples[:, i] = samples[:, i] * (max_val - min_val) + min_val
    
    print(f"Generated samples shape: {scaled_samples.shape}")
    
    # Create result DataFrame with expanded columns
    print("Expanding template to accommodate samples...")
    total_cols = 6 + N_SAMPLES  # Original 6 columns + samples
    result_df = template_df.copy()
    
    # Add empty columns for samples
    for col_idx in range(len(template_df.columns), total_cols):
        result_df[f'Col_{col_idx}'] = pd.Series([''] * len(result_df), dtype=object)
    
    print(f"Expanded from {len(template_df.columns)} to {total_cols} columns")
    
    # Insert samples as columns in first num_params rows
    print("Inserting samples...")
    for i in range(num_params):
        param_name = PARAMETERS['Parameter_Name'][i]
        print(f"Parameter {i} ({param_name}): {N_SAMPLES} samples")
        
        # Insert samples starting from column G (index 6)
        start_col = 6
        for j in range(N_SAMPLES):
            result_df.iloc[i, start_col + j] = scaled_samples[j, i]
    
    # Save result
    print(f"Saving to: {OUTPUT_FILE}")
    result_df.to_excel(OUTPUT_FILE, index=False, header=False)
    print(f"SUCCESS: Sampling plan saved as {OUTPUT_FILE}")
    print(f"File contains {result_df.shape[0]} rows and {result_df.shape[1]} columns")
    end_col_letter = get_column_letter(start_col + N_SAMPLES)  # start_col=6 -> 'G' is column 7 (1-based)
    print(f"Samples inserted in first {num_params} rows, columns G through {end_col_letter}")
    
    # Create parameter plots
    param_names = PARAMETERS['Parameter_Name']
    units = PARAMETERS['Units']
    plot_parameters(scaled_samples, param_names, units, f'{OUTPUT_FILE.replace(".xlsx", "_parameter_distributions.png")}')

if __name__ == "__main__":
    main()
