"""
Configuration file for the neuroscience simulation project.
Defines paths, settings, and constants used throughout the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Analysis directories
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
NEURON_MODELS_DIR = ANALYSIS_DIR / "neuron_models"
NETWORK_ANALYSIS_DIR = ANALYSIS_DIR / "network_analysis"
NUMERICAL_PRECISION_DIR = ANALYSIS_DIR / "numerical_precision"
UTILS_DIR = ANALYSIS_DIR / "utils"
PLOTTING_DIR = ANALYSIS_DIR / "plotting"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METRICS_DIR = DATA_DIR / "metrics"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Figure subdirectories
FIGURAS_DIR = FIGURES_DIR / "figuras"
IMAGES_DIR = FIGURES_DIR / "images"
IMAGES_COMPARE_DIR = FIGURES_DIR / "images_compare"
SWEEP_PLOTS_DIR = FIGURES_DIR / "sweep_plots"
RESULTS_DIR = FIGURES_DIR / "results"

# Ensure all directories exist
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        ANALYSIS_DIR, NEURON_MODELS_DIR, NETWORK_ANALYSIS_DIR,
        NUMERICAL_PRECISION_DIR, UTILS_DIR, PLOTTING_DIR,
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, METRICS_DIR,
        OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR,
        FIGURAS_DIR, IMAGES_DIR, IMAGES_COMPARE_DIR, SWEEP_PLOTS_DIR, RESULTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Simulation parameters
SIMULATION_PARAMS = {
    'default_dt': 0.1,  # ms
    'default_duration': 1000,  # ms
    'default_v_rest': -70,  # mV
    'default_v_reset': -60,  # mV
    'default_v_threshold': -55,  # mV
}

# Precision types
PRECISION_TYPES = ['float16', 'float32', 'float64', 'posit16']

# Neuron types for Izhikevich model
IZHIKEVICH_TYPES = {
    'RS': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 2},      # Regular spiking
    'IB': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},      # Intrinsically bursting
    'CH': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},      # Chattering
    'FS': {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2}        # Fast spiking
}

# Plotting settings
PLOT_SETTINGS = {
    'figure_size': (12, 8),
    'dpi': 300,
    'save_format': 'png',
    'style': 'seaborn-v0_8',
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True,
}

if __name__ == "__main__":
    ensure_directories()
    print("Project directories created successfully!") 