# TFG Evaluacion de SNNs con distintos formatos numericos

This project implements and analyzes neural network simulations using Brian2 with different numerical precision types (float16, float32, float64, posit16).

## Project Structure

```
tfg/
├── analysis/                    # Analysis scripts organized by category
│   ├── neuron_models/          # Single neuron and basic neuron group simulations
│   ├── network_analysis/       # Network-level simulations and analysis
│   ├── numerical_precision/    # Precision comparison and numerical analysis
│   ├── utils/                  # Utility functions and helper scripts
│   └── plotting/               # Plotting and visualization scripts
├── data/                       # Data files
│   ├── raw/                    # Raw simulation data
│   ├── processed/              # Processed data
│   └── metrics/                # Metrics and comparison results
├── outputs/                    # Output files
│   ├── figures/                # Generated plots and visualizations
│   │   ├── figuras/            # Main analysis figures
│   │   ├── images/             # General images
│   │   ├── images_compare/     # Comparison plots
│   │   ├── sweep_plots/        # Parameter sweep visualizations
│   │   └── results/            # Analysis result figures
│   ├── reports/                # Analysis reports
│   └── logs/                   # Simulation and analysis logs
├── brian2/                     # Brian2 library source code
├── universal/                  # Universal number system library
└── venv/                       # Python virtual environment
```

## Analysis Categories

### 1. Neuron Models (`analysis/neuron_models/`)
- Single neuron simulations with different parameters
- Parameter sweeps (current, time constants, thresholds)
- Basic neuron group implementations
- Izhikevich neuron models

### 2. Network Analysis (`analysis/network_analysis/`)
- Spiking neural network simulations
- STDP learning implementations
- Multi-layer network analysis
- Pulse detection and noise analysis

### 3. Numerical Precision (`analysis/numerical_precision/`)
- Float16 vs Float32 vs Float64 vs Posit16 comparisons
- Numerical error analysis
- Stress test cases
- Debugging numerical issues

### 4. Utilities (`analysis/utils/`)
- Analysis helper functions
- Metrics calculation
- Data processing utilities
- Testing and validation scripts

### 5. Plotting (`analysis/plotting/`)
- Visualization scripts
- Comparison plots
- Parameter sweep visualizations

## Key Features

- **Multi-precision support**: Float16, Float32, Float64, and Posit16
- **Comprehensive analysis**: From single neurons to complex networks
- **Parameter sweeps**: Systematic exploration of parameter spaces
- **Error analysis**: RMSE, accumulated error, spike timing analysis
- **Visualization**: Heatmaps, scatter plots, time series, and statistical plots

## Getting Started

1. **Setup environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run basic simulations**:
   ```bash
   cd analysis/neuron_models
   python neuron1.py
   ```

3. **Run precision comparisons**:
   ```bash
   cd analysis/numerical_precision
   python comprehensive_numerical_analysis.py
   ```

## Data Organization

- **Metrics**: Stored in `data/metrics/` as CSV files
- **Figures**: Generated in `outputs/figures/` organized by analysis type
- **Logs**: Simulation logs stored in `outputs/logs/`
- **Results**: Analysis results in `outputs/figures/results/`

## Dependencies

- Brian2 (neural simulation library)
- NumPy (numerical computing)
- Matplotlib (plotting)
- Pandas (data analysis)
- SciPy (scientific computing)
- Posit wrapper (custom implementation)

## Contributing

When adding new analysis scripts:
1. Place them in the appropriate category directory
2. Follow the existing naming conventions
3. Update this README if adding new categories
4. Ensure outputs go to the correct organized directories 
