# Figures Directory

This directory contains all generated figures and visualizations from the neuroscience simulation project.

## Directory Structure

### 📊 **Main Analysis Categories**

- **`precision_comparisons/`** - Figures comparing different numerical precision formats
- **`rmse_analysis/`** - Root Mean Square Error analysis across precision formats
- **`spike_analysis/`** - Spike timing, firing rates, and spiking behavior analysis
- **`error_analysis/`** - Error accumulation and distribution analysis
- **`simulation_results/`** - Raw simulation outputs and test results
- **`metrics_analysis/`** - Quality metrics and comparison measures

### 📁 **Specialized Directories**

- **`figuras/`** - Main analysis figures (organized by analysis type)
- **`images/`** - General simulation images
- **`images_compare/`** - Comparison plots between different approaches
- **`sweep_plots/`** - Parameter sweep visualizations
- **`results/`** - Detailed analysis results and reports

## Figure Types

### **Precision Format Comparisons**
- Float16 vs Float32 vs Float64 vs Posit16
- Numerical accuracy analysis
- Trade-offs between precision and efficiency

### **Error Analysis**
- RMSE (Root Mean Square Error) measurements
- Accumulated error over time
- Logarithmic error analysis

### **Spiking Behavior**
- Spike timing accuracy
- Firing rate analysis
- Spike train comparisons

### **Quality Metrics**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Other numerical quality measures

## Usage

Each subdirectory contains a README.md file explaining the specific figures and their purpose. This organization makes it easy to find relevant visualizations for different types of analysis.

## File Naming Convention

Figures are named descriptively to indicate:
- What is being measured (e.g., `rmse_vs_I`)
- Which precision format (e.g., `_float32`, `_posit16`)
- What parameter is varied (e.g., `_vs_current`, `_vs_time`) 