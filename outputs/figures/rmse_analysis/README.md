# RMSE Analysis

This directory contains figures related to Root Mean Square Error (RMSE) analysis across different numerical precision formats.

## Contents

### RMSE vs Current (I)
- **rmse_vs_I.png** - RMSE vs input current for different formats
- **rmse_vs_I_extended.png** - Extended RMSE analysis vs current
- **rmse_vs_I_p24.png** - RMSE vs current for 24-bit precision
- **rmse_vs_I_p32.png** - RMSE vs current for 32-bit precision
- **rmse_vs_I_p24_p20.png** - RMSE vs current comparing 24-bit and 20-bit
- **rmse_vs_I_bfloat.png** - RMSE vs current for bfloat format
- **rmse_vs_I_golden.png** - RMSE vs current for golden reference
- **rmse_vs_I_cercacero.png** - RMSE vs current near zero values

### RMSE Heatmaps
- **heatmap_rmse_posit16.png** - RMSE heatmap for posit16 format
- **heatmap_rmse_float32.png** - RMSE heatmap for float32 format
- **heatmap_rmse_float16.png** - RMSE heatmap for float16 format

### Other RMSE Metrics
- **rmse_spike_vs_current.png** - RMSE of spike timing vs current
- **rmse_barplot_red.png** - Bar plot of RMSE values
- **rmse_vs_corriente.png** - RMSE vs current (Spanish labeling)

## Description

These figures analyze the numerical accuracy of different precision formats by measuring the Root Mean Square Error between simulation results and reference values. Lower RMSE indicates better numerical accuracy. 