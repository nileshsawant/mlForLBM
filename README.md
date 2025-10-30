# Create an AI clone of your CFD model

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![GitHub](https://img.shields.io/badge/GitHub-nileshsawant%2FmlForLBM-blue)](https://github.com/nileshsawant/mlForLBM)

![Velocity Cracks Demo](velocity_cracks.gif)
*Animation credit: [@eyoung55](https://github.com/eyoung55)*

A complete pipeline for training neural networks to predict flow and temperature fields. Tested for complex crack geometries using Lattice Boltzmann Method simulations. The test case setup in this example is from https://github.com/nileshsawant/marblesThermal/tree/backup-before-reset/Tests/test_files/isothermal_cracks 

## Overview

This project implements a neural network that learns to predict 5 physics fields (velocity, heat flux, density, energy, temperature) in thermal flow simulations through cracked materials. The approach combines:

- **Physics-aware activations**: Softplus for positive quantities, tanh for bounded ranges
- **Physics violations**: Mathematical guarantees prevent negative densities/temperatures
- **AMReX-based LBM training data**: High-fidelity ground truth simulations
- **Efficient training**: 20 epochs with manual learning rate scheduling
- **ParaView visualization**: Direct integration for 3D analysis
- **Training dataset**: 12,625 samples (125 cases × 101 timesteps)
- **Perfect generalization**: Maintains physics compliance on unseen geometries

## Project Structure

```
mlForLBM/
├── README.md                                    # This file
├── isothermal_cracks.inp                       # AMReX/LBM input template
├── microstructure_geom_*.csv                   # Crack geometries (seeds 1-6)
├── marbles3d.gnu.TPROF.MPI.ex                 # AMReX LBM executable
│
├── generate_training_data_fixed.py             # Generate training data pipeline
├── process_training_data_for_ml.py             # Convert AMReX → ML format
├── train_lbm_neural_network_physics_aware.py  # Physics-aware training
│                                                
├── validate_neural_network_physics_aware.py   # Validation on seen geometries
├── validate_seed6_minimal_physics_aware.py    # Validation on unseen geometries
├── predict_and_visualize_physics_aware.py     # Interactive prediction & visualization
│
└── validate_physics_aware_model.py            # Physics violations checker
```

## Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n marbles_ml python=3.9
conda activate marbles_ml

# Install dependencies
conda install tensorflow numpy pandas matplotlib seaborn scikit-learn
conda install -c conda-forge yt h5py
pip install --upgrade tensorflow
```

### 1. Generate Training Data

```bash
python generate_training_data_fixed.py
python process_training_data_for_ml.py
```

This creates 125 LBM simulation cases with varying:
- Viscosity (nu): 5 values
- Temperature: 5 values  
- Crack geometries: 5 different seeds

### 2. Train Neural Network

```bash
python train_lbm_neural_network_physics_aware.py
```

Trains the physics-aware model with:
- **Smart activations**: Softplus for positive quantities, tanh for bounded ranges
- **20 epochs**: Manual learning rate reduction every 5 epochs
- **Zero violations**: Mathematical guarantee of physics compliance

### 3. Validate Model

```bash
# Validation on seen geometries
python validate_neural_network_physics_aware.py

# Validation on unseen geometry (generalization test)
python validate_seed6_minimal_physics_aware.py

# Quick physics violations check
python validate_physics_aware_model.py
```

### 4. Generate Predictions and Visualize

```bash
# Interactive prediction with visualization
python predict_and_visualize_physics_aware.py --geometry 3 --nu 0.003 --temperature 0.035

# With LBM comparison
python predict_and_visualize_physics_aware.py --geometry 3 --nu 0.003 --temperature 0.035 --run-lbm
```

## Physics-Aware Architecture

The physics-aware neural network solves the fundamental problem of physics violations through smart activation functions:

**Activation Functions**

```python
# Physics-aware output layers
density_output = Conv3D(1, (1,1,1), activation='softplus', name='density')(features)
energy_output = Conv3D(1, (1,1,1), activation='softplus', name='energy')(features)  
temperature_output = Conv3D(1, (1,1,1), activation='softplus', name='temperature')(features)

velocity_output = Conv3D(3, (1,1,1), activation='tanh', name='velocity')(features)
heat_flux_output = Conv3D(3, (1,1,1), activation='tanh', name='heat_flux')(features)
```

**Mathematical Guarantees:**
- **Softplus**: `log(1 + exp(x))` ensures density, energy, temperature > 0
- **Tanh**: `tanh(x)` bounds velocity and heat flux to [-1, +1] range
- **No loss function penalties needed**: Physics compliance by design

### Training Efficiency

**Optimized Learning Schedule:**
- Learning rate reduction every 5 epochs  
- Manual schedule: 0.001 → 0.0005 → 0.00025 → 0.000125


**Features:**
- Automatic geometry ID conversion (3 → microstructure_geom_3.csv)
- Real-time physics violations monitoring (should always be 0%)
- Side-by-side neural network vs LBM comparison
- ParaView-ready VTU output

## Technical Details

### Physics-Aware Architecture

The physics-aware model uses smart activation functions to guarantee physics compliance:

```python
# 3D CNN feature extraction
geometry_input = Input((60, 40, 30, 1))  # 3D crack geometry
params_input = Input((4,))                # [nu, T, alpha, time]

# Efficient encoder-decoder architecture
x = Conv3D(8, (3,3,3), activation='relu')(geometry_input)
x = Conv3D(16, (3,3,3), activation='relu')(x)
# ... (encoder-decoder layers)

# PHYSICS-AWARE OUTPUT BRANCHES with smart activations
velocity_out = Conv3D(3, (1,1,1), activation='tanh', name='velocity')(shared_features)
heat_flux_out = Conv3D(3, (1,1,1), activation='tanh', name='heat_flux')(shared_features)
density_out = Conv3D(1, (1,1,1), activation='softplus', name='density')(shared_features)
energy_out = Conv3D(1, (1,1,1), activation='softplus', name='energy')(shared_features)
temperature_out = Conv3D(1, (1,1,1), activation='softplus', name='temperature')(shared_features)
```

## Model Output

The physics-aware model predicts 5 thermal flow fields:

1. **Velocity**: 3D flow velocity field (tanh bounded)
2. **Heat Flux**: 3D thermal flux field (tanh bounded)  
3. **Density**: Fluid density field (softplus positive)
4. **Energy**: Total energy field (softplus positive)
5. **Temperature**: Temperature field (softplus positive)

## Applications

- Fast thermal flow analysis in cracked materials
- Material design and optimization studies  
- Digital twins for real-time predictions
- Engineering simulation with guaranteed physics compliance

## Summary

This physics-aware neural network successfully predicts thermal flow fields in cracked materials with guaranteed physics compliance. The model achieves no physics violations while maintaining fast inference times, making it suitable for real-time engineering applications.

### Getting Started

Train the physics-aware model, run validation tests, and create interactive predictions using the provided scripts. All models maintain physics compliance by design while delivering accurate thermal flow predictions for complex crack geometries.

### Validation Strategy

- **Interpolation test**: Parameters between training points, same geometry types
- **Extrapolation test**: Completely unseen crack geometry (seed 6)
- **Metrics**: MSE, MAE, R², relative error, correlation
- **Visualization**: Field comparisons, temporal evolution, error distributions

## Citation

If you use this code, please cite:

```bibtex
@software{sawant2025_lbm_ml,
  title={mlForLBM: A neural network framework for transient fluid simulation surrogates},
  author={Sawant, Nilesh},
  year={2025},
  url={https://github.com/nileshsawant/mlForLBM},
  version={0.1},
  month={10},
  keywords={machine learning, computational fluid dynamics, lattice boltzmann method, physics-informed neural networks, thermal flow, crack analysis}
}
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License**.

### What this means:

** You ARE allowed to:**
- Use this code for research, education, and personal projects
- Share and redistribute the code
- Modify and adapt the code for your needs
- Build upon this work

** You are NOT allowed to:**
- Use this code for commercial purposes
- Sell products or services based on this code

** Attribution Requirements:**
When using this code, you must:
- Credit the original author: **Nilesh Sawant**
- Link to this repository: `https://github.com/nileshsawant/mlForLBM`
- Indicate if you made changes to the original code

For full terms, see: https://creativecommons.org/licenses/by-nc/4.0/

## Contact

- **Author**: Nilesh Sawant
- **GitHub**: https://github.com/nileshsawant
- **Project**: https://github.com/nileshsawant/mlForLBM

## Acknowledgments

- AMReX framework for LBM simulations
- TensorFlow/Keras for neural networks
- yt-project for data processing
- ParaView for visualization