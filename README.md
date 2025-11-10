# Create an AI surrogate of your CFD model

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![GitHub](https://img.shields.io/badge/GitHub-nileshsawant%2FmlForLBM-blue)](https://github.com/nileshsawant/mlForLBM)

![Velocity Cracks Demo](velocity_cracks.gif)
*Animation credit: [@eyoung55](https://github.com/eyoung55)*

A complete pipeline for training neural networks to predict flow and temperature fields. It can be easily adapted to AMReX based codes and extended to other CFD codes. Tested for complex crack geometries using Lattice Boltzmann Method simulations. The test case setup in this example is from https://github.com/nileshsawant/marblesThermal/tree/main/Tests/test_files/isothermal_cracks .

This repository contains the end-to-end workflow used to build a physics-aware neural surrogate for thermal Lattice Boltzmann simulations in fractured materials. Training data are generated with the AMReX-based `marblesThermal` solver, converted into machine-learning ready tensors, and used to train a TensorFlow/Keras model that predicts velocity, heat flux, density, energy, and temperature fields while respecting sign and range constraints enforced by the network architecture.

## Key Capabilities

- Physics-aware activations (softplus / tanh) eliminate negative density, temperature, and energy predictions without extra loss terms.
- Automated training-data generation for 125 parameter combinations (5 ν values × 5 temperature values × 5 geometries).
- Validation pipelines for interpolation cases (`validate_neural_network_physics_aware.py`) and an unseen geometry seed (`validate_seed6_minimal_physics_aware.py`) that export metrics, plots, and summary JSON reports.
- Interactive prediction utility (`predict_and_visualize_physics_aware.py`) with optional AMReX reruns for side-by-side comparisons and ParaView-ready VTU conversion (`convert_nn_to_vtu_physics_aware.py`).
- JSON summaries keep large validation outputs manageable by storing dataset metadata instead of full 3D arrays.

## Repository Tour

| Script / Folder | Purpose |
| --- | --- |
| `generate_training_data_fixed.py` | Creates `.inp` files for the 5×5×5 parameter sweep and stages AMReX runs. |
| `process_training_data_for_ml.py` | Converts AMReX plotfiles into ML tensors (NPZ/HDF5/CSV) and records metadata. |
| `train_lbm_neural_network_physics_aware.py` | Trains the physics-aware 3D CNN (default: 20 epochs with manual LR schedule). |
| `validate_neural_network_physics_aware.py` | Runs interpolation validation, saves plots to `validation_physics_aware/plots`, and writes `physics_aware_validation_report.json`. |
| `validate_seed6_minimal_physics_aware.py` | Repeats the validation workflow on geometry seed 6 and writes the generalization report. |
| `predict_and_visualize_physics_aware.py` | Generates predictions for a chosen geometry/parameter set and can launch an AMReX comparison run. |
| `convert_nn_to_vtu_physics_aware.py` | Converts NPZ prediction bundles into ParaView VTU files for inspection. |
| `validation_physics_aware/`, `validation_seed6_physics_aware/` | Contain latest validation artifacts, plots, summary JSON, and VTU outputs. |

Additional notes and archived analysis live in `PHYSICS_AWARE_IMPROVEMENTS.md`, `VALIDATION_RESULTS_SUMMARY.md`, and related documents in the repository root.

## Environment Setup

The workflow was developed with Python 3.9 inside a conda environment named `marbles_ml`.

```bash
conda create -n marbles_ml python=3.9
conda activate marbles_ml

# Core dependencies
conda install tensorflow numpy pandas matplotlib seaborn scikit-learn
conda install -c conda-forge yt h5py

# Optional: upgrade TensorFlow or install GPU builds as needed
pip install --upgrade tensorflow
```

Running AMReX simulations requires the `marbles3d.gnu.TPROF.MPI.ex` executable present in the repo root (compiled from the `marblesThermal/` source). The helper scripts assume they are invoked from the repository directory.

## Workflow

1. **Generate LBM cases and run simulations**

   ```bash
   python generate_training_data_fixed.py
   ```

   Creates 125 case directories (5×5×5 sweep), generates geometries via `genCracks`, copies the solver, and then calls `run_all_simulations()` to execute `marbles3d.gnu.TPROF.MPI.ex isothermal_cracks.inp` inside each case. Plotfiles (`plt*`) will be written into the case folders.

2. **Convert training data**

   ```bash
   python process_training_data_for_ml.py
   ```

   Reads the plotfiles, extracts fields with `yt`, and writes compressed tensors in `ml_training_data/` alongside CSV summaries and metadata JSON.

3. **Train the surrogate**

   ```bash
   python train_lbm_neural_network_physics_aware.py
   ```

   Loads the processed tensors, trains for 20 epochs (default batch size 8), saves `lbm_flow_predictor_physics_aware.h5`, and exports `physics_aware_normalization_params.json` plus a `training_loss_physics_aware.png` curve.

4. **Validate**

   ```bash
   python validate_neural_network_physics_aware.py
   python validate_seed6_minimal_physics_aware.py
   ```

   Each script runs the AMReX solver as needed, generates neural predictions, compares them with LBM outputs, logs physics-violation checks, and writes plots plus a compact `physics_aware_*_validation_report.json` with metrics and dataset summaries.

5. **Predict & visualize**

   ```bash
   python predict_and_visualize_physics_aware.py --geometry 3 --nu 0.003 --temperature 0.035
   python predict_and_visualize_physics_aware.py --geometry 3 --nu 0.003 --temperature 0.035 --run-lbm
   ```

   Produces NPZ bundles under `custom_predictions_physics_aware/`. Use `convert_nn_to_vtu_physics_aware.py` to obtain VTU files for ParaView.

6. **Convert predictions to ParaView VTU**

   ```bash
   python convert_nn_to_vtu_physics_aware.py
   ```

   Finds neural-network prediction archives in `validation_physics_aware/`, `validation_seed6_physics_aware/`, and `custom_predictions_physics_aware/`, then writes matching `paraview_vtu/` directories for inspection in ParaView.

## Outputs and Metrics

- **Validation reports** (`validation_physics_aware/physics_aware_validation_report.json`, `validation_seed6_physics_aware/physics_aware_generalization_validation_report.json`)
  - Store per-case metrics: MSE, RMSE, MAE, R², mean relative error, max absolute error for each field.
  - Include lightweight `lbm_summary` and `nn_summary` sections with timestep ranges and array shapes instead of raw voxel data.
- **Plots**
  - Overall metric comparisons across physics fields.
  - Temporal evolution plots for each case showing MSE and R² per timestep.
- **Generated VTUs**
  - Located in `paraview_vtu/` subfolders beside prediction bundles for quick visualization.

## Physics-Aware Model

`train_lbm_neural_network_physics_aware.py` builds a compact 3D encoder–decoder that fuses geometry voxels with simulation parameters:

| Stage | Layers | Kernel / stride | Purpose |
| --- | --- | --- | --- |
| Geometry encoder | Conv3D → BN → ReLU (16 channels) | `3×3×3`, stride 1 | Looks at each voxel and its 26 neighbours to learn local crack patterns. |
| | Conv3D → BN → ReLU (24) + MaxPool3D | `3×3×3`, stride 1 → pooling `2×2×2` | Extracts richer features then halves the grid size to reduce cost. |
| | Conv3D → BN → ReLU (32) | `3×3×3`, stride 1 | Continues capturing neighbourhood interactions on the coarser grid. |
| | Conv3D → BN → ReLU (32) + MaxPool3D | `3×3×3`, stride 1 → pooling `2×2×2` | Further compresses spatial dimensions to `(15,10,7)` before global pooling. |
| Parameter tower | Dense 16 → Dense 32 (ReLU + dropout) | n/a | Encodes `[nu, temperature, alpha, normalized_time]` into a compact vector. |
| Fusion | Concatenate → Dense 128 → Dense (15×10×7×16) | n/a | Mixes geometry and parameter context, then reshapes back to a small 3D feature block. |
| Decoder | Conv3DTranspose 32 → BN → ReLU, UpSample | `3×3×3`, stride 1 → upsample `2×2×2` | Expands the spatial grid while blending neighbouring context. |
| | Conv3D 24 → BN → ReLU, UpSample | `3×3×3`, stride 1 → upsample `2×2×2` | Restores the original `(60,40,30)` resolution. |
| | ZeroPadding3D depth + Conv3D 16 → BN → ReLU | `3×3×3`, stride 1 | Aligns depth and produces a refined shared feature map. |
| Output heads | Conv3D (velocity/heat flux) | `1×1×1`, stride 1 | Acts like a per-voxel linear layer that mixes the 16-channel features into 3 bounded components (tanh). |
| | Conv3D (density/energy/temperature) | `1×1×1`, stride 1 | Same idea but outputs a single positive scalar per voxel via softplus. |

Why both kernel sizes? The `3×3×3` convolutions gather information from a voxel and its neighbours, so the receptive field grows as the network deepens. Once that shared feature map is built, the `1×1×1` convolutions simply apply a physics-aware activation to each voxel independently—no extra spatial mixing is needed because all the context has already been captured upstream.

**Outputs:**

   ```python
   velocity   = Conv3D(3, (1, 1, 1), activation="tanh",     name="velocity")(shared)
   heat_flux  = Conv3D(3, (1, 1, 1), activation="tanh",     name="heat_flux")(shared)
   density    = Conv3D(1, (1, 1, 1), activation="softplus", name="density")(shared)
   energy     = Conv3D(1, (1, 1, 1), activation="softplus", name="energy")(shared)
   temperature= Conv3D(1, (1, 1, 1), activation="softplus", name="temperature")(shared)
   ```

Every convolutional layer in the encoder/decoder uses a `3×3×3` receptive field (except the `1×1×1` output heads). Softplus activations enforce positive density/energy/temperature, while tanh bounds velocity and heat flux. The `ActivationPhysicsMetrics` callback samples validation batches each epoch to verify these constraints stay satisfied.

## Directory Layout ( abridged )

```text
mlForLBM/
├── README.md
├── isothermal_cracks.inp
├── marbles3d.gnu.TPROF.MPI.ex
├── generate_training_data_fixed.py
├── process_training_data_for_ml.py
├── train_lbm_neural_network_physics_aware.py
├── validate_neural_network_physics_aware.py
├── validate_seed6_minimal_physics_aware.py
├── predict_and_visualize_physics_aware.py
├── convert_nn_to_vtu_physics_aware.py
├── training_data/
├── ml_training_data/
├── validation_physics_aware/
├── validation_seed6_physics_aware/
└── custom_predictions_physics_aware/
```

## Utilities

- `validate_physics_aware_model.py` checks trained checkpoints for activation-induced physics violations.
- `preview_training_data.py` and `verify_training_data.py` offer quick sanity checks on generated datasets.
- `cleanup_training_data.py` removes intermediate AMReX artifacts if storage becomes a concern.

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
  keywords={machine learning, computational fluid dynamics, lattice boltzmann method, physics-aware neural networks, thermal flow}
}
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License**. You may use, share, and adapt the materials for non-commercial purposes with attribution to **Nilesh Sawant**. Commercial use is not permitted. See <https://creativecommons.org/licenses/by-nc/4.0/> for details.

## Contact

- **Author**: Nilesh Sawant
- **GitHub**: <https://github.com/nileshsawant>
- **Project**: <https://github.com/nileshsawant/mlForLBM>

## Acknowledgments

- AMReX (`marblesThermal`) for the reference LBM solver.
- TensorFlow / Keras for model development.
- `yt-project` for handling AMReX plotfiles.
- ParaView for visualization of VTU outputs.
