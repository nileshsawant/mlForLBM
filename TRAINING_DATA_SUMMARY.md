# Training Data Generation Summary

## Overview
Successfully generated **125 training cases** for Marbles LBM thermal simulation neural network training.

## Structure
- **5 viscosities** (lbm.nu): 3.238e-03 to 7.286e-03 (reference ± 50%)
- **5 temperatures** (lbm.body_temperature): 0.044 to 0.099 (reference ± 50%) 
- **5 geometries** (generated with genCracks 1-5)
- **Total cases**: 5 × 5 × 5 = 125

## Reference Values
From `isothermal_cracks.inp`:
- `lbm.nu = 4.857e-3` (viscosity)
- `lbm.body_temperature = 0.066` (isothermal body temperature)
- `lbm.alpha = lbm.nu/0.7` (thermal diffusivity, always coupled to viscosity)

## Directory Structure
```
training_data/
├── case_000_nu_0_temp_0_geom_1/
│   ├── isothermal_cracks.inp              # Simulation parameters
│   ├── microstructure_geom_1.csv          # Geometry file (738KB)
│   └── marbles3d.gnu.TPROF.MPI.ex         # Executable copy (4.3MB)
├── case_001_nu_0_temp_0_geom_2/
│   ├── isothermal_cracks.inp
│   ├── microstructure_geom_2.csv
│   └── marbles3d.gnu.TPROF.MPI.ex
...
└── case_124_nu_4_temp_4_geom_5/
    ├── isothermal_cracks.inp
    ├── microstructure_geom_5.csv
    └── marbles3d.gnu.TPROF.MPI.ex
```

## Parameter Sweeps

### Viscosity Values (lbm.nu)
| Index | Value (m²/s) | Ratio to Reference |
|-------|--------------|-------------------|
| 0     | 3.238e-03    | 0.67× (1/1.5)     |
| 1     | 4.250e-03    | 0.87×             |
| 2     | 5.262e-03    | 1.08×             |
| 3     | 6.274e-03    | 1.29×             |
| 4     | 7.286e-03    | 1.50×             |

### Temperature Values (lbm.body_temperature)
| Index | Value | Ratio to Reference |
|-------|-------|-------------------|
| 0     | 0.044 | 0.67× (1/1.5)     |
| 1     | 0.058 | 0.87×             |
| 2     | 0.072 | 1.08×             |
| 3     | 0.085 | 1.29×             |
| 4     | 0.099 | 1.50×             |

### Geometry IDs
- **geom_1**: Generated with `genCracks 1` (seed=1)
- **geom_2**: Generated with `genCracks 2` (seed=2)  
- **geom_3**: Generated with `genCracks 3` (seed=3)
- **geom_4**: Generated with `genCracks 4` (seed=4)
- **geom_5**: Generated with `genCracks 5` (seed=5)

Each geometry represents a different crack microstructure in a 60×40×30 domain.

## Case Naming Convention
`case_{id:03d}_nu_{nu_index}_temp_{temp_index}_geom_{geom_id}`

Examples:
- `case_000_nu_0_temp_0_geom_1`: Lowest viscosity, lowest temperature, geometry 1
- `case_062_nu_2_temp_2_geom_3`: Middle values, geometry 3  
- `case_124_nu_4_temp_4_geom_5`: Highest viscosity, highest temperature, geometry 5

## File Sizes
- **Each case directory**: ~5.0 MB
  - Input file: ~2.5 KB
  - Geometry file: ~738 KB  
  - Executable: ~4.3 MB
- **Total training_data/**: ~625 MB (125 cases × 5 MB)

## Next Steps

### To run a single simulation:
```bash
cd training_data/case_000_nu_0_temp_0_geom_1/
mpirun -np 4 ./marbles3d.gnu.TPROF.MPI.ex isothermal_cracks.inp
```

### To convert output for ML training:
```bash
python3 ../../convert_amrex_to_ml.py plt00010 output_case_000.npz
```

### Batch simulation execution:
Consider creating a script to:
1. Loop through all 125 cases
2. Run simulation in each directory  
3. Convert output to ML format
4. Collect all outputs for neural network training

## Generation Script
The training data was generated using `generate_training_data_fixed.py` with the function:
```python
generate_all_training_data('isothermal_cracks.inp')
```

This process:
1. Reads reference parameters from `isothermal_cracks.inp`
2. Generates 5×5×5 parameter combinations
3. Runs `genCracks 1-5` to create geometries
4. Creates 125 case directories with modified input files
5. Copies appropriate geometry and executable to each case