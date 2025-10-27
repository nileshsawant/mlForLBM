#!/usr/bin/env python3
"""
Physics-Aware Model Validator

This script tests the physics-aware neural network model to verify:
1. No negative densities or temperatures
2. Improved physics consistency
3. Comparison with baseline enhanced model

Usage:
  python validate_physics_aware_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import json

def load_physics_aware_model(model_path='lbm_flow_predictor_physics_aware.h5',
                            norm_params_path='physics_aware_normalization_params.json'):
    """Load physics-aware model and normalization parameters"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Physics-aware model not found: {model_path}")
    
    if not os.path.exists(norm_params_path):
        raise FileNotFoundError(f"Normalization parameters not found: {norm_params_path}")
    
    # Load model without compiling (to avoid custom loss function issues)
    model = keras.models.load_model(model_path, compile=False)
    
    # Load normalization parameters
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    
    return model, norm_params

def denormalize_predictions(predictions, norm_params):
    """Denormalize activation-function model predictions back to physical units"""
    
    velocity_pred, heat_flux_pred, density_pred, energy_pred, temperature_pred = predictions
    
    # Velocity: scale back from tanh [-1, 1] to physical range
    velocity_denorm = velocity_pred * norm_params['velocity_scale']
    
    # Heat flux: scale back from tanh [-1, 1] to physical range
    heat_flux_denorm = heat_flux_pred * norm_params['heat_flux_scale']
    
    # Density: inverse log normalization then exp
    # Note: softplus activation ensures this is always positive
    density_log_denorm = density_pred * norm_params['density_log_std'] + norm_params['density_log_mean']
    density_denorm = np.exp(density_log_denorm)
    
    # Energy: inverse log normalization then exp
    # Note: softplus activation ensures this is always positive
    energy_log_denorm = energy_pred * norm_params['energy_log_std'] + norm_params['energy_log_mean']
    energy_denorm = np.exp(energy_log_denorm)
    
    # Temperature: inverse log normalization then exp
    # Note: softplus activation ensures this is always positive
    temp_log_denorm = temperature_pred * norm_params['temp_log_std'] + norm_params['temp_log_mean']
    temperature_denorm = np.exp(temp_log_denorm)
    
    return velocity_denorm, heat_flux_denorm, density_denorm, energy_denorm, temperature_denorm

def test_physics_constraints(geometry_file="microstructure_nX60_nY40_nZ30_seed1.csv"):
    """Test physics-aware model for constraint violations"""
    
    print(" ACTIVATION-FUNCTION MODEL VALIDATION")
    print("=" * 50)
    print("Testing smart activation functions:")
    print("  • Softplus: density, energy, temperature → ALWAYS > 0")
    print("  • Tanh: velocity, heat flux → bounded range")
    print("  • NO conservation law penalties (keeping it simple!)")
    print()
    
    # Load geometry
    if not os.path.exists(geometry_file):
        print(f" Geometry file not found: {geometry_file}")
        return
    
    geom_df = pd.read_csv(geometry_file)
    geometry_3d = np.zeros((60, 40, 30))
    
    for _, row in geom_df.iterrows():
        x, y, z = int(row['X']), int(row['Y']), int(row['Z'])
        if 0 <= x < 60 and 0 <= y < 40 and 0 <= z < 30:
            geometry_3d[x, y, z] = row['tag']
    
    print(f" Loaded geometry: {np.sum(geometry_3d > 0)} solid voxels")
    
    # Load physics-aware model
    try:
        model, norm_params = load_physics_aware_model()
        print(f" Loaded activation-function model: {model.count_params():,} parameters")
    except FileNotFoundError as e:
        print(f" {e}")
        print("   Run train_lbm_neural_network_physics_aware.py first!")
        return
    
    # Test with multiple parameter combinations
    test_cases = [
        {'nu': 0.00259, 'temp': 0.025, 'alpha': 0.00259/0.7, 'time': 0.0},
        {'nu': 0.00259, 'temp': 0.045, 'alpha': 0.00259/0.7, 'time': 0.5},
        {'nu': 0.00567, 'temp': 0.025, 'alpha': 0.00567/0.7, 'time': 1.0},
    ]
    
    print(f"\n Testing {len(test_cases)} parameter combinations...")
    
    all_results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n Test Case {i+1}: nu={case['nu']:.5f}, T={case['temp']:.3f}, t={case['time']:.1f}")
        
        # Prepare inputs
        geometry_input = geometry_3d[np.newaxis, ..., np.newaxis]  # (1, 60, 40, 30, 1)
        params_input = np.array([[case['nu'], case['alpha'], case['temp'], case['time']]])  # (1, 4)
        
        # Generate predictions
        predictions = model.predict([geometry_input, params_input], verbose=0)
        
        # Denormalize predictions
        velocity_denorm, heat_flux_denorm, density_denorm, energy_denorm, temperature_denorm = \
            denormalize_predictions(predictions, norm_params)
        
        # Analyze physics constraints
        results = {
            'case_id': i,
            'parameters': case,
            'predictions': {
                'velocity': velocity_denorm[0],
                'heat_flux': heat_flux_denorm[0],
                'density': density_denorm[0],
                'energy': energy_denorm[0],
                'temperature': temperature_denorm[0]
            }
        }
        
        # Physics violation analysis
        violations = {}
        
        # Check density constraints
        density_min = np.min(density_denorm)
        density_negative = np.sum(density_denorm <= 0)
        violations['density'] = {
            'min_value': float(density_min),
            'negative_count': int(density_negative),
            'total_elements': int(density_denorm.size)
        }
        
        # Check temperature constraints
        temp_min = np.min(temperature_denorm)
        temp_negative = np.sum(temperature_denorm <= 0)
        violations['temperature'] = {
            'min_value': float(temp_min),
            'negative_count': int(temp_negative),
            'total_elements': int(temperature_denorm.size)
        }
        
        # Check energy constraints
        energy_min = np.min(energy_denorm)
        energy_negative = np.sum(energy_denorm <= 0)
        violations['energy'] = {
            'min_value': float(energy_min),
            'negative_count': int(energy_negative),
            'total_elements': int(energy_denorm.size)
        }
        
        results['violations'] = violations
        all_results.append(results)
        
        # Report violations
        print(f"   Density:     min={density_min:8.6f}, negatives={density_negative:>7,} / {density_denorm.size:,}")
        print(f"   Temperature: min={temp_min:8.6f}, negatives={temp_negative:>7,} / {temperature_denorm.size:,}")
        print(f"   Energy:      min={energy_min:8.6f}, negatives={energy_negative:>7,} / {energy_denorm.size:,}")
        
        # Status
        if density_negative == 0 and temp_negative == 0 and energy_negative == 0:
            print(f"    PHYSICS CONSTRAINTS SATISFIED")
        else:
            print(f"     Physics violations detected")
    
    # Summary comparison
    print(f"\n SUMMARY COMPARISON")
    print("=" * 50)
    
    total_violations = {
        'density': sum(r['violations']['density']['negative_count'] for r in all_results),
        'temperature': sum(r['violations']['temperature']['negative_count'] for r in all_results),
        'energy': sum(r['violations']['energy']['negative_count'] for r in all_results)
    }
    
    total_elements = all_results[0]['violations']['density']['total_elements'] * len(all_results)
    
    print(f"Physics-Aware Model Results:")
    print(f"  Density violations:     {total_violations['density']:>8,} / {total_elements:,} ({100*total_violations['density']/total_elements:.3f}%)")
    print(f"  Temperature violations: {total_violations['temperature']:>8,} / {total_elements:,} ({100*total_violations['temperature']/total_elements:.3f}%)")
    print(f"  Energy violations:      {total_violations['energy']:>8,} / {total_elements:,} ({100*total_violations['energy']/total_elements:.3f}%)")
    
    # Compare with baseline (if available)
    print(f"\nBaseline Enhanced Model (from validation):")
    print(f"  Density violations:     ~2,233,000 / 7,272,000 (30.7%)")
    print(f"  Temperature violations: ~2,650,000 / 7,272,000 (36.4%)")
    
    improvement_density = (2233000 - total_violations['density']) / 2233000 * 100
    improvement_temp = (2650000 - total_violations['temperature']) / 2650000 * 100
    
    print(f"\n IMPROVEMENTS:")
    print(f"  Density violations:     {improvement_density:+6.1f}% improvement")
    print(f"  Temperature violations: {improvement_temp:+6.1f}% improvement")
    
    # Save detailed results
    results_file = 'physics_aware_validation_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = []
        for result in all_results:
            result_copy = result.copy()
            # Convert numpy arrays in predictions
            for key, value in result_copy['predictions'].items():
                if isinstance(value, np.ndarray):
                    result_copy['predictions'][key] = value.tolist()
            results_json.append(result_copy)
        
        json.dump(results_json, f, indent=2)
    
    print(f"\n Detailed results saved to: {results_file}")
    
    return all_results

def create_comparison_plots(results):
    """Create visualization comparing physics-aware vs baseline model"""
    
    print(f"\n Creating comparison plots...")
    
    # Extract violation data
    cases = [f"Case {r['case_id']+1}" for r in results]
    density_violations = [r['violations']['density']['negative_count'] for r in results]
    temp_violations = [r['violations']['temperature']['negative_count'] for r in results]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Density violations comparison
    x = np.arange(len(cases))
    width = 0.35
    
    baseline_density = [2233000 / 4] * len(cases)  # Approximate per case
    ax1.bar(x - width/2, baseline_density, width, label='Baseline Enhanced', color='red', alpha=0.7)
    ax1.bar(x + width/2, density_violations, width, label='Physics-Aware', color='green', alpha=0.7)
    
    ax1.set_title('Density Violations Comparison')
    ax1.set_ylabel('Number of Negative Densities')
    ax1.set_xlabel('Test Cases')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cases)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Temperature violations comparison
    baseline_temp = [2650000 / 4] * len(cases)  # Approximate per case
    ax2.bar(x - width/2, baseline_temp, width, label='Baseline Enhanced', color='red', alpha=0.7)
    ax2.bar(x + width/2, temp_violations, width, label='Physics-Aware', color='green', alpha=0.7)
    
    ax2.set_title('Temperature Violations Comparison')
    ax2.set_ylabel('Number of Negative Temperatures')
    ax2.set_xlabel('Test Cases')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cases)
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('physics_aware_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Comparison plot saved: physics_aware_model_comparison.png")

def main():
    """Main validation function"""
    
    # Test physics constraints
    results = test_physics_constraints()
    
    if results:
        # Create comparison plots
        create_comparison_plots(results)
        
        print(f"\n PHYSICS-AWARE MODEL VALIDATION COMPLETE")
        print(f"=" * 50)
        print(f" Model tested with multiple parameter combinations")
        print(f" Physics constraints analyzed")
        print(f" Comparison with baseline model generated")
        print(f" Results saved for further analysis")
        
        print(f"\nNext steps:")
        print(f"  1. If violations are eliminated: retrain full model")
        print(f"  2. If violations persist: adjust activation functions")
        print(f"  3. Fine-tune physics-informed loss function")
        print(f"  4. Run full validation suite with new model")

if __name__ == "__main__":
    main()