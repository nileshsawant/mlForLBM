#!/usr/bin/env python3
"""
Neural Network Prediction Viewer

Load and examine the 3D neural network predictions saved during validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_nn_predictions(case_name):
    """Load neural network predictions for a specific case"""
    prediction_file = f"validation/neural_network_predictions/{case_name}_nn_predictions.npz"
    
    if not os.path.exists(prediction_file):
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
    
    data = np.load(prediction_file, allow_pickle=True)
    
    print(f" Loaded NN predictions: {case_name}")
    print(f"   Timesteps: {len(data['timesteps'])}")
    print(f"   Velocity shape: {data['velocity_fields'].shape}")
    print(f"   Heat flux shape: {data['heat_flux_fields'].shape}")
    print(f"   Density shape: {data['density_fields'].shape}")
    print(f"   Energy shape: {data['energy_fields'].shape}")
    print(f"   Temperature shape: {data['temperature_fields'].shape}")
    
    return data

def visualize_field_slice(field_data, field_name, timestep=50, slice_axis='z', slice_idx=15):
    """Visualize a 2D slice of a 3D field"""
    
    if len(field_data.shape) == 5:  # Vector field (velocity, heat flux)
        if field_name == 'velocity':
            # Show velocity magnitude
            field_slice = np.sqrt(np.sum(field_data[timestep, :, :, :, :]**2, axis=-1))
        elif field_name == 'heat_flux':
            # Show heat flux magnitude  
            field_slice = np.sqrt(np.sum(field_data[timestep, :, :, :, :]**2, axis=-1))
    else:  # Scalar field
        field_slice = field_data[timestep, :, :, :, 0]  # Remove channel dimension
    
    # Extract 2D slice
    if slice_axis == 'z':
        slice_2d = field_slice[:, :, slice_idx]
        xlabel, ylabel = 'Y', 'X'
    elif slice_axis == 'y':
        slice_2d = field_slice[:, slice_idx, :]
        xlabel, ylabel = 'Z', 'X'
    elif slice_axis == 'x':
        slice_2d = field_slice[slice_idx, :, :]
        xlabel, ylabel = 'Z', 'Y'
    
    plt.figure(figsize=(10, 8))
    plt.imshow(slice_2d, origin='lower', cmap='viridis')
    plt.colorbar(label=field_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{field_name.title()} - Timestep {timestep}, {slice_axis.upper()}={slice_idx}')
    plt.show()

def compare_timesteps(field_data, field_name, timesteps=[0, 25, 50, 75, 100]):
    """Compare field evolution over multiple timesteps"""
    fig, axes = plt.subplots(1, len(timesteps), figsize=(20, 4))
    
    for i, t in enumerate(timesteps):
        if t >= field_data.shape[0]:
            continue
            
        if len(field_data.shape) == 5:  # Vector field
            field_slice = np.sqrt(np.sum(field_data[t, :, :, 15, :]**2, axis=-1))
        else:  # Scalar field
            field_slice = field_data[t, :, :, 15, 0]
        
        im = axes[i].imshow(field_slice, origin='lower', cmap='viridis')
        axes[i].set_title(f't={t}')
        axes[i].set_xlabel('Y')
        axes[i].set_ylabel('X')
        plt.colorbar(im, ax=axes[i])
    
    fig.suptitle(f'{field_name.title()} Evolution (Z=15 slice)')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to explore neural network predictions"""
    
    print(" Neural Network Prediction Viewer")
    print("=" * 50)
    
    # List available prediction files
    pred_dir = "validation/neural_network_predictions"
    if not os.path.exists(pred_dir):
        print(" No neural network predictions found!")
        print("   Run validation script first.")
        return
    
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('_nn_predictions.npz')]
    
    if not pred_files:
        print(" No prediction files found!")
        return
    
    print(f" Available prediction files:")
    for f in pred_files:
        print(f"   {f}")
    
    # Load first case as example
    case_name = pred_files[0].replace('_nn_predictions.npz', '')
    data = load_nn_predictions(case_name)
    
    print(f"\\n Sample Statistics for {case_name}:")
    print("-" * 40)
    
    # Show field statistics
    fields = {
        'velocity': data['velocity_fields'],
        'heat_flux': data['heat_flux_fields'],
        'density': data['density_fields'], 
        'energy': data['energy_fields'],
        'temperature': data['temperature_fields']
    }
    
    for field_name, field_data in fields.items():
        print(f"{field_name.title():>12}: min={field_data.min():.6f}, max={field_data.max():.6f}, mean={field_data.mean():.6f}")
    
    print(f"\\n To visualize specific fields, use:")
    print(f"   data = load_nn_predictions('{case_name}')")
    print(f"   visualize_field_slice(data['velocity_fields'], 'velocity')")
    print(f"   compare_timesteps(data['temperature_fields'], 'temperature')")

if __name__ == "__main__":
    main()