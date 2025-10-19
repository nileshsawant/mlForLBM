#!/usr/bin/env python3
"""
Quick preview script to check training data status before ML conversion
"""

import os
import glob

def preview_training_data(training_data_dir="training_data"):
    """
    Preview the current state of training data
    """
    print("Training Data Preview")
    print("=" * 50)
    
    # Find all case directories
    case_pattern = os.path.join(training_data_dir, "case_*")
    case_dirs = sorted(glob.glob(case_pattern))
    
    if not case_dirs:
        print(f" No case directories found in {training_data_dir}")
        return
    
    print(f"Found {len(case_dirs)} case directories")
    
    # Analyze a few representative cases
    sample_cases = [case_dirs[0], case_dirs[len(case_dirs)//2], case_dirs[-1]]
    
    total_plt_files = 0
    cases_with_data = 0
    
    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir)
        
        # Count plt files
        plt_files = glob.glob(os.path.join(case_dir, "plt*"))
        plt_dirs = [f for f in plt_files if os.path.isdir(f)]
        
        if plt_dirs:
            cases_with_data += 1
            total_plt_files += len(plt_dirs)
    
    print(f"\nData Summary:")
    print(f"  Cases with simulation data: {cases_with_data}/{len(case_dirs)}")
    print(f"  Total plt files: {total_plt_files}")
    print(f"  Average plt files per case: {total_plt_files/max(cases_with_data,1):.1f}")
    
    print(f"\nSample Cases:")
    for case_dir in sample_cases:
        case_name = os.path.basename(case_dir)
        
        # Check files in directory
        files = os.listdir(case_dir)
        plt_files = [f for f in files if f.startswith('plt') and os.path.isdir(os.path.join(case_dir, f))]
        
        print(f"  {case_name}:")
        print(f"    plt files: {len(plt_files)}")
        if plt_files:
            print(f"    Latest: {sorted(plt_files)[-1]}")
        
        # Show parameter info from filename
        parts = case_name.split('_')
        if len(parts) >= 8:
            nu_idx = parts[3]
            temp_idx = parts[5] 
            geom_id = parts[7]
            print(f"    Parameters: nu_{nu_idx}, temp_{temp_idx}, geom_{geom_id}")
    
    if cases_with_data == len(case_dirs):
        print(f"\n All cases have simulation data - ready for ML conversion!")
        print(f"\nTo convert to ML format, run:")
        print(f"  conda activate marbles_ml")
        print(f"  python3 process_training_data_for_ml.py")
    elif cases_with_data == 0:
        print(f"\n No simulation data found - run batch simulations first")
        print(f"  python3 run_batch_simulations.py")
    else:
        print(f"\n  Partial data - {len(case_dirs)-cases_with_data} cases missing simulation results")

if __name__ == "__main__":
    preview_training_data()