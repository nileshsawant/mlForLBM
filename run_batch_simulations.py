#!/usr/bin/env python3
"""
Single command to run all 125 training simulations
"""

from generate_training_data_fixed import run_all_simulations

if __name__ == "__main__":
    print(" Starting batch simulation of all 125 training cases...")
    print("Command: ./marbles3d.gnu.TPROF.MPI.ex isothermal_cracks.inp")
    print("=" * 60)
    
    successful = run_all_simulations()
    
    print("\n" + "=" * 60)
    if successful == 125:
        print(" SUCCESS: All 125 simulations completed!")
        print("Ready for ML data conversion and neural network training.")
    else:
        print(f"  PARTIAL SUCCESS: {successful}/125 simulations completed.")
        print("Check the output above for any failed cases.")