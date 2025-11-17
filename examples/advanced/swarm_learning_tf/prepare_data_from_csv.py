import os
import shutil
import sys
from math import isclose

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <iteration> <num_clients> <weight1> <weight2> ...")
        sys.exit(1)

    iteration = sys.argv[1]
    num_clients = int(sys.argv[2])
    weights = [float(w) for w in sys.argv[3:3+num_clients]]
    
    sum_is_one = isclose(sum(weights), 100.0, abs_tol=1e-1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not sum_is_one:
        source_range = range(2, num_clients + 2)
    else:
        weights.insert(0, weights.pop())
        source_range = range(1, num_clients + 1)

    for dest_idx, source_num in enumerate(source_range, start=1):
        try:
            weight_idx = source_num - 2 if not sum_is_one else source_num - 1
            
            if weight_idx >= len(weights) or weight_idx < 0:
                print(f"WARNING: Missing weight for client {source_num}")
                continue
                
            weight = weights[weight_idx]
            if weight.is_integer():
                weight = int(weight)
            
            # Path sorgente (usa sempre la numerazione originale)
            num_nodes = num_clients - 1 if sum_is_one else num_clients
            source_file = os.path.join(
                script_dir, "..", "mimic", "dataset", 
                f"mimiciv_{num_nodes}_nodes", 
                f"{source_num}_{weight}_{iteration}.csv"
            )
            
            # Path destinazione (rinumerato da 1)
            dest_file = f"/tmp/mimic_data/site-{dest_idx}.csv"
            
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            
            if os.path.isfile(source_file):
                shutil.copy(source_file, dest_file)
                # print(f"Copied {source_file} to {dest_file}")
            else:
                print(f"WARNING: Source file {source_file} does not exist!")
        except Exception as e:
            print(f"ERROR processing client {source_num}: {str(e)}")

if __name__ == "__main__":
    main()