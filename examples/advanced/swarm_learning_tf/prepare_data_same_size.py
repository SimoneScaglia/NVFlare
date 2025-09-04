import os
import shutil
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <iteration> <num_clients>")
        sys.exit(1)

    iteration = sys.argv[1]
    num_clients = int(sys.argv[2])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, "..", "mimic", "dataset", "mimiciv_same_size")
    dest_dir = "/tmp/mimic_data"
    os.makedirs(dest_dir, exist_ok=True)

    for i in range(1, num_clients + 1):
        source_file = os.path.join(source_dir, f"node{i}_{iteration}.csv")
        dest_file = os.path.join(dest_dir, f"site-{i}.csv")

        try:
            if os.path.isfile(source_file):
                shutil.copy(source_file, dest_file)
                print(f"Copied {source_file} to {dest_file}")
            else:
                print(f"WARNING: Source file {source_file} does not exist!")
        except Exception as e:
            print(f"ERROR processing node {i}: {str(e)}")

if __name__ == "__main__":
    main()
