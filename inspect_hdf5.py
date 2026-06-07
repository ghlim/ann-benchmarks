import h5py
import os

# Look at one file to understand structure
filepath = 'results/sift-128-euclidean/10/hnswlib/euclidean_M_12_efConstruction_500_10.hdf5'

print(f"Inspecting: {filepath}\n")

with h5py.File(filepath, 'r') as f:
    print("Attributes:")
    for key, value in f.attrs.items():
        print(f"  {key}: {value}")
    
    print("\nDatasets:")
    for key in f.keys():
        dataset = f[key]
        print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        if dataset.shape[0] <= 10:
            print(f"    Data: {dataset[:]}")
