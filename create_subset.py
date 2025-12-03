"""
Create a subset of training data that fits in memory.
"""
import h5py
import numpy as np
import sys
from pathlib import Path

def create_subset(input_path, output_path, max_samples=30000):
    """
    Create a smaller HDF5 file with subset of data
    """
    print(f"Reading from: {input_path}")
    print(f"Creating subset with {max_samples} samples...")

    with h5py.File(input_path, 'r') as h5f_in:
        total_samples = h5f_in['rgb'].shape[0]

        if total_samples <= max_samples:
            print(f"Input has only {total_samples} samples, no need to subset")
            return

        # Take evenly distributed samples
        indices = np.linspace(0, total_samples-1, max_samples, dtype=int)

        with h5py.File(output_path, 'w') as h5f_out:
            # Create datasets
            rgb_dset = h5f_out.create_dataset(
                'rgb',
                shape=(max_samples, 600, 800, 3),
                dtype=np.uint8,
                chunks=(100, 600, 800, 3),
                compression='gzip',
                compression_opts=1
            )

            seg_dset = h5f_out.create_dataset(
                'segmentation',
                shape=(max_samples, 600, 800),
                dtype=np.uint8,
                chunks=(100, 600, 800),
                compression='gzip',
                compression_opts=1
            )

            # Copy data in batches
            batch_size = 1000
            for i in range(0, max_samples, batch_size):
                end_idx = min(i + batch_size, max_samples)
                batch_indices = indices[i:end_idx]

                rgb_dset[i:end_idx] = h5f_in['rgb'][batch_indices]
                seg_dset[i:end_idx] = h5f_in['segmentation'][batch_indices]

                print(f"  Copied {end_idx}/{max_samples} samples", end='\r')

            print(f"\n  ✓ Copied {max_samples} samples")

            # Copy metadata
            for key, value in h5f_in.attrs.items():
                h5f_out.attrs[key] = value

    output_size = Path(output_path).stat().st_size / (1024**3)
    print(f"\nCreated: {output_path}")
    print(f"Size: {output_size:.2f} GB")
    print(f"This should fit in memory for fast training!")

if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else './training_data/training_data.h5'
    output_path = sys.argv[2] if len(sys.argv) > 2 else './training_data/training_data_subset.h5'
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 30000

    create_subset(input_path, output_path, max_samples)
