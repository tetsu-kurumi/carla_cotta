"""
Check the status of training data collection.
"""
import h5py
from pathlib import Path
import sys

def check_h5_file(h5_path):
    """Check the contents and status of an HDF5 file"""
    h5_path = Path(h5_path)

    if not h5_path.exists():
        print(f"File not found: {h5_path}")
        return

    file_size_gb = h5_path.stat().st_size / (1024**3)
    print(f"\nFile: {h5_path}")
    print(f"Size: {file_size_gb:.2f} GB")
    print("="*70)

    with h5py.File(h5_path, 'r') as h5f:
        # Print datasets
        print("\nDatasets:")
        for key in h5f.keys():
            dataset = h5f[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")

        # Print attributes
        print("\nMetadata:")
        for key, value in h5f.attrs.items():
            if key == 'weathers':
                print(f"  {key}: {list(value)}")
            else:
                print(f"  {key}: {value}")

        # Check completion status
        if 'completed_weathers' in h5f.attrs:
            completed = set(h5f.attrs['completed_weathers'])
            all_weathers = list(h5f.attrs['weathers']) if 'weathers' in h5f.attrs else []
            remaining = [w for w in all_weathers if w not in completed]

            print(f"\nProgress:")
            print(f"  Completed: {len(completed)}/{len(all_weathers)} weather conditions")
            print(f"  Total frames: {h5f['rgb'].shape[0]}")

            if completed:
                print(f"\n  ✓ Completed weathers:")
                for weather in completed:
                    print(f"    - {weather}")

            if remaining:
                print(f"\n  ⏳ Remaining weathers:")
                for weather in remaining:
                    print(f"    - {weather}")
            else:
                print("\n  🎉 All weather conditions completed!")
        else:
            print("\n⚠️  No completion tracking found (old format)")

    print("="*70)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        h5_path = './training_data/training_data.h5'
    else:
        h5_path = sys.argv[1]

    check_h5_file(h5_path)
