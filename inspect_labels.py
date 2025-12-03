"""
Inspect segmentation labels in HDF5 training data.
"""
import h5py
import numpy as np
import sys

def inspect_labels(h5_path):
    """Check the range and distribution of labels"""
    print(f"Inspecting: {h5_path}\n")

    with h5py.File(h5_path, 'r') as h5f:
        seg_data = h5f['segmentation']
        total_samples = seg_data.shape[0]

        print(f"Total samples: {total_samples}")
        print(f"Image shape: {seg_data.shape[1:]}")

        # Sample a subset to check label distribution
        sample_size = min(100, total_samples)
        print(f"\nSampling {sample_size} images to check labels...")

        unique_labels = set()
        for i in range(0, total_samples, total_samples // sample_size):
            labels = seg_data[i]
            unique_labels.update(np.unique(labels))

        unique_labels = sorted(unique_labels)

        print(f"\nUnique label values found: {unique_labels}")
        print(f"Min label: {min(unique_labels)}")
        print(f"Max label: {max(unique_labels)}")
        print(f"Number of unique labels: {len(unique_labels)}")

        # CARLA semantic segmentation classes
        carla_classes = {
            0: "Unlabeled",
            1: "Building",
            2: "Fence",
            3: "Other",
            4: "Pedestrian",
            5: "Pole",
            6: "RoadLine",
            7: "Road",
            8: "SideWalk",
            9: "Vegetation",
            10: "Vehicles",
            11: "Wall",
            12: "TrafficSign",
            13: "Sky",
            14: "Ground",
            15: "Bridge",
            16: "RailTrack",
            17: "GuardRail",
            18: "TrafficLight",
            19: "Static",
            20: "Dynamic",
            21: "Water",
            22: "Terrain"
        }

        print("\nLabel distribution:")
        for label in unique_labels:
            class_name = carla_classes.get(int(label), "Unknown")
            print(f"  {label}: {class_name}")

        print("\n" + "="*70)
        print("RECOMMENDATION:")
        if max(unique_labels) >= 13:
            print(f"Your data contains labels up to {max(unique_labels)}, but your model")
            print("is configured for only 13 classes.")
            print("\nOptions:")
            print("1. Change model to use num_classes=23 (or 24 to be safe)")
            print("2. Remap labels to consolidate into 13 classes")
            print("3. Use ignore_index in loss for labels >= 13")
        else:
            print("Labels are within range for 13 classes!")
        print("="*70)

if __name__ == '__main__':
    h5_path = sys.argv[1] if len(sys.argv) > 1 else './training_data/training_data.h5'
    inspect_labels(h5_path)
