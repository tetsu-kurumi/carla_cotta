"""
Check what's in a checkpoint file.
"""
import torch
import sys

def check_checkpoint(path):
    """Check checkpoint contents"""
    print(f"Loading: {path}\n")

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        print("Checkpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")

        if 'epoch' in checkpoint:
            print(f"\nEpoch: {checkpoint['epoch']}")
        if 'miou' in checkpoint:
            print(f"mIoU: {checkpoint['miou']:.4f}")
        if 'train_loss' in checkpoint:
            print(f"Train Loss: {checkpoint['train_loss']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    else:
        print("Checkpoint is a state_dict (model weights only)")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else './checkpoints_new/best_model.pth'
    check_checkpoint(path)
