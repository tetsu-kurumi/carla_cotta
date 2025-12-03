"""
Debug model predictions.
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/tetsuk/Desktop/CARLA_0.9.16/cotta_proj')
from fast_scnn_pytorch import FastSCNN

# Load model
print("Loading model...")
model = FastSCNN(num_classes=28)
state_dict = torch.load('checkpoints_new/best_model.pth', weights_only=False)
model.load_state_dict(state_dict)
model.eval()
model = model.cuda()

# Create dummy input
print("\nCreating dummy input (256x512 RGB image)...")
dummy_input = torch.rand(1, 3, 256, 512).cuda()

# Forward pass
print("Running forward pass...")
with torch.no_grad():
    output = model(dummy_input)

print(f"\nOutput shape: {output.shape}")
print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

# Get predictions
pred = torch.argmax(output, dim=1).squeeze()
unique_preds = torch.unique(pred)

print(f"\nPrediction shape: {pred.shape}")
print(f"Unique predicted classes: {unique_preds.cpu().numpy()}")
print(f"Number of unique classes predicted: {len(unique_preds)}")

# Check if model is predicting reasonably
if len(unique_preds) == 1:
    print("\n⚠️  WARNING: Model is predicting only ONE class for the entire image!")
    print("This suggests the model might not be working correctly.")
else:
    print("\n✓ Model is predicting multiple classes (looks good)")
