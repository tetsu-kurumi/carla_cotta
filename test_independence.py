"""
Test if models are actually independent by modifying weights.
"""
import torch
from copy import deepcopy
import sys
sys.path.insert(0, '/home/tetsuk/Desktop/CARLA_0.9.16/cotta_proj')
from fast_scnn_pytorch import FastSCNN
from run_evaluation import setup_methods

# Load base model
print("Loading base model...")
base_model = FastSCNN(num_classes=28)
state_dict = torch.load('checkpoints_new/best_model.pth', weights_only=False)
base_model.load_state_dict(state_dict)

# Setup methods
print("Setting up methods...")
methods = setup_methods(base_model, device='cpu')

# Get models
static_model = methods['Static'][0]
ttda_wrapper = methods['TTDA'][0]
cotta_wrapper = methods['CoTTA'][0]

# Access internal models
ttda_model = ttda_wrapper.model
cotta_model = cotta_wrapper.model

# Get first parameter from each
static_param = next(static_model.parameters())
ttda_param = next(ttda_model.parameters())
cotta_param = next(cotta_model.parameters())

print(f"\nBefore modification:")
print(f"Static first value: {static_param.data.flatten()[0].item():.6f}")
print(f"TTDA first value: {ttda_param.data.flatten()[0].item():.6f}")
print(f"CoTTA first value: {cotta_param.data.flatten()[0].item():.6f}")

# Modify TTDA weights
print(f"\nModifying TTDA weights by adding 1.0...")
with torch.no_grad():
    ttda_param.data += 1.0

print(f"\nAfter modifying TTDA:")
print(f"Static first value: {static_param.data.flatten()[0].item():.6f}")
print(f"TTDA first value: {ttda_param.data.flatten()[0].item():.6f}")
print(f"CoTTA first value: {cotta_param.data.flatten()[0].item():.6f}")

if abs(static_param.data.flatten()[0].item() - (ttda_param.data.flatten()[0].item() - 1.0)) > 0.001:
    print("\n✓ Models are INDEPENDENT (good!)")
else:
    print("\n✗ Models are SHARING WEIGHTS (bug!)")
