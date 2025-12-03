"""
Verify that Static, TTDA, and CoTTA models are truly separate.
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

# Get first layer weights from each model
static_model = methods['Static'][0]
ttda_model = methods['TTDA'][0].model if hasattr(methods['TTDA'][0], 'model') else methods['TTDA'][0]
cotta_model = methods['CoTTA'][0].model if hasattr(methods['CoTTA'][0], 'model') else methods['CoTTA'][0]

# Check if models share memory
print("\nChecking weight memory addresses:")
for name, model in [('Static', static_model), ('TTDA', ttda_model), ('CoTTA', cotta_model)]:
    first_param = next(model.parameters())
    print(f"{name}: {id(first_param.data)}")

# Check if weights are identical (they should be initially)
static_param = next(static_model.parameters())
ttda_param = next(ttda_model.parameters())
cotta_param = next(cotta_model.parameters())

print("\nInitial weight equality:")
print(f"Static == TTDA: {torch.allclose(static_param, ttda_param)}")
print(f"Static == CoTTA: {torch.allclose(static_param, cotta_param)}")
print(f"TTDA == CoTTA: {torch.allclose(ttda_param, cotta_param)}")

# Check training/eval mode
print("\nModel modes:")
print(f"Static training: {static_model.training}")
print(f"TTDA training: {ttda_model.training}")
print(f"CoTTA training: {cotta_model.training}")

# Check if BatchNorm is in eval mode for Static
print("\nStatic BatchNorm status:")
for name, module in static_model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        print(f"  {name}: training={module.training}")
        break
