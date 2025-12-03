"""
Tent for Semantic Segmentation
Simpler baseline compared to CoTTA - only updates BatchNorm parameters.
"""

from copy import deepcopy
import torch
import torch.nn as nn


class TentSegmentation(nn.Module):
    """
    Tent adapts a segmentation model by entropy minimization during testing.
    Simpler than CoTTA - only updates BatchNorm layers.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "Tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # Save initial state for reset
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data."""
        # Forward
        outputs = model(x)

        # Handle different model output formats
        if isinstance(outputs, dict):
            # SegFormer returns {'logits': tensor}
            if 'logits' in outputs:
                logits = outputs['logits']
            # DeepLabV3 returns {'out': tensor}
            elif 'out' in outputs:
                logits = outputs['out']
            else:
                # Fallback for unknown dict format
                logits = outputs
        else:
            # Direct tensor output (Fast-SCNN)
            logits = outputs

        # Calculate entropy loss (per-pixel)
        loss = softmax_entropy(logits).mean()

        # Adapt
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return outputs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """
    Entropy of softmax distribution from logits.
    For segmentation: [B, C, H, W] -> [B, H, W]
    """
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """
    Collect only BatchNorm parameters (Tent's approach).

    This is more conservative than CoTTA which updates all parameters.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """
    Configure model for use with Tent.
    Only BatchNorm layers are set to trainable.
    """
    # Train mode for most layers
    model.train()

    # Disable grad for all parameters
    model.requires_grad_(False)

    # Enable grad only for BatchNorm parameters
    # But keep BatchNorm in eval mode to avoid batch size issues
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # Enable gradient computation for parameters
            m.requires_grad_(True)
            # But keep in eval mode to avoid "Expected more than 1 value" error
            # This is common practice in test-time adaptation
            m.eval()
            # Use running stats (which get updated via gradient descent)
            m.track_running_stats = True

    return model


def check_model(model):
    """Check model compatibility with Tent."""
    is_training = model.training
    assert is_training, "Tent needs train mode: call model.train()"

    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)

    assert has_any_params, "Tent needs params to update: check which require grad"
    assert not has_all_params, "Tent should not update all params: check which require grad"

    has_bn = any([isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) for m in model.modules()])
    assert has_bn, "Tent needs normalization layers for its optimization"
