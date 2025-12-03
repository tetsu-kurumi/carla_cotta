"""
CoTTA for Semantic Segmentation - Adapted for CARLA
Based on the CoTTA paper: https://arxiv.org/abs/2203.13591
"""

from copy import deepcopy
import torch
import torch.nn as nn
import PIL
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def get_tta_transforms_segmentation(soft=False):
    """
    Test-Time Augmentation for segmentation.
    Uses simple horizontal flip to avoid size inconsistencies.
    """
    # For segmentation TTA, we use only horizontal flip to maintain size consistency
    # More complex augmentations (rotation, scale) cause size mismatches
    class SegmentationTTA:
        def __init__(self, soft=False):
            self.soft = soft
            self.p_hflip = 0.5

        def __call__(self, x):
            # Random horizontal flip (preserves size)
            if torch.rand(1) < self.p_hflip:
                x = TF.hflip(x)

            # Color jitter (preserves size)
            brightness_factor = 0.9 + torch.rand(1).item() * 0.2 if self.soft else 0.8 + torch.rand(1).item() * 0.4
            contrast_factor = 0.9 + torch.rand(1).item() * 0.2 if self.soft else 0.8 + torch.rand(1).item() * 0.4

            x = TF.adjust_brightness(x, brightness_factor)
            x = TF.adjust_contrast(x, contrast_factor)

            return x

    return SegmentationTTA(soft=soft)


def update_ema_variables(ema_model, model, alpha_teacher):
    """Update EMA model parameters"""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTASegmentation(nn.Module):
    """
    CoTTA for semantic segmentation.
    Adapts a segmentation model during test-time using entropy minimization.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False,
                 mt_alpha=0.999, rst_m=0.01, ap=0.92, num_classes=13):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.num_classes = num_classes
        assert steps > 0, "CoTTA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # Copy model states
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # Test-time augmentation
        self.transform = get_tta_transforms_segmentation(soft=True)

        # Hyperparameters
        self.mt = mt_alpha  # EMA momentum (higher = more stable)
        self.rst = rst_m     # Stochastic restoration rate (lower = less restoration)
        self.ap = ap         # Anchor probability threshold

    def forward(self, x):
        """
        Forward pass with test-time adaptation.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output segmentation logits [B, num_classes, H, W]
        """
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        """Reset to initial model state"""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        """
        Forward pass with adaptation step.
        """
        # Helper function to extract logits from model output
        def extract_logits(output):
            if isinstance(output, dict):
                # SegFormer returns {'logits': tensor}
                if 'logits' in output:
                    return output['logits']
                # DeepLabV3 returns {'out': tensor}
                elif 'out' in output:
                    return output['out']
                else:
                    return output
            else:
                # Direct tensor output (Fast-SCNN)
                return output

        # Student prediction
        student_output = self.model(x)
        outputs = extract_logits(student_output)

        # Anchor prediction (original model)
        with torch.no_grad():
            anchor_out = self.model_anchor(x)
            anchor_out = extract_logits(anchor_out)
            anchor_prob = torch.nn.functional.softmax(anchor_out, dim=1).max(1)[0]

            # Standard EMA prediction
            standard_ema = self.model_ema(x)
            standard_ema = extract_logits(standard_ema)

        # Augmentation-averaged prediction (if low confidence)
        # Reduced from 32 to 8 for segmentation (computational efficiency)
        N = 8

        if anchor_prob.mean() < self.ap:
            # Use augmentation-averaged prediction
            outputs_emas = []
            with torch.no_grad():
                for i in range(N):
                    x_aug = self.transform(x)
                    out_aug = self.model_ema(x_aug)
                    out_aug = extract_logits(out_aug)
                    outputs_emas.append(out_aug)
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            # Use standard EMA prediction (high confidence)
            outputs_ema = standard_ema

        # Loss: cross-entropy between student and teacher
        # For segmentation, we compute per-pixel entropy
        loss = segmentation_entropy_loss(outputs, outputs_ema)

        # Backward and update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update teacher model (EMA)
        self.model_ema = update_ema_variables(
            ema_model=self.model_ema,
            model=self.model,
            alpha_teacher=self.mt
        )

        # Stochastic restoration (prevent catastrophic forgetting)
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < self.rst).float().to(p.device)
                    with torch.no_grad():
                        # Move model_state tensor to same device as parameter
                        original_param = self.model_state[f"{nm}.{npp}"].to(p.device)
                        p.data = original_param * mask + p * (1. - mask)

        # Return in the same format as the model output for compatibility
        # This ensures postprocess_output can handle SegFormer's lower-resolution output correctly
        if isinstance(student_output, dict):
            # Return dict with EMA logits
            result = {}
            for k, v in student_output.items():
                result[k] = v
            if 'logits' in result:
                result['logits'] = outputs_ema
            elif 'out' in result:
                result['out'] = outputs_ema
            return result
        else:
            # Return tensor directly
            return outputs_ema


def segmentation_entropy_loss(student_logits, teacher_logits):
    """
    Compute entropy loss for segmentation.

    Args:
        student_logits: [B, C, H, W] - student model output
        teacher_logits: [B, C, H, W] - teacher model output

    Returns:
        Scalar loss value
    """
    # Cross-entropy between teacher (soft target) and student
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=1)
    student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=1)

    # Per-pixel cross-entropy: -sum(teacher_probs * log(student_probs))
    loss = -(teacher_probs * student_log_probs).sum(1)  # [B, H, W]

    return loss.mean()


def collect_params(model):
    """
    Collect all trainable parameters.
    For CoTTA, we typically update all parameters, not just BatchNorm.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            if np in ['weight', 'bias'] and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)

    # Detach EMA model parameters
    for param in ema_model.parameters():
        param.detach_()

    # Set to eval mode
    model_anchor.eval()
    ema_model.eval()

    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """
    Configure model for use with CoTTA.
    Unlike Tent, CoTTA updates all parameters, not just BatchNorm.
    """
    model.train()
    model.requires_grad_(True)

    # For BatchNorm layers, keep in eval mode to avoid batch size=1 issues
    # This is a common practice in test-time adaptation
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # Keep BatchNorm in eval mode
            m.eval()
            # Use running stats (updated via gradient descent on parameters)
            m.track_running_stats = True

    return model
