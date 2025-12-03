"""
Fast-SCNN in PyTorch
Converted from TensorFlow implementation for CARLA segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution block with BN and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNetV2 style)."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6):
        super().__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels

        expand_channels = in_channels * expand_ratio

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expand_channels, 1, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, 3, stride, 1, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        if self.use_res_connect:
            out = out + x

        return out


class LearningToDownsample(nn.Module):
    """Learning to Downsample module."""
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(3, 32, kernel_size=3, stride=2, padding=1)
        self.dsconv1 = DepthwiseSeparableConv(32, 48, stride=2)
        self.dsconv2 = DepthwiseSeparableConv(48, 64, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global Feature Extractor module."""
    def __init__(self):
        super().__init__()

        # First set of bottlenecks
        self.bottleneck1 = nn.Sequential(
            InvertedResidual(64, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6)
        )

        # Second set
        self.bottleneck2 = nn.Sequential(
            InvertedResidual(64, 96, stride=2, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6)
        )

        # Third set
        self.bottleneck3 = nn.Sequential(
            InvertedResidual(96, 128, stride=1, expand_ratio=6),
            InvertedResidual(128, 128, stride=1, expand_ratio=6),
            InvertedResidual(128, 128, stride=1, expand_ratio=6)
        )

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module."""
    def __init__(self, high_in_channels=64, low_in_channels=128, out_channels=128):
        super().__init__()

        # Process high resolution features
        self.high_res_conv = DepthwiseSeparableConv(high_in_channels, out_channels, stride=1)

        # Fuse
        self.fuse = ConvBlock(low_in_channels + out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, low_res, high_res):
        # Upsample low resolution to match high resolution
        low_res_up = F.interpolate(low_res, size=high_res.shape[2:], mode='bilinear', align_corners=False)

        # Process high resolution
        high_res = self.high_res_conv(high_res)

        # Concatenate and fuse
        concat = torch.cat([low_res_up, high_res], dim=1)
        fused = self.fuse(concat)

        return fused


class Classifier(nn.Module):
    """Classifier module."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.dsconv1 = DepthwiseSeparableConv(in_channels, 128, stride=1)
        self.dsconv2 = DepthwiseSeparableConv(128, 128, stride=1)
        self.conv = nn.Conv2d(128, num_classes, 1)

    def forward(self, x, output_size):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)

        # Upsample to original input size
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)

        return x


class FastSCNN(nn.Module):
    """Fast-SCNN model for semantic segmentation."""
    def __init__(self, num_classes=13):
        super().__init__()

        self.lds = LearningToDownsample()
        self.gfe = GlobalFeatureExtractor()
        self.ffm = FeatureFusionModule(high_in_channels=64, low_in_channels=128, out_channels=128)
        self.classifier = Classifier(128, num_classes)

    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)

        # Learning to Downsample
        lds_out = self.lds(x)

        # Global Feature Extractor
        gfe_out = self.gfe(lds_out)

        # Feature Fusion
        ffm_out = self.ffm(gfe_out, lds_out)

        # Classifier
        out = self.classifier(ffm_out, input_size)

        return out


def load_tensorflow_weights(pytorch_model, tf_checkpoint_path):
    """
    Load TensorFlow/Keras weights into PyTorch model.

    Note: This requires tensorflow to be installed.
    The weight mapping is approximate and may need fine-tuning.
    """
    try:
        import tensorflow as tf
        import h5py
    except ImportError:
        raise ImportError("TensorFlow and h5py are required to load .h5 weights")

    print(f"Loading TensorFlow weights from: {tf_checkpoint_path}")

    # Load the Keras model
    tf_model = tf.keras.models.load_model(
        tf_checkpoint_path,
        compile=False,
        custom_objects={
            'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy,
            'MeanIoU': tf.keras.metrics.MeanIoU
        }
    )

    # Get TensorFlow weights
    tf_weights = {}
    for layer in tf_model.layers:
        for weight in layer.weights:
            tf_weights[weight.name] = weight.numpy()

    # Manual weight mapping (this is approximate)
    # You may need to adjust based on exact layer structure
    print("Note: Weight conversion is approximate. Fine-tuning may be needed.")

    return pytorch_model


def create_fast_scnn(num_classes=13, pretrained_path=None):
    """
    Create Fast-SCNN model, optionally loading pretrained weights.

    Args:
        num_classes: Number of segmentation classes (13 for CARLA)
        pretrained_path: Path to TensorFlow .h5 checkpoint (optional)

    Returns:
        PyTorch Fast-SCNN model
    """
    model = FastSCNN(num_classes=num_classes)

    if pretrained_path is not None:
        # Note: Full weight conversion from TF to PyTorch is complex
        # For best results, retrain in PyTorch or use a pre-converted checkpoint
        print(f"Warning: Loading TensorFlow weights to PyTorch requires manual weight mapping.")
        print(f"Consider retraining the model in PyTorch for best results.")
        # model = load_tensorflow_weights(model, pretrained_path)

    return model


# For compatibility with run_evaluation.py
def get_fast_scnn_model(num_classes=13, checkpoint_path=None):
    """
    Get Fast-SCNN model for use with CoTTA/Tent.

    This is a drop-in replacement for get_base_model() in run_evaluation.py
    """
    model = create_fast_scnn(num_classes=num_classes)

    if checkpoint_path is not None and checkpoint_path.endswith('.pth'):
        # Load PyTorch checkpoint
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        print(f"Loaded PyTorch weights from: {checkpoint_path}")
    elif checkpoint_path is not None and checkpoint_path.endswith('.h5'):
        print(f"Warning: .h5 files are TensorFlow format.")
        print(f"To use with PyTorch, you need to:")
        print(f"  1. Train Fast-SCNN in PyTorch on CARLA data, OR")
        print(f"  2. Convert TensorFlow weights to PyTorch (complex)")

    return model


if __name__ == '__main__':
    # Test the model
    model = FastSCNN(num_classes=13)

    # Test with input matching CARLA images
    x = torch.randn(1, 3, 256, 512)  # Same as TF config: (256, 512)

    model.eval()
    with torch.no_grad():
        out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected output: torch.Size([1, 13, 256, 512])")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
