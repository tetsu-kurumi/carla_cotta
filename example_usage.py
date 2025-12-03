"""
Example: Using CoTTA with a segmentation model for CARLA

This shows how to:
1. Load a pre-trained segmentation model
2. Wrap it with CoTTA for test-time adaptation
3. Integrate with the CARLA pipeline
"""

import torch
import torchvision.models.segmentation as seg_models
from cotta_segmentation import CoTTASegmentation, configure_model, collect_params
from carla_camera import CARLACamera
from eval import SegmentationEvaluator
import cv2
import numpy as np


def get_base_model(num_classes=13, pretrained=True):
    """
    Get a base segmentation model.

    Options:
    1. DeepLabV3+ with ResNet50 backbone (good balance)
    2. DeepLabV3+ with ResNet101 backbone (higher accuracy)
    3. FCN ResNet50 (faster, lower accuracy)
    """
    # Option 1: DeepLabV3 ResNet50 (Recommended)
    weights = 'DEFAULT' if pretrained else None
    model = seg_models.deeplabv3_resnet50(weights=weights)

    # Modify the classifier for CARLA's 13 classes
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    # aux_classifier only exists with pretrained weights
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    return model


def setup_cotta_model(base_model, learning_rate=1e-3):
    """
    Setup CoTTA wrapper around base model.

    Args:
        base_model: Pre-trained segmentation model
        learning_rate: Learning rate for test-time adaptation

    Returns:
        CoTTA-wrapped model ready for inference
    """
    # Configure model for adaptation
    model = configure_model(base_model)

    # Collect parameters to optimize
    # For CoTTA, we update all trainable parameters
    params, param_names = collect_params(model)

    # Setup optimizer (usually Adam or SGD)
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0)

    # Wrap with CoTTA
    cotta_model = CoTTASegmentation(
        model=model,
        optimizer=optimizer,
        steps=1,              # Number of adaptation steps per sample
        episodic=False,       # Set True if you want to reset after each sequence
        mt_alpha=0.999,       # EMA momentum (0.999 = slow updates)
        rst_m=0.01,          # Stochastic restoration rate (1% restored)
        ap=0.92,             # Anchor probability threshold
        num_classes=13       # CARLA has 13 classes
    )

    return cotta_model


class CoTTASegmentationPipeline:
    """Segmentation pipeline with CoTTA adaptation"""

    def __init__(self, cotta_model, device='cuda', input_size=(512, 512)):
        self.model = cotta_model
        self.device = device
        self.input_size = input_size
        self.model.to(device)

        # Normalization (ImageNet statistics)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def preprocess(self, rgb_image):
        """
        Preprocess RGB image for model input.

        Args:
            rgb_image: numpy array [H, W, 3] in range [0, 255]

        Returns:
            tensor [1, 3, H, W] normalized
        """
        # Convert to tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(rgb_image).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        # Resize if needed
        if img_tensor.shape[-2:] != self.input_size:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor, size=self.input_size, mode='bilinear', align_corners=False
            )

        # Normalize with ImageNet stats
        img_tensor = img_tensor.to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor

    def postprocess(self, output, original_size):
        """
        Post-process model output to get segmentation mask.

        Args:
            output: Model output tensor [1, num_classes, H, W]
            original_size: (H, W) of original image

        Returns:
            Segmentation mask [H, W] with class indices
        """
        # Get class predictions
        pred_mask = torch.argmax(output, dim=1).squeeze(0)  # [H, W]

        # Resize to original size
        if pred_mask.shape != original_size:
            pred_mask = pred_mask.unsqueeze(0).unsqueeze(0).float()
            pred_mask = torch.nn.functional.interpolate(
                pred_mask, size=original_size, mode='nearest'
            )
            pred_mask = pred_mask.squeeze().long()

        return pred_mask.cpu().numpy()

    def predict(self, rgb_image):
        """
        Run segmentation with CoTTA adaptation.

        Args:
            rgb_image: numpy array [H, W, 3]

        Returns:
            Segmentation mask [H, W] with class indices
        """
        original_size = rgb_image.shape[:2]

        # Preprocess
        input_tensor = self.preprocess(rgb_image)

        # Forward pass with adaptation
        # CoTTA will automatically adapt the model during this forward pass
        output = self.model(input_tensor)

        # Handle dict output (e.g., from DeepLab)
        if isinstance(output, dict):
            output = output['out']

        # Post-process
        pred_mask = self.postprocess(output, original_size)

        return pred_mask


def main_with_cotta():
    """Main function integrating CoTTA with CARLA"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load base model
    print("Loading base segmentation model...")
    base_model = get_base_model(num_classes=13, pretrained=True)

    # Optional: Load your own pre-trained weights
    # checkpoint = torch.load('path/to/your/checkpoint.pth')
    # base_model.load_state_dict(checkpoint['model_state_dict'])

    # 2. Setup CoTTA
    print("Setting up CoTTA adaptation...")
    cotta_model = setup_cotta_model(base_model, learning_rate=1e-3)

    # 3. Create pipeline
    segmentation_pipeline = CoTTASegmentationPipeline(
        cotta_model=cotta_model,
        device=device,
        input_size=(512, 512)
    )

    # 4. Initialize CARLA
    print("Initializing CARLA...")
    carla_cam = CARLACamera(host='localhost', port=2000)
    vehicle, rgb_cam, seg_cam = carla_cam.setup_vehicle_and_cameras()
    vehicle.set_autopilot(True)

    # 5. Initialize evaluator
    evaluator = SegmentationEvaluator(num_classes=13)

    try:
        frame_count = 0
        max_frames = 1000

        print("Starting test-time adaptation...")

        while frame_count < max_frames:
            # Get frames
            rgb_image, gt_seg = carla_cam.get_next_frame()

            if rgb_image is None:
                continue

            # Run CoTTA segmentation
            # The model will adapt during each forward pass!
            pred_mask = segmentation_pipeline.predict(rgb_image)

            # Update metrics
            evaluator.update(pred_mask, gt_seg)

            # Visualize
            if frame_count % 10 == 0:
                cv2.imshow('RGB', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                cv2.imshow('Ground Truth', gt_seg)
                cv2.imshow('CoTTA Prediction', pred_mask * 20)
                cv2.waitKey(1)

            frame_count += 1

            # Print metrics periodically
            if frame_count % 100 == 0:
                metrics = evaluator.compute_metrics()
                print(f"Frame {frame_count}: "
                      f"mIoU={metrics['mean_iou']:.4f}, "
                      f"Pixel Acc={metrics['pixel_accuracy']:.4f}")

    finally:
        carla_cam.cleanup()
        cv2.destroyAllWindows()

        # Final results
        final_metrics = evaluator.compute_metrics()
        print("\n" + "="*50)
        print("FINAL EVALUATION RESULTS")
        print("="*50)
        print(f"Mean IoU: {final_metrics['mean_iou']:.4f}")
        print(f"Pixel Accuracy: {final_metrics['pixel_accuracy']:.4f}")
        print(f"Total Frames: {final_metrics['total_frames']}")
        print("\nPer-class IoU:")
        class_names = ['Unlabeled', 'Building', 'Fence', 'Other', 'Pedestrian',
                       'Pole', 'RoadLine', 'Road', 'Sidewalk', 'Vegetation',
                       'Vehicles', 'Wall', 'TrafficSign']
        for i, (name, iou) in enumerate(zip(class_names, final_metrics['iou_per_class'])):
            print(f"  {name:15s}: {iou:.4f}")


def compare_with_baseline():
    """
    Compare CoTTA with baseline (no adaptation).
    This helps you see the benefit of test-time adaptation.
    """
    print("Running comparison: CoTTA vs Baseline")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup two models: one with CoTTA, one without
    base_model_cotta = get_base_model(num_classes=13)
    base_model_baseline = get_base_model(num_classes=13)

    # Load same weights for fair comparison
    # base_model_baseline.load_state_dict(base_model_cotta.state_dict())

    cotta_model = setup_cotta_model(base_model_cotta)
    baseline_model = base_model_baseline.to(device)
    baseline_model.eval()

    # Create pipelines
    cotta_pipeline = CoTTASegmentationPipeline(cotta_model, device)

    # Run evaluation...
    # (Similar to main_with_cotta but track metrics for both models)


if __name__ == '__main__':
    main_with_cotta()

    # Optionally run comparison
    # compare_with_baseline()
