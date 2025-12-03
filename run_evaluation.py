"""
Main Evaluation Script for CoTTA Research Proposal

Runs comprehensive evaluation comparing:
- Static (no adaptation)
- TTDA (Tent)
- CoTTA

Across multiple scenarios with gradual domain shifts.
"""

import torch
import torchvision.models.segmentation as seg_models
import numpy as np
import cv2
from copy import deepcopy
from pathlib import Path

from transformers import SegformerForSemanticSegmentation

from carla_camera import CARLACamera
from carla_scenarios import WeatherTransitionManager, ScenarioGenerator
from benchmark import AdaptationBenchmark
from cotta_segmentation import CoTTASegmentation, configure_model as configure_cotta, collect_params
from tent_segmentation import TentSegmentation, configure_model as configure_tent, collect_params as collect_bn_params


def get_base_model(num_classes=13, checkpoint_path=None, model_type='deeplabv3'):
    """
    Load base segmentation model.

    Args:
        num_classes: Number of semantic classes (13 for CARLA)
        checkpoint_path: Optional path to pre-trained weights (.pth for PyTorch)
        model_type: 'deeplabv3', 'fast_scnn', or 'segformer_b3'

    Returns:
        Base model
    """
    if model_type == 'fast_scnn':
        from fast_scnn_pytorch import FastSCNN
        model = FastSCNN(num_classes=num_classes)
        print(f"Using Fast-SCNN model with {num_classes} classes")

        if checkpoint_path and checkpoint_path.endswith('.pth'):
            print(f"Loading PyTorch checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
        elif checkpoint_path and checkpoint_path.endswith('.h5'):
            print(f"WARNING: .h5 files are TensorFlow format and cannot be loaded directly.")
            print(f"Using randomly initialized Fast-SCNN instead.")
            print(f"To use your trained model, please train Fast-SCNN in PyTorch.")

    elif model_type == 'segformer_b3':
        print(f"Using SegFormer B3 model with {num_classes} classes")

        if checkpoint_path:
            # More efficient: load config only, then load checkpoint
            # Avoids downloading 200MB+ ADE20K weights we'll immediately discard
            from transformers import SegformerConfig

            print("Loading SegFormer B3 architecture (no pretrained weights)...")
            config = SegformerConfig.from_pretrained(
                "nvidia/segformer-b3-finetuned-ade-512-512"
            )
            config.num_labels = num_classes
            model = SegformerForSemanticSegmentation(config)

            print(f"Loading fine-tuned checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
        else:
            # No checkpoint - use ADE20K pretrained weights (transfer learning)
            print("WARNING: No checkpoint provided. Using pretrained weights from ADE20K.")
            print("Model may not perform well on CARLA without fine-tuning.")
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b3-finetuned-ade-512-512",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )

    else:  # deeplabv3
        # Use weights='DEFAULT' for pretrained, or None for random initialization
        model = seg_models.deeplabv3_resnet50(weights='DEFAULT')

        # Modify for CARLA classes
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

        # aux_classifier only exists with pretrained weights
        if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
            model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

        print(f"Using DeepLabV3-ResNet50 model with {num_classes} classes")

        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

    return model


def setup_methods(base_model, device='cuda', learning_rate=1e-3):
    """
    Setup all three methods: Static, TTDA, CoTTA

    Returns:
        Dictionary of {method_name: (model, preprocessing_fn)}
    """
    methods = {}

    # 1. Static (no adaptation)
    static_model = deepcopy(base_model).to(device)
    static_model.eval()
    methods['Static'] = (static_model, lambda x: x)

    # 2. TTDA (Tent) - only updates BatchNorm
    tent_base = deepcopy(base_model)
    tent_model = configure_tent(tent_base)
    tent_params, _ = collect_bn_params(tent_model)
    tent_optimizer = torch.optim.Adam(tent_params, lr=learning_rate * 0.1)  # Lower LR for Tent

    tent_wrapper = TentSegmentation(
        model=tent_model,
        optimizer=tent_optimizer,
        steps=1,
        episodic=False
    ).to(device)

    methods['TTDA'] = (tent_wrapper, lambda x: x)

    # 3. CoTTA - updates all parameters with teacher/anchor
    cotta_base = deepcopy(base_model)
    cotta_model = configure_cotta(cotta_base)
    cotta_params, _ = collect_params(cotta_model)
    cotta_optimizer = torch.optim.Adam(cotta_params, lr=learning_rate)

    cotta_wrapper = CoTTASegmentation(
        model=cotta_model,
        optimizer=cotta_optimizer,
        steps=1,
        episodic=False,
        mt_alpha=0.999,
        rst_m=0.01,
        ap=0.92,
        num_classes=13
    ).to(device)

    methods['CoTTA'] = (cotta_wrapper, lambda x: x)

    return methods


def preprocess_image(rgb_image, device='cuda', input_size=(256, 512)):
    """Preprocess RGB image for model input"""
    # Convert to tensor and normalize
    # Copy the array to make it writable (CARLA images are read-only)
    img_tensor = torch.from_numpy(rgb_image.copy()).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # Resize
    if img_tensor.shape[-2:] != input_size:
        img_tensor = torch.nn.functional.interpolate(
            img_tensor, size=input_size, mode='bilinear', align_corners=False
        )

    # Move to device (no ImageNet normalization - Fast-SCNN trained on [0,1] range)
    img_tensor = img_tensor.to(device)

    return img_tensor


def postprocess_output(output, original_size, model_type='deeplabv3'):
    """Post-process model output to get segmentation mask"""
    # Handle different model output formats
    if isinstance(output, dict):
        # SegFormer returns {'logits': tensor}
        if 'logits' in output:
            logits = output['logits']
            # SegFormer outputs at lower resolution, upsample first
            logits = torch.nn.functional.interpolate(
                logits,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
            pred_mask = torch.argmax(logits, dim=1).squeeze(0)
        # DeepLabV3 returns {'out': tensor}
        elif 'out' in output:
            output = output['out']
            pred_mask = torch.argmax(output, dim=1).squeeze(0)
        else:
            # Fallback for unknown dict format
            pred_mask = torch.argmax(output, dim=1).squeeze(0)
    else:
        # Direct tensor output (Fast-SCNN)
        pred_mask = torch.argmax(output, dim=1).squeeze(0)

    # Resize to original size if needed
    if pred_mask.shape != original_size:
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0).float()
        pred_mask = torch.nn.functional.interpolate(
            pred_mask, size=original_size, mode='nearest'
        )
        pred_mask = pred_mask.squeeze().long()

    return pred_mask.cpu().numpy()


def run_evaluation(
    scenario_name: str,
    weather_sequence: list,
    frames_per_transition: int = 500,
    checkpoint_path: str = None,
    device: str = 'cuda',
    model_type: str = 'deeplabv3',
    num_classes: int = 28,
    visualize: bool = True,
    results_dir: str = './results'
):
    """
    Run complete evaluation on a scenario.

    Args:
        scenario_name: Name of the scenario (for saving results)
        weather_sequence: List of weather preset names
        frames_per_transition: Frames for each weather transition
        checkpoint_path: Path to pre-trained model checkpoint
        device: 'cuda' or 'cpu'
        visualize: Whether to show live visualization
        results_dir: Directory to save results
    """

    print(f"\n{'='*70}")
    print(f"EVALUATION: {scenario_name}")
    print(f"{'='*70}")
    print(f"Weather sequence: {' → '.join(weather_sequence)}")
    print(f"Frames per transition: {frames_per_transition}")
    print(f"Device: {device}\n")

    # Setup results directory
    scenario_dir = Path(results_dir) / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CARLA
    print("Initializing CARLA...")
    carla_cam = CARLACamera(host='localhost', port=2000)
    vehicle, rgb_cam, seg_cam = carla_cam.setup_vehicle_and_cameras()
    vehicle.set_autopilot(True)

    # Setup weather manager
    weather_manager = WeatherTransitionManager(carla_cam.world)

    # Setup models
    print("Loading models...")
    base_model = get_base_model(num_classes=num_classes, checkpoint_path=checkpoint_path, model_type=model_type)
    methods = setup_methods(base_model, device=device)

    # Setup benchmark
    benchmark = AdaptationBenchmark(num_classes=num_classes, save_dir=str(scenario_dir))
    for method_name in methods.keys():
        benchmark.register_method(method_name)

    # Create transition schedule
    transitions = weather_manager.create_gradual_transition(
        weather_sequence,
        frames_per_transition=frames_per_transition
    )

    print(f"Total frames to process: {len(transitions)}")
    print("Starting evaluation...\n")

    try:
        for idx, (start_preset, end_preset, alpha, frame_id) in enumerate(transitions):
            # Update weather
            start = weather_manager.PRESETS[start_preset]
            end = weather_manager.PRESETS[end_preset]
            weather_manager.interpolate_weather(start, end, alpha)

            # Get current domain name
            if alpha < 0.5:
                current_domain = start_preset
            else:
                current_domain = end_preset

            # Get frame from CARLA
            rgb_image, gt_seg = carla_cam.get_next_frame()
            if rgb_image is None:
                continue

            # Preprocess
            original_size = rgb_image.shape[:2]
            input_tensor = preprocess_image(rgb_image, device=device)

            # Run all methods
            for method_name, (model, _) in methods.items():
                with torch.set_grad_enabled(method_name != 'Static'):
                    output = model(input_tensor)
                    pred_mask = postprocess_output(output, original_size, model_type=model_type)

                # Update benchmark
                benchmark.update(method_name, pred_mask, gt_seg, current_domain, frame_id)

            # Print progress
            if (idx + 1) % 100 == 0:
                # Show weather transition info
                if start_preset == end_preset:
                    weather_info = f"{start_preset}"
                else:
                    weather_info = f"{start_preset} → {end_preset} ({alpha:.1%})"

                print(f"[{idx+1}/{len(transitions)}] Weather: {weather_info:35s} | ", end="")

                for method_name in methods.keys():
                    metrics = benchmark.methods[method_name]
                    current_miou = metrics.compute_overall_miou()
                    print(f"{method_name}: {current_miou:.4f} | ", end="")

                print()

            # Visualize
            if visualize and (idx % 10 == 0):
                vis_frame = rgb_image.copy()
                cv2.putText(vis_frame, f"Domain: {current_domain}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(vis_frame, f"Frame: {frame_id}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow('Evaluation', cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    finally:
        print("\nCleaning up...")
        carla_cam.cleanup()
        if visualize:
            cv2.destroyAllWindows()

    # Compute and save results
    print("\nComputing final results...")
    benchmark.save_results(filename=f'{scenario_name}_results.json')
    benchmark.plot_results(save_prefix=scenario_name)
    benchmark.print_summary()

    return benchmark


def run_all_scenarios(checkpoint_path=None, device='cuda', model_type='deeplabv3', num_classes=28):
    """
    Run all evaluation scenarios from the proposal.
    """
    scenarios = [
        {
            'name': 'weather_progression',
            'sequence': ScenarioGenerator.weather_progression_scenario(),
            'frames_per_transition': 500
        },
        {
            'name': 'time_progression',
            'sequence': ScenarioGenerator.time_progression_scenario(),
            'frames_per_transition': 500
        },
        {
            'name': 'combined',
            'sequence': ScenarioGenerator.combined_scenario(),
            'frames_per_transition': 400
        },
        {
            'name': 'cyclic_10_cycles',
            'sequence': ScenarioGenerator.cyclic_scenario(
                ['clear_noon', 'heavy_rain', 'fog'],
                num_cycles=10
            ),
            'frames_per_transition': 300
        },
        {
            'name': 'stress_test',
            'sequence': ScenarioGenerator.stress_test_scenario(),
            'frames_per_transition': 200
        }
    ]

    results = {}

    for scenario in scenarios:
        benchmark = run_evaluation(
            scenario_name=scenario['name'],
            weather_sequence=scenario['sequence'],
            frames_per_transition=scenario['frames_per_transition'],
            checkpoint_path=checkpoint_path,
            device=device,
            model_type=model_type,
            num_classes=num_classes,
            visualize=True
        )
        results[scenario['name']] = benchmark

    # Generate combined report
    print("\n" + "="*70)
    print("ALL SCENARIOS COMPLETED")
    print("="*70)

    for name, benchmark in results.items():
        print(f"\n{name.upper()}:")
        print("-" * 70)
        hypothesis_results = benchmark.compute_hypothesis_results()

        h1 = hypothesis_results['hypothesis_1']
        h2 = hypothesis_results['hypothesis_2']
        h3 = hypothesis_results['hypothesis_3']

        if 'passed' in h1:
            print(f"  H1 (Performance): {'✓' if h1['passed'] else '✗'} "
                  f"({h1['improvement_percentage']:.2f}% improvement)")

        if 'passed' in h2:
            print(f"  H2 (Stability): {'✓' if h2['passed'] else '✗'} "
                  f"({h2['variance_reduction_percentage']:.2f}% variance reduction)")

        if 'passed' in h3:
            print(f"  H3 (Forgetting): {'✓' if h3['passed'] else '✗'} "
                  f"(CoTTA: {h3['cotta_retention']:.1f}% retention, "
                  f"TTDA: {h3['ttda_degradation']:.1f}% degradation)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run CoTTA evaluation in CARLA')
    parser.add_argument('--scenario', type=str, default='all',
                       choices=['all', 'weather', 'time', 'combined', 'cyclic', 'stress'],
                       help='Which scenario to run')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to pre-trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--model-type', type=str, default='deeplabv3',
                       choices=['deeplabv3', 'fast_scnn', 'segformer_b3'],
                       help='Model architecture to use')
    parser.add_argument('--num-classes', type=int, default=28,
                       help='Number of segmentation classes (default: 28 for CARLA)')
    parser.add_argument('--frames', type=int, default=500,
                       help='Frames per weather transition')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')

    args = parser.parse_args()

    if args.scenario == 'all':
        run_all_scenarios(checkpoint_path=args.checkpoint, device=args.device, model_type=args.model_type, num_classes=args.num_classes)
    else:
        scenario_map = {
            'weather': ScenarioGenerator.weather_progression_scenario(),
            'time': ScenarioGenerator.time_progression_scenario(),
            'combined': ScenarioGenerator.combined_scenario(),
            'cyclic': ScenarioGenerator.cyclic_scenario(['clear_noon', 'heavy_rain', 'fog'], 10),
            'stress': ScenarioGenerator.stress_test_scenario()
        }

        run_evaluation(
            scenario_name=args.scenario,
            weather_sequence=scenario_map[args.scenario],
            frames_per_transition=args.frames,
            checkpoint_path=args.checkpoint,
            device=args.device,
            model_type=args.model_type,
            num_classes=args.num_classes,
            visualize=not args.no_viz
        )
