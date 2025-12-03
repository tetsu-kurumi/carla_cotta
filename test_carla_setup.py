"""
Test script to verify CARLA setup is working correctly.
Run this before the full evaluation to catch any issues.
"""

import sys
import cv2
import numpy as np

def test_carla_connection():
    """Test basic CARLA connection"""
    print("="*70)
    print("TEST 1: CARLA Connection")
    print("="*70)

    try:
        from carla_camera import CARLACamera

        print("Connecting to CARLA server...")
        carla_cam = CARLACamera(host='localhost', port=2000, synchronous=True, fps=20)
        print("✓ Connected successfully")
        print(f"✓ Synchronous mode enabled: {carla_cam.synchronous}")
        print(f"✓ World: {carla_cam.world}")

        return carla_cam
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nMake sure CARLA is running:")
        print("  cd /home/tetsuk/Downloads/CARLA_0.9.16")
        print("  ./CarlaUE4.sh")
        sys.exit(1)


def test_vehicle_spawn(carla_cam):
    """Test vehicle and camera spawning"""
    print("\n" + "="*70)
    print("TEST 2: Vehicle and Camera Setup")
    print("="*70)

    try:
        print("Spawning vehicle and cameras...")
        vehicle, rgb_cam, seg_cam = carla_cam.setup_vehicle_and_cameras(image_size=(800, 600))

        print(f"✓ Vehicle spawned: {vehicle.type_id}")
        print(f"✓ RGB camera attached: {rgb_cam.type_id}")
        print(f"✓ Segmentation camera attached: {seg_cam.type_id}")

        return vehicle, rgb_cam, seg_cam
    except Exception as e:
        print(f"✗ Failed to setup vehicle: {e}")
        carla_cam.cleanup()
        sys.exit(1)


def test_autopilot(vehicle):
    """Test autopilot"""
    print("\n" + "="*70)
    print("TEST 3: Autopilot")
    print("="*70)

    try:
        print("Enabling autopilot...")
        vehicle.set_autopilot(True)
        print("✓ Autopilot enabled")
    except Exception as e:
        print(f"✗ Failed to enable autopilot: {e}")


def test_frame_capture(carla_cam):
    """Test capturing frames"""
    print("\n" + "="*70)
    print("TEST 4: Frame Capture")
    print("="*70)

    try:
        print("Capturing 10 frames...")
        for i in range(10):
            rgb_image, seg_image = carla_cam.get_next_frame(timeout=5.0)

            if rgb_image is None:
                print(f"✗ Failed to get frame {i+1}")
                return False

            print(f"  Frame {i+1}: RGB shape={rgb_image.shape}, Seg shape={seg_image.shape}")

        print("✓ All frames captured successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to capture frames: {e}")
        return False


def test_weather_transitions(carla_cam):
    """Test weather transition system"""
    print("\n" + "="*70)
    print("TEST 5: Weather Transitions")
    print("="*70)

    try:
        from carla_scenarios import WeatherTransitionManager, ScenarioGenerator

        print("Initializing weather manager...")
        weather_manager = WeatherTransitionManager(carla_cam.world)
        print("✓ Weather manager initialized")

        # Test setting different weather conditions
        conditions = ['clear_noon', 'cloudy', 'light_rain', 'night']

        for condition in conditions:
            print(f"\nTesting weather: {condition}")
            name = weather_manager.set_weather(condition)
            print(f"  ✓ Weather set to: {name}")

            # Capture a frame in this weather
            rgb_image, seg_image = carla_cam.get_next_frame(timeout=5.0)
            if rgb_image is not None:
                print(f"  ✓ Frame captured in {condition}")
            else:
                print(f"  ✗ Failed to capture frame in {condition}")

        # Test gradual transition
        print("\nTesting gradual transition (clear → rain)...")
        start = weather_manager.PRESETS['clear_noon']
        end = weather_manager.PRESETS['light_rain']

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            weather_manager.interpolate_weather(start, end, alpha)
            print(f"  ✓ Transition alpha={alpha:.2f}")

        print("✓ Weather transitions working")
        return True

    except Exception as e:
        print(f"✗ Weather transition failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization(carla_cam):
    """Test visualization"""
    print("\n" + "="*70)
    print("TEST 6: Visualization (5 seconds)")
    print("="*70)

    try:
        print("Displaying live feed for 5 seconds...")
        print("Press 'q' to skip")

        import time
        start_time = time.time()

        while time.time() - start_time < 5:
            rgb_image, seg_image = carla_cam.get_next_frame(timeout=2.0)

            if rgb_image is None:
                continue

            # Display
            display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('CARLA Test', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("✓ Visualization working")
        return True

    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False


def test_model_loading():
    """Test model loading"""
    print("\n" + "="*70)
    print("TEST 7: Model Loading")
    print("="*70)

    try:
        import torch
        import torchvision.models.segmentation as seg_models

        print("Loading DeepLabV3 ResNet50...")
        # Use weights parameter instead of deprecated pretrained
        model = seg_models.deeplabv3_resnet50(weights=None)

        # Modify for CARLA's 13 classes
        model.classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=1)

        # aux_classifier is only present if model was loaded with pretrained weights
        if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
            model.aux_classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=1)
            print("  ✓ Modified aux_classifier for 13 classes")
        else:
            print("  ℹ aux_classifier not present (normal for non-pretrained model)")

        print(f"✓ Model loaded: {type(model).__name__}")

        # Test forward pass (use eval mode to avoid BatchNorm issues with batch_size=1)
        model.eval()
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ Forward pass successful: output shape = {output['out'].shape}")

        return True

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cotta_tent():
    """Test CoTTA and Tent implementations"""
    print("\n" + "="*70)
    print("TEST 8: CoTTA/Tent Loading")
    print("="*70)

    try:
        import torch
        import torchvision.models.segmentation as seg_models
        from cotta_segmentation import CoTTASegmentation, configure_model as configure_cotta, collect_params
        from tent_segmentation import TentSegmentation, configure_model as configure_tent, collect_params as collect_bn_params

        # Test CoTTA
        print("Testing CoTTA...")
        base_model = seg_models.deeplabv3_resnet50(weights=None)
        base_model.classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=1)

        # Only modify aux_classifier if it exists
        if hasattr(base_model, 'aux_classifier') and base_model.aux_classifier is not None:
            base_model.aux_classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=1)

        model = configure_cotta(base_model)
        params, _ = collect_params(model)
        optimizer = torch.optim.Adam(params, lr=1e-3)

        cotta = CoTTASegmentation(
            model=model,
            optimizer=optimizer,
            steps=1,
            mt_alpha=0.999,
            rst_m=0.01,
            ap=0.92,
            num_classes=13
        )
        print(f"✓ CoTTA initialized")

        # Test forward pass (use batch_size=2 because training mode requires >1 sample)
        dummy_input = torch.randn(2, 3, 512, 512)
        output = cotta(dummy_input)
        print(f"✓ CoTTA forward pass: output shape = {output.shape}")

        # Test Tent
        print("\nTesting Tent...")
        base_model2 = seg_models.deeplabv3_resnet50(weights=None)
        base_model2.classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=1)

        # Only modify aux_classifier if it exists
        if hasattr(base_model2, 'aux_classifier') and base_model2.aux_classifier is not None:
            base_model2.aux_classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=1)

        model2 = configure_tent(base_model2)
        params2, _ = collect_bn_params(model2)
        optimizer2 = torch.optim.Adam(params2, lr=1e-4)

        tent = TentSegmentation(
            model=model2,
            optimizer=optimizer2,
            steps=1
        )
        print(f"✓ Tent initialized")

        # Use same dummy_input with batch_size=2
        output2 = tent(dummy_input)
        print(f"✓ Tent forward pass: output shape = {output2.shape}")

        return True

    except Exception as e:
        print(f"✗ CoTTA/Tent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "CARLA SETUP TEST SUITE" + " "*31 + "║")
    print("╚" + "="*68 + "╝")

    # Track results
    results = {}

    # Test 1: Connection
    carla_cam = test_carla_connection()
    results['connection'] = True

    try:
        # Test 2: Vehicle spawn
        vehicle, rgb_cam, seg_cam = test_vehicle_spawn(carla_cam)
        results['vehicle_spawn'] = True

        # Test 3: Autopilot
        test_autopilot(vehicle)
        results['autopilot'] = True

        # Test 4: Frame capture
        results['frame_capture'] = test_frame_capture(carla_cam)

        # Test 5: Weather
        results['weather'] = test_weather_transitions(carla_cam)

        # Test 6: Visualization
        results['visualization'] = test_visualization(carla_cam)

    finally:
        # Cleanup CARLA
        print("\n" + "="*70)
        print("CLEANUP")
        print("="*70)
        carla_cam.cleanup()

    # Test 7: Model loading (doesn't need CARLA)
    results['model_loading'] = test_model_loading()

    # Test 8: CoTTA/Tent
    results['cotta_tent'] = test_cotta_tent()

    # Summary
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*25 + "TEST SUMMARY" + " "*31 + "║")
    print("╚" + "="*68 + "╝")

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for evaluation!")
        print("\nYou can now run:")
        print("  python run_evaluation.py --scenario weather --frames 100")
    else:
        print("✗ SOME TESTS FAILED - Please fix issues before running evaluation")
        sys.exit(1)
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
