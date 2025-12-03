"""
Collect CARLA images for EVALUATION/TESTING.
Uses SAME weather conditions as training but collects DIFFERENT images.
Saves data in HDF5 format for efficient PyTorch loading.

IMPORTANT: This creates a separate evaluation dataset with distinct images
from the training set, even though weather conditions are the same.
"""

import carla
import numpy as np
import h5py
import time
from pathlib import Path
from tqdm import tqdm
import random
import shutil


class EvalDataCollector:
    """
    Collects evaluation/test data from CARLA.
    Uses SAME weather scenarios as training but DIFFERENT images.
    """

    # Evaluation weather scenarios - SAME as training scenarios
    # Images will be distinct due to different spawn points/trajectories/random seed
    EVAL_WEATHERS = [
        'clear_noon',       # Clear conditions
        'cloudy',           # Cloudy
        'light_rain',       # Light rain
        'heavy_rain',       # Heavy rain
        'fog',              # Fog
        'sunset',           # Sunset
        'dusk',             # Dusk
        'night',            # Night
    ]

    def __init__(self, host='localhost', port=2000, output_dir='./eval_data', random_seed=42):
        """
        Initialize collector

        Args:
            host: CARLA server host
            port: CARLA server port
            output_dir: Directory to save evaluation data
            random_seed: Random seed for spawn point selection (different from training)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducible but distinct spawn points
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Connect to CARLA
        print(f"Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        self.actors = []
        self.rgb_image = None
        self.seg_image = None

        print("CARLA initialized in synchronous mode")
        print(f"Random seed: {random_seed} (for reproducible but distinct data)")

    def setup_vehicle_and_cameras(self):
        """Spawn vehicle with RGB and segmentation cameras"""
        blueprint_library = self.world.get_blueprint_library()

        # Spawn vehicle
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        vehicle = None
        for spawn_point in spawn_points:
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.actors.append(vehicle)
                break
            except RuntimeError:
                continue

        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")

        # Enable autopilot
        vehicle.set_autopilot(True)

        # RGB Camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        rgb_cam = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        self.actors.append(rgb_cam)
        rgb_cam.listen(lambda image: self._process_rgb_image(image))

        # Segmentation Camera
        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', '640')
        seg_bp.set_attribute('image_size_y', '480')
        seg_bp.set_attribute('fov', '110')

        seg_cam = self.world.spawn_actor(seg_bp, camera_transform, attach_to=vehicle)
        self.actors.append(seg_cam)
        seg_cam.listen(lambda image: self._process_seg_image(image))

        return vehicle

    def _process_rgb_image(self, image):
        """Process RGB camera callback"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.rgb_image = array[:, :, :3].copy()

    def _process_seg_image(self, image):
        """Process segmentation camera callback"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.seg_image = array[:, :, 2].copy()  # Red channel contains labels

    def set_weather(self, weather_name):
        """Set weather preset - matches carla_scenarios.py weather definitions"""
        weather_presets = {
            # Main evaluation weather scenarios
            'clear_noon': carla.WeatherParameters.ClearNoon,
            'cloudy': carla.WeatherParameters.CloudyNoon,
            'light_rain': carla.WeatherParameters.WetNoon,  # Light rain
            'heavy_rain': carla.WeatherParameters.HardRainNoon,
            'fog': carla.WeatherParameters.ClearNoon,  # Will add fog manually
            'sunset': carla.WeatherParameters.ClearSunset,
            'dusk': carla.WeatherParameters.ClearSunset,  # Will modify sun angle
            'night': carla.WeatherParameters.ClearNight,
        }

        if weather_name in weather_presets:
            weather = weather_presets[weather_name]

            # Custom modifications to match carla_scenarios.py
            if weather_name == 'fog':
                weather.fog_density = 70
                weather.fog_distance = 10
                weather.cloudiness = 70
                weather.wetness = 20
            elif weather_name == 'dusk':
                weather.sun_altitude_angle = -5  # Just below horizon
                weather.cloudiness = 40
            elif weather_name == 'light_rain':
                weather.precipitation = 30
                weather.precipitation_deposits = 50
                weather.wetness = 50

            self.world.set_weather(weather)

    def collect_batch(self, num_frames, weather_name, description="Collecting"):
        """Collect a batch of frames under specific weather"""
        print(f"\n{description}: {weather_name}")
        self.set_weather(weather_name)

        # Warmup - let weather stabilize
        for _ in range(10):
            self.world.tick()

        rgb_batch = []
        seg_batch = []

        for _ in tqdm(range(num_frames), desc=weather_name):
            self.world.tick()
            time.sleep(0.01)  # Small delay for callbacks

            if self.rgb_image is not None and self.seg_image is not None:
                rgb_batch.append(self.rgb_image.copy())
                seg_batch.append(self.seg_image.copy())

        return rgb_batch, seg_batch

    def check_disk_space(self, required_gb=50):
        """Check if sufficient disk space is available"""
        stat = shutil.disk_usage(self.output_dir)
        free_gb = stat.free / (1024**3)
        print(f"\nDisk space available: {free_gb:.2f} GB")
        if free_gb < required_gb:
            print(f"WARNING: Less than {required_gb} GB available. Consider freeing up space.")
            return False
        return True

    def collect_dataset(self, frames_per_weather=350, num_vehicles=3, resume=True):
        """
        Collect comprehensive evaluation dataset.

        Args:
            frames_per_weather: Number of frames per weather condition (typically less than training)
            num_vehicles: Number of different vehicle spawns per weather
            resume: If True, resume from existing file if it exists
        """
        total_collected = 0
        completed_weathers = set()

        # Create HDF5 file
        h5_path = self.output_dir / 'eval_data.h5'

        # Check if file exists and we're in resume mode
        file_exists = h5_path.exists()
        mode = 'a' if (resume and file_exists) else 'w'

        if file_exists and resume:
            print(f"\nResuming from existing file: {h5_path}")
            # Load existing progress
            with h5py.File(h5_path, 'r') as h5f_read:
                if 'completed_weathers' in h5f_read.attrs:
                    completed_weathers = set(h5f_read.attrs['completed_weathers'])
                    total_collected = h5f_read['rgb'].shape[0]
                    print(f"Already collected: {total_collected} frames")
                    print(f"Completed weathers: {list(completed_weathers)}")
        else:
            print(f"\nSaving to: {h5_path}")

        # Check disk space
        self.check_disk_space(required_gb=20)

        with h5py.File(h5_path, mode) as h5f:
            # Create or get existing datasets
            max_samples = len(self.EVAL_WEATHERS) * frames_per_weather * num_vehicles

            if 'rgb' in h5f:
                rgb_dset = h5f['rgb']
                seg_dset = h5f['segmentation']
                print(f"Using existing datasets with {rgb_dset.shape[0]} frames")
            else:
                rgb_dset = h5f.create_dataset(
                    'rgb',
                    shape=(0, 480, 640, 3),
                    maxshape=(max_samples, 480, 640, 3),
                    dtype=np.uint8,
                    chunks=(100, 480, 640, 3),
                    compression='gzip',
                    compression_opts=1
                )

                seg_dset = h5f.create_dataset(
                    'segmentation',
                    shape=(0, 480, 640),
                    maxshape=(max_samples, 480, 640),
                    dtype=np.uint8,
                    chunks=(100, 480, 640),
                    compression='gzip',
                    compression_opts=1
                )

            # Update metadata
            h5f.attrs['weathers'] = self.EVAL_WEATHERS
            h5f.attrs['frames_per_weather'] = frames_per_weather
            h5f.attrs['num_vehicles'] = num_vehicles
            h5f.attrs['dataset_type'] = 'evaluation'

            # Collect from each weather condition
            for weather_idx, weather in enumerate(self.EVAL_WEATHERS):
                # Skip if already completed
                if weather in completed_weathers:
                    print(f"\n{'='*70}")
                    print(f"Weather {weather_idx+1}/{len(self.EVAL_WEATHERS)}: {weather} [SKIPPING - already completed]")
                    print(f"{'='*70}")
                    continue

                print(f"\n{'='*70}")
                print(f"Weather {weather_idx+1}/{len(self.EVAL_WEATHERS)}: {weather}")
                print(f"{'='*70}")

                weather_frames_collected = 0

                # Collect from multiple vehicle spawns for diversity
                for vehicle_idx in range(num_vehicles):
                    # Clean up previous vehicle
                    self.cleanup()

                    # Spawn new vehicle at random location
                    try:
                        vehicle = self.setup_vehicle_and_cameras()
                    except Exception as e:
                        print(f"Failed to spawn vehicle: {e}")
                        continue

                    # Collect frames
                    frames_per_spawn = frames_per_weather // num_vehicles
                    rgb_batch, seg_batch = self.collect_batch(
                        frames_per_spawn,
                        weather,
                        f"Vehicle {vehicle_idx+1}/{num_vehicles}"
                    )

                    if len(rgb_batch) > 0:
                        try:
                            # Append to HDF5
                            current_size = rgb_dset.shape[0]
                            new_size = current_size + len(rgb_batch)

                            rgb_dset.resize(new_size, axis=0)
                            seg_dset.resize(new_size, axis=0)

                            rgb_dset[current_size:new_size] = np.array(rgb_batch)
                            seg_dset[current_size:new_size] = np.array(seg_batch)

                            total_collected += len(rgb_batch)
                            weather_frames_collected += len(rgb_batch)

                            print(f"  Collected {len(rgb_batch)} frames | Total: {total_collected}")

                            # Flush to disk periodically
                            h5f.flush()
                        except OSError as e:
                            if e.errno == 28:  # No space left
                                print(f"\n{'!'*70}")
                                print(f"ERROR: Disk full! Collected {total_collected} frames total.")
                                print(f"Free up space and re-run to continue from this point.")
                                print(f"{'!'*70}")
                                raise
                            else:
                                raise

                # Mark weather as completed
                completed_weathers.add(weather)
                h5f.attrs['completed_weathers'] = list(completed_weathers)
                h5f.flush()
                print(f"  ✓ Completed {weather}: {weather_frames_collected} frames")

        print(f"\n{'='*70}")
        print(f"Evaluation data collection complete!")
        print(f"Total frames collected: {total_collected}")
        print(f"Saved to: {h5_path}")
        print(f"File size: {h5_path.stat().st_size / 1024**3:.2f} GB")
        print(f"{'='*70}")

    def cleanup(self):
        """Clean up actors"""
        for actor in self.actors:
            if actor is not None:
                try:
                    actor.destroy()
                except RuntimeError:
                    # Actor already destroyed
                    pass
        self.actors.clear()
        self.rgb_image = None
        self.seg_image = None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect CARLA evaluation data')
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA host')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA port')
    parser.add_argument('--output-dir', type=str, default='./eval_data',
                       help='Output directory')
    parser.add_argument('--frames-per-weather', type=int, default=500,
                       help='Frames to collect per weather condition (typically less than training)')
    parser.add_argument('--num-vehicles', type=int, default=3,
                       help='Number of vehicle spawns per weather')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for spawn point selection (use different from training)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start from scratch instead of resuming')

    args = parser.parse_args()

    collector = EvalDataCollector(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )

    try:
        collector.collect_dataset(
            frames_per_weather=args.frames_per_weather,
            num_vehicles=args.num_vehicles,
            resume=not args.no_resume
        )
    finally:
        collector.cleanup()
        print("\nCleaned up CARLA actors")
