import carla
import numpy as np
import queue
from threading import Thread

class CARLACamera:
    def __init__(self, host='localhost', port=2000, synchronous=True, fps=20):
        """
        Initialize CARLA camera system.

        Args:
            host: CARLA server host
            port: CARLA server port
            synchronous: Enable synchronous mode (recommended for reproducible research)
            fps: Target FPS (only in synchronous mode)
        """
        # Connect to CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Enable synchronous mode for reproducible experiments
        self.synchronous = synchronous
        if synchronous:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / fps  # Fixed time-step
            self.world.apply_settings(settings)
            print(f"[CARLA] Synchronous mode enabled with {fps} FPS")

        # Queue for camera data
        self.image_queue = queue.Queue()
        self.segmentation_queue = queue.Queue()

        # For cleanup
        self.actors = []
        
    def setup_vehicle_and_cameras(self, image_size=(800, 600)):
        """
        Setup vehicle and cameras.

        Args:
            image_size: (width, height) for camera images
        """
        # Get blueprint library
        blueprint_library = self.world.get_blueprint_library()

        # Spawn vehicle with retry logic
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) == 0:
            raise RuntimeError("No spawn points available in the map")

        import random
        random.shuffle(spawn_points)

        self.vehicle = None
        for spawn_point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.actors.append(self.vehicle)
                print(f"[CARLA] Vehicle spawned at {spawn_point.location}")
                break
            except RuntimeError as e:
                if "collision" in str(e).lower():
                    continue
                raise

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle: all spawn points have collisions")

        # RGB Camera setup
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(image_size[0]))
        camera_bp.set_attribute('image_size_y', str(image_size[1]))
        camera_bp.set_attribute('fov', '110')

        # Semantic Segmentation Camera (ground truth)
        seg_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', str(image_size[0]))
        seg_camera_bp.set_attribute('image_size_y', str(image_size[1]))
        seg_camera_bp.set_attribute('fov', '110')

        # Camera transform (attach to vehicle)
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )

        # Spawn cameras
        self.rgb_camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.seg_camera = self.world.spawn_actor(
            seg_camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.actors.extend([self.rgb_camera, self.seg_camera])

        # Register callbacks
        self.rgb_camera.listen(lambda image: self._process_rgb_image(image))
        self.seg_camera.listen(lambda image: self._process_seg_image(image))

        print(f"[CARLA] Cameras attached to vehicle")

        # Tick once to ensure actors are spawned
        if self.synchronous:
            self.world.tick()

        return self.vehicle, self.rgb_camera, self.seg_camera
    
    def _process_rgb_image(self, image):
        """Convert CARLA image to numpy array"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        self.image_queue.put((image.frame, array))
    
    def _process_seg_image(self, image):
        """Convert semantic segmentation to numpy array"""
        # Don't convert to CityScapesPalette - keep raw label IDs
        # Raw data contains label IDs in the red channel
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # Extract red channel which contains the label IDs (0-27 for CARLA)
        labels = array[:, :, 2]  # Red channel contains labels
        self.segmentation_queue.put((image.frame, labels))
    
    def get_next_frame(self, timeout=2.0):
        """
        Get synchronized RGB and ground truth segmentation.

        In synchronous mode, this will tick the world and wait for the next frame.
        """
        # Tick the world to advance simulation (only in synchronous mode)
        if self.synchronous:
            self.world.tick()

        try:
            # Get data from both cameras
            rgb_frame, rgb_image = self.image_queue.get(timeout=timeout)
            seg_frame, seg_image = self.segmentation_queue.get(timeout=timeout)

            # Verify frames are synchronized (should match in synchronous mode)
            if abs(rgb_frame - seg_frame) > 1:
                print(f"[WARNING] Frame mismatch: RGB={rgb_frame}, Seg={seg_frame}")

            return rgb_image, seg_image
        except queue.Empty:
            print("[WARNING] Camera data timeout - no frames received")
            return None, None
    
    def cleanup(self):
        """Clean up actors and restore settings"""
        print("[CARLA] Cleaning up...")

        # Stop cameras first
        if hasattr(self, 'rgb_camera') and self.rgb_camera is not None:
            self.rgb_camera.stop()
        if hasattr(self, 'seg_camera') and self.seg_camera is not None:
            self.seg_camera.stop()

        # Destroy all actors
        for actor in self.actors:
            if actor is not None and actor.is_alive:
                actor.destroy()

        # Restore asynchronous mode
        if self.synchronous:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            print("[CARLA] Synchronous mode disabled")

        print("[CARLA] Cleanup complete")