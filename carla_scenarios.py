"""
CARLA Dynamic Weather and Lighting Scenarios
For evaluating continual test-time adaptation
"""

import carla
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WeatherPreset:
    """Weather configuration preset"""
    name: str
    cloudiness: float  # 0-100
    precipitation: float  # 0-100
    precipitation_deposits: float  # 0-100
    wind_intensity: float  # 0-100
    sun_azimuth_angle: float  # 0-360
    sun_altitude_angle: float  # -90 to 90
    fog_density: float  # 0-100
    fog_distance: float  # meters
    wetness: float  # 0-100


class WeatherTransitionManager:
    """
    Manages gradual weather transitions in CARLA for test-time adaptation evaluation.
    Automatically handles street lights for night scenarios.
    """

    # Define weather presets
    PRESETS = {
        'clear_noon': WeatherPreset(
            name='Clear Noon',
            cloudiness=0, precipitation=0, precipitation_deposits=0,
            wind_intensity=5, sun_azimuth_angle=0, sun_altitude_angle=75,
            fog_density=0, fog_distance=100, wetness=0
        ),
        'cloudy': WeatherPreset(
            name='Cloudy',
            cloudiness=80, precipitation=0, precipitation_deposits=0,
            wind_intensity=20, sun_azimuth_angle=0, sun_altitude_angle=75,
            fog_density=10, fog_distance=75, wetness=0
        ),
        'light_rain': WeatherPreset(
            name='Light Rain',
            cloudiness=90, precipitation=30, precipitation_deposits=50,
            wind_intensity=40, sun_azimuth_angle=0, sun_altitude_angle=75,
            fog_density=20, fog_distance=50, wetness=50
        ),
        'heavy_rain': WeatherPreset(
            name='Heavy Rain',
            cloudiness=100, precipitation=80, precipitation_deposits=90,
            wind_intensity=80, sun_azimuth_angle=0, sun_altitude_angle=75,
            fog_density=30, fog_distance=30, wetness=90
        ),
        'fog': WeatherPreset(
            name='Fog',
            cloudiness=70, precipitation=0, precipitation_deposits=0,
            wind_intensity=10, sun_azimuth_angle=0, sun_altitude_angle=75,
            fog_density=70, fog_distance=10, wetness=20
        ),
        'sunset': WeatherPreset(
            name='Sunset',
            cloudiness=30, precipitation=0, precipitation_deposits=0,
            wind_intensity=10, sun_azimuth_angle=270, sun_altitude_angle=10,
            fog_density=5, fog_distance=100, wetness=0
        ),
        'dusk': WeatherPreset(
            name='Dusk',
            cloudiness=40, precipitation=0, precipitation_deposits=0,
            wind_intensity=15, sun_azimuth_angle=270, sun_altitude_angle=-5,
            fog_density=10, fog_distance=50, wetness=0
        ),
        'night': WeatherPreset(
            name='Night',
            cloudiness=20, precipitation=0, precipitation_deposits=0,
            wind_intensity=20, sun_azimuth_angle=0, sun_altitude_angle=-75,
            fog_density=0, fog_distance=100, wetness=0
        ),
    }

    def __init__(self, world: carla.World):
        self.world = world
        self.current_weather = None

        # Get light manager for controlling street lights
        try:
            self.light_manager = self.world.get_lightmanager()
            self.lights_available = True
        except:
            print("[WARNING] Light manager not available")
            self.lights_available = False

    def set_weather(self, preset_name: str):
        """Set weather to a specific preset and manage lights"""
        if preset_name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")

        preset = self.PRESETS[preset_name]
        weather = carla.WeatherParameters(
            cloudiness=preset.cloudiness,
            precipitation=preset.precipitation,
            precipitation_deposits=preset.precipitation_deposits,
            wind_intensity=preset.wind_intensity,
            sun_azimuth_angle=preset.sun_azimuth_angle,
            sun_altitude_angle=preset.sun_altitude_angle,
            fog_density=preset.fog_density,
            fog_distance=preset.fog_distance,
            wetness=preset.wetness
        )
        self.world.set_weather(weather)

        # Manage street lights based on sun altitude
        # Turn on lights when sun is below horizon (altitude < 0)
        self._manage_lights(preset.sun_altitude_angle < 0)

        self.current_weather = preset_name
        return preset.name

    def interpolate_weather(self, preset1: WeatherPreset, preset2: WeatherPreset, alpha: float):
        """
        Interpolate between two weather presets.

        Args:
            preset1: Starting weather
            preset2: Ending weather
            alpha: Interpolation factor (0=preset1, 1=preset2)
        """
        sun_altitude = self._lerp(preset1.sun_altitude_angle, preset2.sun_altitude_angle, alpha)

        weather = carla.WeatherParameters(
            cloudiness=self._lerp(preset1.cloudiness, preset2.cloudiness, alpha),
            precipitation=self._lerp(preset1.precipitation, preset2.precipitation, alpha),
            precipitation_deposits=self._lerp(preset1.precipitation_deposits, preset2.precipitation_deposits, alpha),
            wind_intensity=self._lerp(preset1.wind_intensity, preset2.wind_intensity, alpha),
            sun_azimuth_angle=self._lerp(preset1.sun_azimuth_angle, preset2.sun_azimuth_angle, alpha),
            sun_altitude_angle=sun_altitude,
            fog_density=self._lerp(preset1.fog_density, preset2.fog_density, alpha),
            fog_distance=self._lerp(preset1.fog_distance, preset2.fog_distance, alpha),
            wetness=self._lerp(preset1.wetness, preset2.wetness, alpha)
        )
        self.world.set_weather(weather)

        # Manage lights during transition
        self._manage_lights(sun_altitude < 0)

    def _manage_lights(self, turn_on: bool):
        """
        Manage street and building lights.

        Args:
            turn_on: If True, turn lights on. If False, turn them off.
        """
        if not self.lights_available:
            return

        try:
            # Get all light groups
            street_lights = self.light_manager.get_all_lights(carla.LightGroup.Street)
            building_lights = self.light_manager.get_all_lights(carla.LightGroup.Building)

            if turn_on:
                self.light_manager.turn_on(street_lights)
                self.light_manager.turn_on(building_lights)
            else:
                self.light_manager.turn_off(street_lights)
                self.light_manager.turn_off(building_lights)
        except Exception as e:
            # Lights management is optional, don't fail if it doesn't work
            pass

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation"""
        return a + (b - a) * t

    def create_gradual_transition(
        self,
        sequence: List[str],
        frames_per_transition: int = 500
    ) -> List[Tuple[str, float, int]]:
        """
        Create a gradual transition sequence.

        Args:
            sequence: List of preset names (e.g., ['clear_noon', 'cloudy', 'light_rain'])
            frames_per_transition: Number of frames for each transition

        Returns:
            List of (current_condition, alpha, total_frame_count) tuples
        """
        transitions = []
        frame_count = 0

        for i in range(len(sequence) - 1):
            start_preset = sequence[i]
            end_preset = sequence[i + 1]

            for frame in range(frames_per_transition):
                alpha = frame / frames_per_transition
                transitions.append((start_preset, end_preset, alpha, frame_count))
                frame_count += 1

        # Hold on final condition
        transitions.append((sequence[-1], sequence[-1], 1.0, frame_count))

        return transitions


class ScenarioGenerator:
    """Generate evaluation scenarios for the proposal"""

    @staticmethod
    def weather_progression_scenario():
        """
        Scenario 1: Weather progression
        Clear → Cloudy → Light Rain → Heavy Rain → Fog
        """
        return [
            'clear_noon',
            'cloudy',
            'light_rain',
            'heavy_rain',
            'fog'
        ]

    @staticmethod
    def time_progression_scenario():
        """
        Scenario 2: Time progression
        Noon → Sunset → Dusk → Night
        """
        return [
            'clear_noon',
            'sunset',
            'dusk',
            'night'
        ]

    @staticmethod
    def combined_scenario():
        """
        Scenario 3: Combined weather and time changes
        Simulating a full day of driving
        """
        return [
            'clear_noon',      # Start: clear morning
            'cloudy',          # Clouds roll in
            'light_rain',      # Light rain
            'sunset',          # Rain clears at sunset
            'dusk',            # Getting dark
            'night',           # Night driving
        ]

    @staticmethod
    def cyclic_scenario(base_sequence: List[str], num_cycles: int = 10):
        """
        Create cyclic scenario for long-term evaluation.
        Tests catastrophic forgetting by returning to source domain.

        Args:
            base_sequence: Base sequence to cycle through
            num_cycles: Number of times to repeat

        Returns:
            Extended sequence with cycles
        """
        # Always start and end with clear_noon (source domain)
        sequence = []
        for cycle in range(num_cycles):
            sequence.extend(base_sequence)
            # Return to source domain at end of each cycle
            sequence.append('clear_noon')

        return sequence

    @staticmethod
    def stress_test_scenario():
        """
        Scenario 4: Rapid domain shifts
        Tests adaptation speed and stability
        """
        return [
            'clear_noon',
            'heavy_rain',    # Sudden shift
            'clear_noon',    # Back to source
            'fog',           # Different shift
            'clear_noon',    # Back to source
            'night',         # Another shift
            'clear_noon',    # Back to source
        ]


# Example usage
if __name__ == '__main__':
    """
    Example of how to use the scenario generator
    """

    # Generate scenarios
    print("Scenario 1: Weather Progression")
    scenario1 = ScenarioGenerator.weather_progression_scenario()
    print(f"  Sequence: {' → '.join(scenario1)}")
    print(f"  Total conditions: {len(scenario1)}\n")

    print("Scenario 2: Time Progression")
    scenario2 = ScenarioGenerator.time_progression_scenario()
    print(f"  Sequence: {' → '.join(scenario2)}")
    print(f"  Total conditions: {len(scenario2)}\n")

    print("Scenario 3: Combined")
    scenario3 = ScenarioGenerator.combined_scenario()
    print(f"  Sequence: {' → '.join(scenario3)}")
    print(f"  Total conditions: {len(scenario3)}\n")

    print("Scenario 4: Cyclic (10 cycles)")
    base = ['clear_noon', 'heavy_rain', 'fog']
    scenario4 = ScenarioGenerator.cyclic_scenario(base, num_cycles=10)
    print(f"  Total conditions: {len(scenario4)}")
    print(f"  Returns to source {scenario4.count('clear_noon')} times\n")
