#!/usr/bin/env python3
"""
Configuration module for sonar 3D reconstruction.

This module contains default configurations for sensor parameters,
grid settings, and data paths.
"""

from typing import Tuple, Dict, Any
from pathlib import Path


class SonarConfig:
    """
    Configuration class for sonar sensor parameters.
    
    Attributes:
        max_range: Maximum sonar range in meters
        fov_degrees: Horizontal field of view in degrees
        vertical_aperture_degrees: Vertical beam aperture in degrees
        intensity_threshold: Threshold for occupied detection (0-255)
        min_range: Minimum range to filter noise in meters
    """
    
    def __init__(self):
        # Oculus M750D default parameters
        self.max_range: float = 5.0
        self.fov_degrees: float = 130.0
        self.vertical_aperture_degrees: float = 20.0
        self.intensity_threshold: int = 35
        self.min_range: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_range': self.max_range,
            'fov_degrees': self.fov_degrees,
            'vertical_aperture_degrees': self.vertical_aperture_degrees,
            'intensity_threshold': self.intensity_threshold,
            'min_range': self.min_range
        }


class GridConfig:
    """
    Configuration class for occupancy grid parameters.
    
    Attributes:
        resolution: Grid resolution in meters
        x_range: (min, max) range in X direction (lateral)
        y_range: (min, max) range in Y direction (longitudinal)
        z_range: (min, max) range in Z direction (depth, negative underwater)
    """
    
    def __init__(self):
        self.resolution: float = 0.03  # 30mm default
        self.x_range: Tuple[float, float] = (-0.75, 0.75)  # 1.5m width
        self.y_range: Tuple[float, float] = (-0.75, 0.75)  # 1.5m depth
        self.z_range: Tuple[float, float] = (-1.5, 0.0)    # 1.5m height (underwater)
    
    def get_grid_limits(self) -> Dict[str, Tuple[float, float]]:
        """Get grid limits as dictionary."""
        return {
            'x': self.x_range,
            'y': self.y_range,
            'z': self.z_range
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'resolution': self.resolution,
            'x_range': self.x_range,
            'y_range': self.y_range,
            'z_range': self.z_range
        }


class SensorPoseConfig:
    """
    Configuration class for sensor position and orientation.
    
    Default configuration is downward-looking sonar at water surface.
    
    Attributes:
        position: (x, y, z) position in meters
        pose: (heading, tilt, roll) orientation in degrees
    """
    
    def __init__(self):
        # Default: sensor at water surface, looking downward
        self.position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (heading, tilt, roll)
    
    def get_heading(self) -> float:
        """Get heading angle in degrees."""
        return self.pose[0]
    
    def get_tilt(self) -> float:
        """Get tilt angle in degrees."""
        return self.pose[1]
    
    def get_roll(self) -> float:
        """Get roll angle in degrees."""
        return self.pose[2]
    
    def set_pose(self, heading: float = None, tilt: float = None, roll: float = None):
        """
        Update sensor pose.
        
        Args:
            heading: Yaw angle in degrees (rotation around Z)
            tilt: Pitch angle in degrees (rotation around Y)
            roll: Roll angle in degrees (rotation around X)
        """
        current_heading, current_tilt, current_roll = self.pose
        
        if heading is not None:
            current_heading = heading
        if tilt is not None:
            current_tilt = tilt
        if roll is not None:
            current_roll = roll
        
        self.pose = (current_heading, current_tilt, current_roll)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'position': self.position,
            'pose': self.pose
        }


class DataConfig:
    """
    Configuration class for data paths and settings.
    
    Attributes:
        base_path: Base path to data directory
        bag_file: Name of the ROS2 bag file
        output_dir: Directory for output files
    """
    
    def __init__(self):
        self.base_path: Path = Path("/workspace/data/3_janggil_ri/20250801_blueboat_sonar_lidar")
        self.bag_file: str = "sonar-scenario1-v1.deg90.bag"
        self.output_dir: Path = Path("/workspace/ros2_ws/src/sonar_3d_reconstruction/output")
    
    def get_bag_path(self) -> Path:
        """Get full path to bag file."""
        return self.base_path / self.bag_file
    
    def ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'base_path': str(self.base_path),
            'bag_file': self.bag_file,
            'output_dir': str(self.output_dir)
        }


class VisualizationConfig:
    """
    Configuration class for visualization settings.
    
    Attributes:
        display_mode: 'points' or 'cubes' for voxel display
        figure_size: (width, height) in inches
        initial_view: (elevation, azimuth) in degrees
        show_grid: Whether to show grid lines
        show_legend: Whether to show legend
    """
    
    def __init__(self):
        self.display_mode: str = 'points'  # or 'cubes'
        self.figure_size: Tuple[int, int] = (14, 10)
        self.initial_view: Tuple[int, int] = (20, -60)  # (elev, azim)
        self.show_grid: bool = True
        self.show_legend: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'display_mode': self.display_mode,
            'figure_size': self.figure_size,
            'initial_view': self.initial_view,
            'show_grid': self.show_grid,
            'show_legend': self.show_legend
        }


class MappingConfig:
    """
    Main configuration class that combines all sub-configurations.
    
    This class provides a centralized configuration management system
    for the sonar mapping application.
    """
    
    def __init__(self):
        self.sonar = SonarConfig()
        self.grid = GridConfig()
        self.sensor_pose = SensorPoseConfig()
        self.data = DataConfig()
        self.visualization = VisualizationConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all configurations to nested dictionary."""
        return {
            'sonar': self.sonar.to_dict(),
            'grid': self.grid.to_dict(),
            'sensor_pose': self.sensor_pose.to_dict(),
            'data': self.data.to_dict(),
            'visualization': self.visualization.to_dict()
        }
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        
        print("\nSonar Configuration:")
        print(f"  Max range: {self.sonar.max_range}m")
        print(f"  FOV: {self.sonar.fov_degrees}°")
        print(f"  Vertical aperture: {self.sonar.vertical_aperture_degrees}°")
        print(f"  Intensity threshold: {self.sonar.intensity_threshold}")
        print(f"  Min range: {self.sonar.min_range}m")
        
        print("\nGrid Configuration:")
        print(f"  Resolution: {self.grid.resolution*1000:.0f}mm")
        print(f"  X range: {self.grid.x_range}")
        print(f"  Y range: {self.grid.y_range}")
        print(f"  Z range: {self.grid.z_range}")
        
        print("\nSensor Pose Configuration:")
        print(f"  Position: {self.sensor_pose.position}")
        print(f"  Pose (H,T,R): {self.sensor_pose.pose}°")
        
        print("\nData Configuration:")
        print(f"  Bag file: {self.data.bag_file}")
        print(f"  Output dir: {self.data.output_dir}")
        
        print("\nVisualization Configuration:")
        print(f"  Display mode: {self.visualization.display_mode}")
        print(f"  Figure size: {self.visualization.figure_size}")
        
        print("="*60)


# Create default configuration instance
default_config = MappingConfig()