#!/usr/bin/env python3
"""
Octree-based Sonar Mapping Library
Core classes for 3D sonar mapping with octree storage

Extracted from octree_mapper_proper.py for modular use.
This file contains only the core mapping functionality without ROS dependencies.

Author: Sonar 3D Reconstruction Team
Date: 2025
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Any


class SimpleOctree:
    """Simple dictionary-based octree for sparse voxel storage"""
    
    def __init__(self, resolution: float = 0.03, dynamic_expansion: bool = False):
        self.resolution = resolution
        self.voxels = {}  # (i,j,k) -> log_odds
        self.min_bounds = np.array([float('inf')] * 3)
        self.max_bounds = np.array([float('-inf')] * 3)
        self.dynamic_expansion = dynamic_expansion
        
    def world_to_key(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        i = int(np.floor(x / self.resolution))
        j = int(np.floor(y / self.resolution))
        k = int(np.floor(z / self.resolution))
        return (i, j, k)
    
    def key_to_world(self, key: Tuple[int, int, int]) -> Tuple[float, float, float]:
        x = (key[0] + 0.5) * self.resolution
        y = (key[1] + 0.5) * self.resolution
        z = (key[2] + 0.5) * self.resolution
        return (x, y, z)
    
    def update(self, x: float, y: float, z: float, log_odds_delta: float):
        # Check if within bounds (only if not in dynamic expansion mode)
        if not self.dynamic_expansion and not np.isinf(self.min_bounds[0]):  # Bounds are initialized and not dynamic
            if (x < self.min_bounds[0] or x > self.max_bounds[0] or
                y < self.min_bounds[1] or y > self.max_bounds[1] or
                z < self.min_bounds[2] or z > self.max_bounds[2]):
                return  # Skip voxels outside bounds
        
        key = self.world_to_key(x, y, z)
        if key not in self.voxels:
            self.voxels[key] = 0.0
        self.voxels[key] += log_odds_delta
        self.voxels[key] = np.clip(self.voxels[key], -10.0, 10.0)
        
        # Update bounds: always if dynamic expansion, only if not pre-set otherwise
        if self.dynamic_expansion or np.isinf(self.min_bounds[0]):
            self.min_bounds = np.minimum(self.min_bounds, [x, y, z])
            self.max_bounds = np.maximum(self.max_bounds, [x, y, z])
    
    def get_log_odds(self, x: float, y: float, z: float) -> float:
        key = self.world_to_key(x, y, z)
        return self.voxels.get(key, 0.0)
    
    def get_probability(self, x: float, y: float, z: float) -> float:
        log_odds = self.get_log_odds(x, y, z)
        return 1.0 / (1.0 + np.exp(-log_odds))


class OctreeBasedSonarMapperWithBoat:
    """
    Octree-based sonar occupancy grid mapper using intensity-based ray processing
    """
    
    def __init__(self, 
                 resolution: float = 0.03,
                 sensor_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 sensor_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 max_range: float = 5.0,
                 fov_degrees: float = 130.0,
                 vertical_aperture_degrees: float = 20.0,
                 intensity_threshold: int = 35,
                 min_range: float = 0.5,
                 dynamic_expansion: bool = True,
                 fov_margin_degrees: float = 5.0,
                 log_odds_occupied: float = 1.5,
                 log_odds_free: float = -2.0):
        """
        Initialize octree-based sonar mapper.
        
        Args:
            resolution: Voxel resolution in meters
            sensor_position: (x, y, z) sensor position relative to boat
            sensor_pose: (heading, tilt, roll) in degrees
            max_range: Maximum sonar range in meters
            fov_degrees: Horizontal field of view in degrees
            vertical_aperture_degrees: Vertical beam aperture in degrees
            intensity_threshold: Threshold for occupied detection (0-255)
            min_range: Minimum range to filter noise in meters
            dynamic_expansion: If True, map expands dynamically
            fov_margin_degrees: FOV margin to exclude from processing (degrees)
            log_odds_occupied: Log-odds weight for occupied space updates
            log_odds_free: Log-odds weight for free space updates
        """
        # Initialize octree for sparse storage
        self.dynamic_expansion = dynamic_expansion
        self.octree = SimpleOctree(resolution, dynamic_expansion=dynamic_expansion)
        self.resolution = resolution
        self.sensor_position = sensor_position
        self.max_range = max_range
        self.fov_degrees = fov_degrees
        self.fov_margin_degrees = fov_margin_degrees
        self.vertical_aperture_degrees = vertical_aperture_degrees
        self.vertical_aperture_radians = np.radians(vertical_aperture_degrees)
        self.intensity_threshold = intensity_threshold
        self.min_range = min_range
        
        # Calculate effective FOV after margin
        self.effective_fov_degrees = fov_degrees - 2 * fov_margin_degrees
        
        # Ray processing parameters (configurable)
        self.log_odds_free = log_odds_free  # Negative update for free space
        self.log_odds_occupied = log_odds_occupied  # Positive update for occupied
        
        # Sonar orientation parameters (heading, tilt, roll)
        self.sensor_pose = sensor_pose
        self.heading_degrees = sensor_pose[0]  # Yaw (rotation around Z axis)
        self.tilt_degrees = sensor_pose[1]     # Pitch (rotation around Y axis)  
        self.roll_degrees = sensor_pose[2]     # Roll (rotation around X axis)
        
        # Boat pose state (for 3-stage transformation)
        self.boat_position = np.array([0.0, 0.0, 0.0])
        self.boat_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        
        # Transformation matrices for 3-stage transformation
        self.sonar_to_boat_transform = np.eye(4)
        self.boat_to_map_transform = np.eye(4)
        
        # Initialize coordinate transformation matrices
        self.update_transform_matrix()
    
    def create_sonar_to_boat_transform(self, 
                                       heading_deg: float = 0.0,
                                       tilt_deg: float = 0.0,
                                       roll_deg: float = 0.0,
                                       offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """
        Create transformation from sonar to boat coordinates.
        
        Sonar frame: +X forward, +Y right, +Z down (standard sonar coordinates)
        Boat frame: +X forward, +Y right, +Z up (standard boat coordinates)
        
        tilt_deg: Rotation around Y axis (pitch)
                  0° = forward-looking
                  -90° = downward-looking (results in xb=ys, yb=-zs, zb=-xs)
        """
        # Apply full 3D rotation (Z-Y-X intrinsic rotations)
        # Convert to radians
        yaw_rad = np.radians(heading_deg)    # Z axis rotation
        pitch_rad = np.radians(tilt_deg)     # Y axis rotation
        roll_rad = np.radians(roll_deg)      # X axis rotation
        
        # Rotation matrices
        Rz = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        Ry = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad),  np.cos(roll_rad)]
        ])
        
        # Combined rotation: R = Rz * Ry * Rx (Z-Y-X order)
        R_sonar_to_boat = Rz @ Ry @ Rx
        
        # Build 4x4 homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R_sonar_to_boat
        T[:3, 3] = offset
        
        return T
    
    def create_boat_to_map_transform(self,
                                     position: Tuple[float, float, float],
                                     quaternion: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Create transformation from boat to map coordinates using odometry.
        
        Args:
            position: (x, y, z) position from odometry
            quaternion: (qx, qy, qz, qw) orientation from odometry
        """
        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = quaternion
        
        # Quaternion to rotation matrix conversion
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        # Build 4x4 homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        return T
    
    def set_boat_pose_from_odometry(self,
                                    position: Tuple[float, float, float],
                                    quaternion: Tuple[float, float, float, float]):
        """
        Set boat pose from odometry data and update transformations.
        
        Args:
            position: (x, y, z) position from odometry
            quaternion: (qx, qy, qz, qw) orientation from odometry
        """
        self.boat_position = np.array(position)
        self.boat_quaternion = np.array(quaternion)
        
        # Update transformation matrices
        self.update_transform_matrix()
    
    def update_transform_matrix(self):
        """Update transformation matrices based on current sonar position and orientation"""
        # Create sonar to boat transformation
        self.sonar_to_boat_transform = self.create_sonar_to_boat_transform(
            heading_deg=self.heading_degrees,
            tilt_deg=self.tilt_degrees,
            roll_deg=self.roll_degrees,
            offset=self.sensor_position
        )
        
        # Create boat to map transformation
        self.boat_to_map_transform = self.create_boat_to_map_transform(
            self.boat_position,
            self.boat_quaternion
        )
        
        # Chain transformations: Sonar → Boat → Map
        self.sonar_to_map_transform = self.boat_to_map_transform @ self.sonar_to_boat_transform
        
        # Compute inverse for map to sonar transformation
        self.map_to_sonar_transform = np.linalg.inv(self.sonar_to_map_transform)
    
    def sonar_to_map_coords(self, sonar_x: float, sonar_y: float, sonar_z: float) -> np.ndarray:
        """Transform coordinates from sonar frame to map frame"""
        sonar_point = np.array([sonar_x, sonar_y, sonar_z, 1.0])
        map_point = self.sonar_to_map_transform @ sonar_point
        return map_point[:3]
    
    def map_to_sonar_coords(self, map_x: float, map_y: float, map_z: float) -> np.ndarray:
        """Transform coordinates from map frame to sonar frame"""
        map_point = np.array([map_x, map_y, map_z, 1.0])
        sonar_point = self.map_to_sonar_transform @ map_point
        return sonar_point[:3]
    
    def is_bearing_in_valid_fov(self, bearing_angle: float) -> bool:
        """Check if bearing angle is within valid FOV (excluding margins)"""
        effective_half_fov = np.radians(self.effective_fov_degrees / 2)
        return abs(bearing_angle) <= effective_half_fov
    
    def process_polar_image_voxel_centric(self, polar_image: np.ndarray) -> Dict[str, Any]:
        """
        Process polar image using hybrid voxel-centric approach for octree.
        First finds occupied regions with rays, then fills full aperture volume.
        
        Args:
            polar_image: (range_bins, bearing_bins) polar sonar image
            
        Returns:
            Processing statistics
        """
        # Processing with hybrid voxel-centric approach (quiet mode)
        start_time = time.time()
        
        # Get image dimensions
        range_bins, bearing_bins = polar_image.shape
        range_resolution = self.max_range / range_bins
        bearing_resolution = np.radians(self.fov_degrees) / bearing_bins
        half_fov = np.radians(self.fov_degrees / 2)
        half_aperture = self.vertical_aperture_radians / 2
        
        # Processing counters
        free_updates = 0
        occupied_updates = 0
        rays_processed = 0
        
        # Pre-compute bearing angles
        bearing_angles = np.linspace(-half_fov, half_fov, bearing_bins)
        min_range_idx = int(self.min_range / range_resolution)
        
        # Step 1: Find all occupied regions using sparse ray sampling
        # Finding occupied regions (quiet)
        occupied_regions = []  # List of (bearing_idx, range_idx, intensity)
        
        for bearing_idx in range(bearing_bins):
            bearing_angle = bearing_angles[bearing_idx]
            
            # Skip bearings outside valid FOV (FOV margin filtering)
            if not self.is_bearing_in_valid_fov(bearing_angle):
                continue
            
            for range_idx in range(min_range_idx, range_bins):
                if polar_image[range_idx, bearing_idx] >= self.intensity_threshold:
                    occupied_regions.append((bearing_idx, range_idx, polar_image[range_idx, bearing_idx]))
        
        # Found occupied points (quiet)
        
        # Step 2: Process each occupied region with full aperture volume
        # Filling aperture volumes around occupied points (quiet)
        processed_voxels = set()  # Track processed voxels to avoid duplicates
        
        for bearing_idx, range_idx, intensity in occupied_regions:
            bearing_angle = bearing_angles[bearing_idx]
            range_m = range_idx * range_resolution
            
            # Calculate aperture volume at this range
            vertical_spread = range_m * np.tan(half_aperture)
            num_vertical_steps = max(3, int(vertical_spread / self.resolution))
            
            # Fill the aperture cone with voxels
            for v_step in range(-num_vertical_steps, num_vertical_steps + 1):
                # Calculate vertical angle within aperture
                vertical_angle = (v_step / max(1, num_vertical_steps)) * half_aperture
                
                # Calculate position in sonar coordinates
                # Sonar coordinates: +X forward, +Y right, +Z down
                # Bearing: rotation around Z axis (in XY plane, horizontal)
                # Vertical: rotation around Y axis (in XZ plane, elevation)
                sonar_x = range_m * np.cos(vertical_angle) * np.cos(bearing_angle)
                sonar_y = range_m * np.cos(vertical_angle) * np.sin(bearing_angle)
                sonar_z = range_m * np.sin(vertical_angle)
                
                # Transform to map coordinates
                map_pos = self.sonar_to_map_coords(sonar_x, sonar_y, sonar_z)
                voxel_key = self.octree.world_to_key(*map_pos)
                
                if voxel_key not in processed_voxels:
                    self.octree.update(map_pos[0], map_pos[1], map_pos[2], self.log_odds_occupied)
                    processed_voxels.add(voxel_key)
                    occupied_updates += 1
        
        # Step 3: Process free space (before obstacles)
        # Processing free space (quiet)
        
        # Process ALL bearings for accurate coverage (256 bearings total)
        bearing_sample_step = 1  # Process every bearing
        
        for bearing_idx in range(0, bearing_bins, bearing_sample_step):
            bearing_angle = bearing_angles[bearing_idx]
            
            # Skip bearings outside valid FOV (FOV margin filtering)
            if not self.is_bearing_in_valid_fov(bearing_angle):
                continue
            
            # Find first obstacle in this bearing
            first_obstacle_range = self.max_range
            for range_idx in range(min_range_idx, range_bins):
                if polar_image[range_idx, bearing_idx] >= self.intensity_threshold:
                    first_obstacle_range = range_idx * range_resolution
                    break
            
            # Sample free space before obstacle with vertical spreading
            num_range_samples = min(20, int(first_obstacle_range / range_resolution))
            for r_sample in range(num_range_samples):
                range_m = self.min_range + (r_sample / max(1, num_range_samples - 1)) * (first_obstacle_range - self.min_range)
                
                # Add vertical spread for free space too (but less dense)
                vertical_spread = range_m * np.tan(half_aperture)
                num_vertical_steps = max(2, int(vertical_spread / (self.resolution * 2)))  # Sparser for free space
                
                for v_step in range(-num_vertical_steps, num_vertical_steps + 1):
                    # Calculate vertical angle within aperture
                    vertical_angle = (v_step / max(1, num_vertical_steps)) * half_aperture
                    
                    # Calculate position in sonar coordinates (same as occupied)
                    # Sonar coordinates: +X forward, +Y right, +Z down
                    # Bearing: rotation around Z axis (in XY plane, horizontal)
                    # Vertical: rotation around Y axis (in XZ plane, elevation)
                    sonar_x = range_m * np.cos(vertical_angle) * np.cos(bearing_angle)
                    sonar_y = range_m * np.cos(vertical_angle) * np.sin(bearing_angle)
                    sonar_z = range_m * np.sin(vertical_angle)
                    
                    map_pos = self.sonar_to_map_coords(sonar_x, sonar_y, sonar_z)
                    voxel_key = self.octree.world_to_key(*map_pos)
                    
                    if voxel_key not in processed_voxels:
                        self.octree.update(map_pos[0], map_pos[1], map_pos[2], self.log_odds_free)
                        processed_voxels.add(voxel_key)
                        free_updates += 1
            
            rays_processed += 1
        
        processing_time = time.time() - start_time
        
        stats = {
            "processing_time": processing_time,
            "rays_processed": rays_processed,
            "free_updates": free_updates,
            "occupied_updates": occupied_updates,
            "voxels_created": len(self.octree.voxels),
            "total_voxels": len(self.octree.voxels)
        }
        
        return stats
    
    def get_classified_voxels(self) -> Dict[str, np.ndarray]:
        """
        Get voxels classified by probability from the octree.
        
        Returns:
            Dictionary with 'free', 'unknown', 'occupied' voxel coordinates
        """
        free_coords = []
        unknown_coords = []
        occupied_coords = []
        
        # Iterate through all voxels in the octree
        for key, log_odds in self.octree.voxels.items():
            world_pos = self.octree.key_to_world(key)
            probability = 1.0 / (1.0 + np.exp(-log_odds))
            
            if probability < 0.3:
                free_coords.append(list(world_pos))
            elif probability > 0.7:
                occupied_coords.append(list(world_pos))
            else:
                unknown_coords.append(list(world_pos))
        
        result = {
            'free': np.array(free_coords) if free_coords else np.array([]).reshape(0, 3),
            'unknown': np.array(unknown_coords) if unknown_coords else np.array([]).reshape(0, 3),
            'occupied': np.array(occupied_coords) if occupied_coords else np.array([]).reshape(0, 3)
        }
        
        return result
    
    def get_classified_voxels_with_probability(self, min_probability: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get voxels with probability above threshold with their probability values.
        
        Args:
            min_probability: Minimum probability to include voxel (default 0.5)
                            Free space (prob < 0.5) is filtered out
        
        Returns:
            Tuple of (points, probabilities) arrays for occupied/unknown voxels only
        """
        points = []
        probabilities = []
        
        # Iterate through all voxels in the octree
        for key, log_odds in self.octree.voxels.items():
            # Convert log-odds to probability
            probability = 1.0 / (1.0 + np.exp(-log_odds))
            
            # Skip free space voxels (probability < threshold)
            if probability < min_probability:
                continue
                
            world_pos = self.octree.key_to_world(key)
            points.append(list(world_pos))
            probabilities.append(probability)
        
        return (np.array(points) if points else np.array([]).reshape(0, 3),
                np.array(probabilities) if probabilities else np.array([]))