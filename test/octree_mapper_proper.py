#!/usr/bin/env python3
"""
Octree-based Sonar Occupancy Grid Mapper with Boat Coordinate Transformation
- Uses sparse octree storage instead of dense grid
- Implements ray-casting based sonar occupancy mapping
- Processes intensity along each ray to determine free/occupied/shadow zones
- Supports realistic sonar physics with uncertainty and shadowing
- Adds 3-stage coordinate transformation: Sonar → Boat → Map
- Dynamic map expansion without fixed boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
import csv


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
                 initial_map_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
                 dynamic_expansion: bool = False,
                 fov_margin_degrees: float = 5.0):
        """
        Initialize octree-based sonar mapper.
        
        The default configuration (pose = 0,0,0) creates a downward-looking sonar,
        which is the typical configuration for seafloor mapping.
        Map expands dynamically as needed - no fixed boundaries.
        
        Args:
            resolution: Voxel resolution in meters
            sensor_position: (x, y, z) sensor position in map frame
            sensor_pose: (heading, tilt, roll) in degrees:
                - heading: Yaw angle (rotation around Z axis)
                - tilt: Pitch angle (rotation around Y axis)
                - roll: Roll angle (rotation around X axis)
            max_range: Maximum sonar range in meters
            fov_degrees: Horizontal field of view in degrees
            vertical_aperture_degrees: Vertical beam aperture in degrees
            intensity_threshold: Threshold for occupied detection (0-255)
            min_range: Minimum range to filter noise in meters
            initial_map_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max)) initial map bounds
                                If None, map starts empty and expands dynamically
            dynamic_expansion: If True, ignore initial_map_bounds and expand dynamically as robot moves
            fov_margin_degrees: FOV margin to exclude from processing (degrees). Excludes noisy edges
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
        self.initial_map_bounds = initial_map_bounds
        
        # Calculate effective FOV after margin
        self.effective_fov_degrees = fov_degrees - 2 * fov_margin_degrees
        
        # Initialize map bounds if specified and not in dynamic mode
        if dynamic_expansion:
            map_mode = "Dynamic expansion (starts small, expands with robot movement)"
        elif initial_map_bounds is not None:
            # Pre-set the octree bounds
            x_range, y_range, z_range = initial_map_bounds
            self.octree.min_bounds = np.array([x_range[0], y_range[0], z_range[0]])
            self.octree.max_bounds = np.array([x_range[1], y_range[1], z_range[1]])
            x_size = x_range[1] - x_range[0]
            y_size = y_range[1] - y_range[0]
            z_size = z_range[1] - z_range[0]
            map_mode = f"Fixed bounds: X[{x_range[0]:.1f},{x_range[1]:.1f}] Y[{y_range[0]:.1f},{y_range[1]:.1f}] Z[{z_range[0]:.1f},{z_range[1]:.1f}]"
        else:
            map_mode = "Dynamic (auto-expanding)"
        
        # Print initialization summary
        print(f"Initializing Octree-based Sonar Mapper:")
        print(f"  Map mode: {map_mode}")
        print(f"  Resolution: {resolution}m ({resolution*1000:.0f}mm)")
        print(f"  Sparse voxel storage (memory efficient)")
        print(f"  Sensor position: {sensor_position}")
        print(f"  Sensor pose (H,T,R): {sensor_pose}°")
        print(f"  Sonar parameters: {max_range}m range, {fov_degrees}° FOV ({self.effective_fov_degrees}° effective), {vertical_aperture_degrees}° vertical aperture")
        if fov_margin_degrees > 0:
            print(f"  FOV filtering: ±{fov_margin_degrees}° margin (edge exclusion)")
        
        # Log-odds are stored in the octree, not a dense grid
        
        # Initialize averaging buffers for frame processing
        self.update_buffer = {}  # Dictionary to store accumulated updates
        self.update_counts = {}  # Dictionary to store update counts
        
        # Ray processing parameters
        self.log_odds_free = -2.0  # Strong negative update for free space
        self.log_odds_occupied = 1.5  # Positive update for occupied
        
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
        
        print(f"  Log-odds updates: free={self.log_odds_free}, occupied={self.log_odds_occupied}")
        print(f"  Mapper initialized successfully")
    
    def create_sonar_to_map_transform(self, position: Tuple[float, float, float],
                                      heading_deg: float = 0, 
                                      tilt_deg: float = 0,
                                      roll_deg: float = 0) -> np.ndarray:
        """
        Create 4x4 homogeneous transformation matrix from sonar to map coordinates
        
        Sonar coordinate system (looking forward):
        - +X: Forward direction
        - +Y: Right side
        - +Z: Downward
        
        Map coordinate system:
        - X: [-0.75, 0.75] (side to side)
        - Y: [-0.75, 0.75] (front to back) 
        - Z: [-1.5, 0] (depth, negative is underwater)
        
        When sonar is at (0,0,0) looking straight down:
        - Sonar +X (forward) → Map -Z (downward)
        - Sonar +Y (right) → Map +X (right)
        - Sonar +Z (down) → Map +Y (backward)
        """
        # Convert angles to radians
        heading_rad = np.radians(heading_deg)
        tilt_rad = np.radians(tilt_deg)
        roll_rad = np.radians(roll_deg)
        
        # Base rotation matrix for downward-looking sonar (default orientation)
        # When pose=(0,0,0), the sonar looks downward:
        # Sonar +X (forward) -> Map -Z (down)
        # Sonar +Y (right) -> Map +Y (right)
        # Sonar +Z (down) -> Map +X (forward)
        R_base = np.array([
            [0, 0, 1],   # Map X = Sonar Z
            [0, 1, 0],   # Map Y = Sonar Y
            [-1, 0, 0]   # Map Z = -Sonar X
        ])
        
        # Additional rotation matrices for orientation changes
        # Roll (rotation around X axis - sonar forward axis)
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        
        # Pitch/Tilt (rotation around Y axis - sonar right axis)
        R_pitch = np.array([
            [np.cos(tilt_rad), 0, np.sin(tilt_rad)],
            [0, 1, 0],
            [-np.sin(tilt_rad), 0, np.cos(tilt_rad)]
        ])
        
        # Yaw/Heading (rotation around Z axis - sonar down axis)
        R_yaw = np.array([
            [np.cos(heading_rad), -np.sin(heading_rad), 0],
            [np.sin(heading_rad), np.cos(heading_rad), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Apply rotations in order Roll → Pitch → Yaw
        # Then apply base transformation
        R_sonar = R_yaw @ R_pitch @ R_roll
        R_combined = R_base @ R_sonar
        
        # Create 4x4 homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R_combined
        T[:3, 3] = position
        
        return T
    
    def create_sonar_to_boat_transform(self, 
                                       tilt_deg: float = 0.0,
                                       offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """
        Create transformation from sonar to boat coordinates.
        
        Sonar frame: +X forward, +Y right, +Z down
        Boat frame: xb=ys, yb=-zs, zb=-xs (corrected)
        """
        # Base rotation matrix for the corrected transformation
        R_sonar_to_boat = np.array([
            [0,  1,  0],   # boat X = sonar Y
            [0,  0, -1],   # boat Y = -sonar Z
            [-1, 0,  0]    # boat Z = -sonar X
        ])
        
        # Apply tilt if specified (rotation around Y axis)
        if abs(tilt_deg) > 0.001:
            tilt_rad = np.radians(tilt_deg)
            R_tilt = np.array([
                [np.cos(tilt_rad), 0, np.sin(tilt_rad)],
                [0, 1, 0],
                [-np.sin(tilt_rad), 0, np.cos(tilt_rad)]
            ])
            R_sonar_to_boat = R_sonar_to_boat @ R_tilt
        
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
    
    def load_odometry_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """
        Load odometry data from CSV file.
        
        Returns:
            List of odometry measurements with timestamp, position, and quaternion
        """
        odometry_data = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                odometry_data.append({
                    'timestamp': float(row['timestamp_sec']) + float(row['timestamp_nanosec']) / 1e9,
                    'position': (float(row['x']), float(row['y']), float(row['z'])),
                    'quaternion': (float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw']))
                })
        
        return odometry_data
    
    def update_transform_matrix(self):
        """Update transformation matrices based on current sonar position and orientation"""
        # Create sonar to boat transformation
        self.sonar_to_boat_transform = self.create_sonar_to_boat_transform(
            tilt_deg=self.tilt_degrees,
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
    
    def set_sensor_pose(self, sensor_pose: Tuple[float, float, float] = None,
                       heading: float = None, 
                       tilt: float = None, 
                       roll: float = None):
        """
        Update sensor orientation.
        
        Can either provide a complete pose tuple or individual angle updates.
        
        Args:
            sensor_pose: (heading, tilt, roll) tuple in degrees
            heading: Yaw angle in degrees (rotation around Z)
            tilt: Pitch angle in degrees (rotation around Y)
            roll: Roll angle in degrees (rotation around X)
        """
        if sensor_pose is not None:
            self.sensor_pose = sensor_pose
            self.heading_degrees = sensor_pose[0]
            self.tilt_degrees = sensor_pose[1]
            self.roll_degrees = sensor_pose[2]
        else:
            if heading is not None:
                self.heading_degrees = heading
            if tilt is not None:
                self.tilt_degrees = tilt
            if roll is not None:
                self.roll_degrees = roll
            self.sensor_pose = (self.heading_degrees, self.tilt_degrees, self.roll_degrees)
        
        # Update transformation matrices with new pose
        self.update_transform_matrix()
        
        print(f"Sensor pose updated: (H,T,R) = {self.sensor_pose}°")
    
    def set_sensor_position(self, x: float = None, y: float = None, z: float = None):
        """
        Update sensor position
        
        Args:
            x: X coordinate
            y: Y coordinate  
            z: Z coordinate
        """
        current_x, current_y, current_z = self.sensor_position
        
        if x is not None:
            current_x = x
        if y is not None:
            current_y = y
        if z is not None:
            current_z = z
            
        self.sensor_position = (current_x, current_y, current_z)
        
        # Update transformation matrices with new position
        self.update_transform_matrix()
        
        print(f"Sensor position updated: {self.sensor_position}")
    
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
    
    
    def world_to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates to voxel key - no bounds checking with octree"""
        return self.octree.world_to_key(x, y, z)
    
    def grid_to_world(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Convert voxel key to world coordinates (center of voxel)"""
        return self.octree.key_to_world((i, j, k))
    
    def buffer_update(self, x: float, y: float, z: float, log_odds_delta: float):
        """Buffer an update for averaging later"""
        key = self.world_to_grid(x, y, z)
        if key not in self.update_buffer:
            self.update_buffer[key] = 0.0
            self.update_counts[key] = 0
        self.update_buffer[key] += log_odds_delta
        self.update_counts[key] += 1
    
    def apply_buffered_updates(self):
        """Apply all buffered updates using weighted averaging"""
        for key, accumulated_update in self.update_buffer.items():
            count = self.update_counts[key]
            # Use weighted average: reduce effect only if many updates
            # If count is 1-3, use full update. Otherwise, scale down logarithmically
            if count <= 3:
                weighted_update = accumulated_update / count
            else:
                # Scale down by log of count to prevent over-dilution
                weight = 1.0 + np.log(count)
                weighted_update = accumulated_update / weight
            
            # Convert key to world coordinates and update octree
            x, y, z = self.octree.key_to_world(key)
            self.octree.update(x, y, z, weighted_update)
        
        # Clear buffers for next frame
        self.update_buffer.clear()
        self.update_counts.clear()
    
    def update_log_odds(self, x: float, y: float, z: float, log_odds_delta: float):
        """Update log-odds for a single voxel (direct update, no averaging)"""
        self.octree.update(x, y, z, log_odds_delta)
    
    def get_probability(self, x: float, y: float, z: float) -> float:
        """Get occupancy probability for a world coordinate"""
        return self.octree.get_probability(x, y, z)
    
    def ray_cast_3d(self, start: Tuple[float, float, float], 
                    end: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        3D ray casting using linear interpolation (optimized)
        Returns list of world coordinates along the ray
        """
        start_array = np.array(start)
        end_array = np.array(end)
        
        # Calculate ray direction and length
        direction = end_array - start_array
        length = np.linalg.norm(direction)
        
        if length < self.resolution:
            return []
        
        # Use larger step size for efficiency (double resolution)
        step_size = self.resolution * 2.0
        num_steps = max(1, int(length / step_size))
        
        # Generate points along the ray using vectorized operations
        if num_steps == 1:
            return [(start_array[0], start_array[1], start_array[2])]
        
        t_values = np.linspace(0, 1, num_steps)
        points_array = start_array + np.outer(t_values, direction)
        
        return [(p[0], p[1], p[2]) for p in points_array]
    
    def get_all_voxels_in_aperture(self, bearing_angle: float, range_m: float) -> List[Tuple[int, int, int]]:
        """
        Get ALL voxel indices within the vertical aperture cone at a specific range
        (No sampling - finds all voxels geometrically)
        
        Args:
            bearing_angle: Horizontal bearing angle in radians (in sonar coordinates)
            range_m: Range in meters
            
        Returns:
            List of voxel indices (i, j, k) within the aperture
        """
        half_aperture = self.vertical_aperture_radians / 2
        
        # Calculate ray center in sonar coordinates
        # In sonar frame: bearing angle is in X-Y plane from +X axis
        sonar_ray_x = range_m * np.cos(bearing_angle)
        sonar_ray_y = range_m * np.sin(bearing_angle)
        sonar_ray_z = 0  # Center ray is horizontal in sonar frame
        
        # Transform ray center from sonar to map coordinates
        map_center = self.sonar_to_map_coords(sonar_ray_x, sonar_ray_y, sonar_ray_z)
        world_center_x, world_center_y, world_center_z = map_center
        
        # Get center voxel
        i_center, j_center, k_center = self.world_to_grid(world_center_x, world_center_y, world_center_z)
        if i_center < 0:
            return []
        
        # Calculate vertical spread at this range
        vertical_radius = range_m * np.tan(half_aperture)
        
        # Calculate search radius in voxel units
        search_radius_voxels = int(np.ceil(vertical_radius / self.resolution)) + 2
        
        voxels = []
        unique_voxels = set()
        
        # Check all voxels in the potential cone region
        # Increase search range to account for all orientations
        for di in range(-search_radius_voxels, search_radius_voxels + 1):
            for dj in range(-search_radius_voxels, search_radius_voxels + 1):
                for dk in range(-search_radius_voxels, search_radius_voxels + 1):
                    i = i_center + di
                    j = j_center + dj
                    k = k_center + dk
                    
                    # No bounds checking with octree - it expands dynamically
                    
                    # Skip if already checked
                    if (i, j, k) in unique_voxels:
                        continue
                    
                    # Get world coordinates of voxel center
                    voxel_x, voxel_y, voxel_z = self.grid_to_world(i, j, k)
                    
                    # Transform voxel to sonar coordinates
                    sonar_voxel = self.map_to_sonar_coords(voxel_x, voxel_y, voxel_z)
                    voxel_sonar_x, voxel_sonar_y, voxel_sonar_z = sonar_voxel
                    
                    # Distance from sensor in sonar coordinates
                    dist = np.sqrt(voxel_sonar_x**2 + voxel_sonar_y**2 + voxel_sonar_z**2)
                    
                    # Check if at correct range (within voxel size tolerance)
                    if abs(dist - range_m) > self.resolution * 1.5:
                        continue
                    
                    # Check bearing angle in sonar coordinates
                    if dist > 0:
                        # Calculate bearing in sonar X-Y plane
                        voxel_bearing = np.arctan2(voxel_sonar_y, voxel_sonar_x)
                        bearing_tolerance = 2.0 * self.resolution / range_m
                        if abs(voxel_bearing - bearing_angle) > bearing_tolerance:
                            continue
                        
                        # Check vertical angle (within aperture) in sonar coordinates
                        horizontal_dist = np.sqrt(voxel_sonar_x**2 + voxel_sonar_y**2)
                        if horizontal_dist > 0:
                            vertical_angle = np.arctan2(voxel_sonar_z, horizontal_dist)
                        else:
                            vertical_angle = 0 if voxel_sonar_z == 0 else (np.pi/2 if voxel_sonar_z > 0 else -np.pi/2)
                        
                        if abs(vertical_angle) <= half_aperture:
                            unique_voxels.add((i, j, k))
                            voxels.append((i, j, k))
        
        return voxels

    def process_polar_image(self, polar_image: np.ndarray) -> Dict[str, Any]:
        """
        Process a single polar sonar image using volumetric ray-based approach
        
        Args:
            polar_image: (range_bins, bearing_bins) polar sonar image
            
        Returns:
            Processing statistics and metadata
        """
        print(f"\n🔄 Processing polar image with volumetric ray-based mapping...")
        start_time = time.time()
        
        # Get image dimensions
        range_bins, bearing_bins = polar_image.shape
        print(f"   📊 Image size: {range_bins} range bins × {bearing_bins} bearing bins")
        
        # Calculate range and bearing resolutions
        range_resolution = self.max_range / range_bins
        bearing_resolution = np.radians(self.fov_degrees) / bearing_bins
        half_fov = np.radians(self.fov_degrees / 2)
        
        print(f"   📏 Range resolution: {range_resolution:.3f}m per bin")
        print(f"   📐 Bearing resolution: {np.degrees(bearing_resolution):.2f}° per bin")
        
        # Processing counters
        rays_processed = 0
        free_updates = 0
        occupied_updates = 0
        shadow_zones = 0
        
        # Pre-allocate arrays for batch processing
        bearing_angles = np.linspace(-half_fov, half_fov, bearing_bins)
        
        # Process each bearing ray
        for bearing_idx in range(bearing_bins):
            bearing_angle = bearing_angles[bearing_idx]
            
            # Skip bearings outside valid FOV (FOV margin filtering)
            if not self.is_bearing_in_valid_fov(bearing_angle):
                continue
            
            min_range_idx = int(self.min_range / range_resolution)
            
            # Track previous state for shadow detection
            prev_was_occupied = False
            shadow_count = 0
            
            # Process ranges along this ray
            for range_idx in range(min_range_idx, range_bins):
                range_m = range_idx * range_resolution
                
                # Skip if beyond grid bounds
                if range_m > 1.5:
                    break
                
                # Get current intensity
                current_intensity = polar_image[range_idx, bearing_idx]
                
                # Get ALL voxels within aperture at this range (no sampling)
                voxel_indices = self.get_all_voxels_in_aperture(bearing_angle, range_m)
                
                if len(voxel_indices) == 0:
                    continue
                
                # Determine update type based on intensity
                if current_intensity >= self.intensity_threshold:
                    # High intensity - always occupied (could be seafloor or new object)
                    for (i, j, k) in voxel_indices:
                        key = (i, j, k)
                        if key not in self.update_buffer:
                            self.update_buffer[key] = 0.0
                            self.update_counts[key] = 0
                        self.update_buffer[key] += self.log_odds_occupied
                        self.update_counts[key] += 1
                    occupied_updates += len(voxel_indices)
                    prev_was_occupied = True
                    
                elif prev_was_occupied:
                    # Low intensity immediately after high intensity = shadow zone
                    # Skip update (no information about this region)
                    shadow_count += 1
                    shadow_zones += len(voxel_indices)
                    prev_was_occupied = False  # Reset for next potential object
                    
                else:
                    # Low intensity not immediately after high intensity = free space
                    for (i, j, k) in voxel_indices:
                        key = (i, j, k)
                        if key not in self.update_buffer:
                            self.update_buffer[key] = 0.0
                            self.update_counts[key] = 0
                        self.update_buffer[key] += self.log_odds_free
                        self.update_counts[key] += 1
                    free_updates += len(voxel_indices)
                    prev_was_occupied = False
            
            rays_processed += 1
        
        print()  # New line after progress bar
        
        # Apply all buffered updates with averaging
        print(f"   📝 Applying {len(self.update_buffer):,} buffered updates...")
        self.apply_buffered_updates()
        
        # Calculate processing statistics
        processing_time = time.time() - start_time
        
        stats = {
            "processing_time": processing_time,
            "rays_processed": rays_processed,
            "free_updates": free_updates,
            "occupied_updates": occupied_updates,
            "shadow_zones": shadow_zones,
            "total_range_bins": range_bins * bearing_bins,
            "range_resolution": range_resolution,
            "bearing_resolution_deg": np.degrees(bearing_resolution)
        }
        
        print(f"   ✅ Volumetric ray-based processing completed in {processing_time:.3f}s")
        print(f"   📊 Statistics:")
        print(f"      Rays processed: {rays_processed:,}")
        print(f"      Free space updates: {free_updates:,}")
        print(f"      Occupied updates: {occupied_updates:,}")
        print(f"      Shadow zones: {shadow_zones:,}")
        
        return stats
    
    def process_polar_image_ray_based(self, polar_image: np.ndarray) -> Dict[str, Any]:
        """
        Process polar image using ray-based approach.
        Only creates voxels along rays, much more efficient for sparse octree.
        
        Args:
            polar_image: (range_bins, bearing_bins) polar sonar image
            
        Returns:
            Processing statistics
        """
        print(f"\n🔄 Processing polar image with ray-based octree approach...")
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
        
        # Process each bearing ray
        for bearing_idx in range(bearing_bins):
            bearing_angle = bearing_angles[bearing_idx]
            
            # Find first occupied range
            occupied_idx = -1
            for i in range(min_range_idx, range_bins):
                if polar_image[i, bearing_idx] >= self.intensity_threshold:
                    occupied_idx = i
                    break
            
            # Process vertical aperture samples (more samples for better quality)
            num_vertical_samples = 11  # Increased from 5 for better coverage
            for v_angle in np.linspace(-half_aperture, half_aperture, num_vertical_samples):
                # Process ranges along this ray
                for range_idx in range(min_range_idx, min(range_bins, occupied_idx + 10 if occupied_idx > 0 else range_bins)):
                    range_m = range_idx * range_resolution
                    
                    # Calculate position in sonar coordinates
                    horizontal_dist = range_m * np.cos(v_angle)
                    sonar_x = horizontal_dist * np.cos(bearing_angle)
                    sonar_y = horizontal_dist * np.sin(bearing_angle)
                    sonar_z = range_m * np.sin(v_angle)
                    
                    # Transform to map coordinates
                    map_pos = self.sonar_to_map_coords(sonar_x, sonar_y, sonar_z)
                    
                    # Determine update type
                    if occupied_idx > 0 and abs(range_idx - occupied_idx) <= 2:
                        # Near occupied range
                        self.octree.update(map_pos[0], map_pos[1], map_pos[2], self.log_odds_occupied)
                        occupied_updates += 1
                    elif occupied_idx < 0 or range_idx < occupied_idx:
                        # Free space
                        self.octree.update(map_pos[0], map_pos[1], map_pos[2], self.log_odds_free)
                        free_updates += 1
                    # else: shadow zone, skip
            
            rays_processed += 1
        
        print()  # New line after progress bar
        
        processing_time = time.time() - start_time
        
        stats = {
            "processing_time": processing_time,
            "rays_processed": rays_processed,
            "free_updates": free_updates,
            "occupied_updates": occupied_updates,
            "voxels_created": len(self.octree.voxels),
            "total_voxels": len(self.octree.voxels)
        }
        
        print(f"   ✅ Ray-based octree processing completed in {processing_time:.3f}s")
        print(f"   📊 Statistics:")
        print(f"      Rays processed: {rays_processed:,}")
        print(f"      Free space updates: {free_updates:,}")
        print(f"      Occupied updates: {occupied_updates:,}")
        print(f"      Total voxels created: {len(self.octree.voxels):,}")
        
        return stats
    
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
                v_offset = (v_step / max(1, num_vertical_steps)) * vertical_spread
                
                # Calculate position in sonar coordinates
                sonar_x = range_m * np.cos(bearing_angle)
                sonar_y = range_m * np.sin(bearing_angle)
                sonar_z = v_offset
                
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
                    v_offset = (v_step / max(1, num_vertical_steps)) * vertical_spread
                    
                    sonar_x = range_m * np.cos(bearing_angle)
                    sonar_y = range_m * np.sin(bearing_angle)
                    sonar_z = v_offset
                    
                    map_pos = self.sonar_to_map_coords(sonar_x, sonar_y, sonar_z)
                    voxel_key = self.octree.world_to_key(*map_pos)
                    
                    if voxel_key not in processed_voxels:
                        self.octree.update(map_pos[0], map_pos[1], map_pos[2], self.log_odds_free)
                        processed_voxels.add(voxel_key)
                        free_updates += 1
            
            rays_processed += 1
        
        print()  # New line after progress bars
        
        processing_time = time.time() - start_time
        
        stats = {
            "processing_time": processing_time,
            "rays_processed": rays_processed,
            "free_updates": free_updates,
            "occupied_updates": occupied_updates,
            "voxels_created": len(self.octree.voxels),
            "total_voxels": len(self.octree.voxels)
        }
        
        # Processing completed (quiet mode)
        
        return stats
    
    def _process_polar_image_voxel_centric_dense(self, polar_image: np.ndarray) -> Dict[str, Any]:
        """
        Process a single polar sonar image using voxel-centric approach
        (More efficient - processes all rays affecting each voxel together)
        
        Args:
            polar_image: (range_bins, bearing_bins) polar sonar image
            
        Returns:
            Processing statistics and metadata
        """
        print(f"\n🔄 Processing polar image with voxel-centric approach...")
        start_time = time.time()
        
        # Get image dimensions
        range_bins, bearing_bins = polar_image.shape
        print(f"   📊 Image size: {range_bins} range bins × {bearing_bins} bearing bins")
        
        # Calculate range and bearing resolutions
        range_resolution = self.max_range / range_bins
        bearing_resolution = np.radians(self.fov_degrees) / bearing_bins
        half_fov = np.radians(self.fov_degrees / 2)
        half_aperture = self.vertical_aperture_radians / 2
        
        print(f"   📏 Range resolution: {range_resolution:.3f}m per bin")
        print(f"   📐 Bearing resolution: {np.degrees(bearing_resolution):.2f}° per bin")
        
        # Pre-compute bearing angles for all rays
        bearing_angles = np.linspace(-half_fov, half_fov, bearing_bins)
        
        # Step 1: Find occupied ranges for all rays at once
        print(f"   🔍 Finding obstacles in all rays...")
        min_range_idx = int(self.min_range / range_resolution)
        
        # Store ALL occupied ranges for each ray (not just first continuous region)
        # Each ray can have multiple occupied regions
        occupied_ranges_per_ray = []
        
        for bearing_idx in range(bearing_bins):
            ray_intensities = polar_image[min_range_idx:, bearing_idx]
            occupied_indices = np.where(ray_intensities >= self.intensity_threshold)[0]
            
            # Store all occupied indices for this ray (adjusted for min_range offset)
            if len(occupied_indices) > 0:
                occupied_ranges_per_ray.append(occupied_indices + min_range_idx)
            else:
                occupied_ranges_per_ray.append(np.array([]))
        
        # Step 2: Process voxels in the grid
        print(f"   🎯 Processing voxels affected by sonar cone...")
        
        # Debug: Verify coordinate transformation
        print(f"\n   🔍 Debug: Coordinate transformation verification")
        test_point_sonar = np.array([1.0, 0.0, 0.0])  # 1m forward in sonar
        test_point_map = self.sonar_to_map_coords(*test_point_sonar)
        print(f"      Sonar [1,0,0] → Map {test_point_map}")
        test_point_sonar2 = np.array([0.0, 1.0, 0.0])  # 1m right in sonar
        test_point_map2 = self.sonar_to_map_coords(*test_point_sonar2)
        print(f"      Sonar [0,1,0] → Map {test_point_map2}")
        test_point_sonar3 = np.array([0.0, 0.0, 1.0])  # 1m down in sonar
        test_point_map3 = self.sonar_to_map_coords(*test_point_sonar3)
        print(f"      Sonar [0,0,1] → Map {test_point_map3}")
        
        # With octree, no grid boundaries - use full sonar range
        max_range_in_grid = self.max_range
        
        print(f"   📏 Using full sonar range: {max_range_in_grid:.2f}m (octree auto-expands)")
        
        # For each voxel in the potential sonar coverage area
        free_updates = 0
        occupied_updates = 0
        
        # Statistics for debugging
        total_voxels_checked = 0
        voxels_in_range = 0
        voxels_in_fov = 0
        voxels_in_aperture = 0
        
        # Vertical angle distribution statistics
        vertical_angle_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'count': 0,
            'histogram': {}  # Bins for angle distribution
        }
        
        # Range distribution statistics
        range_distribution = {}  # range별 voxel 수
        occupied_voxel_coords = []  # occupied voxel들의 좌표와 정보
        free_voxel_angles = []  # free voxel들의 vertical angle
        debug_voxel_count = 0  # 디버그 출력 제한
        
        # Specific debugging for 170° problem
        near_sonar_voxels = []  # 소나 근처 voxel들의 각도 분포
        
        # Debug: Sample some edge cases
        debug_sample_count = 0
        max_debug_samples = 5
        
        # Calculate bounds for voxels that could be affected
        # Since sensor is at (0, 0, 0) and grid goes from negative to positive
        # we need to check the full grid within range
        
        # With octree, we don't iterate through all grid voxels
        # Instead, we process rays and update only the voxels along each ray
        # This is much more efficient and allows dynamic expansion
        
        print(f"   🎯 Processing {len(bearing_angles)} rays with octree storage...")
        
        # Process each ray and create voxels as needed
        for bearing_idx in range(bearing_bins):
            bearing_angle = bearing_angles[bearing_idx]
            
            # Get occupied ranges for this ray
            occupied_ranges = occupied_ranges_per_ray[bearing_idx]
            
            # Process all ranges along this ray
            for range_idx in range(min_range_idx, range_bins):
                range_m = range_idx * range_resolution
                
                # Skip if beyond reasonable range
                if range_m > max_range_in_grid:
                    break
                    
                # Calculate position in sonar coordinates
                sonar_x = range_m * np.cos(bearing_angle)
                sonar_y = range_m * np.sin(bearing_angle)
                sonar_z = 0  # Center of beam
                
                # Transform to map coordinates
                map_pos = self.sonar_to_map_coords(sonar_x, sonar_y, sonar_z)
                map_x, map_y, map_z = map_pos
                
                # Get intensity at this position
                intensity = polar_image[range_idx, bearing_idx]
                
                # Determine update type
                if len(occupied_ranges) > 0 and range_idx in occupied_ranges:
                    # Occupied
                    self.octree.update(map_x, map_y, map_z, self.log_odds_occupied)
                    occupied_updates += 1
                elif len(occupied_ranges) == 0 or range_idx < np.min(occupied_ranges):
                    # Free space (no obstacles or before first obstacle)
                    self.octree.update(map_x, map_y, map_z, self.log_odds_free)
                    free_updates += 1
                # else: shadow zone, no update
                
                total_voxels_checked += 1
        
        print()  # New line after progress bar
        
        # Print overall statistics
        print(f"\n   📊 Overall voxel processing statistics:")
        print(f"      Total voxels checked: {total_voxels_checked:,}")
        print(f"      Voxels in range: {voxels_in_range:,} ({voxels_in_range/total_voxels_checked*100:.1f}%)")
        print(f"      Voxels in FOV: {voxels_in_fov:,} ({voxels_in_fov/total_voxels_checked*100:.1f}%)")
        print(f"      Voxels in aperture: {voxels_in_aperture:,} ({voxels_in_aperture/total_voxels_checked*100:.1f}%)")
        print(f"      Free updates: {free_updates:,}")
        print(f"      Occupied updates: {occupied_updates:,}")
        
        # Print vertical angle distribution
        if vertical_angle_stats['count'] > 0:
            print(f"\n   📐 Vertical Angle Distribution Analysis:")
            print(f"      Aperture setting: ±{self.vertical_aperture_degrees/2:.1f}° (total {self.vertical_aperture_degrees}°)")
            print(f"      Actual angle range: [{vertical_angle_stats['min']:.1f}°, {vertical_angle_stats['max']:.1f}°]")
            
            # Calculate theoretical maximum possible angle given grid constraints
            # Maximum Z depth is -1.5m, minimum range is 0.5m
            max_theoretical_angle = np.degrees(np.arctan2(abs(self.z_range[0]), self.min_range))
            print(f"      Max theoretical angle with grid: ±{max_theoretical_angle:.1f}°")
            
            # Print histogram
            print(f"      Angle distribution (10° bins):")
            sorted_bins = sorted(vertical_angle_stats['histogram'].keys())
            for bin_key in sorted_bins:
                count = vertical_angle_stats['histogram'][bin_key]
                bar_length = int(count / max(vertical_angle_stats['histogram'].values()) * 20)
                bar = '█' * bar_length
                print(f"         [{bin_key:3d}° to {bin_key+10:3d}°]: {bar} ({count:,})")
            
            # Warning if aperture exceeds physical limits
            if self.vertical_aperture_degrees/2 > max_theoretical_angle:
                print(f"\n      ⚠️  WARNING: Requested aperture (±{self.vertical_aperture_degrees/2:.1f}°) exceeds")
                print(f"         maximum achievable angle (±{max_theoretical_angle:.1f}°) with current grid!")
                print(f"         Consider: - Increasing Z range (currently {self.z_range})")
                print(f"                    - Decreasing min_range (currently {self.min_range}m)")
        
        # Debug near-sonar voxels for 170° problem
        if near_sonar_voxels:
            print(f"\n   🔍 Near-sonar voxel analysis (< 1m from sonar):")
            print(f"      Total voxels within 1m: {len(near_sonar_voxels)}")
            
            # Analyze angle distribution
            angles_in_aperture = [v['vertical_angle'] for v in near_sonar_voxels if v['in_aperture']]
            angles_outside = [v['vertical_angle'] for v in near_sonar_voxels if not v['in_aperture']]
            
            if angles_in_aperture:
                print(f"      Voxels in aperture: {len(angles_in_aperture)}")
                print(f"      Min/Max angles in aperture: [{min(angles_in_aperture):.1f}°, {max(angles_in_aperture):.1f}°]")
            
            if angles_outside:
                print(f"      Voxels outside aperture: {len(angles_outside)}")
                print(f"      Min/Max angles outside: [{min(angles_outside):.1f}°, {max(angles_outside):.1f}°]")
            
            # Check specific distance ranges
            for dist_threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                voxels_at_dist = [v for v in near_sonar_voxels if v['dist'] <= dist_threshold]
                if voxels_at_dist:
                    angles = [v['vertical_angle'] for v in voxels_at_dist]
                    print(f"      At distance ≤{dist_threshold:.1f}m: angle range [{min(angles):.1f}°, {max(angles):.1f}°]")
        
        # Debug occupied voxel patterns
        if occupied_voxel_coords:
            print(f"\n   🔴 Occupied voxel pattern analysis (first {len(occupied_voxel_coords)} voxels):")
            
            # Group by similar range
            range_groups = {}
            for voxel in occupied_voxel_coords:
                range_key = round(voxel['dist'], 1)  # Group by 0.1m intervals
                if range_key not in range_groups:
                    range_groups[range_key] = []
                range_groups[range_key].append(voxel)
            
            print(f"      Range distribution of occupied voxels:")
            for range_key in sorted(range_groups.keys())[:10]:  # Show first 10 ranges
                voxels = range_groups[range_key]
                bearings = [v['bearing'] for v in voxels]
                verticals = [v['vertical'] for v in voxels]
                print(f"         {range_key:.1f}m: {len(voxels)} voxels")
                print(f"            Bearing range: [{min(bearings):.1f}°, {max(bearings):.1f}°]")
                print(f"            Vertical range: [{min(verticals):.1f}°, {max(verticals):.1f}°]")
            
            # Check if occupied voxels form arcs
            print(f"\n      Arc consistency check:")
            for i in range(min(5, len(occupied_voxel_coords))):
                v = occupied_voxel_coords[i]
                print(f"         Voxel {i}: Range={v['dist']:.3f}m, Bearing={v['bearing']:.1f}°, Vertical={v['vertical']:.1f}°")
                print(f"                   World=({v['world'][0]:.3f}, {v['world'][1]:.3f}, {v['world'][2]:.3f})")
                print(f"                   Sonar=({v['sonar'][0]:.3f}, {v['sonar'][1]:.3f}, {v['sonar'][2]:.3f})")
        
        # Calculate processing statistics
        processing_time = time.time() - start_time
        
        stats = {
            "processing_time": processing_time,
            "rays_processed": bearing_bins,
            "free_updates": free_updates,
            "occupied_updates": occupied_updates,
            "range_resolution": range_resolution,
            "bearing_resolution_deg": np.degrees(bearing_resolution),
            "voxels_checked": total_voxels_checked,
            "voxels_in_range": voxels_in_range,
            "voxels_in_fov": voxels_in_fov,
            "voxels_in_aperture": voxels_in_aperture
        }
        
        print(f"   ✅ Voxel-centric processing completed in {processing_time:.3f}s")
        print(f"   📊 Statistics:")
        print(f"      Rays analyzed: {bearing_bins:,}")
        print(f"      Free space updates: {free_updates:,}")
        print(f"      Occupied updates: {occupied_updates:,}")
        
        return stats
    
    def get_probability_grid(self) -> np.ndarray:
        """Get probability grid as dense array for visualization compatibility"""
        if not self.octree.voxels:
            return np.ones((10, 10, 10)) * 0.5
        
        # Calculate bounds and create dense grid for visualization
        min_b = self.octree.min_bounds
        max_b = self.octree.max_bounds
        
        size = np.ceil((max_b - min_b) / self.resolution).astype(int)
        size = np.maximum(size, 1)
        
        grid = np.ones(size) * 0.5  # Initialize with unknown
        
        for key, log_odds in self.octree.voxels.items():
            world_pos = self.octree.key_to_world(key)
            idx = ((world_pos - min_b) / self.resolution).astype(int)
            if np.all(idx >= 0) and np.all(idx < size):
                grid[tuple(idx)] = 1.0 / (1.0 + np.exp(-log_odds))
        
        return grid
    
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
    
    def visualize_occupancy_grid(self, title: str = "Octree-based Sonar Occupancy Mapping"):
        """
        Visualize the occupancy grid with free/unknown/occupied voxels
        """
        print(f"\n📊 Creating octree occupancy grid visualization...")
        
        # Get classified voxels
        voxels = self.get_classified_voxels()
        
        # Calculate statistics
        total_voxels = len(self.octree.voxels)  # Only stored voxels
        free_count = len(voxels['free'])
        unknown_count = len(voxels['unknown'])
        occupied_count = len(voxels['occupied'])
        
        print(f"   📈 Octree voxel statistics:")
        print(f"      Stored voxels: {total_voxels:,}")
        print(f"      Free: {free_count:,}")
        print(f"      Unknown: {unknown_count:,}")
        print(f"      Occupied: {occupied_count:,}")
        
        # Create visualization
        fig = plt.figure(figsize=(16, 12))
        
        # 3D visualization
        ax1 = fig.add_subplot(231, projection='3d')
        
        # Plot voxels with different colors
        if len(voxels['free']) > 0:
            # Sample for visualization if too many
            free_viz = voxels['free']
            if len(free_viz) > 5000:
                idx = np.random.choice(len(free_viz), 5000, replace=False)
                free_viz = free_viz[idx]
            ax1.scatter(free_viz[:, 0], free_viz[:, 1], free_viz[:, 2],
                       c='blue', s=1, alpha=0.3, label=f'Free ({free_count:,})')
        
        if len(voxels['unknown']) > 0:
            unknown_viz = voxels['unknown']
            if len(unknown_viz) > 3000:
                idx = np.random.choice(len(unknown_viz), 3000, replace=False)
                unknown_viz = unknown_viz[idx]
            ax1.scatter(unknown_viz[:, 0], unknown_viz[:, 1], unknown_viz[:, 2],
                       c='gray', s=2, alpha=0.5, label=f'Unknown ({unknown_count:,})')
        
        if len(voxels['occupied']) > 0:
            ax1.scatter(voxels['occupied'][:, 0], voxels['occupied'][:, 1], voxels['occupied'][:, 2],
                       c='red', s=3, alpha=0.8, label=f'Occupied ({occupied_count:,})')
        
        # Mark sensor position
        ax1.scatter(*self.sensor_position, c='black', s=50, marker='s', 
                   label='Sensor')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Occupancy Grid')
        ax1.legend()
        # Use octree bounds for axis limits
        if len(self.octree.voxels) > 0:
            margin = 0.5  # Add some margin around the data
            ax1.set_xlim(self.octree.min_bounds[0] - margin, self.octree.max_bounds[0] + margin)
            ax1.set_ylim(self.octree.min_bounds[1] - margin, self.octree.max_bounds[1] + margin)
            ax1.set_zlim(self.octree.min_bounds[2] - margin, self.octree.max_bounds[2] + margin)
        else:
            ax1.set_xlim(-1, 1)
            ax1.set_ylim(-1, 1)
            ax1.set_zlim(-1, 1)
        
        # XY view (top-down)
        ax2 = fig.add_subplot(232)
        if len(voxels['free']) > 0:
            ax2.scatter(voxels['free'][:, 0], voxels['free'][:, 1], 
                       c='blue', s=0.5, alpha=0.3)
        if len(voxels['unknown']) > 0:
            ax2.scatter(voxels['unknown'][:, 0], voxels['unknown'][:, 1], 
                       c='gray', s=1, alpha=0.5)
        if len(voxels['occupied']) > 0:
            ax2.scatter(voxels['occupied'][:, 0], voxels['occupied'][:, 1], 
                       c='red', s=2, alpha=0.8)
        ax2.scatter(self.sensor_position[0], self.sensor_position[1], 
                   c='black', s=50, marker='s')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View (XY plane)')
        if len(self.octree.voxels) > 0:
            ax2.set_xlim(self.octree.min_bounds[0] - 0.5, self.octree.max_bounds[0] + 0.5)
            ax2.set_ylim(self.octree.min_bounds[1] - 0.5, self.octree.max_bounds[1] + 0.5)
        else:
            ax2.set_xlim(-1, 1)
            ax2.set_ylim(-1, 1)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # XZ view (side view)
        ax3 = fig.add_subplot(233)
        if len(voxels['free']) > 0:
            ax3.scatter(voxels['free'][:, 0], voxels['free'][:, 2], 
                       c='blue', s=0.5, alpha=0.3)
        if len(voxels['unknown']) > 0:
            ax3.scatter(voxels['unknown'][:, 0], voxels['unknown'][:, 2], 
                       c='gray', s=1, alpha=0.5)
        if len(voxels['occupied']) > 0:
            ax3.scatter(voxels['occupied'][:, 0], voxels['occupied'][:, 2], 
                       c='red', s=2, alpha=0.8)
        ax3.scatter(self.sensor_position[0], self.sensor_position[2], 
                   c='black', s=50, marker='s')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Side View (XZ plane)')
        if len(self.octree.voxels) > 0:
            ax3.set_xlim(self.octree.min_bounds[0] - 0.5, self.octree.max_bounds[0] + 0.5)
            ax3.set_ylim(self.octree.min_bounds[2] - 0.5, self.octree.max_bounds[2] + 0.5)
        else:
            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-1, 0)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # YZ view (front view)
        ax4 = fig.add_subplot(234)
        if len(voxels['free']) > 0:
            ax4.scatter(voxels['free'][:, 1], voxels['free'][:, 2], 
                       c='blue', s=0.5, alpha=0.3)
        if len(voxels['unknown']) > 0:
            ax4.scatter(voxels['unknown'][:, 1], voxels['unknown'][:, 2], 
                       c='gray', s=1, alpha=0.5)
        if len(voxels['occupied']) > 0:
            ax4.scatter(voxels['occupied'][:, 1], voxels['occupied'][:, 2], 
                       c='red', s=2, alpha=0.8)
        ax4.scatter(self.sensor_position[1], self.sensor_position[2], 
                   c='black', s=50, marker='s')
        ax4.set_xlabel('Y (m)')
        ax4.set_ylabel('Z (m)')
        ax4.set_title('Front View (YZ plane)')
        if len(self.octree.voxels) > 0:
            ax4.set_xlim(self.octree.min_bounds[1] - 0.5, self.octree.max_bounds[1] + 0.5)
            ax4.set_ylim(self.octree.min_bounds[2] - 0.5, self.octree.max_bounds[2] + 0.5)
        else:
            ax4.set_xlim(-1, 1)
            ax4.set_ylim(-1, 0)
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
        # Probability histogram
        ax5 = fig.add_subplot(235)
        prob_grid = self.get_probability_grid()
        ax5.hist(prob_grid.flatten(), bins=50, alpha=0.7, color='purple')
        ax5.axvline(0.3, color='blue', linestyle='--', label='Free threshold')
        ax5.axvline(0.7, color='red', linestyle='--', label='Occupied threshold')
        ax5.set_xlabel('Probability')
        ax5.set_ylabel('Voxel Count')
        ax5.set_title('Probability Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Statistics text
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        
        # Calculate octree bounds
        if len(self.octree.voxels) > 0:
            bounds_size = self.octree.max_bounds - self.octree.min_bounds
        else:
            bounds_size = np.zeros(3)
        
        stats_text = f"Octree-based Sonar Mapping Statistics\n"
        stats_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        stats_text += f"Dynamic Octree (auto-expanding)\n"
        stats_text += f"Resolution: {self.resolution*1000:.0f}mm\n"
        stats_text += f"Current Bounds: {bounds_size[0]:.1f}×{bounds_size[1]:.1f}×{bounds_size[2]:.1f}m\n"
        stats_text += f"Stored Voxels: {total_voxels:,} (sparse storage)\n\n"
        stats_text += f"Voxel Classification:\n"
        stats_text += f"  🔵 Free: {free_count:,} ({free_count/total_voxels*100:.1f}%)\n"
        stats_text += f"  ⚪ Unknown: {unknown_count:,} ({unknown_count/total_voxels*100:.1f}%)\n"
        stats_text += f"  🔴 Occupied: {occupied_count:,} ({occupied_count/total_voxels*100:.1f}%)\n\n"
        stats_text += f"Sensor Configuration:\n"
        stats_text += f"  📍 Position: {self.sensor_position}\n"
        stats_text += f"  📏 Range: {self.max_range}m\n"
        stats_text += f"  📐 FOV: {self.fov_degrees}°\n"
        stats_text += f"  🎯 Threshold: {self.intensity_threshold}\n"
        
        ax6.text(0.05, 0.95, stats_text, fontsize=10, fontfamily='monospace',
                verticalalignment='top', transform=ax6.transAxes)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive mapping statistics"""
        voxels = self.get_classified_voxels()
        total_voxels = len(self.octree.voxels)
        
        # Calculate log-odds statistics from octree
        if total_voxels > 0:
            log_odds_values = [v for v in self.octree.voxels.values()]
            log_odds_min = min(log_odds_values)
            log_odds_max = max(log_odds_values)
            log_odds_mean = np.mean(log_odds_values)
            log_odds_std = np.std(log_odds_values)
            
            # Calculate probability statistics
            probs = [1.0 / (1.0 + np.exp(-lo)) for lo in log_odds_values]
            prob_min = min(probs)
            prob_max = max(probs)
            prob_mean = np.mean(probs)
            prob_std = np.std(probs)
        else:
            log_odds_min = log_odds_max = log_odds_mean = log_odds_std = 0.0
            prob_min = prob_max = prob_mean = prob_std = 0.5
        
        # Calculate percentages
        free_pct = len(voxels['free']) / total_voxels * 100 if total_voxels > 0 else 0
        unknown_pct = len(voxels['unknown']) / total_voxels * 100 if total_voxels > 0 else 0
        occupied_pct = len(voxels['occupied']) / total_voxels * 100 if total_voxels > 0 else 0
        
        # Calculate bounds
        if len(self.octree.voxels) > 0:
            bounds_size = self.octree.max_bounds - self.octree.min_bounds
        else:
            bounds_size = np.zeros(3)
        
        stats = {
            "octree_stats": {
                "stored_voxels": total_voxels,
                "bounds_min": self.octree.min_bounds.tolist(),
                "bounds_max": self.octree.max_bounds.tolist(),
                "bounds_size": bounds_size.tolist()
            },
            "resolution": self.resolution,
            "total_voxels": total_voxels,
            "free_voxels": len(voxels['free']),
            "unknown_voxels": len(voxels['unknown']),
            "occupied_voxels": len(voxels['occupied']),
            "free_percentage": free_pct,
            "unknown_percentage": unknown_pct,
            "occupied_percentage": occupied_pct,
            "probability_stats": {
                "min": prob_min,
                "max": prob_max,
                "mean": prob_mean,
                "std": prob_std
            },
            "log_odds_stats": {
                "min": log_odds_min,
                "max": log_odds_max,
                "mean": log_odds_mean,
                "std": log_odds_std
            },
            "sensor_config": {
                "position": self.sensor_position,
                "max_range": self.max_range,
                "fov_degrees": self.fov_degrees,
                "intensity_threshold": self.intensity_threshold,
                "min_range": self.min_range
            }
        }
        
        return stats


if __name__ == "__main__":
    # Test the octree-based mapper
    print("🧪 Testing Octree-based Sonar Mapper...")
    
    # Create mapper
    mapper = OctreeBasedSonarMapperWithBoat(
        resolution=0.03,
        sensor_position=(0.0, 0.0, 0.0)
    )
    
    # Create synthetic sonar data for testing
    range_bins = 495
    bearing_bins = 512
    
    # Create test pattern: circular object at depth
    test_image = np.zeros((range_bins, bearing_bins), dtype=np.uint8)
    
    # Add a circular object at range ~2m, intensity 100
    object_range_bin = int(2.0 / 5.0 * range_bins)  # 2m depth
    for bearing_idx in range(bearing_bins):
        bearing_angle = -np.pi/3 + bearing_idx * (2*np.pi/3) / bearing_bins
        # Create circular pattern
        if abs(bearing_angle) < np.pi/6:  # 60° sector
            test_image[object_range_bin:object_range_bin+10, bearing_idx] = 80
    
    # Process the test image
    stats = mapper.process_polar_image(test_image)
    
    # Visualize results
    fig = mapper.visualize_occupancy_grid("Octree-based Mapper Test")
    
    # Print statistics
    mapping_stats = mapper.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   Stored voxels: {mapping_stats['total_voxels']:,}")
    print(f"   Free voxels: {mapping_stats['free_voxels']:,} ({mapping_stats['free_percentage']:.1f}%)")
    print(f"   Unknown voxels: {mapping_stats['unknown_voxels']:,} ({mapping_stats['unknown_percentage']:.1f}%)")
    print(f"   Occupied voxels: {mapping_stats['occupied_voxels']:,} ({mapping_stats['occupied_percentage']:.1f}%)")
    
    plt.show()