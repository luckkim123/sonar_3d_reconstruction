#!/usr/bin/env python3
"""
3D Sonar Mapping Library with Probabilistic Octree
Based on feature_extraction_3d.py with adaptations for real data

This module provides octree-based sparse storage and probabilistic mapping
for 3D sonar reconstruction using log-odds Bayesian updates.

Author: Sonar 3D Reconstruction Team
Date: 2025
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict, Any, Optional
import time


class SimpleOctree:
    """
    Sparse voxel storage using dictionary with dynamic expansion
    Stores log-odds values for each voxel with adaptive updating
    """
    
    def __init__(self, resolution: float = 0.03, dynamic_expansion: bool = True):
        """
        Initialize octree with given resolution
        
        Args:
            resolution: Size of each voxel in meters
            dynamic_expansion: Enable dynamic map expansion
        """
        self.resolution = resolution
        self.voxels = defaultdict(float)  # Store log-odds values
        self.dynamic_expansion = dynamic_expansion
        
        # Map bounds (for dynamic expansion)
        self.min_bounds = np.array([float('inf')] * 3)
        self.max_bounds = np.array([-float('inf')] * 3)
        
        # Log-odds parameters (will be set from config)
        self.log_odds_occupied = 1.5      # Log-odds increment for occupied
        self.log_odds_free = -2.0         # Log-odds decrement for free space
        self.log_odds_min = -10.0         # Minimum log-odds (clamping)
        self.log_odds_max = 10.0          # Maximum log-odds (clamping)
        self.log_odds_threshold = 0.0     # Threshold for considering occupied
        
        # Adaptive update parameters
        self.adaptive_update = True       # Enable adaptive updating
        self.adaptive_threshold = 0.5     # Protection threshold
        self.adaptive_max_ratio = 0.5     # Maximum update ratio at threshold
    
    def world_to_key(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """
        Convert world coordinates to voxel key
        
        Args:
            x, y, z: World coordinates in meters
            
        Returns:
            Tuple (ix, iy, iz) as voxel index
        """
        i = int(np.floor(x / self.resolution))
        j = int(np.floor(y / self.resolution))
        k = int(np.floor(z / self.resolution))
        return (i, j, k)
    
    def key_to_world(self, key: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert voxel key to world coordinates (center of voxel)
        
        Args:
            key: Tuple (ix, iy, iz) voxel index
            
        Returns:
            numpy array [x, y, z] world coordinates
        """
        x = (key[0] + 0.5) * self.resolution
        y = (key[1] + 0.5) * self.resolution
        z = (key[2] + 0.5) * self.resolution
        return np.array([x, y, z])
    
    def update_voxel(self, point: np.ndarray, log_odds_update: float, adaptive: bool = True):
        """
        Update voxel log-odds value with optional adaptive updating
        
        Args:
            point: [x, y, z] numpy array in world coordinates
            log_odds_update: Log-odds increment/decrement
            adaptive: If True, use adaptive updating for occupied updates
        """
        key = self.world_to_key(point[0], point[1], point[2])
        
        # Adaptive update: reduce occupied updates for voxels that are likely free
        if adaptive and self.adaptive_update and log_odds_update > 0:
            current_log_odds = self.voxels.get(key, 0.0)
            current_prob = 1.0 / (1.0 + np.exp(-current_log_odds))
            
            # Linear interpolation for adaptive update
            if current_prob <= self.adaptive_threshold:
                update_scale = (current_prob / self.adaptive_threshold) * self.adaptive_max_ratio
                log_odds_update *= update_scale
        
        # Apply update
        if key not in self.voxels:
            self.voxels[key] = 0.0
        self.voxels[key] += log_odds_update
        
        # Clamp to prevent overflow
        self.voxels[key] = np.clip(self.voxels[key], self.log_odds_min, self.log_odds_max)
        
        # Update bounds for dynamic expansion
        if self.dynamic_expansion:
            self.min_bounds = np.minimum(self.min_bounds, point)
            self.max_bounds = np.maximum(self.max_bounds, point)
    
    def get_log_odds(self, x: float, y: float, z: float) -> float:
        """Get log-odds value for a voxel"""
        key = self.world_to_key(x, y, z)
        return self.voxels.get(key, 0.0)
    
    def get_probability(self, x: float, y: float, z: float) -> float:
        """Get probability from log-odds value"""
        log_odds = self.get_log_odds(x, y, z)
        return 1.0 / (1.0 + np.exp(-log_odds))
    
    def get_occupied_voxels(self, min_probability: float = 0.5) -> List[Tuple[np.ndarray, float]]:
        """
        Get all occupied voxels above probability threshold
        
        Args:
            min_probability: Minimum probability to consider occupied
            
        Returns:
            List of (point, probability) tuples
        """
        occupied = []
        
        # Convert probability to log-odds threshold
        if min_probability >= 1.0:
            min_log_odds = self.log_odds_max - 0.01
        elif min_probability <= 0.0:
            min_log_odds = self.log_odds_min
        else:
            min_log_odds = np.log(min_probability / (1.0 - min_probability))
        
        for key, log_odds in self.voxels.items():
            if log_odds > min_log_odds:
                point = self.key_to_world(key)
                probability = 1.0 / (1.0 + np.exp(-log_odds))
                occupied.append((point, probability))
        
        return occupied
    
    def get_all_voxels_classified(self, min_probability: float = 0.7) -> Dict[str, List]:
        """
        Get all voxels classified as free, unknown, or occupied
        
        Args:
            min_probability: Minimum probability to consider occupied
            
        Returns:
            Dictionary with 'free', 'unknown', 'occupied' lists
        """
        free = []
        unknown = []
        occupied = []
        
        # Thresholds
        free_threshold = np.log(0.3 / 0.7)  # prob < 0.3 = free
        occupied_threshold = np.log(min_probability / (1.0 - min_probability))
        
        for key, log_odds in self.voxels.items():
            point = self.key_to_world(key)
            probability = 1.0 / (1.0 + np.exp(-log_odds))
            
            if log_odds < free_threshold:
                free.append((point, probability))
            elif log_odds > occupied_threshold:
                occupied.append((point, probability))
            else:
                unknown.append((point, probability))
        
        return {
            'free': free,
            'unknown': unknown,
            'occupied': occupied
        }
    
    def clear(self):
        """Clear all voxels"""
        self.voxels.clear()
        self.min_bounds = np.array([float('inf')] * 3)
        self.max_bounds = np.array([-float('inf')] * 3)


class SonarTo3DMapper:
    """
    Convert sonar images to 3D point clouds with probabilistic mapping
    Accumulates multiple frames and updates voxel probabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize mapper with configuration
        
        Args:
            config: Configuration dictionary (overrides defaults)
        
        Note: Parameter priority (highest to lowest):
            1. ROS2 launch args (handled by node)
            2. YAML file (handled by node)
            3. Launch file params (handled by node)
            4. Node defaults (handled by node)
            5. Config dict passed here (overrides defaults below)
            6. Default values in this class (lowest priority)
        """
        # Default configuration (lowest priority - priority 5)
        # These will be overridden by any values in the config dict
        default_config = {
            # Sonar parameters
            'horizontal_fov': 130.0,       # degrees
            'vertical_aperture': 20.0,     # degrees
            'max_range': 10.0,             # meters
            'min_range': 0.5,              # meters
            'intensity_threshold': 35,     # 0-255 scale
            'image_width': 512,            # bearings
            'image_height': 500,           # ranges
            
            # Sonar mounting (relative to base_link)
            'sonar_position': [0.0, 0.0, -0.5],  # xyz
            'sonar_orientation': [0.0, 1.5708, 0.0],  # rpy (0, 90deg, 0)
            
            # Octree parameters
            'voxel_resolution': 0.05,      # meters
            'min_probability': 0.6,        # for occupied
            'dynamic_expansion': True,
            
            # Adaptive update
            'adaptive_update': True,
            'adaptive_threshold': 0.5,
            'adaptive_max_ratio': 0.3,
            
            # Log-odds parameters
            'log_odds_occupied': 1.5,
            'log_odds_free': -2.0,
            'log_odds_min': -10.0,
            'log_odds_max': 10.0,
            
        }
        
        # Update with provided config (config dict has priority over defaults)
        if config:
            default_config.update(config)
        
        # Store parameters
        self.horizontal_fov = np.radians(default_config['horizontal_fov'])
        self.vertical_aperture = np.radians(default_config['vertical_aperture'])
        self.max_range = default_config['max_range']
        self.min_range = default_config['min_range']
        self.intensity_threshold = default_config['intensity_threshold']
        self.image_width = default_config['image_width']
        self.image_height = default_config['image_height']
        self.voxel_resolution = default_config['voxel_resolution']
        self.min_probability = default_config['min_probability']
        self.dynamic_expansion = default_config['dynamic_expansion']
        
        # Z-axis filtering
        self.z_filter_min = default_config.get('z_filter_min', -5.0)
        self.z_filter_enabled = default_config.get('z_filter_enabled', False)
        
        # Sonar mounting transform
        self.sonar_position = np.array(default_config['sonar_position'])
        self.sonar_orientation = np.array(default_config['sonar_orientation'])
        
        # Pre-compute sonar to base_link transform
        self.T_sonar_to_base = self.create_transform_matrix(
            self.sonar_position,
            self.sonar_orientation
        )
        
        # Initialize octree with settings
        self.octree = SimpleOctree(self.voxel_resolution, self.dynamic_expansion)
        
        # Configure octree parameters
        self.octree.log_odds_occupied = default_config['log_odds_occupied']
        self.octree.log_odds_free = default_config['log_odds_free']
        self.octree.log_odds_min = default_config['log_odds_min']
        self.octree.log_odds_max = default_config['log_odds_max']
        self.octree.adaptive_update = default_config['adaptive_update']
        self.octree.adaptive_threshold = default_config['adaptive_threshold']
        self.octree.adaptive_max_ratio = default_config['adaptive_max_ratio']
        
        # Pre-compute bearing angles
        self.bearing_angles = np.linspace(
            -self.horizontal_fov/2,
            self.horizontal_fov/2,
            self.image_width
        )
        
        
        # Frame counter
        self.frame_count = 0
        self.processed_frame_count = 0
        
        # Debug: Track update counts per voxel
        self.voxel_update_counts = defaultdict(int)
        self.frame_update_counts = defaultdict(int)  # Updates in current frame
        
        # Processing statistics
        self.last_processing_time = 0.0
        self.total_processing_time = 0.0
    
    def create_transform_matrix(self, position: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        """
        Create 4x4 homogeneous transform matrix from position and RPY
        
        Args:
            position: [x, y, z] translation
            rpy: [roll, pitch, yaw] rotation in radians
            
        Returns:
            4x4 numpy array transform matrix
        """
        # Create rotation matrix from RPY
        cr = np.cos(rpy[0])
        sr = np.sin(rpy[0])
        cp = np.cos(rpy[1])
        sp = np.sin(rpy[1])
        cy = np.cos(rpy[2])
        sy = np.sin(rpy[2])
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        # Build 4x4 homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        return T
    
    def quaternion_to_matrix(self, quaternion: List[float]) -> np.ndarray:
        """
        Convert quaternion to rotation matrix
        
        Args:
            quaternion: [x, y, z, w] quaternion
            
        Returns:
            3x3 rotation matrix
        """
        x, y, z, w = quaternion
        
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    def create_odometry_transform(self, position: List[float], quaternion: List[float]) -> np.ndarray:
        """
        Create transformation matrix from odometry data
        
        Args:
            position: [x, y, z] position
            quaternion: [x, y, z, w] orientation
            
        Returns:
            4x4 transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = self.quaternion_to_matrix(quaternion)
        T[:3, 3] = position
        return T
    
    def is_bearing_in_valid_fov(self, bearing_angle: float) -> bool:
        """Check if bearing angle is within valid FOV"""
        half_fov = self.horizontal_fov / 2
        return abs(bearing_angle) <= half_fov
    
    def process_sonar_ray(self, bearing_angle: float, intensity_profile: np.ndarray, 
                          T_sonar_to_world: np.ndarray) -> List[Tuple[np.ndarray, float, str]]:
        """
        Process a single sonar ray and return voxel updates
        
        Args:
            bearing_angle: Horizontal angle in radians
            intensity_profile: 1D array of intensities along range
            T_sonar_to_world: 4x4 transform matrix
            
        Returns:
            List of (point, log_odds_update, type) tuples
        """
        updates = []
        
        # Find first hit
        first_hit_idx = -1
        range_resolution = self.max_range / len(intensity_profile)
        
        for r_idx, intensity in enumerate(intensity_profile):
            if intensity > self.intensity_threshold:
                first_hit_idx = r_idx
                break
        
        # If no hit, process entire ray as free
        if first_hit_idx == -1:
            first_hit_idx = len(intensity_profile)
        
        # Calculate vertical aperture parameters
        half_aperture = self.vertical_aperture / 2
        
        # Process free space before first hit (sparse)
        free_sampling_step = 10
        for r_idx in range(0, first_hit_idx, free_sampling_step):
            range_m = r_idx * range_resolution
            if range_m < self.min_range:
                continue
            
            # Calculate vertical spread
            vertical_spread = range_m * np.tan(half_aperture)
            num_vertical = max(1, int(vertical_spread / (self.voxel_resolution * 4)))
            
            for v_step in range(-num_vertical, num_vertical + 1):
                vertical_angle = (v_step / max(1, num_vertical)) * half_aperture
                
                # Sonar coordinates (X=forward, Y=right, Z=down)
                # Note: Negative y for correct right-hand coordinate system
                x_sonar = range_m * np.cos(vertical_angle) * np.cos(bearing_angle)
                y_sonar = -range_m * np.cos(vertical_angle) * np.sin(bearing_angle)
                z_sonar = range_m * np.sin(vertical_angle)
                
                # Transform to world
                pt_sonar = np.array([x_sonar, y_sonar, z_sonar, 1.0])
                pt_world = T_sonar_to_world @ pt_sonar
                
                # Apply Z-axis filter if enabled
                if self.z_filter_enabled and pt_world[2] < self.z_filter_min:
                    continue
                
                updates.append((pt_world[:3], self.octree.log_odds_free, 'free'))
        
        # Process occupied regions (dense)
        if first_hit_idx < len(intensity_profile):
            # Find all high intensity regions
            for r_idx in range(first_hit_idx, min(first_hit_idx + 50, len(intensity_profile))):
                if intensity_profile[r_idx] > self.intensity_threshold:
                    range_m = r_idx * range_resolution
                    
                    # Check both min and max range
                    if range_m < self.min_range:
                        continue
                    if range_m > self.max_range:
                        break
                    
                    # Calculate vertical spread
                    vertical_spread = range_m * np.tan(half_aperture)
                    num_vertical = max(2, int(vertical_spread / (self.voxel_resolution * 1.5)))
                    
                    for v_step in range(-num_vertical, num_vertical + 1):
                        vertical_angle = (v_step / max(1, num_vertical)) * half_aperture
                        
                        # Sonar coordinates (X=forward, Y=right, Z=down)
                        x_sonar = range_m * np.cos(vertical_angle) * np.cos(bearing_angle)
                        y_sonar = -range_m * np.cos(vertical_angle) * np.sin(bearing_angle)
                        z_sonar = range_m * np.sin(vertical_angle)
                        
                        # Transform to world
                        pt_sonar = np.array([x_sonar, y_sonar, z_sonar, 1.0])
                        pt_world = T_sonar_to_world @ pt_sonar
                        
                        # Apply Z-axis filter if enabled
                        if self.z_filter_enabled and pt_world[2] < self.z_filter_min:
                            continue
                        
                        updates.append((pt_world[:3], self.octree.log_odds_occupied, 'occupied'))
        
        return updates
    
    def process_sonar_image(self, polar_image: np.ndarray, 
                           robot_position: List[float], 
                           robot_orientation: List[float]) -> Dict[str, Any]:
        """
        Process sonar image and update probabilistic map
        
        Args:
            polar_image: 2D numpy array (height x width) with intensity values
            robot_position: [x, y, z] position from odometry
            robot_orientation: [x, y, z, w] quaternion from odometry
            
        Returns:
            Processing statistics dictionary
        """
        self.frame_count += 1
        start_time = time.time()
        self.processed_frame_count += 1
        
        # Ensure image is numpy array
        if not isinstance(polar_image, np.ndarray):
            polar_image = np.array(polar_image)
        
        # Get image dimensions
        range_bins, bearing_bins = polar_image.shape
        
        # Update bearing angles if needed
        if bearing_bins != self.image_width:
            self.bearing_angles = np.linspace(
                -self.horizontal_fov/2,
                self.horizontal_fov/2,
                bearing_bins
            )
            self.image_width = bearing_bins
        
        # Create transformation matrices
        T_base_to_world = self.create_odometry_transform(robot_position, robot_orientation)
        T_sonar_to_world = T_base_to_world @ self.T_sonar_to_base
        
        # Accumulate updates per voxel
        voxel_updates = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'type': 'unknown'})
        self.frame_update_counts.clear()  # Reset for this frame
        
        # Process subset of bearings for efficiency
        bearing_step = max(1, bearing_bins // 256)
        
        for b_idx in range(0, bearing_bins, bearing_step):
            bearing_angle = self.bearing_angles[b_idx]
            
            # Skip bearings outside valid FOV
            if not self.is_bearing_in_valid_fov(bearing_angle):
                continue
            
            # Process this ray
            intensity_profile = polar_image[:, b_idx]
            ray_updates = self.process_sonar_ray(bearing_angle, intensity_profile, T_sonar_to_world)
            
            # Accumulate updates
            for point, log_odds, update_type in ray_updates:
                key = self.octree.world_to_key(point[0], point[1], point[2])
                if voxel_updates[key]['type'] != 'occupied':  # Occupied has priority
                    voxel_updates[key]['type'] = update_type
                voxel_updates[key]['sum'] += log_odds
                voxel_updates[key]['count'] += 1
                
                # Debug: Track updates
                self.frame_update_counts[key] += 1
                self.voxel_update_counts[key] += 1
        
        # Apply averaged updates to octree
        num_occupied = 0
        num_free = 0
        
        for key, update_info in voxel_updates.items():
            if update_info['count'] > 0:
                avg_update = update_info['sum'] / update_info['count']
                point = self.octree.key_to_world(key)
                
                if update_info['type'] == 'occupied':
                    self.octree.update_voxel(point, avg_update, adaptive=True)
                    num_occupied += 1
                elif update_info['type'] == 'free':
                    self.octree.update_voxel(point, avg_update, adaptive=False)
                    num_free += 1
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.last_processing_time = processing_time
        self.total_processing_time += processing_time
        
        # Debug statistics
        if self.frame_update_counts:
            max_updates_frame = max(self.frame_update_counts.values())
            avg_updates_frame = sum(self.frame_update_counts.values()) / len(self.frame_update_counts)
            max_updates_total = max(self.voxel_update_counts.values())
            
            if self.frame_count % 10 == 0:  # Log every 10 frames
                print(f"[DEBUG] Frame {self.frame_count}:")
                print(f"  Max updates in frame: {max_updates_frame}")
                print(f"  Avg updates in frame: {avg_updates_frame:.1f}")
                print(f"  Max total updates: {max_updates_total}")
                print(f"  Voxels with >10 updates in frame: {sum(1 for v in self.frame_update_counts.values() if v > 10)}")
        
        return {
            'frame_count': self.frame_count,
            'processed_count': self.processed_frame_count,
            'num_occupied': num_occupied,
            'num_free': num_free,
            'num_voxels': len(self.octree.voxels),
            'processing_time': processing_time,
            'avg_processing_time': self.total_processing_time / max(1, self.processed_frame_count)
        }
    
    def get_point_cloud(self, include_free: bool = False) -> Dict[str, Any]:
        """
        Get current point cloud from probabilistic map
        
        Args:
            include_free: Whether to include free space voxels
            
        Returns:
            Dictionary containing point cloud data and statistics
        """
        if include_free:
            classified = self.octree.get_all_voxels_classified(self.min_probability)
            
            return {
                'occupied': classified['occupied'],
                'free': classified['free'],
                'unknown': classified['unknown'],
                'num_voxels': len(self.octree.voxels),
                'num_occupied': len(classified['occupied']),
                'num_free': len(classified['free']),
                'num_unknown': len(classified['unknown']),
                'frame_count': self.frame_count,
                'processed_count': self.processed_frame_count,
                'bounds': {
                    'min': self.octree.min_bounds.copy() if self.octree.dynamic_expansion else None,
                    'max': self.octree.max_bounds.copy() if self.octree.dynamic_expansion else None
                }
            }
        else:
            occupied_voxels = self.octree.get_occupied_voxels(self.min_probability)
            
            if occupied_voxels:
                points = np.array([v[0] for v in occupied_voxels])
                probabilities = np.array([v[1] for v in occupied_voxels])
            else:
                points = np.empty((0, 3))
                probabilities = np.empty(0)
            
            return {
                'points': points,
                'probabilities': probabilities,
                'num_voxels': len(self.octree.voxels),
                'num_occupied': len(occupied_voxels),
                'frame_count': self.frame_count,
                'processed_count': self.processed_frame_count
            }
    
    def reset_map(self):
        """Reset the probabilistic map"""
        self.octree.clear()
        self.frame_count = 0
        self.processed_frame_count = 0
        self.total_processing_time = 0.0
        print("Map reset")


if __name__ == "__main__":
    # Test the module
    print("Testing 3D Mapper...")
    
    # Create instance with test config
    config = {
        'voxel_resolution': 0.1,
        'min_probability': 0.6,
        'intensity_threshold': 30
    }
    
    mapper = SonarTo3DMapper(config)
    
    # Create synthetic sonar image
    test_image = np.zeros((500, 512), dtype=np.uint8)
    test_image[100:150, 200:300] = 100  # Object at ~2m
    test_image[300:350, 100:150] = 150  # Object at ~6m
    
    # Process with fake odometry
    for i in range(3):
        position = [i * 0.1, 0, 0]
        orientation = [0, 0, 0, 1]  # Identity quaternion
        
        stats = mapper.process_sonar_image(test_image, position, orientation)
        print(f"Frame {i+1}: {stats}")
    
    # Get result
    result = mapper.get_point_cloud()
    print(f"\nGenerated {result['num_occupied']} occupied voxels")
    print(f"Total voxels: {result['num_voxels']}")
    print(f"Processed frames: {result['processed_count']}/{result['frame_count']}")