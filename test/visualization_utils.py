#!/usr/bin/env python3
"""
Visualization utilities for sonar occupancy mapping.

This module provides reusable visualization functions for 3D point clouds,
voxel grids, and sensor orientations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, Optional, Dict, Any, List


def create_cube_vertices(center: np.ndarray, size: float) -> List[List[np.ndarray]]:
    """
    Create vertices for a 3D cube.
    
    Args:
        center: 3D position of cube center [x, y, z]
        size: Side length of the cube
        
    Returns:
        List of 6 faces, each face is a list of 4 vertices
    """
    x, y, z = center
    s = size / 2
    
    # Define 8 vertices of the cube
    vertices = [
        [x-s, y-s, z-s], [x+s, y-s, z-s], [x+s, y+s, z-s], [x-s, y+s, z-s],  # bottom
        [x-s, y-s, z+s], [x+s, y-s, z+s], [x+s, y+s, z+s], [x-s, y+s, z+s]   # top
    ]
    
    # Define the 6 faces using the vertices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    return faces


def visualize_occupancy_grid_3d(points: np.ndarray, 
                                probabilities: np.ndarray,
                                sensor_position: Tuple[float, float, float],
                                grid_limits: Dict[str, Tuple[float, float]],
                                resolution: float,
                                processing_time: float,
                                frame_idx: int = 0,
                                display_mode: str = 'points') -> plt.Figure:
    """
    Visualize 3D occupancy grid with probability-based coloring.
    
    Args:
        points: Nx3 array of voxel center positions
        probabilities: N-length array of occupancy probabilities [0, 1]
        sensor_position: (x, y, z) position of the sensor
        grid_limits: Dictionary with 'x', 'y', 'z' range tuples
        resolution: Grid resolution in meters
        processing_time: Time taken to process the frame
        frame_idx: Frame index for title
        display_mode: 'points' or 'cubes' visualization mode
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(points) == 0:
        ax.text2D(0.5, 0.5, "No voxels updated from neutral state", 
                 transform=ax.transAxes, ha='center')
    else:
        # Separate voxels by probability
        free_mask = probabilities < 0.5
        occupied_mask = probabilities > 0.5
        
        if display_mode == 'cubes':
            voxel_size = resolution * 0.9  # Slightly smaller for visibility
            
            # Plot free voxels (blue gradient)
            if np.any(free_mask):
                free_points = points[free_mask]
                free_probs = probabilities[free_mask]
                
                for point, prob in zip(free_points, free_probs):
                    intensity = 1.0 - (prob / 0.5)  # 0.5 -> 0 (light), 0.0 -> 1 (dark)
                    color = (0, 0, intensity)  # Blue channel varies
                    alpha = 0.2 + 0.3 * intensity
                    
                    cube = create_cube_vertices(point, voxel_size)
                    poly = Poly3DCollection(cube, alpha=alpha, facecolor=color, edgecolor='none')
                    ax.add_collection3d(poly)
            
            # Plot occupied voxels (red gradient)
            if np.any(occupied_mask):
                occupied_points = points[occupied_mask]
                occupied_probs = probabilities[occupied_mask]
                
                for point, prob in zip(occupied_points, occupied_probs):
                    intensity = (prob - 0.5) / 0.5  # 0.5 -> 0 (light), 1.0 -> 1 (dark)
                    color = (intensity, 0, 0)  # Red channel varies
                    alpha = 0.4 + 0.4 * intensity
                    
                    cube = create_cube_vertices(point, voxel_size)
                    poly = Poly3DCollection(cube, alpha=alpha, facecolor=color, edgecolor='none')
                    ax.add_collection3d(poly)
            
            # Create legend entries
            ax.scatter([], [], [], c='blue', marker='s', s=100, alpha=0.5, 
                      label=f'Free ({np.sum(free_mask):,})')
            ax.scatter([], [], [], c='red', marker='s', s=100, alpha=0.7, 
                      label=f'Occupied ({np.sum(occupied_mask):,})')
            
        else:  # points mode
            # Plot free space with blue gradient
            if np.any(free_mask):
                free_points = points[free_mask]
                free_probs = probabilities[free_mask]
                ax.scatter(free_points[:, 0], free_points[:, 1], free_points[:, 2],
                          c=free_probs, cmap='Blues_r', vmin=0.0, vmax=0.5,
                          s=1, alpha=0.3)
            
            # Plot occupied space with red gradient
            if np.any(occupied_mask):
                occupied_points = points[occupied_mask]
                occupied_probs = probabilities[occupied_mask]
                ax.scatter(occupied_points[:, 0], occupied_points[:, 1], occupied_points[:, 2],
                          c=occupied_probs, cmap='Reds', vmin=0.5, vmax=1.0,
                          s=3, alpha=0.7)
            
            # Add legend entries
            ax.scatter([], [], [], c='blue', s=50, alpha=0.5, 
                      label=f'Free (<0.5): {np.sum(free_mask):,}')
            ax.scatter([], [], [], c='red', s=50, alpha=0.7, 
                      label=f'Occupied (>0.5): {np.sum(occupied_mask):,}')
    
    # Mark sensor position
    ax.scatter(*sensor_position, c='black', s=100, marker='s', label='Sensor')
    
    # Set labels and properties
    ax.set_xlabel('X (Lateral) [m]')
    ax.set_ylabel('Y (Longitudinal) [m]')
    ax.set_zlabel('Z (Depth) [m]')
    ax.set_title(f'Ray-based 3D Occupancy Grid - Frame {frame_idx}\n'
                 f'Resolution: {resolution*1000:.0f}mm, '
                 f'Processing time: {processing_time:.2f}s', 
                 fontsize=14, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(grid_limits['x'])
    ax.set_ylim(grid_limits['y'])
    ax.set_zlim(grid_limits['z'])
    
    # Add legend
    if len(points) > 0:
        ax.legend(loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set initial viewing angle
    ax.view_init(elev=20, azim=-60)
    
    # Enable mouse interaction
    ax.mouse_init()
    
    plt.tight_layout()
    
    return fig


def visualize_sensor_orientation(sensor_position: Tuple[float, float, float],
                                 sensor_pose: Tuple[float, float, float],
                                 transform_matrix: np.ndarray,
                                 axis_length: float = 0.3) -> None:
    """
    Visualize sensor coordinate axes after transformation.
    
    Args:
        sensor_position: (x, y, z) position of sensor
        sensor_pose: (heading, tilt, roll) in degrees
        transform_matrix: 4x4 homogeneous transformation matrix
        axis_length: Length of coordinate axes for visualization
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Transform coordinate axes
    origin = np.array([0, 0, 0, 1])
    x_axis = np.array([axis_length, 0, 0, 1])
    y_axis = np.array([0, axis_length, 0, 1])
    z_axis = np.array([0, 0, axis_length, 1])
    
    # Apply transformation
    origin_t = (transform_matrix @ origin)[:3]
    x_axis_t = (transform_matrix @ x_axis)[:3]
    y_axis_t = (transform_matrix @ y_axis)[:3]
    z_axis_t = (transform_matrix @ z_axis)[:3]
    
    # Plot transformed axes
    ax.plot([origin_t[0], x_axis_t[0]], 
            [origin_t[1], x_axis_t[1]], 
            [origin_t[2], x_axis_t[2]], 
            'r-', linewidth=3, label='X (Forward)')
    
    ax.plot([origin_t[0], y_axis_t[0]], 
            [origin_t[1], y_axis_t[1]], 
            [origin_t[2], y_axis_t[2]], 
            'g-', linewidth=3, label='Y (Right)')
    
    ax.plot([origin_t[0], z_axis_t[0]], 
            [origin_t[1], z_axis_t[1]], 
            [origin_t[2], z_axis_t[2]], 
            'b-', linewidth=3, label='Z (Down)')
    
    # Mark sensor position
    ax.scatter(*sensor_position, c='black', s=100, marker='o')
    
    # Draw water surface
    xx, yy = np.meshgrid([-0.5, 0.5], [-0.5, 0.5])
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='cyan')
    
    # Set labels and title
    ax.set_xlabel('Map X [m]')
    ax.set_ylabel('Map Y [m]')
    ax.set_zlabel('Map Z [m]')
    ax.set_title(f'Sensor Orientation\n'
                 f'Heading: {sensor_pose[0]:.1f}°, '
                 f'Tilt: {sensor_pose[1]:.1f}°, '
                 f'Roll: {sensor_pose[2]:.1f}°',
                 fontsize=12, fontweight='bold')
    
    # Set limits
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    plt.show()


def print_processing_summary(processing_stats: Dict[str, Any],
                            mapping_stats: Dict[str, Any],
                            frame_idx: int,
                            resolution: float) -> None:
    """
    Print formatted processing summary.
    
    Args:
        processing_stats: Dictionary with processing statistics
        mapping_stats: Dictionary with mapping statistics
        frame_idx: Frame index
        resolution: Grid resolution in meters
    """
    print("\n" + "="*60)
    print("RAY-BASED MAPPING SUMMARY")
    print("="*60)
    print(f"Processed frame: {frame_idx}")
    print(f"Grid resolution: {resolution*1000:.0f}mm")
    print(f"Processing time: {processing_stats['processing_time']:.3f}s")
    print(f"\nVoxel Classification:")
    print(f"  Free space: {mapping_stats['free_voxels']:,} voxels "
          f"({mapping_stats['free_percentage']:.1f}%)")
    print(f"  Unknown: {mapping_stats['unknown_voxels']:,} voxels "
          f"({mapping_stats['unknown_percentage']:.1f}%)")
    print(f"  Occupied: {mapping_stats['occupied_voxels']:,} voxels "
          f"({mapping_stats['occupied_percentage']:.1f}%)")
    print("="*60)