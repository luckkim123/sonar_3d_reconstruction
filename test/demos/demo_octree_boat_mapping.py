#!/usr/bin/env python3
"""
Octree-based Boat Sonar Occupancy Mapping Demonstration

This demo loads real sonar data from ROS2 bag files and processes it using
an octree-based occupancy mapping approach with boat coordinate transformation.
Uses odometry data to track boat position and orientation.

Key features:
- Sparse octree storage (memory efficient)
- Dynamic map expansion (no fixed boundaries)
- Boat coordinate transformation with odometry
- Probability-based visualization

Author: Sonar 3D Reconstruction Team
Date: 2025-08-18
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import time
from tqdm import tqdm

# Import custom modules
from octree_mapper_proper import OctreeBasedSonarMapperWithBoat
from bag_processor import BagProcessor
from config import MappingConfig
from visualization_utils import print_processing_summary


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Octree-based Sonar Occupancy Mapping Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--bag-path', type=str, default=None,
                       help='Path to ROS2 bag file')
    parser.add_argument('--odometry-csv', type=str, 
                       default='/workspace/ros2_ws/src/sonar_3d_reconstruction/test/full_odometry_data.csv',
                       help='Path to odometry CSV file')
    parser.add_argument('--num-frames', type=int, default=3,
                       help='Number of frames to process')
    parser.add_argument('--frame-step', type=int, default=15,
                       help='Frame interval (1=consecutive, 15=1sec interval at 15Hz)')
    
    # Grid configuration (octree with optional initial bounds)
    parser.add_argument('--resolution', type=float, default=0.03,
                       help='Voxel resolution in meters')
    parser.add_argument('--dynamic-map', action='store_true',
                       help='Enable dynamic map expansion (ignore initial bounds, start small and expand with robot movement)')
    parser.add_argument('--initial-size', type=float, default=1.0,
                       help='Initial map size radius in meters (used only with --dynamic-map)')
    parser.add_argument('--x-range', nargs=2, type=float, default=[-2.0, 5.0],
                       help='X-axis range (min max) in meters. Ignored if --dynamic-map is used')
    parser.add_argument('--y-range', nargs=2, type=float, default=[-2.0, 5.0],
                       help='Y-axis range (min max) in meters. Ignored if --dynamic-map is used')
    parser.add_argument('--z-range', nargs=2, type=float, default=[-1.5, 0],
                       help='Z-axis range (min max) in meters. Ignored if --dynamic-map is used')
    
    # Sensor configuration
    parser.add_argument('--sensor-x', type=float, default=0.0,
                       help='Sensor X position')
    parser.add_argument('--sensor-y', type=float, default=0.0,
                       help='Sensor Y position')
    parser.add_argument('--sensor-z', type=float, default=0.0,
                       help='Sensor Z position')
    parser.add_argument('--heading', type=float, default=0.0,
                       help='Sensor heading in degrees')
    parser.add_argument('--tilt', type=float, default=0.0,
                       help='Sensor tilt in degrees')
    parser.add_argument('--roll', type=float, default=0.0,
                       help='Sensor roll in degrees')
    
    # Sonar parameters
    parser.add_argument('--max-range', type=float, default=5.0,
                       help='Maximum sonar range in meters')
    parser.add_argument('--fov', type=float, default=130.0,
                       help='Horizontal field of view in degrees')
    parser.add_argument('--vertical-aperture', type=float, default=20.0,
                       help='Vertical beam aperture in degrees')
    parser.add_argument('--intensity-threshold', type=int, default=35,
                       help='Intensity threshold for occupied detection')
    parser.add_argument('--min-range', type=float, default=0.5,
                       help='Minimum range to filter noise')
    
    # FOV filtering parameters
    parser.add_argument('--fov-margin', type=float, default=0.0,
                       help='FOV margin to exclude from processing (degrees). Removes noisy edges')
    
    # Visualization
    parser.add_argument('--no-show', action='store_true',
                       help='Do not show visualization')
    parser.add_argument('--save-fig', type=str, default=None,
                       help='Save figure to file')
    parser.add_argument('--show-per-frame', action='store_true',
                       help='Show separate 3D visualization after each frame')
    parser.add_argument('--viz-mode', type=str, default='point', choices=['point', 'cube'],
                       help='Visualization mode: point (scatter) or cube (voxel)')
    parser.add_argument('--point-size', type=float, default=1.0,
                       help='Point size for scatter visualization (0.5-10.0)')
    parser.add_argument('--aspect-mode', type=str, default='auto', choices=['equal', 'auto'],
                       help='Aspect ratio mode for 3D plot')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame number')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                       help='Compare with dense grid mapper')
    
    return parser.parse_args()


def initialize_mapper(args: argparse.Namespace) -> OctreeBasedSonarMapperWithBoat:
    """
    Initialize the octree-based sonar mapper with boat transformation.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Initialized OctreeBasedSonarMapperWithBoat object
    """
    print("\nInitializing octree-based sonar mapper...")
    
    # Determine map configuration
    if args.dynamic_map:
        initial_map_bounds = None
        print(f"  🔄 Dynamic map expansion enabled")
        print(f"     Initial size: ±{args.initial_size:.1f}m radius around robot")
        print("     Map will expand automatically as robot moves")
    else:
        # Calculate initial map size from ranges
        initial_map_bounds = (args.x_range, args.y_range, args.z_range)
        x_size = args.x_range[1] - args.x_range[0]
        y_size = args.y_range[1] - args.y_range[0]
        z_size = args.z_range[1] - args.z_range[0]
        print(f"  📏 Fixed map bounds: X[{args.x_range[0]:.1f}, {args.x_range[1]:.1f}], "
              f"Y[{args.y_range[0]:.1f}, {args.y_range[1]:.1f}], Z[{args.z_range[0]:.1f}, {args.z_range[1]:.1f}]m")
        print(f"     Map size: {x_size:.1f}×{y_size:.1f}×{z_size:.1f}m")
    
    print("  💾 Sparse storage - only stores measured voxels")
    
    mapper = OctreeBasedSonarMapperWithBoat(
        resolution=args.resolution,
        sensor_position=(args.sensor_x, args.sensor_y, args.sensor_z),
        sensor_pose=(args.heading, args.tilt, args.roll),
        max_range=args.max_range,
        fov_degrees=args.fov,
        vertical_aperture_degrees=args.vertical_aperture,
        intensity_threshold=args.intensity_threshold,
        min_range=args.min_range,
        initial_map_bounds=initial_map_bounds,
        dynamic_expansion=args.dynamic_map,
        fov_margin_degrees=args.fov_margin
    )
    
    return mapper


def create_synthetic_test_data(num_frames: int = 3) -> tuple:
    """
    Create synthetic sonar data for testing.
    
    Args:
        num_frames: Number of frames to generate
        
    Returns:
        (frame_data, odometry_data) tuple
    """
    print(f"\nGenerating {num_frames} synthetic test frames...")
    
    # Create synthetic sonar frame
    range_bins = 200
    bearing_bins = 256
    frame = np.zeros((range_bins, bearing_bins), dtype=np.uint8)
    
    # Add some objects
    # Object 1: Wall at 2m
    wall_range = int(2.0 / 5.0 * range_bins)
    frame[wall_range-2:wall_range+2, 100:150] = 150
    
    # Object 2: Point target at 1.5m
    target_range = int(1.5 / 5.0 * range_bins)
    frame[target_range-1:target_range+1, 50:55] = 200
    
    # Create synthetic odometry for requested number of frames
    odometry_data = []
    frames = []
    
    for i in range(num_frames):
        # Generate position along a path
        t = i / max(1, num_frames - 1) if num_frames > 1 else 0
        position = (t * 1.0, t * 0.3, 0.0)  # Linear movement
        
        # Generate rotation (yaw increases with position)
        yaw_rad = t * np.radians(20)  # Up to 20 degrees rotation
        quaternion = (0.0, 0.0, np.sin(yaw_rad/2), np.cos(yaw_rad/2))
        
        odometry_data.append({
            'position': position,
            'quaternion': quaternion,
            'timestamp': float(i)
        })
        frames.append(frame.copy())
    
    return frames, odometry_data


def process_with_octree(mapper: OctreeBasedSonarMapperWithBoat,
                        frames: list,
                        odometry_data: list,
                        show_per_frame: bool = False,
                        viz_mode: str = 'point',
                        point_size: float = 1.0,
                        aspect_mode: str = 'equal') -> Dict[str, Any]:
    """
    Process sonar frames with octree mapper.
    
    Args:
        mapper: Initialized octree mapper
        frames: List of sonar frames
        odometry_data: List of odometry measurements
        
    Returns:
        Processing statistics
    """
    print("\n🔄 Processing with octree mapper...")
    start_time = time.time()
    
    total_stats = {
        'processing_time': 0.0,
        'rays_processed': 0,
        'free_updates': 0,
        'occupied_updates': 0,
        'frame_data': frames,
        'odometry_data': odometry_data
    }
    
    # Store mappers for each frame if showing per-frame
    frame_mappers = [] if show_per_frame else None
    
    # Use tqdm for clean progress display
    with tqdm(total=len(frames), desc="  Processing frames", ncols=80, unit="frame") as pbar:
        for i, (frame, odom) in enumerate(zip(frames, odometry_data)):
            # Update boat pose on main mapper
            mapper.set_boat_pose_from_odometry(odom['position'], odom['quaternion'])
            
            # Process frame on main mapper
            frame_stats = mapper.process_polar_image_voxel_centric(frame)
            
            # Accumulate statistics
            total_stats['processing_time'] += frame_stats['processing_time']
            total_stats['rays_processed'] += frame_stats['rays_processed']
            total_stats['free_updates'] += frame_stats.get('free_updates', 0)
            total_stats['occupied_updates'] += frame_stats.get('occupied_updates', 0)
            
            # Store a deep copy of mapper state for per-frame visualization
            if show_per_frame:
                # Create a new mapper with same configuration for visualization
                vis_mapper = OctreeBasedSonarMapperWithBoat(
                    resolution=mapper.resolution,
                    sensor_position=mapper.sensor_position,
                    sensor_pose=(mapper.heading_degrees, mapper.tilt_degrees, mapper.roll_degrees),
                    max_range=mapper.max_range,
                    fov_degrees=mapper.fov_degrees,
                    vertical_aperture_degrees=mapper.vertical_aperture_degrees,
                    intensity_threshold=mapper.intensity_threshold,
                    min_range=mapper.min_range,
                    initial_map_bounds=([mapper.octree.min_bounds[0], mapper.octree.max_bounds[0]],
                                      [mapper.octree.min_bounds[1], mapper.octree.max_bounds[1]],
                                      [mapper.octree.min_bounds[2], mapper.octree.max_bounds[2]])
                )
                # Deep copy current octree state
                vis_mapper.octree.voxels = dict(mapper.octree.voxels)
                vis_mapper.octree.min_bounds = mapper.octree.min_bounds.copy()
                vis_mapper.octree.max_bounds = mapper.octree.max_bounds.copy()
                
                frame_mappers.append((vis_mapper, frames[:i+1], odometry_data[:i+1], i+1))
            
            # Update progress bar
            pbar.update(1)
    
    print("\n  ✅ All frames processed!")
    
    # Show all frame visualizations at once if requested
    if show_per_frame:
        import matplotlib.pyplot as plt
        figures = []
        for mapper_state, frame_data, odom_data, frame_num in frame_mappers:
            fig = visualize_octree_results(
                mapper=mapper_state,
                stats={'frame_data': frame_data, 'odometry_data': odom_data,
                       'processing_time': total_stats['processing_time'],
                       'rays_processed': total_stats['rays_processed'],
                       'mapping_stats': mapper_state.get_statistics()},
                save_path=None,
                show=False,  # Don't show yet, just return figure
                title_suffix=f" - Frame {frame_num}/{len(frames)}",
                viz_mode=viz_mode,
                point_size=point_size,
                aspect_mode=aspect_mode)
            figures.append(fig)
        plt.show()  # Show all figures at once
    
    total_stats['processing_time'] = time.time() - start_time
    total_stats['mapping_stats'] = mapper.get_statistics()
    
    return total_stats


def draw_voxel_cube(ax, position, size, color, alpha):
    """Draw a cube at the given position."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Define the vertices of a cube
    r = size / 2
    x, y, z = position
    vertices = [
        [x-r, y-r, z-r], [x+r, y-r, z-r], [x+r, y+r, z-r], [x-r, y+r, z-r],  # bottom
        [x-r, y-r, z+r], [x+r, y-r, z+r], [x+r, y+r, z+r], [x-r, y+r, z+r]   # top
    ]
    
    # Define the 6 faces of the cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    # Create the 3D polygon collection
    cube = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='none')
    ax.add_collection3d(cube)


def visualize_octree_results(mapper: OctreeBasedSonarMapperWithBoat,
                             stats: Dict[str, Any],
                             save_path: Optional[str] = None,
                             show: bool = True,
                             title_suffix: str = "",
                             viz_mode: str = 'point',
                             point_size: float = 1.0,
                             aspect_mode: str = 'equal') -> Optional[plt.Figure]:
    """
    Visualize the octree mapping results with probability-based colors.
    Returns the figure if show=False, None otherwise.
    """
    print("\n📊 Creating octree visualization...")
    
    # Get all voxels with probabilities
    all_voxels = []
    all_probs = []
    for key, log_odds in mapper.octree.voxels.items():
        world_pos = mapper.octree.key_to_world(key)
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        all_voxels.append(world_pos)
        all_probs.append(prob)
    
    all_voxels = np.array(all_voxels) if all_voxels else np.array([]).reshape(0, 3)
    all_probs = np.array(all_probs) if all_probs else np.array([])
    
    # Get all sensor positions for trajectory
    sensor_positions = []
    for odom in stats.get('odometry_data', []):
        mapper.set_boat_pose_from_odometry(odom['position'], odom['quaternion'])
        sensor_pos = mapper.sonar_to_map_coords(0, 0, 0)
        sensor_positions.append(sensor_pos)
    
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(14, 10))
    from matplotlib.gridspec import GridSpec
    
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1], width_ratios=[1, 1])
    
    # 3D plot - centered on top half
    ax1 = fig.add_subplot(gs[0, :], projection='3d')
    
    # Plot voxels with probability-based colors
    if len(all_voxels) > 0:
        # Separate by probability threshold
        low_prob_mask = all_probs < 0.5
        high_prob_mask = ~low_prob_mask
        
        # Plot low probability voxels (< 0.5) - semi-transparent blue
        if np.any(low_prob_mask):
            low_voxels = all_voxels[low_prob_mask]
            # Subsample if too many
            if len(low_voxels) > 5000:
                idx = np.random.choice(len(low_voxels), 5000, replace=False)
                low_voxels = low_voxels[idx]
            
            if viz_mode == 'cube':
                # Draw cubes for voxels
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                for voxel_pos in low_voxels[::max(1, len(low_voxels)//1000)]:  # Limit cubes for performance
                    draw_voxel_cube(ax1, voxel_pos, mapper.resolution, 
                                   color='blue', alpha=0.1)
            else:
                # Draw points
                ax1.scatter(low_voxels[:, 0], low_voxels[:, 1], low_voxels[:, 2],
                           c='blue', s=point_size, alpha=0.1, label=f'Free (p<0.5): {np.sum(low_prob_mask)}')
        
        # Plot high probability voxels (>= 0.5) - red gradient based on probability
        if np.any(high_prob_mask):
            high_voxels = all_voxels[high_prob_mask]
            high_probs = all_probs[high_prob_mask]
            
            if viz_mode == 'cube':
                # Draw cubes with gradient colors
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                import matplotlib.cm as cm
                cmap = cm.get_cmap('Reds')
                
                # Limit number of cubes for performance
                max_cubes = 2000
                if len(high_voxels) > max_cubes:
                    idx = np.random.choice(len(high_voxels), max_cubes, replace=False)
                    cube_voxels = high_voxels[idx]
                    cube_probs = high_probs[idx]
                else:
                    cube_voxels = high_voxels
                    cube_probs = high_probs
                
                for voxel_pos, prob in zip(cube_voxels, cube_probs):
                    # Map probability to color
                    norm_prob = (prob - 0.5) / 0.5  # Normalize 0.5-1.0 to 0-1
                    color = cmap(norm_prob)
                    draw_voxel_cube(ax1, voxel_pos, mapper.resolution,
                                   color=color[:3], alpha=0.9)
                
                # Add dummy scatter for colorbar
                dummy = ax1.scatter([0], [0], [0], c=[0], cmap='Reds', vmin=0.5, vmax=1.0, s=0)
                cbar = plt.colorbar(dummy, ax=ax1, pad=0.1, shrink=0.6)
                cbar.set_label('Occupancy Probability', rotation=270, labelpad=15)
            else:
                # Draw points with gradient
                scatter = ax1.scatter(high_voxels[:, 0], high_voxels[:, 1], high_voxels[:, 2],
                                     c=high_probs, cmap='Reds', vmin=0.5, vmax=1.0,
                                     s=point_size * 3, alpha=0.9, label=f'Occupied (p≥0.5): {np.sum(high_prob_mask)}')
                
                # Add colorbar for occupied voxels
                cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.6)
                cbar.set_label('Occupancy Probability', rotation=270, labelpad=15)
    
    # Plot sensor trajectory
    if len(sensor_positions) > 0:
        positions_array = np.array(sensor_positions)
        ax1.plot(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2],
                'g--', alpha=0.5, linewidth=2, label='Boat Path')
        ax1.scatter(positions_array[-1, 0], positions_array[-1, 1], positions_array[-1, 2],
                   c='lime', s=100, marker='o', edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Octree 3D Occupancy Map')
    ax1.legend()
    
    # Auto-set limits based on octree bounds
    if len(mapper.octree.voxels) > 0:
        margin = 0.5
        ax1.set_xlim(mapper.octree.min_bounds[0] - margin, mapper.octree.max_bounds[0] + margin)
        ax1.set_ylim(mapper.octree.min_bounds[1] - margin, mapper.octree.max_bounds[1] + margin)
        ax1.set_zlim(mapper.octree.min_bounds[2] - margin, mapper.octree.max_bounds[2] + margin)
    
    # Set aspect ratio
    if aspect_mode == 'equal':
        # Make axes have equal aspect ratio
        ax1.set_box_aspect([1,1,1])
    elif aspect_mode == 'auto':
        # Use automatic aspect ratio based on data ranges
        if len(mapper.octree.voxels) > 0:
            x_range = mapper.octree.max_bounds[0] - mapper.octree.min_bounds[0]
            y_range = mapper.octree.max_bounds[1] - mapper.octree.min_bounds[1] 
            z_range = mapper.octree.max_bounds[2] - mapper.octree.min_bounds[2]
            # Normalize the largest dimension to 1, others proportionally
            max_range = max(x_range, y_range, z_range)
            if max_range > 0:
                aspect_ratios = [x_range/max_range, y_range/max_range, z_range/max_range]
                ax1.set_box_aspect(aspect_ratios)
    
    # Top view - bottom left
    ax2 = fig.add_subplot(gs[1, 0])
    if len(all_voxels) > 0:
        # Plot low probability voxels - blue
        if np.any(low_prob_mask):
            ax2.scatter(all_voxels[low_prob_mask, 0], all_voxels[low_prob_mask, 1],
                       c='blue', s=0.5, alpha=0.1)
        # Plot high probability voxels - red gradient
        if np.any(high_prob_mask):
            ax2.scatter(all_voxels[high_prob_mask, 0], all_voxels[high_prob_mask, 1],
                       c=all_probs[high_prob_mask], cmap='Reds', vmin=0.5, vmax=1.0,
                       s=1, alpha=0.9)
    
    if len(sensor_positions) > 0:
        positions_array = np.array(sensor_positions)
        ax2.plot(positions_array[:, 0], positions_array[:, 1], 'g--', alpha=0.5, linewidth=2)
        ax2.scatter(positions_array[-1, 0], positions_array[-1, 1], c='lime', s=50, marker='o')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (XY)')
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    # Statistics panel - bottom right
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    octree_stats = stats['mapping_stats'].get('octree_stats', {})
    bounds_size = octree_stats.get('bounds_size', [0, 0, 0])
    
    # Count voxels by probability
    free_count = np.sum(all_probs < 0.5) if len(all_probs) > 0 else 0
    occupied_count = np.sum(all_probs >= 0.5) if len(all_probs) > 0 else 0
    
    stats_text = f"""Octree Mapping Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Storage Efficiency:
  Stored Voxels: {len(mapper.octree.voxels):,}
  Resolution: {mapper.resolution*1000:.0f}mm
  
Dynamic Bounds:
  Size: {bounds_size[0]:.1f} × {bounds_size[1]:.1f} × {bounds_size[2]:.1f}m
  Min: ({octree_stats.get('bounds_min', [0,0,0])[0]:.2f}, {octree_stats.get('bounds_min', [0,0,0])[1]:.2f}, {octree_stats.get('bounds_min', [0,0,0])[2]:.2f})
  Max: ({octree_stats.get('bounds_max', [0,0,0])[0]:.2f}, {octree_stats.get('bounds_max', [0,0,0])[1]:.2f}, {octree_stats.get('bounds_max', [0,0,0])[2]:.2f})
  
Probability-Based Classification:
  🔵 Free (p<0.5): {free_count:,} ({free_count/len(mapper.octree.voxels)*100 if len(mapper.octree.voxels) > 0 else 0:.1f}%)
  🔴 Occupied (p≥0.5): {occupied_count:,} ({occupied_count/len(mapper.octree.voxels)*100 if len(mapper.octree.voxels) > 0 else 0:.1f}%)
  
Processing:
  Time: {stats['processing_time']:.2f}s
  Frames: {len(stats.get('frame_data', []))}
  Rays: {stats['rays_processed']:,}
"""
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top')
    
    # Add memory efficiency info to stats text
    if bounds_size[0] > 0 and bounds_size[1] > 0 and bounds_size[2] > 0:
        dense_voxels = int((bounds_size[0] / mapper.resolution) * 
                          (bounds_size[1] / mapper.resolution) * 
                          (bounds_size[2] / mapper.resolution))
        sparse_voxels = len(mapper.octree.voxels)
        memory_ratio = dense_voxels / sparse_voxels if sparse_voxels > 0 else float('inf')
        
        memory_text = f"""
Memory Level:
  Dense: {dense_voxels:,}
  Sparse: {sparse_voxels:,}
  Ratio: {memory_ratio:.1f}x"""
        
        # Append to existing stats text
        stats_text += memory_text
        
        # Update the text
        ax3.clear()
        ax3.axis('off')
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                fontsize=9, fontfamily='monospace', verticalalignment='top')
    
    plt.suptitle(f'Octree-based Sonar Mapping (Dynamic Expansion){title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        # Ensure output directory exists
        output_dir = os.path.join(os.path.dirname(os.path.dirname(save_path)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(save_path))
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    
    if show:
        plt.show()
        return None
    else:
        return fig  # Return figure for later display


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print("="*80)
    print("OCTREE-BASED SONAR OCCUPANCY MAPPING DEMO")
    print("Dynamic Map Expansion | Sparse Storage | Memory Efficient")
    print("="*80)
    
    # Initialize mapper
    mapper = initialize_mapper(args)
    
    # Load or create test data
    if args.bag_path and os.path.exists(args.bag_path):
        print(f"\nLoading data from: {args.bag_path}")
        # TODO: Implement bag loading
        print("  ⚠️  Bag loading not yet implemented, using synthetic data")
        frames, odometry_data = create_synthetic_test_data(args.num_frames)
    else:
        # Use real data from default bag file like demo_boat_mapping.py does
        from bag_processor import BagProcessor
        from config import MappingConfig
        
        config = MappingConfig()
        bag_path = config.data.get_bag_path()
        
        if bag_path.exists():
            print(f"\nLoading real data from: {bag_path}")
            processor = BagProcessor(str(bag_path))
            
            # Load odometry
            odometry_data = mapper.load_odometry_from_csv(args.odometry_csv)
            print(f"Loaded {len(odometry_data)} odometry measurements")
            
            # Extract frames
            frames = []
            selected_odometry = []
            
            # Select frame indices based on start_frame and frame_step
            frame_indices = []
            for i in range(args.num_frames):
                frame_idx = args.start_frame + (i * args.frame_step)
                if frame_idx < 1280:  # Total frames in this bag
                    frame_indices.append(frame_idx)
                else:
                    break
            
            if len(frame_indices) < args.num_frames:
                print(f"⚠️  Only {len(frame_indices)} frames available starting from frame {args.start_frame}")
            
            print(f"Selecting frames: {frame_indices} (start={args.start_frame}, step={args.frame_step})")
            
            # Pre-extract timestamps for frame mapping and establish time baseline
            frame_timestamps = {}
            first_sonar_time = None
            
            print("Extracting frame timestamps...")
            for idx in frame_indices:
                frame, metadata = processor.sonar_extractor.extract_oculus_frame(idx)
                if frame is not None:
                    timestamp_ns = metadata.get('timestamp_ns', 0)
                    frame_timestamps[idx] = timestamp_ns
                    if first_sonar_time is None:
                        first_sonar_time = timestamp_ns
            
            # Get odometry baseline time (first odometry timestamp converted to seconds)
            first_odom_time = odometry_data[0]['timestamp']
            
            print(f"Time baseline: first_sonar={first_sonar_time*1e-9:.3f}s, first_odom={first_odom_time:.3f}s")
            
            # Now process frames with relative time matching
            for idx in frame_indices:
                frame, metadata = processor.sonar_extractor.extract_oculus_frame(idx)
                if frame is not None:
                    frames.append(frame)
                    
                    # Convert to relative times (seconds from first frame/measurement)
                    timestamp_ns = frame_timestamps[idx]
                    sonar_relative_time = (timestamp_ns - first_sonar_time) * 1e-9  # seconds from first sonar
                    
                    # Find closest odometry using relative time matching
                    best_odom = None
                    best_diff = float('inf')
                    for odom in odometry_data:
                        odom_relative_time = odom['timestamp'] - first_odom_time  # seconds from first odom
                        diff = abs(odom_relative_time - sonar_relative_time)
                        if diff < best_diff:
                            best_diff = diff
                            best_odom = odom
                    
                    # Debug info for verification
                    if len(frame_indices) <= 10:  # Show debug for reasonable test sizes
                        best_odom_rel_time = best_odom['timestamp'] - first_odom_time
                        print(f"  Frame {idx}: sonar_rel_time={sonar_relative_time:.3f}s, odom_rel_time={best_odom_rel_time:.3f}s, diff={best_diff:.3f}s")
                        print(f"           position=({best_odom['position'][0]:.3f}, {best_odom['position'][1]:.3f}, {best_odom['position'][2]:.3f})")
                    
                    selected_odometry.append(best_odom)
            
            processor.close()
            odometry_data = selected_odometry
            print(f"Extracted {len(frames)} frames from bag file")
        else:
            print(f"\nBag file not found, using synthetic data")
            frames, odometry_data = create_synthetic_test_data(args.num_frames)
    
    # Process with octree
    stats = process_with_octree(mapper, frames, odometry_data, show_per_frame=args.show_per_frame,
                               viz_mode=args.viz_mode, point_size=args.point_size, aspect_mode=args.aspect_mode)
    
    # Print summary
    print("\n" + "="*60)
    print("OCTREE MAPPING COMPLETE")
    print("="*60)
    print(f"Total voxels stored: {len(mapper.octree.voxels):,} (sparse)")
    print(f"Map bounds: {mapper.octree.max_bounds - mapper.octree.min_bounds}")
    print(f"Processing time: {stats['processing_time']:.2f}s")
    
    # Visualize results
    visualize_octree_results(
        mapper=mapper,
        stats=stats,
        save_path=args.save_fig,
        show=not args.no_show,
        viz_mode=args.viz_mode,
        point_size=args.point_size,
        aspect_mode=args.aspect_mode
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())