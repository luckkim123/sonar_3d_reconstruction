#!/usr/bin/env python3
"""
Minimal launch file for sonar mapping node
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('sonar_3d_reconstruction')
    config = os.path.join(pkg, 'config', 'octree_mapper.yaml')
    
    return LaunchDescription([
        Node(
            package='sonar_3d_reconstruction',
            executable='octree_sonar_mapper_node.py',
            name='octree_sonar_mapper',
            parameters=[config, {'use_sim_time': True}],
            output='screen'
        )
    ])