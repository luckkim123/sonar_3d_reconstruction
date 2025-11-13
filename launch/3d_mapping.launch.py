#!/usr/bin/env python3
"""
Launch file for 3D Sonar Mapping System
Launches both Fast-LIO and 3D mapper nodes with optional bag playback

Author: Sonar 3D Reconstruction Team
Date: 2025
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directories
    sonar_pkg = get_package_share_directory('sonar_3d_reconstruction')
    fast_lio_pkg = get_package_share_directory('fast_lio')
    
    # Config files
    mapper_config = "/workspace/ros2_ws/src/sonar_3d_reconstruction/config/3d_mapper.yaml"
    fast_lio_config = os.path.join(fast_lio_pkg, 'config', 'mid360.yaml')
    rviz_config = "/workspace/ros2_ws/src/sonar_3d_reconstruction/rviz/3d_mapping.rviz"
    
    # Load YAML configuration to get default values
    import yaml
    with open(mapper_config, 'r') as f:
        yaml_config = yaml.safe_load(f)
        yaml_params = yaml_config['sonar_3d_mapper']['ros__parameters']
    
    # Launch arguments (can be overridden by command line)
    # Priority: command line args > YAML defaults
    use_sim_time = LaunchConfiguration('use_sim_time')
    launch_fast_lio = LaunchConfiguration('launch_fast_lio')
    launch_rviz = LaunchConfiguration('launch_rviz')
    play_bag = LaunchConfiguration('play_bag')
    bag_file = LaunchConfiguration('bag_file')
    bag_playback_rate = LaunchConfiguration('bag_playback_rate')
    record_bag = LaunchConfiguration('record_bag')
    record_output_path = LaunchConfiguration('record_output_path')
    
    # Sonar orientation parameters
    sonar_roll = LaunchConfiguration('sonar_orientation.roll')
    sonar_pitch = LaunchConfiguration('sonar_orientation.pitch')
    sonar_yaw = LaunchConfiguration('sonar_orientation.yaw')
    
    # Declare launch arguments with YAML defaults
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value=str(yaml_params.get('use_sim_time', True)),
        description='Use simulation time for bag playback'
    )
    
    declare_launch_fast_lio_cmd = DeclareLaunchArgument(
        'launch_fast_lio',
        default_value=str(yaml_params.get('launch_fast_lio', True)),
        description='Launch Fast-LIO for odometry'
    )
    
    declare_launch_rviz_cmd = DeclareLaunchArgument(
        'launch_rviz',
        default_value=str(yaml_params.get('launch_rviz', True)),
        description='Launch RViz for visualization'
    )
    
    declare_play_bag_cmd = DeclareLaunchArgument(
        'play_bag',
        default_value=str(yaml_params.get('play_bag', False)),
        description='Play ROS2 bag file'
    )
    
    declare_bag_file_cmd = DeclareLaunchArgument(
        'bag_file',
        default_value=yaml_params.get('bag_file', '/workspace/data/2_kiro_watertank/20250926_blueboat_sonar_lidar/oculus-tilt90-gain50-freq2'),
        description='Path to ROS2 bag file'
    )
    
    declare_bag_playback_rate_cmd = DeclareLaunchArgument(
        'bag_playback_rate',
        default_value=str(yaml_params.get('bag_playback_rate', 1.0)),
        description='Bag playback rate (1.0 = normal, 0.5 = half speed, 2.0 = double speed)'
    )
    
    declare_record_bag_cmd = DeclareLaunchArgument(
        'record_bag',
        default_value='false',
        description='Record ROS2 bag file during mapping'
    )
    
    declare_record_output_path_cmd = DeclareLaunchArgument(
        'record_output_path',
        default_value='/workspace/data/recorded_mapping',
        description='Path to save recorded bag file'
    )
    
    # Declare sonar orientation arguments
    declare_sonar_roll_cmd = DeclareLaunchArgument(
        'sonar_orientation.roll',
        default_value=str(yaml_params['sonar_orientation']['roll']),
        description='Sonar roll angle in degrees'
    )
    
    declare_sonar_pitch_cmd = DeclareLaunchArgument(
        'sonar_orientation.pitch',
        default_value=str(yaml_params['sonar_orientation']['pitch']),
        description='Sonar pitch angle in degrees'
    )
    
    declare_sonar_yaw_cmd = DeclareLaunchArgument(
        'sonar_orientation.yaw',
        default_value=str(yaml_params['sonar_orientation']['yaw']),
        description='Sonar yaw angle in degrees'
    )
    
    # Fast-LIO launch (with rviz=false to avoid conflict)
    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_pkg, 'launch', 'mapping.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rviz': 'false',  # Disable Fast-LIO's RViz to avoid conflict
            'config_file': 'mid360.yaml'
        }.items(),
        condition=IfCondition(launch_fast_lio)
    )
    
    # 3D Mapper node
    # Parameter priority order:
    # 1. Command line args (--ros-args -p param:=value) - highest priority
    # 2. YAML file (mapper_config)
    # 3. Launch file parameters (defined here)
    # 4. Node defaults in 3d_mapper_node.py
    # 5. Library defaults in 3d_mapper.py - lowest priority
    mapper_node = Node(
        package='sonar_3d_reconstruction',
        executable='3d_mapper_node.py',
        name='sonar_3d_mapper',
        parameters=[
            mapper_config,  # Priority 2: YAML file
            {
                'use_sim_time': use_sim_time,  # Priority 3: Launch parameters
                'sonar_orientation.roll': sonar_roll,
                'sonar_orientation.pitch': sonar_pitch,
                'sonar_orientation.yaw': sonar_yaw
            }
        ],
        output='screen'
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(launch_rviz)
    )
    
    # Bag player process with playback rate
    bag_player = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_file, '--clock', '--rate', bag_playback_rate],
        output='screen',
        condition=IfCondition(play_bag)
    )
    
    # Bag recorder process
    bag_recorder = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-a', '-o', record_output_path],
        output='screen',
        condition=IfCondition(record_bag)
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_launch_fast_lio_cmd)
    ld.add_action(declare_launch_rviz_cmd)
    ld.add_action(declare_play_bag_cmd)
    ld.add_action(declare_bag_file_cmd)
    ld.add_action(declare_bag_playback_rate_cmd)
    ld.add_action(declare_record_bag_cmd)
    ld.add_action(declare_record_output_path_cmd)
    ld.add_action(declare_sonar_roll_cmd)
    ld.add_action(declare_sonar_pitch_cmd)
    ld.add_action(declare_sonar_yaw_cmd)
    
    # Add nodes
    ld.add_action(fast_lio_launch)  # Changed to launch instead of node
    ld.add_action(mapper_node)
    ld.add_action(rviz_node)
    ld.add_action(bag_player)
    ld.add_action(bag_recorder)
    
    return ld