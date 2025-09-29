#!/usr/bin/env python3
"""
ROS2 Node for 3D Sonar Mapping
Subscribes to sonar images and odometry, publishes 3D map as PointCloud2

Author: Sonar 3D Reconstruction Team
Date: 2025
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import time
import struct

# ROS2 message imports
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import StaticTransformBroadcaster

# Message filters for time synchronization
from message_filters import Subscriber, ApproximateTimeSynchronizer

# OpenCV for image processing
from cv_bridge import CvBridge

# Import core mapping class (using import with underscores since Python doesn't allow starting with numbers)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# We'll rename the file or import it differently
import importlib.util
spec = importlib.util.spec_from_file_location("mapper_3d", 
    os.path.join(os.path.dirname(__file__), "3d_mapper.py"))
mapper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper_module)
SonarTo3DMapper = mapper_module.SonarTo3DMapper


class SonarMapperNode(Node):
    """ROS2 node for 3D sonar mapping with probabilistic octree"""
    
    def __init__(self):
        super().__init__('sonar_3d_mapper')
        
        # Declare parameters with default values (lowest priority - priority 4)
        # These defaults will be overridden by YAML, launch file, or command line args
        self.declare_parameters(
            namespace='',
            parameters=[
                # Sonar parameters
                ('horizontal_fov', 130.0),
                ('vertical_aperture', 20.0),
                ('max_range', 10.0),
                ('min_range', 0.5),
                ('intensity_threshold', 35),
                
                # Sonar mounting (relative to base_link)
                ('sonar_position.x', 0.0),
                ('sonar_position.y', 0.0),
                ('sonar_position.z', -0.5),
                ('sonar_orientation.roll', 0.0),  # degrees
                ('sonar_orientation.pitch', 90.0),  # degrees (90 = pointing down)
                ('sonar_orientation.yaw', 0.0),  # degrees
                
                # Octree parameters
                ('voxel_resolution', 0.05),
                ('min_probability', 0.6),
                ('dynamic_expansion', True),
                
                # Adaptive update parameters
                ('adaptive_update', True),
                ('adaptive_threshold', 0.5),
                ('adaptive_max_ratio', 0.3),
                
                # Log-odds parameters
                ('log_odds_occupied', 1.5),
                ('log_odds_free', -2.0),
                ('log_odds_min', -10.0),
                ('log_odds_max', 10.0),
                
                # Publishing parameters
                ('show_free_space', False),
                
                # Frame IDs
                ('sonar_frame_id', 'sonar_link'),
                ('base_frame_id', 'base_link'),
                ('map_frame_id', 'map'),
                ('publish_tf', True),
                
                # Topics
                ('sonar_topic', '/sensor/sonar/oculus/m750d/image'),
                ('odometry_topic', '/fast_lio/odometry'),
                ('pointcloud_topic', '/sonar_3d_map'),
                ('marker_topic', '/sonar_3d_map_markers')
            ]
        )
        
        # Get parameters and create config dictionary
        # Parameter priority order (highest to lowest):
        # 1. Command line args (ros2 run ... --ros-args -p param:=value)
        # 2. YAML file (specified in launch or --params-file)
        # 3. Launch file parameters
        # 4. Node defaults (declared above)
        # 5. 3d_mapper.py defaults (will be overridden by config dict)
        
        # ROS2 automatically handles priority 1-4, we just get the final values
        config = {
            'horizontal_fov': self.get_parameter('horizontal_fov').value,
            'vertical_aperture': self.get_parameter('vertical_aperture').value,
            'max_range': self.get_parameter('max_range').value,
            'min_range': self.get_parameter('min_range').value,
            'intensity_threshold': self.get_parameter('intensity_threshold').value,
            'sonar_position': [
                self.get_parameter('sonar_position.x').value,
                self.get_parameter('sonar_position.y').value,
                self.get_parameter('sonar_position.z').value
            ],
            'sonar_orientation': [
                np.radians(self.get_parameter('sonar_orientation.roll').value),  # Convert degrees to radians
                np.radians(self.get_parameter('sonar_orientation.pitch').value),  # Convert degrees to radians
                np.radians(self.get_parameter('sonar_orientation.yaw').value)  # Convert degrees to radians
            ],
            'voxel_resolution': self.get_parameter('voxel_resolution').value,
            'min_probability': self.get_parameter('min_probability').value,
            'dynamic_expansion': self.get_parameter('dynamic_expansion').value,
            'adaptive_update': self.get_parameter('adaptive_update').value,
            'adaptive_threshold': self.get_parameter('adaptive_threshold').value,
            'adaptive_max_ratio': self.get_parameter('adaptive_max_ratio').value,
            'log_odds_occupied': self.get_parameter('log_odds_occupied').value,
            'log_odds_free': self.get_parameter('log_odds_free').value,
            'log_odds_min': self.get_parameter('log_odds_min').value,
            'log_odds_max': self.get_parameter('log_odds_max').value
        }
        
        # Get other parameters
        self.show_free_space = self.get_parameter('show_free_space').value
        self.sonar_frame_id = self.get_parameter('sonar_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.publish_tf = self.get_parameter('publish_tf').value
        
        # Get topic names
        sonar_topic = self.get_parameter('sonar_topic').value
        odometry_topic = self.get_parameter('odometry_topic').value
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        marker_topic = self.get_parameter('marker_topic').value
        
        # Initialize mapper
        self.mapper = SonarTo3DMapper(config)
        
        # Create CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize static TF broadcaster if enabled
        if self.publish_tf:
            from tf2_ros import StaticTransformBroadcaster
            self.tf_static_broadcaster = StaticTransformBroadcaster(self)
            self.sonar_position = config['sonar_position']
            self.sonar_orientation = config['sonar_orientation']
            # Publish static transform once
            self.publish_static_tf()
        
        # Initialize latest_odometry to None
        self.latest_odometry = None
        
        # Frame counter
        self.frame_count = 0
        self.last_publish_time = time.time()
        
        # QoS profile for best effort subscription
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create synchronized subscribers using message_filters
        self.sonar_sub = Subscriber(
            self,
            Image,
            sonar_topic,
            qos_profile=qos_profile
        )
        
        self.odom_sub = Subscriber(
            self,
            Odometry,
            odometry_topic,
            qos_profile=qos_profile
        )
        
        # Create time synchronizer with 0.1 second tolerance
        self.time_sync = ApproximateTimeSynchronizer(
            [self.sonar_sub, self.odom_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance for time synchronization
        )
        self.time_sync.registerCallback(self.synchronized_callback)
        
        # Create publishers
        self.pc_pub = self.create_publisher(
            PointCloud2,
            pointcloud_topic,
            10
        )
        
        self.marker_pub = self.create_publisher(
            MarkerArray,
            marker_topic,
            10
        )
        
        # Create timer for periodic publishing (fixed at 10Hz)
        self.timer = self.create_timer(
            0.1,  # 10Hz publishing rate
            self.publish_pointcloud
        )
        
        # No need for TF timer anymore since we use static transform
        
        self.get_logger().info('3D Sonar Mapper Node initialized')
        self.get_logger().info(f'  Horizontal FOV: {config["horizontal_fov"]}°')
        self.get_logger().info(f'  Vertical aperture: {config["vertical_aperture"]}°')
        self.get_logger().info(f'  Voxel resolution: {config["voxel_resolution"]}m')
        self.get_logger().info(f'  Min probability: {config["min_probability"]}')
        self.get_logger().info(f'  Adaptive update: {config["adaptive_update"]}')
        self.get_logger().info(f'  Sonar orientation (deg): roll={self.get_parameter("sonar_orientation.roll").value}, '
                              f'pitch={self.get_parameter("sonar_orientation.pitch").value}, '
                              f'yaw={self.get_parameter("sonar_orientation.yaw").value}')
        self.get_logger().info(f'  Subscribing to sonar: {sonar_topic}')
        self.get_logger().info(f'  Subscribing to odometry: {odometry_topic}')
        self.get_logger().info(f'  Publishing to: {pointcloud_topic}')
        self.get_logger().info(f'  Time synchronization enabled with {0.1}s tolerance')
    
    def synchronized_callback(self, sonar_msg: Image, odom_msg: Odometry):
        """
        Process synchronized sonar image and odometry data
        
        Args:
            sonar_msg: Sonar image message
            odom_msg: Odometry message
        """
        # Convert ROS Image to numpy array
        try:
            # Handle different encodings
            if sonar_msg.encoding == 'mono8' or sonar_msg.encoding == '8UC1':
                sonar_image = self.bridge.imgmsg_to_cv2(sonar_msg, desired_encoding='mono8')
            elif sonar_msg.encoding == 'mono16' or sonar_msg.encoding == '16UC1':
                sonar_image = self.bridge.imgmsg_to_cv2(sonar_msg, desired_encoding='mono16')
                # Convert to 8-bit
                sonar_image = (sonar_image / 256).astype(np.uint8)
            else:
                self.get_logger().error(f'Unsupported image encoding: {sonar_msg.encoding}')
                return
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        # Extract odometry position and orientation
        position = [
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ]
        
        orientation = [
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        ]
        
        # Process the sonar image
        stats = self.mapper.process_sonar_image(sonar_image, position, orientation)
        
        self.frame_count += 1
        
        # Store latest odometry for TF publishing
        self.latest_odometry = odom_msg
        
        # Log statistics periodically
        if not stats.get('skipped', False) and self.frame_count % 10 == 0:
            # Calculate time difference for debug
            sonar_time = sonar_msg.header.stamp.sec + sonar_msg.header.stamp.nanosec * 1e-9
            odom_time = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
            time_diff = abs(sonar_time - odom_time)
            
            self.get_logger().info(
                f'Frame {self.frame_count}: '
                f'{stats["num_occupied"]} occupied, {stats["num_free"]} free, '
                f'{stats["num_voxels"]} total voxels, '
                f'time_diff={time_diff:.3f}s, '
                f'proc_time={stats["processing_time"]:.3f}s'
            )
    
    def publish_static_tf(self):
        """Publish static TF transform from base_link to sonar_link"""
        if not self.publish_tf:
            return
        
        # Create static transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_frame_id
        t.child_frame_id = self.sonar_frame_id
        
        # Set translation
        t.transform.translation.x = self.sonar_position[0]
        t.transform.translation.y = self.sonar_position[1]
        t.transform.translation.z = self.sonar_position[2]
        
        # Convert RPY to quaternion
        roll, pitch, yaw = self.sonar_orientation
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy
        
        # Send static transform
        self.tf_static_broadcaster.sendTransform(t)
        self.get_logger().info(f'Published static TF: {self.base_frame_id} -> {self.sonar_frame_id}')
    
    def publish_pointcloud(self):
        """Publish accumulated point cloud"""
        # Get point cloud from mapper
        result = self.mapper.get_point_cloud(include_free=self.show_free_space)
        
        if self.show_free_space:
            # Publish as marker array with colors
            self.publish_marker_array(result)
        else:
            # Publish as PointCloud2
            if result['num_occupied'] > 0:
                self.publish_pointcloud2(result['points'], result['probabilities'])
    
    def publish_pointcloud2(self, points: np.ndarray, probabilities: np.ndarray):
        """
        Publish PointCloud2 message with intensity as probability
        
        Args:
            points: Nx3 array of points
            probabilities: N array of probabilities
        """
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame_id
        
        # Create PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        
        # Pack data
        data = []
        for i in range(len(points)):
            data.append(struct.pack('ffff',
                                   points[i, 0], points[i, 1], points[i, 2],
                                   probabilities[i]))
        
        cloud.data = b''.join(data)
        
        # Publish
        self.pc_pub.publish(cloud)
    
    def publish_marker_array(self, result: dict):
        """
        Publish MarkerArray with colored voxels
        
        Args:
            result: Dictionary with classified voxels
        """
        marker_array = MarkerArray()
        marker_id = 0
        
        # Create marker for occupied voxels (red)
        if len(result['occupied']) > 0:
            marker = Marker()
            marker.header.frame_id = self.map_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker_id
            marker.type = Marker.CUBE_LIST
            marker.action = Marker.ADD
            marker.scale.x = self.mapper.voxel_resolution
            marker.scale.y = self.mapper.voxel_resolution
            marker.scale.z = self.mapper.voxel_resolution
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            for point, prob in result['occupied']:
                p = marker.points.add()
                p.x, p.y, p.z = point
            
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Create marker for free voxels (blue) if enabled
        if self.show_free_space and len(result['free']) > 0:
            marker = Marker()
            marker.header.frame_id = self.map_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker_id
            marker.type = Marker.CUBE_LIST
            marker.action = Marker.ADD
            marker.scale.x = self.mapper.voxel_resolution
            marker.scale.y = self.mapper.voxel_resolution
            marker.scale.z = self.mapper.voxel_resolution
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.3
            
            for point, prob in result['free']:
                p = marker.points.add()
                p.x, p.y, p.z = point
            
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Create marker for unknown voxels (yellow)
        if len(result.get('unknown', [])) > 0:
            marker = Marker()
            marker.header.frame_id = self.map_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker_id
            marker.type = Marker.CUBE_LIST
            marker.action = Marker.ADD
            marker.scale.x = self.mapper.voxel_resolution
            marker.scale.y = self.mapper.voxel_resolution
            marker.scale.z = self.mapper.voxel_resolution
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            
            for point, prob in result['unknown']:
                p = marker.points.add()
                p.x, p.y, p.z = point
            
            marker_array.markers.append(marker)
        
        # Publish marker array
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    
    node = SonarMapperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Print final statistics
        result = node.mapper.get_point_cloud()
        node.get_logger().info(
            f'\nFinal statistics:\n'
            f'  Total frames: {result["frame_count"]}\n'
            f'  Processed frames: {result["processed_count"]}\n'
            f'  Total voxels: {result["num_voxels"]}\n'
            f'  Occupied voxels: {result["num_occupied"]}'
        )
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()