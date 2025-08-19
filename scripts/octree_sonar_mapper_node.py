#!/usr/bin/env python3
"""
ROS2 Node for Octree-based Sonar Mapping
Subscribes to sonar images and odometry, publishes 3D map as PointCloud2

This node uses the octree_mapper library for the core mapping functionality.

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
from tf2_ros import TransformBroadcaster

# OpenCV for image processing
from cv_bridge import CvBridge

# Scipy for rotation conversions
from scipy.spatial.transform import Rotation

# Import core mapping classes from octree_mapper module
from octree_mapper import SimpleOctree, OctreeBasedSonarMapperWithBoat


class OctreeSonarMapperNode(Node):
    """ROS2 node for octree-based sonar mapping"""
    
    def __init__(self):
        super().__init__('octree_sonar_mapper')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                # Octree parameters
                ('resolution', 0.03),
                ('dynamic_expansion', True),
                ('prob_hit', 0.7),
                ('prob_miss', 0.3),
                
                # Sonar parameters
                ('max_range', 5.0),
                ('min_range', 0.5),
                ('fov_degrees', 130.0),
                ('fov_margin_degrees', 15.0),
                ('vertical_aperture_degrees', 20.0),
                ('intensity_threshold', 35),
                
                # Sonar-to-boat transform parameters
                ('sonar_to_boat_transform.position.x', 0.0),
                ('sonar_to_boat_transform.position.y', 0.0),
                ('sonar_to_boat_transform.position.z', -0.5),
                ('sonar_to_boat_transform.orientation.roll', 0.0),
                ('sonar_to_boat_transform.orientation.pitch', 90.0),
                ('sonar_to_boat_transform.orientation.yaw', 0.0),
                
                # Visualization parameters
                ('publish_rate', 10.0),
                ('min_probability_threshold', 0.6),
                
                # Frame IDs
                ('sonar_frame_id', 'sonar_link'),
                ('body_frame_id', 'body'),
                ('map_frame_id', 'camera_init'),
                ('publish_tf', True),
                
                # Topics
                ('sonar_topic', '/oculus/sonar_image_raw'),
                ('odometry_topic', '/Odometry'),
                ('pointcloud_topic', '/sonar_3d_map')
            ]
        )
        
        # Get parameters
        self.resolution = self.get_parameter('resolution').value
        self.dynamic_expansion = self.get_parameter('dynamic_expansion').value
        prob_hit = self.get_parameter('prob_hit').value
        prob_miss = self.get_parameter('prob_miss').value
        
        # Convert probabilities to log-odds
        self.log_odds_occupied = np.log(prob_hit / (1.0 - prob_hit))
        self.log_odds_free = np.log(prob_miss / (1.0 - prob_miss))
        self.max_range = self.get_parameter('max_range').value
        self.min_range = self.get_parameter('min_range').value
        self.fov_degrees = self.get_parameter('fov_degrees').value
        self.fov_margin_degrees = self.get_parameter('fov_margin_degrees').value
        self.vertical_aperture_degrees = self.get_parameter('vertical_aperture_degrees').value
        self.intensity_threshold = self.get_parameter('intensity_threshold').value
        
        # Get transform parameters
        sonar_position = (
            self.get_parameter('sonar_to_boat_transform.position.x').value,
            self.get_parameter('sonar_to_boat_transform.position.y').value,
            self.get_parameter('sonar_to_boat_transform.position.z').value
        )
        
        # IMPORTANT: sensor_pose expects (heading, tilt, roll) = (yaw, pitch, roll)
        sonar_orientation = (
            self.get_parameter('sonar_to_boat_transform.orientation.yaw').value,
            self.get_parameter('sonar_to_boat_transform.orientation.pitch').value,
            self.get_parameter('sonar_to_boat_transform.orientation.roll').value
        )
        
        
        # Get visualization parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.min_probability_threshold = self.get_parameter('min_probability_threshold').value
        
        # Get frame IDs
        self.sonar_frame_id = self.get_parameter('sonar_frame_id').value
        self.body_frame_id = self.get_parameter('body_frame_id').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.publish_tf = self.get_parameter('publish_tf').value
        
        # Get topic names
        sonar_topic = self.get_parameter('sonar_topic').value
        odometry_topic = self.get_parameter('odometry_topic').value
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        
        # Initialize mapper using the library
        self.mapper = OctreeBasedSonarMapperWithBoat(
            resolution=self.resolution,
            sensor_position=sonar_position,
            sensor_pose=sonar_orientation,
            max_range=self.max_range,
            fov_degrees=self.fov_degrees,
            vertical_aperture_degrees=self.vertical_aperture_degrees,
            intensity_threshold=self.intensity_threshold,
            min_range=self.min_range,
            dynamic_expansion=self.dynamic_expansion,
            fov_margin_degrees=self.fov_margin_degrees,
            log_odds_occupied=self.log_odds_occupied,
            log_odds_free=self.log_odds_free
        )
        
        # Create CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize TF broadcaster if enabled
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
            self.sonar_position = sonar_position
            self.sonar_orientation = sonar_orientation  # Store full orientation (yaw, pitch, roll)
            self.get_logger().info(f'  Publishing TF: {self.body_frame_id} -> {self.sonar_frame_id}')
        
        # Latest odometry data
        self.latest_odometry = None
        
        # Frame counter
        self.frame_count = 0
        self.last_publish_time = time.time()
        
        # QoS profile for best effort subscription (for bag playback)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscribers
        self.sonar_sub = self.create_subscription(
            Image,
            sonar_topic,
            self.sonar_callback,
            qos_profile
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            odometry_topic,
            self.odometry_callback,
            qos_profile
        )
        
        # Create publisher for PointCloud2
        self.pc_pub = self.create_publisher(
            PointCloud2,
            pointcloud_topic,
            10
        )
        
        # Create timer for periodic publishing
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_pointcloud
        )
        
        # Create separate timer for TF publishing if enabled
        if self.publish_tf:
            self.tf_timer = self.create_timer(
                0.1,  # Publish TF at 10Hz
                self.publish_tf_callback
            )
            self.get_logger().info('TF publishing timer created')
        
        self.get_logger().info(f'Octree Sonar Mapper Node initialized')
        self.get_logger().info(f'  Resolution: {self.resolution}m')
        self.get_logger().info(f'  Sonar FOV: {self.fov_degrees}° (margin: {self.fov_margin_degrees}°)')
        self.get_logger().info(f'  Sonar position: {sonar_position}')
        self.get_logger().info(f'  Sonar orientation (Y,P,R): {sonar_orientation}°')
        self.get_logger().info(f'  Update probabilities: hit={prob_hit}, miss={prob_miss}')
        self.get_logger().info(f'  Using process_polar_image_voxel_centric() method')
        self.get_logger().info(f'  Frame IDs: {self.sonar_frame_id} -> {self.body_frame_id} -> {self.map_frame_id}')
        self.get_logger().info(f'  Subscribing to: {sonar_topic}')
        self.get_logger().info(f'  Subscribing to: {odometry_topic}')
        self.get_logger().info(f'  Publishing to: {pointcloud_topic}')
    
    def sonar_callback(self, msg: Image):
        """Process incoming sonar image"""
        # Check if we have odometry data
        if self.latest_odometry is None:
            if self.frame_count == 0:
                self.get_logger().warn('No odometry data received yet, waiting...')
            return
        
        # Convert ROS Image to numpy array
        try:
            # Handle different encodings
            if msg.encoding == 'mono8' or msg.encoding == '8UC1':
                sonar_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            elif msg.encoding == 'mono16' or msg.encoding == '16UC1':
                sonar_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
                # Convert to 8-bit
                sonar_image = (sonar_image / 256).astype(np.uint8)
            else:
                self.get_logger().error(f'Unsupported image encoding: {msg.encoding}')
                return
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        # Process the sonar image using voxel-centric method (same as demo)
        stats = self.mapper.process_polar_image_voxel_centric(sonar_image)
        
        self.frame_count += 1
        
        # Log processing statistics every 10 frames
        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f'Frame {self.frame_count}: '
                f'{stats["rays_processed"]} rays, '
                f'{stats["occupied_updates"]} occupied, '
                f'{stats["free_updates"]} free, '
                f'{len(self.mapper.octree.voxels)} total voxels, '
                f'{stats["processing_time"]:.3f}s'
            )
    
    def odometry_callback(self, msg: Odometry):
        """Update boat pose from odometry"""
        # Extract position
        position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )
        
        # Extract quaternion
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        
        # Update mapper with new boat pose
        self.mapper.set_boat_pose_from_odometry(position, quaternion)
        
        # Store latest odometry
        self.latest_odometry = msg
    
    def publish_pointcloud(self):
        """Publish the current map as PointCloud2"""
        # Check if we have any data to publish
        if len(self.mapper.octree.voxels) == 0:
            return
        
        # Get voxels with probability for intensity visualization
        # Filter out free space (probability < threshold)
        points, probabilities = self.mapper.get_classified_voxels_with_probability(
            min_probability=self.min_probability_threshold
        )
        
        if len(points) == 0:
            return
        
        # Subsample if too many points
        if len(points) > 50000:
            indices = np.random.choice(len(points), 50000, replace=False)
            points = points[indices]
            probabilities = probabilities[indices]
        
        # TF is now published by separate timer
        
        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame_id  # Use configurable frame ID
        
        # Create PointCloud2 message with intensity (probability)
        pc_msg = self.create_pointcloud2_msg_with_intensity(header, points, probabilities)
        
        # Publish
        self.pc_pub.publish(pc_msg)
        
        # Log statistics periodically
        current_time = time.time()
        if current_time - self.last_publish_time > 5.0:  # Every 5 seconds
            total_voxels = len(self.mapper.octree.voxels)
            self.get_logger().info(
                f'Map statistics: {total_voxels} total voxels'
            )
            self.last_publish_time = current_time
    
    def create_pointcloud2_msg_with_intensity(self, header, points, intensities):
        """Create a PointCloud2 message with XYZ and intensity (probability)"""
        # Define the point fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Create the PointCloud2 message
        pc_msg = PointCloud2()
        pc_msg.header = header
        pc_msg.height = 1
        pc_msg.width = len(points)
        pc_msg.is_dense = True
        pc_msg.is_bigendian = False
        pc_msg.fields = fields
        pc_msg.point_step = 16  # 4 bytes * 4 fields
        pc_msg.row_step = pc_msg.point_step * pc_msg.width
        
        # Pack the data
        buffer = []
        for i in range(len(points)):
            x, y, z = points[i]
            intensity = intensities[i]  # probability value (0.0 to 1.0)
            
            # Pack the point with intensity
            buffer.append(struct.pack('ffff', x, y, z, intensity))
        
        pc_msg.data = b''.join(buffer)
        
        return pc_msg
    
    def create_sonar_to_boat_rotation_matrix(self, yaw_deg, pitch_deg, roll_deg):
        """
        Create rotation matrix from sonar to boat coordinates.
        This mirrors the logic in octree_mapper.py for consistency.
        
        Args:
            yaw_deg: Rotation around Z axis (heading)
            pitch_deg: Rotation around Y axis (tilt)
            roll_deg: Rotation around X axis
        
        Returns:
            3x3 rotation matrix
        """
        # Special case for downward-looking sonar (pitch = -90)
        if abs(pitch_deg + 90) < 0.001:
            # Direct transformation for downward-looking sonar
            # This achieves xb=ys, yb=-zs, zb=-xs
            R = np.array([
                [0,  1,  0],   # xb = ys
                [0,  0, -1],   # yb = -zs
                [-1, 0,  0]    # zb = -xs
            ])
        else:
            # General case: apply rotations in order (Z-Y-X convention)
            # Convert to radians
            yaw_rad = np.radians(yaw_deg)
            pitch_rad = np.radians(pitch_deg)
            roll_rad = np.radians(roll_deg)
            
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
            
            # Combined rotation: R = Rz * Ry * Rx
            R = Rz @ Ry @ Rx
        
        return R
    
    def publish_sonar_transform(self):
        """Publish TF transform from body to sonar_link frame"""
        t = TransformStamped()
        
        # Use odometry timestamp for time synchronization
        if self.latest_odometry is not None:
            t.header.stamp = self.latest_odometry.header.stamp
        else:
            t.header.stamp = self.get_clock().now().to_msg()
        
        t.header.frame_id = self.body_frame_id  # parent: body
        t.child_frame_id = self.sonar_frame_id  # child: sonar_link
        
        # Set translation (sonar position relative to body)
        t.transform.translation.x = self.sonar_position[0]
        t.transform.translation.y = self.sonar_position[1]
        t.transform.translation.z = self.sonar_position[2]
        
        # Get rotation matrix using the same logic as octree_mapper.py
        yaw_deg, pitch_deg, roll_deg = self.sonar_orientation
        rotation_matrix = self.create_sonar_to_boat_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
        
        # Convert rotation matrix to quaternion using scipy
        rotation = Rotation.from_matrix(rotation_matrix)
        quat = rotation.as_quat()  # Returns [x, y, z, w]
        
        # Set rotation quaternion
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)
    
    def publish_tf_callback(self):
        """Callback to publish TF transform periodically"""
        # Only publish if we have received odometry (body frame exists)
        if self.latest_odometry is not None:
            self.publish_sonar_transform()


def main(args=None):
    rclpy.init(args=args)
    node = OctreeSonarMapperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()