#!/usr/bin/env python3
"""
ROS2 Bag Merger
Merges sonar and lidar bag files into a single synchronized bag file.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import argparse

# ROS2 imports
try:
    from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions
    from rosbag2_py import TopicMetadata
    from rclpy.serialization import deserialize_message, serialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f"❌ ROS2 import error: {e}")
    print("Make sure to run: source /opt/ros/humble/setup.bash")
    sys.exit(1)


def merge_bags(bag1_path: str, bag2_path: str, output_path: str, verbose: bool = True):
    """
    Merge two ROS2 bag files into a single bag file.
    
    Args:
        bag1_path: Path to first bag file
        bag2_path: Path to second bag file  
        output_path: Path for merged output bag
        verbose: Print progress messages
    """
    if verbose:
        print(f"🔄 Merging bag files:")
        print(f"   📁 Bag 1: {bag1_path}")
        print(f"   📁 Bag 2: {bag2_path}")
        print(f"   📁 Output: {output_path}")
        print()
    
    # Create readers for both bags
    reader1 = SequentialReader()
    reader2 = SequentialReader()
    
    storage_options1 = StorageOptions(uri=bag1_path, storage_id='sqlite3')
    storage_options2 = StorageOptions(uri=bag2_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader1.open(storage_options1, converter_options)
    reader2.open(storage_options2, converter_options)
    
    # Get all topics from both bags
    topics1 = reader1.get_all_topics_and_types()
    topics2 = reader2.get_all_topics_and_types()
    
    if verbose:
        print(f"📊 Bag 1 topics: {len(topics1)}")
        for topic in topics1:
            print(f"   - {topic.name}: {topic.type}")
        print()
        
        print(f"📊 Bag 2 topics: {len(topics2)}")
        for topic in topics2:
            print(f"   - {topic.name}: {topic.type}")
        print()
    
    # Create writer for merged bag
    writer = SequentialWriter()
    storage_options_out = StorageOptions(uri=output_path, storage_id='sqlite3')
    writer.open(storage_options_out, converter_options)
    
    # Register all topics in the writer
    all_topics = {}
    
    # Add topics from bag1
    for topic_info in topics1:
        if topic_info.name not in all_topics:
            all_topics[topic_info.name] = topic_info
            topic_meta = TopicMetadata(
                name=topic_info.name,
                type=topic_info.type,
                serialization_format='cdr'
            )
            writer.create_topic(topic_meta)
    
    # Add topics from bag2
    for topic_info in topics2:
        if topic_info.name not in all_topics:
            all_topics[topic_info.name] = topic_info
            topic_meta = TopicMetadata(
                name=topic_info.name,
                type=topic_info.type,
                serialization_format='cdr'
            )
            writer.create_topic(topic_meta)
    
    if verbose:
        print(f"✅ Registered {len(all_topics)} unique topics in merged bag")
        print()
    
    # Read all messages from both bags
    messages = []
    
    # Read from bag1
    if verbose:
        print("📖 Reading messages from Bag 1...")
    msg_count1 = 0
    while reader1.has_next():
        topic, data, timestamp = reader1.read_next()
        messages.append((timestamp, topic, data))
        msg_count1 += 1
    
    # Read from bag2
    if verbose:
        print(f"   ✅ Read {msg_count1} messages from Bag 1")
        print("📖 Reading messages from Bag 2...")
    msg_count2 = 0
    while reader2.has_next():
        topic, data, timestamp = reader2.read_next()
        messages.append((timestamp, topic, data))
        msg_count2 += 1
    
    if verbose:
        print(f"   ✅ Read {msg_count2} messages from Bag 2")
        print(f"   📊 Total messages: {len(messages)}")
        print()
    
    # Sort messages by timestamp
    if verbose:
        print("🔄 Sorting messages by timestamp...")
    messages.sort(key=lambda x: x[0])
    
    # Write sorted messages to output bag
    if verbose:
        print("💾 Writing merged bag...")
    for timestamp, topic, data in messages:
        writer.write(topic, data, timestamp)
    
    # Clean up
    del writer
    del reader1
    del reader2
    
    if verbose:
        print(f"✅ Successfully merged {len(messages)} messages into {output_path}")
        print()
        
        # Print time range
        if messages:
            start_time = messages[0][0] / 1e9  # Convert nanoseconds to seconds
            end_time = messages[-1][0] / 1e9
            duration = end_time - start_time
            print(f"📊 Merged bag statistics:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Start time: {start_time:.6f}")
            print(f"   End time: {end_time:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Merge two ROS2 bag files')
    parser.add_argument('--bag1', type=str, 
                       default='/workspace/data/3_janggil_ri/20250801_blueboat_sonar_lidar/sonar-scenario1-v1.deg90.bag',
                       help='Path to first bag file')
    parser.add_argument('--bag2', type=str,
                       default='/workspace/data/3_janggil_ri/20250801_blueboat_sonar_lidar/lidar-scenario1-v1.2.deg90.bag',
                       help='Path to second bag file')
    parser.add_argument('--output', type=str,
                       default='/workspace/data/3_janggil_ri/20250801_blueboat_sonar_lidar/merged-scenario1-v1.deg90.bag',
                       help='Path for merged output bag')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Check if input bags exist
    if not Path(args.bag1).exists():
        print(f"❌ Bag 1 not found: {args.bag1}")
        sys.exit(1)
    
    if not Path(args.bag2).exists():
        print(f"❌ Bag 2 not found: {args.bag2}")
        sys.exit(1)
    
    # Check if output already exists
    if Path(args.output).exists():
        response = input(f"⚠️  Output file already exists: {args.output}\n   Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        # Remove existing directory
        import shutil
        shutil.rmtree(args.output)
    
    # Merge the bags
    try:
        merge_bags(args.bag1, args.bag2, args.output, verbose=not args.quiet)
    except Exception as e:
        print(f"❌ Error merging bags: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # Source ROS2
    os.system('source /opt/ros/humble/setup.bash 2>/dev/null')
    main()