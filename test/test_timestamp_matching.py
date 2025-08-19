#!/usr/bin/env python3
"""Test timestamp matching between sonar and odometry data."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bag_processor import BagProcessor
import csv

# Load odometry data
odometry_data = []
with open('full_odometry_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamp = float(row['timestamp_sec']) + float(row['timestamp_nanosec']) * 1e-9
        odometry_data.append({
            'timestamp': timestamp,
            'position': [float(row['x']), float(row['y']), float(row['z'])]
        })

print(f"Loaded {len(odometry_data)} odometry measurements")
print(f"Odometry time range: {odometry_data[0]['timestamp']:.3f} - {odometry_data[-1]['timestamp']:.3f}s")
print(f"Odometry duration: {odometry_data[-1]['timestamp'] - odometry_data[0]['timestamp']:.3f}s")

# Load sonar timestamps
bag_path = "/workspace/data/3_janggil_ri/20250801_blueboat_sonar_lidar/sonar-scenario1-v1.deg90.bag"
processor = BagProcessor(bag_path)

# Get first few sonar timestamps
sonar_timestamps = []
for i in range(3):
    try:
        frame, metadata = processor.sonar_extractor.extract_oculus_frame(i)
        if frame is not None:
            sonar_timestamps.append(metadata.get('timestamp_ns', 0))
            print(f"\nSonar frame {i}:")
            print(f"  Timestamp: {metadata.get('timestamp_ns', 0) * 1e-9:.3f}s")
    except:
        break

if sonar_timestamps:
    print(f"\nSonar time range (first 3): {sonar_timestamps[0] * 1e-9:.3f} - {sonar_timestamps[-1] * 1e-9:.3f}s")
    print(f"Sonar interval: {(sonar_timestamps[-1] - sonar_timestamps[0]) * 1e-9:.3f}s")
    
    # Calculate relative times
    print("\n--- Relative Time Matching ---")
    sonar_start = sonar_timestamps[0]
    odom_start = odometry_data[0]['timestamp']
    
    for i, sonar_ts in enumerate(sonar_timestamps):
        sonar_relative = (sonar_ts - sonar_start) * 1e-9
        
        # Find closest odometry
        best_idx = 0
        best_diff = float('inf')
        for j, odom in enumerate(odometry_data):
            odom_relative = odom['timestamp'] - odom_start
            diff = abs(odom_relative - sonar_relative)
            if diff < best_diff:
                best_diff = diff
                best_idx = j
        
        print(f"\nSonar frame {i} (relative: {sonar_relative:.3f}s)")
        print(f"  Closest odometry: index {best_idx} (relative: {odometry_data[best_idx]['timestamp'] - odom_start:.3f}s)")
        print(f"  Time difference: {best_diff:.3f}s")
        print(f"  Position: {odometry_data[best_idx]['position']}")

processor.close()