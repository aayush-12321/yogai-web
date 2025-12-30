import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yogaproject.settings')
django.setup()

from yoga_backend.pose_detector import get_detector
import numpy as np

print("Testing pose detector predictions...")

# Test the detector
detector = get_detector()
print('Models loaded:', list(detector.models.keys()))

# Test with dummy keypoints for each pose
test_poses = ['plank', 'mountain', 'warrior2']

for pose in test_poses:
    # Get the expected number of features for this pose
    num_landmarks = len(detector.pose_landmarks[pose])
    num_features = num_landmarks * 4  # x, y, z, visibility

    print(f"\nTesting {pose} pose:")
    print(f"  Expected landmarks: {num_landmarks}")
    print(f"  Expected features: {num_features}")

    # Create dummy keypoints
    dummy_keypoints = np.random.rand(num_features)
    print(f"  Created dummy keypoints shape: {dummy_keypoints.shape}")

    # Test prediction
    result = detector.predict_pose(dummy_keypoints, pose)
    print(f"  Prediction result: {result}")