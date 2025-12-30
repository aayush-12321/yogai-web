import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yogaproject.settings')
django.setup()

from yoga_backend.pose_detector import get_detector
import numpy as np

print("Testing pose detector...")

# Test the detector
detector = get_detector()
print('Models loaded:', list(detector.models.keys()))

# Create some dummy keypoints (68 features for plank pose)
dummy_keypoints = np.random.rand(68)
print("Created dummy keypoints with shape:", dummy_keypoints.shape)

# Test prediction
print("Testing prediction for 'plank'...")
result = detector.predict_pose(dummy_keypoints, 'plank')
print('Prediction result:', result)

# Test with unsupported pose
print("Testing prediction for unsupported 'tree'...")
result2 = detector.predict_pose(dummy_keypoints, 'tree')
print('Unsupported pose result:', result2)

print("Test completed.")