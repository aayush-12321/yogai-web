from django.test import TestCase
from yoga_backend.warrior2_pose_service import detect_is_left_facing

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class Warrior2OrientationTest(TestCase):
    def test_left_facing_detection(self):
        # Create a list of 33 mock landmarks
        landmarks = [MockLandmark(0.0, 0.0) for _ in range(33)]
        
        # Test Case 1: Left knee bent (left knee angle is smaller than right knee angle)
        # Left leg: hip=23, knee=25, ankle=27
        # Right leg: hip=24, knee=26, ankle=28
        
        # Left leg bent at 90 degrees:
        # Hip at (0.5, 0.5), Knee at (0.5, 0.7), Ankle at (0.7, 0.7)
        landmarks[23] = MockLandmark(0.5, 0.5)
        landmarks[25] = MockLandmark(0.5, 0.7)
        landmarks[27] = MockLandmark(0.7, 0.7)
        
        # Right leg straight (180 degrees):
        # Hip at (0.5, 0.5), Knee at (0.3, 0.7), Ankle at (0.1, 0.9)
        landmarks[24] = MockLandmark(0.5, 0.5)
        landmarks[26] = MockLandmark(0.3, 0.7)
        landmarks[28] = MockLandmark(0.1, 0.9)
        
        # Since left knee is bent (~90) and right knee is straight (~180),
        # left_knee_angle should be smaller than right_knee_angle, returning True.
        self.assertTrue(detect_is_left_facing(landmarks))

    def test_right_facing_detection(self):
        landmarks = [MockLandmark(0.0, 0.0) for _ in range(33)]
        
        # Test Case 2: Right knee bent (right knee angle is smaller than left knee angle)
        # Left leg straight:
        landmarks[23] = MockLandmark(0.5, 0.5)
        landmarks[25] = MockLandmark(0.7, 0.7)
        landmarks[27] = MockLandmark(0.9, 0.9)
        
        # Right leg bent at 90 degrees:
        landmarks[24] = MockLandmark(0.5, 0.5)
        landmarks[26] = MockLandmark(0.5, 0.7)
        landmarks[28] = MockLandmark(0.3, 0.7)
        
        # Since right knee is bent (~90) and left knee is straight (~180),
        # right_knee_angle should be smaller than left_knee_angle, returning False.
        self.assertFalse(detect_is_left_facing(landmarks))
