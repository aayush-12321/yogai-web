# Yoga Pose Detection Models

This directory should contain the trained scikit-learn models for each yoga pose.

## Required Models

Place the following model files in this directory:

- `plank_model.pkl` - Logistic regression model trained to detect plank pose
- `mountain_model.pkl` - Logistic regression model trained to detect mountain pose
- `warrior2_model.pkl` - Logistic regression model trained to detect warrior II pose

## Model Specifications

Each model should be trained on pose-specific landmark features:

### Plank Pose Landmarks (17 landmarks × 4 values = 68 features):
- Selected landmarks: [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
- Names: NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX
- For each landmark: [x, y, z, visibility]
- Input shape: (68,)

### Tree Pose Landmarks (19 landmarks × 4 values = 76 features):
- Selected landmarks: [0, 7, 8, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30]
- Names: NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_INDEX, RIGHT_INDEX, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL
- For each landmark: [x, y, z, visibility]
- Input shape: (76,)

### Warrior II Pose Landmarks (17 landmarks × 4 values = 68 features):
- Selected landmarks: [0, 2, 5, 11, 12, 13, 14, 19, 20, 23, 24, 25, 26, 29, 30, 31, 32]
- Names: NOSE, LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_INDEX, RIGHT_INDEX, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX
- For each landmark: [x, y, z, visibility]
- Input shape: (68,)

- Output: Binary classification (pose vs not pose)

## Training Data

Models should be trained using:
- MediaPipe Pose landmarks
- Pose-specific landmark sets as listed above
- Binary classification for each pose type
- Confidence threshold: 75% for correct pose detection

## Demo Mode

If models are not found, the system will run in demo mode with random predictions for testing purposes.