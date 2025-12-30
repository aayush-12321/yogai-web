# Yoga Pose Detection Models

This directory should contain the trained TensorFlow/Keras models for each yoga pose.

## Required Models

Place the following model files in this directory:

- `plank_model.h5` - Model trained to detect plank pose
- `tree_model.h5` - Model trained to detect tree pose
- `warrior2_model.h5` - Model trained to detect warrior II pose

## Model Specifications

Each model should be trained on the following landmark features (17 landmarks × 4 values = 68 features):
- Selected landmarks: [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
- For each landmark: [x, y, z, visibility]
- Input shape: (68,)
- Output: Binary classification (pose vs not pose)

## Training Data

Models should be trained using:
- MediaPipe Pose landmarks
- Only the selected 17 landmarks listed above
- Binary classification for each pose type
- Confidence threshold: 75% for correct pose detection

## Demo Mode

If models are not found, the system will run in demo mode with random predictions for testing purposes.