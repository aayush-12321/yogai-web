import cv2
import numpy as np
import mediapipe as mp
import os
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class YogaPoseDetector:
    """
    Yoga pose detection using MediaPipe for keypoint extraction
    and separate trained models for each pose classification
    """
    
    def __init__(self):
        self.models = {}  # Dictionary to store models for each pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define yoga pose classes - each will have its own model
        self.pose_classes = [
            'plank',
            'tree',
            'warrior2'
        ]
        
        # Define landmark indices to use (15-20 landmarks instead of all 33)
        # Selected key landmarks for pose recognition
        self.selected_landmarks = [
            # Core body landmarks
            0,   # nose
            11,  # left_shoulder
            12,  # right_shoulder
            13,  # left_elbow
            14,  # right_elbow
            15,  # left_wrist
            16,  # right_wrist
            23,  # left_hip
            24,  # right_hip
            25,  # left_knee
            26,  # right_knee
            27,  # left_ankle
            28,  # right_ankle
            29,  # left_heel
            30,  # right_heel
            31,  # left_foot_index
            32,  # right_foot_index
        ]
        
        self._load_models()
    
    def _load_models(self):
        """Load the trained yoga pose classification models for each pose"""
        try:
            # Try to load sklearn models first (.pkl files)
            sklearn_loaded = False
            try:
                import pickle
                for pose in self.pose_classes:
                    model_path = os.path.join(
                        settings.BASE_DIR, 
                        'yoga_backend', 
                        'trained_models', 
                        f'{pose}_model.pkl'
                    )
                    
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            self.models[pose] = pickle.load(f)
                        logger.info(f"Scikit-learn model loaded successfully for {pose} from {model_path}")
                        sklearn_loaded = True
                    else:
                        logger.warning(f"Scikit-learn model file not found for {pose} at {model_path}")
            except ImportError:
                logger.warning("Pickle not available, trying TensorFlow models")
            
            # If no sklearn models loaded, try TensorFlow models
            if not sklearn_loaded:
                try:
                    from tensorflow import keras
                    
                    for pose in self.pose_classes:
                        model_path = os.path.join(
                            settings.BASE_DIR, 
                            'yoga_backend', 
                            'trained_models', 
                            f'{pose}_model.h5'
                        )
                        
                        if os.path.exists(model_path):
                            self.models[pose] = keras.models.load_model(model_path)
                            logger.info(f"TensorFlow model loaded successfully for {pose} from {model_path}")
                        else:
                            logger.warning(f"TensorFlow model file not found for {pose} at {model_path}")
                            logger.warning(f"Pose detection for {pose} will work in demo mode")
                except ImportError:
                    logger.warning("TensorFlow not available, models will run in demo mode")
            
            if not self.models:
                logger.warning("No models loaded, running in demo mode")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.warning("Running in demo mode without models")
    
    def extract_keypoints(self, image):
        """
        Extract pose keypoints using MediaPipe
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            numpy array of selected keypoints or None if no pose detected
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract selected landmarks only
                landmarks = []
                for idx in self.selected_landmarks:
                    landmark = results.pose_landmarks.landmark[idx]
                    landmarks.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])
                
                # Debug: Print landmarks for first frame
                if not hasattr(self, '_debug_printed'):
                    logger.info(f"DEBUG: Extracted {len(self.selected_landmarks)} landmarks with {len(landmarks)} values")
                    logger.info(f"DEBUG: First few landmark values: {landmarks[:20]}")
                    self._debug_printed = True
                
                return np.array(landmarks)
            
            return None
        except Exception as e:
            logger.error(f"Error extracting keypoints: {e}")
            return None
    
    def predict_pose(self, keypoints, target_pose=None):
        """
        Predict yoga pose from keypoints using the appropriate model
        
        Args:
            keypoints: numpy array of pose keypoints
            target_pose: the pose to classify (uses specific model)
            
        Returns:
            dict with prediction results
        """
        # If no target pose specified, use general classification (not implemented)
        if target_pose is None:
            return {
                'success': False,
                'error': 'Target pose must be specified for classification'
            }
        
        # Check if we have a model for this pose
        if target_pose not in self.models or self.models[target_pose] is None:
            # Demo mode - return random predictions for testing
            import random
            confidence = random.uniform(0.6, 0.95)
            return {
                'success': True,
                'pose': target_pose,
                'confidence': confidence,
                'is_correct': confidence > 0.75,  # Changed threshold to 75%
                'mode': 'demo',
                'target_pose': target_pose
            }
        
        try:
            # Reshape keypoints for model input
            keypoints_input = keypoints.reshape(1, -1)
            
            # Get the specific model for this pose
            model = self.models[target_pose]
            
            # Make prediction
            predictions = model.predict(keypoints_input, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # For single-pose models, the prediction is binary: this pose or not
            # Assuming the model outputs [not_pose_confidence, pose_confidence]
            is_correct = confidence > 0.75  # 75% threshold as requested
            
            return {
                'success': True,
                'pose': target_pose if is_correct else 'unknown',
                'confidence': confidence,
                'is_correct': is_correct,
                'mode': 'model',
                'target_pose': target_pose
            }
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_frame(self, frame_data, target_pose=None):
        """
        Process a base64 encoded frame
        
        Args:
            frame_data: base64 encoded image string
            target_pose: the pose to classify
            
        Returns:
            dict with detection results
        """
        try:
            import base64
            
            # Decode base64 image
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    'success': False,
                    'message': 'Invalid image data'
                }
            
            # Extract keypoints
            keypoints = self.extract_keypoints(image)
            
            if keypoints is None:
                return {
                    'success': False,
                    'message': 'No pose detected in frame'
                }
            
            # Predict pose
            result = self.predict_pose(keypoints, target_pose)
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pose'):
            self.pose.close()


# Global instance (singleton pattern)
_detector_instance = None

def get_detector():
    """Get or create the global detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YogaPoseDetector()
    return _detector_instance