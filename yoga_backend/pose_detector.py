# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# from django.conf import settings
# import logging

# logger = logging.getLogger(__name__)


# class YogaPoseDetector:
#     """
#     Yoga pose detection using MediaPipe for keypoint extraction
#     and separate trained models for each pose classification
#     """
    
#     def __init__(self):
#         self.models = {}  # Dictionary to store models for each pose
#         self.mp_pose = mp.solutions.pose
#         self.pose = self.mp_pose.Pose(
#             static_image_mode=False,
#             model_complexity=1,
#             smooth_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
        
#         # Define yoga pose classes - each will have its own model
#         self.pose_classes = [
#             'plank',
#             'mountain',
#             'warrior2'
#         ]
        
#         # Define pose-specific landmark indices (different landmarks are important for each pose)
#         self.pose_landmarks = {
#             'plank': [
#                 # Plank pose landmarks (17 landmarks × 4 values = 68 features)
#                 0,   # NOSE
#                 11,  # LEFT_SHOULDER
#                 12,  # RIGHT_SHOULDER
#                 13,  # LEFT_ELBOW
#                 14,  # RIGHT_ELBOW
#                 15,  # LEFT_WRIST
#                 16,  # RIGHT_WRIST
#                 23,  # LEFT_HIP
#                 24,  # RIGHT_HIP
#                 25,  # LEFT_KNEE
#                 26,  # RIGHT_KNEE
#                 27,  # LEFT_ANKLE
#                 28,  # RIGHT_ANKLE
#                 29,  # LEFT_HEEL
#                 30,  # RIGHT_HEEL
#                 31,  # LEFT_FOOT_INDEX
#                 32,  # RIGHT_FOOT_INDEX
#             ],
#             'mountain': [
#                 # mountain pose landmarks (19 landmarks × 4 values = 76 features)
#                 0,   # NOSE
#                 7,   # LEFT_EAR
#                 8,   # RIGHT_EAR
#                 11,  # LEFT_SHOULDER
#                 12,  # RIGHT_SHOULDER
#                 13,  # LEFT_ELBOW
#                 14,  # RIGHT_ELBOW
#                 15,  # LEFT_WRIST
#                 16,  # RIGHT_WRIST
#                 19,  # LEFT_INDEX
#                 20,  # RIGHT_INDEX
#                 23,  # LEFT_HIP
#                 24,  # RIGHT_HIP
#                 25,  # LEFT_KNEE
#                 26,  # RIGHT_KNEE
#                 27,  # LEFT_ANKLE
#                 28,  # RIGHT_ANKLE
#                 29,  # LEFT_HEEL
#                 30,  # RIGHT_HEEL
#             ],
#             'warrior2': [
#                 # Warrior II pose landmarks (17 landmarks × 4 values = 68 features)
#                 0,   # NOSE
#                 2,   # LEFT_EYE
#                 5,   # RIGHT_EYE
#                 11,  # LEFT_SHOULDER
#                 12,  # RIGHT_SHOULDER
#                 13,  # LEFT_ELBOW
#                 14,  # RIGHT_ELBOW
#                 19,  # LEFT_INDEX
#                 20,  # RIGHT_INDEX
#                 23,  # LEFT_HIP
#                 24,  # RIGHT_HIP
#                 25,  # LEFT_KNEE
#                 26,  # RIGHT_KNEE
#                 29,  # LEFT_HEEL
#                 30,  # RIGHT_HEEL
#                 31,  # LEFT_FOOT_INDEX
#                 32,  # RIGHT_FOOT_INDEX
#             ]
#         }
        
#         self._load_models()
    
#     def _load_models(self):
#         """Load the trained yoga pose classification models for each pose"""
#         # Check if Django settings are available
#         try:
#             from django.conf import settings
#             base_dir = settings.BASE_DIR
#             logger.info(f"Django BASE_DIR: {base_dir}")
#         except:
#             logger.warning("Django settings not available, skipping model loading")
#             return
            
#         try:
#             # Try to load sklearn models first (.pkl files)
#             sklearn_loaded = False
            
#             # Check if pickle is available
#             try:
#                 import pickle
#                 pickle_available = True
#                 logger.info("Pickle is available")
#             except ImportError as e:
#                 pickle_available = False
#                 logger.warning(f"Pickle not available for loading sklearn models: {e}")
            
#             if pickle_available:
#                 try:
#                     import sklearn
#                     sklearn_available = True
#                     logger.info("Scikit-learn is available")
#                 except ImportError as e:
#                     sklearn_available = False
#                     logger.warning(f"Scikit-learn not available: {e}")
                
#                 if sklearn_available:
#                     for pose in self.pose_classes:
#                         model_path = os.path.join(
#                             base_dir, 
#                             'yoga_backend', 
#                             'trained_models', 
#                             f'{pose}_model.pkl'
#                         )
#                         print(f"model paath: {model_path}")
#                         logger.info(f"Checking model path: {model_path}")
#                         logger.info(f"Model file exists: {os.path.exists(model_path)}")
                        
#                         if os.path.exists(model_path):
#                             try:
#                                 with open(model_path, 'rb') as f:
#                                     self.models[pose] = pickle.load(f)
#                                 logger.info(f"Scikit-learn model loaded successfully for {pose} from {model_path}")
#                                 sklearn_loaded = True
#                             except Exception as e:
#                                 logger.error(f"Error loading model for {pose}: {e}")
#                         else:
#                             logger.warning(f"Scikit-learn model file not found for {pose} at {model_path}")
#                 else:
#                     logger.warning("Scikit-learn not available, cannot load sklearn models")
            
#             # If no sklearn models loaded, try TensorFlow models
#             if not sklearn_loaded:
#                 try:
#                     from tensorflow import keras
                    
#                     for pose in self.pose_classes:
#                         model_path = os.path.join(
#                             base_dir, 
#                             'yoga_backend', 
#                             'trained_models', 
#                             f'{pose}_model.h5'
#                         )
                        
#                         if os.path.exists(model_path):
#                             self.models[pose] = keras.models.load_model(model_path)
#                             logger.info(f"TensorFlow model loaded successfully for {pose} from {model_path}")
#                         else:
#                             logger.warning(f"TensorFlow model file not found for {pose} at {model_path}")
#                             logger.warning(f"Pose detection for {pose} will work in demo mode")
#                 except ImportError:
#                     logger.warning("TensorFlow not available, models will run in demo mode")
            
#             if not self.models:
#                 logger.warning("No models loaded, running in demo mode")
#             else:
#                 logger.info(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")
                
#         except Exception as e:
#             logger.error(f"Error loading models: {e}")
#             logger.warning("Running in demo mode without models")
    
#     def extract_keypoints(self, image, pose_type=None):
#         """
#         Extract pose keypoints using MediaPipe
        
#         Args:
#             image: OpenCV image (BGR format)
#             pose_type: the pose type to extract landmarks for (uses pose-specific landmarks)
            
#         Returns:
#             numpy array of selected keypoints or None if no pose detected
#         """
#         try:
#             # Convert BGR to RGB
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
#             # Process the image
#             results = self.pose.process(image_rgb)
            
#             if results.pose_landmarks:
#                 # Get the appropriate landmarks for this pose
#                 if pose_type and pose_type in self.pose_landmarks:
#                     selected_landmarks = self.pose_landmarks[pose_type]
#                 else:
#                     # Fallback to a default set if pose_type not specified or not found
#                     selected_landmarks = self.pose_landmarks['plank']  # Default to plank landmarks
                
#                 # Extract selected landmarks only
#                 landmarks = []
#                 for idx in selected_landmarks:
#                     landmark = results.pose_landmarks.landmark[idx]
#                     landmarks.extend([
#                         landmark.x,
#                         landmark.y,
#                         landmark.z,
#                         landmark.visibility
#                     ])
                
#                 # Debug: Print landmarks for first frame
#                 if not hasattr(self, '_debug_printed'):
#                     logger.info(f"DEBUG: Extracted {len(selected_landmarks)} landmarks with {len(landmarks)} values for pose {pose_type}")
#                     logger.info(f"DEBUG: First few landmark values: {landmarks[:20]}")
#                     self._debug_printed = True
                
#                 return np.array(landmarks)
            
#             return None
#         except Exception as e:
#             logger.error(f"Error extracting keypoints: {e}")
#             return None
    
#     def predict_pose(self, keypoints, target_pose=None):
#         """
#         Predict yoga pose from keypoints using the appropriate model
        
#         Args:
#             keypoints: numpy array of pose keypoints
#             target_pose: the pose to classify (uses specific model)
            
#         Returns:
#             dict with prediction results
#         """
#         # Validate keypoints
#         if keypoints is None:
#             return {
#                 'success': False,
#                 'error': 'No keypoints provided for classification'
#             }
        
#         if not isinstance(keypoints, np.ndarray):
#             return {
#                 'success': False,
#                 'error': 'Keypoints must be a numpy array'
#             }
        
#         # If no target pose specified, use general classification (not implemented)
#         if target_pose is None:
#             return {
#                 'success': False,
#                 'error': 'Target pose must be specified for classification'
#             }
        
#         # Normalize target_pose to lowercase
#         target_pose = target_pose.lower()
        
#         # Check if we have a model for this pose
#         if target_pose not in self.models or self.models[target_pose] is None:
#             # Demo mode - return random predictions for testing
#             import random
#             confidence = random.uniform(0.6, 0.95)
#             return {
#                 'success': True,
#                 'pose': target_pose,
#                 'confidence': confidence,
#                 'is_correct': confidence > 0.75,  # Changed threshold to 75%
#                 'mode': 'demo',
#                 'target_pose': target_pose
#             }
        
#         try:
#             # Reshape keypoints for model input
#             keypoints_input = keypoints.reshape(1, -1)
            
#             # Get the specific model for this pose
#             model = self.models[target_pose]
            
#             # Make prediction - use predict_proba for confidence scores
#             # Suppress the feature names warning since we don't have the original column names
#             import warnings
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore", UserWarning)
#                 probabilities = model.predict_proba(keypoints_input)
            
#             # For binary classification, probabilities[0][1] is the probability of the positive class (pose)
#             confidence = float(probabilities[0][1])
            
#             # For single-pose models, the prediction is binary: this pose or not
#             is_correct = confidence > 0.75  # 75% threshold as requested
            
#             return {
#                 'success': True,
#                 'pose': target_pose if is_correct else 'unknown',
#                 'confidence': confidence,
#                 'is_correct': is_correct,
#                 'mode': 'model',
#                 'target_pose': target_pose
#             }
#         except Exception as e:
#             logger.error(f"Error making prediction: {e}")
#             return {
#                 'success': False,
#                 'error': str(e)
#             }
    
#     def process_frame(self, frame_data, target_pose=None):
#         """
#         Process a base64 encoded frame
        
#         Args:
#             frame_data: base64 encoded image string
#             target_pose: the pose to classify
            
#         Returns:
#             dict with detection results
#         """
#         try:
#             import base64
            
#             # Decode base64 image
#             if ',' in frame_data:
#                 frame_data = frame_data.split(',')[1]
            
#             img_data = base64.b64decode(frame_data)
#             nparr = np.frombuffer(img_data, np.uint8)
#             image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if image is None:
#                 return {
#                     'success': False,
#                     'message': 'Invalid image data'
#                 }
            
#             # Extract keypoints using pose-specific landmarks
#             keypoints = self.extract_keypoints(image, target_pose)
            
#             if keypoints is None:
#                 return {
#                     'success': False,
#                     'message': 'No pose detected in frame'
#                 }
            
#             # Predict pose
#             result = self.predict_pose(keypoints, target_pose)
#             return result
            
#         except Exception as e:
#             logger.error(f"Error processing frame: {e}")
#             return {
#                 'success': False,
#                 'error': str(e)
#             }
    
#     def __del__(self):
#         """Cleanup"""
#         if hasattr(self, 'pose'):
#             self.pose.close()


# # Global instance (singleton pattern)
# _detector_instance = None

# def get_detector():
#     """Get or create the global detector instance"""
#     global _detector_instance
#     if _detector_instance is None:
#         _detector_instance = YogaPoseDetector()
#     return _detector_instance













#####################################################################################33333


import cv2
import numpy as np
import mediapipe as mp
import os
import logging

logger = logging.getLogger(__name__)


class YogaPoseDetector:
    """
    MediaPipe + ML based yoga pose detector
    """

    def __init__(self):
        self.models = {}
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pose_classes = ["plank", "mountain", "warrior2"]

        self.pose_landmarks = {
            "plank": [0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32],
            "mountain": [0,7,8,11,12,13,14,15,16,19,20,23,24,25,26,27,28,29,30],
            "warrior2": [0,2,5,11,12,13,14,19,20,23,24,25,26,29,30,31,32]
        }

        self.load_models()

    def load_models(self):
        """
        Load trained ML models (.pkl)
        """
        from django.conf import settings
        import pickle

        base_dir = settings.BASE_DIR

        for pose in self.pose_classes:
            path = os.path.join(base_dir, "yoga_backend", "trained_models", f"{pose}_model.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.models[pose] = pickle.load(f)
                logger.info(f"{pose} model loaded")
            else:
                logger.warning(f"{pose} model not found – demo mode")

    def extract_keypoints(self, image, pose_type):
        """
        Extract selected landmarks
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        points = []
        for idx in self.pose_landmarks[pose_type]:
            lm = results.pose_landmarks.landmark[idx]
            points.extend([lm.x, lm.y, lm.z, lm.visibility])

        return np.array(points)

    def predict_pose(self, keypoints, target_pose):
        """
        Predict pose probability
        """
        if target_pose not in self.models:
            # Demo fallback
            confidence = np.random.uniform(0.6, 0.95)
            return {
                "success": True,
                "pose": target_pose,
                "confidence": round(confidence, 3),
                "is_correct": confidence > 0.75
            }

        model = self.models[target_pose]
        keypoints = keypoints.reshape(1, -1)

        probs = model.predict_proba(keypoints)
        confidence = float(probs[0][1])

        return {
            "success": True,
            "pose": target_pose if confidence > 0.75 else "unknown",
            "confidence": round(confidence, 3),
            "is_correct": confidence > 0.75
        }

    def process_frame(self, frame_b64, target_pose):
        """
        Full frame → prediction pipeline
        """
        import base64

        frame_b64 = frame_b64.split(",")[1]
        image_bytes = base64.b64decode(frame_b64)
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return {"success": False}

        keypoints = self.extract_keypoints(image, target_pose)
        if keypoints is None:
            return {"success": False}

        return self.predict_pose(keypoints, target_pose)


# Singleton
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = YogaPoseDetector()
    return _detector
