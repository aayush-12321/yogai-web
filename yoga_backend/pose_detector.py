









#####################################################################################33333


import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
from yoga_backend.angle_utils import (
    compute_angle_from_landmarks
)

class YogaPoseDetector:

    def __init__(self):
        self.models = {}  # stores pipelines (scaling is built-in)
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pose_classes = ["plank", "mountain", "warrior2"]

        # Named landmarks — same order as training
        self.pose_landmarks = {
            "plank": [
                "NOSE",
                "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_ELBOW",    "RIGHT_ELBOW",
                "LEFT_WRIST",    "RIGHT_WRIST",
                "LEFT_HIP",      "RIGHT_HIP",
                "LEFT_KNEE",     "RIGHT_KNEE",
                "LEFT_ANKLE",    "RIGHT_ANKLE",
                "LEFT_HEEL",     "RIGHT_HEEL",
                "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX",
            ],
            # Add mountain/warrior2 landmark lists when ready
            "mountain": [],
            "warrior2": [],
        }

        # Column names for landmark features (matches HEADERS[1:] from training)
        self.landmark_columns = {
            "plank": [
                f"{lm.lower()}_{axis}"
                for lm in self.pose_landmarks["plank"]
                for axis in ["x", "y", "z", "v"]
            ],
            # Add others when ready
        }

        self.label_maps = {
            "plank": {0: "correct", 1: "high_back", 2: "low_back"},
            "mountain": {0: "correct"},
            "warrior2": {0: "correct"},
        }

        self.feedback_map = {
            "plank": {
                "high_back": "Lower your hips",
                "low_back":  "Raise your hips",
                "correct":   "Good plank!"
            }
        }

        self.pose_plausibility_checks = {
            "plank": [

            #     {
            #         # In plank, shoulders and hips are at similar heights (horizontal body)
            # # In standing poses, hips Y is much greater than shoulders Y
            # # Difference threshold found by testing — tune if needed
            # "compute": lambda lms: abs(
            #     ((lms[11].y + lms[12].y) / 2) -   # avg shoulder Y
            #     ((lms[23].y + lms[24].y) / 2)      # avg hip Y
            # ),
            # "min": None,
            # "max": 0.3,   # if difference > 0.3, body is upright not horizontal
            # "feedback": "This doesn't look like a plank. Make sure your body is horizontal."
        
                # }
                {
                    # Hip should be straight (close to 180°) — shoulder→hip→knee
                    "compute": lambda lms: (
                        compute_angle_from_landmarks(lms, 11, 23, 25) +  # left
                        compute_angle_from_landmarks(lms, 12, 24, 26)    # right
                    ) / 2,
                    "min": 70,
                    "max": None,
                    "feedback": "This doesn't look like a plank. Keep your body straight and horizontal."
                },
                {
                    # Knees should be straight
                    "compute": lambda lms: (
                        compute_angle_from_landmarks(lms, 23, 25, 27) +  # left hip→knee→ankle
                        compute_angle_from_landmarks(lms, 24, 26, 28)    # right
                    ) / 2,
                    "min": 100,
                    "max": None,
                    "feedback": "Your knees appear bent. Keep your legs straight for a plank."
                },
            ],

            # "mountain": [
            #     {
            #         "compute": lambda lms: (
            #             compute_angle_from_landmarks(lms, 11, 23, 25) +
            #             compute_angle_from_landmarks(lms, 12, 24, 26)
            #         ) / 2,
            #         "min": 160,
            #         "max": None,
            #         "feedback": "Stand tall and straight for mountain pose."
            #     },
            # ],

            # "warrior2": [
            #     {
            #         # Front knee must be bent
            #         "compute": lambda lms: min(
            #             compute_angle_from_landmarks(lms, 23, 25, 27),
            #             compute_angle_from_landmarks(lms, 24, 26, 28)
            #         ),
            #         "min": None,
            #         "max": 120,
            #         "feedback": "Bend your front knee more for Warrior 2."
            #     },
            # ],
        }

        self.prediction_threshold = 0.8

        self.load_models()

    def load_models(self):
        from django.conf import settings
        import pickle

        base_dir = settings.BASE_DIR

        for pose in self.pose_classes:
            # Each pose now uses a pipeline .pkl (scaler + model bundled)
            path = os.path.join(base_dir, "yoga_backend", "trained_models", f"{pose}_pipeline.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.models[pose] = pickle.load(f)
                logger.info(f"{pose} pipeline loaded")
            else:
                logger.warning(f"{pose} pipeline not found – demo mode")

    # def extract_keypoints(self, image, pose_type):
    #     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     results = self.pose.process(rgb)

    #     if not results.pose_landmarks:
    #         return None

    #     points = []
    #     for lm_name in self.pose_landmarks[pose_type]:
    #         idx = self.mp_pose.PoseLandmark[lm_name].value
    #         lm = results.pose_landmarks.landmark[idx]
    #         points.extend([lm.x, lm.y, lm.z, lm.visibility])

    #     return np.array(points)
    def extract_keypoints(self, image, pose_type):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, None  # ← return tuple now

        points = []
        for lm_name in self.pose_landmarks[pose_type]:
            idx = self.mp_pose.PoseLandmark[lm_name].value
            lm = results.pose_landmarks.landmark[idx]
            points.extend([lm.x, lm.y, lm.z, lm.visibility])

        return np.array(points), results.pose_landmarks.landmark  # ← return both

    def is_target_pose_plausible(self, results_landmarks, pose_type):
        """
        Sanity check using raw MediaPipe landmarks before running the classifier.
        Works for any pose — no dependency on which landmarks were selected for training.
        Returns (is_plausible, feedback_message)
        """
        from yoga_backend.angle_utils import compute_angle_from_landmarks

        checks = self.pose_plausibility_checks.get(pose_type, [])
        for check in checks:
            try:
                value = check["compute"](results_landmarks)
                min_val = check.get("min")
                max_val = check.get("max")
                if (min_val is not None and value < min_val) or \
                (max_val is not None and value > max_val):
                    return False, check["feedback"]
            except Exception as e:
                logger.warning(f"Plausibility check failed for {pose_type}: {e}")
                continue  # skip this check if something goes wrong

        return True, ""
    
    def is_full_body_visible(self, raw_landmarks, pose_type, threshold=0.5):
        """
        Check if key landmarks are visible enough.
        For symmetric pairs (e.g. shoulders), at least one side must be visible.
        """
        required_landmarks = {
            "plank": [
                (11, 12),  # shoulders
                (23, 24),  # hips
                (25, 26),  # knees
                (27, 28),  # ankles
            ],
            "mountain": [
                (11, 12),
                (23, 24),
                (25, 26),
                (27, 28),
            ],
            "warrior2": [
                (11, 12),
                (23, 24),
                (25, 26),
                (27, 28),
            ],
        }

        required = required_landmarks.get(pose_type, [])
        for pair in required:
            # At least one of the pair must be visible
            if not any(raw_landmarks[idx].visibility >= threshold for idx in pair):
                return False, f"Part of your body is not visible. Make sure your full body is in frame."

        return True, ""


    def build_feature_dataframe(self, keypoints, pose_type):
        """
        Convert raw keypoints → DataFrame with landmarks + angles,
        matching exactly what the model was trained on.
        """
        from yoga_backend.angle_utils import compute_all_angles

        # Step 1: landmarks DataFrame
        cols = self.landmark_columns[pose_type]
        X_landmarks = pd.DataFrame([keypoints], columns=cols)

        # Step 2: compute angles
        row = X_landmarks.iloc[0]
        angles, angle_cols = compute_all_angles(row, pose_type)

        if angles:
            X_angles = pd.DataFrame([angles], columns=angle_cols)
            X_full = pd.concat([X_landmarks, X_angles], axis=1)
        else:
            X_full = X_landmarks  # fallback if no angles defined yet

        return X_full

    def predict_pose(self, keypoints, raw_landmarks, target_pose):
        is_visible, visibility_feedback = self.is_full_body_visible(raw_landmarks, target_pose)
        print(f"🔍 Visibility check: {is_visible}, feedback: {visibility_feedback}")

        is_plausible, plausibility_feedback = self.is_target_pose_plausible(raw_landmarks, target_pose)
        print(f"🔍 Plausibility check: {is_plausible}, feedback: {plausibility_feedback}")
        if target_pose not in self.models:
            # Demo fallback
            confidence = np.random.uniform(0.6, 0.95)
            return {
                "success": True,
                "pose": target_pose,
                "confidence": round(confidence, 3),
                "is_correct": confidence > 0.75
            }

        if not self.is_full_body_visible(raw_landmarks, target_pose):
            return {
                "success": True,
                "pose": target_pose,
                "prediction": "unknown",
                "confidence": 0.0,
                "is_correct": False,
                "feedback": "Make sure your full body is visible to the camera."
            }

        # Plausibility check using raw MediaPipe landmarks
        is_plausible, plausibility_feedback = self.is_target_pose_plausible(raw_landmarks, target_pose)
        if not is_plausible:
            return {
                "success": True,
                "pose": target_pose,
                "prediction": "unknown",
                "confidence": 0.0,
                "is_correct": False,
                "feedback": plausibility_feedback
            }



        model = self.models[target_pose]  # this is the full pipeline

        # Build the full feature DataFrame (landmarks + angles)
        X_full = self.build_feature_dataframe(keypoints, target_pose)

        # Pipeline handles scaling internally
        probs = model.predict_proba(X_full)[0]
        pred_class = int(model.predict(X_full)[0])
        confidence = float(probs[pred_class])

        label_map = self.label_maps.get(target_pose, {})
        pred_label = label_map.get(pred_class, "unknown")

        print(f"\n📌 Pose: {target_pose}")
        print(f"Predicted class: {pred_class} → {pred_label}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Probabilities: {probs}")

        return {
            "success": True,
            "pose": target_pose,
            "prediction": pred_label if confidence >= self.prediction_threshold else "unknown",
            "confidence": round(confidence, 3),
            "is_correct": pred_label == "correct",
            "feedback": self.feedback_map.get(target_pose, {}).get(pred_label, "")
        }

    def process_frame(self, frame_b64, target_pose):
        import base64

        frame_b64 = frame_b64.split(",")[1]
        image_bytes = base64.b64decode(frame_b64)
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return {"success": False}

        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        keypoints, raw_landmarks = self.extract_keypoints(image, target_pose)  # ← unpack here
        if keypoints is None:
            return {"success": False, "message": "No pose detected"}

        return self.predict_pose(keypoints, raw_landmarks, target_pose)  # ← pass both
            


_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = YogaPoseDetector()
    return _detector










# #####################################################################################33333


# import cv2
# import numpy as np
# import mediapipe as mp
# import os
# import logging

# logger = logging.getLogger(__name__)


# class YogaPoseDetector:
#     """
#     MediaPipe + ML based yoga pose detector
#     """

#     def __init__(self):
#         self.models = {}
#         self.mp_pose = mp.solutions.pose

#         self.pose = self.mp_pose.Pose(
#             static_image_mode=False,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )

#         self.pose_classes = ["plank", "mountain", "warrior2"]

#         self.pose_landmarks = {
#             "plank": [0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32],
#             "mountain": [0,7,8,11,12,13,14,15,16,19,20,23,24,25,26,27,28,29,30],
#             "warrior2": [0,2,5,11,12,13,14,19,20,23,24,25,26,29,30,31,32]
#         }

#         self.label_maps = {
#         "plank": {
#             0: "correct",
#             1: "high_back",
#             2: "low_back"
#         },
#         "mountain": {
#             0: "correct",
#             # 1: "leaning_forward",
#             # 2: "leaning_backward"
#         },
#         "warrior2": {
#             0: "correct",
#             # 1: "front_knee_not_bent",
#             # 2: "arms_not_horizontal"
#         }}

#         self.feedback_map = {
#             "plank": {
#                 "high_back": "Lower your hips",
#                 "low_back": "Raise your hips",
#                 "correct": "Good plank!"
#             }
#         }

#         self.debug_counter = 0


#         self.load_models()

#     def load_models(self):
#         """
#         Load trained ML models (.pkl)
#         """
#         from django.conf import settings
#         import pickle

#         # logger.info(f"{self.pose} classes: {self.models[self.pose].classes_}")

#         base_dir = settings.BASE_DIR

#         for pose in self.pose_classes:
#             path = os.path.join(base_dir, "yoga_backend", "trained_models", f"{pose}_model.pkl")
#             if os.path.exists(path):
#                 with open(path, "rb") as f:
#                     self.models[pose] = pickle.load(f)
#                 logger.info(f"{pose} model loaded")
#             else:
#                 logger.warning(f"{pose} model not found – demo mode")

#     def extract_keypoints(self, image, pose_type):
#         """
#         Extract selected landmarks
#         """
#         rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = self.pose.process(rgb)

#         if not results.pose_landmarks:
#             # print("❌ No pose detected")
#             return None

#         # print("✅ Pose detected")
#         # for i, lm in enumerate(results.pose_landmarks.landmark[:5]):
#         #     print(f"LM {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}, vis={lm.visibility:.3f}")

#         points = []
#         for idx in self.pose_landmarks[pose_type]:
#             lm = results.pose_landmarks.landmark[idx]
#             points.extend([lm.x, lm.y, lm.z, lm.visibility])

#         # print(f"Feature vector length: {len(points)}")

#         return np.array(points)

#     def predict_pose(self, keypoints, target_pose):
#         """
#         Predict pose probability
#         """
#         print("Model input shape:", keypoints.shape)

#         if target_pose not in self.models:
#             # Demo fallback
#             confidence = np.random.uniform(0.6, 0.95)
#             return {
#                 "success": True,
#                 "pose": target_pose,
#                 "confidence": round(confidence, 3),
#                 "is_correct": confidence > 0.75
#             }

#         model = self.models[target_pose]
#         keypoints = keypoints.reshape(1, -1)

#         probs = model.predict_proba(keypoints)[0]
#         pred_class = int(model.predict(keypoints)[0])
#         confidence = float(probs[pred_class])
#         # confidence = float(probs[0][1])

#         label_map = self.label_maps.get(target_pose, {})
#         pred_label = label_map.get(pred_class, "unknown")

#         # 🔍 DEBUG PRINTS
#         # self.debug_counter += 1
#         # if self.debug_counter % 30 == 0:

#         print(f"\n📌 Pose: {target_pose}")
#         print(f"Predicted class index: {pred_class}")
#         print(f"Predicted label: {pred_label}")
#         print(f"Confidence: {confidence:.3f}")
#         print(f"Probabilities: {probs}")

#         return {
#             "success": True,
#             "pose": target_pose,
#             "prediction": pred_label if confidence > 0.8 else "unknown",
#             "confidence": round(confidence, 3),
#             "is_correct": pred_label == "correct",
#             "feedback": self.feedback_map[target_pose].get(pred_label, "")
#             # "success": True,
#             # "pose": target_pose if confidence > 0.75 else "unknown",
#             # "confidence": round(confidence, 3),
#             # "is_correct": confidence > 0.75
#         }

#     def process_frame(self, frame_b64, target_pose):
#         """
#         Full frame → prediction pipeline
#         """
#         import base64

#         frame_b64 = frame_b64.split(",")[1]
#         image_bytes = base64.b64decode(frame_b64)
#         image_np = np.frombuffer(image_bytes, np.uint8)
#         image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

#         if image is None:
#             return {"success": False}

#         keypoints = self.extract_keypoints(image, target_pose)
#         if keypoints is None:
#             return {"success": False}

#         return self.predict_pose(keypoints, target_pose)


# # Singleton
# _detector = None

# def get_detector():
#     global _detector
#     if _detector is None:
#         _detector = YogaPoseDetector()
#     return _detector
