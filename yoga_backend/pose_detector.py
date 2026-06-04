"""
pose_detector.py

Central inference hub for all three poses.
Uses the MediaPipe Tasks API (mediapipe >= 0.10.x) — mp.solutions.pose is NOT used.

Running modes:
  Live webcam  → IMAGE mode  (stateless, isolated frames, matches training images)
  Video upload → VIDEO mode  (temporal tracking, matches training video extraction)
                 A FRESH landmarker is created per video — this is mandatory because
                 VIDEO mode is stateful and timestamps must be monotonically increasing
                 within a single landmarker instance. Reusing across videos causes
                 "Input timestamp must be monotonically increasing" crashes.

Pose routing:
  mountain  → MountainPoseService (ML pipeline, YAML feature engineering)
  warrior2  → rule-based classify_warrior2() from angle_utils.py
  plank     → ML pipeline via _predict_ml()
"""

import logging
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

from yoga_backend.angle_utils import compute_angle_from_landmarks

logger = logging.getLogger(__name__)

# ── MediaPipe landmark index map ─────────────────────────────────────────────
LANDMARK_INDICES = {
    "NOSE":              0,
    "LEFT_EYE_INNER":   1,
    "LEFT_EYE":         2,
    "LEFT_EYE_OUTER":   3,
    "RIGHT_EYE_INNER":  4,
    "RIGHT_EYE":        5,
    "RIGHT_EYE_OUTER":  6,
    "LEFT_EAR":         7,
    "RIGHT_EAR":        8,
    "MOUTH_LEFT":       9,
    "MOUTH_RIGHT":      10,
    "LEFT_SHOULDER":    11,
    "RIGHT_SHOULDER":   12,
    "LEFT_ELBOW":       13,
    "RIGHT_ELBOW":      14,
    "LEFT_WRIST":       15,
    "RIGHT_WRIST":      16,
    "LEFT_PINKY":       17,
    "RIGHT_PINKY":      18,
    "LEFT_INDEX":       19,
    "RIGHT_INDEX":      20,
    "LEFT_THUMB":       21,
    "RIGHT_THUMB":      22,
    "LEFT_HIP":         23,
    "RIGHT_HIP":        24,
    "LEFT_KNEE":        25,
    "RIGHT_KNEE":       26,
    "LEFT_ANKLE":       27,
    "RIGHT_ANKLE":      28,
    "LEFT_HEEL":        29,
    "RIGHT_HEEL":       30,
    "LEFT_FOOT_INDEX":  31,
    "RIGHT_FOOT_INDEX": 32,
}

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_task_model(model_path: Path) -> None:
    """Download the .task model file if not already present."""
    if model_path.exists():
        return
    logger.info(f"Downloading MediaPipe .task model to {model_path} ...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, model_path)
    logger.info("MediaPipe model downloaded.")


def build_image_landmarker(model_path: Path) -> mp_vision.PoseLandmarker:
    """
    IMAGE mode — stateless, one detection per call, no timestamps needed.
    Used for live webcam frames. Matches training-time image extraction.
    """
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def build_video_landmarker(model_path: Path) -> mp_vision.PoseLandmarker:
    """
    VIDEO mode — stateful, requires monotonically increasing timestamps.
    Must be created fresh for each video file. Matches training-time video
    extraction exactly (same options as _process_video in the training script).
    """
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


class YogaPoseDetector:

    def __init__(self):
        self.models       = {}
        self.pose_classes = ["plank", "mountain", "warrior2"]

        self._lm_index = LANDMARK_INDICES

        self.pose_landmarks = {
            "plank": [
                "NOSE",
                "LEFT_SHOULDER",   "RIGHT_SHOULDER",
                "LEFT_ELBOW",      "RIGHT_ELBOW",
                "LEFT_WRIST",      "RIGHT_WRIST",
                "LEFT_HIP",        "RIGHT_HIP",
                "LEFT_KNEE",       "RIGHT_KNEE",
                "LEFT_ANKLE",      "RIGHT_ANKLE",
                "LEFT_HEEL",       "RIGHT_HEEL",
                "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
            ],
            "warrior2": [
                "NOSE",
                "LEFT_EYE",        "RIGHT_EYE",
                "LEFT_SHOULDER",   "RIGHT_SHOULDER",
                "LEFT_ELBOW",      "RIGHT_ELBOW",
                "LEFT_INDEX",      "RIGHT_INDEX",
                "LEFT_HIP",        "RIGHT_HIP",
                "LEFT_KNEE",       "RIGHT_KNEE",
                "LEFT_HEEL",       "RIGHT_HEEL",
                "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
            ],
            "mountain": [
                "NOSE",
                "LEFT_EAR",        "RIGHT_EAR",
                "LEFT_SHOULDER",   "RIGHT_SHOULDER",
                "LEFT_ELBOW",      "RIGHT_ELBOW",
                "LEFT_WRIST",      "RIGHT_WRIST",
                "LEFT_INDEX",      "RIGHT_INDEX",
                "LEFT_HIP",        "RIGHT_HIP",
                "LEFT_KNEE",       "RIGHT_KNEE",
                "LEFT_ANKLE",      "RIGHT_ANKLE",
                "LEFT_HEEL",       "RIGHT_HEEL",
            ],
        }

        self.landmark_columns = {
            pose: [
                f"{lm.lower()}_{axis}"
                for lm in landmarks
                for axis in ["x", "y", "z", "v"]
            ]
            for pose, landmarks in self.pose_landmarks.items()
        }

        self.label_maps = {
            "plank": {0: "correct", 1: "high_back", 2: "low_back"},
            "warrior2": {
                0: "correct",
                1: "bent_front_knee",
                2: "arms_dropped",
                3: "arms_bent",
                4: "hips_rotated",
                5: "back_foot_wrong_angle",
                6: "leaning_forward",
                7: "narrow_stance",
                8: "shoulders_not_aligned",
            },
        }

        self.feedback_map = {
            "plank": {
                "high_back": "Lower your hips.",
                "low_back":  "Raise your hips.",
                "correct":   "Good plank!",
            },
            "warrior2": {
                "correct":               "Great Warrior 2!",
                "bent_front_knee":       "Bend your front knee more — aim for 90°.",
                "arms_dropped":          "Raise your arms to shoulder height.",
                "arms_bent":             "Straighten your arms fully.",
                "hips_rotated":          "Square your hips to the side.",
                "back_foot_wrong_angle": "Turn your back foot to about 90°.",
                "leaning_forward":       "Keep your torso upright — don't lean forward.",
                "narrow_stance":         "Widen your stance for better stability.",
                "shoulders_not_aligned": "Align your shoulders over your hips.",
            },
        }

        self.pose_plausibility_checks = {
            "plank": [
                {
                    "compute": lambda lms: (
                        compute_angle_from_landmarks(lms, 11, 23, 25)
                        + compute_angle_from_landmarks(lms, 12, 24, 26)
                    ) / 2,
                    "min": 70,
                    "max": None,
                    "feedback": "This doesn't look like a plank. Keep your body straight and horizontal.",
                },
                {
                    "compute": lambda lms: (
                        compute_angle_from_landmarks(lms, 23, 25, 27)
                        + compute_angle_from_landmarks(lms, 24, 26, 28)
                    ) / 2,
                    "min": 100,
                    "max": None,
                    "feedback": "Your knees appear bent. Keep your legs straight for a plank.",
                },
            ],
            "warrior2": [
                {
                    "compute": lambda lms: min(
                        compute_angle_from_landmarks(lms, 23, 25, 27),
                        compute_angle_from_landmarks(lms, 24, 26, 28),
                    ),
                    "min": None,
                    "max": 120,
                    "feedback": "Bend your front knee more for Warrior 2.",
                },
            ],
        }

        self.prediction_threshold = 0.8
        self._warrior2_threshold  = 0.6
        self._pose_services: dict = {}

        # ── MediaPipe model path ─────────────────────────────────────────────
        from django.conf import settings
        self._task_model_path = (
            Path(settings.BASE_DIR)
            / "yoga_backend"
            / "trained_models"
            / "pose_landmarker_lite.task"
        )
        _ensure_task_model(self._task_model_path)

        # IMAGE mode landmarker — one shared instance for live webcam.
        # Stateless: safe to reuse across requests.
        self._landmarker_image = build_image_landmarker(self._task_model_path)

        # There is NO shared video landmarker here.
        # VideoAnalysisView creates a fresh one per video — see views.py.

        logger.info("MediaPipe PoseLandmarker (IMAGE mode) initialised for live webcam.")

        self.load_models()
        self._init_pose_services()

    # ── Model loading ────────────────────────────────────────────────────────

    def load_models(self) -> None:
        from django.conf import settings
        import pickle

        base_dir = Path(settings.BASE_DIR)
        for pose in ["plank"]:
            path = base_dir / "yoga_backend" / "trained_models" / f"{pose}_pipeline.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self.models[pose] = pickle.load(f)
                logger.info(f"{pose} pipeline loaded.")
            else:
                logger.warning(f"{pose} pipeline not found — demo mode active.")

    def _init_pose_services(self) -> None:
        from django.conf import settings
        from yoga_backend.mountain_pose_service import MountainPoseService

        base_dir       = Path(settings.BASE_DIR)
        trained_models = base_dir / "yoga_backend" / "trained_models"

        artefacts_root = trained_models / "mountain_pose_files"
        yaml_path      = artefacts_root / "mountain_pose.yaml"

        svc = MountainPoseService(
            artefacts_root=artefacts_root,
            yaml_path=yaml_path,
            task_model_path=self._task_model_path,
        )
        self._pose_services["mountain"] = svc

        if svc.is_loaded:
            logger.info("MountainPoseService ready.")
        else:
            logger.warning("MountainPoseService failed to load artefacts.")

    # ── MediaPipe detection ──────────────────────────────────────────────────

    def _detect_landmarks_image(self, bgr_frame: np.ndarray):
        """
        IMAGE mode detection for a single webcam frame.
        No resize — matches training (images processed at original resolution).
        Returns landmark list or None.
        """
        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker_image.detect(mp_image)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    @staticmethod
    def detect_landmarks_video(bgr_frame: np.ndarray, timestamp_ms: int,
                               landmarker: mp_vision.PoseLandmarker):
        """
        VIDEO mode detection for a single frame from a video file.
        Static method — called by VideoAnalysisView with its own per-video landmarker.
        No resize — matches training (videos processed at original resolution).
        Returns landmark list or None.
        """
        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    # ── Keypoint extraction ──────────────────────────────────────────────────

    def _extract_keypoints(self, landmarks, pose_type: str) -> np.ndarray:
        points = []
        for lm_name in self.pose_landmarks[pose_type]:
            idx = self._lm_index[lm_name]
            lm  = landmarks[idx]
            points.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(points)

    # ── Visibility gate ──────────────────────────────────────────────────────

    def _is_body_visible(self, landmarks, pose_type: str, threshold: float = 0.5) -> tuple:
        pairs = {
            "plank":    [(11, 12), (23, 24), (25, 26), (27, 28)],
            "mountain": [(11, 12), (23, 24), (25, 26), (29, 30)],
            "warrior2": [(11, 12), (23, 24), (25, 26), (27, 28)],
        }
        for pair in pairs.get(pose_type, []):
            if not any(landmarks[i].visibility >= threshold for i in pair):
                return False, "Part of your body is not visible. Make sure your full body is in frame."
        return True, ""

    # ── Plausibility gate ────────────────────────────────────────────────────

    def _is_plausible(self, landmarks, pose_type: str) -> tuple:
        for check in self.pose_plausibility_checks.get(pose_type, []):
            try:
                value = check["compute"](landmarks)
                lo, hi = check.get("min"), check.get("max")
                if (lo is not None and value < lo) or (hi is not None and value > hi):
                    return False, check["feedback"]
            except Exception as exc:
                logger.warning(f"Plausibility check failed for {pose_type}: {exc}")
        return True, ""

    # ── Prediction routing ───────────────────────────────────────────────────

    def predict_from_landmarks(self, landmarks, target_pose: str) -> dict:
        """
        Run the appropriate prediction backend given detected landmarks.
        Public so VideoAnalysisView can call it directly after detecting
        landmarks with its own per-video landmarker.
        """
        ok, msg = self._is_body_visible(landmarks, target_pose)
        if not ok:
            return _unknown_result(target_pose, msg)

        if target_pose == "mountain":
            svc = self._pose_services.get("mountain")
            if svc is None or not svc.is_loaded:
                return _unknown_result("mountain", "Mountain pose model not loaded.")
            return svc.predict_from_landmarks(landmarks)

        if target_pose == "warrior2":
            return self._predict_warrior2(landmarks, target_pose)

        ok, msg = self._is_plausible(landmarks, target_pose)
        if not ok:
            return _unknown_result(target_pose, msg)

        keypoints = self._extract_keypoints(landmarks, target_pose)
        return self._predict_ml(keypoints, target_pose)

    def _predict_ml(self, keypoints: np.ndarray, target_pose: str) -> dict:
        if target_pose not in self.models:
            return _unknown_result(target_pose, "Model not loaded.")

        model  = self.models[target_pose]
        X_full = self._build_feature_df(keypoints, target_pose)

        probs      = model.predict_proba(X_full)[0]
        pred_class = int(model.predict(X_full)[0])
        confidence = float(probs[pred_class])

        label    = self.label_maps[target_pose].get(pred_class, "unknown")
        feedback = self.feedback_map.get(target_pose, {}).get(label, "")
        display  = label if confidence >= self.prediction_threshold else "unknown"

        return {
            "success":    True,
            "pose":       target_pose,
            "prediction": display,
            "confidence": round(confidence, 3),
            "is_correct": label == "correct",
            "feedback":   feedback,
        }

    def _predict_warrior2(self, landmarks, target_pose: str) -> dict:
        from yoga_backend.angle_utils import classify_warrior2

        keypoints = self._extract_keypoints(landmarks, target_pose)
        cols      = self.landmark_columns[target_pose]
        row       = pd.DataFrame([keypoints], columns=cols).iloc[0]

        label, confidence, feedback = classify_warrior2(row)
        display = label if confidence >= self._warrior2_threshold else "unknown"

        return {
            "success":    True,
            "pose":       target_pose,
            "prediction": display,
            "confidence": confidence,
            "is_correct": label == "correct",
            "feedback":   feedback,
        }

    def _build_feature_df(self, keypoints: np.ndarray, pose_type: str) -> pd.DataFrame:
        from yoga_backend.angle_utils import compute_all_angles

        cols               = self.landmark_columns[pose_type]
        df                 = pd.DataFrame([keypoints], columns=cols)
        angles, angle_cols = compute_all_angles(df.iloc[0], pose_type)

        if angles:
            return pd.concat([df, pd.DataFrame([angles], columns=angle_cols)], axis=1)
        return df

    # ── Public API: live webcam ──────────────────────────────────────────────

    def process_frame(self, frame_b64: str, target_pose: str) -> dict:
        """
        Decode a base64 JPEG and run IMAGE mode detection + prediction.
        Used by PoseDetectionView (live webcam).
        No resize — training images were not resized.
        """
        import base64

        try:
            # Handle both bare base64 and data-URI prefixed strings
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",")[1]
            raw   = base64.b64decode(frame_b64)
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.error(f"Failed to decode webcam frame: {exc}")
            return {"success": False, "message": "Invalid frame data."}

        if image is None:
            return {"success": False, "message": "Could not decode frame."}

        landmarks = self._detect_landmarks_image(image)
        if landmarks is None:
            return {"success": False, "message": "No pose detected."}

        return self.predict_from_landmarks(landmarks, target_pose)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _unknown_result(pose: str, feedback: str) -> dict:
    return {
        "success":    True,
        "pose":       pose,
        "prediction": "unknown",
        "confidence": 0.0,
        "is_correct": False,
        "feedback":   feedback,
    }


# ── Singleton ────────────────────────────────────────────────────────────────

_detector: YogaPoseDetector | None = None


def get_detector() -> YogaPoseDetector:
    global _detector
    if _detector is None:
        _detector = YogaPoseDetector()
    return _detector