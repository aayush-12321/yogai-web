"""
mountain_pose_service.py

ML inference service for Mountain Pose.

Receives a MediaPipe landmark list from YogaPoseDetector (already detected)
and runs the trained sklearn pipeline using YAML-driven feature engineering.

MediaPipe detection is handled centrally in pose_detector.py — this service
only does feature extraction + ML inference.
"""

import logging
import math
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

EPSILON = 1e-6

POSE_NAME        = "mountain_pose"
EXPERIMENT_TAG   = "4class_no_bent_forward"
AVAILABLE_MODELS = ["logistic_regression", "svc", "knn", "xgboost"]

FEATURE_COLUMNS = [
    "left_torso_hip_angle",
    "right_torso_hip_angle",
    "left_shoulder_arm_angle",
    "right_shoulder_arm_angle",
    "neck_tilt_angle",
    "feet_distance_normalized",
    "ear_shoulder_lateral_delta",
    "plumb_line_alignment",
]

PROBABILITY_THRESHOLD = 0.5


#  Feature engineering (exact port from notebook Cell 4) ─

def _xyz(landmarks, idx: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba     = a - b
    bc     = c - b
    cosine = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + EPSILON),
        -1.0, 1.0,
    )
    return float(np.degrees(np.arccos(cosine)))


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _point_to_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab    = b - a
    cross = np.cross(p - a, ab)
    return float(np.linalg.norm(cross) / (np.linalg.norm(ab) + EPSILON))


def _joint_angle(landmarks, cfg: dict) -> float:
    j = cfg["joints"]
    return _angle_at_vertex(_xyz(landmarks, j[0]), _xyz(landmarks, j[1]), _xyz(landmarks, j[2]))


def _spatial_distance(landmarks, cfg: dict) -> float:
    j    = cfg["joints"]
    dist = _euclidean(_xyz(landmarks, j[0]), _xyz(landmarks, j[1]))
    if "normalization_factor" in cfg:
        nf   = cfg["normalization_factor"]
        norm = _euclidean(_xyz(landmarks, nf[0]), _xyz(landmarks, nf[1]))
        dist = dist / (norm + EPSILON)
    return dist


def _alignment_offset(landmarks, cfg: dict) -> float:
    j    = cfg["joints"]
    name = cfg["name"]
    n    = len(j)

    try:
        if n == 2:
            val = abs(_xyz(landmarks, j[0])[0] - _xyz(landmarks, j[1])[0])

        elif n == 3:
            val = _point_to_line(
                _xyz(landmarks, j[1]),
                _xyz(landmarks, j[0]),
                _xyz(landmarks, j[2]),
            )

        elif n == 4 and "plumb" in name.lower():
            outer_a = _xyz(landmarks, j[0])
            outer_b = _xyz(landmarks, j[3])
            val = (
                _point_to_line(_xyz(landmarks, j[1]), outer_a, outer_b)
                + _point_to_line(_xyz(landmarks, j[2]), outer_a, outer_b)
            )

        elif n == 4:
            lm0, lm1 = _xyz(landmarks, j[0]), _xyz(landmarks, j[1])
            lm2, lm3 = _xyz(landmarks, j[2]), _xyz(landmarks, j[3])
            a01   = np.arctan2(lm1[1] - lm0[1], lm1[0] - lm0[0])
            a23   = np.arctan2(lm3[1] - lm2[1], lm3[0] - lm2[0])
            delta = (a01 - a23 + math.pi) % (2 * math.pi) - math.pi
            val   = math.degrees(abs(delta))

        else:
            raise ValueError(f"Alignment offset '{name}' has {n} joints; expected 2, 3, or 4.")

    except Exception as exc:
        warnings.warn(f"Failed computing alignment offset '{name}': {exc}")
        return float("nan")

    if "normalization_factor" in cfg:
        nf    = cfg["normalization_factor"]
        scale = _euclidean(_xyz(landmarks, nf[0]), _xyz(landmarks, nf[1]))
        val   = val / (scale + EPSILON)

    return val


def extract_features(landmarks, feat_config: dict, feature_columns: list) -> Optional[np.ndarray]:
    """
    Compute all engineered features for one frame from a MediaPipe landmark list.
    Returns a 1-D float64 array in feature_columns order, or None if any value is NaN.
    """
    fmap: dict[str, float] = {}

    for cfg in feat_config.get("joint_angles", []):
        fmap[cfg["name"]] = _joint_angle(landmarks, cfg)

    for cfg in feat_config.get("spatial_distances", []):
        fmap[cfg["name"]] = _spatial_distance(landmarks, cfg)

    for cfg in feat_config.get("alignment_offsets", []):
        fmap[cfg["name"]] = _alignment_offset(landmarks, cfg)

    try:
        row = np.array([fmap[col] for col in feature_columns], dtype=np.float64)
    except KeyError as e:
        raise KeyError(
            f"Feature {e} not computed. Check that the YAML covers all required features."
        ) from e

    return None if np.any(np.isnan(row)) else row


#  Service 

class MountainPoseService:
    """
    Loads the mountain pose ML pipeline once at startup.
    predict_from_landmarks() is the only public method — MediaPipe detection
    is handled upstream in YogaPoseDetector.
    """

    def __init__(
        self,
        artefacts_root: Path,
        yaml_path: Path,
        task_model_path: Optional[Path] = None,   # kept for API compatibility, not used
        force_model: Optional[str] = None,
    ):
        self.pipeline      = None
        self.label_encoder = None
        self.feat_config   = {}
        self.yaml_feedback: dict[str, str] = {}
        self.has_proba     = False
        self._loaded       = False

        self._load(artefacts_root, yaml_path, force_model)

    #  Loading 

    def _load(self, artefacts_root: Path, yaml_path: Path, force_model: Optional[str]) -> None:

        # 1. YAML config
        if not yaml_path.exists():
            logger.error(f"Mountain pose YAML not found: {yaml_path}")
            return

        with open(yaml_path, "r") as f:
            pose_yaml = yaml.safe_load(f)

        self.feat_config   = pose_yaml.get("features_config", {}).get("engineered_features", {})
        self.yaml_feedback = pose_yaml.get("feedback", {})

        if not self.feat_config:
            logger.error("No 'features_config.engineered_features' in mountain pose YAML.")
            return

        logger.info(f"Mountain YAML loaded. Feedback classes: {list(self.yaml_feedback.keys())}")

        # 2. Model selection
        dir_models  = artefacts_root / "models"
        dir_results = artefacts_root / "results"

        if force_model is not None:
            selected = force_model
            logger.info(f"Mountain pose model forced: {selected}")
        else:
            results_csv = dir_results / f"{POSE_NAME}_{EXPERIMENT_TAG}_experiment_results.csv"
            if not results_csv.exists():
                logger.error(f"Results CSV not found: {results_csv}")
                return
            df       = pd.read_csv(results_csv)
            best     = df.sort_values("test_f1_weighted", ascending=False).iloc[0]
            selected = best["model"]
            logger.info(
                f"Mountain pose auto-selected: {selected} "
                f"(test_f1={best['test_f1_weighted']:.4f})"
            )

        # 3. Pipeline + label encoder
        prefix  = f"{POSE_NAME}_{EXPERIMENT_TAG}_{selected}"
        p_path  = dir_models / f"{prefix}_pipeline.joblib"
        le_path = dir_models / f"{prefix}_label_encoder.joblib"

        if not p_path.exists() or not le_path.exists():
            logger.error(
                f"Mountain pose artefacts not found.\n"
                f"  Expected: {p_path}\n"
                f"            {le_path}"
            )
            return

        self.pipeline      = joblib.load(p_path)
        self.label_encoder = joblib.load(le_path)
        self.has_proba     = hasattr(self.pipeline, "predict_proba")
        self._loaded       = True

        logger.info(
            f"Mountain pose pipeline loaded: {p_path.name} | "
            f"classes: {list(self.label_encoder.classes_)}"
        )

    #  Inference 

    def predict_from_landmarks(self, landmarks) -> dict:
        """
        Run feature extraction + ML inference on a MediaPipe landmark list.
        landmarks: list of NormalizedLandmark with .x .y .z .visibility
        """
        if not self._loaded:
            return _unavailable()

        feat_vec = extract_features(landmarks, self.feat_config, FEATURE_COLUMNS)

        if feat_vec is None:
            return {
                "success":    True,
                "pose":       "mountain",
                "prediction": "unknown",
                "confidence": 0.0,
                "is_correct": False,
                "feedback":   "Pose unclear — ensure your full body is visible.",
            }

        X = feat_vec.reshape(1, -1)

        pred_enc   = self.pipeline.predict(X)[0]
        pred_class = self.label_encoder.inverse_transform([pred_enc])[0]

        if self.has_proba:
            proba      = self.pipeline.predict_proba(X)[0]
            confidence = float(proba[pred_enc])
        else:
            confidence = 1.0

        display_class = pred_class if confidence >= PROBABILITY_THRESHOLD else "unknown"
        feedback      = self.yaml_feedback.get(display_class, "")

        logger.debug(f"Mountain | class={pred_class} conf={confidence:.3f} → {display_class}")

        return {
            "success":    True,
            "pose":       "mountain",
            "prediction": display_class,
            "confidence": round(confidence, 3),
            "is_correct": display_class == "correct",
            "feedback":   feedback,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def _unavailable() -> dict:
    return {
        "success":    True,
        "pose":       "mountain",
        "prediction": "unknown",
        "confidence": 0.0,
        "is_correct": False,
        "feedback":   "Mountain pose model not loaded.",
    }