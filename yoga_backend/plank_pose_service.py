"""
plank_pose_service.py

ML inference service for Plank Pose.

Receives a MediaPipe landmark list from YogaPoseDetector (already detected)
and runs the trained sklearn pipeline using YAML-driven feature engineering.

The plank YAML uses text-based joint names (e.g. "left_shoulder") rather than
numeric indices.  This service maps those names to MediaPipe landmark indices
at load time, then uses the same feature-engineering maths as the training
script (plank_specific_fe.py).

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

POSE_NAME        = "plank_pose"
EXPERIMENT_TAG   = "full_class"
AVAILABLE_MODELS = ["logistic_regression", "svc", "knn", "xgboost"]

# Plank YAML uses these text names for joints.
# Map each name to the MediaPipe PoseLandmark index.
LANDMARK_NAME_TO_INDEX: dict[str, int] = {
    "nose":           0,
    "left_eye_inner": 1,
    "left_eye":       2,
    "left_eye_outer": 3,
    "right_eye_inner":4,
    "right_eye":      5,
    "right_eye_outer":6,
    "left_ear":       7,
    "right_ear":      8,
    "mouth_left":     9,
    "mouth_right":    10,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_pinky":     17,
    "right_pinky":    18,
    "left_index":     19,
    "right_index":    20,
    "left_thumb":     21,
    "right_thumb":    22,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_heel":      29,
    "right_heel":     30,
    "left_foot_index":31,
    "right_foot_index":32,
}

# These feature names must match what the trained pipeline expects.
# They are derived from the YAML engineered_features section.
FEATURE_COLUMNS = [
    "neck_tilt_angle",
    "torso_hip_angle",
    "hip_knee_angle",
    "hip_shoulder_vertical_ratio",
    "plumb_line_alignment",
]

PROBABILITY_THRESHOLD = 0.5


#  Low-level helpers ─

def _resolve_joint(name_or_int) -> int:
    """Convert a text landmark name to its MediaPipe index."""
    if isinstance(name_or_int, int):
        return name_or_int
    return LANDMARK_NAME_TO_INDEX[name_or_int]


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


#  Per-feature-type compute functions ─
# These mirror plank_specific_fe.py but operate on a single MediaPipe
# landmark list instead of a DataFrame row.

def _joint_angle(landmarks, cfg: dict) -> float:
    j = [_resolve_joint(name) for name in cfg["joints"]]
    return _angle_at_vertex(
        _xyz(landmarks, j[0]),
        _xyz(landmarks, j[1]),
        _xyz(landmarks, j[2]),
    )


def _spatial_distance(landmarks, cfg: dict) -> float:
    j    = [_resolve_joint(name) for name in cfg["joints"]]
    name = cfg["name"]

    # Match training script: vertical features use Y-axis only
    if "vertical" in name.lower() or "y_axis" in name.lower():
        y_a = float(landmarks[j[0]].y)
        y_b = float(landmarks[j[1]].y)
        dist = abs(y_a - y_b)
    else:
        dist = _euclidean(_xyz(landmarks, j[0]), _xyz(landmarks, j[1]))

    if "normalization_factor" in cfg:
        nf   = [_resolve_joint(name) for name in cfg["normalization_factor"]]
        norm = _euclidean(_xyz(landmarks, nf[0]), _xyz(landmarks, nf[1]))
        dist = dist / (norm + EPSILON)

    return dist


def _alignment_offset(landmarks, cfg: dict) -> float:
    j    = [_resolve_joint(name) for name in cfg["joints"]]
    name = cfg["name"]
    n    = len(j)

    try:
        if n == 2:
            val = abs(float(landmarks[j[0]].x) - float(landmarks[j[1]].x))

        elif n == 3:
            val = _point_to_line(
                _xyz(landmarks, j[1]),
                _xyz(landmarks, j[0]),
                _xyz(landmarks, j[2]),
            )

        elif n == 4 and "plumb" in name.lower():
            # Spine chain line: deviation of inner joints from outer-joint vector
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
        nf    = [_resolve_joint(name) for name in cfg["normalization_factor"]]
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


#  Service ─

class PlankPoseService:
    """
    Loads the plank pose ML pipeline once at startup.
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

    #  Loading ─

    def _load(self, artefacts_root: Path, yaml_path: Path, force_model: Optional[str]) -> None:

        # 1. YAML config
        if not yaml_path.exists():
            logger.error(f"Plank pose YAML not found: {yaml_path}")
            return

        with open(yaml_path, "r") as f:
            pose_yaml = yaml.safe_load(f)

        self.feat_config   = pose_yaml.get("features_config", {}).get("engineered_features", {})
        self.yaml_feedback = pose_yaml.get("feedback", {})

        if not self.feat_config:
            logger.error("No 'features_config.engineered_features' in plank pose YAML.")
            return

        logger.info(f"Plank YAML loaded. Feedback classes: {list(self.yaml_feedback.keys())}")

        # 2. Model selection
        dir_models  = artefacts_root / "models"
        dir_results = artefacts_root / "results"

        if force_model is not None:
            selected = force_model
            logger.info(f"Plank pose model forced: {selected}")
        else:
            results_csv = dir_results / f"{POSE_NAME}_{EXPERIMENT_TAG}_experiment_results.csv"
            if not results_csv.exists():
                logger.error(f"Results CSV not found: {results_csv}")
                return
            df       = pd.read_csv(results_csv)
            best     = df.sort_values("test_f1_weighted", ascending=False).iloc[0]
            selected = best["model"]
            logger.info(
                f"Plank pose auto-selected: {selected} "
                f"(test_f1={best['test_f1_weighted']:.4f})"
            )

        # 3. Pipeline + label encoder
        prefix  = f"{POSE_NAME}_{EXPERIMENT_TAG}_{selected}"
        p_path  = dir_models / f"{prefix}_pipeline.joblib"
        le_path = dir_models / f"{prefix}_label_encoder.joblib"

        if not p_path.exists() or not le_path.exists():
            logger.error(
                f"Plank pose artefacts not found.\n"
                f"  Expected: {p_path}\n"
                f"            {le_path}"
            )
            return

        self.pipeline      = joblib.load(p_path)
        self.label_encoder = joblib.load(le_path)
        self.has_proba     = hasattr(self.pipeline, "predict_proba")
        self._loaded       = True

        logger.info(
            f"Plank pose pipeline loaded: {p_path.name} | "
            f"classes: {list(self.label_encoder.classes_)}"
        )

    #  Inference ─

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
                "pose":       "plank",
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

        # Map predictions to expected YAML feedback keys:
        # C/c -> correct, L/l -> low_back, H/h -> high_back
        class_mapping = {
            "C": "correct",
            "c": "correct",
            "L": "low_back",
            "l": "low_back",
            "H": "high_back",
            "h": "high_back",
        }
        pred_class = class_mapping.get(pred_class, pred_class)

        display_class = pred_class if confidence >= PROBABILITY_THRESHOLD else "unknown"
        feedback      = self.yaml_feedback.get(display_class, "")

        logger.debug(f"Plank | class={pred_class} conf={confidence:.3f} → {display_class}")

        return {
            "success":    True,
            "pose":       "plank",
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
        "pose":       "plank",
        "prediction": "unknown",
        "confidence": 0.0,
        "is_correct": False,
        "feedback":   "Plank pose model not loaded.",
    }
