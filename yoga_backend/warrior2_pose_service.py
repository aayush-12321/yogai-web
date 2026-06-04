"""
warrior2_pose_service.py

ML inference service for Warrior 2 Pose.

Receives a MediaPipe landmark list from YogaPoseDetector (already detected)
and runs the trained sklearn pipeline using YAML-driven feature engineering.
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

POSE_NAME        = "warrior2_pose"
EXPERIMENT_TAG   = "reduced_class"
AVAILABLE_MODELS = ["logistic_regression", "svc", "knn", "xgboost"]

# MediaPipe landmark index constants
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_WRIST     = 15
_RIGHT_WRIST    = 16
_LEFT_HIP       = 23
_RIGHT_HIP      = 24
_LEFT_KNEE      = 25
_RIGHT_KNEE     = 26
_LEFT_ANKLE     = 27
_RIGHT_ANKLE    = 28

# Orientation map: semantic role -> (right-facing idx, left-facing idx)
_ORIENTATION_MAP: dict[str, tuple[int, int]] = {
    "front_hip"      : (_RIGHT_HIP,      _LEFT_HIP),
    "front_knee"     : (_RIGHT_KNEE,     _LEFT_KNEE),
    "front_ankle"    : (_RIGHT_ANKLE,    _LEFT_ANKLE),
    "front_shoulder" : (_RIGHT_SHOULDER, _LEFT_SHOULDER),
    "front_wrist"    : (_RIGHT_WRIST,    _LEFT_WRIST),
    "back_hip"       : (_LEFT_HIP,       _RIGHT_HIP),
    "back_knee"      : (_LEFT_KNEE,      _RIGHT_KNEE),
    "back_ankle"     : (_LEFT_ANKLE,     _RIGHT_ANKLE),
    "back_shoulder"  : (_LEFT_SHOULDER,  _RIGHT_SHOULDER),
    "back_wrist"     : (_LEFT_WRIST,     _RIGHT_WRIST),
}

FEATURE_COLUMNS = [
    "back_knee_flexion_angle",
    "front_knee_flexion_angle",
    "spine_hip_angle",
    "stance_width_normalized",
    "front_knee_ankle_alignment",
    "center_of_mass_torso_offset",
    "wrist_height_symmetry",
    "front_wrist_shoulder_vertical_offset",
    "back_wrist_shoulder_vertical_offset",
]

PROBABILITY_THRESHOLD = 0.6


def detect_is_left_facing(landmarks) -> bool:
    """
    Determine whether the subject is left-facing in the frame.
    Returns True if left-facing (left knee = front/bent leg).
    Uses the knee flexion angles to determine the front (bent) leg,
    making it extremely robust to head tilt or facing direction.
    """
    # Left knee landmarks: 23 (hip), 25 (knee), 27 (ankle)
    left_knee_angle = _angle_at_vertex_2d(
        _xy(landmarks, 23),
        _xy(landmarks, 25),
        _xy(landmarks, 27)
    )
    # Right knee landmarks: 24 (hip), 26 (knee), 28 (ankle)
    right_knee_angle = _angle_at_vertex_2d(
        _xy(landmarks, 24),
        _xy(landmarks, 26),
        _xy(landmarks, 28)
    )
    return left_knee_angle < right_knee_angle




def _resolve_joint(role_or_int, is_left: bool) -> int:
    if isinstance(role_or_int, str):
        right_idx, left_idx = _ORIENTATION_MAP[role_or_int]
        return left_idx if is_left else right_idx
    return int(role_or_int)


def resolve_joints(joints: list, is_left: bool) -> list[int]:
    return [_resolve_joint(j, is_left) for j in joints]


#  Low-level coordinate helpers 

def _xy(landmarks, idx: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=np.float64)


def _xyz(landmarks, idx: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def _x(landmarks, idx: int) -> float:
    return float(landmarks[idx].x)


def _y(landmarks, idx: int) -> float:
    return float(landmarks[idx].y)


#  Geometry primitives 

def _angle_at_vertex_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba     = a - b
    bc     = c - b
    cosine = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + EPSILON),
        -1.0, 1.0,
    )
    return float(np.degrees(np.arccos(cosine)))


def _angle_at_vertex_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba     = a - b
    bc     = c - b
    cosine = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + EPSILON),
        -1.0, 1.0,
    )
    return float(np.degrees(np.arccos(cosine)))


def _euclidean_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _point_to_line_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab    = b - a
    ap    = p - a
    cross = abs(ab[0] * ap[1] - ab[1] * ap[0])
    return cross / (np.linalg.norm(ab) + EPSILON)


#  Per-feature-type compute functions 

def _compute_joint_angle(landmarks, cfg: dict, is_left: bool) -> float:
    j = resolve_joints(cfg["joints"], is_left)
    if cfg.get("mode") == "2d":
        return _angle_at_vertex_2d(
            _xy(landmarks, j[0]),
            _xy(landmarks, j[1]),
            _xy(landmarks, j[2]),
        )
    return _angle_at_vertex_3d(
        _xyz(landmarks, j[0]),
        _xyz(landmarks, j[1]),
        _xyz(landmarks, j[2]),
    )


def _compute_spatial_distance(landmarks, cfg: dict, is_left: bool) -> float:
    j    = resolve_joints(cfg["joints"], is_left)
    dist = _euclidean_2d(_xy(landmarks, j[0]), _xy(landmarks, j[1]))
    if "normalization_factor" in cfg:
        nf    = resolve_joints(cfg["normalization_factor"], is_left)
        scale = _euclidean_2d(_xy(landmarks, nf[0]), _xy(landmarks, nf[1]))
        dist  = dist / (scale + EPSILON)
    return dist


def _compute_alignment_offset(landmarks, cfg: dict, is_left: bool) -> float:
    name    = cfg["name"]
    compute = cfg.get("compute", "").strip().lower()
    j       = resolve_joints(cfg["joints"], is_left)

    try:
        if compute == "lateral_delta":
            val = abs(_x(landmarks, j[0]) - _x(landmarks, j[1]))

        elif compute == "vertical_delta":
            val = abs(_y(landmarks, j[1]) - _y(landmarks, j[0]))

        elif compute == "midpoint_delta":
            mid_shoulder_x = (_x(landmarks, j[0]) + _x(landmarks, j[1])) / 2.0
            mid_hip_x      = (_x(landmarks, j[2]) + _x(landmarks, j[3])) / 2.0
            val = abs(mid_shoulder_x - mid_hip_x)

        elif compute == "perpendicular":
            val = _point_to_line_2d(
                _xy(landmarks, j[1]),
                _xy(landmarks, j[0]),
                _xy(landmarks, j[2]),
            )

        elif compute == "slope_delta":
            angle_01 = math.atan2(
                _y(landmarks, j[1]) - _y(landmarks, j[0]),
                _x(landmarks, j[1]) - _x(landmarks, j[0]),
            )
            angle_23 = math.atan2(
                _y(landmarks, j[3]) - _y(landmarks, j[2]),
                _x(landmarks, j[3]) - _x(landmarks, j[2]),
            )
            delta_rad = (angle_01 - angle_23 + math.pi) % (2 * math.pi) - math.pi
            val = math.degrees(abs(delta_rad))

        else:
            raise ValueError(
                f"Alignment offset '{name}' has unsupported compute type '{compute}'."
            )

    except Exception as exc:
        warnings.warn(f"Failed computing alignment offset '{name}': {exc}")
        return float("nan")

    if "normalization_factor" in cfg:
        nf    = resolve_joints(cfg["normalization_factor"], is_left)
        scale = _euclidean_2d(_xy(landmarks, nf[0]), _xy(landmarks, nf[1]))
        val   = val / (scale + EPSILON)

    return float(val)


def extract_features(landmarks, feat_config: dict, feature_columns: list, is_left: bool) -> Optional[np.ndarray]:
    feature_map: dict[str, float] = {}

    for cfg in feat_config.get("joint_angles", []):
        feature_map[cfg["name"]] = _compute_joint_angle(landmarks, cfg, is_left)

    for cfg in feat_config.get("spatial_distances", []):
        feature_map[cfg["name"]] = _compute_spatial_distance(landmarks, cfg, is_left)

    for cfg in feat_config.get("alignment_offsets", []):
        feature_map[cfg["name"]] = _compute_alignment_offset(landmarks, cfg, is_left)

    try:
        row = np.array([feature_map[col] for col in feature_columns], dtype=np.float64)
    except KeyError as e:
        raise KeyError(
            f"Feature {e} not computed. Check that the YAML config covers all required features."
        ) from e

    return None if np.any(np.isnan(row)) else row


#  Service 

class Warrior2PoseService:
    def __init__(
        self,
        artefacts_root: Path,
        yaml_path: Path,
        task_model_path: Optional[Path] = None,
        force_model: Optional[str] = None,
    ):
        self.pipeline      = None
        self.label_encoder = None
        self.feat_config   = {}
        self.yaml_feedback: dict[str, str] = {}
        self.has_proba     = False
        self._loaded       = False

        self._load(artefacts_root, yaml_path, force_model)

    def _load(self, artefacts_root: Path, yaml_path: Path, force_model: Optional[str]) -> None:
        if not yaml_path.exists():
            logger.error(f"Warrior 2 pose YAML not found: {yaml_path}")
            return

        with open(yaml_path, "r") as f:
            pose_yaml = yaml.safe_load(f)

        self.feat_config   = pose_yaml.get("features_config", {}).get("engineered_features", {})
        self.yaml_feedback = pose_yaml.get("feedback", {})

        if not self.feat_config:
            logger.error("No 'features_config.engineered_features' in Warrior 2 pose YAML.")
            return

        logger.info(f"Warrior 2 YAML loaded. Feedback classes: {list(self.yaml_feedback.keys())}")

        dir_models  = artefacts_root / "models"
        dir_results = artefacts_root / "results"

        if force_model is not None:
            selected = force_model
            logger.info(f"Warrior 2 pose model forced: {selected}")
        else:
            results_csv = dir_results / f"{POSE_NAME}_{EXPERIMENT_TAG}_experiment_results.csv"
            if not results_csv.exists():
                logger.error(f"Results CSV not found: {results_csv}")
                return
            df       = pd.read_csv(results_csv)
            best     = df.sort_values("test_f1_weighted", ascending=False).iloc[0]
            selected = best["model"]
            logger.info(
                f"Warrior 2 pose auto-selected: {selected} "
                f"(test_f1={best['test_f1_weighted']:.4f})"
            )

        prefix  = f"{POSE_NAME}_{EXPERIMENT_TAG}_{selected}"
        p_path  = dir_models / f"{prefix}_pipeline.joblib"
        le_path = dir_models / f"{prefix}_label_encoder.joblib"

        if not p_path.exists() or not le_path.exists():
            logger.error(
                f"Warrior 2 pose artefacts not found.\n"
                f"  Expected: {p_path}\n"
                f"            {le_path}"
            )
            return

        self.pipeline      = joblib.load(p_path)
        self.label_encoder = joblib.load(le_path)
        self.has_proba     = hasattr(self.pipeline, "predict_proba")
        self._loaded       = True

        logger.info(
            f"Warrior 2 pose pipeline loaded: {p_path.name} | "
            f"classes: {list(self.label_encoder.classes_)}"
        )

    def predict_from_landmarks(self, landmarks) -> dict:
        if not self._loaded:
            return _unavailable()

        is_left  = detect_is_left_facing(landmarks)

        logger.debug(f"Warrior2 orientation: is_left={is_left}, nose_x={landmarks[0].x:.3f}, mid_hip_x={(landmarks[23].x + landmarks[24].x)/2:.3f}")


        feat_vec = extract_features(landmarks, self.feat_config, FEATURE_COLUMNS, is_left)

        if feat_vec is None:
            return {
                "success":    True,
                "pose":       "warrior2",
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

        logger.debug(f"Warrior2 | class={pred_class} conf={confidence:.3f} → {display_class}")

        return {
            "success":    True,
            "pose":       "warrior2",
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
        "pose":       "warrior2",
        "prediction": "unknown",
        "confidence": 0.0,
        "is_correct": False,
        "feedback":   "Warrior 2 pose model not loaded.",
    }
