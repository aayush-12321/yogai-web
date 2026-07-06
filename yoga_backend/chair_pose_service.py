"""
chair_pose_service.py

ML inference service for Chair Pose (Utkatasana).

Receives a MediaPipe landmark list from YogaPoseDetector (already detected)
and runs the trained sklearn pipeline using YAML-driven feature engineering.

Chair Pose is recorded side-on (sagittal plane), so only the camera-facing
side is reliable.  All features use `camera_*` semantic roles that resolve
to the correct anatomical side via is_left_facing.

Orientation convention (matches chair_specific_fe.py / training notebook):
  right-facing (is_left=False): person faces RIGHT → LEFT side faces camera
                                 → camera_* roles resolve to LEFT landmarks
  left-facing  (is_left=True) : person faces LEFT  → RIGHT side faces camera
                                 → camera_* roles resolve to RIGHT landmarks

Detection strategy: nose + shoulder consensus vs. hip centre (NOT knee-flexion,
which is warrior2's approach).
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

EPSILON = 1e-6

POSE_NAME        = "chair_pose"
EXPERIMENT_TAG   = "full_class"
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

# Orientation map: semantic role → (right-facing idx, left-facing idx)
#
# right-facing (is_left=False): person faces right, LEFT side faces camera
#                                → use index 0 (LEFT landmark)
# left-facing  (is_left=True) : person faces left, RIGHT side faces camera
#                                → use index 1 (RIGHT landmark)
_ORIENTATION_MAP: dict[str, tuple[int, int]] = {
    "camera_shoulder": (_LEFT_SHOULDER, _RIGHT_SHOULDER),
    "camera_wrist":    (_LEFT_WRIST,    _RIGHT_WRIST),
    "camera_hip":      (_LEFT_HIP,      _RIGHT_HIP),
    "camera_knee":     (_LEFT_KNEE,     _RIGHT_KNEE),
    "camera_ankle":    (_LEFT_ANKLE,    _RIGHT_ANKLE),
}

# Must match column order used during training (chair_pose.yaml YAML order):
#   joint_angles → spatial_distances → alignment_offsets
FEATURE_COLUMNS = [
    "camera_knee_flexion_angle",
    "camera_torso_hip_knee_angle",
    "camera_shoulder_arm_angle",
    "camera_wrist_shoulder_vertical_delta",
    "camera_hip_knee_vertical_delta",
    "camera_shoulder_ankle_horizontal_offset",
    "camera_knee_ankle_horizontal_overhang",
]

PROBABILITY_THRESHOLD  = 0.6
VISIBILITY_THRESHOLD   = 0.5
DEFAULT_IS_LEFT_FACING = False   # dataset default: right-facing


# ── Orientation detection ─────────────────────────────────────────────────────

def detect_is_left_facing(landmarks) -> bool:
    """
    Determine whether the subject is left-facing in the frame.

    Strategy: nose + shoulder consensus vs. hip centre.
    (Matches chair_specific_fe.py — NOT the knee-flexion method used in warrior2.)

        nose.x > hip_center.x  AND  shoulder_center.x > hip_center.x
            → person faces RIGHT  → is_left_facing = False  (default)

        nose.x < hip_center.x  AND  shoulder_center.x < hip_center.x
            → person faces LEFT   → is_left_facing = True

    Falls back to DEFAULT_IS_LEFT_FACING (False) when signals disagree or
    any key landmark is below VISIBILITY_THRESHOLD.

    Returns:
        True  → left-facing  (RIGHT side of body faces camera)
        False → right-facing (LEFT  side of body faces camera, dataset default)
    """
    nose       = landmarks[0]
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_hip      = landmarks[23]
    r_hip      = landmarks[24]

    key_lms = [nose, l_shoulder, r_shoulder, l_hip, r_hip]
    if any(lm.visibility < VISIBILITY_THRESHOLD for lm in key_lms):
        return DEFAULT_IS_LEFT_FACING

    hip_center_x      = (l_hip.x      + r_hip.x)      / 2.0
    shoulder_center_x = (l_shoulder.x + r_shoulder.x) / 2.0

    nose_ahead     = nose.x            > hip_center_x
    shoulder_ahead = shoulder_center_x > hip_center_x

    if nose_ahead and shoulder_ahead:
        return False          # both agree → right-facing
    elif not nose_ahead and not shoulder_ahead:
        return True           # both agree → left-facing
    else:
        return DEFAULT_IS_LEFT_FACING   # signals disagree → use default


# ── Joint resolution ──────────────────────────────────────────────────────────

def _resolve_joint(role_or_int, is_left: bool) -> int:
    """Resolve a semantic camera_* role or raw int to a MediaPipe landmark index."""
    if isinstance(role_or_int, str):
        right_idx, left_idx = _ORIENTATION_MAP[role_or_int]
        return left_idx if is_left else right_idx
    return int(role_or_int)


def resolve_joints(joints: list, is_left: bool) -> list[int]:
    return [_resolve_joint(j, is_left) for j in joints]


# ── Low-level coordinate helpers ──────────────────────────────────────────────

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


# ── Geometry primitives ───────────────────────────────────────────────────────

def _angle_at_vertex_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba     = a - b
    bc     = c - b
    cosine = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + EPSILON),
        -1.0, 1.0,
    )
    return float(np.degrees(np.arccos(cosine)))


def _angle_at_vertex_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """2-D screen-plane angle (X/Y only).  Excludes Z to avoid depth-jitter."""
    ba     = a - b
    bc     = c - b
    cosine = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + EPSILON),
        -1.0, 1.0,
    )
    return float(np.degrees(np.arccos(cosine)))


def _euclidean_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ── Per-feature-type compute functions ────────────────────────────────────────

def _compute_joint_angle(landmarks, cfg: dict, is_left: bool) -> float:
    """Joint angle — 2d when cfg has `mode: '2d'`, 3d otherwise."""
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


def _compute_distance_or_offset(landmarks, cfg: dict, is_left: bool) -> float:
    """
    Compute a spatial-distance or alignment-offset feature.

    Chair Pose uses two compute types (matching chair_specific_fe.py):
        vertical_delta  — |y_b - y_a|           (wrist height, hip-knee drop)
        lateral_delta   — |x_a - x_b|           (shoulder-ankle, knee-ankle overhang)
    Falls back to 2-D Euclidean distance for unspecified compute types.

    All results are optionally normalised by torso length (camera_shoulder →
    camera_hip), which is fully visible and stable in side-view recordings.
    """
    name    = cfg["name"]
    compute = cfg.get("compute", "").strip().lower()
    j       = resolve_joints(cfg["joints"], is_left)

    try:
        if compute == "vertical_delta":
            # |y_b - y_a|  (MediaPipe Y increases downward)
            val = abs(_y(landmarks, j[1]) - _y(landmarks, j[0]))

        elif compute == "lateral_delta":
            # |x_a - x_b|  — horizontal separation on screen
            val = abs(_x(landmarks, j[0]) - _x(landmarks, j[1]))

        else:
            # Euclidean 2-D distance (fallback)
            val = _euclidean_2d(_xy(landmarks, j[0]), _xy(landmarks, j[1]))

    except Exception as exc:
        warnings.warn(f"Failed computing feature '{name}': {exc}")
        return float("nan")

    if "normalization_factor" in cfg:
        nf    = resolve_joints(cfg["normalization_factor"], is_left)
        scale = _euclidean_2d(_xy(landmarks, nf[0]), _xy(landmarks, nf[1]))
        val   = val / (scale + EPSILON)

    return float(val)


# ── Master feature extractor ──────────────────────────────────────────────────

def extract_features(
    landmarks,
    feat_config: dict,
    feature_columns: list,
    is_left: bool,
) -> Optional[np.ndarray]:
    """
    Compute all engineered features for a single Chair Pose frame.

    Args:
        landmarks      : pose_landmarks list from MediaPipe.
        feat_config    : the ``engineered_features`` block from the pose YAML.
        feature_columns: ordered feature names matching training order.
        is_left        : orientation flag for this frame.

    Returns:
        1-D numpy float64 array of shape (n_features,), or None if any value
        is NaN (pose landmarks not usable for inference).
    """
    feature_map: dict[str, float] = {}

    for cfg in feat_config.get("joint_angles", []):
        feature_map[cfg["name"]] = _compute_joint_angle(landmarks, cfg, is_left)

    # Both spatial_distances and alignment_offsets use the same compute helper
    # (vertical_delta / lateral_delta) — no separate alignment handler needed.
    for cfg in feat_config.get("spatial_distances", []):
        feature_map[cfg["name"]] = _compute_distance_or_offset(landmarks, cfg, is_left)

    for cfg in feat_config.get("alignment_offsets", []):
        feature_map[cfg["name"]] = _compute_distance_or_offset(landmarks, cfg, is_left)

    try:
        row = np.array([feature_map[col] for col in feature_columns], dtype=np.float64)
    except KeyError as e:
        raise KeyError(
            f"Feature {e} not computed. Check that the YAML config covers all required features."
        ) from e

    return None if np.any(np.isnan(row)) else row


# ── Service ───────────────────────────────────────────────────────────────────

class ChairPoseService:
    """
    Loads the chair pose ML pipeline once at startup.
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

    def _load(self, artefacts_root: Path, yaml_path: Path, force_model: Optional[str]) -> None:
        if not yaml_path.exists():
            logger.error(f"Chair pose YAML not found: {yaml_path}")
            return

        with open(yaml_path, "r") as f:
            pose_yaml = yaml.safe_load(f)

        self.feat_config   = pose_yaml.get("features_config", {}).get("engineered_features", {})
        self.yaml_feedback = pose_yaml.get("feedback", {})

        if not self.feat_config:
            logger.error("No 'features_config.engineered_features' in chair pose YAML.")
            return

        logger.info(f"Chair pose YAML loaded. Feedback classes: {list(self.yaml_feedback.keys())}")

        dir_models  = artefacts_root / "models"
        dir_results = artefacts_root / "results"

        if force_model is not None:
            selected = force_model
            logger.info(f"Chair pose model forced: {selected}")
        else:
            results_csv = dir_results / f"{POSE_NAME}_{EXPERIMENT_TAG}_experiment_results.csv"
            if not results_csv.exists():
                logger.error(f"Results CSV not found: {results_csv}")
                return
            df       = pd.read_csv(results_csv)
            best     = df.sort_values("test_f1_weighted", ascending=False).iloc[0]
            selected = best["model"]
            logger.info(
                f"Chair pose auto-selected: {selected} "
                f"(test_f1={best['test_f1_weighted']:.4f})"
            )

        prefix  = f"{POSE_NAME}_{EXPERIMENT_TAG}_{selected}"
        p_path  = dir_models / f"{prefix}_pipeline.joblib"
        le_path = dir_models / f"{prefix}_label_encoder.joblib"

        if not p_path.exists() or not le_path.exists():
            logger.error(
                f"Chair pose artefacts not found.\n"
                f"  Expected: {p_path}\n"
                f"            {le_path}"
            )
            return

        self.pipeline      = joblib.load(p_path)
        self.label_encoder = joblib.load(le_path)
        self.has_proba     = hasattr(self.pipeline, "predict_proba")
        self._loaded       = True

        logger.info(
            f"Chair pose pipeline loaded: {p_path.name} | "
            f"classes: {list(self.label_encoder.classes_)}"
        )

    def predict_from_landmarks(self, landmarks) -> dict:
        """
        Run feature extraction + ML inference on a MediaPipe landmark list.
        landmarks: list of NormalizedLandmark with .x .y .z .visibility
        """
        if not self._loaded:
            return _unavailable()

        is_left = detect_is_left_facing(landmarks)

        logger.debug(
            f"Chair orient: is_left={is_left}, "
            f"nose_x={landmarks[0].x:.3f}, "
            f"hip_center_x={(landmarks[23].x + landmarks[24].x) / 2:.3f}"
        )

        feat_vec = extract_features(landmarks, self.feat_config, FEATURE_COLUMNS, is_left)

        if feat_vec is None:
            return {
                "success":    True,
                "pose":       "chair",
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

        logger.debug(f"Chair | class={pred_class} conf={confidence:.3f} → {display_class}")

        return {
            "success":    True,
            "pose":       "chair",
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
        "pose":       "chair",
        "prediction": "unknown",
        "confidence": 0.0,
        "is_correct": False,
        "feedback":   "Chair pose model not loaded.",
    }
