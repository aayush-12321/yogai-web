"""
master_pose_service.py

ML inference service for the Master Pose Classifier.
Predicts WHICH pose the user is performing (mountain, plank, or warrior2).

Feature engineering is an exact per-sample port of delete.py (master_model_fe.py).
All features are 2D (x, y only). Key rules:
  - vector_incline_angle: arctan2(|dx|, |dy|)  -> 0 = vertical, 90 = horizontal
  - vertical_delta:       |y[joints[1]] - y[joints[0]]|
  - perpendicular:        2D point-to-line distance via cross product
"""

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

EPSILON = 1e-6

POSE_NAME      = "master_pose_master"
EXPERIMENT_TAG = "3class"

# Feature order must exactly match the column order produced by delete.py:
#   joint_angles (YAML order) -> spatial_distances (YAML order) -> alignment_offsets (YAML order)
FEATURE_COLUMNS = [
    # joint_angles
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    # spatial_distances
    "stance_width_normalized",
    "wrist_span_normalized",
    # alignment_offsets
    "torso_incline_angle",
    "leg_incline_angle",
    "left_arm_elevation",
    "right_arm_elevation",
    "left_hip_line_deviation",
    "right_hip_line_deviation",
]

PROBABILITY_THRESHOLD = 0.0


# ─── 2-D helpers (exact per-sample equivalents of delete.py batch helpers) ───

def _xy(landmarks, idx: int) -> np.ndarray:
    """Return 2-D (x, y) for one MediaPipe landmark."""
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=np.float64)


def _angle_at_vertex_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    2D angle at vertex B, in degrees.
    Port of delete.py _angle_at_vertex_2d for a single sample.
    """
    ba = a - b
    bc = c - b
    dot = float(np.dot(ba, bc))
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    cosine = np.clip(dot / (norm_ba * norm_bc + EPSILON), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _euclidean_2d(a: np.ndarray, b: np.ndarray) -> float:
    """2D Euclidean distance. Port of delete.py _euclidean_distance_2d."""
    return float(np.linalg.norm(a - b))


def _point_to_line_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    2D perpendicular distance from point p to line through a and b.
    Port of delete.py _point_to_line_distance_2d (cross product method).
    """
    ab = b - a
    ap = p - a
    cross = abs(ab[0] * ap[1] - ab[1] * ap[0])
    return float(cross / (np.linalg.norm(ab) + EPSILON))


# ─── Per-feature-type computation ────────────────────────────────────────────

def _joint_angle(landmarks, cfg: dict) -> float:
    """Port of delete.py _compute_joint_angles for a single sample."""
    j = cfg["joints"]
    return _angle_at_vertex_2d(_xy(landmarks, j[0]), _xy(landmarks, j[1]), _xy(landmarks, j[2]))


def _spatial_distance(landmarks, cfg: dict) -> float:
    """Port of delete.py _compute_spatial_distances for a single sample."""
    j    = cfg["joints"]
    dist = _euclidean_2d(_xy(landmarks, j[0]), _xy(landmarks, j[1]))
    if "normalization_factor" in cfg:
        nf    = cfg["normalization_factor"]
        scale = _euclidean_2d(_xy(landmarks, nf[0]), _xy(landmarks, nf[1]))
        dist  = dist / (scale + EPSILON)
    return dist


def _alignment_offset(landmarks, cfg: dict) -> float:
    """
    Port of delete.py _compute_alignment_offsets for a single sample.

    Supported compute types (same semantics as delete.py):
        vertical_delta       |y[joints[1]] - y[joints[0]]|
        perpendicular        2D distance of joints[1] from line(joints[0], joints[2])
        vector_incline_angle arctan2(|dx|, |dy|) — 0 = vertical, 90 = horizontal
    """
    name    = cfg["name"]
    compute = cfg.get("compute", "").strip().lower()
    joints  = cfg["joints"]

    if compute == "vertical_delta":
        # |y_joints[1] - y_joints[0]|  (matches delete.py line 298)
        val = abs(landmarks[joints[1]].y - landmarks[joints[0]].y)

    elif compute == "perpendicular":
        # perpendicular distance of joints[1] from line(joints[0], joints[2])
        # (matches delete.py lines 301-303)
        val = _point_to_line_2d(
            _xy(landmarks, joints[1]),
            _xy(landmarks, joints[0]),
            _xy(landmarks, joints[2]),
        )

    elif compute == "vector_incline_angle":
        # (matches delete.py lines 306-318)
        if len(joints) == 2:
            p0 = _xy(landmarks, joints[0])
            p1 = _xy(landmarks, joints[1])
        elif len(joints) == 4:
            p0 = (_xy(landmarks, joints[0]) + _xy(landmarks, joints[1])) / 2.0
            p1 = (_xy(landmarks, joints[2]) + _xy(landmarks, joints[3])) / 2.0
        else:
            raise ValueError(
                f"'vector_incline_angle' feature '{name}' needs 2 or 4 joints, "
                f"got {len(joints)}."
            )
        delta = p1 - p0
        # arctan2(|dx|, |dy|): 0 deg when vertical, 90 deg when horizontal
        val = float(np.degrees(np.arctan2(abs(delta[0]), abs(delta[1]) + EPSILON)))

    else:
        raise ValueError(
            f"Alignment offset '{name}' has unsupported compute type '{compute}'. "
            f"Expected one of: vertical_delta, perpendicular, vector_incline_angle."
        )

    if "normalization_factor" in cfg:
        nf    = cfg["normalization_factor"]
        scale = _euclidean_2d(_xy(landmarks, nf[0]), _xy(landmarks, nf[1]))
        val   = val / (scale + EPSILON)

    return float(val)


# ─── Top-level extraction ─────────────────────────────────────────────────────

def extract_features(landmarks, feat_config: dict) -> Optional[np.ndarray]:
    """
    Compute all master-model features for one frame from a MediaPipe landmark list.

    Column order: joint_angles → spatial_distances → alignment_offsets (YAML order).
    This exactly matches the order produced by delete.py's generate_master_features().

    Returns a 1-D float64 array matching FEATURE_COLUMNS, or None if any value is
    non-finite (NaN / Inf).
    """
    fmap: dict[str, float] = {}

    for cfg in feat_config.get("joint_angles", []):
        fmap[cfg["name"]] = _joint_angle(landmarks, cfg)

    for cfg in feat_config.get("spatial_distances", []):
        fmap[cfg["name"]] = _spatial_distance(landmarks, cfg)

    for cfg in feat_config.get("alignment_offsets", []):
        fmap[cfg["name"]] = _alignment_offset(landmarks, cfg)

    try:
        row = np.array([fmap[col] for col in FEATURE_COLUMNS], dtype=np.float64)
    except KeyError as e:
        raise KeyError(
            f"Feature {e} not computed. Check that the YAML covers all required features."
        ) from e

    return None if not np.all(np.isfinite(row)) else row


# ─── Stabilizer ──────────────────────────────────────────────────────────────

class PosePredictionStabilizer:
    """
    Rolling window + majority voting.
    Prevents rapid pose-label oscillation without adding significant latency.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.last_stable: Optional[str] = None

    def push(self, pose: str) -> str:
        if pose and pose != "unknown":
            self.history.append(pose)

        if not self.history:
            return "unknown"

        counts: dict[str, int] = {}
        for p in self.history:
            counts[p] = counts.get(p, 0) + 1

        max_count = max(counts.values())
        best      = [p for p, c in counts.items() if c == max_count]

        # Tie-breaker: keep last stable if still in the tie group
        if self.last_stable in best:
            return self.last_stable

        self.last_stable = best[0]
        return self.last_stable

    def reset(self) -> None:
        self.history.clear()
        self.last_stable = None


# ─── Classifier ──────────────────────────────────────────────────────────────

class MasterPoseClassifier:
    """
    Loads the best master pose ML pipeline at startup (chosen by test_f1_weighted).
    predict_from_landmarks() returns a lowercase pose name or 'unknown'.
    """

    def __init__(
        self,
        artefacts_root: Path,
        yaml_path: Path,
        force_model: Optional[str] = None,
    ):
        self.pipeline:      Optional[object] = None
        self.label_encoder: Optional[object] = None
        self.feat_config:   dict             = {}
        self.has_proba:     bool             = False
        self._loaded:       bool             = False

        self._load(artefacts_root, yaml_path, force_model)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load(self, artefacts_root: Path, yaml_path: Path, force_model: Optional[str]) -> None:
        if not yaml_path.exists():
            logger.error(f"Master pose YAML not found: {yaml_path}")
            return

        with open(yaml_path, "r") as f:
            pose_yaml = yaml.safe_load(f)

        self.feat_config = pose_yaml.get("features_config", {}).get("engineered_features", {})
        if not self.feat_config:
            logger.error("No 'features_config.engineered_features' in master pose YAML.")
            return

        dir_models  = artefacts_root / "models"
        dir_results = artefacts_root / "results"

        if force_model is not None:
            selected = force_model
            logger.info(f"Master pose model forced: {selected}")
        else:
            results_csv = dir_results / f"{POSE_NAME}_{EXPERIMENT_TAG}_experiment_results.csv"
            if not results_csv.exists():
                logger.error(f"Results CSV not found: {results_csv}")
                return
            df       = pd.read_csv(results_csv)
            best_row = df.sort_values("test_f1_weighted", ascending=False).iloc[0]
            selected = best_row["model"]
            logger.info(
                f"Master pose auto-selected: {selected} "
                f"(test_f1={best_row['test_f1_weighted']:.4f})"
            )

        prefix  = f"{POSE_NAME}_{EXPERIMENT_TAG}_{selected}"
        p_path  = dir_models / f"{prefix}_pipeline.joblib"
        le_path = dir_models / f"{prefix}_label_encoder.joblib"

        if not p_path.exists() or not le_path.exists():
            logger.error(
                f"Master pose artefacts not found.\n"
                f"  Expected: {p_path}\n"
                f"            {le_path}"
            )
            return

        self.pipeline      = joblib.load(p_path)
        self.label_encoder = joblib.load(le_path)
        self.has_proba     = hasattr(self.pipeline, "predict_proba")
        self._loaded       = True

        logger.info(
            f"Master pose pipeline loaded: {p_path.name} | "
            f"classes: {list(self.label_encoder.classes_)}"
        )

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_from_landmarks(self, landmarks) -> str:
        """
        Returns the predicted pose name ('mountain', 'plank', 'warrior2') or 'unknown'.
        """
        if not self._loaded:
            return "unknown"

        try:
            feat_vec = extract_features(landmarks, self.feat_config)
        except Exception as exc:
            logger.warning(f"Master feature extraction failed: {exc}")
            return "unknown"

        if feat_vec is None:
            return "unknown"

        X          = feat_vec.reshape(1, -1)
        pred_enc   = self.pipeline.predict(X)[0]
        pred_class = self.label_encoder.inverse_transform([pred_enc])[0]

        if self.has_proba:
            proba      = self.pipeline.predict_proba(X)[0]
            confidence = float(proba[pred_enc])
        else:
            confidence = 1.0

        return pred_class.lower() if confidence >= PROBABILITY_THRESHOLD else "unknown"

    @property
    def is_loaded(self) -> bool:
        return self._loaded
