# angle_utils.py

import numpy as np

def vector_from_points(p1, p2):
    return p2 - p1

def angle_between_vectors(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def compute_angle_from_landmarks(results_landmarks, p1_idx, vertex_idx, p2_idx):
    """
    Generic angle computation directly from MediaPipe landmark results.
    No DataFrame needed — works for any pose regardless of which landmarks were selected.
    
    p1_idx, vertex_idx, p2_idx: MediaPipe landmark indices
    Returns angle in degrees at the vertex point.
    """
    def get_point(idx):
        lm = results_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    p1     = get_point(p1_idx)
    vertex = get_point(vertex_idx)
    p2     = get_point(p2_idx)

    return angle_between_vectors(
        vector_from_points(p1, vertex),
        vector_from_points(p2, vertex)
    )

# ── DataFrame-based functions (used by pipeline during training/prediction) ──
# These require the landmark to exist in the DataFrame, so only call them
# when you know the pose includes those landmarks.

def compute_shoulder_angle(row, side="left"):
    hip      = row[[f"{side}_hip_x",      f"{side}_hip_y",      f"{side}_hip_z"]].values
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    elbow    = row[[f"{side}_elbow_x",    f"{side}_elbow_y",    f"{side}_elbow_z"]].values
    return angle_between_vectors(vector_from_points(hip, shoulder), vector_from_points(elbow, shoulder))

def compute_elbow_angle(row, side="left"):
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    elbow    = row[[f"{side}_elbow_x",    f"{side}_elbow_y",    f"{side}_elbow_z"]].values
    wrist    = row[[f"{side}_wrist_x",    f"{side}_wrist_y",    f"{side}_wrist_z"]].values
    return angle_between_vectors(vector_from_points(shoulder, elbow), vector_from_points(wrist, elbow))

def compute_hip_angle(row, side="left"):
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    hip      = row[[f"{side}_hip_x",      f"{side}_hip_y",      f"{side}_hip_z"]].values
    knee     = row[[f"{side}_knee_x",     f"{side}_knee_y",     f"{side}_knee_z"]].values
    return angle_between_vectors(vector_from_points(shoulder, hip), vector_from_points(knee, hip))

def compute_knee_angle(row, side="left"):
    hip   = row[[f"{side}_hip_x",   f"{side}_hip_y",   f"{side}_hip_z"]].values
    knee  = row[[f"{side}_knee_x",  f"{side}_knee_y",  f"{side}_knee_z"]].values
    ankle = row[[f"{side}_ankle_x", f"{side}_ankle_y", f"{side}_ankle_z"]].values
    return angle_between_vectors(vector_from_points(hip, knee), vector_from_points(ankle, knee))

# Add these new functions for warrior2
def compute_shoulder_line_angle(row, side="left"):
    if side == "left":
        opp_shoulder = row[["right_shoulder_x", "right_shoulder_y", "right_shoulder_z"]].values
    else:
        opp_shoulder = row[["left_shoulder_x", "left_shoulder_y", "left_shoulder_z"]].values
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    elbow    = row[[f"{side}_elbow_x",    f"{side}_elbow_y",    f"{side}_elbow_z"]].values
    return angle_between_vectors(vector_from_points(opp_shoulder, shoulder), vector_from_points(elbow, shoulder))

def compute_torso_angle(row, side="left"):
    hip      = row[[f"{side}_hip_x",      f"{side}_hip_y",      f"{side}_hip_z"]].values
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    nose     = row[["nose_x", "nose_y", "nose_z"]].values
    return angle_between_vectors(vector_from_points(hip, shoulder), vector_from_points(nose, shoulder))

# warrior2 elbow uses index finger instead of wrist
def compute_elbow_angle_warrior2(row, side="left"):
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    elbow    = row[[f"{side}_elbow_x",    f"{side}_elbow_y",    f"{side}_elbow_z"]].values
    index    = row[[f"{side}_index_x",    f"{side}_index_y",    f"{side}_index_z"]].values
    return angle_between_vectors(vector_from_points(shoulder, elbow), vector_from_points(index, elbow))

# warrior2 knee uses heel instead of ankle
def compute_knee_angle_warrior2(row, side="left"):
    hip  = row[[f"{side}_hip_x",  f"{side}_hip_y",  f"{side}_hip_z"]].values
    knee = row[[f"{side}_knee_x", f"{side}_knee_y", f"{side}_knee_z"]].values
    heel = row[[f"{side}_heel_x", f"{side}_heel_y", f"{side}_heel_z"]].values
    return angle_between_vectors(vector_from_points(hip, knee), vector_from_points(heel, knee))

def compute_all_angles(row, pose_type):
    if pose_type == "plank":
        angles = [
            compute_shoulder_angle(row, "left"),
            compute_elbow_angle(row, "left"),
            compute_hip_angle(row, "left"),
            compute_knee_angle(row, "left"),
            compute_shoulder_angle(row, "right"),
            compute_elbow_angle(row, "right"),
            compute_hip_angle(row, "right"),
            compute_knee_angle(row, "right"),
        ]
        columns = [
            "left_shoulder_angle", "left_elbow_angle", "left_hip_angle", "left_knee_angle",
            "right_shoulder_angle", "right_elbow_angle", "right_hip_angle", "right_knee_angle",
        ]
        return angles, columns

    # Add mountain, warrior2 here when ready
    if pose_type == "warrior2":
        angles = [
            compute_elbow_angle_warrior2(row, "left"),
            compute_elbow_angle_warrior2(row, "right"),
            compute_shoulder_line_angle(row, "left"),
            compute_shoulder_line_angle(row, "right"),
            compute_knee_angle_warrior2(row, "left"),
            compute_knee_angle_warrior2(row, "right"),
            compute_hip_angle(row, "left"),
            compute_hip_angle(row, "right"),
            compute_torso_angle(row, "left"),
            compute_torso_angle(row, "right"),
        ]
        columns = [
            "left_elbow_angle", "right_elbow_angle",
            "left_shoulder_line_angle", "right_shoulder_line_angle",
            "left_knee_angle", "right_knee_angle",
            "left_hip_angle", "right_hip_angle",
            "left_torso_angle", "right_torso_angle",
        ]
        return angles, columns
    return [], []


# ── Warrior2 
def _normalize(value, ideal, worst):
    """
    Normalize distance from ideal to 0-1 severity score.
    0 = perfect, 1 = worst case
    """
    diff_value = abs(value - ideal)
    diff_worst = abs(worst - ideal)
    if diff_worst == 0:
        return 0.0
    return round(min(diff_value / diff_worst, 1.0), 3)


def _correct_confidence(scores):
    """Average of (1 - severity) for all checks."""
    return round(sum(1 - s for s in scores) / len(scores), 3)


def classify_warrior2(row):
    """
    Rule-based classifier for Warrior 2 pose.
    Camera-distance independent — all measurements normalized by shoulder width.
    Returns: (predicted_label, confidence, feedback)
    """

    # ── Compute angles ────────────────────────────────────────
    left_knee  = compute_knee_angle_warrior2(row, "left")
    right_knee = compute_knee_angle_warrior2(row, "right")

    left_elbow  = compute_elbow_angle_warrior2(row, "left")
    right_elbow = compute_elbow_angle_warrior2(row, "right")

    left_shoulder_line  = compute_shoulder_line_angle(row, "left")
    right_shoulder_line = compute_shoulder_line_angle(row, "right")

    left_hip  = compute_hip_angle(row, "left")
    right_hip = compute_hip_angle(row, "right")

    left_torso  = compute_torso_angle(row, "left")
    right_torso = compute_torso_angle(row, "right")

    # ── Detect front leg (more bent = front) ─────────────────
    front_knee = min(left_knee, right_knee)

    # ── Averages ──────────────────────────────────────────────
    avg_elbow         = (left_elbow + right_elbow) / 2
    avg_shoulder_line = (left_shoulder_line + right_shoulder_line) / 2
    avg_torso         = (left_torso + right_torso) / 2
    avg_hip           = (left_hip + right_hip) / 2

    # ── Distance-independent stance width ─────────────────────
    # Normalize heel distance by shoulder width so camera distance doesn't matter
    shoulder_width  = abs(row["left_shoulder_x"] - row["right_shoulder_x"]) + 1e-6
    stance_width    = abs(row["left_heel_x"]     - row["right_heel_x"])
    relative_stance = stance_width / shoulder_width

    # Debug — remove after tuning
    print(f"front_knee={front_knee:.1f}, avg_elbow={avg_elbow:.1f}, "
          f"avg_shoulder_line={avg_shoulder_line:.1f}, avg_torso={avg_torso:.1f}, "
          f"avg_hip={avg_hip:.1f}, relative_stance={relative_stance:.2f}")

    # ── Rules ─────────────────────────────────────────────────
    # Format: (condition, label, severity, feedback)
    rules = [
        (
            front_knee > 120,
            "bent_front_knee",
            _normalize(front_knee, ideal=90, worst=160),
            "Bend your front knee more — aim for 90°."
        ),
        (
            avg_elbow < 100,          # ← was 150, now 100 based on real data
            "arms_bent",
            _normalize(avg_elbow, ideal=120, worst=60),  # ← ideal is 120 not 180
            "Straighten your arms fully."
        ),
        (
            avg_shoulder_line < 120,
            "arms_dropped",
            _normalize(avg_shoulder_line, ideal=180, worst=60),
            "Raise your arms to shoulder height — keep them parallel to the floor."
        ),
        (
            avg_torso < 90,           # ← was 100, slightly loosened
            "leaning_forward",
            _normalize(avg_torso, ideal=150, worst=60),  # ← ideal based on real data
            "Keep your torso upright — don't lean forward."
        ),
        (
            abs(avg_hip - 112) > 35,  # ← center changed from 105 to 112 based on real data
            "hips_rotated",
            _normalize(abs(avg_hip - 112), ideal=0, worst=45),
            "Square your hips to the side."
        ),
        (
            relative_stance < 1.5,
            "narrow_stance",
            _normalize(relative_stance, ideal=3.5, worst=1.5),  # ← ideal based on real data
            "Widen your stance for better stability."
        ),
    ]

    # ── Evaluate rules ────────────────────────────────────────
    triggered = [
        (label, severity, feedback)
        for condition, label, severity, feedback in rules
        if condition
    ]

    if not triggered:
        # All checks passed — compute correct confidence
        all_severities = [
            _normalize(front_knee,         ideal=90,  worst=160),
            _normalize(avg_elbow,          ideal=120, worst=60),   # ← fixed
            _normalize(avg_shoulder_line,  ideal=180, worst=60),
            _normalize(avg_torso,          ideal=150, worst=60),   # ← fixed
            _normalize(abs(avg_hip - 112), ideal=0,   worst=45),   # ← fixed
            _normalize(relative_stance,    ideal=3.5, worst=1.5),  # ← fixed
        ]
        confidence = _correct_confidence(all_severities)
        return "correct", confidence, "Great Warrior 2!"

    # Return the most severe mistake
    triggered.sort(key=lambda x: x[1], reverse=True)
    top_label, top_severity, top_feedback = triggered[0]
    return top_label, top_severity, top_feedback