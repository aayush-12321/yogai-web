import numpy as np

def vector_from_points(p1, p2):
    return p2 - p1

def angle_between_vectors(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

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

def compute_all_angles(row, pose_type):
    """
    Compute angles for a given pose type.
    row: pandas Series with landmark columns
    Returns: (angles list, angle column names list)
    """
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

    # 👇 Easily add mountain, warrior2 here later
    # elif pose_type == "mountain":
    #     ...

    return [], []