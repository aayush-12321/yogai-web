import os
import numpy as np
import pandas as pd
import yaml

# Configurations for your setup
YAML_FILE = "configs/poses/plank_pose.yaml"
RAW_CSV = "data/annotations/plank_pose/plank_train.csv"
OUTPUT_DIR = None
METADATA_COLS = ["source_id", "label", "frame_number"]
EPSILON = 1e-6


def _validate_inputs(yaml_path: str, raw_csv_path: str) -> None:
    """Confirm files exist before starting processing pipeline."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config not found: '{yaml_path}'")
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw landmarks CSV not found: '{raw_csv_path}'")


def _validate_landmark_columns(df: pd.DataFrame, required_names: set[str]) -> None:
    """Validates that your text-named coordinate columns exist in the dataframe."""
    missing = [
        f"{joint}_{axis}"
        for joint in required_names
        for axis in ("x", "y", "z")
        if f"{joint}_{axis}" not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Raw CSV is missing {len(missing)} expected landmark column(s): {missing}"
        )


def _collect_required_indices(feat_config: dict) -> set[str]:
    """Scans all config blocks (angles, distances, offsets) for named text joint keys."""
    indices: set[str] = set()
    for section in ("joint_angles", "spatial_distances", "alignment_offsets"):
        for cfg in feat_config.get(section, []):
            indices.update(cfg.get("joints", []))
            if "normalization_factor" in cfg:
                indices.update(cfg["normalization_factor"])
    return indices


def _xyz(df: pd.DataFrame, joint_name: str) -> np.ndarray:
    """Loads text-named columns into an (N, 3) numpy matrix layout."""
    return df[[f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"]].to_numpy(dtype=np.float64)


def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Computes continuous hinge 3D angles at vertex point B (in degrees)."""
    ba = a - b
    bc = c - b
    dot = np.einsum("ij,ij->i", ba, bc)
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    cosine = np.clip(dot / (norm_ba * norm_bc + EPSILON), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (N,) array of Euclidean distances between paired rows of a and b."""
    return np.linalg.norm(a - b, axis=1)


def _point_to_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute perpendicular distance from each point P to the line through A and B."""
    ab = b - a
    ap = p - a
    cross = np.cross(ap, ab)
    return np.linalg.norm(cross, axis=1) / (np.linalg.norm(ab, axis=1) + EPSILON)


def _compute_joint_angles(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """Loops through joint entries to compute vector mathematical features."""
    result = {}
    for cfg in configs:
        j = cfg["joints"]
        result[cfg["name"]] = _angle_at_vertex(_xyz(df, j[0]), _xyz(df, j[1]), _xyz(df, j[2]))
    return pd.DataFrame(result, index=df.index)


def _compute_spatial_distances(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """Vectorized computation of spatial distance features (supports vertical Y tracking)."""
    result = {}
    for cfg in configs:
        j = cfg["joints"]
        name = cfg["name"]

        # If it's your new vertical feature, look at literal Y displacement down the gravity plane
        if "vertical" in name.lower() or "y_axis" in name.lower():
            y_a = df[f"{j[0]}_y"].to_numpy(dtype=np.float64)
            y_b = df[f"{j[1]}_y"].to_numpy(dtype=np.float64)
            dist = np.abs(y_a - y_b)
        else:
            dist = _euclidean_distance(_xyz(df, j[0]), _xyz(df, j[1]))

        # Apply dynamic scale-normalization factor if declared in YAML
        if "normalization_factor" in cfg:
            nf = cfg["normalization_factor"]
            norm_dist = _euclidean_distance(_xyz(df, nf[0]), _xyz(df, nf[1]))
            dist = dist / (norm_dist + EPSILON)

        result[name] = dist
    return pd.DataFrame(result, index=df.index)


def _compute_alignment_offsets(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """Vectorized computation of alignment offset features (Supports 4-joint Plumb lines)."""
    result = {}
    for cfg in configs:
        j = cfg["joints"]
        name = cfg["name"]
        n = len(j)

        try:
            if n == 2:
                vals = np.abs(
                    df[f"{j[0]}_x"].to_numpy(dtype=np.float64) - df[f"{j[1]}_x"].to_numpy(dtype=np.float64)
                )
            elif n == 3:
                vals = _point_to_line_distance(_xyz(df, j[1]), _xyz(df, j[0]), _xyz(df, j[2]))
            elif n == 4 and "plumb" in name.lower():
                # Tracks spine chain line: deviation of hips and knees from the shoulder-to-ankle vector
                outer_a = _xyz(df, j[0])
                outer_b = _xyz(df, j[3])
                dev_1 = _point_to_line_distance(_xyz(df, j[1]), outer_a, outer_b)
                dev_2 = _point_to_line_distance(_xyz(df, j[2]), outer_a, outer_b)
                vals = dev_1 + dev_2
            elif n == 4:
                angle_01 = np.arctan2(
                    df[f"{j[1]}_y"].to_numpy(dtype=np.float64) - df[f"{j[0]}_y"].to_numpy(dtype=np.float64),
                    df[f"{j[1]}_x"].to_numpy(dtype=np.float64) - df[f"{j[0]}_x"].to_numpy(dtype=np.float64),
                )
                angle_23 = np.arctan2(
                    df[f"{j[3]}_y"].to_numpy(dtype=np.float64) - df[f"{j[2]}_y"].to_numpy(dtype=np.float64),
                    df[f"{j[3]}_x"].to_numpy(dtype=np.float64) - df[f"{j[2]}_x"].to_numpy(dtype=np.float64),
                )
                delta_rad = angle_01 - angle_23
                delta_rad = (delta_rad + np.pi) % (2 * np.pi) - np.pi
                vals = np.degrees(np.abs(delta_rad))
            else:
                raise ValueError(f"Alignment offset '{name}' has {n} joints; expected 2, 3, or 4.")
        except Exception as exc:
            print(f"Warning: failed computing alignment offset '{name}': {exc}")
            vals = np.full(len(df), np.nan)

        if "normalization_factor" in cfg:
            nf = cfg["normalization_factor"]
            scale = _euclidean_distance(_xyz(df, nf[0]), _xyz(df, nf[1]))
            vals = vals / (scale + EPSILON)

        result[name] = vals
    return pd.DataFrame(result, index=df.index)


def generate_engineered_features(yaml_path: str, raw_csv_path: str, output_dir: str | None = None) -> str:
    """Calculates geometric features directly from text-named landmark CSV tables."""
    _validate_inputs(yaml_path, raw_csv_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    feat_config = config.get("features_config", {}).get("engineered_features", {})
    if not feat_config:
        raise ValueError("Missing 'features_config.engineered_features' block in YAML.")

    df_raw = pd.read_csv(raw_csv_path)
    print(f"Loaded '{raw_csv_path}' -- {len(df_raw)} rows, {len(df_raw.columns)} columns.")

    # Check which tracking metadata components are actively sitting inside the source file
    present_meta = [col for col in METADATA_COLS if col in df_raw.columns]
    if "label" in df_raw.columns and "label" not in present_meta:
        present_meta.insert(0, "label")

    # Dynamic Validation
    required_joints = _collect_required_indices(feat_config)
    _validate_landmark_columns(df_raw, required_joints)

    # Core Calculation Pipeline
    feature_frames = [df_raw[present_meta].copy()]

    if feat_config.get("joint_angles"):
        feature_frames.append(_compute_joint_angles(df_raw, feat_config["joint_angles"]))

    if feat_config.get("spatial_distances"):
        feature_frames.append(_compute_spatial_distances(df_raw, feat_config["spatial_distances"]))

    if feat_config.get("alignment_offsets"):
        feature_frames.append(_compute_alignment_offsets(df_raw, feat_config["alignment_offsets"]))
    
    df_features = pd.concat(feature_frames, axis=1)
    
    # Path Resolution Logic
    base = os.path.basename(raw_csv_path)
    out_filename = base.replace("_raw.csv", "_features.csv") if "_raw.csv" in base else f"{os.path.splitext(base)[0]}_features.csv"
    save_dir = os.path.abspath(output_dir) if output_dir else os.path.dirname(raw_csv_path)
    output_path = os.path.join(save_dir, out_filename)
    
    df_features.to_csv(output_path, index=False)
    print(f"Saved features CSV ({df_features.shape[0]} rows x {df_features.shape[1]} cols) -> {output_path}")
    return output_path


if __name__ == "__main__":
    generate_engineered_features(YAML_FILE, RAW_CSV, OUTPUT_DIR)