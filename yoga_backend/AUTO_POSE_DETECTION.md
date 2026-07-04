# Auto Pose Detection — Implementation Notes

## What Was Added

A new **Automatic Pose Detection** page that detects which yoga pose the user is performing in real time, without the user needing to select a pose first. Existing manual session pages are **completely untouched**.

---

## New Files

| File | Purpose |
|------|---------|
| `yoga_backend/master_pose_service.py` | Loads the master pose classifier and the prediction stabilizer |
| `yoga/templates/yoga/auto_session.html` | New page — copy of `session.html` adapted for auto mode |

## Modified Files (Minimal Changes)

| File | Change |
|------|--------|
| `yoga_backend/pose_detector.py` | ~20 lines added: loads `MasterPoseClassifier` + `PosePredictionStabilizer`, handles `target_pose="auto"` in `predict_from_landmarks()` |
| `yoga_backend/views.py` | 4 lines changed: `matches_target` logic handles `target_pose="auto"` |
| `yoga/views.py` | Added `auto_session_view` at the bottom (6 lines) |
| `yoga/urls.py` | 1 new route: `path("auto-session/", ...)` |
| `yoga/templates/yoga/base.html` | 1 new nav link: "Auto Detect" |

---

## Architecture

```
Incoming Frame  (target_pose = "auto")
      │
      ▼
pose_detector.py — predict_from_landmarks()
      │
      ├── MasterPoseClassifier.predict_from_landmarks(landmarks)
      │       └─ Feature extraction from YAML config
      │       └─ Best model chosen from results CSV automatically
      │       └─ Returns: "mountain" | "plank" | "warrior2" | "unknown"
      │
      ├── PosePredictionStabilizer.push(raw_prediction)
      │       └─ Rolling window (size=5), majority voting
      │       └─ Tie-breaker: keeps previous stable prediction
      │       └─ Returns: stable pose name
      │
      ├── (routes to existing service as if target_pose was set manually)
      │       ├── MountainPoseService  → correction feedback
      │       ├── PlankPoseService     → correction feedback
      │       └── Warrior2PoseService → correction feedback
      │
      ▼
Same JSON response as manual mode (pose, prediction, confidence, feedback, is_correct)
      │
      ▼
auto_session.html — identical UI logic, dynamically updates "Detected Pose" label
```

---

## Prediction Stabilization Strategy

**Rolling Window + Majority Voting** (`PosePredictionStabilizer`, window=5 frames)

- Every raw prediction from the master classifier is pushed into a `deque(maxlen=5)`.
- The **mode** (most frequent prediction) is returned as the stable prediction.
- **Tie-breaking**: if the previous stable prediction is among tied candidates, it wins. This avoids unnecessary switching at the boundary between poses.
- Result: a 1–2 frame flip is fully absorbed; the UI only switches when 3+ of the last 5 frames agree on a new pose.

This keeps the experience smooth but still reacts in ~2–3 seconds when the user intentionally changes poses.

---

## Adding a New Pose in the Future

1. Train the new pose's error classifier and save its files under `yoga_backend/trained_models/<new_pose>_pose_files/`.
2. Create `yoga_backend/<new_pose>_pose_service.py` following the same pattern as `mountain_pose_service.py`.
3. In `pose_detector.py → _init_pose_services()`, add ~10 lines to instantiate and register the new service in `self._pose_services["<new_pose>"]`.
4. Re-train the master classifier to include the new class (or just update the YAML/model files).

No other files need to change.
