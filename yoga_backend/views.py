"""
views.py  (yoga_backend app)

API endpoints for yoga pose detection, session management, and user stats.

Endpoints
---------
POST   /api/yoga/detect/             PoseDetectionView   — live webcam frame
POST   /api/yoga/analyze_video/      VideoAnalysisView   — uploaded video file
POST   /api/yoga/session/start/      StartSessionView
POST   /api/yoga/session/end/        EndSessionView
GET    /api/yoga/sessions/           SessionHistoryView
GET    /api/yoga/user/stats/         UserStatsView
GET    /api/yoga/poses/              GetAvailablePosesView
GET    /api/yoga/model_status/       ModelStatusView

Bug fixes in this version:
  - VideoAnalysisView: cap.release() now in its own finally block so it always runs,
    even if the video landmarker throws mid-analysis.
  - VideoAnalysisView: MAX_ANALYZED_FRAMES now counts every sampled frame (not just
    frames where landmarks were detected), so the limit is predictable.
  - VideoAnalysisView: video_landmarker is explicitly closed in its own finally block
    to avoid resource leaks if an exception escapes the with-block.
"""

import logging
import os
import tempfile

import cv2
from django.db.models import Avg
from django.db.models.functions import Coalesce
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import PoseDetection, YogaSession
from .pose_detector import get_detector, build_video_landmarker

logger = logging.getLogger(__name__)

# Maximum video upload size: 100 MB
MAX_VIDEO_BYTES = 100 * 1024 * 1024


# ── Live webcam detection ────────────────────────────────────────────────────

class PoseDetectionView(APIView):
    """
    POST /api/yoga/detect/
    Body: { "frame": "<base64 jpeg>", "target_pose": "mountain", "session_id": 123 }
    """

    def post(self, request):
        frame_data  = request.data.get("frame")
        target_pose = request.data.get("target_pose")
        session_id  = request.data.get("session_id")

        if not frame_data:
            return Response({"error": "No frame data provided"}, status=status.HTTP_400_BAD_REQUEST)
        if not target_pose:
            return Response({"error": "target_pose is required"}, status=status.HTTP_400_BAD_REQUEST)

        detector = get_detector()
        result   = detector.process_frame(frame_data, target_pose)

        if not result.get("success"):
            return Response(result, status=status.HTTP_200_OK)

        result["matches_target"] = result.get("pose", "").lower() == target_pose.lower()

        if session_id:
            self._record_detection(session_id, result)

        return Response(result, status=status.HTTP_200_OK)

    def _record_detection(self, session_id: int, result: dict) -> None:
        try:
            session = YogaSession.objects.get(id=session_id)
            session.total_frames += 1
            if result.get("is_correct"):
                session.correct_frames += 1
            session.accuracy = (session.correct_frames / session.total_frames) * 100
            session.save()

            PoseDetection.objects.create(
                session=session,
                predicted_pose=result.get("prediction", "unknown"),
                confidence=result.get("confidence", 0.0),
                is_correct=result.get("is_correct", False),
            )
        except YogaSession.DoesNotExist:
            logger.warning(f"Session {session_id} not found — detection not recorded.")
        except Exception as exc:
            logger.error(f"Error recording detection for session {session_id}: {exc}")


# ── Video file analysis ──────────────────────────────────────────────────────

@method_decorator(csrf_exempt, name="dispatch")
class VideoAnalysisView(APIView):
    """
    POST /api/yoga/analyze_video/
    Form data: video=<file>, target_pose=<str>

    A FRESH VIDEO mode landmarker is created for each upload.
    This is mandatory — VIDEO mode is stateful and timestamps must be
    monotonically increasing within one landmarker instance. Sharing a
    single instance across requests causes timestamp ordering crashes.

    Frame processing matches the training script exactly:
      - VIDEO mode + detect_for_video() with frame-based monotonic timestamps
      - Every Nth frame (FRAME_SKIP)
      - No frame resizing (training videos were not resized)
    """

    permission_classes  = [AllowAny]
    # ~2 minutes at 30fps with FRAME_SKIP=5 (counts ALL sampled frames, not just detections)
    MAX_ANALYZED_FRAMES = 720
    FRAME_SKIP          = 5

    def post(self, request):
        video_file  = request.FILES.get("video")
        target_pose = request.POST.get("target_pose")

        if not video_file:
            return Response({"error": "Video file is required."}, status=400)
        if not target_pose:
            return Response({"error": "target_pose is required."}, status=400)

        if video_file.size > MAX_VIDEO_BYTES:
            mb = MAX_VIDEO_BYTES // (1024 * 1024)
            return Response(
                {"error": f"Video too large. Maximum allowed size is {mb} MB."},
                status=400,
            )

        detector = get_detector()

        try:
            result = self._analyze(video_file, target_pose, detector)
        except Exception as exc:
            logger.error(f"Video analysis failed: {exc}", exc_info=True)
            return Response({"error": str(exc)}, status=500)

        return Response(result, status=200)

    def _analyze(self, video_file, target_pose: str, detector) -> dict:
        results = {
            "target_pose":        target_pose,
            "total_frames":       0,
            "analyzed_frames":    0,
            "correct_frames":     0,
            "average_confidence": 0.0,
            "fps":                30.0,
            "predictions":        [],
        }
        confidences = []

        # Write the entire upload to disk before opening with OpenCV.
        # The file must be fully written and closed before cv2.VideoCapture
        # can read it — reading from a half-written file causes errors.
        tmp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                for chunk in video_file.chunks():
                    tmp_file.write(chunk)
            # File is now fully flushed and closed.

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                raise RuntimeError(
                    "OpenCV could not open the uploaded video. "
                    "Check that the file is a valid MP4/MOV/AVI."
                )

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            results["fps"] = round(fps, 3)

            # BUG FIX: Create a FRESH VIDEO mode landmarker for this video.
            # Timestamps start at 0 for every new video; a shared instance crashes
            # because its internal clock carries over from the previous video.
            video_landmarker = build_video_landmarker(detector._task_model_path)

            frame_index     = 0
            sampled_frames  = 0   # BUG FIX: count sampled frames, not just detections

            try:
                while cap.isOpened() and sampled_frames < self.MAX_ANALYZED_FRAMES:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_index += 1
                    results["total_frames"] += 1

                    if frame_index % self.FRAME_SKIP != 0:
                        continue

                    # BUG FIX: Increment sampled_frames for every frame we attempt,
                    # not just for frames where landmarks were detected.
                    # This makes MAX_ANALYZED_FRAMES a reliable upper bound.
                    sampled_frames += 1

                    # Monotonic timestamp in ms — matches training script exactly:
                    #   timestamp_ms = int((frame_count / fps) * 1000)
                    timestamp_ms = int((frame_index / fps) * 1000)

                    # No resize — training videos were processed at full resolution.
                    landmarks = detector.detect_landmarks_video(
                        frame, timestamp_ms, video_landmarker
                    )

                    if landmarks is None:
                        continue

                    prediction = detector.predict_from_landmarks(landmarks, target_pose)

                    if prediction.get("success"):
                        results["analyzed_frames"] += 1
                        conf = prediction["confidence"]
                        confidences.append(conf)
                        if prediction["is_correct"]:
                            results["correct_frames"] += 1

                        results["predictions"].append({
                            "frame":         frame_index,
                            "timestamp_sec": round(frame_index / fps, 3),
                            "prediction":    prediction.get("prediction", "unknown"),
                            "confidence":    round(conf, 4),
                            "is_correct":    prediction["is_correct"],
                            "feedback":      prediction.get("feedback", ""),
                        })

            finally:
                # BUG FIX: Always release landmarker and cap in separate finally blocks
                # so neither leaks if the other throws.
                try:
                    video_landmarker.close()
                except Exception as e:
                    logger.warning(f"Error closing video landmarker: {e}")
                cap.release()

        finally:
            # Always remove the temp file
            try:
                os.remove(temp_path)
            except OSError:
                pass

        if confidences:
            results["average_confidence"] = round(
                sum(confidences) / len(confidences), 3
            )

        analyzed = results["analyzed_frames"]
        results["accuracy"] = (
            round((results["correct_frames"] / analyzed) * 100, 2)
            if analyzed > 0
            else 0.0
        )

        return results


# ── Session management ───────────────────────────────────────────────────────

class StartSessionView(APIView):
    """
    POST /api/yoga/session/start/
    Body: { "pose": "mountain", "user_id": 1 }
    """

    def post(self, request):
        pose_name = request.data.get("pose", "plank")
        user_id   = request.data.get("user_id")

        session_data = {"pose_name": pose_name}

        if user_id:
            from django.contrib.auth.models import User
            try:
                session_data["user"] = User.objects.get(id=user_id)
            except User.DoesNotExist:
                pass

        try:
            session = YogaSession.objects.create(**session_data)
            return Response(
                {
                    "success":     True,
                    "session_id":  session.id,
                    "target_pose": pose_name,
                    "message":     "Session started successfully",
                    "started_at":  session.started_at.isoformat(),
                },
                status=status.HTTP_201_CREATED,
            )
        except Exception as exc:
            logger.error(f"Error starting session: {exc}")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class EndSessionView(APIView):
    """
    POST /api/yoga/session/end/
    Body: { "session_id": 123 }
    """

    def post(self, request):
        session_id = request.data.get("session_id")

        if not session_id:
            return Response({"error": "session_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session          = YogaSession.objects.get(id=session_id)
            session.ended_at = timezone.now()
            session.save()

            duration = (session.ended_at - session.started_at).total_seconds()

            return Response({
                "success":          True,
                "session_id":       session.id,
                "pose":             session.pose_name,
                "duration_seconds": duration,
                "total_frames":     session.total_frames,
                "correct_frames":   session.correct_frames,
                "accuracy":         session.accuracy,
                "message":          "Session ended successfully",
            })

        except YogaSession.DoesNotExist:
            return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            logger.error(f"Error ending session: {exc}")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ── Session history ──────────────────────────────────────────────────────────

class SessionHistoryView(APIView):
    """GET /api/yoga/sessions/?user_id=1"""

    def get(self, request):
        try:
            user_id  = request.query_params.get("user_id")
            sessions = YogaSession.objects.all()
            if user_id:
                sessions = sessions.filter(user_id=user_id)

            data = [
                {
                    "id":             s.id,
                    "pose":           s.pose_name,
                    "started_at":     s.started_at.isoformat(),
                    "ended_at":       s.ended_at.isoformat() if s.ended_at else None,
                    "total_frames":   s.total_frames,
                    "correct_frames": s.correct_frames,
                    "accuracy":       s.accuracy,
                }
                for s in sessions[:20]
            ]
            return Response({"sessions": data, "count": len(data)})

        except Exception as exc:
            logger.error(f"Error fetching session history: {exc}")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ── User stats ───────────────────────────────────────────────────────────────

class UserStatsView(APIView):
    """GET /api/yoga/user/stats/?user_id=1"""

    def get(self, request):
        try:
            user_id  = request.query_params.get("user_id")
            sessions = YogaSession.objects.all()
            if user_id:
                sessions = sessions.filter(user_id=user_id)

            completed = sessions.filter(ended_at__isnull=False)

            total_sessions = completed.count()
            avg_accuracy   = completed.aggregate(
                avg_acc=Coalesce(Avg("accuracy"), 0.0)
            )["avg_acc"]

            total_seconds = sum(
                (s.ended_at - s.started_at).total_seconds()
                for s in completed
                if s.started_at and s.ended_at
            )
            total_hours = round(total_seconds / 3600, 2)

            longest_streak          = self._longest_streak(completed)
            longest_continuous_time = self._longest_continuous_correct(completed)
            per_pose                = self._per_pose_breakdown(completed)

            recent = []
            for s in completed.order_by("-started_at")[:5]:
                dur = (s.ended_at - s.started_at).total_seconds() if s.ended_at else 0
                recent.append({
                    "id":       s.id,
                    "pose":     s.pose_name,
                    "date":     s.started_at.strftime("%Y-%m-%d"),
                    "time":     s.started_at.strftime("%H:%M"),
                    "duration": f"{int(dur // 60)}m {int(dur % 60)}s",
                    "accuracy": round(s.accuracy, 1),
                })

            return Response({
                "total_sessions":          total_sessions,
                "avg_accuracy":            round(avg_accuracy, 1),
                "total_hours":             total_hours,
                "longest_streak":          longest_streak,
                "longest_continuous_time": longest_continuous_time,
                "recent_sessions":         recent,
                "per_pose_breakdown":      per_pose,
            })

        except Exception as exc:
            logger.error(f"Error fetching user stats: {exc}")
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _longest_streak(self, sessions) -> int:
        dates = sorted({s.started_at.date() for s in sessions if s.started_at})
        if not dates:
            return 0
        longest = current = 1
        for i in range(1, len(dates)):
            if (dates[i] - dates[i - 1]).days == 1:
                current += 1
                longest  = max(longest, current)
            else:
                current = 1
        return longest

    def _longest_continuous_correct(self, sessions) -> float:
        max_time = 0.0
        for session in sessions:
            detections = session.detections.filter(is_correct=True).order_by("timestamp")
            if not detections.exists():
                continue
            streak_start = None
            streak_time  = 0.0
            for det in detections:
                if streak_start is None:
                    streak_start = det.timestamp
                else:
                    diff = (det.timestamp - streak_start).total_seconds()
                    if diff <= 2.0:
                        streak_time = diff
                    else:
                        max_time     = max(max_time, streak_time)
                        streak_start = det.timestamp
                        streak_time  = 0.0
            max_time = max(max_time, streak_time)
        return round(max_time, 1)

    def _per_pose_breakdown(self, sessions) -> dict:
        breakdown: dict = {}
        for s in sessions:
            pose = s.pose_name
            if pose not in breakdown:
                breakdown[pose] = {"sessions": 0, "accuracy_sum": 0.0}
            breakdown[pose]["sessions"]     += 1
            breakdown[pose]["accuracy_sum"] += s.accuracy
        return {
            pose: {
                "sessions":     v["sessions"],
                "avg_accuracy": round(v["accuracy_sum"] / v["sessions"], 1),
            }
            for pose, v in breakdown.items()
        }


# ── Utility endpoints ────────────────────────────────────────────────────────

class GetAvailablePosesView(APIView):
    """GET /api/yoga/poses/"""

    def get(self, request):
        detector = get_detector()
        return Response({"poses": detector.pose_classes, "count": len(detector.pose_classes)})


class ModelStatusView(APIView):
    """GET /api/yoga/model_status/  — debug"""

    def get(self, request):
        detector = get_detector()
        return Response({
            "ml_models":     {pose: detector.models.get(pose) is not None for pose in ["plank"]},
            "pose_services": {pose: svc.is_loaded for pose, svc in detector._pose_services.items()},
            "pose_classes":  detector.pose_classes,
        })