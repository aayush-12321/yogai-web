from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from .pose_detector import get_detector
from .models import YogaSession, PoseDetection
import logging

from django.db.models import Avg, Sum, Count
from django.db.models.functions import Coalesce
from datetime import timedelta

logger = logging.getLogger(__name__)


class PoseDetectionView(APIView):
    """
    API endpoint to detect yoga pose from a video frame
    
    POST /api/yoga/detect/
    Body: {
        "frame": "base64_encoded_image",
        "target_pose": "plank",  # required
        "session_id": 123       # optional
    }
    """
    
    def post(self, request):
        try:
            # Get request data
            frame_data = request.data.get('frame')
            target_pose = request.data.get('target_pose')
            session_id = request.data.get('session_id')
            
            if not frame_data:
                return Response(
                    {'error': 'No frame data provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get detector and process frame
            detector = get_detector()
            result = detector.process_frame(frame_data, target_pose)
            
            if not result.get('success'):
                return Response(result, status=status.HTTP_200_OK)
            
            # Check if predicted pose matches target
            if target_pose:
                result['matches_target'] = (
                    result['pose'].lower() == target_pose.lower()
                )
            
            # Save detection to database if session exists
            if session_id:
                try:
                    session = YogaSession.objects.get(id=session_id)
                    session.total_frames += 1
                    if result.get('is_correct'):
                        session.correct_frames += 1
                    session.accuracy = (session.correct_frames / session.total_frames) * 100
                    session.save()
                    
                    # Create detection record
                    PoseDetection.objects.create(
                        session=session,
                        predicted_pose=result['pose'],
                        confidence=result['confidence'],
                        is_correct=result.get('is_correct', False)
                    )
                except YogaSession.DoesNotExist:
                    logger.warning(f"Session {session_id} not found")
            
            return Response(result, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in pose detection: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class StartSessionView(APIView):
    """
    API endpoint to start a new yoga session
    
    POST /api/yoga/session/start/
    Body: {
        "pose": "plank|mountain|warrior2",
        "user_id": 1  # optional
    }
    """
    
    def post(self, request):
        try:
            pose_name = request.data.get('pose', 'plank')
            user_id = request.data.get('user_id')
            
            # Create new session
            session_data = {
                'pose_name': pose_name,
            }
            
            if user_id:
                from django.contrib.auth.models import User
                try:
                    user = User.objects.get(id=user_id)
                    session_data['user'] = user
                except User.DoesNotExist:
                    pass
            
            session = YogaSession.objects.create(**session_data)
            
            return Response({
                'success': True,
                'session_id': session.id,
                'target_pose': pose_name,
                'message': 'Session started successfully',
                'started_at': session.started_at.isoformat()
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class EndSessionView(APIView):
    """
    API endpoint to end a yoga session
    
    POST /api/yoga/session/end/
    Body: {
        "session_id": 123
    }
    """
    
    def post(self, request):
        try:
            session_id = request.data.get('session_id')
            
            if not session_id:
                return Response(
                    {'error': 'session_id is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            session = YogaSession.objects.get(id=session_id)
            session.ended_at = timezone.now()
            session.save()
            
            duration = (session.ended_at - session.started_at).total_seconds()
            
            return Response({
                'success': True,
                'session_id': session.id,
                'pose': session.pose_name,
                'duration_seconds': duration,
                'total_frames': session.total_frames,
                'correct_frames': session.correct_frames,
                'accuracy': session.accuracy,
                'message': 'Session ended successfully'
            })
            
        except YogaSession.DoesNotExist:
            return Response(
                {'error': 'Session not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GetAvailablePosesView(APIView):
    """
    API endpoint to get list of available yoga poses
    
    GET /api/yoga/poses/
    """
    
    def get(self, request):
        detector = get_detector()
        return Response({
            'poses': detector.pose_classes,
            'count': len(detector.pose_classes)
        })


class SessionHistoryView(APIView):
    """
    API endpoint to get session history
    
    GET /api/yoga/sessions/?user_id=1
    """
    
    def get(self, request):
        try:
            user_id = request.query_params.get('user_id')
            
            sessions = YogaSession.objects.all()
            
            if user_id:
                sessions = sessions.filter(user_id=user_id)
            
            # Get recent sessions
            sessions = sessions[:20]
            
            data = []
            for session in sessions:
                data.append({
                    'id': session.id,
                    'pose': session.pose_name,
                    'started_at': session.started_at.isoformat(),
                    'ended_at': session.ended_at.isoformat() if session.ended_at else None,
                    'total_frames': session.total_frames,
                    'correct_frames': session.correct_frames,
                    'accuracy': session.accuracy
                })
            
            return Response({
                'sessions': data,
                'count': len(data)
            })
            
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
class UserStatsView(APIView):
    """
    API endpoint to get user statistics
    
    GET /api/yoga/user/stats/?user_id=1
    """
    
    def get(self, request):
        try:
            user_id = request.query_params.get('user_id')
            
            # Get all sessions (with or without user_id)
            if user_id:
                sessions = YogaSession.objects.filter(user_id=user_id)
            else:
                # For anonymous users, get all sessions
                sessions = YogaSession.objects.all()
            
            # Filter only completed sessions (those with ended_at)
            completed_sessions = sessions.filter(ended_at__isnull=False)
            
            # Calculate total sessions
            total_sessions = completed_sessions.count()
            
            # Calculate average accuracy
            avg_accuracy = completed_sessions.aggregate(
                avg_acc=Coalesce(Avg('accuracy'), 0.0)
            )['avg_acc']
            
            # Calculate total hours
            total_seconds = 0
            for session in completed_sessions:
                if session.started_at and session.ended_at:
                    duration = (session.ended_at - session.started_at).total_seconds()
                    total_seconds += duration
            
            total_hours = total_seconds / 3600  # Convert to hours
            
            # Calculate longest streak
            longest_streak = self.calculate_longest_streak(completed_sessions)
            
            # Calculate longest continuous correct pose time
            longest_continuous_time = self.calculate_longest_continuous_time(completed_sessions)
            
            # Get recent sessions
            recent_sessions = completed_sessions.order_by('-started_at')[:5]
            recent_sessions_data = []
            for session in recent_sessions:
                duration = (session.ended_at - session.started_at).total_seconds() if session.ended_at else 0
                recent_sessions_data.append({
                    'id': session.id,
                    'pose': session.pose_name,
                    'date': session.started_at.strftime('%Y-%m-%d'),
                    'time': session.started_at.strftime('%H:%M'),
                    'duration': f"{int(duration // 60)}m {int(duration % 60)}s",
                    'accuracy': round(session.accuracy, 1)
                })
            
            return Response({
                'total_sessions': total_sessions,
                'avg_accuracy': round(avg_accuracy, 1),
                'total_hours': round(total_hours, 2),
                'longest_streak': longest_streak,
                'longest_continuous_time': longest_continuous_time,
                'recent_sessions': recent_sessions_data
            })
            
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def calculate_longest_streak(self, sessions):
        """Calculate longest consecutive days streak"""
        if not sessions.exists():
            return 0
        
        # Get unique dates of sessions
        session_dates = set()
        for session in sessions:
            if session.started_at:
                session_dates.add(session.started_at.date())
        
        if not session_dates:
            return 0
        
        sorted_dates = sorted(session_dates)
        longest_streak = 1
        current_streak = 1
        
        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i] - sorted_dates[i-1]).days == 1:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 1
        
        return longest_streak
    
    def calculate_longest_continuous_time(self, sessions):
        """Calculate longest continuous time spent in correct pose"""
        if not sessions.exists():
            return 0
        
        max_continuous_time = 0
        
        for session in sessions:
            # Get all detections for this session
            detections = session.detections.filter(is_correct=True).order_by('timestamp')
            
            if not detections.exists():
                continue
            
            # Calculate continuous correct pose periods
            current_streak_start = None
            current_streak_time = 0
            
            for detection in detections:
                if current_streak_start is None:
                    current_streak_start = detection.timestamp
                    current_streak_time = 0
                else:
                    # Check if this detection is within 2 seconds of the previous one
                    time_diff = (detection.timestamp - current_streak_start).total_seconds()
                    if time_diff <= 2.0:  # Assuming detections are ~0.5s apart
                        current_streak_time = time_diff
                    else:
                        # Streak broken, reset
                        max_continuous_time = max(max_continuous_time, current_streak_time)
                        current_streak_start = detection.timestamp
                        current_streak_time = 0
            
            # Check the last streak
            if current_streak_start is not None:
                max_continuous_time = max(max_continuous_time, current_streak_time)
        
        return round(max_continuous_time, 1)  # Return in seconds


class ModelStatusView(APIView):
    """
    Debug endpoint to check model loading status
    """
    
    def get(self, request):
        from .pose_detector import get_detector
        detector = get_detector()
        
        model_status = {}
        for pose in detector.pose_classes:
            model_status[pose] = detector.models.get(pose) is not None
        
        return Response({
            'models_loaded': model_status,
            'pose_classes': detector.pose_classes,
            'total_models': len([m for m in detector.models.values() if m is not None])
        })


# class VideoAnalysisView(APIView):
#     """
#     API endpoint to analyze uploaded video for yoga pose detection
    
#     POST /api/yoga/analyze_video/
#     Body: {
#         "video": file,
#         "target_pose": "plank|mountain|warrior2",
#         "user_id": 1  # optional
#     }
#     """
#     def post(self, request):
#         try:
#             # Get uploaded video file
#             video_file = request.FILES.get('video')
#             target_pose = request.POST.get('target_pose') or request.data.get('target_pose')
#             user_id = request.POST.get('user_id') or request.data.get('user_id')
            
#             logger.info(f"Video analysis request: video_file={video_file}, target_pose={target_pose}, user_id={user_id}")
            
#             if not video_file:
#                 return Response(
#                     {'error': 'No video file provided'},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )
            
#             if not target_pose:
#                 return Response(
#                     {'error': 'target_pose is required'},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )
            
#             # Process the video
#             detector = get_detector()
#             results = self._process_video(video_file, target_pose, user_id)
            
#             logger.info(f"Video analysis completed: {results}")
#             return Response(results, status=status.HTTP_200_OK)
            
#         except Exception as e:
#             logger.error(f"Error in video analysis: {e}")
#             return Response(
#                 {'error': str(e)},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )
    
#     def _process_video(self, video_file, target_pose, user_id):
#         """Process video file and return analysis results"""
#         import tempfile
#         import cv2
        
#         logger.info(f"Processing video: {video_file.name}, size: {video_file.size}")
        
#         results = {
#             'target_pose': target_pose,
#             'total_frames': 0,
#             'analyzed_frames': 0,
#             'correct_frames': 0,
#             'average_confidence': 0.0,
#             'predictions': []
#         }
        
#         confidences = []
        
#         try:
#             # Save uploaded file to temporary location
#             with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
#                 for chunk in video_file.chunks():
#                     temp_file.write(chunk)
#                 temp_path = temp_file.name
            
#             # Open video with OpenCV
#             cap = cv2.VideoCapture(temp_path)
            
#             if not cap.isOpened():
#                 logger.error(f"Could not open video file: {temp_path}")
#                 return {'error': 'Could not open video file'}
            
#             logger.info(f"Video opened successfully: {temp_path}")
            
#             frame_count = 0
#             analyzed_count = 0
#             max_frames = 100  # Limit to 100 frames max to avoid excessive processing
            
#             while analyzed_count < max_frames:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 frame_count += 1
                
#                 # Process every 5th frame to balance speed and coverage
#                 if frame_count % 5 == 0:
#                     try:
#                         logger.info(f"Processing frame {frame_count}")
#                         # Convert frame to base64 for processing
#                         import base64
#                         _, buffer = cv2.imencode('.jpg', frame)
#                         frame_data = base64.b64encode(buffer).decode('utf-8')
#                         frame_data = f'data:image/jpeg;base64,{frame_data}'
                        
#                         # Detect pose
#                         result = detector.process_frame(frame_data, target_pose)
                        
#                         if result.get('success'):
#                             analyzed_count += 1
#                             prediction = {
#                                 'frame': frame_count,
#                                 'pose': result.get('pose'),
#                                 'confidence': result.get('confidence'),
#                                 'is_correct': result.get('is_correct'),
#                                 'timestamp': frame_count / 30.0  # Assuming 30fps
#                             }
#                             results['predictions'].append(prediction)
                            
#                             if result.get('is_correct'):
#                                 results['correct_frames'] += 1
                            
#                             confidences.append(result.get('confidence', 0))
                            
#                     except Exception as e:
#                         logger.error(f"Error processing frame {frame_count}: {e}")
#                         continue
            
#             cap.release()
            
#             # Calculate statistics
#             results['total_frames'] = frame_count
#             results['analyzed_frames'] = analyzed_count
            
#             if confidences:
#                 results['average_confidence'] = sum(confidences) / len(confidences)
            
#             if analyzed_count > 0:
#                 results['accuracy'] = (results['correct_frames'] / analyzed_count) * 100
#             else:
#                 results['accuracy'] = 0
            
#             # Create session record if user provided
#             if user_id and analyzed_count > 0:
#                 try:
#                     from django.contrib.auth.models import User
#                     user = User.objects.get(id=user_id)
                    
#                     # Create session
#                     session = YogaSession.objects.create(
#                         user=user,
#                         pose_name=target_pose,
#                         total_frames=analyzed_count,
#                         correct_frames=results['correct_frames'],
#                         accuracy=results['accuracy']
#                     )
                    
#                     # Create detection records
#                     for pred in results['predictions']:
#                         PoseDetection.objects.create(
#                             session=session,
#                             predicted_pose=pred['pose'],
#                             confidence=pred['confidence'],
#                             is_correct=pred['is_correct']
#                         )
                    
#                     results['session_id'] = session.id
                    
#                 except User.DoesNotExist:
#                     logger.warning(f"User {user_id} not found")
#                 except Exception as e:
#                     logger.error(f"Error creating session: {e}")
            
#             # Clean up temp file
#             import os
#             os.unlink(temp_path)
            
#         except Exception as e:
#             logger.error(f"Error processing video: {e}")
#             return {'error': str(e)}
        
#         return results











###################################################################################


# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from django.utils import timezone
# from .pose_detector import get_detector
# from .models import YogaSession, PoseDetection
# import logging
import cv2
import tempfile
import base64
import os

# logger = logging.getLogger(__name__)


class VideoAnalysisView(APIView):
    """
    Analyze uploaded video and return pose predictions + probabilities
    """

    def post(self, request):
        video_file = request.FILES.get("video")
        target_pose = request.POST.get("target_pose")

        if not video_file:
            return Response({"error": "Video file is required"}, status=400)

        if not target_pose:
            return Response({"error": "target_pose is required"}, status=400)

        detector = get_detector()
        result = self.process_video(video_file, target_pose, detector)

        return Response(result, status=200)

    def process_video(self, video_file, target_pose, detector):
        """
        Main video processing logic
        """

        results = {
            "target_pose": target_pose,
            "total_frames": 0,
            "processed_frames": 0,
            "correct_frames": 0,
            "average_confidence": 0,
            "predictions": []
        }

        confidences = []

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            for chunk in video_file.chunks():
                temp.write(chunk)
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return {"error": "Unable to open video"}

        frame_index = 0
        MAX_FRAMES = 100          # safety limit
        FRAME_SKIP = 5            # process every 5th frame

        while cap.isOpened() and results["processed_frames"] < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            results["total_frames"] += 1

            if frame_index % FRAME_SKIP != 0:
                continue

            # Convert frame → base64
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode()
            frame_b64 = f"data:image/jpeg;base64,{frame_b64}"

            prediction = detector.process_frame(frame_b64, target_pose)

            if prediction.get("success"):
                results["processed_frames"] += 1

                results["predictions"].append({
                    "frame": frame_index,
                    "pose": prediction["pose"],
                    "confidence": prediction["confidence"],
                    "is_correct": prediction["is_correct"]
                })

                confidences.append(prediction["confidence"])

                if prediction["is_correct"]:
                    results["correct_frames"] += 1

        cap.release()
        os.remove(temp_path)

        # Statistics
        if confidences:
            results["average_confidence"] = round(sum(confidences) / len(confidences), 3)

        if results["processed_frames"] > 0:
            results["accuracy"] = round(
                (results["correct_frames"] / results["processed_frames"]) * 100, 2
            )
        else:
            results["accuracy"] = 0

        return results
