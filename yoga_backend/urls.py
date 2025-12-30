from django.urls import path
from .views import (
    PoseDetectionView,
    StartSessionView,
    EndSessionView,
    GetAvailablePosesView,
    SessionHistoryView,
    UserStatsView,
    VideoAnalysisView,
    ModelStatusView
)

app_name = 'yoga_backend'

urlpatterns = [
    # Pose detection
    path('detect/', PoseDetectionView.as_view(), name='pose-detection'),
    
    # Video analysis
    path('analyze_video/', VideoAnalysisView.as_view(), name='video-analysis'),
    
    # Session management
    path('session/start/', StartSessionView.as_view(), name='start-session'),
    path('session/end/', EndSessionView.as_view(), name='end-session'),
    path('sessions/', SessionHistoryView.as_view(), name='session-history'),

    # User statistics
    path('user/stats/', UserStatsView.as_view(), name='user-stats'),
    
    # Utility
    path('poses/', GetAvailablePosesView.as_view(), name='available-poses'),
    path('model_status/', ModelStatusView.as_view(), name='model-status'),
]