from django.urls import path
from . import views
from .views import CustomLoginView,custom_logout
from django.contrib.auth.views import LogoutView

app_name = 'yoga'
urlpatterns = [
    path('', views.landing, name='landing'),
    path('poses/', views.poses_view, name='poses'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', CustomLoginView.as_view(), name='login'),
    # path('logout/', LogoutView.as_view(), name='logout'),
    path('logout/', custom_logout, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('profile/edit/', views.edit_profile_view, name='edit_profile'),
    path('add_practice/', views.add_practice, name='add_practice'),
    path('stats/', views.stats_view, name='stats'),
    path("session/", views.session_view, name="session"),
    path("session/<str:pose>/", views.session_view, name="session_with_pose"),
    path('api/practice_dates/', views.practice_dates_json, name='practice_dates_json'),
]
