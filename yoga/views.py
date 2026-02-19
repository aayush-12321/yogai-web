from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate , logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .forms import SignUpForm, ProfileForm, PracticeForm
from .models import Pose, Practice, Profile
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.utils import timezone
from datetime import timedelta
import calendar
from datetime import date, datetime
import requests


def landing(request):
    top_poses = Pose.objects.all()[:4]
    return render(request, 'yoga/landing.html', {'top_poses': top_poses})

def poses_view(request):
    poses = Pose.objects.all()[:8]  # show first 8
    
    # Add pose_slug to each pose for URL generation
    pose_name_mapping = {
        'plank': 'plank',
        'mountain': 'mountain', 
        'warrior2': 'warrior2',
        'warrior ii': 'warrior2',
        'warrior': 'warrior2',
    }
    
    for pose in poses:
        pose_title_lower = pose.title.lower()
        # Try to find a matching pose name
        pose.pose_slug = None
        for key in pose_name_mapping:
            if key in pose_title_lower:
                pose.pose_slug = pose_name_mapping[key]
                break
        # If no match found, try to create slug from title
        if not pose.pose_slug:
            pose.pose_slug = pose_title_lower.replace(' pose', '').replace(' ', '')
        
        # Ensure we have a valid slug
        if pose.pose_slug not in ['plank', 'mountain', 'warrior2']:
            pose.pose_slug = 'plank'  # default fallback
    
    return render(request, 'yoga/poses.html', {'poses': poses})



# def signup_view(request):
#     if request.method == 'POST':
#         form = SignUpForm(request.POST, request.FILES)
#         if form.is_valid():
#             user = form.save()
#             # create profile with additional fields
#             Profile.objects.create(
#                 user=user,
#                 date_of_birth=form.cleaned_data.get('date_of_birth'),
#                 medical_condition=form.cleaned_data.get('medical_condition'),
#                 avatar=form.cleaned_data.get('avatar')
#             )
#             login(request, user)
#             return redirect('yoga:profile')
#     else:
#         form = SignUpForm()
#     return render(request, 'yoga/signup.html', {'form': form})


def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()  # This also saves the Profile
            login(request, user)
            return redirect('yoga:profile')
    else:
        form = SignUpForm()
    
    return render(request, 'yoga/signup.html', {'form': form})


from django.contrib.auth.views import LoginView, LogoutView
class CustomLoginView(LoginView):
    template_name = 'yoga/login.html'

@login_required
# views.py
def profile_view(request):
    profile = request.user.profile

    # full queryset for logic
    practices_qs = request.user.practices.all()

    # slice only for display if you need
    practices = practices_qs[:30]

    # collect distinct days practiced
    practiced_days = set(practices_qs.values_list("date", flat=True))

    practice_form = PracticeForm()

     # Get month/year from query params or use current
    month = int(request.GET.get("month", date.today().month))
    year = int(request.GET.get("year", date.today().year))
    today = date.today()

    cal = calendar.Calendar(firstweekday=6)
    month_days = cal.itermonthdates(year, month)

    # Calendar setup
    calendar_days = []
    for day in month_days:
        calendar_days.append({
            "label": day.day,
            "in_month": day.month == month,
            "practiced": day in practiced_days and day <= today,
            "is_today": day == today,
            "future": day > today,
        })

    # Previous / next month for navigation
    first_day = date(year, month, 1)
    prev_month = first_day - timedelta(days=1)
    next_month = (first_day + timedelta(days=31)).replace(day=1)

    try:
        user_id = request.user.id if request.user.is_authenticated else None
        
        # Call the stats API
        api_url = request.build_absolute_uri('/api/yoga/user/stats/')
        if user_id:
            api_url += f'?user_id={user_id}'
        
        response = requests.get(api_url)
        stats = response.json() if response.status_code == 200 else {}
        
    except Exception as e:
        print(f"Error fetching stats: {e}")
        stats = {
            'total_sessions': 0,
            'avg_accuracy': 0,
            'total_hours': 0,
            'longest_streak': 0,
            'recent_sessions': []
        }
    
    # context = {
    #     'total_sessions': stats.get('total_sessions', 0),
    #     'avg_accuracy': stats.get('avg_accuracy', 0),
    #     'total_hours': stats.get('total_hours', 0),
    #     'longest_streak': stats.get('longest_streak', 0),
    #     'recent_sessions': stats.get('recent_sessions', []),
    #     'user': request.user
    # } 

    # return render(request, 'yoga/profile.html', context)


    return render(request, "yoga/profile.html",{
        "profile": profile,
        "practices": practices[:30],
        "practice_form": practice_form,
        "calendar_days": calendar_days,
        "month_name": first_day.strftime("%B"),
        "year": year,
        "prev_month": prev_month,
        "next_month": next_month,
        'total_sessions': stats.get('total_sessions', 0),
        'avg_accuracy': stats.get('avg_accuracy', 0),
        'total_hours': stats.get('total_hours', 0),
        'longest_streak': stats.get('longest_streak', 0),
        'recent_sessions': stats.get('recent_sessions', []),
        'user': request.user

    })




def edit_profile_view(request):
    profile = request.user.profile
    if request.method == "POST":
        form = ProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            return redirect('yoga:profile')
    else:
        form = ProfileForm(instance=profile)

    return render(request, 'yoga/edit_profile.html', {
        'form': form
    })

@login_required
def add_practice(request):
    if request.method == 'POST':
        form = PracticeForm(request.POST)
        if form.is_valid():
            p = form.save(commit=False)
            p.user = request.user
            p.save()
            form.save_m2m()
            return redirect('yoga:profile')
    return redirect('yoga:profile')

@login_required
def stats_view(request):
    user = request.user
    practices = user.practices.all()
    total_days = practices.count()
    total_minutes = sum([p.duration_minutes for p in practices])
    # simple weekly last-7 days breakdown
    today = timezone.localdate()
    last7 = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    breakdown = []
    for d in last7:
        exists = practices.filter(date=d).exists()
        breakdown.append({'date': d.isoformat(), 'done': exists})
    return render(request, 'yoga/stats.html', {
        'total_days': total_days,
        'total_minutes': total_minutes,
        'breakdown': breakdown
    })

# JSON endpoint for calendar to fetch practice dates
@login_required
@require_GET
def practice_dates_json(request):
    dates = list(request.user.practices.values_list('date', flat=True))
    # convert to ISO string list
    iso_dates = [d.isoformat() for d in dates]
    return JsonResponse({'dates': iso_dates})


# @login_required
# def session_view(request, pose=None):
#     return render(request, "yoga/session.html", {
#         'user': request.user,
#         'user_id': request.user.id,
#         'selected_pose': pose
#     })

@login_required
def session_view(request, pose=None):
    pose_obj = None
    if pose:
        from yoga.models import Pose
        # print(f"🔍 pose parameter: '{pose}'")
        all_poses = list(Pose.objects.values_list('title', flat=True))
        # print(f"📋 All titles: {all_poses}")
        pose_obj = Pose.objects.filter(title__icontains=pose).first()
        # print(f"✅ pose_obj: {pose_obj}")
        # print(f"📝 description: {pose_obj.description[:50] if pose_obj else 'None'}")
    
    return render(request, "yoga/session.html", {
        'user':          request.user,
        'user_id':       request.user.id,
        'selected_pose': pose,
        'pose1':         pose_obj,
    })



@login_required
def custom_logout(request):
    logout(request)
    return redirect('yoga:landing')


# def poses(request):
#     poses = [
#         {"name": "Mountain Pose", "difficulty": "Beginner", "duration": "30s", "image": "🏔️"},
#         {"name": "Warrior I", "difficulty": "Intermediate", "duration": "45s", "image": "🗡️"},
#         {"name": "Tree Pose", "difficulty": "Beginner", "duration": "60s", "image": "🌳"},
#         {"name": "Downward Dog", "difficulty": "Beginner", "duration": "45s", "image": "🐕"},
#         {"name": "Cobra Pose", "difficulty": "Intermediate", "duration": "30s", "image": "🐍"},
#         {"name": "Warrior III", "difficulty": "Advanced", "duration": "45s", "image": "⚔️"},
#     ]
#     return render(request, "yoga/poses.html", {"poses": poses})