from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to='static/yoga/images/', blank=True, null=True)
    date_of_birth = models.DateField(blank=True, null=True)
    medical_condition = models.TextField(blank=True, help_text="Any medical conditions or injuries that might affect your practice")

    def __str__(self):
        return f"{self.user.username} Profile"

class Pose(models.Model):
    title = models.CharField(max_length=100)
    difficulty = models.CharField(max_length=30, blank=True)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='static/yoga/poses/', blank=True, null=True)
    duration_minutes = models.PositiveSmallIntegerField(default=2)

    def __str__(self):
        return self.title

class Practice(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='practices')
    date = models.DateField(default=timezone.localdate)
    poses = models.ManyToManyField(Pose, blank=True)
    duration_minutes = models.PositiveIntegerField(default=0)
    notes = models.TextField(blank=True)

    class Meta:
        unique_together = ('user', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.user.username} {self.date}"
