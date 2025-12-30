from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Profile, Practice

from django.core.exceptions import ValidationError
import re

# class SignUpForm(UserCreationForm):
#     email = forms.EmailField(required=True)
#     first_name = forms.CharField(required=False)
#     last_name = forms.CharField(required=False)
#     date_of_birth = forms.DateField(
#         required=False,
#         widget=forms.DateInput(attrs={'type': 'date'}),
#         help_text="Optional: Helps us provide better guidance"
#     )
#     medical_condition = forms.CharField(
#         required=False,
#         widget=forms.Textarea(attrs={'rows': 3}),
#         help_text="Optional: Any medical conditions or injuries"
#     )
#     avatar = forms.ImageField(required=False)

#     class Meta:
#         model = User
#         fields = ('username','first_name','last_name','email','password1','password2')
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile
from django.core.exceptions import ValidationError
import re
from datetime import date

class SignUpForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'placeholder': 'you@example.com'})
    )
    first_name = forms.CharField(required=False)
    last_name = forms.CharField(required=False)
    date_of_birth = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'type': 'date'}),
        help_text="Optional: Helps us provide better guidance"
    )
    medical_condition = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 3}),
        help_text="Optional: Any medical conditions or injuries"
    )
    avatar = forms.ImageField(required=False)

    class Meta:
        model = User
        fields = (
            'username', 'first_name', 'last_name', 'email',
            'password1', 'password2', 'date_of_birth',
            'medical_condition', 'avatar'
        )

    # Username must be unique
    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise ValidationError("Username already exists.")
        return username

    # Email must be unique
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError("Email already registered.")
        return email

    # Password complexity
    def clean_password1(self):
        password = self.cleaned_data.get('password1')
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters long.")
        if not re.search(r'\d', password):
            raise ValidationError("Password must contain at least one number.")
        if not re.search(r'[A-Z]', password):
            raise ValidationError("Password must contain at least one uppercase letter.")
        return password

    # Confirm passwords match
    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get("password1")
        password2 = cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            self.add_error('password2', "Passwords do not match.")

    def clean_date_of_birth(self):
        dob = self.cleaned_data.get('date_of_birth')
        if dob and dob > date.today():
            raise ValidationError("Date of Birth cannot be in the future.")
        return dob

    # Save profile fields
    def save(self, commit=True):
        user = super().save(commit=commit)
        profile, created = Profile.objects.get_or_create(user=user)
        profile.date_of_birth = self.cleaned_data.get('date_of_birth')
        profile.medical_condition = self.cleaned_data.get('medical_condition')
        avatar_file = self.cleaned_data.get('avatar')
        if avatar_file:
            profile.avatar = avatar_file
        if commit:
            profile.save()
        return user



class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ('bio','avatar', 'date_of_birth', 'medical_condition')
        widgets = {
            'date_of_birth': forms.DateInput(attrs={'type': 'date'}),
            'medical_condition': forms.Textarea(attrs={'rows': 3}),
        }

class PracticeForm(forms.ModelForm):
    class Meta:
        model = Practice
        fields = ('date','poses','duration_minutes','notes')
        widgets = {
            'date': forms.DateInput(attrs={'type':'date'}),
            'poses': forms.CheckboxSelectMultiple(),
        }
