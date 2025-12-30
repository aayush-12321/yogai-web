from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Profile, Practice

class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)
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
        fields = ('username','first_name','last_name','email','password1','password2')

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
