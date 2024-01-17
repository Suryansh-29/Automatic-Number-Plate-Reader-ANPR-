# anpr_app/forms.py
from django import forms
from .models import NumberPlate
from .models import NumberPlate2

class NumberPlateForm(forms.ModelForm):
    class Meta:
        model = NumberPlate
        fields = ['image']

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = NumberPlate2
        fields = ['file']

