# anpr_app/models.py
from django.db import models

class NumberPlate(models.Model):
    plate_text = models.CharField(max_length=20, blank=True)
    image = models.ImageField(upload_to='number_plate/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.plate_text


class NumberPlate2(models.Model):
    file = models.FileField(upload_to='videos/')

    def __str__(self):
        return self.file.name
