from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class Doctor(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # Add additional fields specific to doctors
    specialty = models.CharField(max_length=100)
    experience_years = models.IntegerField()

class Patient(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # Add additional fields specific to patients
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    
    # detection/models.py
from django.db import models

class Symptom(models.Model):
    AGE_CHOICES = [(i, str(i)) for i in range(1, 101)]
    GENDER_CHOICES = [(1, 'Female'), (3, 'Male')]
    YES_NO_CHOICES = [(1, 'Yes'), (3, 'No')]

    age = models.IntegerField(choices=AGE_CHOICES)
    gender = models.IntegerField(choices=GENDER_CHOICES)
    smoking = models.IntegerField(choices=YES_NO_CHOICES)
    yellow_fingers = models.IntegerField(choices=YES_NO_CHOICES)
    anxiety = models.IntegerField(choices=YES_NO_CHOICES)
    peer_pressure = models.IntegerField(choices=YES_NO_CHOICES)
    chronic_disease = models.IntegerField(choices=YES_NO_CHOICES)
    fatigue = models.IntegerField(choices=YES_NO_CHOICES)
    allergy = models.IntegerField(choices=YES_NO_CHOICES)
    wheezing = models.IntegerField(choices=YES_NO_CHOICES)
    alcohol_consumption = models.IntegerField(choices=YES_NO_CHOICES)
    coughing = models.IntegerField(choices=YES_NO_CHOICES)
    shortness_of_breath = models.IntegerField(choices=YES_NO_CHOICES)
    swallowing_difficulty = models.IntegerField(choices=YES_NO_CHOICES)
    chest_pain = models.IntegerField(choices=YES_NO_CHOICES)

    def __str__(self):
        return f"Symptom entry for Age: {self.age}, Gender: {self.get_gender_display()}"

