from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth.models import AbstractUser
from config import settings

class Question(models.Model):
    author=models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True)
    content = models.TextField()
    create_date = models.DateTimeField()

class Answer(models.Model):
    question = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True)
    content = models.TextField()
    create_date = models.DateTimeField()
    poster_link = models.TextField(null=True, blank=True)

class User(AbstractUser):
    gender = models.CharField(max_length=1, default="", null=True, blank=True)
    genre = models.CharField(max_length=50, null=True, blank=True)
    age_group = models.IntegerField(null=True, blank=True)

class Document(models.Model):
    upload = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)