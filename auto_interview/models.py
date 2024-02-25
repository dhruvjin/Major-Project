from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    USER_ROLE_CHOICES = (
        ('test_giver', 'Test Giver'),
        ('test_taker', 'Test Taker'),
    )
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=USER_ROLE_CHOICES)

    def __str__(self):
        return self.user.username
