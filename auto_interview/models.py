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

class Test(models.Model):
    name = models.CharField(max_length=255)
    created_by = models.ForeignKey(Profile, related_name='created_tests', on_delete=models.CASCADE)
    candidates = models.ManyToManyField(Profile, related_name='assigned_tests', limit_choices_to={'role': 'test_taker'})
    has_taken = models.BooleanField(default=False)
    max_marks = models.IntegerField(default=100)
    duration = models.IntegerField(default=120)
    def __str__(self):
        return self.name

class Question(models.Model):
    test = models.ForeignKey(Test, related_name='questions', on_delete=models.CASCADE)
    text = models.TextField()
    correct_answer = models.TextField()
    marks = models.IntegerField(default=0) 
    def __str__(self):
        return self.text

class Answer(models.Model):
    question = models.ForeignKey(Question, related_name='answers', on_delete=models.CASCADE)
    candidate = models.ForeignKey(Profile, related_name='answers', on_delete=models.CASCADE)
    text = models.TextField()
    is_correct = models.BooleanField(default=False)
    test_completed = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.candidate.user.username}: {self.question.text[:50]}"

class TestResult(models.Model):
    test = models.ForeignKey(Test, on_delete=models.CASCADE)
    candidate = models.ForeignKey(Profile, on_delete=models.CASCADE)
    score = models.FloatField(default=0)
    cheated = models.BooleanField(default=False)
    session_data = models.FileField(upload_to='session_data/', null=True, blank=True)
    status = models.CharField(max_length=10, default='Pass')
    def __str__(self):
        return f"Test: {self.test.name}, Candidate: {self.candidate.user.username}, Score: {self.score}"

class CandidateTestStatus(models.Model):
    candidate = models.ForeignKey(Profile, on_delete=models.CASCADE)
    test = models.ForeignKey(Test, on_delete=models.CASCADE)
    has_taken = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.candidate.user.username} - {self.test.name}: {'Taken' if self.has_taken else 'Not Taken'}"


class Tip(models.Model):
    text = models.TextField()
    category = models.CharField(max_length=100)  # E.g., 'General', 'Technical', 'Behavioral'

class VirtualCoachSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    current_step = models.IntegerField(default=0)
    completed = models.BooleanField(default=False)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    points = models.PositiveIntegerField(default=0)