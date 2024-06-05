# admin.py

from django.contrib import admin
from .models import Profile
from .models import Profile, Test, Question, Answer,TestResult,CandidateTestStatus,Tip,VirtualCoachSession


admin.site.register(Profile)
admin.site.register(Test)
admin.site.register(Question)
admin.site.register(Answer)
admin.site.register(TestResult)
admin.site.register(CandidateTestStatus)
admin.site.register(Tip)
admin.site.register(VirtualCoachSession)