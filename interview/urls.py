from django.contrib import admin
from django.urls import path
from auto_interview import views

urlpatterns = [
    path('', views.user_login, name='login'),
    path('admin/', admin.site.urls),
    path('login/', views.user_login, name='login'),
    path('signup/', views.user_signup, name='signup'),
    path('home_giver/', views.home_giver, name='home_giver'),
    path('home_taker/', views.home_taker, name='home_taker'),
]
