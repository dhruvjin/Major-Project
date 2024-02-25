from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import authenticate, login
from .models import Profile
from django.contrib.auth.models import User

from django.contrib.auth.decorators import login_required

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            try:
                profile = user.profile
                if profile.role == 'test_giver':
                    return redirect('home_giver')
                else:
                    return redirect('home_taker')
            except Profile.DoesNotExist:
                return redirect('signup')  # Redirect to signup if UserProfile does not exist
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})


def user_signup(request):
    print("Inside user_signup function")
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password1')
        role = request.POST.get('role')

        # Save user object
        user = User.objects.create_user(username=username, password=password)
        
        # Create profile
        profile = Profile.objects.create(user=user, role=role)
        profile.save()
        # Authenticate user
        user = authenticate(username=username, password=password)
        
        # Login user
        if user is not None:
            login(request, user)
            if role == 'test_giver':
                return redirect('login')
            else:
                return redirect('login')
    return render(request, 'signup.html')



def home_giver(request):
    return render(request, 'home_giver.html')


def home_taker(request):
    return render(request, 'home_taker.html')
