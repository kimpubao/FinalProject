from django.shortcuts import render, redirect
from django.contrib.auth import login, logout


def index(request):
    return render(request, 'chatPage/main.html')

def login(request):
    return render(request, 'chatPage/login.html')

def logout_view(request):
    logout(request)
    return redirect('/')

def signup(request):
    return render(request, 'chatPage/signup.html')