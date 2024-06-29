from django.shortcuts import render

def index(request):
    return render(request, 'chatPage/main.html')

def login(request):
    return render(request, 'chatPage/login.html')

def signup(request):
    return render(request, 'chatPage/signup.html')