from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from common.forms import UserForm
from django.contrib import messages

def index(request):
    return render(request, 'chatPage/main.html')

def loginPage(request):
    return render(request, 'chatPage/login.html')

def logout_view(request):
    logout(request)
    return redirect('/')

def signup(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('/')
    else:
        form = UserForm()
    return render(request, 'chatPage/signup.html', {'form': form})