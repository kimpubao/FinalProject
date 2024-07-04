from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from common.forms import UserForm
from django.utils import timezone
from .forms import QuestionForm
from .models import Question
from django.http import JsonResponse
import json


def index(request):
    user_chat_list = Question.objects.order_by('create_date')
    # user_chat_list = user_chat_list.filter(Q(author__username__icontains = request.user.id))
    context = {'context': user_chat_list}
    return render(request, 'chatPage/main.html', context)

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

def userchat(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        content = data.get('content')
        author_id = data.get('author')

        if content and author_id:
            form = QuestionForm(data)
            if form.is_valid():
                user = form.save(commit=False)
                user.create_date = timezone.now()
                user.author_id = author_id
                user.save()

            return JsonResponse({'message': 'Message received successfully!'})
    return JsonResponse({'error': 'Invalid request'}, status=400)