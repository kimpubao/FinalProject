import io
import sqlite3
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from common.forms import UserForm
from django.utils import timezone
from .forms import QuestionForm
from .models import Question
from django.http import HttpResponse, JsonResponse
import json
from django.db.models import Q
from django.views.decorators.csrf import csrf_exempt
import speech_recognition as sr
import requests
from bs4 import BeautifulSoup
from django.contrib.auth.models import User
import datetime
def index(request):
    user_chat_list = Question.objects.order_by('create_date')
    user_chat_list = user_chat_list.filter(Q(author__id = request.user.id))
    # user_chat_list = user_chat_list.filter(Q(author__username__icontains = request.user))
    url = 'https://www.yna.co.kr/entertainment/movies'
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    news_data = soup.select('#container > div > div > div.section01 > section > div.list-type038 > ul .news-con .tit-news')
    news_link = soup.select('#container > div > div > div.section01 > section > div.list-type038 > ul .news-con .tit-wrap')
    news_text = []
    # news_links = []
    for i in range(len(news_data)):
        news_text.append([news_link[i].attrs['href'], news_data[i].text])
    context = {'context': user_chat_list, 'news': news_text}
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


@csrf_exempt
def mike(request):
    if request.method == 'POST':
        recognizer = sr.Recognizer()
        with sr.Microphone() as source: # 마이크에서 입력받기
            audio = recognizer.listen(source)

        try:
            transcribed_text = recognizer.recognize_google(audio, language='ko-KR')
        except sr.UnknownValueError:
            transcribed_text = "(당신의 말을 이해하지 못했습니다.)"
        except sr.RequestError as e:
            transcribed_text = "(서비스에 문제가 발생했습니다; {e})"

        return JsonResponse({'transcribed_text': transcribed_text})
    return JsonResponse({'error': 'Invalid request'}, status=400)

def mypage(request):
    return render(request, 'chatPage/mypage.html')

def delete_account(request):
    if request.method == 'POST':
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM auth_user WHERE id = ?", (request.user.id,))
        conn.commit()
        conn.close()
        return redirect('/')
    
def change_password(request):
    if request.method == 'POST':
        if request.POST.get('password1') == request.POST.get('password2'):
            user = User.objects.get(username = request.user)
            user.set_password(request.POST.get('password2'))
            user.save()
            user = authenticate(username=request.user, password=request.POST.get('password2'))
            login(request, user)
            return redirect('/')
        return JsonResponse({'error': 'password error'}, status=400)

def chat_export(request):
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    cursor.execute("SELECT content, create_date FROM chatPage_question WHERE author_id = ?", (request.user.id,))
    chat_log = cursor.fetchall()
    print('test')
    output = io.StringIO()
    for content, create_date in chat_log: # 텍스트 파일 작성
        output.write(f"[{create_date}]: {content}\n")

    # 텍스트 파일을 HTTP 응답으로 반환
    response = HttpResponse(output.getvalue(), content_type='text/plain')
    now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    response['Content-Disposition'] = f'attachment; filename=chat_log{now_time}.txt'
    return response