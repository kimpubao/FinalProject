from glob import glob
import io
import sqlite3
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
import pandas as pd
from common.forms import UserForm
from django.utils import timezone
from .forms import AnswerForm, QuestionForm
from .models import Answer, Document, Question
from django.http import HttpResponse, JsonResponse
from django.db.models import Q
from django.views.decorators.csrf import csrf_exempt
import speech_recognition as sr
import requests
from bs4 import BeautifulSoup
from django.contrib.auth.models import User
import datetime
import os
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, BartForConditionalGeneration
from .forms import DocumentForm
from django.contrib.auth import get_user_model

tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
summary_model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

# device = torch.device('cuda:0')
device = torch.device('cpu')

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

model_path = "chatbot_model_185457.pth" 
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)


def index(request):
    user_chat_list = Question.objects.order_by('create_date')
    user_chat_list = user_chat_list.filter(Q(author__id = request.user.id))
    bot_chat_list = Answer.objects.order_by('create_date')
    bot_chat_list = bot_chat_list.filter(Q(question__id = request.user.id))
    
    chat_log = []
    for i in range(len(bot_chat_list)):
        chat_log.append(['user',user_chat_list[i].content, user_chat_list[i].create_date])
        chat_log.append(['bot',bot_chat_list[i].content, bot_chat_list[i].create_date])
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
    context = {'context': chat_log, 'news': news_text}
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
            # 장르는 리스트 저장이 안되므로 따로 업데이트
            genres = request.POST.getlist('genre')
            genres_str = ','.join(genres)
            conn = sqlite3.connect('db.sqlite3')
            cursor = conn.cursor()
            query = """update chatPage_user set genre = ? where username = ?"""
            data = (genres_str, username)
            cursor.execute(query, data)
            conn.commit()
            cursor.close()
            conn.close()
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('/')
    else:
        form = UserForm()
    return render(request, 'chatPage/signup.html', {'form': form})


def userchat(request):
    if request.method == 'POST':
        # device = torch.device('cuda:0')
        device = torch.device('cpu')
        content = request.POST.get('content')
        author_id = request.POST.get('author')
        upload = request.FILES.get('upload')

        if upload:
            if upload.name.endswith('.txt'):
                document = Document(upload=upload)
                document.save()
                text = upload.read().decode('utf-8')
                lst = glob('documents\\*.txt')
                filename = ''
                ftime = 0
                for i in lst:
                    creation_time = os.path.getctime(i)
                    if ftime < creation_time:
                        ftime = creation_time
                        filename = i

                answer = '네 요약해드리겠습니다. \n\n'
                f = open(filename, 'r', encoding='utf-8')
                text = f.read()
                
                # 요약 모델 적용
                raw_input_ids = tokenizer.encode(text)
                input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

                summary_ids = summary_model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
                data = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
                
                # 답변 내용
                answer += data # data 대신 요약 내용이 들어가야 함

                form = QuestionForm({'content': content, 'author': author_id})
                user = form.save(commit=False)
                user.create_date = timezone.now()
                user.author_id = author_id
                user.save()

                form = AnswerForm()
                answer_form = form.save(commit=False)
                answer_form.content = answer
                answer_form.create_date = timezone.now()
                answer_form.question_id = request.user.id
                answer_form.save()

                return JsonResponse({'message': answer}) # 여기서 요약 모델을 적용하면 됨
            
            document = Document(upload=upload)
            document.save()
            answer = 'txt확장자의 파일만 요약할 수 있습니다.'

            form = AnswerForm()
            answer_form = form.save(commit=False)
            answer_form.content = answer
            answer_form.create_date = timezone.now()
            answer_form.question_id = request.user.id
            answer_form.save()
            
            return JsonResponse({'message': answer})

        if content and author_id:
            form = QuestionForm({'content': content, 'author': author_id})
            if form.is_valid():
                user = form.save(commit=False)
                user.create_date = timezone.now()
                user.author_id = author_id
                user.save()

            # koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            #             bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            #             pad_token=PAD, mask_token=MASK) 
            # model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

            # model_path = "chatbot_model_185457.pth" 
            # model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
            # checkpoint = torch.load(model_path)
            # model.load_state_dict(checkpoint["model_state_dict"])
            # model.eval()
            # model.to(device)

            with torch.no_grad():
                q = content.strip()
                a = ""
                while True:
                    input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)
                    input_ids = input_ids.to(device)
                    pred = model(input_ids)
                    pred = pred.logits
                    gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).to('cpu').squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace("▁", " ")
                
                print("User > {}".format(q))
                print("Chatbot > {}".format(a.replace('<pad>', '').strip()))

            answer = a.replace('<pad>', '').strip()

            form = AnswerForm()
            answer_form = form.save(commit=False)
            answer_form.content = answer
            answer_form.create_date = timezone.now()
            answer_form.question_id = request.user.id
            answer_form.save()

            return JsonResponse({'message': answer})
    
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


def upload_file(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return JsonResponse({'message': 'upload'})
    else:
        form = DocumentForm()
    return render(request, 'upload.html', {'form': form})

########## 마이페이지 기능 ##########

def mypage(request):
    if request.user.is_authenticated:
        UserData = get_user_model() # 기존 유저 클래스
        user = UserData.objects.get(id = request.user.id)
        age_group = user.age_group
        gender = user.gender
        genres = user.genre
        user_data = {"age": age_group, "gender": gender, "genre": genres}
    return render(request, 'chatPage/mypage.html', user_data)


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
            UserData = get_user_model() # 기존 유저 클래스
            user = UserData.objects.get(username = request.user)
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
    user_chat_log = cursor.fetchall()
    cursor.execute("SELECT content, create_date FROM chatPage_answer WHERE question_id = ?", (request.user.id,))
    bot_chat_log = cursor.fetchall()
    output = io.StringIO()
    log = [] # 해당 유저 채팅 기록 전체 리스트

    for content, create_date in user_chat_log:
        log.append((create_date, 'user:', content))

    for content, create_date in bot_chat_log:
        log.append((create_date, 'bot:', content))

    log.sort(key=lambda x:x[0]) # 시간 기준 정렬

    for create_date, author, content in log:  # 텍스트 파일 작성
        output.write(f"[{create_date}] [{author}] {content}\n")

    # 텍스트 파일을 HTTP 응답으로 반환
    response = HttpResponse(output.getvalue(), content_type='text/plain')
    now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    response['Content-Disposition'] = f'attachment; filename=chat_log{now_time}.txt'
    return response


def change_gender(request):
    if request.method == 'POST':
        UserData = get_user_model() # 기존 유저 클래스
        user = UserData.objects.get(username = request.user)
        user.gender = request.POST.get('gender')
        user.save()
        return redirect('/')
    

def change_genre(request):
    if request.method == 'POST':
        UserData = get_user_model() # 기존 유저 클래스
        user = UserData.objects.get(username = request.user)
        user.genre = request.POST.getlist('genre')
        user.save()
        return redirect('/')


def change_age_group(request):
    if request.method == 'POST':
        UserData = get_user_model() # 기존 유저 클래스
        user = UserData.objects.get(username = request.user)
        try:
            age = int(request.POST.get('age_group'))
        except:
            age = None
        user.age_group = age
        print('data',type(request.POST.get('age_group')), user.age_group)
        user.save()
        return redirect('/')