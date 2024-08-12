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
import datetime
import os
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, BartForConditionalGeneration
from .forms import DocumentForm
from django.contrib.auth import get_user_model
import tensorflow as tf
from konlpy.tag import Okt, Komoran
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras import layers
import re
import json
import numpy as np
import pickle
from .cnn_classifier import CNNClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import spacy
from .recommend import *

nlp = spacy.load("ko_core_news_sm")
movie_data = pd.read_csv('movie_data')
movie_name_genre= pd.read_pickle('movie_genre13306.pkl')
genre_weights_df = pd.read_csv('genre_weights_df')
with open('movie_info_titles.pkl', 'rb') as f:
    movie_info, movie_titles = pickle.load(f)
with open('scraping/poster_link3.pkl', 'rb') as f:
    movie_poster_links = pickle.load(f)

DATA_IN_PATH = 'data_in/'
model_name = 'cnn_classifier_kr'
stop_words = set(['은', '는', '이', '가', '하', '아', '것', '들','의', '있', '되', '수', '보', '주', '등', '한'])
prepro_configs = json.load(open(DATA_IN_PATH + 'data_configs.json', 'r'))

kargs = {'model_name': model_name,
        'vocab_size': prepro_configs['vocab_size'],
        'embedding_size': 128,
        'num_filters': 100,
        'dropout_rate': 0.5,
        'hidden_dimension': 250,
        'output_dimension':1}

okt=Okt()
komoran = Komoran()

# DataFrame의 장르 정규화
movie_titles['info'] = movie_titles['info'].apply(lambda x: {**x, '장르': normalize_genre(x['장르'])})
# 처리된 텍스트 데이터 벡터화
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movie_data['Processed_Text'])
# 코사인 유사도 행렬 계산
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
genres = movie_data['장르'].str.split(',').explode().str.strip().unique().tolist()
# 영화명, 감독, 출연 배우, 장르 및 인물 찾기
genres = get_genres(movie_titles)
movies = movie_data['영화명'].dropna().apply(lambda x: x.split(', ')).explode().unique()
directors = movie_data['감독'].dropna().apply(lambda x: x.split(', ')).explode().unique()
cast_members = movie_data['출연진'].dropna().apply(lambda x: x.split(', ')).explode().unique()
movie_name_genre['text']= movie_name_genre['영화명'] +' '+ movie_name_genre['장르']

with open('data_out/cnn_classifier_kr/vocab_list.pkl', 'rb') as f:
    d = pickle.load(f)
tokenizer_clf = Tokenizer()
tokenizer_clf.fit_on_texts(d)
cnn_model = CNNClassifier(**kargs)
dummy_input = tf.random.uniform((1, 40))
cnn_model(dummy_input)
cnn_model.load_weights('data_out/cnn_classifier_kr/cnn_classifier_kr_weights.h5')

def preprocessing(text, okt, remove_stopwords = False, stop_words = []):
    user_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", text)
    word_text = okt.morphs(user_text, stem=True)
    if remove_stopwords:
        word_text = [token for token in word_text if not token in stop_words]
    return word_text

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

model_path = "chatbot_model0801_432634.pth" 
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
        chat_log.append(['bot',bot_chat_list[i].content, bot_chat_list[i].create_date, bot_chat_list[i].poster_link])
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
        if request.user.is_authenticated:
            # device = torch.device('cuda:0')
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
                link = ''
                form = QuestionForm({'content': content, 'author': author_id})
                if form.is_valid():
                    user = form.save(commit=False)
                    user.create_date = timezone.now()
                    user.author_id = author_id
                    user.save()

                q = content.strip()
                a = ""
                if (q != '') and (type(q) == str):
                    clf_text = [preprocessing(q, okt, remove_stopwords=True, stop_words=stop_words)]
                    test_sequences = tokenizer_clf.texts_to_sequences(clf_text)
                    test_inputs = pad_sequences(test_sequences, maxlen=40, padding='post')
                    res = (cnn_model.predict(test_inputs) > 0.5).astype(np.int32)
                
                if res:
                    with torch.no_grad():
                        while True:
                            input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)
                            input_ids = input_ids.to(device)
                            pred = model(input_ids)
                            pred = pred.logits
                            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).to('cpu').squeeze().numpy().tolist())[-1]
                            if gen == EOS:
                                break
                            a += gen.replace("▁", " ")
                        
                        # print("User > {}".format(q))
                        # print("Chatbot > {}".format(a.replace('<pad>', '').strip()))

                    answer = a.replace('<pad>', '').strip()
                else:
                    UserData = get_user_model() # 기존 유저 클래스
                    user = UserData.objects.get(username = request.user)
                    
                    a = int(user.age_group) if user.age_group else None
                    if user.gender == 'M':
                        g = '남성'
                    elif user.gender == 'F':
                        g = '여성'
                    else:
                        g = None
                    favored_genres = user.genre if user.genre else None
                    filtered_review, nnp_and_nng_review, movie_found, genre, director, actor = process_user_input(q, komoran, movie_name_genre, directors, cast_members, genres)
                    movie_title = movie_found['영화명'].tolist()[0] if movie_found is not None and not movie_found.empty else None
                    answer = display_recommendations(movie_title, genre, director, actor, g, a, favored_genres, cosine_sim, movie_data, nnp_and_nng_review, movie_name_genre)
                    for poster in movie_poster_links:
                        if poster[0] == answer[0][0]:
                            link = poster[1]
                    output = movie_data[movie_data['영화명'] == answer[0][0]]
                    # 영화명,평점,연도,상영시간,연령,감독,출연진,장르 출력 줄거리는 공간 확인 필요
                    movie_name = output['영화명'].values[0]
                    movie_lating = output['평점'].values[0]
                    movie_year = output['연도'].values[0]
                    movie_playtime = output['상영시간'].values[0]
                    movie_age = output['연령'].values[0]
                    movie_director = output['감독'].values[0]
                    movie_actors = output['출연진'].values[0]
                    movie_genre = output['장르'].values[0]
                    answer = f'영화명:{movie_name}\n평점:{movie_lating}  연도:{movie_year}  상영시간:{movie_playtime}  연령:{movie_age}\n감독 및 출연진: {movie_director}, {movie_actors}\n장르: {movie_genre}'
                    

                form = AnswerForm()
                answer_form = form.save(commit=False)
                answer_form.content = answer
                answer_form.create_date = timezone.now()
                answer_form.question_id = request.user.id
                if link:
                    answer_form.poster_link = link
                answer_form.save()
                print('link:', link)
                return JsonResponse({'message': answer, 'link': link})
        else:
            answer = '로그인 후 이용해주세요'
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