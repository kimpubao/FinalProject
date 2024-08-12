import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import spacy
from konlpy.tag import Okt

genre_weights_df = pd.read_csv('genre_weights_df')
tfidf = TfidfVectorizer()
data = pd.read_csv('movie_data')
tfidf_matrix = tfidf.fit_transform(data['Processed_Text'])
okt=Okt()


# 텍스트를 토큰화하고 원형으로 복원하는 함수 정의
def tokenize_and_lemmatize(text):
    if isinstance(text, str):  # Check if 'text' is a string
        tokens = okt.morphs(text, stem=True)
        return ' '.join(tokens)
    else:
        return ''  # Return an empty string or handle NaN case appropriately
    
# 텍스트 데이터에 함수 적용
data['synopsis_processed'] = data['줄거리'].apply(tokenize_and_lemmatize)
# 처리된 텍스트 데이터 벡터화
tfidf_matrix_synopsys = tfidf.transform(data['synopsis_processed'])

stop_words_ko = {"및", "도", "에", "의", "가", "이", "은", "는", "을", "를", "에서", "로", "과", "와", "한", "그", "이", "하", "여", "속", "전", "자", "이다"}
# 장르 동의어 매핑 생성
genre_synonyms_ko = {
    '공포': ['공포', '호러', '스릴러', '무서운'],
    '로맨스': ['로맨스', '사랑', '연애', '로맨틱'],
    '멜로': ['멜로', '멜로드라마'],
    '액션': ['액션', '모험', '전투', '스릴'],
    '코미디': ['코미디', '웃긴', '유머', '희극'],
    '서부': ['서부', '웨스턴', '카우보이', '서부극', 'western'],
}



# 장르 동의어를 정규화하는 함수
def normalize_genre_from_synonyms_ko(keyword):
    keyword = keyword.strip().lower()
    for genre, synonyms in genre_synonyms_ko.items():
        if keyword in [synonym.lower() for synonym in synonyms]:
            return genre
    return keyword

# 장르 정규화 함수 (괄호 및 슬래시 처리 포함)
def normalize_genre(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'\s*\(/?\s*|\s*/\s*|\s*\)\s*', ',', text)
    text = re.sub(r'\s+', ' ', text)
    genres = [normalize_genre_from_synonyms_ko(genre.strip()) for genre in text.split(',')]
    return ','.join(sorted(set(genres)))

# 문장에서 키워드 추출하는 함수
def extract_keywords(sentence):
    doc = nlp(sentence)
    keywords = [token.text.strip() for token in doc if token.text.strip() not in stop_words_ko and token.pos_ in ['NOUN', 'ADJ', 'PROPN']]

    # 불용어를 제거한 키워드 정리
    cleaned_keywords = []
    for keyword in keywords:
        # 공백으로 나누고, 불용어 제거
        parts = re.split(r'\s+', keyword)
        cleaned_parts = [part for part in parts if part not in stop_words_ko]
        cleaned_keyword = ' '.join(cleaned_parts)
        cleaned_keywords.append(cleaned_keyword)

    return cleaned_keywords

# 장르로 영화 추천
def recommend_movies_by_genre(genre):
    normalized_genre = normalize_genre_from_synonyms_ko(genre)
    recommended_movies = movie_titles[movie_titles['info'].apply(lambda x: '장르' in x and normalized_genre in x['장르'])]
    return recommended_movies[['제목', 'info']]

# 감독 또는 출연 배우로 영화 추천
def recommend_movies_by_person(person_name):
    person_name = person_name.strip().lower()
    recommended_movies = movie_titles[movie_titles['info'].apply(
        lambda x: ('감독' in x and person_name in x['감독'].strip().lower())
    ) | movie_titles['info'].apply(
        lambda x: ('출연진' in x and person_name in x['출연진'].strip().lower())
    )]
    return recommended_movies[['제목', 'info']]

# 장르 목록을 가져오는 함수
def get_genres(movie_titles):
    genres = set()
    for info in movie_titles['info']:
        if '장르' in info:
            text = info['장르']
            genres.update([normalize_genre_from_synonyms_ko(genre.strip()) for genre in text.split(',')])
    return sorted(genres)

# 영화 제목으로 정보를 가져오는 함수
def get_movie_info(title):
    filtered_title = title.strip()
    matching_titles = movie_titles[movie_titles['제목'].str.contains(filtered_title, na=False)]

    if matching_titles.empty:
        return f"영화를 찾지 못했습니다: {title}"

    movie_id = matching_titles.index[0]
    return movie_titles[movie_titles.index == movie_id]['info'].iloc[0]

def preprocessing(user_input, komoran, remove_stopwords=False, stop_words=[]):
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ0-9\\sSFsf]", "", user_input) # 한글, 숫자, S/F 문자만 남김
    word_review = komoran.pos(review_text, flatten=False) # 형태소 분석

    # 고유명사(NNP)와 일반명사(NNG)를 분리
    filtered_review = []      # 고유명사(NNP)만 담을 리스트
    nnp_and_nng_review = []   # 고유명사(NNP)와 일반명사(NNG) 모두 담을 리스트

    for sentence in word_review:  # 문장 단위로 분리된 리스트
        for word, tag in sentence:
            if tag == 'NNP':
                filtered_review.append((word, tag))  # 고유명사만 담기
                nnp_and_nng_review.append((word, tag))  # 고유명사도 같이 담기
            elif tag == 'NNG':
                nnp_and_nng_review.append((word, tag))  # 일반명사 담기

    if remove_stopwords:
        filtered_review = [token for token in filtered_review if not token[0] in stop_words]
        nnp_and_nng_review = [token for token in nnp_and_nng_review if not token[0] in stop_words]

    return filtered_review, nnp_and_nng_review

def find_matching_movie(filtered_review, movie_name_genre):
    if filtered_review: # filtered_review가 비어 있지 않은 경우에만 수행
        for word, tag in filtered_review:
            for movie_name in movie_name_genre['영화명']:
                if re.search(word, movie_name):
                    return movie_name_genre[movie_name_genre['영화명'] == movie_name]

    return None

def find_matching_cast(filtered_review, cast_members):
    if filtered_review: # filtered_review가 비어 있지 않은 경우에만 수행
        matching_cast = []
        for word, _ in filtered_review:
            if word in cast_members:
                matching_cast.append(word)
        return matching_cast[0] if matching_cast else None

    return None

def find_matching_director(filtered_review, directors):
    if filtered_review: # filtered_review가 비어 있지 않은 경우에만 수행
        matching_director = []
        for word, _ in filtered_review:
            if word in directors:
                matching_director.append(word)
        return matching_director[0] if matching_director else None

    return None

def find_matching_genre(nnp_and_nng_review, genres):
    if nnp_and_nng_review: # nnp_and_nng_review가 비어 있지 않은 경우에만 수행
        matching_genres = []
        genre_list = genres  # 장르 목록을 리스트로 변환하여 일치 여부를 빠르게 검사

        for word, _ in nnp_and_nng_review:
            # 명사 사이에 띄어쓰기가 있을 경우, 분리하여 장르를 찾기
            words_to_check = word.split()  # 띄어쓰기로 분리된 단어들

            for w in words_to_check:
                if w in genre_list:
                    matching_genres.append(w)

        # 중복된 장르를 제거하기 위해 set을 사용한 후 리스트로 변환
        matching_genres = list(set(matching_genres))
        return ', '.join(matching_genres)

    return None

# 연령대와 성별을 받아 해당하는 가중치 행을 찾는 함수
def get_genre_weights(age, gender):
    # 주어진 나이에 따라 연령대 범위를 결정
    if age == 10:
        age_range = '0~18세'
    elif age == 20:
        age_range = '19~23세'
    elif age == 30:
        age_range = '30~34세'
    elif age == 40:
        age_range = '40~49세'
    elif age == 50:
        age_range = '50~90세'
    elif age >= 60:
        age_range = '50~90세'
    else:
        raise ValueError("유효하지 않은 나이 범위입니다.")

    age_gender_key = f"{age_range} {gender}"

    # 해당 연령대와 성별에 맞는 가중치 행 선택 및 평균 계산
    if age_gender_key == '19~23세 남성':
        weights = (genre_weights_df.loc[genre_weights_df['연령대'] == age_gender_key].reset_index(drop=True).iloc[:, 1:] +
                   genre_weights_df.loc[genre_weights_df['연령대'] == '24~29세 남성'].reset_index(drop=True).iloc[:, 1:]) / 2
    elif age_gender_key == '30~34세 남성':
        weights = (genre_weights_df.loc[genre_weights_df['연령대'] == age_gender_key].reset_index(drop=True).iloc[:, 1:] +
                   genre_weights_df.loc[genre_weights_df['연령대'] == '35~39세 남성'].reset_index(drop=True).iloc[:, 1:]) / 2
    elif age_gender_key == '19~23세 여성':
        weights = (genre_weights_df.loc[genre_weights_df['연령대'] == age_gender_key].reset_index(drop=True).iloc[:, 1:] +
                   genre_weights_df.loc[genre_weights_df['연령대'] == '24~29세 여성'].reset_index(drop=True).iloc[:, 1:]) / 2
    elif age_gender_key == '30~34세 여성':
        weights = (genre_weights_df.loc[genre_weights_df['연령대'] == age_gender_key].reset_index(drop=True).iloc[:, 1:] +
                   genre_weights_df.loc[genre_weights_df['연령대'] == '35~39세 여성'].reset_index(drop=True).iloc[:, 1:]) / 2
    else:
        weights = genre_weights_df.loc[genre_weights_df['연령대'] == age_gender_key].iloc[:, 1:]

    return weights


def get_recommendations(movie_title=None, genre=None, director=None, actor = None, gender=None, age=None, favored_genres=None, cosine_sim=None, data=None, nnp_and_nng_review = None, movie_name_genre=None):
    filtered_data = data.copy()
    # print('get', nnp_and_nng_review)
    # 기본 가중치 설정
    weights = {'genre_weight': 0.0, 'director_weight': 0.0}
    entered_criteria = 0

    # 입력된 조건에 대해 가중치 증가
    if genre:
        weights['genre_weight'] = 0.2
        entered_criteria += 1
    if director:
        weights['director_weight'] = 0.2
        entered_criteria += 1
    if actor:
        weights['actor_weight'] = 0.2
        entered_criteria += 1
    if entered_criteria > 0:
        equal_weight = 1.0 / entered_criteria
        for key in weights.keys():
            if weights[key] == 0.0:
                weights[key] = equal_weight

    #가입 할 때 기입하는 선호 장르 가중치
    if favored_genres:
        weights['favored_genre_weight'] = 0.1

    # 연령과 성별에 따른 장르 가중치 계산
    if gender and age:
        genre_weights = get_genre_weights(age, gender)  # 여성일 때 드라마 장르 가중치

    if movie_title:
        if movie_title not in data['영화명'].values:
            # 코사인 유사도 계산 없이 가장 유사한 영화명 추천
            tfidf_title_genre = tfidf.transform([tokenize_and_lemmatize(movie_name_genre[movie_name_genre['영화명']==movie_title]['text'].values[0])])
            cosine_sim_title = linear_kernel(tfidf_title_genre, tfidf_matrix)
            sim_scores = list(enumerate(cosine_sim_title[0]))

            # 입력조건, 연령과 성별에 따른 장르 가중치 적용
            for i, (idx, score) in enumerate(sim_scores):
                #입력 조건 가중치 적용
                #장르 가중치
                genre_score = 0
                if genre:
                    serched_m_genres = [g.strip().lower() for g in data.loc[idx, '장르'].split(',')]
                    if genre.strip().lower() in serched_m_genres:
                        genre_score = weights['genre_weight']

                #감독, 배우 가중치
                director_score = weights['director_weight'] if director and director in data.loc[idx, '감독'] else 0
                actor_score = weights['actor_weight'] if actor and actor in data.loc[idx, '출연진'] else 0

                #해당 영화의 장르 확인
                genre_list = data.loc[idx, '장르'].split(',')

                # 초기화
                genre_weight_score = 0
                genre_weights_list = [] # 장르 가중치 리스트

                # 장르 가중치 계산
                if gender and age:
                    for genre1 in genre_list:
                        genre1 = genre1.strip()
                        for column in genre_weights.columns:
                            if any(genre_part.strip() == genre1 for genre_part in column.split('/')):
                                genre_weight = genre_weights[column].values[0]
                                genre_weights_list.append(genre_weight)  # 각 장르 가중치를 리스트에 추가
                                break

                    # 가장 큰 두 개의 가중치를 선택하여 합산
                    if len(genre_weights_list) >= 2:
                        genre_weights_list.sort(reverse=True)
                        genre_weight_score = genre_weights_list[0] + genre_weights_list[1]
                        genre_weight_score = genre_weight_score/2  # 두 개의 가중치를 반영한 점수 계산
                    else:
                        genre_weight_score = sum(genre_weights_list)  # 장르가 두 개 미만일 경우 모든 가중치를 합산

                # 선호 장르 가중치 적용
                favored_genre_weight_score = 0
                if favored_genres:
                    for favor_genre in favored_genres:
                        if favor_genre in genre_list:
                            favored_genre_weight_score += weights['favored_genre_weight']

                # 기존 점수와 새로운 가중치를 반영한 점수 계산
                total_score = score + genre_score + director_score + actor_score +  genre_weight_score + favored_genre_weight_score
                
                # 최종 점수와 함께 저장
                sim_scores[i] = (idx, total_score)
            # print(movie_title,genre,director,actor)
            # 유사도 순으로 정렬
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # print(sim_scores)
            sim_scores = sim_scores[:5]
            movie_indices = [i[0] for i in sim_scores]
            recommendations = data.iloc[movie_indices]

            # 시각화를 위한 유사도 점수 포함
            similarity_scores = [i[1] for i in sim_scores]
            return recommendations[['영화명','장르']].values.tolist(), similarity_scores
        else: #영화 제목이 데이터에 있는 경우
            # 영화명에 해당하는 인덱스 가져오기
            idx = data[data['영화명'] == movie_title].index[0]

            # 필터링된 데이터 내에서 영화명과 유사한 영화들을 추천하기 위해 코사인 유사도 계산
            filtered_indices = filtered_data.index

            sim_scores = []
            for filtered_idx in filtered_indices:
                if filtered_idx == idx:
                    continue  # 자기 자신은 제외
                sim_score = cosine_sim[idx, filtered_idx]

                # 가중치 적용
                #장르 가중치
                genre_score = 0
                if genre:
                    serched_m_genres = [g.strip().lower() for g in data.loc[filtered_idx, '장르'].split(',')]
                    if genre.strip().lower() in serched_m_genres:
                        genre_score = weights['genre_weight']

                #감독, 배우 가중치
                director_score = weights['director_weight'] if director and director in data.loc[filtered_idx, '감독'] else 0
                actor_score = weights['actor_weight'] if actor and actor in data.loc[filtered_idx, '출연진'] else 0

                #해당 영화의 장르 확인
                genre_list = data.loc[filtered_idx, '장르'].split(',')

                # 초기화
                genre_weight_score = 0
                genre_weights_list = [] # 장르 가중치 리스트

                if gender and age:
                    for genre1 in genre_list:
                        genre1 = genre1.strip()
                        for column in genre_weights.columns:
                            if any(genre_part.strip() == genre1 for genre_part in column.split('/')):
                                genre_weight = genre_weights[column].values[0]
                                genre_weights_list.append(genre_weight)  # 각 장르 가중치를 리스트에 추가
                                break

                    # 가장 큰 두 개의 가중치를 선택하여 합산
                    if len(genre_weights_list) >= 2:
                        genre_weights_list.sort(reverse=True)
                        genre_weight_score = genre_weights_list[0] + genre_weights_list[1]
                        genre_weight_score = genre_weight_score/2  # 두 개의 가중치를 반영한 점수 계산
                    else:
                        genre_weight_score = sum(genre_weights_list)  # 장르가 두 개 미만일 경우 모든 가중치를 합산

                # 선호 장르 가중치 적용
                favored_genre_weight_score = 0
                if favored_genres:
                    for favor_genre in favored_genres:
                        if favor_genre in genre_list:
                            favored_genre_weight_score += weights['favored_genre_weight']

                # 기존 점수와 새로운 가중치를 반영한 점수 계산
                total_score = sim_score + genre_score + director_score + actor_score + genre_weight_score + favored_genre_weight_score
                # 디버깅 또는 로그 출력을 위해 idx와 관련 정보를 출력
                sim_scores.append((filtered_idx, total_score))
            # # 유사도 순으로 정렬
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # 상위 5개 추천
            sim_scores = sim_scores[:5]
            movie_indices = [i[0] for i in sim_scores]
            recommendations = data.iloc[movie_indices]

            # 시각화를 위한 유사도 점수 포함
            similarity_scores = [i[1] for i in sim_scores]
            return recommendations[['영화명','장르']].values.tolist(), similarity_scores
    else:
        # movie_title이 없는 경우 나머지 조건들을 사용하여 추천
        sim_scores = []
        # print("nnp_and_nng_review 내용:", nnp_and_nng_review)
        exclude_words = ['영화', '추천','내용','줄거리']
        words_only = [word for word, tag in nnp_and_nng_review if word not in exclude_words]
        words_text = ' '.join(words_only)
        # print("words_text 내용:", words_text)
        if any(word == '내용' or word == '줄거리' for word, tag in nnp_and_nng_review):
            # user_input을 벡터화한 것과 영화를 비교하는 로직
            user_tfidf_vector = tfidf.transform([tokenize_and_lemmatize(words_text)])
            cosine_sim_user = linear_kernel(user_tfidf_vector, tfidf_matrix_synopsys)

            for idx in filtered_data.index:
                sim_score = cosine_sim_user[0, idx]

                # 최종 점수 계산 및 저장
                total_score = sim_score  # 여기에 필요한 가중치를 추가하여 계산할 수 있습니다.
                sim_scores.append((idx, total_score))

            # 유사도 점수로 정렬 후 상위 5개를 선택
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:5]
            movie_indices = [i[0] for i in sim_scores]
            similarity_scores = [i[1] for i in sim_scores]

            # 영화 제목과 장르를 리스트로 반환
            recommendations = data.iloc[movie_indices][['영화명', '평점', '연도', '상영시간', '연령', '감독', '출연진', '줄거리', '장르']].values.tolist()
            return recommendations, similarity_scores
        else:
            for idx in filtered_data.index:
                genre_score = 0
                year_score = 0  # 각 영화마다 year_score을 초기화

                # '최신'이란 단어가 있으면 연도에 따라 점수 부여
                if any(word == '최신' for word, tag in nnp_and_nng_review):
                    if data.loc[idx, '연도'] == 2024:
                        year_score = 2

                if genre:
                    searched_m_genres = [g.strip().lower() for g in data.loc[idx, '장르'].split(',')]
                    if genre.strip().lower() in searched_m_genres:
                        genre_score = weights['genre_weight']

                director_score = weights['director_weight'] if director and director in data.loc[idx, '감독'] else 0
                actor_score = weights['actor_weight'] if actor and actor in data.loc[idx, '출연진'] else 0

                genre_list = data.loc[idx, '장르'].split(',')
                genre_weight_score = 0
                genre_weights_list = []

                if gender and age:
                    for genre1 in genre_list:
                        genre1 = genre1.strip()
                        for column in genre_weights.columns:
                            if any(genre_part.strip() == genre1 for genre_part in column.split('/')):
                                genre_weight = genre_weights[column].values[0]
                                genre_weights_list.append(genre_weight)
                                break

                    if len(genre_weights_list) >= 2:
                        genre_weights_list.sort(reverse=True)
                        genre_weight_score = genre_weights_list[0] + genre_weights_list[1]
                        genre_weight_score /= 2
                    else:
                        genre_weight_score = sum(genre_weights_list)

                favored_genre_weight_score = 0
                if favored_genres:
                    for favor_genre in favored_genres:
                        if favor_genre in genre_list:
                            favored_genre_weight_score += weights['favored_genre_weight']

                total_score = genre_score + director_score + actor_score + genre_weight_score + favored_genre_weight_score + year_score
                sim_scores.append((idx, total_score))

            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:5]
            movie_indices = [i[0] for i in sim_scores]
            recommendations = data.iloc[movie_indices]
            similarity_scores = [i[1] for i in sim_scores]
            return recommendations[['영화명', '평점', '연도', '상영시간', '연령', '감독', '출연진', '줄거리', '장르']].values.tolist(), similarity_scores

def get_user_input():
    """사용자 입력을 받아 전처리 후 반환"""
    user_input = input("영화 제목, 원하는 장르, 감독 또는 출연 배우를 입력하세요 (예: 공포, 액션, 멜로, 서부, 로버트 다우니 주니어): ")
    gender_input = input("사용자의 성별을 입력하세요 (남성, 여성) (입력하지 않으면 무시됩니다): ").strip()
    age_input = input("사용자의 나이를 입력하세요 (입력하지 않으면 무시됩니다): ").strip()
    favored_genres_input = input("사용자의 선호 장르를 입력하세요 (입력하지 않으면 무시됩니다): ").strip()

    gender = gender_input if gender_input else None
    age = int(age_input) if age_input.isdigit() else None
    favored_genres = favored_genres_input if favored_genres_input else None

    return user_input, gender, age, favored_genres

def process_user_input(user_input, komoran, movie_name_genre, directors, cast_members, genres):
    """사용자 입력을 전처리하고 영화 정보 검색"""
    filtered_review, nnp_and_nng_review = preprocessing(user_input, komoran, remove_stopwords=False, stop_words=stop_words_ko)
    
    movie_found = find_matching_movie(filtered_review, movie_name_genre)
    director_found = find_matching_director(filtered_review, directors)
    genre_found = find_matching_genre(nnp_and_nng_review, genres)
    person_found = find_matching_cast(filtered_review, cast_members)

    return filtered_review, nnp_and_nng_review, movie_found, genre_found, director_found, person_found

def display_recommendations(movie_title, genre, director, actor, gender, age, favored_genres, cosine_sim, data, nnp_and_nng_review, movie_name_genre):
    """영화 추천 결과를 출력"""
    recommendations, similarity_scores = get_recommendations(movie_title, genre, director, actor, gender, age, favored_genres, cosine_sim, data, nnp_and_nng_review, movie_name_genre)
    # print("추천된 영화:")
    # print(recommendations)
    # print("유사도 점수:")
    # print(similarity_scores)
    return recommendations
    