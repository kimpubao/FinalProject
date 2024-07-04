# FinalProject

## 프로젝트 개요

### 프로젝트 이름
Bert와 GPT를 이용한 영화추천 챗봇 개발

### 팀 구성
**팀명:** 신호등 사천황

**팀원:**
- 최동현 (팀장)
- 김상준
- 안성진
- 오승주

### 목적
채팅과 동시에 이미지 인식 지원 모델을 개발하여 다양한 사용자 요구를 충족시키며 영화 추천 시스템을 중심으로 
텍스트, 이미지, 음성 기능을 통합하는 프로젝트

## 기능

### 주요 기능
1. **자연어 이해 및 전처리**
2. **영화 추천 시스템**
3. **일반 대화 및 텍스트 요약**
4. **음성 인식 기능**
5. **웹을 이용한 사용자 편의성 기능 및 디자인**
6. **데이터 시각화를 통한 분석 내용 출력**

### 기대 효과
- 영화 추천에 특화된 가벼운 모델
- 텍스트 요약 기능으로 독서 시간 절약
- 음성을 이용한 접근성 향상
- 전문가가 아니어도 쉽게 사용할 수 있는 환경 조성
- 추천 근거를 시각화하여 신뢰도 향상

## 사용 기술

### 프로그래밍 언어
- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### 라이브러리 및 프레임워크
- ![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)
- ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
- ![Konlpy](https://img.shields.io/badge/Konlpy-0277BD?style=for-the-badge&logo=konlpy&logoColor=white)
- ![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)
- ![PyQt5](https://img.shields.io/badge/PyQt5-41CD52?style=for-the-badge&logo=qt&logoColor=white)
- ![Numpy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)

### 주요 API 및 데이터 소스
1. The Movie Database (TMDb) API
2. MovieLens 데이터셋
3. IMDb Datasets
4. OMDb API (Open Movie Database)
5. JustWatch API (비공식)
6. Trakt.tv API
7. TVmaze API
8. YTS API (YIFY Torrents)
9. YouTube Data API
10. 비공식 Netflix API
11. uNoGS (unofficial Netflix Online Global Search)
12. AIhub
    
## 역할 분담 및 일정

### 역할 분담
- **최동현:** 백엔드, 음성인식, 프로젝트 관리
- **김상준:** 프론트엔드, 데이터 분석, 머신러닝
- **안성진:** 데이터 분석, 머신러닝(추천시스템)
- **오승주:** 데이터 분석, 자연어 처리(GPT2)

### 일정
- ~ 6/28: 주제 선정 및 일정 수립
- **7/1 ~ 7/6:** 데이터 수집 및 분석, 웹 디자인 초안 구성, DB 설계 및 테이블 생성, 데이터 분석 진행
- **7/7 ~ 7/13:** 데이터 전처리, 머신러닝 모델 개발 및 테스트, 웹 개발 시작, 데이터 분석 및 머신러닝 적용, Bert를 통한 이미지 생성(stable diffusion)
- **7/14 ~ 7/20:** 웹 디자인 및 페이지 구성 완료, 챗봇 기능 개발, 영상 처리 진행, 음성 인식 및 텍스트 변환 기능 개발, 머신러닝 모델 테스트 및 튜닝
- **7/21 ~ 7/31:** 챗봇 기능 테스트 및 수정, 영상 처리 기능 진행, 백엔드 데이터 처리 테스트, 웹에 머신러닝 모델 이식, 음성 인식 기능 테스트 및 웹에 이식
- **8/1 ~ 8/11:** 챗봇 기능 추가 학습 및 테스트, 영상 처리 및 챗봇 연결(멀티모달)
- **8/12 ~ 8/17:** 기능 보완 및 테스트, 발표 자료 준비
- **8/19:** 프로젝트 발표



## 프로그램 사용 전 라이브러리 설치
```
pip install -r requirements.txt
```
---
