from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.index, name='chatPage'),
    path('loginPage/', auth_views.LoginView.as_view(template_name='chatPage/login.html'), name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('signup/', views.signup, name='signup'),
    path('userchat/', views.userchat, name='userchat'),
    path('mike/', views.mike, name='mike'),
    path('mypage/', views.mypage, name='mypage'),
    path('delete_account/', views.delete_account, name='delete_account'),
    path('change_password/', views.change_password, name='change_password'),
    path('chat_export/', views.chat_export, name='chat_export'),
    path('upload/', views.upload_file, name='upload_file'),
    path('change_gender/', views.change_gender, name='change_gender'),
    path('change_genre/', views.change_genre, name='change_genre'),
    path('change_age_group/', views.change_age_group, name='change_age_group'),
]