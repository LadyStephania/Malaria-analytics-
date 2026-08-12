# malaria/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('decision-support/', views.decision_view, name='decision'),
    path('weather-forecast/', views.weather_view, name='weather'),
    path('upload-data/', views.upload_view, name='upload'),
    path('upload-data/sample.csv', views.sample_csv_view, name='sample_csv'),
    path('user-management/', views.users_view, name='users'),
]

