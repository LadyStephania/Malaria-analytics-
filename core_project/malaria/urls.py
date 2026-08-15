# malaria/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('login/', views.login_view),  # alias — '/login/' is the natural guess, keep the root path as the canonical `{% url 'login' %}` target
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('dashboard/year/<int:year>/', views.year_breakdown_view, name='year_breakdown'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('analytics/monthly-estimate.csv', views.monthly_estimate_csv_view, name='monthly_estimate_csv'),
    path('data-quality/', views.data_quality_view, name='data_quality'),
    path('hotspot-analysis/', views.hotspot_view, name='hotspot_analysis'),
    path('decision-support/', views.decision_view, name='decision'),
    path('weather-forecast/', views.weather_view, name='weather'),
    path('upload-data/', views.upload_view, name='upload'),
    path('upload-data/sample.csv', views.sample_csv_view, name='sample_csv'),
    path('user-management/', views.users_view, name='users'),
]

