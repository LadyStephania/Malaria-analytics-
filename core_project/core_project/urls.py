from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('malaria.urls')), # Points traffic straight to your app routes
]
