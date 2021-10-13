# todos/urls.py
from django.urls import path, include
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', views.index, name='home'),
    path('video_feed', views.video_feed, name='video_feed'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)   # FOR IMAGES
