"""
URL patterns for content app
"""
from django.urls import path
from . import views

app_name = 'content'

urlpatterns = [
    # Home
    path('', views.home, name='home'),
    
    # Songs
    path('songs/', views.songs_list, name='songs_list'),
    path('songs/upload/', views.songs_upload, name='songs_upload'),
    path('songs/<int:song_id>/delete/', views.delete_song, name='delete_song'),
    
    # Database Explorer
    path('database/', views.database_explorer, name='database_explorer'),
    
    # Backgrounds
    path('backgrounds/', views.backgrounds, name='backgrounds'),
    
    # Colors
    path('colors/', views.colors, name='colors'),
    path('colors/<str:profile_name>/delete/', views.delete_color, name='delete_color'),
    
    # Highways
    path('highways/', views.highways, name='highways'),
    path('highways/<str:hw_type>/<str:highway_name>/delete/', views.delete_highway, name='delete_highway'),
    
    # Song Generator
    path('generator/', views.song_generator, name='song_generator'),
]
