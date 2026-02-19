from django.contrib import admin
from .models import Song


@admin.register(Song)
class SongAdmin(admin.ModelAdmin):
    list_display = ('title', 'artist', 'album', 'created_at')
    search_fields = ('title', 'artist', 'album')
    list_filter = ('artist', 'album')
    readonly_fields = ('created_at', 'updated_at')
