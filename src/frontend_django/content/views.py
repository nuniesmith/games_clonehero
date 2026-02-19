"""
Views for Clone Hero Content Manager
Replicates functionality from Streamlit pages
"""
import requests
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.conf import settings
from loguru import logger
from .models import Song


# ============================================================================
# HOME / DASHBOARD
# ============================================================================
def home(request):
    """Landing page with navigation to all features"""
    return render(request, 'home.html')


# ============================================================================
# SONGS MANAGEMENT
# ============================================================================
def songs_list(request):
    """Display all songs with pagination"""
    page_number = request.GET.get('page', 1)
    songs = Song.objects.all()
    paginator = Paginator(songs, 10)  # 10 songs per page
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'total_songs': songs.count(),
    }
    return render(request, 'songs/list.html', context)


def songs_upload(request):
    """Handle song upload form"""
    if request.method == 'POST':
        return upload_song_handler(request)
    return render(request, 'songs/upload.html')


def upload_song_handler(request):
    """Process song upload via API"""
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)
    
    uploaded_file = request.FILES['file']
    
    # Validate file size
    if uploaded_file.size > settings.MAX_UPLOAD_SIZE:
        return JsonResponse({
            'error': f'File size exceeds {settings.MAX_FILE_SIZE_GB}GB limit'
        }, status=413)
    
    # Validate file extension
    allowed_extensions = ['.zip', '.rar']
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if f'.{file_ext}' not in allowed_extensions:
        return JsonResponse({
            'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
        }, status=400)
    
    try:
        # Forward to API
        files = {'file': (uploaded_file.name, uploaded_file, 'application/octet-stream')}
        data = {'content_type': 'songs'}
        
        response = requests.post(
            f"{settings.API_URL}/upload_content/",
            files=files,
            data=data,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'error' in result:
                return JsonResponse({'error': result['error']}, status=400)
            return JsonResponse({'success': True, 'message': 'Song uploaded successfully'})
        else:
            return JsonResponse({'error': response.text}, status=response.status_code)
            
    except Exception as e:
        logger.exception(f"Error uploading song: {e}")
        return JsonResponse({'error': str(e)}, status=500)


# ============================================================================
# DATABASE EXPLORER
# ============================================================================
def database_explorer(request):
    """Search and explore songs in the database"""
    search_query = request.GET.get('search', '').strip()
    page_number = request.GET.get('page', 1)
    
    songs = Song.objects.all()
    
    # Apply search filter
    if search_query:
        songs = songs.filter(
            title__icontains=search_query
        ) | songs.filter(
            artist__icontains=search_query
        ) | songs.filter(
            album__icontains=search_query
        )
    
    paginator = Paginator(songs, 10)
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_results': songs.count(),
    }
    return render(request, 'database_explorer.html', context)


@require_http_methods(["DELETE", "POST"])
def delete_song(request, song_id):
    """Delete a song by ID via API"""
    try:
        response = requests.delete(
            f"{settings.API_URL}/songs/{song_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'error': response.text}, status=response.status_code)
            
    except Exception as e:
        logger.exception(f"Error deleting song {song_id}: {e}")
        return JsonResponse({'error': str(e)}, status=500)


# ============================================================================
# BACKGROUNDS MANAGEMENT
# ============================================================================
def backgrounds(request):
    """Manage image and video backgrounds"""
    if request.method == 'POST':
        return upload_background_handler(request)
    
    # Fetch existing backgrounds from API
    try:
        img_response = requests.get(
            f"{settings.API_URL}/list_content/",
            params={'content_type': 'image_backgrounds'},
            timeout=30
        )
        vid_response = requests.get(
            f"{settings.API_URL}/list_content/",
            params={'content_type': 'video_backgrounds'},
            timeout=30
        )
        
        image_backgrounds = img_response.json().get('files', []) if img_response.status_code == 200 else []
        video_backgrounds = vid_response.json().get('files', []) if vid_response.status_code == 200 else []
        
    except Exception as e:
        logger.error(f"Error fetching backgrounds: {e}")
        image_backgrounds = []
        video_backgrounds = []
    
    context = {
        'image_backgrounds': image_backgrounds,
        'video_backgrounds': video_backgrounds,
    }
    return render(request, 'backgrounds.html', context)


def upload_background_handler(request):
    """Upload background via API"""
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)
    
    uploaded_file = request.FILES['file']
    bg_type = request.POST.get('bg_type', 'image_backgrounds')
    
    try:
        files = {'file': (uploaded_file.name, uploaded_file, 'application/octet-stream')}
        data = {'content_type': bg_type}
        
        response = requests.post(
            f"{settings.API_URL}/upload_content/",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'error': response.text}, status=400)
            
    except Exception as e:
        logger.exception(f"Error uploading background: {e}")
        return JsonResponse({'error': str(e)}, status=500)


# ============================================================================
# COLORS MANAGEMENT
# ============================================================================
def colors(request):
    """Manage color profiles"""
    if request.method == 'POST':
        return upload_color_handler(request)
    
    # Fetch existing color profiles
    try:
        response = requests.get(
            f"{settings.API_URL}/list_content/",
            params={'content_type': 'colors'},
            timeout=30
        )
        color_profiles = response.json().get('files', []) if response.status_code == 200 else []
    except Exception as e:
        logger.error(f"Error fetching colors: {e}")
        color_profiles = []
    
    context = {'color_profiles': color_profiles}
    return render(request, 'colors.html', context)


def upload_color_handler(request):
    """Upload color profile via API"""
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)
    
    uploaded_file = request.FILES['file']
    
    try:
        files = {'file': (uploaded_file.name, uploaded_file, 'application/octet-stream')}
        data = {'content_type': 'colors'}
        
        response = requests.post(
            f"{settings.API_URL}/upload_content/",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'error': response.text}, status=400)
            
    except Exception as e:
        logger.exception(f"Error uploading color profile: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["DELETE", "POST"])
def delete_color(request, profile_name):
    """Delete color profile via API"""
    try:
        response = requests.delete(
            f"{settings.API_URL}/delete_content/",
            params={'content_type': 'colors', 'file': profile_name},
            timeout=30
        )
        
        if response.status_code == 200:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'error': response.text}, status=response.status_code)
            
    except Exception as e:
        logger.exception(f"Error deleting color profile: {e}")
        return JsonResponse({'error': str(e)}, status=500)


# ============================================================================
# HIGHWAYS MANAGEMENT
# ============================================================================
def highways(request):
    """Manage highway image and video files"""
    if request.method == 'POST':
        return upload_highway_handler(request)
    
    # Fetch existing highways
    try:
        img_response = requests.get(
            f"{settings.API_URL}/list_content/",
            params={'content_type': 'image_highways'},
            timeout=30
        )
        vid_response = requests.get(
            f"{settings.API_URL}/list_content/",
            params={'content_type': 'video_highways'},
            timeout=30
        )
        
        image_highways = img_response.json().get('files', []) if img_response.status_code == 200 else []
        video_highways = vid_response.json().get('files', []) if vid_response.status_code == 200 else []
        
    except Exception as e:
        logger.error(f"Error fetching highways: {e}")
        image_highways = []
        video_highways = []
    
    context = {
        'image_highways': image_highways,
        'video_highways': video_highways,
    }
    return render(request, 'highways.html', context)


def upload_highway_handler(request):
    """Upload highway via API"""
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)
    
    uploaded_file = request.FILES['file']
    hw_type = request.POST.get('hw_type', 'image_highways')
    
    try:
        files = {'file': (uploaded_file.name, uploaded_file, 'application/octet-stream')}
        data = {'content_type': hw_type}
        
        response = requests.post(
            f"{settings.API_URL}/upload_content/",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'error': response.text}, status=400)
            
    except Exception as e:
        logger.exception(f"Error uploading highway: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["DELETE", "POST"])
def delete_highway(request, hw_type, highway_name):
    """Delete highway via API"""
    try:
        response = requests.delete(
            f"{settings.API_URL}/delete_content/",
            params={'content_type': hw_type, 'file': highway_name},
            timeout=30
        )
        
        if response.status_code == 200:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'error': response.text}, status=response.status_code)
            
    except Exception as e:
        logger.exception(f"Error deleting highway: {e}")
        return JsonResponse({'error': str(e)}, status=500)


# ============================================================================
# SONG GENERATOR
# ============================================================================
def song_generator(request):
    """Process songs into Clone Hero format"""
    if request.method == 'POST':
        return process_song_handler(request)
    return render(request, 'song_generator.html')


def process_song_handler(request):
    """Process song via API"""
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided'}, status=400)
    
    uploaded_file = request.FILES['file']
    
    try:
        files = {'file': (uploaded_file.name, uploaded_file, 'application/octet-stream')}
        
        response = requests.post(
            f"{settings.API_URL}/process_song/",
            files=files,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'error' in result:
                return JsonResponse({'error': result['error']}, status=400)
            return JsonResponse(result)
        else:
            return JsonResponse({'error': response.text}, status=400)
            
    except Exception as e:
        logger.exception(f"Error processing song: {e}")
        return JsonResponse({'error': str(e)}, status=500)
