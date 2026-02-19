## Django Frontend - Quick Reference

### Common Commands

```bash
# Development
cd src/frontend_django
python manage.py runserver 0.0.0.0:8501
python manage.py collectstatic --noinput
python manage.py createsuperuser

# Docker (from project root)
docker compose build frontend
docker compose up -d frontend
docker compose logs -f frontend
docker compose exec frontend bash
docker compose exec frontend python src/frontend_django/manage.py shell

# Static Files
docker compose exec frontend python src/frontend_django/manage.py collectstatic --noinput

# Database (read-only)
docker compose exec db psql -U clonehero -d clonehero
```

### URL Patterns

- `/` - Home/landing page
- `/database/` - Search and browse songs
- `/songs/upload/` - Upload song archives
- `/generator/` - Process audio files
- `/colors/` - Manage color profiles
- `/backgrounds/` - Manage backgrounds
- `/highways/` - Manage highways
- `/admin/` - Django admin interface

### Adding a New Page

1. **View** (`content/views.py`):
```python
def my_page(request):
    context = {'data': 'value'}
    return render(request, 'my_page.html', context)
```

2. **URL** (`content/urls.py`):
```python
path('mypage/', views.my_page, name='my_page'),
```

3. **Template** (`templates/my_page.html`):
```html
{% extends 'base.html' %}
{% block content %}
<h1>My Page</h1>
{% endblock %}
```

4. **Navigation** (`templates/base.html`):
```html
<li><a href="{% url 'content:my_page' %}">My Page</a></li>
```

### AJAX Pattern

```javascript
async function uploadFile(formData) {
    const response = await fetch('/upload/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': getCookie('csrftoken'),
        },
    });
    return await response.json();
}
```

### API Communication

```python
import requests
from django.conf import settings

# GET request
response = requests.get(
    f"{settings.API_URL}/songs/",
    params={'search': query},
    timeout=30
)
songs = response.json()

# POST request (upload)
files = {'file': (filename, file_content, 'application/octet-stream')}
data = {'content_type': 'songs'}
response = requests.post(
    f"{settings.API_URL}/upload_content/",
    files=files,
    data=data,
    timeout=300
)

# DELETE request
response = requests.delete(
    f"{settings.API_URL}/songs/{song_id}",
    timeout=30
)
```

### Template Tags

```html
{% load static %}
<link rel="stylesheet" href="{% static 'css/style.css' %}">
<img src="{% static 'assets/ch_icon.png' %}">

{% load custom_filters %}
<pre>{{ metadata|pprint }}</pre>

{% url 'content:home' %}
{% url 'content:songs_upload' %}
```

### Models (Read-Only)

```python
from content.models import Song

# Query songs
songs = Song.objects.all()
songs = Song.objects.filter(artist__icontains='metallica')
song = Song.objects.get(id=123)

# Pagination
from django.core.paginator import Paginator
paginator = Paginator(songs, 10)
page_obj = paginator.get_page(page_number)
```

### Static Files

CSS: `static/css/style.css`
JS: `static/js/main.js`, `tabs.js`, `upload.js`
Assets: `static/assets/ch_icon.png`

After changes: `python manage.py collectstatic --noinput`

### Settings

Key settings in `clonehero_frontend/settings.py`:
- `DATABASES` - PostgreSQL config
- `STATIC_URL`, `STATIC_ROOT` - Static files
- `TEMPLATES` - Template directories
- `API_URL` - Backend API endpoint
- `MAX_UPLOAD_SIZE` - File upload limits

### Troubleshooting

**Static files not loading:**
```bash
python manage.py collectstatic --noinput
```

**CSRF errors:**
- Check `{% csrf_token %}` in forms
- Include token in AJAX headers
- Verify `ALLOWED_HOSTS` in settings

**Database connection:**
```bash
docker compose ps db
docker compose logs db
```

**View logs:**
```bash
docker compose logs -f frontend
docker compose exec frontend cat /var/log/frontend/django.log
```

### Files to Know

- `manage.py` - Django CLI
- `clonehero_frontend/settings.py` - Configuration
- `clonehero_frontend/urls.py` - Main routing
- `content/views.py` - All view functions
- `content/urls.py` - URL patterns
- `content/models.py` - Song model
- `templates/base.html` - Base template
- `static/css/style.css` - Stylesheet
- `static/js/main.js` - JavaScript

### Migration from Streamlit

See `MIGRATION_GUIDE.md` for:
- Complete migration steps
- Feature comparison
- Troubleshooting guide
- Testing checklist

Run: `./migrate_to_django.sh`
