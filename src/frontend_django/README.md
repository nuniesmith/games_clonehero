# Clone Hero Frontend (Django)

This is the Django-based web frontend for the Clone Hero Content Manager.

## Quick Start

### Development

```bash
# From project root
cd src/frontend_django

# Run development server (outside Docker)
python manage.py runserver 0.0.0.0:8501

# Collect static files
python manage.py collectstatic --noinput

# Access Django admin (create superuser first)
python manage.py createsuperuser
```

### Production (Docker)

```bash
# From project root
docker compose up -d --build frontend
docker compose logs -f frontend
```

## Project Structure

```
src/frontend_django/
├── manage.py                    # Django CLI
├── clonehero_frontend/         # Project settings
│   ├── settings.py             # Configuration
│   ├── urls.py                 # Main URL routing
│   └── wsgi.py                 # WSGI entry point
├── content/                    # Main Django app
│   ├── models.py               # Song model (read-only)
│   ├── views.py                # All view functions
│   ├── urls.py                 # App URL patterns
│   ├── admin.py                # Admin config
│   └── templatetags/           # Custom filters
├── templates/                  # HTML templates
│   ├── base.html              # Base with sidebar
│   ├── home.html              # Landing page
│   ├── database_explorer.html # Search/browse
│   ├── backgrounds.html       # Background mgmt
│   ├── colors.html            # Color profiles
│   ├── highways.html          # Highway mgmt
│   ├── song_generator.html    # Audio processing
│   └── songs/
│       └── upload.html        # Song upload
└── static/                    # Static assets
    ├── css/
    │   └── style.css          # Main stylesheet
    ├── js/
    │   ├── main.js            # Core functions
    │   ├── tabs.js            # Tab switching
    │   └── upload.js          # Upload handler
    └── assets/
        └── ch_icon.png        # Logo
```

## Key Features

- **RESTful URLs**: Clean URL patterns for all pages
- **AJAX Operations**: File uploads, deletes, searches without page reload
- **Responsive Design**: Mobile-friendly CSS with sidebar navigation
- **Static File Optimization**: WhiteNoise compression and caching
- **Database Integration**: Django ORM with read-only Song model
- **API Communication**: All data operations proxy to FastAPI backend
- **Production Ready**: Gunicorn WSGI server with logging

## Available URLs

| URL | View | Description |
|-----|------|-------------|
| `/` | `home` | Landing page with feature overview |
| `/database/` | `database_explorer` | Search and browse all songs |
| `/songs/upload/` | `songs_upload` | Upload song archives |
| `/generator/` | `song_generator` | Process audio into charts |
| `/colors/` | `colors` | Manage color profiles |
| `/backgrounds/` | `backgrounds` | Manage background images/videos |
| `/highways/` | `highways` | Manage highway images/videos |
| `/admin/` | Django Admin | Database administration |

## Configuration

Environment variables in `.env`:

```bash
# Django
DJANGO_SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=*

# Database (existing)
DB_HOST=clonehero_db
DB_NAME=clonehero
DB_USER=clonehero
DB_PASSWORD=clonehero
DB_PORT=5432

# API
API_URL=http://clonehero_api:8000

# File Uploads
MAX_FILE_SIZE_GB=10

# Logging
LOG_DIR=/var/log/frontend
LOG_LEVEL=INFO
```

## API Integration

The Django frontend communicates with the FastAPI backend:

- **Uploads**: POST to `/upload_content/` with multipart/form-data
- **List Content**: GET from `/list_content/?content_type=X`
- **Delete Content**: DELETE to `/delete_content/?content_type=X&file=Y`
- **Songs**: GET from `/content/`, `/songs/`
- **Process Songs**: POST to `/process_song/`

All API calls use `requests` library with proper error handling.

## Development Tips

### Adding a New Page

1. **Create view** in `content/views.py`:
```python
def my_new_page(request):
    context = {'data': 'example'}
    return render(request, 'my_page.html', context)
```

2. **Add URL** in `content/urls.py`:
```python
path('mypage/', views.my_new_page, name='my_page'),
```

3. **Create template** in `templates/my_page.html`:
```html
{% extends 'base.html' %}
{% block content %}
<h1>My New Page</h1>
{% endblock %}
```

4. **Add navigation** in `templates/base.html`:
```html
<li><a href="{% url 'content:my_page' %}">My Page</a></li>
```

### Debugging

```bash
# View logs
docker compose logs -f frontend

# Enter container
docker compose exec frontend bash

# Django shell
docker compose exec frontend python src/frontend_django/manage.py shell

# Check static files
docker compose exec frontend ls -la /app/src/frontend_django/staticfiles
```

### Testing AJAX

Use browser DevTools Network tab to inspect:
- Request payloads
- CSRF tokens
- Response data
- Error messages

## Common Issues

**Static files not loading**: Run `python manage.py collectstatic`

**CSRF errors**: Check CSRF token in form/AJAX request

**Database connection**: Ensure PostgreSQL service is healthy

**Import errors**: Verify `PYTHONPATH=/app` is set

## Production Deployment

The Dockerfile handles:
1. Installing dependencies from `requirements.txt`
2. Creating non-root `appuser`
3. Collecting static files on startup
4. Running Gunicorn with 2 workers on port 8501
5. Health checks via curl

No additional configuration needed for production deployment.

## Legacy Streamlit

The original Streamlit frontend is preserved in `src/frontend/` for reference. The Django version maintains 100% feature parity with improved performance and architecture.

## Support

See `MIGRATION_GUIDE.md` for detailed conversion information.
See `.github/copilot-instructions.md` for development guidelines.
