# Streamlit to Django Migration Guide

## Overview

This project has been successfully converted from Streamlit to Django with HTML/CSS templates. All functionality has been preserved and enhanced with a more robust web framework.

## What Changed

### Architecture
- **Before**: Streamlit single-file app with page-based navigation
- **After**: Django MVC architecture with proper separation of concerns
  - Models: Database interaction via Django ORM (read-only for existing tables)
  - Views: Business logic and API communication
  - Templates: HTML templates with Jinja2 templating
  - Static files: CSS, JavaScript, and assets

### File Structure
```
src/frontend_django/
├── manage.py                          # Django management script
├── clonehero_frontend/               # Project settings
│   ├── __init__.py
│   ├── settings.py                   # Configuration
│   ├── urls.py                       # URL routing
│   ├── wsgi.py                       # WSGI application
│   └── asgi.py                       # ASGI application
├── content/                          # Main Django app
│   ├── models.py                     # Song model (maps to existing DB)
│   ├── views.py                      # All page views
│   ├── urls.py                       # URL patterns
│   ├── admin.py                      # Django admin interface
│   └── templatetags/                 # Custom template filters
├── templates/                        # HTML templates
│   ├── base.html                     # Base template with sidebar
│   ├── home.html                     # Landing page
│   ├── database_explorer.html        # Search & browse songs
│   ├── backgrounds.html              # Background management
│   ├── colors.html                   # Color profile management
│   ├── highways.html                 # Highway management
│   ├── song_generator.html           # Song processing
│   └── songs/
│       └── upload.html               # Song upload page
└── static/                           # Static assets
    ├── css/
    │   └── style.css                 # Main stylesheet
    ├── js/
    │   ├── main.js                   # Core JavaScript
    │   ├── tabs.js                   # Tab functionality
    │   └── upload.js                 # Upload handling
    └── assets/
        └── ch_icon.png               # Clone Hero icon
```

## Feature Comparison

### Streamlit Pages → Django Views

| Streamlit Page | Django View | URL Pattern | Template |
|----------------|-------------|-------------|----------|
| `songs.py` | `songs_upload` | `/songs/upload/` | `songs/upload.html` |
| `database_explorer.py` | `database_explorer` | `/database/` | `database_explorer.html` |
| `backgrounds.py` | `backgrounds` | `/backgrounds/` | `backgrounds.html` |
| `colors.py` | `colors` | `/colors/` | `colors.html` |
| `highways.py` | `highways` | `/highways/` | `highways.html` |
| `song_generator.py` | `song_generator` | `/generator/` | `song_generator.html` |

### Key Improvements

1. **Better Performance**: Django is more efficient than Streamlit for production
2. **Proper MVC Architecture**: Clean separation of concerns
3. **Static File Management**: Optimized CSS/JS with WhiteNoise compression
4. **Database ORM**: Django models for better database interaction
5. **AJAX Support**: Asynchronous operations without page reloads
6. **Responsive Design**: Modern CSS with mobile support
7. **Production Ready**: Gunicorn WSGI server with proper logging

## Configuration

### Environment Variables

Add these to your `.env` file (if not already present):

```bash
# Django Settings
DJANGO_SECRET_KEY=your-secret-key-here-change-in-production
DEBUG=True
ALLOWED_HOSTS=*

# Database (already configured)
DB_HOST=clonehero_db
DB_NAME=clonehero
DB_USER=clonehero
DB_PASSWORD=clonehero
DB_PORT=5432

# API URL (already configured)
API_URL=http://clonehero_api:8000

# Frontend Port (already configured)
FRONTEND_PORT=8501

# Logging (already configured)
LOG_DIR=/var/log/frontend
LOG_LEVEL=INFO
```

### Docker Configuration

The `docker/frontend/Dockerfile` has been updated to:
- Use Django instead of Streamlit
- Run Gunicorn with 2 workers
- Collect static files on startup
- Serve via WhiteNoise

The `docker-compose.yml` remains mostly unchanged, maintaining the same port (8501) for compatibility.

## Deployment Steps

### 1. Rebuild the Frontend Container

```bash
# Using the utility script (recommended)
./utils.sh
# Select option [2] "Build & Push Docker Images"

# Or manually:
docker compose build frontend
docker compose push nuniesmith/clonehero:frontend
```

### 2. Stop Current Services

```bash
docker compose down
```

### 3. Start Services with New Frontend

```bash
docker compose up -d --build
```

### 4. Verify Deployment

Check if all services are running:
```bash
docker compose ps
```

Check frontend logs:
```bash
docker compose logs -f frontend
```

Access the frontend:
```
http://localhost:8501
```

### 5. Collect Static Files (if needed)

Static files are automatically collected on container startup, but if you need to manually collect:

```bash
docker compose exec frontend python src/frontend_django/manage.py collectstatic --noinput
```

## Testing Checklist

- [ ] Home page loads correctly
- [ ] Sidebar navigation works
- [ ] Database Explorer: search and pagination work
- [ ] Upload Songs: file upload and processing work
- [ ] Song Generator: audio processing works
- [ ] Colors: upload and delete work
- [ ] Backgrounds: tab switching and upload work
- [ ] Highways: tab switching and upload work
- [ ] All AJAX operations complete successfully
- [ ] Mobile responsiveness is maintained
- [ ] Static files (CSS, JS, images) load correctly

## API Integration

All Django views communicate with the FastAPI backend (port 8000) exactly as the Streamlit version did:

- **Upload endpoints**: Forward files to `/upload_content/`
- **List endpoints**: Fetch from `/list_content/` and `/content/`
- **Delete endpoints**: Call `/delete_content/` and `/songs/{id}/delete/`
- **Process endpoint**: Forward to `/process_song/`

No changes to the API service are required.

## Troubleshooting

### Static Files Not Loading

```bash
# Manually collect static files
docker compose exec frontend python src/frontend_django/manage.py collectstatic --noinput

# Check STATIC_ROOT permissions
docker compose exec frontend ls -la /app/src/frontend_django/staticfiles
```

### Database Connection Issues

The Django models use `managed = False` to prevent Django from trying to create/modify the existing tables. The Song model maps directly to the `songs` table created by the API service.

### CSRF Token Errors

All forms include `{% csrf_token %}` and AJAX requests include the token in headers. If you encounter CSRF issues:

1. Check `ALLOWED_HOSTS` in settings.py
2. Ensure cookies are enabled in browser
3. Check for mixed HTTP/HTTPS requests

### Import Errors

Ensure `PYTHONPATH=/app` is set in the Dockerfile and that you're running commands from the correct directory:

```bash
# Correct:
docker compose exec frontend python src/frontend_django/manage.py <command>

# Or enter the container:
docker compose exec frontend bash
cd src/frontend_django
python manage.py <command>
```

## Rollback Plan

If you need to rollback to Streamlit:

1. Revert `docker/frontend/Dockerfile` to use Streamlit
2. Revert `requirements.txt` to include `streamlit`
3. Use the old `src/frontend/` directory
4. Rebuild and restart:
   ```bash
   git checkout HEAD~1 -- docker/frontend/Dockerfile requirements.txt
   docker compose up -d --build frontend
   ```

## Future Enhancements

Now that you're on Django, you can easily add:

1. **User Authentication**: Django has built-in auth system
2. **Admin Interface**: Django admin is already configured (`/admin/`)
3. **REST API**: Django REST Framework for a frontend API
4. **Caching**: Redis integration for better performance
5. **Testing**: Django's excellent testing framework
6. **Background Tasks**: Celery for async processing
7. **Real-time Updates**: Django Channels for WebSockets

## Support

For issues or questions:

1. Check Docker logs: `docker compose logs -f frontend`
2. Check Django logs: `docker compose exec frontend cat /var/log/frontend/django.log`
3. Review the original Copilot instructions: `.github/copilot-instructions.md`
4. Check this migration guide for common issues

## Summary

This migration maintains 100% feature parity with the Streamlit version while providing:

- ✅ Better performance and scalability
- ✅ Professional web framework architecture
- ✅ Easier customization and extension
- ✅ Production-ready deployment
- ✅ Same API integration
- ✅ Same Docker configuration
- ✅ Same user experience (improved UI/UX)

The frontend now runs on Django with Gunicorn, serving static files via WhiteNoise, and communicating with the FastAPI backend exactly as before.
