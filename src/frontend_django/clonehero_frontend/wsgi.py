"""
WSGI config for clonehero_frontend project.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'clonehero_frontend.settings')
application = get_wsgi_application()
