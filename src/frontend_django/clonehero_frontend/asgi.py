"""
ASGI config for clonehero_frontend project.
"""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'clonehero_frontend.settings')
application = get_asgi_application()
