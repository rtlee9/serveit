"""Static and environment variables."""
from os import getenv

WSGI_HOST = getenv('WSGI_HOST', '127.0.0.1')
WSGI_PORT = int(getenv('WSGI_PORT', 5000))
