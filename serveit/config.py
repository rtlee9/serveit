"""Static and environment variables."""
from os import getenv

WSGI_HOST = getenv('WSGI_HOST', '127.0.0.1')
WSGI_PORT = getenv('WSGI_PORT', '5000')
WSGI_WORKERS = getenv('WSGI_WORKERS', '1')
WSGI_OPTIONS = {
    'bind': '{}:{}'.format(WSGI_HOST, WSGI_PORT),
    'workers': int(WSGI_WORKERS),
}
