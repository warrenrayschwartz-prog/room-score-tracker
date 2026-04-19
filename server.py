#!/usr/bin/env python3
"""Room Score Tracker — static file server for Railway deployment."""
import http.server
import os
import socketserver

PORT = int(os.environ.get('PORT', 8080))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

    def log_message(self, format, *args):
        pass  # suppress per-request noise

print(f'Room Score Tracker serving on port {PORT}')
with socketserver.TCPServer(('', PORT), Handler) as httpd:
    httpd.serve_forever()
