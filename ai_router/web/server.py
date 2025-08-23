#!/usr/bin/env python3
"""Simple web server for AI Router UI"""

import os
import json
import asyncio
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

from ..router import AIModelRouter

class AIRouterHandler(SimpleHTTPRequestHandler):
    """Custom handler for AI Router web interface"""
    
    def __init__(self, *args, router=None, **kwargs):
        self.router = router
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.serve_ui()
        elif self.path.startswith('/api/'):
            self.handle_api_get()
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path.startswith('/api/'):
            self.handle_api_post()
        else:
            self.send_error(404)
    
    def serve_ui(self):
        """Serve the main UI HTML file"""
        ui_path = Path(__file__).parent / 'ui.html'
        
        if ui_path.exists():
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open(ui_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "UI file not found")
    
    def handle_api_get(self):
        """Handle API GET requests"""
        if self.path == '/api/stats':
            try:
                stats = self.router.get_stats()
                self.send_json_response(stats)
            except Exception as e:
                self.send_error_response(str(e))
        else:
            self.send_error(404)
    
    def handle_api_post(self):
        """Handle API POST requests"""
        if self.path == '/api/route':
            try:
                # Read request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                # Extract parameters
                prompt = data.get('prompt', '')
                task_type = data.get('taskType')
                
                if not prompt:
                    self.send_error_response("Prompt is required")
                    return
                
                # Route the request (this needs to be sync for the simple server)
                # In a real implementation, you'd use an async framework like FastAPI
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.router.route(prompt, task_type))
                loop.close()
                
                self.send_json_response(result)
                
            except json.JSONDecodeError:
                self.send_error_response("Invalid JSON")
            except Exception as e:
                self.send_error_response(str(e))
        else:
            self.send_error(404)
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        json_str = json.dumps(data, indent=2)
        self.wfile.write(json_str.encode('utf-8'))
    
    def send_error_response(self, message):
        """Send error response"""
        error_data = {"error": message, "success": False}
        self.send_response(400)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        json_str = json.dumps(error_data)
        self.wfile.write(json_str.encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def create_handler(router):
    """Create a handler with the router instance"""
    def handler(*args, **kwargs):
        AIRouterHandler(*args, router=router, **kwargs)
    return handler

def start_web_ui(port=8080, open_browser=True):
    """Start the web UI server"""
    print("🚀 Starting AI Router Web UI...")
    
    # Initialize router
    try:
        router = AIModelRouter()
        print("✅ AI Router initialized")
    except Exception as e:
        print(f"❌ Failed to initialize router: {e}")
        print("Make sure you have at least one API key set:")
        print("  export OPENAI_API_KEY=your-key")
        print("  export ANTHROPIC_API_KEY=your-key")
        print("  export GROQ_API_KEY=your-key")
        return
    
    # Change to the web directory to serve static files
    web_dir = Path(__file__).parent
    os.chdir(web_dir)
    
    # Create server
    handler = create_handler(router)
    httpd = HTTPServer(('localhost', port), handler)
    
    print(f"📡 Server starting on http://localhost:{port}")
    
    # Open browser
    if open_browser:
        def open_browser_delayed():
            time.sleep(1)  # Wait for server to start
            webbrowser.open(f'http://localhost:{port}')
        
        browser_thread = threading.Thread(target=open_browser_delayed)
        browser_thread.daemon = True
        browser_thread.start()
        print("🌐 Opening browser...")
    
    print("💡 Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
        httpd.shutdown()

if __name__ == "__main__":
    start_web_ui()