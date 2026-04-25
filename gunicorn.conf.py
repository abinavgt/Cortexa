import os

# Automatically bind to Render's $PORT
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Optimized for Render's free tier
workers = 1
threads = 2
timeout = 120
