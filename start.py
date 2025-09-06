#!/usr/bin/env python3
import os
from waitress import serve
from app_simple import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}")
    serve(app, host="0.0.0.0", port=port, threads=4)
