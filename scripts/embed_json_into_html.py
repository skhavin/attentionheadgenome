"""
embed_json_into_html.py
Reads all_kv_geometry.json and inlines it directly into the HTML viewer
so the file works from file:// without a web server.
"""
import json
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
json_path = os.path.join(base, "outputs", "geometry", "all_kv_geometry.json")
html_path = os.path.join(base, "outputs", "geometry", "interactive_kv_viewer.html")

with open(json_path, "r", encoding="utf-8") as f:
    raw_json = f.read()

# Read the HTML template
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Inject data by replacing the placeholder comment
DATA_BLOCK = f"<script>window._DB_DATA = {raw_json};</script>"
html = html.replace("<!-- DATA_PLACEHOLDER -->", DATA_BLOCK)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Done! File size: {len(html) // 1024} KB at {html_path}")
