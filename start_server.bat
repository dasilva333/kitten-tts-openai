@echo off
title KittenTTS API Server
cd /d "%~dp0"
echo Starting KittenTTS API Server on port 8090...
venv\Scripts\python.exe api_server.py
pause
