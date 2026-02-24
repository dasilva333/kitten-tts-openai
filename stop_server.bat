@echo off
title Stop KittenTTS Server
echo Stopping KittenTTS Server on port 8090...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8090 ^| findstr LISTENING') do (
    echo Killing process %%a
    taskkill /f /pid %%a
)
echo Server stopped.
pause
