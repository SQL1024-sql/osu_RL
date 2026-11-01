@echo off
set "DOTNET_ROOT=%LOCALAPPDATA%\Microsoft\dotnet10"
set "PATH=%LOCALAPPDATA%\Microsoft\dotnet10;%PATH%"
cd /d "%~dp0"
start "" "%~dp0osu!.exe"
