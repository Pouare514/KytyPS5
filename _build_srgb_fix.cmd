@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul
set "PATH=C:\Program Files\LLVM\bin;C:\Program Files\CMake\bin;C:\Users\pouar\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cd /d C:\codes\KytyPS5-main\build\windows
ninja CMakeFiles/kyty_emulator.dir/graphics/host_gpu/renderer/imageView.cpp.obj kyty_emulator
echo BUILD_EXIT=%ERRORLEVEL%
if errorlevel 1 exit /b %ERRORLEVEL%
copy /Y kyty_emulator.exe install\kyty_emulator.exe
