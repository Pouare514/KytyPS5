@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
set "PATH=C:\Program Files\LLVM\bin;C:\Program Files\CMake\bin;C:\Program Files\Git\cmd;C:\Users\pouar\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
set "VULKAN_SDK=C:\VulkanSDK\1.4.350.0"
cd /d C:\codes\KytyPS5-main
if not exist build\windows-debug\externals (
	if exist build\windows\externals (
		echo Reusing ffmpeg prebuilts from build\windows\externals...
		xcopy /E /I /Y build\windows\externals build\windows-debug\externals >nul
	)
)
cmake -S src -B build/windows-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_PREFIX_PATH=C:/Qt/6.8.2/msvc2022_64 -DKYTY_BUILD_LAUNCHER=ON
if errorlevel 1 exit /b 1
cmake --build build/windows-debug --target kyty_emulator launcher
if errorlevel 1 exit /b 1
cmake --install build/windows-debug --prefix build/windows-debug/install
exit /b %ERRORLEVEL%
