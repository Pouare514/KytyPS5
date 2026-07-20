@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
set "PATH=C:\Program Files\LLVM\bin;C:\Program Files\CMake\bin;C:\Program Files\Git\cmd;C:\Users\pouar\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
set "VULKAN_SDK=C:\VulkanSDK\1.4.350.0"
cd /d C:\codes\KytyPS5-main
cmake -S src -B build/windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_PREFIX_PATH=C:/Qt/6.8.2/msvc2022_64 -DKYTY_BUILD_LAUNCHER=ON
if errorlevel 1 exit /b 1
cmake --build build/windows --target kyty_emulator -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b 1
copy /Y build\windows\kyty_emulator.exe build\windows\install\kyty_emulator.exe
echo RECONFIG_BUILD_OK
