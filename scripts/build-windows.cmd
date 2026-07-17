@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
set "PATH=C:\Program Files\LLVM\bin;C:\Program Files\CMake\bin;C:\Program Files\Git\cmd;C:\Users\pouar\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
set "VULKAN_SDK=C:\VulkanSDK\1.4.350.0"
cd /d C:\codes\KytyPS5-main
cmake -S src -B _Build/windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_PREFIX_PATH=C:/Qt/6.8.2/msvc2022_64
if errorlevel 1 exit /b 1
cmake --build _Build/windows --target launcher kyty_emulator shader_cfg_tests scalar_provenance_tests page_manager_tests memory_tracker_tests shader_vertex_metadata_tests shader_stage_runtime_tests resource_tracking_tests resource_mutex_tests shader_recompiler_compute_tests virtual_memory_allocation_tests
if errorlevel 1 exit /b 1
cmake --install _Build/windows --prefix _Build/windows/install
if errorlevel 1 exit /b 1
ctest --test-dir _Build/windows --output-on-failure
exit /b %ERRORLEVEL%
