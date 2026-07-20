@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_validate_phase.ps1" -Phase 59 -Seconds 180 %*
