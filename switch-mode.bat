@echo off
rem =====================================================================
rem kabu-auto モード切替ランチャ
rem ダブルクリックで対話メニューを起動する。
rem =====================================================================
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\switch_mode.ps1" %*
pause
