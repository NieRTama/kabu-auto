@echo off
rem =====================================================================
rem kabu-auto モード切替ランチャ
rem ダブルクリックで対話メニューを起動する。
rem =====================================================================
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\switch_mode.ps1" %*
rem 終了コードで窓を閉じるか決める（switch_mode.ps1 の $AppExitCode を参照）。
rem   0 = アプリが正常終了した     → そのまま閉じる（停止のたびに窓が残らない）
rem   1 = アプリが異常終了した     → 原因を読めるよう残す
rem   2 = アプリを起動していない   → 切替結果を読めるよう残す
rem 無条件に閉じると、起動拒否（CONFIRM_LIVE_TRADING 未設定・多重起動）の
rem メッセージが一瞬で消えて原因が分からなくなるため、失敗時だけ残す。
if errorlevel 1 pause
