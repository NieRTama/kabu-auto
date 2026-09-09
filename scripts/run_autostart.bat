@echo off
rem =====================================================================
rem kabu-auto 自動起動ランチャ（タスクスケジューラ \kabu\kabu-auto 専用）。
rem Windows起動時にこのファイルが呼ばれ、kabu-auto をライブモードで起動する。
rem kabuステーションの起動・ログインは自動化しない（人が行う方針）。
rem
rem set と python の呼び出しは必ず別行に書く。set X=Y ^&^& cmd は環境変数の
rem 値に末尾スペースが混入し("true ")、main.py の厳密比較(== "true")で
rem 弾かれる（2026-09 に switch_mode.ps1 で実際に発生した罠と同じ）。
rem =====================================================================
cd /d C:\Users\garnet\kabu-auto
set CONFIRM_LIVE_TRADING=true

rem Pythonの実体を確認する。パス変更・再インストールに気づかず
rem 「起動していないのに誰も気づかない」状態を避けるため、redirect先の
rem err.log にも分かる形で書き残す（cmdの標準エラーは何も出さないため）。
set PYEXE=C:\Users\garnet\AppData\Local\Programs\Python\Python311\python.exe
if not exist "%PYEXE%" (
  echo [run_autostart] Python not found: %PYEXE% >> data\run_console.err.log
  exit /b 1
)

rem ログは無制限に追記され続けると肥大化する（実測: 3ヶ月放置で20MB・21万行。
rem WebSocket再接続の記録がほとんど）。起動のたびにサイズを見て、しきい値を
rem 超えていたら1世代だけ .old へ退避してから新規に書き始める。
powershell -NoProfile -Command "foreach ($f in @('data\run_console.log','data\run_console.err.log')) { $i = Get-Item $f -EA SilentlyContinue; if ($i -and $i.Length -gt 5MB) { Move-Item $f ($f + '.old') -Force } }"

"%PYEXE%" main.py >> data\run_console.log 2>> data\run_console.err.log
