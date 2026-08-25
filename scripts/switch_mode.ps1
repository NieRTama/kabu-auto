# =====================================================================
# kabu-auto 取引モード切替スクリプト
# ---------------------------------------------------------------------
# config.yaml の trading.mode / kabu_station.base_url と .env の
# KABU_API_PASSWORD をまとめて切り替える。
#
# 接続先の自動整合ルール:
#   paper                     → 検証系 (18081 / 検証パスワード)  ※実口座から隔離
#   dry_run / semi_live / live → 本番系 (18080 / 本番パスワード)
#
# 使い方:
#   対話メニュー: switch-mode.bat をダブルクリック（または本ファイルを直接実行）
#   非対話:       powershell -File switch_mode.ps1 -Mode paper -NoLaunch
# =====================================================================
param(
    [ValidateSet('paper', 'dry_run', 'semi_live', 'live')]
    [string]$Mode,
    [switch]$NoLaunch
)

$ErrorActionPreference = 'Stop'

$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ConfigPath = Join-Path $RepoRoot 'config.yaml'
$EnvPath    = Join-Path $RepoRoot '.env'
$Utf8NoBom  = New-Object System.Text.UTF8Encoding($false)

function Read-Raw([string]$path)  { [System.IO.File]::ReadAllText($path) }
function Write-Raw([string]$path, [string]$text) { [System.IO.File]::WriteAllText($path, $text, $Utf8NoBom) }

# ─── パスワードは .env からのみ読む（値をこのファイルに書かない）──────────
# このスクリプトは「どの値を使うか」を切り替えるだけで、秘密そのものは保持しない。
# .env（.gitignore 済み）に次の2つを定義しておく:
#   KABU_API_PASSWORD_LIVE=<本番用APIパスワード>
#   KABU_API_PASSWORD_VERIFY=<検証用APIパスワード>
# 未定義なら KABU_API_PASSWORD は書き換えず、モードと接続先だけを切り替える（警告を表示）。
function Get-EnvValue([string]$name) {
    if (-not (Test-Path $EnvPath)) { return $null }
    $raw = Read-Raw $EnvPath
    if ($raw -match "(?m)^$([regex]::Escape($name))=(.*)$") { return $Matches[1].Trim() }
    return $null
}

function Get-Target([string]$m) {
    if ($m -eq 'paper') {
        return @{ Port = '18081'; PwVar = 'KABU_API_PASSWORD_VERIFY'; PwLabel = '検証用'; Env = '検証' }
    }
    return @{ Port = '18080'; PwVar = 'KABU_API_PASSWORD_LIVE'; PwLabel = '本番用'; Env = '本番' }
}

function Get-CurrentState {
    $cfg  = Read-Raw $ConfigPath
    $mode = if ($cfg -match '(?m)^[ \t]*mode:[ \t]*"([^"]*)"') { $Matches[1] } else { '不明' }
    $port = if ($cfg -match 'base_url:[ \t]*"http://localhost:(\d+)/kabusapi"') { $Matches[1] } else { '不明' }
    $pwLabel = '不明'
    if (Test-Path $EnvPath) {
        $envRaw = Read-Raw $EnvPath
        if ($envRaw -match '(?m)^KABU_API_PASSWORD=(.*)$') {
            $pwVal   = $Matches[1].Trim()
            $pwLive  = Get-EnvValue 'KABU_API_PASSWORD_LIVE'
            $pwVerif = Get-EnvValue 'KABU_API_PASSWORD_VERIFY'
            if ($pwVal -eq '')                          { $pwLabel = '未設定' }
            elseif ($pwLive  -and $pwVal -eq $pwLive)   { $pwLabel = '本番用' }
            elseif ($pwVerif -and $pwVal -eq $pwVerif)  { $pwLabel = '検証用' }
            else                                        { $pwLabel = 'その他' }
        }
    }
    return @{ Mode = $mode; Port = $port; PwLabel = $pwLabel }
}

function Backup-File([string]$path) {
    if (Test-Path $path) { Copy-Item $path "$path.bak" -Force }
}

function Set-Mode([string]$m) {
    $t = Get-Target $m
    Backup-File $ConfigPath
    Backup-File $EnvPath

    # config.yaml: trading.mode と base_url のポートを置換
    $cfg = Read-Raw $ConfigPath
    $cfg = [regex]::Replace($cfg, '(?m)^([ \t]*mode:[ \t]*")[^"]*(")', "`${1}$m`${2}")
    $cfg = [regex]::Replace($cfg, '(base_url:[ \t]*"http://localhost:)\d+(/kabusapi")', "`${1}$($t.Port)`${2}")
    Write-Raw $ConfigPath $cfg

    # .env: KABU_API_PASSWORD を、対象モード用の値（KABU_API_PASSWORD_LIVE/_VERIFY）で更新する。
    # 値が未定義なら書き換えない（誤って空パスワードにして起動不能にしないため）。
    $pw = Get-EnvValue $t.PwVar
    $t.PwApplied = $false
    if ([string]::IsNullOrEmpty($pw)) {
        Write-Host ("警告: .env に {0} が未定義のため、KABU_API_PASSWORD は変更しませんでした。" -f $t.PwVar) -ForegroundColor Yellow
        Write-Host "      .env に $($t.PwVar)=<パスワード> を追記してください（値はgit管理外の.envにのみ保存）。" -ForegroundColor Yellow
    } elseif (Test-Path $EnvPath) {
        # 置換値は MatchEvaluator で渡す（パスワードに $ が含まれても壊れないようにする）
        $envRaw = Read-Raw $EnvPath
        if ($envRaw -match '(?m)^KABU_API_PASSWORD=') {
            $envRaw = [regex]::Replace($envRaw, '(?m)^KABU_API_PASSWORD=.*$', { param($mt) "KABU_API_PASSWORD=$pw" })
        } else {
            $sep = if ($envRaw.EndsWith("`n")) { '' } else { "`r`n" }
            $envRaw = $envRaw + $sep + "KABU_API_PASSWORD=$pw`r`n"
        }
        Write-Raw $EnvPath $envRaw
        $t.PwApplied = $true
    }
    return $t
}

function Invoke-Switch([string]$m) {
    $t = Set-Mode $m
    $desc = @{ paper = 'ペーパー（仮想取引）'; dry_run = 'ドライラン'; semi_live = 'セミライブ'; live = 'ライブ（本番）' }[$m]
    Write-Host ''
    $pwState = if ($t.PwApplied) { $t.PwLabel } else { '変更なし' }
    Write-Host ("→ {0} に切り替えました（接続先: {1} {2} / パスワード: {3}）" -f $desc, $t.Env, $t.Port, $pwState) -ForegroundColor Green
    Write-Host '  バックアップ: config.yaml.bak / .env.bak を作成しました'
    return $t
}

function Start-App([string]$m) {
    Push-Location $RepoRoot
    try {
        if ($m -eq 'live' -or $m -eq 'semi_live') {
            $env:CONFIRM_LIVE_TRADING = 'true'
            Write-Host ''
            Write-Host '【実発注モード】CONFIRM_LIVE_TRADING=true を設定して起動します。' -ForegroundColor Yellow
        }
        Write-Host 'python main.py を起動します...'
        Write-Host ''
        python main.py
    } finally {
        Pop-Location
    }
}

# ─── 非対話モード（-Mode 指定時。テスト/ショートカット用）───────────
if ($Mode) {
    Invoke-Switch $Mode | Out-Null
    if (-not $NoLaunch) {
        if ($Mode -eq 'live' -or $Mode -eq 'semi_live') {
            Write-Host '非対話モードでは実発注モードの自動起動は行いません（誤起動防止）。メニューから起動してください。' -ForegroundColor Yellow
        } else {
            Start-App $Mode
        }
    }
    return
}

# ─── 対話メニュー ───────────────────────────────────────────────────
while ($true) {
    $s = Get-CurrentState
    Write-Host ''
    Write-Host '======================================'
    Write-Host '  kabu-auto モード切替'
    Write-Host '======================================'
    Write-Host ("  現在: {0} （接続先: {1} / パスワード: {2}）" -f $s.Mode, $s.Port, $s.PwLabel)
    Write-Host ''
    Write-Host '  [1] paper      ペーパー（仮想取引・発注なし）※実口座から隔離'
    Write-Host '  [2] dry_run    ドライラン（実口座読取・発注なし）'
    Write-Host '  [3] semi_live  セミライブ（承認後に実発注）'
    Write-Host '  [4] live       本番（実資金で実発注）'
    Write-Host '  [Q] 終了'
    Write-Host ''
    $raw = Read-Host '選択'
    if ($null -eq $raw) { Write-Host ''; Write-Host '入力が終了しました。'; return }  # stdin が EOF（Ctrl+Z 等）
    $sel = $raw.Trim().ToUpper()

    switch ($sel) {
        '1'  { $m = 'paper' }
        '2'  { $m = 'dry_run' }
        '3'  { $m = 'semi_live' }
        '4'  { $m = 'live' }
        'Q'  { Write-Host '終了します。'; return }
        default { Write-Host '無効な選択です。' -ForegroundColor Red; continue }
    }

    if ($m -eq 'live' -or $m -eq 'semi_live') {
        Write-Host ''
        Write-Host '⚠ 実際の資金で発注するモードです。' -ForegroundColor Red
        $c = Read-Host '本当に切り替えますか？（yes と入力で確定）'
        if ($c -ne 'yes') { Write-Host 'キャンセルしました。' -ForegroundColor Yellow; continue }
    }

    Invoke-Switch $m | Out-Null

    $l = Read-Host '今すぐ python main.py を起動しますか？ (y/N)'
    if ($null -ne $l -and $l.Trim().ToLower() -eq 'y') {
        Start-App $m
        return
    }
}
