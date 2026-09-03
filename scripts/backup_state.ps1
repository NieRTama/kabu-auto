# === kabu-auto 状態バックアップ ===
# 目的: GitHub に存在しない「ローカル生成物・運用状態・機密」をまとめて ZIP 化し、OneDrive へ退避する。
#       端末が故障しても、GitHub のコード + この ZIP の2つで復元できる状態を保つ。
# 原則: 稼働中の本体プロセス(python main.py / 8080)には一切触れない。DB は読み取り専用で開き、
#       SQLite オンラインバックアップAPI で整合スナップショットを取る（WAL 未反映分も含まれる）。
# 注意: 既定では .env と data/auth.json（＝証券APIパスワード等の機密）を含む。
#       機密を除いた ZIP が欲しい場合は -NoSecrets を付ける。

[CmdletBinding()]
param(
    # 退避先。既定は OneDrive の kabu-auto フォルダ。
    [string]$Dest = (Join-Path $env:USERPROFILE 'OneDrive\kabu-auto'),
    # 退避先に残す世代数（古いものから削除）。
    [int]$Keep = 7,
    # 指定すると .env / data/auth.json を含めない。
    [switch]$NoSecrets
)

$ErrorActionPreference = 'Stop'

$repo    = Split-Path -Parent $PSScriptRoot
$stamp   = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$staging = Join-Path $env:TEMP "kabu-auto-state_$stamp"
$zipPath = Join-Path $Dest "kabu-auto-state_$stamp.zip"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-Utf8File([string]$Path, [string]$Text) {
    # Set-Content/Out-File の既定エンコーディングは版により揺れるため使わない（Knowledge.md 1章）
    [System.IO.File]::WriteAllText($Path, $Text, $utf8NoBom)
}

if ($NoSecrets) { $secretLabel = '除外 (-NoSecrets)' } else { $secretLabel = '含める (.env / data/auth.json)' }

Write-Host '=== kabu-auto 状態バックアップ ===' -ForegroundColor Cyan
Write-Host "リポジトリ : $repo"
Write-Host "退避先     : $Dest"
Write-Host "機密の扱い : $secretLabel"

if (-not (Test-Path $Dest)) { New-Item -ItemType Directory -Path $Dest -Force | Out-Null }

# 前回が途中で失敗していると TEMP に作業ディレクトリや中間ZIPが残る。次回実行時に掃除しておく。
foreach ($stale in (Get-ChildItem $env:TEMP -Filter 'kabu-auto-state_*' -ErrorAction SilentlyContinue)) {
    Remove-Item $stale.FullName -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $staging -Force | Out-Null

# ------------------------------------------------------------------
# 1. DB スナップショット（稼働中でも安全なオンラインバックアップ）
# ------------------------------------------------------------------
Write-Host ''
Write-Host '[1/6] DB スナップショットを取得中...' -ForegroundColor Yellow
$dbSrc = Join-Path $repo 'data\kabu_auto.db'
$dbDir = Join-Path $staging 'data'
New-Item -ItemType Directory -Path $dbDir -Force | Out-Null
$dbDst = Join-Path $dbDir 'kabu_auto.db'

# Python の出力は CP932 コンソールで化けるため ASCII のみを出させる（Knowledge.md 1章）
$pySnapshot = @'
import sqlite3, sys, os
src, dst = sys.argv[1], sys.argv[2]
# mode=ro では開かない。異常終了で -wal だけが残り -shm が無い状態のとき
# -shm を作れず開けなくなる（最もバックアップが要る場面で失敗する）。
con = sqlite3.connect(src)
out = sqlite3.connect(dst)
con.backup(out)          # online backup API: WAL の未チェックポイント分も反映された整合スナップショット
out.close(); con.close()
chk = sqlite3.connect(dst)
res = chk.execute("PRAGMA integrity_check").fetchone()[0]
tabs = [r[0] for r in chk.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
rows = []
for t in tabs:
    n = chk.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0]
    rows.append("%s=%d" % (t, n))
chk.close()
print("integrity_check: " + res)
print("size_bytes: %d" % os.path.getsize(dst))
print("rows: " + ", ".join(rows))
if res != "ok":
    sys.exit(1)
'@
$pyFile = Join-Path $staging '_snapshot.py'
Write-Utf8File $pyFile $pySnapshot
$dbReport = & python $pyFile $dbSrc $dbDst
if ($LASTEXITCODE -ne 0) { throw "DB スナップショットの整合性チェックに失敗しました: $dbReport" }
Remove-Item $pyFile -Force
foreach ($line in $dbReport) { Write-Host "      $line" }

# ------------------------------------------------------------------
# 2. ローカルにしか無いファイル
# ------------------------------------------------------------------
Write-Host ''
Write-Host '[2/6] 設定・モデル・運用状態をコピー中...' -ForegroundColor Yellow

# git 管理外＝失うと復元できないもの
$files = @(
    'models\lgb_model.pkl',
    'models\lgb_model.meta.json',
    'models\lstm_model.pt',
    'models\lstm_model.meta.json',
    'risk_profile.json',
    'reference_capital.json',
    'data\trading_halt.json',
    'watchlist.json'
)
# git にもあるが、ZIP 単体で復元内容を確認できるよう同梱する（いずれも小さい）
$files += @('config.yaml', 'watchlists.json', 'requirements.txt', 'requirements-ml.txt', '.env.example')

if (-not $NoSecrets) { $files += @('.env', 'data\auth.json') }

$copied  = @()
$missing = @()
foreach ($rel in $files) {
    $src = Join-Path $repo $rel
    if (Test-Path $src) {
        $dst = Join-Path $staging $rel
        $dstDir = Split-Path -Parent $dst
        if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir -Force | Out-Null }
        Copy-Item $src $dst -Force
        $copied += $rel
    } else {
        $missing += $rel
    }
}
Write-Host "      コピー: $($copied.Count) 件 / 不在: $($missing.Count) 件"
if ($missing.Count -gt 0) {
    Write-Host "      不在(未導入なら想定内): $($missing -join ', ')" -ForegroundColor DarkGray
}

# ------------------------------------------------------------------
# 3. 直近ログ（復元には不要。障害調査用に14日分だけ）
# ------------------------------------------------------------------
Write-Host ''
Write-Host '[3/6] 直近14日分のログを収集中...' -ForegroundColor Yellow
$logSrc = Join-Path $repo 'log'
$logCount = 0
if (Test-Path $logSrc) {
    $logDst = Join-Path $staging 'log'
    New-Item -ItemType Directory -Path $logDst -Force | Out-Null
    $cutoff = (Get-Date).AddDays(-14)
    foreach ($f in (Get-ChildItem $logSrc -Filter '*.log' | Where-Object { $_.LastWriteTime -ge $cutoff })) {
        Copy-Item $f.FullName (Join-Path $logDst $f.Name) -Force
        $logCount++
    }
}
Write-Host "      ログ: $logCount 件"

# ------------------------------------------------------------------
# 4. マニフェストと復元手順
# ------------------------------------------------------------------
Write-Host ''
Write-Host '[4/6] マニフェスト・復元手順を生成中...' -ForegroundColor Yellow
Push-Location $repo
$gitCommit = (& git rev-parse HEAD 2>$null)
$gitShort  = (& git rev-parse --short HEAD 2>$null)
$gitDate   = (& git log -1 --format=%cI 2>$null)
$gitDirty  = (& git status --porcelain 2>$null | Measure-Object -Line).Lines
$gitRemote = (& git remote get-url origin 2>$null)
Pop-Location
$pyVersion = (& python --version 2>&1) -join ''

$hashLines = @()
foreach ($f in (Get-ChildItem $staging -Recurse -File)) {
    $h = Get-FileHash $f.FullName -Algorithm SHA256
    $rel = $f.FullName.Substring($staging.Length + 1)
    $hashLines += ('{0}  {1}  {2} bytes' -f $h.Hash.Substring(0, 16), $rel, $f.Length)
}
if ($NoSecrets) { $secretNote = 'なし (-NoSecrets)' } else { $secretNote = 'あり (.env / data/auth.json)' }

$nl = [Environment]::NewLine
$manifest = @"
kabu-auto 状態バックアップ マニフェスト
========================================
作成日時      : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
作成ホスト    : $env:COMPUTERNAME
リポジトリ    : $repo
git remote    : $gitRemote
git commit    : $gitCommit ($gitShort / $gitDate)
未コミット数  : $gitDirty
Python        : $pyVersion
機密の同梱    : $secretNote

DB スナップショット
$($dbReport -join $nl)

収録ファイル (SHA256 先頭16桁)
$($hashLines -join $nl)
"@
Write-Utf8File (Join-Path $staging 'MANIFEST.txt') $manifest

$restore = @"
# kabu-auto 復元手順

このZIPは「GitHubに無いもの」だけを収めている。コード本体はGitHubから取る。

## 前提
- Python $pyVersion（別バージョンだと models/lgb_model.pkl の読み込みに失敗しうる）
- kabuステーション（Windowsアプリ）のインストールとログイン
- 対象コミット: $gitShort （$gitDate）

## 手順

### 1. コードを取得し、バックアップ時点のコミットに合わせる

    git clone $gitRemote kabu-auto
    cd kabu-auto
    git checkout $gitCommit

最新版で動かすなら checkout は省略可。ただしDBスキーマの前後関係に注意する。

### 2. 依存をインストール

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

### 3. このZIPの中身をリポジトリ直下に上書き展開する

- data/kabu_auto.db      … 取引履歴・建玉・シグナル（最重要）
- data/auth.json         … ダッシュボードのログイン情報 ※機密同梱時のみ
- data/trading_halt.json … 取引停止スイッチの状態
- models/                … 学習済みモデル＋SHA256メタ
- .env                   … APIパスワード等 ※機密同梱時のみ
- config.yaml / watchlists.json / risk_profile.json / reference_capital.json

MANIFEST.txt のSHA256で展開後のファイルを照合できる。

### 4. .env を同梱していない場合

.env.example を写して再設定する（KABU_API_PASSWORD 等）。値は kabuステーション側の設定を参照。

### 5. 必ず paper モードで起動して健全性を確認してから live に戻す

    switch-mode.bat        (paper を選択)
    python main.py

### 6. live へ戻す前に建玉ドリフトを確認する

DBの建玉と実口座の建玉（数量・平均取得単価）が食い違うと kill switch が入り発注停止になる。
復元直後は特に食い違いやすいので、実口座と突き合わせてから live に切り替える。

## 収録していないもの
- data/backups/ の日次DB（同一ディスク上の冗長コピーのため）
- run_console.err.log 等の巨大ログ
- .venv / __pycache__（再作成できる）
"@
Write-Utf8File (Join-Path $staging 'RESTORE.md') $restore

# ------------------------------------------------------------------
# 5. ZIP 化して退避先へ
# ------------------------------------------------------------------
Write-Host ''
Write-Host '[5/6] ZIP を作成・検証して退避先へ配置中...' -ForegroundColor Yellow

# ZIP はまずローカル(TEMP)で作る。OneDrive 等の同期フォルダへ直接書き込むと、
# 同期クライアントが書き込み中のフォルダを掃除して Compress-Archive が
# 「path does not exist」で落ちることがある（実際に発生）。作成→検証→コピーの順にする。
$zipTmp = Join-Path $env:TEMP "kabu-auto-state_$stamp.zip"
Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zipTmp -CompressionLevel Optimal -Force

# 作成した ZIP が実際に開けるか検証する（作りっぱなしにしない）
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipTmp)
$entryCount = $zip.Entries.Count
# PS 5.1 の Compress-Archive はエントリ名を '\' 区切りで書く。'/' 前提で照合すると必ず外れるため正規化する。
$dbEntry = $zip.Entries | Where-Object { $_.FullName.Replace('\', '/') -eq 'data/kabu_auto.db' }
$zip.Dispose()
if ($null -eq $dbEntry) { throw "ZIP に data/kabu_auto.db が入っていません: $zipTmp" }
if ($dbEntry.Length -ne (Get-Item $dbDst).Length) {
    throw "ZIP 内の DB サイズがスナップショットと一致しません: $($dbEntry.Length) != $((Get-Item $dbDst).Length)"
}

# 退避先へコピー（直前に存在確認。同期クライアントが消していたら作り直す）
if (-not (Test-Path $Dest)) { New-Item -ItemType Directory -Path $Dest -Force | Out-Null }
Copy-Item $zipTmp $zipPath -Force
if (-not (Test-Path $zipPath)) { throw "退避先へのコピーに失敗しました: $zipPath" }
if ((Get-Item $zipPath).Length -ne (Get-Item $zipTmp).Length) {
    throw "退避先のファイルサイズが一致しません: $zipPath"
}

$zipSize = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Host "      $zipPath"
Write-Host "      エントリ $entryCount 件 / $zipSize MB / DB同梱 OK"

Remove-Item $zipTmp -Force
Remove-Item $staging -Recurse -Force

# ------------------------------------------------------------------
# 6. 世代管理（このスクリプトが作った ZIP のみを対象に削除）
# ------------------------------------------------------------------
Write-Host ''
Write-Host "[6/6] 古い世代を整理中（$Keep 世代を保持）..." -ForegroundColor Yellow
$olds = Get-ChildItem $Dest -Filter 'kabu-auto-state_*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -Skip $Keep
foreach ($o in $olds) {
    Remove-Item $o.FullName -Force
    Write-Host "      削除: $($o.Name)" -ForegroundColor DarkGray
}
$remain = (Get-ChildItem $Dest -Filter 'kabu-auto-state_*.zip').Count
Write-Host "      保持中の世代: $remain"

Write-Host ''
Write-Host '完了。' -ForegroundColor Green
Write-Host "  $zipPath"
if (-not $NoSecrets) {
    Write-Host '  ※ このZIPには .env / auth.json が平文で含まれる。共有リンクを作らないこと。' -ForegroundColor Yellow
}
