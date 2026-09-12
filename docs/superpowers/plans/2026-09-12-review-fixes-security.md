# レビュー是正 F12〜F14（認証・セッション・ログ）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 初期設定の乗っ取り経路を塞ぎ、ログアウトで全ての認証手段を失効させ、例外文字列から秘密情報が再びログへ出る経路を断つ。

**Architecture:** 認証は「ブラウザ向けの失効できるセッション」と「プログラム向けのヘッダートークン」に役割を分ける。初期設定はローカル操作かコンソールに一度だけ出る資格情報を必須にする。ログは構造化して、例外の文字列表現をそのまま流さない。

**Tech Stack:** Python 3.11 / FastAPI 0.104.1 / pytest / hashlib / secrets

**Spec:** `docs/kabu-auto-detailed-review_20260910.md`（F12・F13・F14、§7）

**現状確認（2026-09-12時点で3件とも現存）:**

| ID | 現存箇所 |
|---|---|
| F12 | `src/dashboard/app.py:182` `_AUTH_EXEMPT_PATHS` に `/api/setup` が含まれ、`src/core/auth.py:46` は読込失敗を `_data=None`（＝未設定）として扱う |
| F13 | `src/dashboard/app.py:378-384` `logout` は `kabu_session` しか破棄せず、`kabu_token` Cookie が残る |
| F14 | `src/core/alerts.py:119` が `last_error`（例外）をそのまま文字列化してログへ出す |

## Global Constraints

- 日時は **JST naive**。現在時刻は `src/core/clock.now()` / `clock.today()` を使い、`datetime.now()` を直接呼ばない。
- **秘密情報を新たにログ・DB・通知本文へ出さない。** 本計画は流出経路を塞ぐものであり、塞ぐ過程で別の経路を作らない。
- **LAN公開時（`dashboard.host: 0.0.0.0`）の既存の認証強制を弱めない。** 既存の `_auth_required` / `_dashboard_token` の仕組みは維持する。
- **既存の利用者をロックアウトしない。** 認証済みの利用者が、この変更だけで締め出されないこと。
- ファイルは UTF-8 **BOM無し**・LF で保存する。確認は `git show <rev>:<path>` でコミット済みblobに対して行う。
- テストは `pytest tests/<file>.py -v` で実行する。ネットワークへ出るテストを書かない。
- コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` を付ける。実装者自身のモデル名を書かない。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `src/core/auth.py`（改修） | 読込失敗と未設定の区別、セットアップ資格情報の発行と検証 |
| `src/dashboard/app.py`（改修） | `/api/setup` の保護、ログアウトでの全認証手段の失効 |
| `src/core/logger.py`（改修） | 秘密情報のマスキングフィルタ |
| `src/core/alerts.py`（改修） | 例外の文字列化をやめ、構造化した情報だけを出す |
| `tests/test_setup_protection.py`（新規） | 初期設定の保護、読込失敗時の挙動 |
| `tests/test_logout_revocation.py`（新規） | ログアウトでの全認証手段の失効 |
| `tests/test_secret_masking.py`（新規） | マスキングと、通知失敗時の出力 |

---

## Task 1: 認証ファイルの読込失敗を「未設定」と区別する

**Files:**
- Modify: `src/core/auth.py:32-54`
- Test: `tests/test_setup_protection.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `load(path, session_ttl_hours=None) -> None` — 読込失敗時に `_load_failed` を立てる（挙動の追加のみ）
  - `is_configured() -> bool` — **挙動不変**
  - `load_failed() -> bool` — 新規
  - `is_setup_allowed() -> bool` — 新規。未設定**かつ**読込失敗でないときだけ True

**背景（F12）:** 現行の `load()` は読み込み例外を捕まえて `_data = None` とする（`src/core/auth.py:45-47`）。その結果、**認証ファイルが破損しただけで「未設定」と同じ状態に戻り、`/api/setup` で誰でもユーザーを作れる**。ディスク障害や部分書き込みが、認証の初期化に化ける。

「読み込めなかった」と「そもそも無い」は別の事象として扱う。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_setup_protection.py` を新規作成する。

```python
"""初期設定の保護（F12）のテスト

認証ファイルの読込失敗を「未設定」と同じ状態に戻すと、ディスク障害が
認証の初期化に化ける。「読み込めなかった」と「そもそも無い」を区別する。
"""
import json

import pytest

from src.core import auth as auth_store


@pytest.fixture(autouse=True)
def _reset():
    """モジュール状態をテストごとに初期化する"""
    auth_store._data = None
    auth_store._load_failed = False
    yield
    auth_store._data = None
    auth_store._load_failed = False


class TestUnconfigured:
    def test_missing_file_is_unconfigured(self, tmp_path):
        auth_store.load(str(tmp_path / "absent.json"))
        assert auth_store.is_configured() is False
        assert auth_store.load_failed() is False

    def test_setup_is_allowed_when_truly_unconfigured(self, tmp_path):
        auth_store.load(str(tmp_path / "absent.json"))
        assert auth_store.is_setup_allowed() is True


class TestLoadFailure:
    def _broken(self, tmp_path):
        path = tmp_path / "auth.json"
        path.write_text("{ this is not valid json", encoding="utf-8")
        return str(path)

    def test_broken_file_is_recorded_as_a_failure(self, tmp_path):
        auth_store.load(self._broken(tmp_path))
        assert auth_store.load_failed() is True

    def test_broken_file_does_not_allow_setup(self, tmp_path):
        """破損しただけで誰でもユーザーを作れる状態に戻さない（F12の核心）"""
        auth_store.load(self._broken(tmp_path))
        assert auth_store.is_setup_allowed() is False

    def test_broken_file_is_still_not_configured(self, tmp_path):
        """is_configured の意味は変えない（認証は通さない）"""
        auth_store.load(self._broken(tmp_path))
        assert auth_store.is_configured() is False

    def test_empty_file_is_a_failure_not_a_reset(self, tmp_path):
        path = tmp_path / "auth.json"
        path.write_text("", encoding="utf-8")
        auth_store.load(str(path))
        assert auth_store.load_failed() is True
        assert auth_store.is_setup_allowed() is False


class TestSuccessfulLoad:
    def _valid(self, tmp_path):
        path = tmp_path / "auth.json"
        auth_store._path = path
        auth_store.create_user("someone", "correct horse battery staple")
        return str(path)

    def test_valid_file_clears_the_failure_flag(self, tmp_path):
        path = self._valid(tmp_path)
        auth_store._data = None
        auth_store._load_failed = True
        auth_store.load(path)
        assert auth_store.load_failed() is False
        assert auth_store.is_configured() is True

    def test_configured_never_allows_setup(self, tmp_path):
        auth_store.load(self._valid(tmp_path))
        assert auth_store.is_setup_allowed() is False

    def test_existing_credentials_still_verify(self, tmp_path):
        """既存の利用者をロックアウトしない"""
        auth_store.load(self._valid(tmp_path))
        assert auth_store.verify("someone", "correct horse battery staple") is True
        assert auth_store.verify("someone", "wrong") is False
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_setup_protection.py -v`
Expected: FAIL — `AttributeError: module 'src.core.auth' has no attribute 'load_failed'`

- [ ] **Step 3: 実装を修正**

`src/core/auth.py` のモジュール変数に `_load_failed` を足す（`_data` の宣言の直後）。

```python
_data: Optional[dict] = None
# 認証ファイルを「読み込もうとして失敗した」か。未設定（ファイルが無い）とは別物。
# 読込失敗を未設定と同じ状態に戻すと、ディスク障害や部分書き込みが
# 「初期設定をやり直せる状態」に化ける（レビューF12）。
_load_failed: bool = False
```

`load()` を次に置き換える。

```python
def load(path: str = "data/auth.json", session_ttl_hours: Optional[int] = None) -> None:
    """Authファイルのパスを設定し、存在すれば読み込む。

    **読込失敗は「未設定」と区別する。** 失敗したまま初期設定を許すと、
    ファイルが壊れただけで誰でもユーザーを作れる状態に戻ってしまう
    （レビューF12）。復旧は人が行う前提で、自動では初期化しない。
    """
    global _path, _data, _SESSION_TTL_HOURS, _load_failed
    _path = Path(path)
    if session_ttl_hours:
        _SESSION_TTL_HOURS = int(session_ttl_hours)
    if _path.exists():
        try:
            with open(_path, encoding="utf-8") as f:
                _data = json.load(f)
            _load_failed = False
            # ユーザーIDも値を出さない（ハッシュ化しているため平文では保持していない）
            logger.info(f"認証情報を読み込み: 設定済み ({_path})")
        except Exception as e:
            logger.error(f"認証ファイルの読み込みに失敗: {e}")
            _data = None
            _load_failed = True
    else:
        _data = None
        _load_failed = False
        logger.info(f"認証ファイル未作成（初回はGUIで初期設定）: {_path}")
```

`is_configured()` の直後に追加する。

```python
def load_failed() -> bool:
    """認証ファイルを読み込もうとして失敗したか。"""
    return _load_failed


def is_setup_allowed() -> bool:
    """初期設定（ユーザー作成）を許してよいか。

    **未設定かつ読込失敗でないときだけ許す。** ファイルが壊れている場合は
    「まだ設定していない」ではなく「壊れている」ので、初期設定ではなく
    復旧が必要（レビューF12）。
    """
    return not is_configured() and not _load_failed
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_setup_protection.py -v`
Expected: PASS（9件）

- [ ] **Step 5: 既存の認証テストが通ることを確認**

Run: `pytest tests/test_auth.py tests/test_dashboard_auth.py -v`
Expected: PASS（`is_configured` の意味を変えていないこと）

- [ ] **Step 6: BOM確認とコミット**

Run: `head -c 3 src/core/auth.py | xxd`（`2222 22` を確認。`efbb bf` なら下記で除去）

```python
for p in ["src/core/auth.py", "tests/test_setup_protection.py"]:
    with open(p, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(p, "wb") as f:
            f.write(data[3:])
```

```bash
git add src/core/auth.py tests/test_setup_protection.py
git commit -m "$(cat <<'EOF'
fix(auth): 認証ファイルの読込失敗が未設定と同じ扱いになっていた問題を修正

読み込み例外を握って_data=Noneとしていたため、ファイルが破損しただけで
「まだ設定していない」状態に戻り、初期設定で誰でもユーザーを作れた。
ディスク障害や部分書き込みが認証の初期化に化ける経路だった。
「読み込めなかった」と「そもそも無い」を区別し、前者では初期設定を
許さない（復旧は人が行う）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `/api/setup` を保護する

**Files:**
- Modify: `src/dashboard/app.py:182`（`_AUTH_EXEMPT_PATHS`）、`src/dashboard/app.py:347-354`（`setup_credentials`）
- Test: `tests/test_setup_protection.py`

**Interfaces:**
- Consumes: Task 1 の `auth_store.is_setup_allowed()`
- Produces:
  - `_setup_token: Optional[str]` — 起動時に一度だけコンソールへ出すセットアップ資格情報
  - `_is_local_request(request) -> bool`
  - `/api/setup` が「ローカルからの要求」または「正しいセットアップトークン」を要求する

**背景（F12）:** `/api/setup` は認証除外パスで、ハンドラー自体もセットアップ用トークンや接続元の確認をしない。LANから到達できる初回起動では、**先にこのAPIを呼べた第三者がアカウントを作成できる**。

**対策:** 次のいずれかを要求する。

1. **ローカルからの要求**（`127.0.0.1` / `::1`）— 端末の前にいる人だけが通る
2. **セットアップトークン** — 初期設定が必要なときだけ起動時にコンソールへ一度だけ出す。`X-Setup-Token` ヘッダーで渡す

**既存利用者への影響:** 設定済みなら `/api/setup` は元から 400 を返す。本変更で新たに締め出される利用者はいない。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_setup_protection.py` の末尾に追記する。

```python
class TestSetupEndpointProtection:
    """LANから到達できる初回起動で、第三者がアカウントを作れない（F12）"""

    def _client(self, tmp_path, monkeypatch, setup_token="s3tup-t0ken"):
        from fastapi.testclient import TestClient
        from src.dashboard import app as dash

        auth_store._path = tmp_path / "auth.json"
        auth_store._data = None
        auth_store._load_failed = False
        monkeypatch.setattr(dash, "_setup_token", setup_token)
        monkeypatch.setattr(dash, "_auth_required", True)
        return TestClient(dash.app)

    def test_rejects_remote_request_without_the_token(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        resp = client.post("/api/setup",
                           json={"username": "attacker", "password": "pw12345678"},
                           headers={"X-Forwarded-For": "192.168.1.50"})
        assert resp.status_code == 403
        assert auth_store.is_configured() is False

    def test_accepts_remote_request_with_the_token(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        resp = client.post("/api/setup",
                           json={"username": "owner", "password": "pw12345678"},
                           headers={"X-Setup-Token": "s3tup-t0ken"})
        assert resp.status_code == 200

    def test_rejects_a_wrong_token(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        resp = client.post("/api/setup",
                           json={"username": "attacker", "password": "pw12345678"},
                           headers={"X-Setup-Token": "wrong"})
        assert resp.status_code == 403

    def test_rejects_when_the_auth_file_is_broken(self, tmp_path, monkeypatch):
        """読込失敗中は正しいトークンでも初期設定させない（復旧が先）"""
        client = self._client(tmp_path, monkeypatch)
        auth_store._load_failed = True
        resp = client.post("/api/setup",
                           json={"username": "owner", "password": "pw12345678"},
                           headers={"X-Setup-Token": "s3tup-t0ken"})
        assert resp.status_code == 409

    def test_rejects_when_already_configured(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch)
        auth_store.create_user("owner", "pw12345678")
        resp = client.post("/api/setup",
                           json={"username": "second", "password": "pw12345678"},
                           headers={"X-Setup-Token": "s3tup-t0ken"})
        assert resp.status_code in (400, 409)


class TestLocalRequestDetection:
    def test_loopback_is_local(self):
        from src.dashboard import app as dash
        assert dash._is_local_address("127.0.0.1") is True
        assert dash._is_local_address("::1") is True

    def test_lan_address_is_not_local(self):
        from src.dashboard import app as dash
        assert dash._is_local_address("192.168.1.50") is False
        assert dash._is_local_address("10.0.0.2") is False

    def test_unknown_address_is_not_local(self):
        from src.dashboard import app as dash
        assert dash._is_local_address(None) is False
        assert dash._is_local_address("") is False
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_setup_protection.py -v`
Expected: FAIL — `AttributeError: module 'src.dashboard.app' has no attribute '_setup_token'`

- [ ] **Step 3: 実装を修正**

`src/dashboard/app.py` の `_dashboard_token` の宣言の近くに追加する。

```python
# 初期設定用の一度きりの資格情報。初期設定が必要なときだけ起動時に生成し、
# コンソールへ一度だけ出す。LANから到達できる初回起動で、先に /api/setup を
# 呼べた第三者がアカウントを作れてしまう経路を塞ぐ（レビューF12）。
_setup_token: Optional[str] = None

_LOCAL_ADDRESSES = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_local_address(addr: Optional[str]) -> bool:
    """ループバックからの要求か（端末の前にいる人だけが通る）。"""
    return bool(addr) and addr in _LOCAL_ADDRESSES
```

`_AUTH_EXEMPT_PATHS` はそのまま（`/api/setup` はミドルウェアを通さず、ハンドラー自身が判定する）。判定をハンドラーに置くのは、拒否の理由（未設定でない／読込失敗中／資格が無い）を区別して返すためである。

`setup_credentials` を次に置き換える。

```python
@app.post("/api/setup")
async def setup_credentials(req: CredentialsRequest, request: Request):
    """初期設定：ユーザーID・パスワードを作成する。

    **ローカルからの要求か、起動時に一度だけ出したセットアップトークンを要求する。**
    LANから到達できる初回起動では、先にこのAPIを呼べた第三者がアカウントを
    作成できてしまうため（レビューF12）。

    認証ファイルが読み込めていない場合は初期設定を許さない。壊れているだけで
    「まだ設定していない」と同じ扱いになると、ディスク障害が認証の初期化に化ける。
    """
    if auth_store.load_failed():
        raise HTTPException(
            status_code=409,
            detail="認証ファイルを読み込めていません。初期設定ではなく復旧が必要です",
        )
    if not auth_store.is_setup_allowed():
        raise HTTPException(status_code=400, detail="初期設定は既に完了しています")

    provided = request.headers.get("X-Setup-Token")
    token_ok = bool(provided and _setup_token
                    and secrets.compare_digest(provided, _setup_token))
    if not (_is_local_address(_client_ip(request)) or token_ok):
        logger.warning(
            f"初期設定の要求を拒否しました（接続元={_client_ip(request)}）")
        raise HTTPException(
            status_code=403,
            detail="初期設定は端末のローカル操作か、起動時に表示された"
                   "セットアップトークン（X-Setup-Token）が必要です",
        )

    try:
        auth_store.create_user(req.username, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _issue_session(JSONResponse({"status": "ok", "message": "初期設定が完了しました"}))
```

- [ ] **Step 4: 起動時のトークン発行を結線する**

`src/dashboard/app.py` の起動処理（`_dashboard_token` を決めている箇所と同じ関数）から `_init_setup_token()` を呼ぶ。**初期設定が必要なときだけ**発行し、コンソールへ一度だけ出す。テストから直接叩けるよう独立した関数にする。

**トークンは `logger` へ渡さない。** `logger.warning()` はファイルシンクにも
配送されるため、初期設定が完了する前にログを読める者へ、LAN経由の初期ユーザー
作成に必要な資格情報が渡る（外部レビューR15）。マスク関数で伏せるのも当てに
できない — 和文の説明に続く裸のトークンは `token=` にも `Authorization:` にも
一致しない。**専用のローカルコンソール出力へ分離する。**

```python
def _print_setup_token(token: str) -> None:
    """初期設定トークンを**コンソールにだけ**出す。

    loguru を通さない。logger はファイルシンク・通知シンクへ配送するため、
    ログを読める者に資格情報が渡る（外部レビューR15）。
    このプロセスを起動した端末の標準エラー出力にだけ書く。

    ログには「発行した」という事実だけを残し、値は残さない。
    """
    import sys

    print("=" * 64, file=sys.stderr)
    print("初期設定用トークン（この起動でのみ有効・一度だけ表示）", file=sys.stderr)
    print(f"  {token}", file=sys.stderr)
    print("LAN経由で初期設定する場合は HTTP ヘッダー X-Setup-Token に指定して",
          file=sys.stderr)
    print("ください。端末のローカル操作（127.0.0.1）なら不要です。", file=sys.stderr)
    print("=" * 64, file=sys.stderr)


def _init_setup_token() -> None:
    """初期設定が必要なときだけトークンを発行する。

    起動処理から呼ぶ。テストが出力先ごとに秘密値の有無を確認できるよう、
    起動処理に埋め込まず独立した関数にする（外部レビューR15）。
    """
    global _setup_token
    if auth_store.is_setup_allowed():
        _setup_token = secrets.token_urlsafe(24)
        _print_setup_token(_setup_token)
        # ログには事実だけ。値は書かない
        logger.warning(
            "初期設定用トークンを発行しました（値はこの端末のコンソールにのみ表示）。"
        )
    else:
        _setup_token = None
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `pytest tests/test_setup_protection.py -v`
Expected: PASS（21件）

- [ ] **Step 6: 既存のダッシュボード認証テストが通ることを確認**

Run: `pytest tests/test_dashboard_auth.py tests/test_dashboard_phase3.py -v`
Expected: PASS

- [ ] **Step 7: コミット**

```bash
git add src/dashboard/app.py tests/test_setup_protection.py
git commit -m "$(cat <<'EOF'
fix(dashboard): 初期設定を先に呼べた第三者がアカウントを作れた問題を修正

/api/setup は認証除外で、ハンドラー自身もセットアップ用トークンや
接続元を確認していなかった。LANから到達できる初回起動では、先にこの
APIを呼べた第三者がアカウントを作成できた。
ローカル操作か、初期設定が必要なときだけ起動時にコンソールへ一度だけ
出すトークンを要求する。認証ファイルが読み込めていない場合は
初期設定ではなく復旧が必要なので409で拒否する。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: ログアウトで全ての認証手段を失効させる

**Files:**
- Modify: `src/dashboard/app.py:378-384`（`logout`）、`_has_valid_token`
- Modify: `config.yaml`（`dashboard.allow_query_token` / `dashboard.query_token_until` を追加。既定は `false` / 未設定）
- Test: `tests/test_logout_revocation.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `logout` が `kabu_session` と `kabu_token` の両方を削除する
  - `_has_valid_token(request)` が **Cookie 由来のトークンを受け付けない**。既定では `X-API-Token` ヘッダーのみ。`?token=` は `dashboard.allow_query_token: true` の移行期間中だけ
  - `_query_token_allowed() -> bool` — `dashboard.allow_query_token`（既定 `false`）と `dashboard.query_token_until`（期限、省略可）で判定する

**背景（F13）:** ミドルウェアは `kabu_session` **または** `kabu_token` Cookie を認証として読む（`src/dashboard/app.py:185-191`）。ログアウトはセッションだけを破棄するため、**過去にクエリトークンで発行した `kabu_token` が残っていれば、ログアウト後も認証が続く**。

**設計（spec §7 の方針）:** ブラウザは**失効できるセッションだけ**に揃える。プログラム向けのAPIトークンは**ヘッダー限定**にする。クエリトークンからCookieへ移す既存の導線は残すが、そのCookieは「次の遷移でセッションを得るまでの繋ぎ」ではなく**認証そのものには使わない**。

**既存利用者への影響:** ブラウザで `?token=` 付きURLを開く運用は、リダイレクト後にセッションが無いと再びログイン画面へ行く。そのため**クエリトークンでの到達時にセッションを発行する**ように変える。curl 等は `X-API-Token` ヘッダーで従来どおり通る。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_logout_revocation.py` を新規作成する。

```python
"""ログアウトでの認証失効（F13）のテスト

ミドルウェアは kabu_session または kabu_token Cookie を認証として読むが、
ログアウトはセッションしか破棄しないため、過去に発行した kabu_token が
残っていると認証が続いてしまう。
"""
import pytest
from fastapi.testclient import TestClient

from src.core import auth as auth_store
from src.dashboard import app as dash


@pytest.fixture
def client(tmp_path, monkeypatch):
    auth_store._path = tmp_path / "auth.json"
    auth_store._data = None
    auth_store._load_failed = False
    auth_store.create_user("owner", "pw12345678")
    monkeypatch.setattr(dash, "_auth_required", True)
    monkeypatch.setattr(dash, "_dashboard_token", "api-t0ken")
    return TestClient(dash.app)


class TestTokenCookieIsNotAuthentication:
    def test_cookie_token_alone_does_not_authenticate(self, client):
        """kabu_token Cookie だけでは通らない（ヘッダー限定にする）"""
        client.cookies.set("kabu_token", "api-t0ken")
        resp = client.get("/api/status")
        assert resp.status_code == 401

    def test_header_token_still_authenticates(self, client):
        """プログラム向けのヘッダートークンは従来どおり通る"""
        resp = client.get("/api/status", headers={"X-API-Token": "api-t0ken"})
        assert resp.status_code == 200

    def test_query_token_still_authenticates(self, client):
        resp = client.get("/api/status", params={"token": "api-t0ken"})
        assert resp.status_code == 200


class TestLogoutRevokesEverything:
    def _login(self, client):
        resp = client.post("/api/login",
                           json={"username": "owner", "password": "pw12345678"})
        assert resp.status_code == 200

    def test_logout_ends_the_session(self, client):
        self._login(client)
        assert client.get("/api/status").status_code == 200
        client.post("/api/logout")
        assert client.get("/api/status").status_code == 401

    def test_logout_clears_the_token_cookie_too(self, client):
        """ログアウト後に kabu_token が残って認証が続かない（F13の核心）"""
        self._login(client)
        client.cookies.set("kabu_token", "api-t0ken")
        client.post("/api/logout")
        assert client.get("/api/status").status_code == 401

    def test_logout_is_idempotent(self, client):
        self._login(client)
        assert client.post("/api/logout").status_code == 200
        assert client.post("/api/logout").status_code == 200

    def test_logout_without_a_session_does_not_error(self, client):
        assert client.post("/api/logout").status_code == 200


class TestQueryTokenIsDisabledByDefault:
    """URLクエリのAPIトークンは既定で受け付けない（外部レビューR16）。

    リダイレクトしても、最初のHTTPリクエストにトークンが載った時点で
    アクセスログ・Referer への流出経路は成立している。漏れたAPIトークンからは
    ログアウト後も新しいセッションを取れる。既定は無効にする。
    """

    def test_query_token_does_not_authenticate_by_default(self, client):
        cfg.get_section("dashboard").pop("allow_query_token", None)
        assert client.get(
            "/api/status", params={"token": "api-t0ken"}).status_code == 401

    def test_header_token_still_authenticates(self, client):
        assert client.get(
            "/api/status", headers={"X-API-Token": "api-t0ken"}
        ).status_code == 200


class TestQueryTokenDuringMigration:
    """移行期間中だけ受け付ける。既存ユーザーを締め出さないための繋ぎ。"""

    def _enable(self, until=None):
        conf = cfg.get_section("dashboard")
        conf["allow_query_token"] = True
        if until is None:
            conf.pop("query_token_until", None)
        else:
            conf["query_token_until"] = until

    def test_query_token_grants_a_usable_session_when_enabled(self, client):
        """?token= で到達したブラウザは、以後セッションで通る

        Cookieトークンを認証から外したので、代わりにセッションを発行する。
        これが無いと、?token= 付きURLを開く運用が次の遷移で止まる。
        """
        self._enable()
        client.get("/api/status", params={"token": "api-t0ken"})
        assert client.get("/api/status").status_code == 200

    def test_that_session_is_revoked_by_logout(self, client):
        self._enable()
        client.get("/api/status", params={"token": "api-t0ken"})
        client.post("/api/logout")
        assert client.get("/api/status").status_code == 401

    def test_usage_is_logged_as_a_warning(self, client, caplog):
        self._enable()
        with caplog.at_level("WARNING"):
            client.get("/api/status", params={"token": "api-t0ken"})
        assert any("X-API-Token" in r.message for r in caplog.records)

    def test_an_expired_migration_window_rejects_the_query_token(self, client):
        """期限を過ぎたら設定が true でも受け付けない"""
        self._enable(until="2020-01-01")
        assert client.get(
            "/api/status", params={"token": "api-t0ken"}).status_code == 401

    def test_a_future_window_still_accepts(self, client):
        self._enable(until="2099-12-31")
        assert client.get(
            "/api/status", params={"token": "api-t0ken"}).status_code in (200, 303)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_logout_revocation.py -v`
Expected: FAIL — `test_cookie_token_alone_does_not_authenticate` が 200 を返す（Cookie で通ってしまう）

- [ ] **Step 3: `_has_valid_token` を ヘッダー・クエリ限定にする**

`src/dashboard/app.py` の `_has_valid_token` を次に置き換える。

```python
def _has_valid_token(request: Request) -> bool:
    """X-API-Token ヘッダー / ?token= クエリのいずれかが有効か。

    **Cookie（kabu_token）は認証として読まない。** ログアウトはセッションを
    破棄するが、過去にクエリトークンで発行した Cookie が残っていると
    認証が続いてしまうため（レビューF13）。ブラウザの認証は失効できる
    セッションだけに揃え、プログラム向けはヘッダー限定にする。
    """
    provided = request.headers.get("X-API-Token")
    if provided is None and _query_token_allowed():
        # 移行期間のみ。既定は無効（下記の判断事項を参照）
        provided = request.query_params.get("token")
    return bool(provided and _dashboard_token is not None
                and secrets.compare_digest(provided, _dashboard_token))


def _query_token_allowed() -> bool:
    """URLクエリでのAPIトークンを受け付けるか。**既定は False。**

    `config.yaml` の `dashboard.allow_query_token`（既定 `false`）と
    `dashboard.query_token_until`（YYYY-MM-DD、省略可）で制御する。
    期限を過ぎたら設定が true でも受け付けない。
    """
    conf = cfg.get_section("dashboard")
    if not conf.get("allow_query_token", False):
        return False
    until = conf.get("query_token_until")
    if until and clock.today() > date.fromisoformat(str(until)):
        logger.warning(
            f"URLクエリトークンの移行期限（{until}）を過ぎています。"
            "X-API-Token ヘッダーへ切り替えてください")
        return False
    return True
```

> **判断事項: URLクエリトークンをいつ止めるか（外部レビューR16）**
>
> レビューの指摘は正しい。`?token=` を受理してセッションを発行し、その後
> トークンの無いURLへリダイレクトしても、**最初のHTTPリクエストに長期APIトークンが
> 載る**。アクセスログ・プロキシログ・Referer への流出経路は残る。漏れた
> APIトークンからはログアウト後も新しいセッションを取得できる。
> 「リダイレクトしたこと」を漏えい防止の合格条件にはしない。
>
> 一方で**即時廃止はユーザーをロックアウトする**。現行の運用はブックマークした
> `?token=` 付きURLからの接続に依存している可能性が高く、それを黙って切ると
> LAN内の端末からダッシュボードへ入れなくなる。既存ユーザーを締め出さないことは
> 本計画の制約である。
>
> **本計画の決定:**
>
> | | 扱い |
> |---|---|
> | ブラウザの認証 | ログイン → **セッションのみ**。Cookieトークンは認証として読まない |
> | プログラムからのアクセス | **`X-API-Token` ヘッダーのみ** |
> | `?token=` | **既定で無効**（`allow_query_token: false`）。明示的に有効化したときだけ、移行期間中に限り受理する |
> | 有効化したとき | 使用のたびに WARNING を出し、ダッシュボードへ移行案内を表示。`query_token_until` の期限を過ぎたら受理しない |
>
> 移行手順はユーザーが決める事項として残す。**期限の既定値は置かない**
> （運用を止めないことが優先のため）。ユーザーがヘッダー認証へ移り終えたら
> `allow_query_token` を消し、この分岐ごと削除する。

- [ ] **Step 4: クエリトークンでの到達時にセッションを発行する**

ミドルウェアの `?token=` 処理（`_set_token_cookie(redirect)` と `_set_token_cookie(response)` の2箇所）を、セッションの発行に差し替える。

```python
    # 既定では受け付けない。有効化されている移行期間中だけ真になる
    # （外部レビューR16。判断事項は Step 3 の表を参照）
    has_query_token = bool(
        request.query_params.get("token")) and _query_token_allowed()
    if has_query_token:
        # **リダイレクトは漏えい防止の合格条件ではない。** 最初のHTTP
        # リクエストに長期APIトークンが載った時点で、アクセスログ・
        # プロキシログ・Referer への流出経路は既に成立している。
        # 使われたこと自体を記録し、移行を促す。
        logger.warning(
            "URLクエリのAPIトークンで到達しました。アクセスログ・Referer に"
            "トークンが残ります。X-API-Token ヘッダーへ移行してください"
        )
    # ブラウザで ?token= 付きURLを直開きした場合、**セッションを発行**してから
    # URLからトークンを取り除いたクリーンなURLへ即リダイレクトする。
    # Cookieトークンを認証から外した（F13）ため、繋ぎとしてセッションを渡す。
    # 以後のリクエストにトークンが載らなくなる効果はあるが、既に載った
    # 1回目は取り消せない。
    if has_query_token and accepts_html and request.method == "GET" \
            and not path.startswith("/api/"):
        clean = request.url.remove_query_params("token")
        target = clean.path + (f"?{clean.query}" if clean.query else "")
        return _issue_session(RedirectResponse(url=target, status_code=303))

    response = await call_next(request)
    if has_query_token:
        # API/XHR で ?token= が来た場合も、以後の fetch はセッションで通す
        return _issue_session(response)
    return response
```

- [ ] **Step 5: ログアウトで両方の Cookie を削除する**

`logout` を次に置き換える。

```python
@app.post("/api/logout")
async def logout(request: Request):
    """ログアウト（セッション破棄・**全ての認証Cookie**を削除）。

    kabu_session だけを消すと、過去に発行した kabu_token が残って認証が
    続いてしまっていた（レビューF13）。認証に使いうるCookieは全て消す。
    """
    auth_store.destroy_session(request.cookies.get("kabu_session"))
    response = JSONResponse({"status": "ok", "message": "ログアウトしました"})
    response.delete_cookie("kabu_session")
    response.delete_cookie("kabu_token")
    return response
```

- [ ] **Step 6: テストを実行して成功を確認**

Run: `pytest tests/test_logout_revocation.py -v`
Expected: PASS（10件）

- [ ] **Step 7: 既存の認証テストが通ることを確認**

Run: `pytest tests/test_dashboard_auth.py tests/test_dashboard_x.py tests/test_dashboard_positions.py -v`
Expected: PASS。`X-API-Token` ヘッダー経由の既存テストは影響を受けない

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

- [ ] **Step 8: コミット**

```bash
git add src/dashboard/app.py tests/test_logout_revocation.py
git commit -m "$(cat <<'EOF'
fix(dashboard): ログアウト後もAPIトークンCookieで認証が続いた問題を修正

ミドルウェアはkabu_sessionまたはkabu_token Cookieを認証として読むが、
ログアウトはセッションしか破棄しないため、過去にクエリトークンで発行した
Cookieが残っていると認証が続いた。
ブラウザの認証は失効できるセッションだけに揃え、プログラム向けの
APIトークンはヘッダー限定にする。?token=での到達時は代わりにセッションを
発行するので、URLからトークンを消す既存の導線は維持される。
ログアウトでは認証に使いうるCookieを全て削除する。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: ログの秘密情報マスキング

**Files:**
- Modify: `src/core/logger.py`
- Test: `tests/test_secret_masking.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `mask_secrets(text: str) -> str` — Webhook URL・APIトークン・パスワード風の値を伏せる
  - `_masked_exception(exc: BaseException) -> BaseException` — 例外の文字列表現を伏せた複製（連鎖例外もたどる。元の例外は書き換えない）
  - `_mask_record(record) -> bool` — loguru フィルタ。`record["message"]` と **`record["exception"]`** の両方を処理する
  - loguru のフォーマッタ／パッチで全ログ出力に適用する

**背景（F14）:** 今回のレビューで実際に、保存済みログから Discord Webhook の ID とトークンを含むURLが1件見つかった。`src/core/alerts.py:119` は `last_error`（例外）をそのまま文字列化しており、**通知失敗の例外にWebhook URLが含まれれば再びログへ残る**。`logger.py` の `diagnose=False` は変数展開を防ぐが、この明示的な例外文字列の出力は防げない。

**マスクの対象:**

| 種類 | パターン |
|---|---|
| Discord Webhook | `https://discord(app)?.com/api/webhooks/<id>/<token>` |
| 汎用の webhook パス | `/api/webhooks/<id>/<token>` |
| クエリのトークン | `token=...` / `api_key=...` / `password=...` |
| ヘッダー風 | `X-API-Token: ...` / `Authorization: ...` |

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_secret_masking.py` を新規作成する。

```python
"""ログの秘密情報マスキング（F14）のテスト

今回のレビューで、保存済みログからDiscord WebhookのIDとトークンを含む
URLが実際に1件見つかった。例外の文字列表現をそのままログへ流す経路が
残っていると、同じことが再び起きる。
"""
from src.core.logger import mask_secrets

_WEBHOOK = ("https://discord.com/api/webhooks/"
            "1234567890123456789/abcDEF-ghiJKL_mnoPQR012345678901234567890123456789")


class TestWebhookMasking:
    def test_masks_a_full_discord_webhook_url(self):
        got = mask_secrets(f"通知に失敗しました: {_WEBHOOK}")
        assert "abcDEF-ghiJKL" not in got
        assert "1234567890123456789" not in got
        assert "REDACTED" in got

    def test_masks_the_discordapp_host_too(self):
        url = _WEBHOOK.replace("discord.com", "discordapp.com")
        got = mask_secrets(url)
        assert "abcDEF-ghiJKL" not in got

    def test_masks_a_hostless_webhook_path(self):
        """ホスト名の無い /api/webhooks/... 形式も伏せる"""
        got = mask_secrets("path=/api/webhooks/999/secrettoken")
        assert "secrettoken" not in got

    def test_keeps_the_surrounding_message(self):
        got = mask_secrets(f"通知に失敗しました: {_WEBHOOK}")
        assert "通知に失敗しました" in got


class TestQueryAndHeaderMasking:
    def test_masks_a_query_token(self):
        got = mask_secrets("GET /api/status?token=s3cr3tvalue")
        assert "s3cr3tvalue" not in got
        assert "/api/status" in got

    def test_masks_an_api_key(self):
        assert "AKIA123" not in mask_secrets("api_key=AKIA123456")

    def test_masks_a_password(self):
        assert "hunter2" not in mask_secrets("password=hunter2")

    def test_masks_an_api_token_header(self):
        assert "t0k3nvalue" not in mask_secrets("X-API-Token: t0k3nvalue")

    def test_masks_an_authorization_header_including_the_token(self):
        """値に空白があってもトークン本体まで伏せる。

        空白で止めると「Authorization: Bearer abc123」が
        「[REDACTED] abc123」になり、肝心のトークンが残る。
        """
        got = mask_secrets("Authorization: Bearer abc123")
        assert "abc123" not in got
        assert "Bearer" not in got

    def test_header_masking_does_not_eat_the_next_line(self):
        got = mask_secrets("Authorization: Bearer abc123\n発注: 7203 100株")
        assert "abc123" not in got
        assert "発注: 7203 100株" in got


class TestHarmlessTextIsUntouched:
    def test_ordinary_message_passes_through(self):
        msg = "シグナルスキャン開始... batch=20260912T162000-ab12cd34"
        assert mask_secrets(msg) == msg

    def test_symbol_and_price_are_not_masked(self):
        msg = "発注: 7203 100株 @2500円"
        assert mask_secrets(msg) == msg

    def test_empty_and_none_are_safe(self):
        assert mask_secrets("") == ""
        assert mask_secrets(None) == ""


class TestAppliedToLogOutput:
    def test_logger_output_is_masked(self, tmp_path):
        """設定した logger を通した出力が実際にマスクされる"""
        from loguru import logger as loguru_logger
        from src.core import logger as log_mod

        path = tmp_path / "test.log"
        sink_id = loguru_logger.add(
            str(path), format="{message}", level="INFO",
            filter=log_mod._mask_record)
        try:
            loguru_logger.info(f"通知失敗: {_WEBHOOK}")
        finally:
            loguru_logger.remove(sink_id)

        content = path.read_text(encoding="utf-8")
        assert "abcDEF-ghiJKL" not in content
        assert "REDACTED" in content

class TestExceptionsAreMaskedInSinkOutput:
    """例外本文・トレースバックまでマスクされること（外部レビューR17）。

    `_mask_record()` が `record["message"]` だけを書き換えると、
    `logger.exception()` の例外は別に整形されるため、Webhook URL を含む
    例外文字列がそのまま出力へ残る。**マスク関数の単体テストではなく、
    実際のシンク出力を読んで確かめる。**
    """

    def _capture(self, tmp_path, emit):
        from loguru import logger as loguru_logger
        from src.core import logger as log_mod

        path = tmp_path / "exc.log"
        sink_id = loguru_logger.add(
            str(path), format="{message}", level="INFO",
            filter=log_mod._mask_record, backtrace=True, diagnose=False)
        try:
            emit(loguru_logger)
        finally:
            loguru_logger.remove(sink_id)
        return path.read_text(encoding="utf-8")

    def test_logger_exception_masks_the_exception_text(self, tmp_path):
        def emit(lg):
            try:
                raise RuntimeError(f"送信に失敗しました: {_WEBHOOK}")
            except RuntimeError:
                lg.exception("通知エラー")

        content = self._capture(tmp_path, emit)
        assert "abcDEF-ghiJKL" not in content
        assert "REDACTED" in content

    def test_opt_exception_true_is_masked(self, tmp_path):
        def emit(lg):
            try:
                raise RuntimeError(f"送信に失敗しました: {_WEBHOOK}")
            except RuntimeError:
                lg.opt(exception=True).error("通知エラー")

        content = self._capture(tmp_path, emit)
        assert "abcDEF-ghiJKL" not in content

    def test_chained_exceptions_are_masked_on_both_sides(self, tmp_path):
        def emit(lg):
            try:
                try:
                    raise ValueError(f"元の失敗: {_WEBHOOK}")
                except ValueError as e:
                    raise RuntimeError("通知の再試行にも失敗") from e
            except RuntimeError:
                lg.exception("通知エラー")

        content = self._capture(tmp_path, emit)
        assert "abcDEF-ghiJKL" not in content

    def test_implicit_context_is_masked(self, tmp_path):
        """`from` を付けない連鎖（__context__）も伏せる"""
        def emit(lg):
            try:
                try:
                    raise ValueError(f"元の失敗: {_WEBHOOK}")
                except ValueError:
                    raise RuntimeError("後続の失敗")
            except RuntimeError:
                lg.exception("通知エラー")

        content = self._capture(tmp_path, emit)
        assert "abcDEF-ghiJKL" not in content

    def test_a_token_in_an_exception_is_masked(self, tmp_path):
        def emit(lg):
            try:
                raise RuntimeError("認証に失敗: X-API-Token: t0k3nvalue1234")
            except RuntimeError:
                lg.exception("認証エラー")

        content = self._capture(tmp_path, emit)
        assert "t0k3nvalue1234" not in content

    def test_the_original_exception_object_is_not_mutated(self, tmp_path):
        """マスクは複製に対して行う。呼び出し側の例外を壊さない"""
        from src.core import logger as log_mod

        original = RuntimeError(f"送信に失敗しました: {_WEBHOOK}")
        masked = log_mod._masked_exception(original)
        assert _WEBHOOK in str(original)      # 元は無傷
        assert _WEBHOOK not in str(masked)
        assert isinstance(masked, RuntimeError)

    def test_a_custom_exception_that_cannot_be_rebuilt_still_masks(self):
        """引数が独自の例外でも、値を残すより型を失うほうを選ぶ"""
        from src.core import logger as log_mod

        class Weird(Exception):
            def __init__(self, a, b, c):
                super().__init__(f"{a} {b} {c}")
                self.a, self.b, self.c = a, b, c

        exc = Weird(1, 2, _WEBHOOK)
        masked = log_mod._masked_exception(exc)
        assert _WEBHOOK not in str(masked)


class TestSetupTokenNeverReachesFileSinks:
    """初期設定トークンがファイルシンクへ出ないこと（外部レビューR15）。

    マスク関数の単体テストでは足りない。和文の説明に続く裸のトークンは
    `token=` にも `Authorization:` にも一致しないため、マスクは効かない。
    **出力先ごとに秘密値の有無を確認する。**
    """

    def test_the_token_value_is_not_written_to_the_log_file(
            self, tmp_path, monkeypatch, capsys):
        from loguru import logger as loguru_logger
        from src.core import logger as log_mod
        from src.dashboard import app as dash

        path = tmp_path / "app.log"
        sink_id = loguru_logger.add(
            str(path), format="{message}", level="INFO",
            filter=log_mod._mask_record)
        try:
            monkeypatch.setattr(dash.auth_store, "is_setup_allowed",
                                lambda: True)
            dash._init_setup_token()          # 起動処理から切り出した関数
            token = dash._setup_token
        finally:
            loguru_logger.remove(sink_id)

        assert token
        content = path.read_text(encoding="utf-8")
        assert token not in content
        assert "発行しました" in content      # 事実だけは残る

    def test_the_token_value_goes_to_the_console(self, monkeypatch, capsys):
        from src.dashboard import app as dash

        monkeypatch.setattr(dash.auth_store, "is_setup_allowed", lambda: True)
        dash._init_setup_token()
        captured = capsys.readouterr()
        assert dash._setup_token in captured.err

    def test_no_token_is_issued_once_setup_is_done(self, monkeypatch, capsys):
        from src.dashboard import app as dash

        monkeypatch.setattr(dash.auth_store, "is_setup_allowed", lambda: False)
        dash._init_setup_token()
        assert dash._setup_token is None
        assert "初期設定用トークン" not in capsys.readouterr().err
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_secret_masking.py -v`
Expected: FAIL — `ImportError: cannot import name 'mask_secrets' from 'src.core.logger'`

- [ ] **Step 3: 実装を追加**

`src/core/logger.py` に追加する。import に `import re` を足す。

```python
_REDACTED = "[REDACTED]"

# 秘密情報のパターン。今回のレビューで、保存済みログからDiscord Webhookの
# IDとトークンを含むURLが実際に見つかった（レビューF14）。例外の文字列表現を
# そのままログへ流す経路が残っていると同じことが再び起きるため、出力の
# 手前で機械的に伏せる。
_SECRET_PATTERNS = (
    # Discord Webhook（ホスト付き・ホスト無しの両方）
    re.compile(r"https?://(?:ptb\.|canary\.)?discord(?:app)?\.com/api/webhooks/[\w-]+/[\w-]+"),
    re.compile(r"/api/webhooks/[\w-]+/[\w-]+"),
    # クエリ・フォームの秘密値
    re.compile(r"(?i)\b(token|api_key|apikey|password|passwd|secret)=[^\s&\"']+"),
    # ヘッダー風の表記。**行末まで**伏せる。空白で止めると
    # 「Authorization: Bearer abc123」が「[REDACTED] abc123」になり、
    # トークン本体が残ってしまう。
    re.compile(r"(?i)\b(x-api-token|x-setup-token|authorization)\s*:\s*[^\r\n]+"),
)


def mask_secrets(text) -> str:
    """ログへ出す前に秘密情報を伏せる。

    完全な防御ではない（未知の形式は素通りする）が、既に実害が出た経路
    （Webhook URL・トークン・パスワード）を機械的に塞ぐ。
    メッセージの前後は残すので、何が起きたかは読める。
    """
    if not text:
        return ""
    out = str(text)
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(_REDACTED, out)
    return out


def _mask_record(record) -> bool:
    """loguru のフィルタ。メッセージを書き換えてから通す。

    フィルタで record["message"] を差し替えると、そのシンクの出力に反映される。
    常に True を返す（落とすのではなく伏せるのが目的）。

    **message だけでは足りない。** `logger.exception()` や
    `logger.opt(exception=True)` の例外は `record["exception"]` に入り、
    シンクへ書き出す段階で message とは別に整形される。message だけを
    書き換えても、例外文字列に含まれる Webhook URL やトークンはそのまま
    出力へ残る（外部レビューR17）。`diagnose=False` は変数値の表示を
    抑えるだけで、例外文字列そのものを除去する設定ではない。
    """
    record["message"] = mask_secrets(record["message"])

    exc = record.get("exception")
    if exc is not None and exc.value is not None:
        # 例外の引数をマスクした複製へ差し替える。型と traceback は保つ
        record["exception"] = exc._replace(value=_masked_exception(exc.value))
    return True


def _masked_exception(exc: BaseException) -> BaseException:
    """例外の文字列表現から秘密情報を伏せた複製を返す。

    連鎖例外（`raise ... from ...` / `__context__`）もたどる。片方だけを
    伏せると、`During handling of the above exception` の側に原文が残る。

    元の例外を書き換えない（呼び出し側が握っている同じオブジェクトなので、
    破壊するとアプリの挙動を変えてしまう）。復元に失敗したら、
    型を保てなくても**伏せるほうを優先**して RuntimeError へ落とす。
    """
    masked_args = tuple(
        mask_secrets(a) if isinstance(a, str) else a for a in exc.args)
    try:
        clone = exc.__class__(*masked_args)
    except Exception:
        # 引数の数や型が独自の例外は再構築できない。値を残すよりは
        # 型を失うほうがましなので、平文の RuntimeError にはしない
        clone = RuntimeError(f"{exc.__class__.__name__}: "
                             f"{mask_secrets(str(exc))}")
    clone.__traceback__ = exc.__traceback__
    if exc.__cause__ is not None:
        clone.__cause__ = _masked_exception(exc.__cause__)
    elif exc.__context__ is not None:
        clone.__context__ = _masked_exception(exc.__context__)
    return clone
```

> **実装者への注記。** loguru の `record["exception"]` は
> `(type, value, traceback)` の namedtuple（`RecordException`）である。
> `_replace(value=...)` で差し替えられる。差し替えが効くかどうかは
> **実際のシンク出力を読んで確かめる**こと（下記テスト）。マスク関数の
> 単体テストは、この経路が繋がっている証拠にならない。

- [ ] **Step 4: 既存のシンクへ適用する**

`src/core/logger.py` の `logger.add(...)` を呼んでいる全ての箇所に `filter=_mask_record` を足す。既存の `_no_leak`（`diagnose=False`）はそのまま残す。

```python
    # 既存の add(...) 呼び出しそれぞれに filter=_mask_record を足す。
    # 例:
    #   logger.add(log_path, ..., **_no_leak)
    #   → logger.add(log_path, ..., filter=_mask_record, **_no_leak)
```

既に `filter=` を渡している `add` があれば、次のように合成する。

```python
def _and_mask(existing_filter):
    """既存のフィルタとマスキングを合成する。"""
    def combined(record):
        if existing_filter is not None and not existing_filter(record):
            return False
        return _mask_record(record)
    return combined
```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `pytest tests/test_secret_masking.py -v`
Expected: PASS（14件）

- [ ] **Step 6: 既存のログテストが通ることを確認**

Run: `pytest tests/test_logger.py -v`
Expected: PASS

- [ ] **Step 7: コミット**

```bash
git add src/core/logger.py tests/test_secret_masking.py
git commit -m "$(cat <<'EOF'
feat(logging): 秘密情報のマスキングを追加

今回のレビューで、保存済みログからDiscord WebhookのIDとトークンを含む
URLが実際に1件見つかった。diagnose=Falseは変数展開を防ぐが、例外の
文字列表現を明示的に出す経路は防げない。
出力の手前でWebhook URL・クエリトークン・パスワード・認証ヘッダーを
機械的に伏せる。完全な防御ではないが、既に実害が出た形式を塞ぐ。
メッセージの前後は残すので何が起きたかは読める。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 通知失敗の出力を構造化する

**Files:**
- Modify: `src/core/alerts.py:97-120`（`_send_one`）
- Test: `tests/test_secret_masking.py`

**Interfaces:**
- Consumes: Task 4 の `mask_secrets`
- Produces: `_send_one` が例外をそのまま文字列化せず、**プロバイダ名・例外クラス名・HTTPステータス・再試行回数**だけを出す

**背景（F14）:** `alerts.py:119` の `logger.error(f"通知送信失敗（{provider.name}）: {last_error}")` は、例外の文字列表現をそのまま流す。`requests` の例外にはリクエストURLが含まれることがあり、Webhook URL が再びログへ残る。

**マスキング（Task 4）は最後の防波堤であって、そもそも秘密を含みうる値を出さないのが先。** 両方を入れる。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_secret_masking.py` の末尾に追記する。

```python
class TestAlertFailureOutput:
    """通知失敗の出力に秘密が混ざらない（F14）"""

    def _provider(self, name="discord"):
        class _P:
            def __init__(self):
                self.name = name

            def send(self, message):
                raise RuntimeError(f"接続失敗: {_WEBHOOK}")
        return _P()

    def test_does_not_log_the_exception_text(self, caplog):
        from src.core import alerts

        alerts._send_one(self._provider(), "テスト通知", retries=0)
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "abcDEF-ghiJKL" not in joined

    def test_logs_the_provider_and_the_exception_type(self, caplog):
        from src.core import alerts

        alerts._send_one(self._provider(), "テスト通知", retries=0)
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "discord" in joined
        assert "RuntimeError" in joined

    def test_logs_the_attempt_count(self, caplog):
        from src.core import alerts

        alerts._send_one(self._provider(), "テスト通知", retries=2)
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "3" in joined   # 初回 + 再試行2回

    def test_returns_false_on_failure(self):
        from src.core import alerts

        assert alerts._send_one(self._provider(), "テスト通知", retries=0) is False
```

> **実装者への注記:** `caplog` で loguru の出力を拾うには、loguru を標準 logging へ流すハンドラーが要る。`tests/test_logger.py` に既存のやり方があればそれに合わせること。無ければ `caplog` の代わりに `logger.add(sink_list.append, format="{message}")` で受けてよい。テストの意図は「出力に例外文字列が混ざらないこと」であり、受け取り方は問わない。

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_secret_masking.py::TestAlertFailureOutput -v`
Expected: FAIL — 例外文字列がそのまま出ているため `abcDEF-ghiJKL` が含まれる

- [ ] **Step 3: 実装を修正**

`src/core/alerts.py` の `_send_one` の失敗ログを次に置き換える。

```python
    # 例外の文字列表現をそのまま出さない。requests 等の例外にはリクエストURLが
    # 含まれることがあり、Webhook URL が再びログへ残る（レビューF14）。
    # 何が起きたかの判断に要るのは、どのプロバイダで・どの種類の失敗が・
    # 何回試して駄目だったか、である。
    status = getattr(getattr(last_error, "response", None), "status_code", None)
    detail = f" HTTP={status}" if status is not None else ""
    logger.error(
        f"通知送信失敗（{provider.name}）: {type(last_error).__name__}{detail} "
        f"試行={retries + 1}回"
    )
```

`last_error` を使っている他の箇所があれば、同じ方針（クラス名・ステータス・回数）に揃える。**例外オブジェクトを f-string へ直接埋めない。**

- [ ] **Step 4: テストを実行して成功を確認**

Run: `pytest tests/test_secret_masking.py -v`
Expected: PASS（18件）

- [ ] **Step 5: 既存の通知テストが通ることを確認**

Run: `pytest tests/test_alerts.py tests/test_alert_levels.py tests/test_health_alerts.py -v`
Expected: PASS

- [ ] **Step 6: 例外を直接埋めている箇所が他に無いか確認**

Run: `grep -rn 'logger\.\(error\|critical\|warning\)(f".*{e}' src/ --include=*.py | head -20`
Expected: 出力を確認する。**秘密を含みうる経路（通知・API・認証）だけ**を同じ方針に直す。データ計算など秘密を含まない箇所は変えない（変更範囲を無闇に広げない）

- [ ] **Step 7: 全体回帰とコミット**

Run: `pytest tests/ -q`
Expected: 失敗が増えていないこと

```bash
git add src/core/alerts.py tests/test_secret_masking.py
git commit -m "$(cat <<'EOF'
fix(alerts): 通知失敗の例外文字列からWebhook URLが漏れる経路を塞いだ

例外の文字列表現をそのままログへ流しており、requests等の例外に含まれる
リクエストURL（=Webhook URL）が再びログへ残る経路だった。
プロバイダ名・例外クラス名・HTTPステータス・試行回数だけを出す。
マスキング（logger側）は最後の防波堤であって、そもそも秘密を含みうる値を
出さないのが先。両方を入れる。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

## 完了条件の確認

- [ ] **確認1: 認証ファイルが壊れても初期設定に戻らない**

Run: `pytest tests/test_setup_protection.py::TestLoadFailure -v`
Expected: PASS（4件）

- [ ] **確認2: LANから第三者が初期設定できない**

Run: `pytest tests/test_setup_protection.py::TestSetupEndpointProtection -v`
Expected: PASS（5件）

- [ ] **確認3: ログアウトで全ての認証手段が失効する**

Run: `pytest tests/test_logout_revocation.py -v`
Expected: PASS（10件）

- [ ] **確認4: 秘密情報がログへ出ない**

Run: `pytest tests/test_secret_masking.py -v`
Expected: PASS（18件）

- [ ] **確認5: 既存の利用者がロックアウトされない**

Run: `pytest tests/test_auth.py tests/test_dashboard_auth.py -v`
Expected: PASS

- [ ] **確認6: 既存経路に回帰が無い**

Run: `pytest tests/ -q`
Expected: 着手前と同じ結果（新規テスト41件ぶんだけ増える）

- [ ] **確認7: 既存ログに秘密が残っていないか一度だけ点検する**

Run: `grep -rlE "api/webhooks/[0-9]+/[A-Za-z0-9_-]{20,}" log/ data/*.log 2>/dev/null || echo "該当なし"`
Expected: 該当が出た場合は**ユーザーへ報告して指示を仰ぐ**。ログの削除・書き換えは勝手に行わない（証跡であるため）

---

## 次の計画

`docs/superpowers/plans/2026-09-12-review-fixes-integrity.md`（F10・F11・F15）で、リスク価格の鮮度、プロファイル適用の原子性、DB制約と来歴を扱う。本計画とは独立に進められる。
