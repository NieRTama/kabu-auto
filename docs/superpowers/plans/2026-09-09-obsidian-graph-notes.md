# Obsidianグラフビュー用ノート生成 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** kabu-autoの49モジュール・71節・既存ドキュメント11件を相互リンクしたMarkdownノートとしてObsidian vaultへ生成し、グラフビューで「どのファイルが何をしていて、誰に使われ、どの事故に関係するか」を辿れるようにする。

**Architecture:** 単一のスタンドアロンスクリプト `scripts/gen_graph_notes.py`。リポジトリを読み取り専用で解析し（`ast`によるimport解析＋設計書の見出し解析）、vaultへノートを書き出す。kabu-auto本体のコード（`src/`）には一切依存せず、`config.yaml`も読まない。生成物はgit管理外。

**Tech Stack:** Python 3.11.6 標準ライブラリのみ（`ast` / `re` / `pathlib` / `argparse` / `dataclasses`）。外部依存を足さない。テストは pytest。

**Spec:** `docs/superpowers/specs/2026-09-09-obsidian-graph-notes-design.md`

## Global Constraints

- **日本語で書く** — ノート本文・コメント・docstring・コミットメッセージは日本語（CLAUDE.md準拠）
- **文字コード** — ファイル読み書きは必ず `encoding="utf-8"`、書き出しは `newline="\n"`、BOMなし。スクリプト冒頭で `sys.stdout.reconfigure(encoding="utf-8")` を呼ぶ（Windows cp932環境での出力失敗を防ぐ）
- **kabu-auto本体に一切依存しない** — `import src.*` を書かない。`src/` はテキストとして読むだけ
- **本体の常駐プロセス・スケジューラを変更しない** — 触るのは `scripts/` と `tests/` のみ
- **既存ドキュメントを書き換えない** — `docs/*.md` と `README.md` は読み取り専用
- **生成物をgitに入れない** — vaultへ出力する。`git status` に生成物が現れてはならない
- **決定性** — 同じ入力からバイト単位で同一の出力。全リストはソート。`index.md` 以外に生成日時を書かない
- **テストは実vaultに書かない** — 必ず `tmp_path` を出力先に渡す
- **既存の1066件のテストを壊さない**
- **テスト実行コマンドは `python -m pytest`** — `pytest` 単体だとCWDが `sys.path` に入らず `import scripts.*` が失敗する
- **コミット時の署名** — `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## File Structure

| ファイル | 責務 |
|----------|------|
| `scripts/__init__.py` | 新規・空。`scripts` をパッケージにしてテストから import 可能にする |
| `scripts/gen_graph_notes.py` | 新規。解析・レンダリング・書き出しの全て |
| `tests/test_gen_graph_notes.py` | 新規。上記のテスト |

`gen_graph_notes.py` は最終的に450行程度になる。以下の順で構成する（タスク順と一致）。

1. 定数・データクラス
2. モジュール探索とimport解析（Task 1）
3. 役割の抽出（Task 2）
4. 節の抽出（Task 3）
5. ファイル名の導出と衝突検査（Task 4）
6. レンダリング（Task 5・6・7）
7. CLI（Task 8）

---

### Task 1: モジュール探索とimport解析

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/gen_graph_notes.py`
- Test: `tests/test_gen_graph_notes.py`

**Interfaces:**
- Consumes: なし（最初のタスク）
- Produces:
  - `Module` dataclass: `path: str`（`"src/risk/manager.py"` 形式・スラッシュ区切り）, `dotted: str`（`"risk.manager"`）, `layer: str`, `docstring: str | None`, `deps: set[str]`（値は `dotted` 形式）, `symbols: set[str]`（トップレベルのクラス／関数名。曖昧なものは除去済み）
  - `discover_modules(repo_root: Path) -> list[Module]` — `path` 昇順でソート済み
  - `resolve_import(module: str | None, names: list[str], repo_root: Path) -> set[str]` — 戻り値は `"src.risk.manager"` 形式のドット表記

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_graph_notes.py` を新規作成:

```python
"""scripts/gen_graph_notes.py のテスト。

実vaultには一切書かない（出力先は必ず tmp_path を渡す）。
"""
import textwrap
from pathlib import Path

import pytest

from scripts.gen_graph_notes import (
    Module,
    discover_modules,
    resolve_import,
)


def _write(path: Path, content: str) -> None:
    """テスト用リポジトリにファイルを作る（親ディレクトリも作る）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    """最小のkabu-auto風リポジトリを作る。"""
    _write(tmp_path / "src" / "__init__.py", "")
    _write(tmp_path / "src" / "core" / "__init__.py", "")
    _write(tmp_path / "src" / "core" / "config.py", '"""設定管理のモジュール。"""\n')
    _write(tmp_path / "src" / "core" / "clock.py", '"""時刻の一元化。"""\n')
    _write(tmp_path / "src" / "risk" / "__init__.py", "")
    _write(
        tmp_path / "src" / "risk" / "manager.py",
        '''
        """リスク管理。

        2行目以降も本文。
        """
        from src.core import clock
        from src.core.config import get_section


        class RiskManager:
            pass
        ''',
    )
    _write(
        tmp_path / "main.py",
        '''
        """エントリポイント。"""
        from src.risk.manager import RiskManager


        def late():
            from src.core import clock  # 関数内import
            return clock
        ''',
    )
    return tmp_path


class TestDiscoverModules:
    def test_finds_all_modules_excluding_init(self, repo):
        mods = discover_modules(repo)
        paths = [m.path for m in mods]
        assert paths == [
            "main.py",
            "src/core/clock.py",
            "src/core/config.py",
            "src/risk/manager.py",
        ]

    def test_dotted_and_layer_are_derived(self, repo):
        by_path = {m.path: m for m in discover_modules(repo)}
        assert by_path["src/risk/manager.py"].dotted == "risk.manager"
        assert by_path["src/risk/manager.py"].layer == "risk"
        assert by_path["main.py"].dotted == "main"
        assert by_path["main.py"].layer == "root"

    def test_docstring_is_captured(self, repo):
        by_path = {m.path: m for m in discover_modules(repo)}
        assert by_path["src/core/config.py"].docstring == "設定管理のモジュール。"

    def test_function_level_import_is_detected(self, repo):
        """関数の中に書かれたimportも辺として拾う（正規表現では取りこぼす箇所）。"""
        by_path = {m.path: m for m in discover_modules(repo)}
        assert "core.clock" in by_path["main.py"].deps

    def test_self_and_external_imports_are_dropped(self, repo):
        by_path = {m.path: m for m in discover_modules(repo)}
        assert by_path["src/risk/manager.py"].deps == {"core.clock", "core.config"}



class TestExtractSymbols:
    def test_collects_class_and_function_names(self, repo):
        by_path = {m.path: m for m in discover_modules(repo)}
        assert "RiskManager" in by_path["src/risk/manager.py"].symbols

    def test_drops_short_and_undistinctive_names(self, repo):
        """短い名前・アンダースコア無しの関数名は本文中の別語に誤マッチするので採らない。"""
        by_path = {m.path: m for m in discover_modules(repo)}
        assert "Cfg" not in by_path["src/core/config.py"].symbols
        assert "load" not in by_path["src/core/config.py"].symbols

    def test_drops_names_defined_in_two_modules(self, repo):
        """同じ名前が2モジュールにあると、どちらを指すか決められないので捨てる。"""
        by_path = {m.path: m for m in discover_modules(repo)}
        assert "SharedThing" not in by_path["src/core/clock.py"].symbols
        assert "SharedThing" not in by_path["src/core/config.py"].symbols

class TestResolveImport:
    def test_from_package_import_module_resolves_to_the_module(self, repo):
        """from src.core import clock は src/core/clock.py が実在するので src.core.clock。"""
        assert resolve_import("src.core", ["clock"], repo) == {"src.core.clock"}

    def test_from_module_import_symbol_resolves_to_the_module(self, repo):
        """from src.core.config import get_section は get_section が実ファイルでないので src.core.config。"""
        assert resolve_import("src.core.config", ["get_section"], repo) == {"src.core.config"}

    def test_non_src_import_is_ignored(self, repo):
        assert resolve_import("pandas", ["DataFrame"], repo) == set()

    def test_relative_import_is_ignored(self, repo):
        assert resolve_import(None, ["x"], repo) == set()
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: 最小の実装を書く**

`scripts/__init__.py` を空ファイルとして作成する。

`scripts/gen_graph_notes.py` を新規作成:

```python
"""kabu-autoの構造をObsidianのグラフビュー用Markdownノートとして生成する。

モジュールのimport関係・設計書の見出し・節本文のパス言及を解析し、相互リンクした
ノート群をObsidian vaultへ書き出す。生成物は派生データなのでgit管理外に置く。

設計: docs/superpowers/specs/2026-09-09-obsidian-graph-notes-design.md

kabu-auto本体（src/）には依存しない。src/ はテキストとして読むだけ。
"""
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    # Windowsのcp932環境で日本語出力が UnicodeEncodeError で落ちるのを防ぐ
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class Module:
    """1つのPythonモジュール（グラフのノード1個に対応）。"""

    path: str                      # "src/risk/manager.py"（スラッシュ区切り）
    dotted: str                    # "risk.manager"（src. を除いたドット表記）
    layer: str                     # "risk" / "core" / ... / main.py は "root"
    docstring: str | None = None
    deps: set[str] = field(default_factory=set)     # dotted形式の依存先
    symbols: set[str] = field(default_factory=set)  # トップレベルのクラス／関数名


def _to_dotted(path: str) -> str:
    """モジュールパスをノート名用のドット表記へ変換する。

    "src/risk/manager.py" -> "risk.manager"
    "main.py"             -> "main"
    """
    stem = path[:-3] if path.endswith(".py") else path
    if stem.startswith("src/"):
        stem = stem[len("src/"):]
    return stem.replace("/", ".")


def _to_layer(path: str) -> str:
    """レイヤ名（グラフの色分け単位）を返す。src/直下のディレクトリ名。"""
    if not path.startswith("src/"):
        return "root"
    return path[len("src/"):].split("/")[0]


def resolve_import(module: str | None, names: list[str], repo_root: Path) -> set[str]:
    """import文の解決先を "src.a.b" 形式のドット表記の集合で返す。

    from src.a import b は、src/a/b.py が実在すれば src.a.b、しなければ src.a。
    src配下でないもの・相対import（module is None）は空集合。
    """
    if not module or not (module == "src" or module.startswith("src.")):
        return set()
    resolved = set()
    for name in names:
        candidate = f"{module}.{name}"
        if (repo_root / (candidate.replace(".", "/") + ".py")).exists():
            resolved.add(candidate)
        else:
            resolved.add(module)
    return resolved


def _parse_deps(source: str, repo_root: Path) -> set[str]:
    """ソースからsrc配下への依存先を集める（dotted形式）。

    ast.walk を使う。関数の中に書かれたimport（main.py:204 等に実在）を
    正規表現では取りこぼすため。
    """
    tree = ast.parse(source)
    raw: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:      # 相対import
                continue
            raw |= resolve_import(node.module, [a.name for a in node.names], repo_root)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "src" or alias.name.startswith("src."):
                    raw.add(alias.name)
    return {_to_dotted(r.replace(".", "/") + ".py") for r in raw}


# 記号名の採用条件。短い名前・ありふれた名前は本文中の別の語に誤マッチするため弾く。
MIN_CLASS_NAME = 5
MIN_FUNC_NAME = 8


def _extract_symbols(tree: ast.Module) -> set[str]:
    """モジュールのトップレベルで定義された、識別力のある名前を集める。

    設計書の事故節はモジュールをパスではなく記号名で書いている
    （例: §1.40 は "src/risk/manager.py" ではなく "RiskManager" と書く。
    実測では src/risk/manager.py は設計書全体で1回しか出てこない）。
    この名前で節とモジュールを繋ぐため控えておく。
    """
    names = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and len(node.name) >= MIN_CLASS_NAME:
            names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if len(node.name) >= MIN_FUNC_NAME and "_" in node.name:
                names.add(node.name)
    return names


def discover_modules(repo_root: Path) -> list[Module]:
    """src配下の全モジュールとmain.pyを探し、依存関係を解析して返す（path昇順）。"""
    files = [p for p in sorted((repo_root / "src").rglob("*.py")) if p.name != "__init__.py"]
    main_py = repo_root / "main.py"
    if main_py.exists():
        files.append(main_py)

    modules = []
    for f in files:
        rel = f.relative_to(repo_root).as_posix()
        source = f.read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        mod = Module(
            path=rel,
            dotted=_to_dotted(rel),
            layer=_to_layer(rel),
            docstring=docstring.strip() if docstring else None,
            deps=_parse_deps(source, repo_root),
            symbols=_extract_symbols(tree),
        )
        mod.deps.discard(mod.dotted)   # 自己参照は落とす
        modules.append(mod)

    # 2つ以上のモジュールに出る名前は、どちらを指すか決められないので捨てる
    counts: dict[str, int] = {}
    for mod in modules:
        for name in mod.symbols:
            counts[name] = counts.get(name, 0) + 1
    for mod in modules:
        mod.symbols = {n for n in mod.symbols if counts[n] == 1}

    return sorted(modules, key=lambda m: m.path)
```

- [ ] **Step 4: テストを実行して通ることを確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -v`
Expected: 12 passed

- [ ] **Step 5: 実リポジトリに対して健全性を確認**

Run:
```bash
python -c "from pathlib import Path; from scripts.gen_graph_notes import discover_modules; m=discover_modules(Path('.')); print(len(m), sum(len(x.deps) for x in m))"
```
Expected: `49 151` に近い値（モジュール49件・辺151本）。モジュール数が49でなければ探索条件を見直す。

- [ ] **Step 6: コミット**

```bash
git add scripts/__init__.py scripts/gen_graph_notes.py tests/test_gen_graph_notes.py
git commit -m "feat(tools): グラフノート生成のモジュール探索とimport解析を追加

ast.walk でimportを拾う。関数内import（main.py:204 等に実在）を
正規表現では取りこぼすため。from src.a import b は src/a/b.py の
有無で src.a.b / src.a に解決を分ける。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: 役割の抽出（見出し→docstring→既定値）

**Files:**
- Modify: `scripts/gen_graph_notes.py`（末尾に追記）
- Test: `tests/test_gen_graph_notes.py`（末尾に追記）

**Interfaces:**
- Consumes: `Module`（Task 1）
- Produces:
  - `RoleHeading` dataclass: `number: str`, `role: str`, `heading_text: str`（`"## "`/`"### "` を除いた見出し全文）
  - `extract_role_headings(detail_doc: str) -> dict[str, RoleHeading]` — キーは `"src/risk/manager.py"` 形式のパス
  - `module_role(module: Module, headings: dict[str, RoleHeading]) -> str`

**背景（実装者向け）:** 設計書の `###` 見出しのうちモジュールパスを含むものは41件あるが、
**役割として使えるのは型A（`### 1.1 src/core/config.py — 設定管理`）の29件だけ**。
残り12件は事故節の小見出しにパスが出ているだけで「そのモジュールの解説節」ではない
（例: `### 1.18.3 FIFOロット台帳（src/execution/lots.py）`）。型Aだけを狙う正規表現にする。
仕様書 §4.3 を参照。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_graph_notes.py` の末尾に追記:

```python
from scripts.gen_graph_notes import (  # noqa: E402
    RoleHeading,
    extract_role_headings,
    module_role,
)

DETAIL_DOC = textwrap.dedent(
    """
    ## 1. モジュール詳細

    ### 1.1 src/core/config.py — 設定管理

    本文。

    ### 1.10 src/risk/manager.py — リスク管理

    本文。

    ## 1.18 注文モデル分離

    ### 1.18.3 FIFOロット台帳（src/execution/lots.py）

    型B。役割として採用してはいけない。

    ### 1.33.4 通知の判断は `src/core/broker_watch.py` へ分離

    型D。役割として採用してはいけない。
    """
).lstrip()


class TestExtractRoleHeadings:
    def test_extracts_type_a_headings_only(self):
        headings = extract_role_headings(DETAIL_DOC)
        assert set(headings) == {"src/core/config.py", "src/risk/manager.py"}

    def test_captures_number_and_role(self):
        h = extract_role_headings(DETAIL_DOC)["src/risk/manager.py"]
        assert h.number == "1.10"
        assert h.role == "リスク管理"

    def test_heading_text_is_kept_for_anchor_links(self):
        h = extract_role_headings(DETAIL_DOC)["src/core/config.py"]
        assert h.heading_text == "1.1 src/core/config.py — 設定管理"

    def test_type_b_heading_is_not_treated_as_a_role(self):
        """事故節の小見出し（末尾が全角括弧）を解説節と誤認しない。"""
        assert "src/execution/lots.py" not in extract_role_headings(DETAIL_DOC)

    def test_type_d_heading_is_not_treated_as_a_role(self):
        """バッククオート囲みの文中パスを解説節と誤認しない。"""
        assert "src/core/broker_watch.py" not in extract_role_headings(DETAIL_DOC)


class TestModuleRole:
    def test_heading_wins_over_docstring(self):
        mod = Module(path="src/risk/manager.py", dotted="risk.manager", layer="risk",
                     docstring="docstringの1行目。")
        headings = extract_role_headings(DETAIL_DOC)
        assert module_role(mod, headings) == "リスク管理"

    def test_falls_back_to_docstring_first_line(self):
        mod = Module(path="src/core/clock.py", dotted="core.clock", layer="core",
                     docstring="アプリ全体の現在時刻を一元化するモジュール。\n\n2行目。")
        assert module_role(mod, {}) == "アプリ全体の現在時刻を一元化するモジュール。"

    def test_falls_back_to_placeholder_when_no_docstring(self):
        """docstringも見出しも無いモジュール（実測では src/core/logger.py の1件）。"""
        mod = Module(path="src/core/logger.py", dotted="core.logger", layer="core",
                     docstring=None)
        assert module_role(mod, {}) == "（説明なし）"
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -k "RoleHeading or ModuleRole" -v`
Expected: FAIL — `ImportError: cannot import name 'RoleHeading'`

- [ ] **Step 3: 最小の実装を書く**

`scripts/gen_graph_notes.py` の末尾に追記:

```python
import re

# 型A の見出しだけを狙う。パスが行頭側にあり、ダッシュで役割が続くもの。
#   ### 1.1 src/core/config.py — 設定管理
# 型B（末尾が全角括弧）・型C（パス2個で区切りが "/"）・型D（バッククオート囲みの文中パス）
# は意図的に弾く。これらは「そのモジュールの解説節」ではないため（仕様書 §4.3）。
ROLE_HEADING_RE = re.compile(
    r"^###\s*([\d.]+)\s+(src/[\w/]+\.py|main\.py)\s*[—–-]\s*(.+)$",
    re.MULTILINE,
)

NO_ROLE = "（説明なし）"


@dataclass
class RoleHeading:
    """型Aの `###` 見出し1件（モジュールの解説節）。"""

    number: str           # "1.10"
    role: str             # "リスク管理"
    heading_text: str     # "1.10 src/risk/manager.py — リスク管理"（アンカーリンク用）


def extract_role_headings(detail_doc: str) -> dict[str, RoleHeading]:
    """詳細設計書から型Aの見出しを抽出する。キーはモジュールパス。"""
    found = {}
    for m in ROLE_HEADING_RE.finditer(detail_doc):
        number, path, role = m.group(1), m.group(2), m.group(3).strip()
        heading_text = m.group(0)[len("###"):].strip()
        found[path] = RoleHeading(number=number, role=role, heading_text=heading_text)
    return found


def module_role(module: Module, headings: dict[str, RoleHeading]) -> str:
    """モジュールの役割を3段フォールバックで決める。

    1. 型Aの見出し（実測29件）
    2. docstringの先頭行（実測20件）
    3. どちらも無ければ既定文言（実測1件: src/core/logger.py）
    """
    heading = headings.get(module.path)
    if heading:
        return heading.role
    if module.docstring:
        return module.docstring.strip().splitlines()[0].strip()
    return NO_ROLE
```

`import re` はファイル冒頭の import 群へ移動する（`import ast` の直後）。

- [ ] **Step 4: テストを実行して通ることを確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -v`
Expected: 20 passed

- [ ] **Step 5: 実データで件数を確認**

Run:
```bash
python -c "
from pathlib import Path
from scripts.gen_graph_notes import discover_modules, extract_role_headings, module_role, NO_ROLE
doc = Path('docs/詳細設計書.md').read_text(encoding='utf-8')
h = extract_role_headings(doc)
mods = discover_modules(Path('.'))
by_head = sum(1 for m in mods if m.path in h)
none = [m.path for m in mods if module_role(m, h) == NO_ROLE]
print(f'見出し由来={by_head} docstring由来={len(mods)-by_head-len(none)} 既定={len(none)} {none}')
"
```
Expected: `見出し由来=29 docstring由来=19 既定=1 ['src/core/logger.py']`

- [ ] **Step 6: コミット**

```bash
git add scripts/gen_graph_notes.py tests/test_gen_graph_notes.py
git commit -m "feat(tools): モジュールの役割を3段フォールバックで決める処理を追加

設計書の ### 見出しのうちパスを含むものは41件あるが、役割として使えるのは
型A（パスが先頭・ダッシュ区切り）の29件だけ。残り12件は事故節の小見出しに
パスが出ているだけで解説節ではないため、型Aだけを狙う正規表現にした。
見出しが無い20件はdocstring先頭行、どちらも無い1件は既定文言へ落とす。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: 節の抽出（見出し分割・パス言及・冒頭抜粋）

**Files:**
- Modify: `scripts/gen_graph_notes.py`
- Test: `tests/test_gen_graph_notes.py`

**Interfaces:**
- Consumes: なし（ドキュメント文字列を受け取る）
- Produces:
  - `Section` dataclass: `doc_key: str`（`"詳細設計書"`）, `doc_path: str`（`"docs/詳細設計書.md"`）, `number: str`, `title: str`（番号を除いた見出し）, `heading_text: str`（番号込みの見出し全文）, `modules: set[str]`（パス形式）, `excerpt: str`
  - `split_sections(doc_key: str, doc_path: str, text: str) -> list[Section]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_graph_notes.py` の末尾に追記:

```python
from scripts.gen_graph_notes import Section, split_sections  # noqa: E402

SECTION_DOC = textwrap.dedent(
    """
    # タイトル

    前置き。節に属さないので無視される。

    ## 1.40 含み損益がOHLCV終値ベースで乖離していた（2026-09-09）

    | 表 | は |
    |----|----|
    | 抜粋 | しない |

    ```python
    # コードブロックも抜粋しない
    ```

    > 引用も抜粋しない

    ユーザーから報告を受けて調査した。真因は src/data/market_data.py の
    終値が2〜3日遅れていたこと。src/risk/manager.py も同じ値を見ていた。

    ## 主な機能

    番号の無い見出し。
    """
).lstrip()


class TestSplitSections:
    def test_splits_on_h2_and_ignores_preamble(self):
        secs = split_sections("詳細設計書", "docs/詳細設計書.md", SECTION_DOC)
        assert [s.title for s in secs] == [
            "含み損益がOHLCV終値ベースで乖離していた（2026-09-09）",
            "主な機能",
        ]

    def test_parses_section_number(self):
        secs = split_sections("詳細設計書", "docs/詳細設計書.md", SECTION_DOC)
        assert secs[0].number == "1.40"
        assert secs[1].number == ""

    def test_heading_text_is_kept_for_anchor_links(self):
        secs = split_sections("詳細設計書", "docs/詳細設計書.md", SECTION_DOC)
        assert secs[0].heading_text == "1.40 含み損益がOHLCV終値ベースで乖離していた（2026-09-09）"

    def test_collects_module_mentions_from_body(self):
        secs = split_sections("詳細設計書", "docs/詳細設計書.md", SECTION_DOC)
        assert secs[0].modules == {"src/data/market_data.py", "src/risk/manager.py"}

    def test_excerpt_skips_tables_code_and_quotes(self):
        """表・コードブロック・引用を飛ばして最初の散文行から抜粋する。"""
        secs = split_sections("詳細設計書", "docs/詳細設計書.md", SECTION_DOC)
        assert secs[0].excerpt.startswith("ユーザーから報告を受けて調査した。")
        assert "|" not in secs[0].excerpt
        assert "```" not in secs[0].excerpt

    def test_excerpt_is_capped(self):
        long_doc = "## 1.1 長い節\n\n" + ("あ" * 500) + "\n"
        secs = split_sections("d", "d.md", long_doc)
        assert len(secs[0].excerpt) <= 200

    def test_section_without_prose_gets_empty_excerpt(self):
        doc = "## 1.1 表だけの節\n\n| a | b |\n|---|---|\n"
        secs = split_sections("d", "d.md", doc)
        assert secs[0].excerpt == ""

    def test_parses_number_followed_by_a_period(self):
        """`## 1. モジュール詳細` 形式（番号の後ろに句点）。実測で24件ある。"""
        secs = split_sections("詳細設計書", "docs/詳細設計書.md", DETAIL_DOC)
        assert secs[0].number == "1"
        assert secs[0].title == "モジュール詳細"

    def test_keeps_the_body_for_symbol_matching(self):
        """記号名によるモジュール照合（Task 8）が本文を必要とする。"""
        secs = split_sections("詳細設計書", "docs/詳細設計書.md", DETAIL_DOC)
        assert "本文。" in secs[0].body

    def test_records_which_modules_this_section_explains(self):
        """節の中にある型A見出し＝その節が解説しているモジュール。"""
        secs = split_sections("詳細設計書", "docs/詳細設計書.md", DETAIL_DOC)
        assert secs[0].role_heading_paths == {"src/core/config.py", "src/risk/manager.py"}
        assert secs[1].role_heading_paths == set()
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -k SplitSections -v`
Expected: FAIL — `ImportError: cannot import name 'Section'`

- [ ] **Step 3: 最小の実装を書く**

`scripts/gen_graph_notes.py` の末尾に追記:

````python
# 節本文からモジュール言及を拾う。型B・C・Dの見出しに出るパスもここで拾われる。
MODULE_PATH_RE = re.compile(r"src/[\w/]+\.py|(?<![\w/])main\.py")

# 見出し先頭の節番号。"1.40 タイトル" と "1. タイトル"（番号の後ろに句点）の
# 両方を受ける。後者は実測で24件あり、受けないと番号がタイトル側へ流れ込む。
SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")

EXCERPT_MAX_CHARS = 200
EXCERPT_MAX_LINES = 3


@dataclass
class Section:
    """ドキュメントの `##` 節1件（グラフのノード1個に対応）。"""

    doc_key: str          # "詳細設計書"（ファイル名から拡張子を除いたもの）
    doc_path: str         # "docs/詳細設計書.md"
    number: str           # "1.40"（番号が無ければ ""）
    title: str            # "含み損益が..."（番号を除いた見出し）
    heading_text: str     # "1.40 含み損益が..."（アンカーリンク用の見出し全文）
    modules: set[str] = field(default_factory=set)   # パス形式
    excerpt: str = ""
    # この節の中に型A見出し（そのモジュールの解説）を持つモジュールのパス。
    # 「関係する設計判断・事故」から自分の解説節を除くために使う。
    role_heading_paths: set[str] = field(default_factory=set)
    body: str = ""        # 節の本文。記号名によるモジュール照合に使う


def _make_excerpt(body: str) -> str:
    """節本文の冒頭から散文を抜粋する。

    表・コードブロック・引用は読んでも意味が通らないので飛ばし、最初の散文行から
    最大3行・200文字を取る。散文が1行も無ければ空文字を返す。
    """
    lines, in_code = [], False
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line:
            if lines:
                break          # 散文が始まった後の空行で打ち切る
            continue
        if line.startswith(("|", ">", "#", "-", "*")):
            if lines:
                break
            continue
        lines.append(line)
        if len(lines) >= EXCERPT_MAX_LINES:
            break
    return " ".join(lines)[:EXCERPT_MAX_CHARS]


def split_sections(doc_key: str, doc_path: str, text: str) -> list[Section]:
    """ドキュメントを `##` 見出しで分割して節のリストを返す（出現順）。"""
    parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    sections = []
    # parts[0] は前置き（節に属さない）。以降 [見出し, 本文] の繰り返し。
    for i in range(1, len(parts), 2):
        heading_text = parts[i].strip()
        body = parts[i + 1]
        m = SECTION_NUMBER_RE.match(heading_text)
        number, title = (m.group(1), m.group(2).strip()) if m else ("", heading_text)
        sections.append(
            Section(
                doc_key=doc_key,
                doc_path=doc_path,
                number=number,
                title=title,
                heading_text=heading_text,
                modules=set(MODULE_PATH_RE.findall(body)),
                excerpt=_make_excerpt(body),
                role_heading_paths={m.group(2) for m in ROLE_HEADING_RE.finditer(body)},
                body=body,
            )
        )
    return sections
````

- [ ] **Step 4: テストを実行して通ることを確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -v`
Expected: 30 passed

- [ ] **Step 5: 実データで件数を確認**

Run:
```bash
python -c "
from pathlib import Path
from scripts.gen_graph_notes import split_sections
tot = 0
for key, p in [('詳細設計書','docs/詳細設計書.md'),('概要設計書','docs/概要設計書.md'),('README','README.md'),('運用Runbook','docs/運用Runbook.md')]:
    s = split_sections(key, p, Path(p).read_text(encoding='utf-8'))
    withmod = sum(1 for x in s if x.modules)
    print(f'{key}: 節{len(s)}件 うちモジュール言及あり{withmod}件')
    tot += len(s)
print('合計', tot)
"
```
Expected: 合計 71。`詳細設計書` が32件。

- [ ] **Step 6: コミット**

```bash
git add scripts/gen_graph_notes.py tests/test_gen_graph_notes.py
git commit -m "feat(tools): ドキュメントの節分割とモジュール言及の抽出を追加

## 見出しで分割し、本文中の src/xxx.py 表記を節→モジュールの辺として拾う。
冒頭抜粋は表・コードブロック・引用を飛ばして散文だけを取る（そのまま読んで
意味が通る要約にするため）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: ファイル名の導出と衝突検査

**Files:**
- Modify: `scripts/gen_graph_notes.py`
- Test: `tests/test_gen_graph_notes.py`

**Interfaces:**
- Consumes: `Module`（Task 1）, `Section`（Task 3）
- Produces:
  - `sanitize_filename(name: str) -> str`
  - `section_note_name(section: Section) -> str` — 拡張子なしのノート名
  - `find_collisions(note_names: list[str], vault_root: Path, graph_dir: Path) -> list[str]`

**注意:** モジュールノート名は `Module.dotted` をそのまま使う（別関数を作らない）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_graph_notes.py` の末尾に追記:

```python
from scripts.gen_graph_notes import (  # noqa: E402
    TITLE_MAX_CHARS,
    find_collisions,
    sanitize_filename,
    section_note_name,
)


def _section(number="1.40", title="含み損益がOHLCV終値ベースで乖離", doc_key="詳細設計書"):
    return Section(
        doc_key=doc_key,
        doc_path=f"docs/{doc_key}.md",
        number=number,
        title=title,
        heading_text=f"{number} {title}".strip(),
    )


class TestSanitizeFilename:
    def test_replaces_windows_forbidden_characters(self):
        assert sanitize_filename('a/b\\c:d*e?f"g<h>i|j') == "a-b-c-d-e-f-g-h-i-j"

    def test_keeps_japanese_and_parentheses(self):
        assert sanitize_filename("含み損益（2026-09-09）") == "含み損益（2026-09-09）"

    def test_strips_leading_and_trailing_whitespace_and_dots(self):
        """Windowsは末尾のドット・空白を含むファイル名を作れない。"""
        assert sanitize_filename("  名前.  ") == "名前"


class TestSectionNoteName:
    def test_includes_doc_key_number_and_title(self):
        assert section_note_name(_section()) == "詳細設計書-1.40-含み損益がOHLCV終値ベースで乖離"

    def test_omits_number_when_absent(self):
        assert section_note_name(_section(number="", title="主な機能", doc_key="README")) == "README-主な機能"

    def test_truncates_long_titles(self):
        name = section_note_name(_section(title="あ" * 100))
        assert name == "詳細設計書-1.40-" + "あ" * TITLE_MAX_CHARS


class TestFindCollisions:
    def test_detects_duplicates_among_generated_notes(self, tmp_path):
        vault = tmp_path / "vault"
        graph = vault / "Claude" / "graph"
        graph.mkdir(parents=True)
        assert find_collisions(["a", "b", "a"], vault, graph) == ["a"]

    def test_detects_clash_with_existing_vault_note(self, tmp_path):
        vault = tmp_path / "vault"
        graph = vault / "Claude" / "graph"
        graph.mkdir(parents=True)
        (vault / "Claude").joinpath("README.md").write_text("既存", encoding="utf-8")
        assert find_collisions(["README", "core.config"], vault, graph) == ["README"]

    def test_ignores_notes_inside_the_graph_dir(self, tmp_path):
        """前回の生成物は上書き対象なので衝突扱いしない。"""
        vault = tmp_path / "vault"
        graph = vault / "Claude" / "graph" / "modules"
        graph.mkdir(parents=True)
        graph.joinpath("core.config.md").write_text("前回の生成物", encoding="utf-8")
        assert find_collisions(["core.config"], vault, vault / "Claude" / "graph") == []

    def test_returns_empty_when_clean(self, tmp_path):
        vault = tmp_path / "vault"
        graph = vault / "Claude" / "graph"
        graph.mkdir(parents=True)
        assert find_collisions(["a", "b"], vault, graph) == []
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -k "Sanitize or SectionNoteName or FindCollisions" -v`
Expected: FAIL — `ImportError: cannot import name 'TITLE_MAX_CHARS'`

- [ ] **Step 3: 最小の実装を書く**

`scripts/gen_graph_notes.py` の末尾に追記:

```python
# Windowsのファイル名に使えない文字
FORBIDDEN_CHARS = '\\/:*?"<>|'
TITLE_MAX_CHARS = 24


def sanitize_filename(name: str) -> str:
    """ファイル名に使えない文字を除いた安全な名前を返す。"""
    for ch in FORBIDDEN_CHARS:
        name = name.replace(ch, "-")
    # Windowsは末尾のドット・空白を含む名前を作れない
    return name.strip().strip(".").strip()


def section_note_name(section: Section) -> str:
    """節スタブのノート名（拡張子なし）を返す。

    "詳細設計書-1.40-含み損益がOHLCV終値ベースで乖離"
    """
    title = sanitize_filename(section.title)[:TITLE_MAX_CHARS]
    parts = [sanitize_filename(section.doc_key)]
    if section.number:
        parts.append(section.number)
    parts.append(title)
    return "-".join(p for p in parts if p)


def find_collisions(note_names: list[str], vault_root: Path, graph_dir: Path) -> list[str]:
    """ノート名の衝突を検出して、衝突した名前をソートして返す。

    Obsidianのwikilinkはノート名で解決するため、vault内で名前が重複すると
    リンクが意図しない先に繋がる。graph_dir配下の既存ファイルは今回の上書き対象
    なので衝突扱いしない。
    """
    collisions = set()

    seen = set()
    for name in note_names:
        if name in seen:
            collisions.add(name)
        seen.add(name)

    graph_dir = graph_dir.resolve()
    generated = set(note_names)
    if vault_root.exists():
        for md in vault_root.rglob("*.md"):
            if graph_dir in md.resolve().parents or md.resolve() == graph_dir:
                continue
            if md.stem in generated:
                collisions.add(md.stem)
    return sorted(collisions)
```

- [ ] **Step 4: テストを実行して通ることを確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -v`
Expected: 40 passed

- [ ] **Step 5: 実vaultに対して衝突がないか確認（読み取りのみ）**

Run:
```bash
python -c "
from pathlib import Path
from scripts.gen_graph_notes import discover_modules, split_sections, section_note_name, find_collisions
vault = Path(r'C:\Users\garnet\iCloudDrive\iCloud~md~obsidian\Tama vault')
graph = vault / 'Claude' / 'graph'
names = [m.dotted for m in discover_modules(Path('.'))]
for key, p in [('詳細設計書','docs/詳細設計書.md'),('概要設計書','docs/概要設計書.md'),('README','README.md'),('運用Runbook','docs/運用Runbook.md')]:
    names += [section_note_name(s) for s in split_sections(key, p, Path(p).read_text(encoding='utf-8'))]
print('生成予定ノート数:', len(names))
print('衝突:', find_collisions(names, vault, graph))
"
```
Expected: `生成予定ノート数: 120` / `衝突: []`
衝突が出た場合は、その名前を報告して命名規則の見直しを相談する（勝手に上書きしない）。

- [ ] **Step 6: コミット**

```bash
git add scripts/gen_graph_notes.py tests/test_gen_graph_notes.py
git commit -m "feat(tools): ノート名の導出とvault内の名前衝突検査を追加

Obsidianのwikilinkはノート名で解決するため、vault内で名前が重複すると
リンクが意図しない先に繋がる。生成前に検査し、1件でもあれば中断する方針。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: モジュールノートのレンダリング

**Files:**
- Modify: `scripts/gen_graph_notes.py`
- Test: `tests/test_gen_graph_notes.py`

**Interfaces:**
- Consumes: `Module`, `RoleHeading`, `Section`, `module_role`, `section_note_name`
- Produces:
  - `escape_wikilinks(text: str) -> str`
  - `render_module_note(module, role, dependents, headings, related_sections) -> str`
    - `dependents: list[str]`（dotted形式・ソート済み）
    - `related_sections: list[Section]`（ソート済み）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_graph_notes.py` の末尾に追記:

```python
from scripts.gen_graph_notes import escape_wikilinks, render_module_note  # noqa: E402


class TestEscapeWikilinks:
    def test_escapes_double_brackets_from_type_annotations(self):
        """Callable[[list[str]], ...] が [[list[str]] というリンクに化けるのを防ぐ。

        実際に詳細設計書.md で発生した事象。
        """
        assert "[[" not in escape_wikilinks("Callable[[list[str]], dict[str, float]]")

    def test_leaves_plain_text_untouched(self):
        assert escape_wikilinks("ふつうの説明文") == "ふつうの説明文"


class TestRenderModuleNote:
    def _render(self, **kw):
        mod = kw.pop("module", Module(
            path="src/risk/manager.py", dotted="risk.manager", layer="risk",
            docstring="リスク管理の説明。", deps={"data.database", "core.halt"},
        ))
        return render_module_note(
            module=mod,
            role=kw.pop("role", "リスク管理"),
            dependents=kw.pop("dependents", ["execution.order_manager", "services.trading"]),
            headings=kw.pop("headings", {"src/risk/manager.py": RoleHeading(
                number="1.10", role="リスク管理",
                heading_text="1.10 src/risk/manager.py — リスク管理")}),
            related_sections=kw.pop("related_sections", [_section()]),
        )

    def test_has_frontmatter_with_tags_and_module_path(self):
        note = self._render()
        assert note.startswith("---\n")
        assert "tags: [kabu-auto/module, layer/risk]" in note
        assert "module: src/risk/manager.py" in note
        assert "generated_by: gen_graph_notes.py" in note

    def test_shows_role_and_docstring(self):
        note = self._render()
        assert "**役割**: リスク管理" in note
        assert "> リスク管理の説明。" in note

    def test_lists_dependencies_sorted_as_wikilinks(self):
        note = self._render()
        assert "- [[core.halt]]\n- [[data.database]]" in note

    def test_lists_dependents(self):
        note = self._render()
        assert "- [[execution.order_manager]]" in note
        assert "- [[services.trading]]" in note

    def test_links_to_explanation_section_with_anchor(self):
        note = self._render()
        assert "[[詳細設計書#1.10 src/risk/manager.py — リスク管理]]" in note

    def test_links_to_related_sections(self):
        note = self._render()
        assert "[[詳細設計書-1.40-含み損益がOHLCV終値ベースで乖離]]" in note

    def test_omits_empty_sections_entirely(self):
        """依存元が無いモジュールに空の見出しを残さない。"""
        mod = Module(path="src/core/clock.py", dotted="core.clock", layer="core",
                     docstring="時刻。", deps=set())
        note = render_module_note(module=mod, role="時刻", dependents=[],
                                  headings={}, related_sections=[])
        assert "依存先" not in note
        assert "依存元" not in note
        assert "解説" not in note
        assert "関係する設計判断・事故" not in note

    def test_docstring_with_type_annotation_does_not_create_a_link(self):
        mod = Module(path="src/risk/manager.py", dotted="risk.manager", layer="risk",
                     docstring="price_fn: Callable[[list[str]], dict[str, float]] を受ける。",
                     deps=set())
        note = render_module_note(module=mod, role="リスク管理", dependents=[],
                                  headings={}, related_sections=[])
        assert "[[" not in note

    def test_contains_no_timestamp(self):
        """生成日時を書くとiCloudが毎回全ファイルを再同期するため入れない。"""
        note = self._render()
        assert "生成日時" not in note
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -k "EscapeWikilinks or RenderModuleNote" -v`
Expected: FAIL — `ImportError: cannot import name 'escape_wikilinks'`

- [ ] **Step 3: 最小の実装を書く**

`scripts/gen_graph_notes.py` の末尾に追記:

```python
def escape_wikilinks(text: str) -> str:
    """本文中の `[[` をエスケープする。

    Callable[[list[str]], dict[str, float]] のような型注釈をObsidianが
    `[[list[str]]` というwikilinkと誤認識するため（詳細設計書.md で実際に発生）。
    """
    return text.replace("[[", r"\[\[")


def _bullet_list(items: list[str]) -> str:
    """wikilinkの箇条書きを作る。"""
    return "\n".join(f"- [[{i}]]" for i in items)


def _quote_block(text: str, max_lines: int = 3) -> str:
    """docstringを引用ブロックにする（最大max_lines行）。"""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()][:max_lines]
    return "\n".join(f"> {escape_wikilinks(ln)}" for ln in lines)


def render_module_note(
    module: Module,
    role: str,
    dependents: list[str],
    headings: dict[str, RoleHeading],
    related_sections: list["Section"],
) -> str:
    """モジュールノート1件のMarkdownを組み立てる。

    生成日時は書かない（決定的な出力にして、iCloudの無駄な再同期を避けるため）。
    中身が空になるセクションは見出しごと省略する。
    """
    out = [
        "---",
        f"tags: [kabu-auto/module, layer/{module.layer}]",
        f"module: {module.path}",
        "generated_by: gen_graph_notes.py",
        "---",
        "",
        f"# {module.dotted}",
        "",
        f"**役割**: {escape_wikilinks(role)}",
    ]
    if module.docstring:
        out += ["", _quote_block(module.docstring)]

    if module.deps:
        out += ["", "## 依存先（このモジュールが使う）", "", _bullet_list(sorted(module.deps))]
    if dependents:
        out += ["", "## 依存元（このモジュールを使う）", "", _bullet_list(sorted(dependents))]

    heading = headings.get(module.path)
    if heading:
        out += ["", "## 解説", "", f"- [[詳細設計書#{heading.heading_text}]]"]

    if related_sections:
        names = sorted(section_note_name(s) for s in related_sections)
        out += ["", "## 関係する設計判断・事故", "", _bullet_list(names)]

    return "\n".join(out) + "\n"
```

- [ ] **Step 4: テストを実行して通ることを確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -v`
Expected: 51 passed

- [ ] **Step 5: コミット**

```bash
git add scripts/gen_graph_notes.py tests/test_gen_graph_notes.py
git commit -m "feat(tools): モジュールノートのレンダリングを追加

依存先・依存元・解説・関係する事故を持つノートを組み立てる。空になる
セクションは見出しごと省略する。型注釈 Callable[[list[str]], ...] が
Obsidianに [[list[str]] と誤認識される実害があるためエスケープする。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: 節スタブのレンダリング

**Files:**
- Modify: `scripts/gen_graph_notes.py`
- Test: `tests/test_gen_graph_notes.py`

**Interfaces:**
- Consumes: `Section`, `escape_wikilinks`
- Produces: `render_section_note(section: Section, module_dotted: dict[str, str]) -> str`
  - `module_dotted` はパス→dotted のマッピング（例 `{"src/risk/manager.py": "risk.manager"}`）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_graph_notes.py` の末尾に追記:

```python
from scripts.gen_graph_notes import render_section_note  # noqa: E402

DOTTED = {
    "src/risk/manager.py": "risk.manager",
    "src/data/market_data.py": "data.market_data",
}


class TestRenderSectionNote:
    def test_has_frontmatter_with_source(self):
        sec = _section()
        note = render_section_note(sec, DOTTED)
        assert "tags: [kabu-auto/section, kabu-auto/詳細設計書]" in note
        assert "source: docs/詳細設計書.md" in note
        assert "generated_by: gen_graph_notes.py" in note

    def test_title_uses_full_heading(self):
        note = render_section_note(_section(), DOTTED)
        assert "# 1.40 含み損益がOHLCV終値ベースで乖離" in note

    def test_links_back_to_source_with_anchor(self):
        note = render_section_note(_section(), DOTTED)
        assert "[[詳細設計書#1.40 含み損益がOHLCV終値ベースで乖離]]" in note

    def test_lists_related_modules_sorted(self):
        sec = _section()
        sec.modules = {"src/risk/manager.py", "src/data/market_data.py"}
        note = render_section_note(sec, DOTTED)
        assert "- [[data.market_data]]\n- [[risk.manager]]" in note

    def test_omits_module_section_when_empty(self):
        note = render_section_note(_section(), DOTTED)
        assert "関係するモジュール" not in note

    def test_includes_excerpt(self):
        sec = _section()
        sec.excerpt = "ユーザーから報告を受けて調査した。"
        note = render_section_note(sec, DOTTED)
        assert "ユーザーから報告を受けて調査した。" in note

    def test_unknown_module_path_is_skipped(self):
        """設計書に書かれているが実在しないモジュールへのリンクは張らない。"""
        sec = _section()
        sec.modules = {"src/does/not_exist.py"}
        note = render_section_note(sec, DOTTED)
        assert "関係するモジュール" not in note

    def test_escapes_wikilinks_in_excerpt(self):
        sec = _section()
        sec.excerpt = "型は Callable[[list[str]], float] である。"
        note = render_section_note(sec, DOTTED)
        assert "[[list" not in note
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -k RenderSectionNote -v`
Expected: FAIL — `ImportError: cannot import name 'render_section_note'`

- [ ] **Step 3: 最小の実装を書く**

`scripts/gen_graph_notes.py` の末尾に追記:

```python
def render_section_note(section: Section, module_dotted: dict[str, str]) -> str:
    """節スタブ1件のMarkdownを組み立てる。

    本文は原本に残し、ここには要約とリンクだけを置く（重複を作らないため）。
    設計書に書かれていても実在しないモジュールへのリンクは張らない。
    """
    out = [
        "---",
        f"tags: [kabu-auto/section, kabu-auto/{section.doc_key}]",
        f"source: {section.doc_path}",
        "generated_by: gen_graph_notes.py",
        "---",
        "",
        f"# {section.heading_text}",
    ]
    if section.excerpt:
        out += ["", escape_wikilinks(section.excerpt)]

    out += ["", "## 本文", "", f"- [[{section.doc_key}#{section.heading_text}]]"]

    linked = sorted(
        module_dotted[p] for p in section.modules if p in module_dotted
    )
    if linked:
        out += ["", "## 関係するモジュール", "", _bullet_list(linked)]

    return "\n".join(out) + "\n"
```

- [ ] **Step 4: テストを実行して通ることを確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -v`
Expected: 59 passed

- [ ] **Step 5: コミット**

```bash
git add scripts/gen_graph_notes.py tests/test_gen_graph_notes.py
git commit -m "feat(tools): 節スタブのレンダリングを追加

本文は原本に残し、スタブには要約とアンカーリンクだけを置く。節スタブは
必ず原本へのリンクを持つため、モジュール言及が無い節も孤立しない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: 書き出し・陳腐化削除・index.md

**Files:**
- Modify: `scripts/gen_graph_notes.py`
- Test: `tests/test_gen_graph_notes.py`

**Interfaces:**
- Consumes: 前タスク全て
- Produces:
  - `GENERATED_MARKER: str` = `"generated_by: gen_graph_notes.py"`
  - `write_note(path: Path, content: str) -> None`
  - `cleanup_stale(graph_dir: Path, keep: set[Path]) -> list[Path]` — 削除したパスを返す
  - `render_index(module_count: int, section_count: int, generated_at: str) -> str`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_graph_notes.py` の末尾に追記:

```python
from scripts.gen_graph_notes import (  # noqa: E402
    GENERATED_MARKER,
    cleanup_stale,
    render_index,
    write_note,
)


class TestWriteNote:
    def test_writes_utf8_without_bom_and_lf_newlines(self, tmp_path):
        target = tmp_path / "sub" / "ノート.md"
        write_note(target, "# 見出し\n本文\n")
        raw = target.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")     # BOMなし
        assert b"\r\n" not in raw                       # CRLFなし
        assert raw.decode("utf-8").startswith("# 見出し")

    def test_creates_parent_directories(self, tmp_path):
        write_note(tmp_path / "a" / "b" / "c.md", "x\n")
        assert (tmp_path / "a" / "b" / "c.md").exists()


class TestCleanupStale:
    def test_removes_generated_files_not_in_keep(self, tmp_path):
        old = tmp_path / "modules" / "old.md"
        write_note(old, f"---\n{GENERATED_MARKER}\n---\n\n古い生成物\n")
        removed = cleanup_stale(tmp_path, keep=set())
        assert removed == [old]
        assert not old.exists()

    def test_keeps_handwritten_files(self, tmp_path):
        """generated_by を持たない手書きノートは消さない。"""
        mine = tmp_path / "私のメモ.md"
        write_note(mine, "# 手で書いたノート\n")
        removed = cleanup_stale(tmp_path, keep=set())
        assert removed == []
        assert mine.exists()

    def test_keeps_files_in_keep_set(self, tmp_path):
        current = tmp_path / "modules" / "core.config.md"
        write_note(current, f"---\n{GENERATED_MARKER}\n---\n\n今回の生成物\n")
        removed = cleanup_stale(tmp_path, keep={current})
        assert removed == []
        assert current.exists()

    def test_returns_empty_when_dir_missing(self, tmp_path):
        assert cleanup_stale(tmp_path / "なし", keep=set()) == []


class TestRenderIndex:
    def test_contains_counts_and_timestamp(self):
        idx = render_index(49, 71, "2026-09-09 15:00:00")
        assert "49" in idx
        assert "71" in idx
        assert "2026-09-09 15:00:00" in idx

    def test_contains_graph_color_group_json(self):
        """色分け設定はコピペ用JSONで案内する（.obsidian は書き換えない）。"""
        idx = render_index(49, 71, "2026-09-09 15:00:00")
        assert "layer/risk" in idx
        assert "colorGroups" in idx
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -k "WriteNote or CleanupStale or RenderIndex" -v`
Expected: FAIL — `ImportError: cannot import name 'GENERATED_MARKER'`

- [ ] **Step 3: 最小の実装を書く**

`scripts/gen_graph_notes.py` の末尾に追記:

````python
GENERATED_MARKER = "generated_by: gen_graph_notes.py"

# グラフビューの色分け設定（index.md に載せるコピペ用）。
# .obsidian/graph.json は書き換えない（ユーザーの既存設定を壊さないため）。
COLOR_GROUPS = [
    ("path:Claude/graph/modules", "モジュール"),
    ("path:Claude/graph/sections", "設計書の節"),
    ("tag:#layer/risk", "リスク管理"),
    ("tag:#layer/execution", "発注"),
    ("tag:#layer/core", "基盤"),
]


def write_note(path: Path, content: str) -> None:
    """ノートを書き出す（UTF-8・BOMなし・LF改行）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def cleanup_stale(graph_dir: Path, keep: set[Path]) -> list[Path]:
    """今回生成しなかった過去の生成物を削除する（削除したパスをソートして返す）。

    frontmatterに GENERATED_MARKER を持つファイルだけを消す。ユーザーが
    graph/ 配下に手で置いたノートを巻き込まないため。
    """
    if not graph_dir.exists():
        return []
    keep_resolved = {p.resolve() for p in keep}
    removed = []
    for md in sorted(graph_dir.rglob("*.md")):
        if md.resolve() in keep_resolved:
            continue
        try:
            head = md.read_text(encoding="utf-8")[:400]
        except (OSError, UnicodeDecodeError):
            continue
        if GENERATED_MARKER in head:
            md.unlink()
            removed.append(md)
    return removed


def render_index(module_count: int, section_count: int, generated_at: str) -> str:
    """graph/index.md を組み立てる。生成日時を書くのはこのファイルだけ。"""
    groups = ",\n    ".join(
        '{"query": "%s", "color": {"a": 1, "rgb": %d}}' % (q, 0x4C78A8 + i * 0x111111)
        for i, (q, _label) in enumerate(COLOR_GROUPS)
    )
    legend = "\n".join(f"| `{q}` | {label} |" for q, label in COLOR_GROUPS)
    return f"""---
tags: [kabu-auto/index]
{GENERATED_MARKER}
---

# kabu-auto 構造グラフ

`scripts/gen_graph_notes.py` が自動生成したノート群の目次。
**手で編集しても次回の生成で上書きされる。**

| 項目 | 値 |
|------|-----|
| モジュールノート | {module_count} 件 |
| 節スタブ | {section_count} 件 |
| 生成日時 | {generated_at} |

## 使い方

グラフビュー（左サイドバーの「グラフビューを開く」）で全体を眺め、
気になるノードをクリックするとそのノートが開く。

- モジュールノート … 役割・依存先・依存元・解説・関係する事故
- 節スタブ … 要約と原本へのリンク・関係するモジュール

## グラフの色分け設定

Obsidianの設定は自動で書き換えていない（既存の設定を壊さないため）。
色を付けたい場合は、グラフビュー右上の設定から「グループ」に以下を追加する。

| クエリ | 意味 |
|--------|------|
{legend}

`.obsidian/graph.json` を直接編集する場合は次を `colorGroups` に入れる。

```json
"colorGroups": [
    {groups}
]
```
"""
````

- [ ] **Step 4: テストを実行して通ることを確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -v`
Expected: 67 passed

- [ ] **Step 5: コミット**

```bash
git add scripts/gen_graph_notes.py tests/test_gen_graph_notes.py
git commit -m "feat(tools): ノートの書き出し・陳腐化削除・index生成を追加

削除は frontmatter に generated_by を持つファイルだけを対象にする
（graph/ 配下に手で置いたノートを巻き込まないため）。
.obsidian/graph.json は書き換えず、色分け設定はコピペ用JSONで案内する。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: 記号名によるモジュール照合

**Files:**
- Modify: `scripts/gen_graph_notes.py`
- Test: `tests/test_gen_graph_notes.py`

**Interfaces:**
- Consumes: `Module`（Task 1。`symbols` を使う）
- Produces: `match_modules_by_symbol(body: str, modules: list[Module]) -> set[str]` — 戻り値はモジュールのパス

**背景（実装者向け）:** 設計書の事故節はモジュールを**パスではなく記号名で書いている**。
実測では `src/risk/manager.py` は設計書全体で1回（モジュール詳細節）しか出てこず、
§1.39・§1.40 は `RiskManager` `unrealized_pnl` と書いている。パス走査だけだと
「この事故はどのファイル発か」という最も価値のある辺がほとんど張れない。

実測: パス走査のみ = 辺47本・繋がる節29/71件 → 記号名も見る = **辺125本・繋がる節48/71件**。

**単語境界の注意:** `` を使ってはいけない。Pythonの `re` はUnicode既定なので
日本語文字が `\w` 扱いになり、`RiskManagerの初期化` のように助詞が直結すると
マッチしない（日本語ドキュメントでは頻出）。ASCIIの英数字とアンダースコアだけを
境界とみなす先読み・後読みを使う。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_graph_notes.py` の末尾に追記:

```python
from scripts.gen_graph_notes import match_modules_by_symbol  # noqa: E402


def _mods_with_symbols():
    return [
        Module(path="src/risk/manager.py", dotted="risk.manager", layer="risk",
               symbols={"RiskManager", "unrealized_pnl"}),
        Module(path="src/data/market_data.py", dotted="data.market_data", layer="data",
               symbols={"latest_closes"}),
    ]


class TestMatchModulesBySymbol:
    def test_matches_symbol_followed_by_a_japanese_particle(self):
        """`RiskManagerの初期化` にマッチすること。

         を使うとPythonのUnicode既定で「の」が単語文字扱いになり、
        境界が成立せずマッチしない。日本語ドキュメントでは頻出の書き方。
        """
        body = "RiskManagerの初期化で例外が出る。"
        assert match_modules_by_symbol(body, _mods_with_symbols()) == {"src/risk/manager.py"}

    def test_matches_symbol_in_japanese_quotes(self):
        body = "「unrealized_pnl」が誤った値を返していた。"
        assert match_modules_by_symbol(body, _mods_with_symbols()) == {"src/risk/manager.py"}

    def test_does_not_match_a_longer_identifier(self):
        """部分一致で誤って繋がないこと。"""
        body = "MyRiskManagerXtra は無関係のクラスである。"
        assert match_modules_by_symbol(body, _mods_with_symbols()) == set()

    def test_matches_multiple_modules(self):
        body = "RiskManager が latest_closes を呼んでいる。"
        assert match_modules_by_symbol(body, _mods_with_symbols()) == {
            "src/risk/manager.py", "src/data/market_data.py"}

    def test_returns_empty_when_nothing_matches(self):
        assert match_modules_by_symbol("関係の無い文章。", _mods_with_symbols()) == set()
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -k MatchModulesBySymbol -v`
Expected: FAIL — `ImportError: cannot import name 'match_modules_by_symbol'`

- [ ] **Step 3: 最小の実装を書く**

`scripts/gen_graph_notes.py` の末尾に追記:

```python
def match_modules_by_symbol(body: str, modules: list[Module]) -> set[str]:
    """節本文に出てくる記号名から、関係するモジュールのパスを集める。

    境界に `` を使わないこと。Pythonの `re` はUnicode既定なので日本語文字が
    `\w` 扱いになり、`RiskManagerの初期化` のように助詞が直結するとマッチしない。
    ASCIIの英数字とアンダースコアだけを境界とみなす。
    """
    hits = set()
    for module in modules:
        for name in module.symbols:
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
            if re.search(pattern, body):
                hits.add(module.path)
                break
    return hits
```

- [ ] **Step 4: テストを実行して通ることを確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -v`
Expected: 72 passed

- [ ] **Step 5: コミット**

```bash
git add scripts/gen_graph_notes.py tests/test_gen_graph_notes.py
git commit -m "feat(tools): 記号名によるモジュール照合を追加

設計書の事故節はモジュールをパスではなく記号名で書いている（src/risk/manager.py
は設計書全体で1回しか出てこず、§1.40は RiskManager と書く）。パス走査だけでは
辺47本・繋がる節29/71件だったのが、記号名も見ると辺125本・48/71件になる。

単語境界に \b を使わない。PythonのreはUnicode既定で日本語文字が \w 扱いになり、
「RiskManagerの初期化」のように助詞が直結するとマッチしないため。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: CLI結線と初回実行・実測

**Files:**
- Modify: `scripts/gen_graph_notes.py`
- Test: `tests/test_gen_graph_notes.py`

**Interfaces:**
- Consumes: 全タスク
- Produces: `build_all(repo_root: Path) -> tuple[dict[str, str], int, int]`（相対パス→本文、モジュール数、節数）、`main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_gen_graph_notes.py` の末尾に追記:

```python
from scripts.gen_graph_notes import build_all, main  # noqa: E402


@pytest.fixture
def full_repo(repo):
    """Task 1 の repo にドキュメントを足した、生成を通しで動かせるリポジトリ。"""
    _write(repo / "docs" / "詳細設計書.md", DETAIL_DOC)
    _write(repo / "docs" / "概要設計書.md", "## 1. 概要\n\n概要の本文。\n")
    _write(repo / "README.md", "## 主な機能\n\n機能の説明。\n")
    _write(repo / "docs" / "運用Runbook.md", "## 1. 手順\n\n手順の本文。\n")
    return repo


class TestBuildAll:
    def test_produces_a_note_per_module_and_section(self, full_repo):
        notes, n_mod, n_sec = build_all(full_repo)
        assert n_mod == 4
        assert n_sec == 5      # 詳細2（1. と 1.18）+ 概要1 + README1 + Runbook1
        assert len(notes) == n_mod + n_sec

    def test_module_notes_go_under_modules_dir(self, full_repo):
        notes, _, _ = build_all(full_repo)
        assert "modules/risk.manager.md" in notes
        assert "sections/README-主な機能.md" in notes

    def test_dependents_are_computed_by_inverting_the_graph(self, full_repo):
        notes, _, _ = build_all(full_repo)
        assert "- [[main]]" in notes["modules/risk.manager.md"]

    def test_links_incident_sections_found_by_symbol_name(self, full_repo):
        """パスを書いていない事故節でも、記号名でモジュールに繋がること。

        実データではこれが主経路（設計書は src/risk/manager.py ではなく
        RiskManager と書く）。
        """
        _write(full_repo / "docs" / "運用Runbook.md",
               "## 9. 障害対応

RiskManager が誤った値を返す事象。
")
        notes, _, _ = build_all(full_repo)
        assert "運用Runbook-9-障害対応" in notes["modules/risk.manager.md"]

    def test_section_notes_also_list_symbol_matched_modules(self, full_repo):
        """節ノート側の「関係するモジュール」にも記号名マッチを反映すること。

        反映しないと、モジュール側からは繋がっているのに節側は空、という
        非対称な表示になる。
        """
        _write(full_repo / "docs" / "運用Runbook.md",
               "## 9. 障害対応

RiskManager が誤った値を返す事象。
")
        notes, _, _ = build_all(full_repo)
        assert "- [[risk.manager]]" in notes["sections/運用Runbook-9-障害対応.md"]

    def test_own_explanation_section_is_not_listed_as_an_incident(self, full_repo):
        """自分の解説がある節を「関係する設計判断・事故」に出さない。

        出すと `## 1. モジュール詳細` のような目次節が全モジュールから参照され、
        グラフ上で意味のない巨大ハブになる。解説は `## 解説` 側でリンク済み。
        """
        notes, _, _ = build_all(full_repo)
        note = notes["modules/risk.manager.md"]
        assert "詳細設計書-1-モジュール詳細" not in note


class TestMain:
    def test_dry_run_writes_nothing(self, full_repo, tmp_path, capsys):
        out = tmp_path / "vault" / "Claude" / "graph"
        out.parent.mkdir(parents=True)
        rc = main(["--repo", str(full_repo), "--out", str(out), "--dry-run"])
        assert rc == 0
        assert not out.exists()
        assert "モジュール" in capsys.readouterr().out

    def test_writes_notes_and_index(self, full_repo, tmp_path):
        out = tmp_path / "vault" / "Claude" / "graph"
        out.parent.mkdir(parents=True)
        assert main(["--repo", str(full_repo), "--out", str(out)]) == 0
        assert (out / "index.md").exists()
        assert (out / "modules" / "risk.manager.md").exists()

    def test_aborts_when_parent_dir_missing(self, full_repo, tmp_path, capsys):
        """iCloudがオフラインのとき、同期されない場所へ書かないための防御。"""
        out = tmp_path / "vault" / "存在しない親" / "graph"
        assert main(["--repo", str(full_repo), "--out", str(out)]) == 1
        assert not out.exists()

    def test_is_deterministic_except_index(self, full_repo, tmp_path):
        """2回実行して index.md 以外がバイト単位で同一なこと。"""
        out = tmp_path / "vault" / "Claude" / "graph"
        out.parent.mkdir(parents=True)
        main(["--repo", str(full_repo), "--out", str(out)])
        first = {p.relative_to(out).as_posix(): p.read_bytes()
                 for p in out.rglob("*.md") if p.name != "index.md"}
        main(["--repo", str(full_repo), "--out", str(out)])
        second = {p.relative_to(out).as_posix(): p.read_bytes()
                  for p in out.rglob("*.md") if p.name != "index.md"}
        assert first == second

    def test_removes_stale_notes_on_rerun(self, full_repo, tmp_path):
        out = tmp_path / "vault" / "Claude" / "graph"
        out.parent.mkdir(parents=True)
        main(["--repo", str(full_repo), "--out", str(out)])
        (full_repo / "src" / "core" / "clock.py").unlink()
        main(["--repo", str(full_repo), "--out", str(out)])
        assert not (out / "modules" / "core.clock.md").exists()
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -k "BuildAll or TestMain" -v`
Expected: FAIL — `ImportError: cannot import name 'build_all'`

- [ ] **Step 3: 最小の実装を書く**

`scripts/gen_graph_notes.py` の末尾に追記:

```python
import argparse
from datetime import datetime

# 既定の出力先（機械固有）。テストは必ず --out で差し替える。
DEFAULT_OUT = Path(
    r"C:\Users\garnet\iCloudDrive\iCloud~md~obsidian\Tama vault\Claude\graph"
)

# 節スタブを作る対象。(ノート名になるキー, リポジトリ相対パス)
TARGET_DOCS = [
    ("詳細設計書", "docs/詳細設計書.md"),
    ("概要設計書", "docs/概要設計書.md"),
    ("README", "README.md"),
    ("運用Runbook", "docs/運用Runbook.md"),
]


def build_all(repo_root: Path) -> tuple[dict[str, str], int, int]:
    """全ノートを組み立てて {出力先の相対パス: 本文} を返す。"""
    modules = discover_modules(repo_root)
    dotted_of = {m.path: m.dotted for m in modules}

    detail = repo_root / "docs" / "詳細設計書.md"
    headings = extract_role_headings(detail.read_text(encoding="utf-8")) if detail.exists() else {}

    sections = []
    for key, rel in TARGET_DOCS:
        doc = repo_root / rel
        if doc.exists():
            sections += split_sections(key, rel, doc.read_text(encoding="utf-8"))

    # 依存元（依存グラフの反転）
    dependents: dict[str, list[str]] = {m.dotted: [] for m in modules}
    for m in modules:
        for dep in m.deps:
            if dep in dependents:
                dependents[dep].append(m.dotted)

    # モジュール -> 関係する節。自分の解説がある節は除く（`## 解説` で既にリンク
    # 済みであり、`## 1. モジュール詳細` のような目次節が全モジュールから
    # 参照される偽のハブになるのを防ぐ）。
    related: dict[str, list[Section]] = {m.path: [] for m in modules}
    for sec in sections:
        # パス表記と記号名の両方で拾う。事故節はパスをほとんど書かず
        # 「RiskManager」のような記号名で書くため、記号名が主な供給源になる。
        hit = sec.modules | match_modules_by_symbol(sec.body, modules)
        # 節ノートの「関係するモジュール」にも反映する。ここを更新しないと、
        # モジュール側からは繋がっているのに節側は空、という非対称が起きる。
        sec.modules = hit
        for path in hit - sec.role_heading_paths:
            if path in related:
                related[path].append(sec)

    notes: dict[str, str] = {}
    for m in modules:
        notes[f"modules/{m.dotted}.md"] = render_module_note(
            module=m,
            role=module_role(m, headings),
            dependents=dependents[m.dotted],
            headings=headings,
            related_sections=related[m.path],
        )
    for sec in sections:
        notes[f"sections/{section_note_name(sec)}.md"] = render_section_note(sec, dotted_of)

    return notes, len(modules), len(sections)


def main(argv: list[str] | None = None) -> int:
    """CLIのエントリポイント。成功で0、中断で1を返す。"""
    parser = argparse.ArgumentParser(
        description="kabu-autoの構造をObsidianグラフビュー用ノートとして生成する"
    )
    parser.add_argument("--repo", default=".", help="リポジトリのルート（既定: カレント）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="出力先ディレクトリ")
    parser.add_argument("--dry-run", action="store_true", help="書かずに件数と衝突だけ報告する")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve()
    out_dir = Path(args.out)

    notes, n_mod, n_sec = build_all(repo_root)
    print(f"モジュール {n_mod} 件 / 節 {n_sec} 件 = ノート {len(notes)} 件")

    # 衝突検査。vaultのルートは graph の2階層上（<vault>/Claude/graph）を想定する。
    vault_root = out_dir.parent.parent
    names = [Path(rel).stem for rel in notes]
    collisions = find_collisions(names, vault_root, out_dir)
    if collisions:
        print("中断: ノート名がvault内の既存ノートと衝突しています")
        for name in collisions:
            print(f"  - {name}")
        return 1

    if args.dry_run:
        print("dry-run のため書き込みませんでした")
        return 0

    if not out_dir.parent.exists():
        # iCloudがオフラインだと親ごと見えない。勝手に作ると同期されない場所へ書いてしまう。
        print(f"中断: 出力先の親ディレクトリがありません: {out_dir.parent}")
        return 1

    written = set()
    for rel, content in sorted(notes.items()):
        target = out_dir / rel
        write_note(target, content)
        written.add(target)

    index = out_dir / "index.md"
    write_note(index, render_index(n_mod, n_sec, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    written.add(index)

    removed = cleanup_stale(out_dir, keep=written)
    print(f"書き込み {len(written)} 件 / 削除 {len(removed)} 件")
    print(f"出力先: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`import argparse` と `from datetime import datetime` はファイル冒頭の import 群へ移動する。

- [ ] **Step 4: テストを実行して通ることを確認**

Run: `python -m pytest tests/test_gen_graph_notes.py -v`
Expected: 84 passed

- [ ] **Step 5: 既存テストを壊していないことを確認**

Run: `python -m pytest -q`
Expected: 1150 passed（既存1066 + 新規84）。失敗が出たら新規テストの副作用を疑う。

- [ ] **Step 6: dry-run で実データを確認**

Run: `python scripts/gen_graph_notes.py --dry-run`
Expected: `モジュール 49 件 / 節 71 件 = ノート 120 件` と表示され、衝突が出ないこと。
衝突が出たら**書き込まずに報告して相談する**。

- [ ] **Step 7: 本番生成**

Run: `python scripts/gen_graph_notes.py`
Expected: `書き込み 121 件 / 削除 0 件`

- [ ] **Step 8: 決定性を実地で確認**

Run: `python scripts/gen_graph_notes.py && python scripts/gen_graph_notes.py`
2回目も同じ件数が出て、`index.md` 以外の内容が変わらないこと。

- [ ] **Step 9: 生成物がgitに現れないことを確認**

Run: `git status --porcelain`
Expected: 生成物（`.md`）が1件も現れない。現れたら出力先が誤っている。

- [ ] **Step 10: 仕様書 §12 の実測を行う**

Run:
```bash
python -c "
from pathlib import Path
from scripts.gen_graph_notes import build_all
notes, n_mod, n_sec = build_all(Path('.'))
sec = {k: v for k, v in notes.items() if k.startswith('sections/')}
mod = {k: v for k, v in notes.items() if k.startswith('modules/')}
no_link = [k for k, v in sec.items() if '関係するモジュール' not in v]
no_expl = [k for k, v in mod.items() if '## 解説' not in v]
print(f'モジュール言及の無い節: {len(no_link)} / {len(sec)}')
print(f'解説節の無いモジュール: {len(no_expl)} / {len(mod)}')
"
```
Expected: おおむね `モジュール言及の無い節: 41 / 71` / `解説節の無いモジュール: 20 / 49`。
**この数字をユーザーへ報告する**（案3の第二段階の判断材料。ここで対応表を勝手に作らない）。

- [ ] **Step 11: コミット**

```bash
git add scripts/gen_graph_notes.py tests/test_gen_graph_notes.py
git commit -m "feat(tools): CLIを結線してグラフノート生成を完成させた

--repo / --out / --dry-run を持つ。出力先の親ディレクトリが無い場合は
中断する（iCloudオフライン時に同期されない場所へ書かないため）。
名前衝突が1件でもあれば何も書かずに中断する。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: 運用手順への反映

**Files:**
- Modify: `C:\Users\garnet\.claude\projects\c--Users-garnet-kabu-auto\memory\obsidian_vault_sync.md`
- Modify: `docs/運用Runbook.md`

**Interfaces:**
- Consumes: 完成した `scripts/gen_graph_notes.py`
- Produces: なし（ドキュメントのみ）

- [ ] **Step 1: 記憶ファイルに手順を追記**

`obsidian_vault_sync.md` の「How to apply」の段落末尾に次を足す。

```markdown
After copying the docs, also regenerate the Obsidian graph notes:

    python scripts/gen_graph_notes.py

This writes ~121 generated notes to `Tama vault/Claude/graph/` (module notes,
section stubs, and an index). They are derived data and intentionally NOT in git —
regenerating is the only way to keep them in sync with the code. Added 2026-09-09.
```

- [ ] **Step 2: 運用Runbookに追記**

`docs/運用Runbook.md` に「Obsidianグラフノートの再生成」の項を足す。内容:

```markdown
## Obsidianグラフノートの再生成

コードやドキュメントを更新してpushしたら、vaultのグラフノートも再生成する。

```bash
python scripts/gen_graph_notes.py
```

- 出力先は `Tama vault/Claude/graph/`（モジュール49・節71・index 1）
- 生成物はgit管理外。派生データなので再生成すれば復元できる
- iCloudがオフラインだと出力先の親が見えず中断する。その場合は同期を待って再実行する
- ノート名がvault内の既存ノートと衝突した場合も中断する。表示された名前を確認すること
```

- [ ] **Step 3: 生成物がコミット対象に入っていないことを確認**

Run: `git status --porcelain`
Expected: 変更は `docs/運用Runbook.md` のみ（記憶ファイルはリポジトリ外）。

- [ ] **Step 4: コミット**

```bash
git add docs/運用Runbook.md
git commit -m "docs(ops): グラフノートの再生成手順をRunbookに追加

push後のvault同期に「python scripts/gen_graph_notes.py」を足した。
生成物はgit管理外の派生データなので、再生成しないとコードとずれる。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: vaultへドキュメントを同期**

更新した `docs/運用Runbook.md` をvaultへコピーする（既存の同期規約どおり）。

```bash
cp docs/運用Runbook.md "C:/Users/garnet/iCloudDrive/iCloud~md~obsidian/Tama vault/Claude/運用Runbook.md"
```

---

## 完了時の確認

- [ ] `python -m pytest -q` が全て通る（既存1066 + 新規84 = 1150）
- [ ] `python scripts/gen_graph_notes.py --dry-run` が衝突ゼロで完走する
- [ ] vaultに モジュール49 + 節71 + index 1 = 121ファイルが生成されている
- [ ] Obsidianのグラフビューで、モジュール群・節群・既存ドキュメントが連結して見える
- [ ] `risk.manager` のノートから依存先・依存元・解説節・関係する事故へ辿れる
- [ ] 2回連続実行で `index.md` 以外に差分が出ない
- [ ] `git status` に生成物が現れない
- [ ] 仕様書 §12 の実測値をユーザーへ報告した
