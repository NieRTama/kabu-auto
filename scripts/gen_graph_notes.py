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
