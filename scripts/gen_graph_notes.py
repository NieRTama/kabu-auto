"""kabu-autoの構造をObsidianのグラフビュー用Markdownノートとして生成する。

モジュールのimport関係・設計書の見出し・節本文のパス言及を解析し、相互リンクした
ノート群をObsidian vaultへ書き出す。生成物は派生データなのでgit管理外に置く。

設計: docs/superpowers/specs/2026-09-09-obsidian-graph-notes-design.md

kabu-auto本体（src/）には依存しない。src/ はテキストとして読むだけ。
"""
import ast
import re
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
