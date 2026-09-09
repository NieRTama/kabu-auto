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
    _write(
        tmp_path / "src" / "core" / "config.py",
        '''
        """設定管理のモジュール。"""


        class Cfg:            # 5文字未満なので採らない
            pass


        class SharedThing:    # clock.py にも同名があるので曖昧として捨てられる
            pass


        def load():           # 8文字未満・アンダースコア無しなので採らない
            pass
        ''',
    )
    _write(
        tmp_path / "src" / "core" / "clock.py",
        '''
        """時刻の一元化。"""


        class SharedThing:    # config.py にも同名がある
            pass
        ''',
    )
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

        単語境界に \\b を使うとPythonのUnicode既定で「の」が単語文字扱いになり、
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
