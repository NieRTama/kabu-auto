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
