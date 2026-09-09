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
