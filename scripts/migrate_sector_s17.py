"""ウォッチリストと保有ポジションのセクターを J-Quants の17業種名（S17Nm）へ一括で書き換える。

2026-11-18: セクターを yfinance の英語分類から17業種へ統一した。混在すると同じ業種が
別セクターとして数えられ、集中判定が効かない。保有ポジションの多くは sector が空で、
集中判定に数えられていなかった（lots.rebuild_position は空のときしか補完しない）。

kabu-auto 停止中に実行する（稼働中はウォッチリストをメモリから書き戻すため、変更が消える）。
  python scripts/migrate_sector_s17.py          # 変更内容の表示のみ
  python scripts/migrate_sector_s17.py --apply  # 書き換え（事前にDBを VACUUM INTO で退避）
取引履歴（trades）の sector は過去の記録なので書き換えない。
"""
import json
import socket
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

from src.data.market_data import jquants_master  # noqa: E402

WATCHLISTS = REPO / "watchlists.json"
DB = REPO / "data" / "kabu_auto.db"


def lookup_all(codes: list[str]) -> dict[str, str]:
    """銘柄→17業種名。取れなかった銘柄は含めない（Free は 5回/分 のため間隔を空ける）"""
    out = {}
    for i, code in enumerate(codes):
        if i:
            time.sleep(13)  # ponytail: Free の 5req/分に合わせた固定待ち。有料化したら縮める
        s17 = jquants_master(code).get("S17Nm")
        if s17:
            out[code] = s17
        print(f"  {code}: {s17 or '（取得できず・変更しない）'}")
    return out


def plan(wl: dict, positions: list[tuple], s17: dict[str, str]) -> tuple[list, list]:
    """(ウォッチリスト変更, ポジション変更) を返す。各要素は (リスト名or'positions', code, 旧, 新)"""
    wl_changes = [(name, e["code"], e.get("sector", ""), s17[e["code"]])
                  for name, entries in wl["lists"].items() for e in entries
                  if e["code"] in s17 and e.get("sector", "") != s17[e["code"]]]
    pos_changes = [("positions", sym, old, s17[sym])
                   for sym, old in positions if sym in s17 and old != s17[sym]]
    return wl_changes, pos_changes


def apply_watchlists(wl: dict, s17: dict[str, str]) -> None:
    for entries in wl["lists"].values():
        for e in entries:
            if e["code"] in s17:
                e["sector"] = s17[e["code"]]


def main() -> int:
    load_dotenv(REPO / ".env")  # import 時に読むとテストの環境変数を汚すため main 内で読む
    if not __import__("os").environ.get("JQUANTS_API_KEY"):
        print(".env に JQUANTS_API_KEY がありません")
        return 1
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", 8080)) == 0:
            print("kabu-auto が稼働中（8080で待受）です。停止してから実行してください")
            return 1
    apply = "--apply" in sys.argv
    wl = json.loads(WATCHLISTS.read_text(encoding="utf-8"))
    con = sqlite3.connect(DB)
    positions = con.execute("select symbol, sector from positions").fetchall()
    codes = sorted({e["code"] for es in wl["lists"].values() for e in es} | {p[0] for p in positions})
    print(f"J-Quants で {len(codes)} 銘柄を照会します（約{len(codes) * 13 // 60}分）")
    s17 = lookup_all(codes)
    wl_changes, pos_changes = plan(wl, positions, s17)
    for c in wl_changes + pos_changes:
        print(f"  [{c[0]}] {c[1]}: {c[2] or '（空）'} → {c[3]}")
    print(f"ウォッチリスト {len(wl_changes)} 件・ポジション {len(pos_changes)} 件")
    if not apply:
        print("表示のみ。書き換えるには --apply を付けて再実行")
        return 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DB.with_name(f"kabu_auto_before_s17_{stamp}.db")
    con.execute("VACUUM INTO ?", (str(backup),))
    WATCHLISTS.with_name(f"watchlists_before_s17_{stamp}.json").write_text(
        WATCHLISTS.read_text(encoding="utf-8"), encoding="utf-8")
    with con:
        con.executemany("update positions set sector=? where symbol=?",
                        [(new, sym) for _, sym, _, new in pos_changes])
    apply_watchlists(wl, s17)
    WATCHLISTS.write_text(json.dumps(wl, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 不変条件: 照会できた銘柄はすべて17業種名になっている
    bad = [sym for sym, sec in con.execute("select symbol, sector from positions")
           if sym in s17 and sec != s17[sym]]
    assert not bad, f"ポジションの書き換え漏れ: {bad}"
    print(f"完了。退避先: {backup.name} / watchlists_before_s17_{stamp}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
