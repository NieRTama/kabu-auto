# kabu-auto 運用Runbook（緊急対応・手動復旧手順）

レビュー P2-4 対応。ライブ運用中の異常時に「まず何を止め、何を確認し、どう復旧するか」を
即座に辿れるようにする手順書。**実資金が動くため、迷ったらまず新規発注を止める（fail-safe）。**

前提:
- ダッシュボード: `http://<host>:8080/`
- 緊急操作系API（停止/解除/緊急決済/逆指値）は `X-Emergency-Token` ヘッダー必須。
  トークンは起動時にコンソールへ一度だけ表示される（または `config.yaml` の
  `dashboard.emergency_token`）。
- 取引停止状態は `data/trading_halt.json` に永続化され、再起動を跨いで維持される。

---

## 0. まず止める（最優先・共通の初動）

新規発注を全停止し、未約定BUYをキャンセルする（損切り・緊急決済は停止中も実行可能）。

ダッシュボードの「取引停止」ボタン、または:

```bash
curl -X POST http://localhost:8080/api/halt \
  -H "X-Emergency-Token: <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"reason":"手動停止：要調査"}'
```

同時に全ポジションを成行決済したい場合は `{"close_positions": true}` を付ける。

状態確認: `GET /api/halt` または `GET /api/status`（`can_place_order` が false になる）。

---

## 1. 緊急全ポジション決済

```bash
curl -X POST http://localhost:8080/api/emergency_close -H "X-Emergency-Token: <TOKEN>"
```

挙動（`OrderManager.close_all_positions`）:
- **live/semi_live はブローカー `/positions` を正本に成行決済する**（ローカルDBは正本にしない）。
- `/positions` 取得に失敗した場合は**自動決済せず critical アラートを出して中断**する
  （実建玉を把握できない状態で当てずっぽうの決済をしないため）。→ §6 の手動決済へ。

---

## 2. API障害（kabuステーション接続不可）

症状: トークン更新失敗・`/orders`・`/positions`・`/wallet` のタイムアウト/例外。

対応:
1. §0 で取引を停止する。
2. kabuステーション本体（PC常駐アプリ）の起動・ログイン状態、`base_url`、ネットワークを確認。
3. 復旧したら `/orders`・`/positions` をダッシュボードで照会し、DBとの整合を確認（§4）。
4. 起動時の preflight（`src/core/preflight.py`）は live/semi_live で API疎通失敗時に
   起動を中断する（fail-closed）。接続が安定してから再起動する。

注意: 照合ジョブ（15秒毎の `reconcile_orders`）は API障害時に例外を飲み込んで次回再試行する。
未約定注文は OPEN のまま残り、復旧後の照合で自動収束する。

---

## 3. 約定不明・部分約定・残注文（UNKNOWN / CANCEL_FAILED）

状態の意味（`src/execution/order_status.py`）:
- `UNKNOWN`: 起動同期/照合でブローカーに該当注文が見つからない（約定/失効/取り逃しの可能性）。
- `CANCEL_FAILED`: キャンセル要求が失敗/拒否（実口座に注文が残存している可能性）。
- `PARTIALLY_FILLED`: 一部約定・残数は生存（追って約定 or タイムアウトキャンセル）。
- `PARTIALLY_FILLED_DONE`: 部分約定のままブローカー側で確定終了（残数は二度と約定しない）。

これらの**未解決注文（UNKNOWN/CANCEL_FAILED）が残る間は新規発注が抑止**され、
取引停止スイッチも解除できない。

手順:
1. 証券会社の取引サイト / kabuステーションで**実際の注文・建玉を直接確認**する。
2. 実口座の状態に合わせてDBを是正する:
   - 実際は約定していた → ブローカー実態に合わせて建玉を確認（次回 `/positions` 照合で
     ドリフト検知される。§4）。
   - 実際は失効/取消だった → 当該 Trade を解消（CANCELLED 相当へ）。
3. 未解決がゼロになったら取引停止を解除（§5）。

---

## 4. 建玉ドリフト（DBとブローカー実建玉の不一致）

検知: 定期照合 `reconcile_positions_with_broker()` がDBの Position とブローカー
`/positions`（LeavesQty）を比較し、差異があれば critical ログ + アラートを出し、
**自動で kill switch を作動**させて新規発注を止める（fail-closed）。

対応:
1. アラートの差分（`DB=○○株 ブローカー=○○株`）を確認。
2. ブローカー実態を正としてDBの Position を是正する。
3. 取り逃した約定が原因なら `/orders` 照合で Fill を取り込めるか確認。
4. 整合後、未解決ゼロを確認して取引停止を解除（§5）。

---

## 5. 取引再開（停止解除）

```bash
curl -X DELETE http://localhost:8080/api/halt -H "X-Emergency-Token: <TOKEN>"
```

- 未解決注文（UNKNOWN/CANCEL_FAILED）が1件でも残っていると **409 で解除を拒否**する。
  先に §3 を完了すること。
- 合計ドローダウン上限（実現損失＋含み損）到達で自動停止した場合は、損失要因を確認し、
  必要なら日次リセット（翌営業日 8:25 の `risk_reset`）まで待つか、建玉を整理する。

---

## 6. 緊急決済が失敗したとき（最終手段：完全手動）

`/positions` 取得失敗や API 全断で自動決済できない場合:
1. **証券会社の取引サイト / 電話注文**で建玉を直接成行決済する。
2. kabu-auto は §0 で停止したまま放置（誤発注を防ぐ）。
3. 市場と口座が落ち着いてから、DBを実態に合わせて是正し、§4→§5 で復旧。

---

## 7. 損切りの保険（ブローカー側逆指値ストップ）

アプリの損切り監視（`stop_loss_check`）が止まっても証券会社側で発動する逆指値成行を発注できる:

```bash
curl -X POST http://localhost:8080/api/stop_loss \
  -H "X-Emergency-Token: <TOKEN>" -H "Content-Type: application/json" \
  -d '{"symbol":"7203"}'
```

トリガー価格は保有建玉の平均取得単価 × (1 + stop_loss_pct) で自動算出される。

---

## チェックリスト（異常検知アラートを受けたら）

1. [ ] §0 で新規発注を停止したか
2. [ ] ブローカー実態（注文・建玉）を直接確認したか
3. [ ] DBとブローカーの差異を是正したか
4. [ ] 未解決注文がゼロになったか
5. [ ] 損失要因を記録したか（再発防止）
6. [ ] 取引停止を解除したか（§5）
