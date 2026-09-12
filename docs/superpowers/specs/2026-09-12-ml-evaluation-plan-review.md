# kabu-auto 修正計画レビュー

レビュー日：2026-09-12

**判定：このまま一式を実装へ渡すのは保留。段階間の接続、価格・時刻の契約、失敗時の整合性を先に修正する。** モジュール分割の方向性は採れるが、記載された実装とテストをそろえても、評価基盤として必要な動作を満たさない箇所がある。特にR01〜R07、R11〜R14は、今回の「正しく測れる状態を作る」という目的に直接関わる。

対象はZIP内のREADME、設計書、元レビュー、実装計画11本。段階AはREADMEで実装済みとされているが、実装リポジトリは同梱されていない。以下は**計画に掲載されたコード・手順のレビュー**であり、本番での発生や既存実装の不具合を確認したという意味ではない。[対象と状態](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/README.md:16)

原本ZIPと計画書は変更していない。Bot・証券API・本番DBには接続せず、pytestも実行していない。算術と日付条件の反例5件は独立したJavaScript計算で確認した。[計算結果](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/counterexamples.json)

P1は該当機能の実装受入れ前に直すべき問題、P2は評価の精度・監査・完了判定に関わる問題としている。採用AUCや利益目標を一律に決めない方針、自動昇格をしない方針自体は欠陥として扱っていない。[設計書の完了範囲](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/specs/2026-09-10-ml-evaluation-foundation-design.md:531)

**R01 / P1：調整後の終値と、別の価格基準のOHLCが混在する。**

段階Aの`load_ohlcv(price_basis="adjusted")`は`close`だけを調整値へ差し替え、OHLはそのまま返す。計画のテスト入力O=1000、H=1010、L=990、C=1005、AdjC=502.5なら、読み出すとL=990 > C=502.5になる。従来のyfinanceの`auto_adjust=True`はOHLにも同じ係数を掛けるため、既存経路も入力が変わる。[A・501〜505行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-10-ml-evaluation-stage-a.md:501)・[yfinance公式実装](https://raw.githubusercontent.com/ranaroussi/yfinance/main/yfinance/utils.py)

修正案：特徴量用の調整OHLCと、執行・現金計算用の価格を別入力にする。調整系列はOHLCに同じ係数を適用する。確認は終値だけでなくOHLCの整合性と、特徴量生成から単一イベントの約定まで通す。段階Aが実装済みとされるため、最初に実コードとの照合が必要。

**R02 / P1：v2学習の返却型を保存側が扱えず、候補モデルを保存できない。**

Fの`_fit_candidate()`は独自ラッパー`CurrentLightGBM`を返す。C2のラッパーが持つのは`_model`と`predict_proba()`で、`booster_`も`save_model()`もない。一方Eは`getattr(model, "booster_", model).save_model(...)`を呼ぶ。この組合せは保存時の`AttributeError`となり、`train_as_candidate()`が捕捉して`None`にする。[F・371〜376行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-f.md:371)・[C2・756〜801行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c2.md:756)・[E・287〜289行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-e.md:287)

修正案：学習型と保存型の契約を合わせる。二値がそろったモデルと、単一クラス時の定数モデルの保存方式も区別する。成功ケースでは`model_id is not None`、保存物の再読込、予測一致を必須にする。現テストの`if res.model_id is not None:`では、保存に失敗しても成功テストが通る。[F・245〜257行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-f.md:245)

**R03 / P1：ダッシュボードのv2経路に特徴量・ルールスコアが供給されない。**

Fはraw OHLCVをそのまま`MarketData`へ渡す。D2も各日の足をそのまま`decide()`へ渡す。最終版`_v2_score_fn()`は存在しない`rule_score`を0、欠けた特徴量も0にする。既定の正の買い閾値では全候補が落ち、特徴量未接続が「取引ゼロの正常なバックテスト」に見える。[F・796〜810行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-f.md:796)・[D2・904〜910行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-d2.md:904)・[F・1003〜1007行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-f.md:1003)

修正案：調整系列から因果的に特徴量とルールスコアを生成し、当日の有効性マスクを含む入力へ結線する。欠落を0で補完しない。テストで`rule_score`を手で渡すだけでなく、合成OHLCV→エンドポイント→T+1約定を通す。期間がデータに覆われない場合も明示的に扱う。

**R04 / P1：過去検証が現在の昇格モデルに固定され、週次再学習も実際の判断に届かない。**

Fは`load_current()`の戻り値を閉包に保持し、過去の全日付に使う。学習期間と検証開始日の前後関係は検査しないため、評価期間を学習済みのモデルでも使用できる。さらに`run_walkforward()`へ`retrain`と`train_model`を渡さず、D2の戦略factoryも受け取った`model`を`score_fn`へ渡さない。[F・994〜1011行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-f.md:994)・[F・804〜810行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-f.md:804)・[D2・1714〜1763行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-d2.md:1714)

修正案：過去評価用には各判断時点で利用可能なモデルを明示し、学習締切・再学習関数・実際に使うモデルIDを接続する。現在モデルを固定して過去へ当てる診断が必要なら、採否用のwalk-forward成績と分ける。確認は、未来の学習データを変えても過去予測が変わらないことと、再学習後のモデル変更が予測に反映されること。

**R05 / P1：推論失敗を握りつぶし、`degraded`が立たない。**

Fの`score_fn()`は推論例外を捕捉して`(rule, None)`を返す。D2が`degraded`を立てるのは外へ届いた例外なので、このエラーは記録されない。Fの説明にある「walkforward側が別途立てる」は成立しない。`StrategyConfig`へ設定中の`on_model_failure`を渡していない点も、`halt_new`指定を既定の`rule_only`へ戻してしまう。[F・1008〜1010行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-f.md:1008)・[D2・1483〜1497行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-d2.md:1483)・[F・800〜802行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-f.md:800)

修正案：意図したML無効・モデル未昇格・推論障害を区別し、障害情報を結果へ伝播させる。設定値を明示的に渡す。実際のscore関数を含めて推論失敗を注入し、永続化されたrunが`degraded`になり、比較・昇格対象から外れることを確認する。

**R06 / P1：学習締切より後に確定したラベルが入る。**

C1は判断日を`train_end`で制限するが、ラベル終了日は`val_start`より前なら許可する。`train_end=1/5、val_start=1/12`のとき、1/5判断・1/8確定のラベルが通る。分割日は候補イベントのある日から作るため、こうした日付の空白を扱う必要がある。これは`train_end`を学習締切と呼ぶ契約との不一致。[C1・479〜485行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c1.md:479)・[締切定義](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c1.md:220)

修正案：学習の実行時点を明示し、判断日とラベル確定日の双方をその締切で制限する。現契約なら`label_end_at <= train_end`が必要。候補がない営業日を挟み、締切後に確定するラベルだけを書き換えるテストを追加する。条件式への独立代入で反例を確認済み。

**R07 / P1：別設定での評価が、過去runの実績ラベルを上書きする。**

B2の`event_id`は銘柄と判断日だけ。一方C2はそのIDだけで実績をupsertし、予測との結合にも使う。同じ銘柄・日付を別のコストや退出条件で評価すると、過去runも新しい実績へ結びつく。Prediction側の`evaluation_run_id`だけでは防げない。[B2・220〜226行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-b2.md:220)・[C2・335〜346行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c2.md:335)・[読込時の結合](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c2.md:366)

修正案：ラベル契約が不変になるキーを実績・予測の双方に持たせる。例は`dataset_id + event_id`。shadowでは将来データ追加でも変わらない契約IDが必要。異なるコストで2回保存しても、最初のrunの明細と指標が変わらないことを確認する。

**R08 / P1：翌朝の約定価格で現金・購入上限を再確認しない。**

前日終値で数量を決め、翌朝の寄り値・スリッページ・手数料でそのまま購入する。D1の`apply_buy()`にも残高不足の拒否がない。例えば現金10万円、前日100円で900株、翌朝112円、手数料0.1%なら残高は−900.8円になる。[D2・878〜887行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-d2.md:878)・[D1・415〜426行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-d1.md:415)

修正案：約定時点で利用可能な価格と手数料を使い、現金・銘柄・セクター上限を満たす数量へ縮小するか注文を拒否する。単元も守る。ギャップ上昇、複数注文、手数料込みの現金非負を確認する。数量配分の段階だけのテストでは足りない。

**R09 / P1：出来高制限が買いだけに作用し、売りは常に全量約定する。**

買いは`entry_fill_limited()`を使う一方、ストップ退出・翌日成行退出は出来高を受け取らない`exit_fill()`へ全数量を渡す。900株保有、当日出来高1000株、参加率上限10%でも900株を売れる計算になる。[D2・566〜571行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-d2.md:566)・[D2・603〜608行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-d2.md:603)

修正案：売りにも数量制限と残数量の繰越を実装する。同日・同銘柄の出来高枠の共有規約も定める。損切りを指示しても全量は売れないケースで、残存保有・翌日の退出意図・買付余力が整合することを確認する。

**R10 / P1：売りスコアによる退出が未接続。**

D2の`_drive_exits()`は`_observation()`へscoreを渡さず、常に既定の`None`になる。B1の`SIGNAL_SELL`条件は成立せず、ストップか期限満了まで保有する。B2の「段階Dで結線する」という記述も未達。[D2・547〜555行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-d2.md:547)・[B1・836行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-b1.md:836)・[B2・630行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-b2.md:630)

修正案：保有銘柄についても当日までの情報から退出スコアを計算し、ラベル生成とwalk-forwardで同じ契約にする。ストップ未到達・期限前・売り閾値到達で翌営業日寄りに退出する統合テストを追加する。

**R11 / P1：モデル参照を切り替えた後のDB失敗から復旧できない。**

Eの`promote()`は`set_current()`を先に実行し、その後で昇格履歴をcommitする。DB書込み失敗や両処理間のプロセス停止では、参照は新モデルなのに昇格記録がない状態が残る。参照ファイル単体の原子的置換は、DBを含む取引の原子性ではない。[E・953〜967行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-e.md:953)・[Python公式 os.replace](https://docs.python.org/3.11/library/os.html#os.replace)

修正案：昇格の状態を永続化し、再起動時に完了・取消を判定できる回復手順を設計する。単純に順序を逆にするだけでも逆向きの不整合が残る。現在の完了確認は事前拒否と参照の読直しだけで、commit失敗・切替直後の停止を試していない。[E・1533〜1537行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-e.md:1533)

**R12 / P1：「不変」の候補ディレクトリを同じIDで上書きできる。**

`save_candidate()`は`exist_ok=True`で既存ディレクトリを受け入れ、モデルとメタを直接上書きする。そのIDをcurrentやrollback先が指していれば、昇格操作なしに実体が変わる。モデル書込み後にメタ保存が失敗した場合も、既存ディレクトリは復元しない。[E・284〜289行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-e.md:284)・[E・1479〜1487行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-e.md:1479)

修正案：既存IDへの書込みを拒否し、別の一時ディレクトリで全成果物の保存・読込検証を完了してから公開する。current・rollback先と同じIDの再保存、モデル保存後の失敗を注入し、元ファイルが不変であることを確認する。

**R13 / P1：実績未確定の予測だけで「評価済み」と判定できる。**

Eの`check_promotable()`は予測明細の行数だけを見る。C2の読込関数は実績がなくても行を返すので、ラベルが全て未確定のshadow予測でもこの条件を通せる。`degraded`も保存済み実行記録から取得せず、呼出し側のboolを信用する。テストの「clean candidate」も実績を保存していない。[E・912〜929行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-e.md:912)・[C2・351〜385行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c2.md:351)・[E・685〜692行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-e.md:685)

修正案：評価実行の完了状態、実績の確定、評価用途、モデル・データ・特徴量契約、保存済みdegraded状態を根拠として検証する。成績の採用閾値を今決める必要はないが、未評価を評価済みにしない条件は必要。実績なし・shadowのみ・不正な評価ID・保存状態との不一致を拒否するテストを加える。

**R14 / P1：設定保存に失敗しても、稼働中のリスク設定だけ変わる。**

integrityは`_custom`の変更と`_apply_if_active()`を保存より先に行う。アクティブ設定の上書き後にJSON書込みや`os.replace`が失敗すると、APIが失敗を返しても実行中の損切り幅等は新設定、保存ファイルは旧設定になる。一時ファイルへの保存だけでは解消しない。[integrity・597〜608行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-integrity.md:597)・[保存処理](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-integrity.md:723)・[元レビューの要求](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/specs/kabu-auto-detailed-review_20260910.md:249)

修正案：検証済み候補設定を共有状態と分離して保存し、成功後に実行中設定を一括で切り替える共通処理にする。import・update・set_activeを同じ契約にまとめる。非アクティブ設定追加の失敗だけでなく、アクティブ設定・履歴・保存内容が全て不変であることを確認する。

**R15 / P1：コンソール限定とする初期設定トークンを、通常ログへ出す。**

securityはトークンを`logger.warning()`へ平文で渡す。計画は通常のファイルシンクも前提としており、専用のコンソール出力にはなっていない。追加マスクの正規表現も、日本語の説明に続く裸のトークンには一致しない。初期設定が完了する前にログを読める者へ、LAN経由の初期ユーザー作成に必要な資格情報が渡る。[security・430〜439行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-security.md:430)・[マスクとファイルシンク](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-security.md:835)・[Loguru公式](https://loguru.readthedocs.io/en/stable/api/logger.html)

修正案：専用のローカルコンソール出力へ分離し、ファイル・通知シンクには配送しない。テストはマスク関数単体ではなく、初回起動を模した実際の出力先ごとに秘密値の有無を確認する。

**R16 / P1：URLクエリの長期APIトークンを受け付け続ける。元レビューF13の未解消。**

securityは`?token=`を受理してセッションを発行し、その後でトークンのないURLへリダイレクトする。しかし最初のHTTPリクエストに長期APIトークンが入るため、アクセスログ等への流出経路は残る。漏れたAPIトークンからはログアウト後も新セッションを取得できる。元レビューはクエリトークンを受け付けない方針。[security・605〜633行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-security.md:605)・[元レビュー・265行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/specs/kabu-auto-detailed-review_20260910.md:265)・[OWASP REST Security](https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html)

修正案：ブラウザはログインからセッションを発行し、APIトークンはヘッダー認証に限定する。互換移行が必要なら、長期APIトークンをURLで再使用しない別の移行手順として決める。単にリダイレクトしたことを漏えい防止の合格条件にしない。

**R17 / P1：共通ログマスクが例外本文・トレースバックを処理しない。**

`_mask_record()`が変更するのは`record["message"]`だけ。`logger.exception()`等の例外フィールドは別に整形されるため、Webhook URL等を含む例外文字列が残る。`diagnose=False`も例外文字列そのものを除去する設定ではない。[security・862〜880行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-security.md:862)・[Loguru公式の例外整形説明](https://loguru.readthedocs.io/en/latest/resources/troubleshooting.html#why-is-the-captured-exception-missing-from-the-formatted-message)

修正案：最終出力に含まれる例外までマスクするか、例外情報を許可した項目へ構造化する。`logger.exception`、`opt(exception=True)`、連鎖例外について実際のシンク出力を検証する。通知コードのf-stringだけを直したことを、共通対策の完了としない。

**R18 / P1：RSI修正がlegacyにも波及し、宣言した互換保証を破る。**

B1は共有の`_rsi()`を直接変更する。単調上昇・横ばい系列のRSIがNaNから100・50に変わり、旧`compute_indicators()`と`build_features()`の戻り値・除外行も変わる。`legacy`へ設定を戻しても旧結果へ戻せない。回帰テストは改修後の旧APIと新APIを比較しており、改修前との比較ではない。[B1・124〜143行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-b1.md:124)・[互換保証](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-b1.md:20)・[回帰テスト](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-b1.md:272)

修正案：v2専用指標へ隔離するか、RSI是正を全経路へ適用する意図的な仕様差分として互換保証を改める。後者も合理的な選択肢だが、現状の「既存公開関数の結果は同一」とは両立しない。旧実装の固定期待値で差分を見えるようにする。

**R19 / P2：学習窓候補ごとに検証期間まで変わり、公平に比較できない。**

C2の`select_training_window()`は候補窓でouter学習集合を切ってから、inner foldを作り直す。短い窓と拡大窓で評価日・件数が違い、比較値も収益率の総和なので、学習窓の効果と評価期間・件数の差が混ざる。[C2・1731〜1752行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c2.md:1731)

修正案：共通のinner検証期間を先に固定し、候補窓は各innerの学習側だけに適用する。全候補で検証event_idが同じことを確認する。また、本関数の選択結果を評価・最終学習へ渡す接続も計画一式では確認できないため、結線先を明示する。

**R20 / P2：実現損益から買付手数料が抜ける。**

D1は買付手数料を現金から引くが平均取得単価に含めず、売却時の`realized`は売付手数料だけを差し引く。同値で10万円分を往復売買し片道手数料0.1%なら、現金は200円減るが実現損益は−100円になる。日次・取引明細にもこの値を記録する。[D1・415〜448行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-d1.md:415)・[D2・899〜901行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-d2.md:899)

修正案：買付手数料を保有原価として保持し、部分売却に応じて配賦する等の契約を定める。全ポジション決済後の実現損益合計と現金増減の一致を、買い増し・部分売却を含めて確認する。ラベル側の往復コストとの一致も必要。

**R21 / P2：shadowで現行側の判断が保存されない。**

`compare()`は現行・候補の両確率と採否を返すが、`record_shadow()`が保存するのは候補の確率だけ。現行モデルID、現行確率、閾値、採否、一致区分を保存しないため、再起動後に当時の比較や見送り理由を復元できない。関数名や一時的な戻り値だけでは「並行記録」を満たさない。[E・1260〜1290行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-e.md:1260)・[E・1294〜1313行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-e.md:1294)

修正案：両モデルID・両確率・判断閾値・採否・入力契約を同じ比較記録へ保存する。DBから読み直して比較内訳を再計算できることを確認する。現行モデル未昇格時の記録も別状態として残す。

**R22 / P2：キャッシュ・DB価格を読むたびに観測時刻を現在へ更新する。**

integrityはキャッシュ値に`observed_at=now`を付け直す。例えば許容鮮度2秒で4秒前のキャッシュを読んでも、鮮度0秒として扱われる。DB終値にも読出し時刻を設定し、足の営業日`session`を埋めないので、何日前の終値かを辿れない。[integrity・403〜407行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-integrity.md:403)・[DB終値](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-integrity.md:424)

修正案：キャッシュには元の観測時刻、DB価格には足の営業日を保持する。読出し時刻が要るなら別項目にする。キャッシュ再読込で観測時刻が変わらず、古いDB足の日付が保存されることを確認する。F10の新規発注経路への未結線は下記の残件として別に扱う。

**R23 / P2：C1のembargoテストは掲載実装と必ず矛盾する。**

テストは検証開始1/7に対し、1/9のイベントがembargoなしの学習集合に入ると要求する。しかし掲載実装は`decision_at < val_start`で先に除外するため、そのassertは満たせない。`Expected: PASS`は記載コードからは成立しない。[C1・429〜437行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c1.md:429)・[実装](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c1.md:479)

修正案：今回が片方向walk-forward専用なら、この不正なFoldを拒否するテストへ変える。検証期間の前後を学習に使う分割も実装するなら、purgeの契約から分けて設計する。今の前方制約を緩めてテストだけ通す修正は避ける。

**R24 / P1：本番DBの「書き換えなし」検査に通常の初期化処理を使う。**

integrityの確認8は「削除も書き換えもしない」としながら`cfg.load(); db.init()`を実行する。同じ計画は`init()`内のWAL設定を前提とし、他計画は自動テーブル作成・列追加を前提としている。旧スキーマの本番DBに対して、この手順では書換禁止を保証できない。[integrity・1189〜1198行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-integrity.md:1189)・[init内の接続](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-integrity.md:904)・[B2の自動移行前提](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-b2.md:18)

修正案：初期化・マイグレーション・WAL設定を呼ばない読取専用の検査接続へ分離し、旧スキーマは変更せず未対応として報告する。実装本体がないため具体的なDDLの実行順は未確認だが、書換禁止を接続モードで担保しない手順は修正対象。この監査コードは今回実行していない。

**今回の実装だけで「是正完了」にしてはいけない残件**

- **F10の新規発注への適用。** `get_price_quotes()`を`validate_buy`等へ使わせる結線は明示的にスコープ外。今回作るのは区別の仕組みであり、古い価格・価格不明時の新規リスク判断を止める変更は完了しない。これは隠れた実装バグではなく、是正範囲を先送りしているという判断事項。[integrity・1208行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-integrity.md:1208)
- **F15の新規不整合防止。** 外部キー宣言はスコープ外。制約のない参照列は`PRAGMA foreign_keys=ON`だけでは保護されず、今回は主に検出段階となる。修正範囲の説明と完了表も合わせる。[integrity・1207行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-review-fixes-integrity.md:1207)・[SQLite公式](https://www.sqlite.org/foreignkeys.html)
- **分割・配当とポートフォリオの接続。** Aの分割不変テストはテスト内で価格を割り、数量へ比率を掛けている。これだけでは実際の読込系列・企業行動イベント・D2の保有に接続されたことを確認できない。分割だけでNAVが増えないという完了条件には、実際の経路を通すテストが要る。「Yahoo価格を二重調整している」とは今回断定しない。[A・619行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-10-ml-evaluation-stage-a.md:619)・[設計書の完了条件](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/specs/2026-09-10-ml-evaluation-foundation-design.md:692)
- **評価実行の再現用記録。** Fの保存は`config_hash=""`、設定JSONもstrategy節のみ。リスク・手数料・数量制限等を含む実行時設定一式、入力データID、コード版、実際のモデル使用履歴を保存する接続が足りない。実行終了後の可変設定を読むのではなく、開始時に固定して全処理へ渡す。新データセット・候補モデルを含むバックアップと復元確認も設計要求にはあるが、E/Fの実装タスクでは確認できない。[F・813〜823行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-f.md:813)・[設計書・608行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/specs/2026-09-10-ml-evaluation-foundation-design.md:608)
- **内側early stopping。** C2は内側で行うと書くが、`fit_inner()`は通常の`fit()`を呼ぶだけで、木数50/200を固定する。固定木数の比較自体を否定するものではない。内側で木数を選ぶ実装を追加するか、実施しない設計差分として明記する。[C2・1246〜1252行](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c2.md:1246)・[モデル定義](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c2.md:786)

**修正と受入れの順序案**

1. 段階Aの実コードを照合し、価格基準と企業行動の契約を確定する。
2. C1/C2の締切・イベント識別・公平な比較条件を直す。評価データを作り直す前にここを固定する。
3. Fの学習→保存→読込、OHLCV→特徴量→判断、再学習→推論→履歴を、実際の型を通す少数の統合テストで結線する。
4. Dの現金・退出・流動性と、E/設定の失敗回復を確認する。事前拒否だけでなく、処理途中の障害を試す。
5. securityの秘密値の出力先と認証経路を確認し、F10/F15の未完範囲を完了表へ明記する。

上記の必要な確認が通った後に実データで比較する。テスト本数の増加、ソースに関数名があること、例外が外に出ないことを機能の成功とは扱わない。現在のFの条件付きassert、Eの事前拒否だけの原子性確認、C1の矛盾テストはそのまま完了基準に使えない。[Fのテスト](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-f.md:245)・[Eの完了確認](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-12-ml-evaluation-stage-e.md:1533)・[C1のテスト](C:/Users/lamoc/codex/reviews/kabu-auto-20260912/kabu-auto-plans/plans/2026-09-11-ml-evaluation-stage-c1.md:429)

**外部確認と限界**

外部技術仕様は、本文にリンクしたPython、yfinance、Loguru、OWASP、SQLiteの一次資料を根拠としている。Google検索URLの取得は失敗し、接続できるブラウザもなかったため、Googleでの裏付けは完了していない。利用可能なWeb検索と一次資料で確認した。計画に載る期待値を本体テストの実行結果としては扱っていない。

対象ZIPのSHA256：`47516CEDACC4FF2556F72E969448F5A66B5129DF16E6E3A573A81BDD99054364`。このレビュー用作業ディレクトリには、展開した原文、レビュー、独立計算の結果を保存している。
