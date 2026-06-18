# サードパーティ・ライセンス表示（THIRD-PARTY NOTICES）

本ソフトウェア「kabu-auto」は、以下のオープンソースソフトウェアを利用しています。
各ソフトウェアはそれぞれの著作権者に帰属し、対応するライセンスの条件の下で
提供されています。再配布・販売にあたっては、各ライセンスが要求する著作権表示・
ライセンス本文の添付義務に従ってください（MIT/BSD/Apache-2.0 はいずれも
バイナリ/ソース配布時の表示を要求します）。

| パッケージ | ライセンス | プロジェクト |
|---|---|---|
| requests | Apache-2.0 | https://github.com/psf/requests |
| yfinance | Apache-2.0 | https://github.com/ranaroussi/yfinance |
| pytz | MIT | https://pythonhosted.org/pytz/ |
| websocket-client | Apache-2.0 | https://github.com/websocket-client/websocket-client |
| pandas | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| numpy | BSD-3-Clause | https://github.com/numpy/numpy |
| scikit-learn | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| lightgbm | MIT | https://github.com/microsoft/LightGBM |
| SQLAlchemy | MIT | https://github.com/sqlalchemy/sqlalchemy |
| FastAPI | MIT | https://github.com/tiangolo/fastapi |
| uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn |
| APScheduler | MIT | https://github.com/agronholm/apscheduler |
| loguru | MIT | https://github.com/Delgan/loguru |
| pydantic | MIT | https://github.com/pydantic/pydantic |
| pydantic-settings | MIT | https://github.com/pydantic/pydantic-settings |
| python-dotenv | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| PyYAML | MIT | https://github.com/yaml/pyyaml |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |

`uvicorn[standard]` は uvloop / httptools / websockets / watchfiles 等の追加依存を
含みます（いずれも MIT または BSD系）。完全な依存ツリーと各ライセンス本文は、
配布物に対し以下で生成・同梱してください。

```bash
pip install pip-licenses
pip-licenses --format=plain-vertical --with-license-file --with-urls > THIRD-PARTY-LICENSES.txt
```

> 注意：本一覧は requirements.txt 記載の直接依存に基づくひな型です。各ライセンスの
> 正確な版・本文は配布時点で `pip-licenses` 等を用いて再生成し、添付してください。
