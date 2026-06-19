"""ネットワーク関連の小さなユーティリティ"""
import socket


def is_port_available(host: str, port: int) -> bool:
    """指定 host:port が空いているか確認する。

    実際にバインドを試して即座に閉じる。SO_REUSEADDR は付けない。
    Windowsでは SO_REUSEADDR を付けると、既に他プロセスがLISTEN中のポートにも
    bind が成功してしまう（いわゆるポート割り込み）ことがあり、誤って
    「空いている」と判定してしまうため。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True
