"""
main.py の WebSocket 入力に、各モードを順に巡るシナリオを送る動作確認用クライアント。

    python tools/ws_test_client.py [--url ws://127.0.0.1:8765]
"""
import argparse
import asyncio
import json

from websockets.asyncio.client import connect

# (秒, メッセージ)
SCENARIO = [
    (1.5, {"arousal": 0.3, "valence": 0.0}),  # NEUTRAL
    (1.5, {"arousal": 0.3, "valence": 0.6}),  # CALM
    (1.5, {"arousal": 0.9, "valence": 0.7}),  # EXCITED
    (1.0, {"event_trigger": "surprise"}),  # IMPULSE (部分更新: 値は直前を引き継ぐ)
    (1.5, {"arousal": 0.85, "valence": -0.7}),  # TENSE
    (1.5, {"arousal": 0.2, "valence": -0.6}),  # GLOOMY
    (1.0, {"event_trigger": "heartbeat"}),  # PULSE
    (1.0, {"event_trigger": "stop"}),  # SAFE_STOP (ラッチ)
    (1.0, {"arousal": 0.9, "valence": 0.9}),  # 停止中なので無視される
    (1.5, {"event_trigger": "resume"}),  # 解除 → EXCITED
    (0.5, {"arousal": 2.0, "valence": "bad"}),  # 不正値 → エラー応答
    (1.0, {"arousal": 0.0, "valence": 0.0}),  # IDLE
]


async def main(url: str) -> None:
    async with connect(url, compression=None) as ws:

        async def print_replies():
            async for reply in ws:
                print("  reply", reply)

        reader = asyncio.create_task(print_replies())
        for wait, msg in SCENARIO:
            print("send", msg)
            await ws.send(json.dumps(msg))
            await asyncio.sleep(wait)
        reader.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8765")
    asyncio.run(main(p.parse_args().url))
