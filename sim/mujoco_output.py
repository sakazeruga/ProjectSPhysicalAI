"""
MuJoCo 上のロボットを出力先とする ActuatorOutput 実装。

SerialController と同じインターフェースなので、main.py の --output を切り替えるだけで
実機の代わりにシミュレーションを動かせる。物理計算は専用スレッドで実時間に同期して進める。
"""
from __future__ import annotations

import logging
import threading
import time

from interfaces import ActuatorCommand, ActuatorOutput
from sim.touch_care import TouchCareConfig, TouchCareSim

logger = logging.getLogger(__name__)


class MujocoOutput(ActuatorOutput):
    def __init__(self, config: TouchCareConfig | None = None, viewer: bool = True) -> None:
        self.sim = TouchCareSim(config)
        self._viewer_enabled = viewer
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mujoco", daemon=True)
        self._frames = 0

    async def start(self) -> None:
        self._thread.start()

    def submit(self, cmd: ActuatorCommand) -> None:
        with self._lock:
            self.sim.command(cmd)
        self._frames += 1

    async def stop(self) -> None:
        with self._lock:
            self.sim.command(ActuatorCommand.safe())
        time.sleep(0.5)  # 離脱の動きを見届ける
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def is_connected(self) -> bool:
        return self._thread.is_alive()

    def stats(self) -> dict[str, float]:
        with self._lock:
            s = self.sim.sample()
        return {"frames": self._frames, "force": s.force, "reflex": self.sim.reflex_count}

    def _run(self) -> None:
        viewer = None
        if self._viewer_enabled:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
        period = 0.01
        next_step = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                self.sim.step(period)
            if viewer is not None:
                if not viewer.is_running():
                    logger.warning("Viewer closed")
                    viewer = None
                else:
                    self.sim.sync_visuals()
                    viewer.sync()
            next_step += period
            delay = next_step - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_step = time.perf_counter()  # 計算が間に合わない周期は詰めない
        if viewer is not None:
            viewer.close()
