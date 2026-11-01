"""
タッチケア シミュレーション: ActuatorCommand → MuJoCo 上の Franka Panda の接触動作。

ハプティクスデバイス用の 3 出力を、前腕をなでるロボットの動作へ次のように写す:

    linear_position     : 前腕に沿ったパッド位置 (0%=手首側 .. 100%=肘側)
    translation_speed   : (StateMapper が max_stroke_speed 以内に制限済み。ここでは二重に上限をかける)
    vibration_intensity : パッド内の振動モータ (押し付け力へ高周波成分として重畳)

制御はトルク指令のカルテシアン・インピーダンス制御。高さ方向は接触力センサを使う
アドミタンス制御で、腕の高さが変わっても押し付け力を一定に保つ。パッドはバネで支持した
直列弾性構造で、接触の瞬間的な変動をアーム本体の慣性から切り離している。

    安全機構 (いずれもホストの指令なしで動く):
      - reflex_force を超える接触力 → 反射的に離脱し reflex_hold の間は再接触しない
      - 指令が watchdog 秒途絶える → 離脱 (ファームウェアのウォッチドッグと同じ動作)
      - SAFE_STOP / IDLE → 離脱
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import mujoco
import numpy as np

from interfaces import ActuatorCommand

MODEL_PATH = Path(__file__).with_name("model") / "touch_care.xml"
LIFTED_MODES = frozenset({"IDLE", "SAFE_STOP"})


class Phase(StrEnum):
    HOVER = "HOVER"  # 腕の上で待機
    APPROACH = "APPROACH"  # 下降中
    CONTACT = "CONTACT"  # 接触中
    RETREAT = "RETREAT"  # 離脱中


@dataclass(slots=True)
class TouchCareConfig:
    # --- ストローク (world x [m]) ---
    stroke_start: float = 0.40  # 手首側 (position=0%)
    stroke_end: float = 0.60  # 肘側 (position=100%)
    hover_height: float = 0.26  # 待機高さ (パッド先端の world z [m])
    # --- 接触 ---
    contact_force: float = 1.5  # 目標押し付け力 [N]
    reflex_force: float = 3.5  # 反射離脱のしきい値 [N]
    force_limit: float = 5.0  # 安全上限 [N] (評価に使う。これを超える前に離れ始める)
    reflex_hold: float = 3.0  # 反射離脱後に再接触しない時間 [s]
    watchdog: float = 0.3  # 指令途絶で離脱するまでの時間 [s]
    # --- 速度 ---
    # 接触中の前腕方向の参照速度の上限 [m/s]。追従の行き過ぎを見込み、実速度が 10cm/s に収まる値にする
    max_contact_speed: float = 0.085
    max_hover_speed: float = 0.25  # 待機中の移動速度上限 [m/s]
    max_accel: float = 0.6  # 前腕方向の加速度上限 [m/s²]
    tracking_tau: float = 0.1  # 位置指令への追従時定数 [s]
    approach_speed: float = 0.10  # 下降速度 [m/s]
    touchdown_speed: float = 0.015  # 腕の直前からの下降速度 [m/s]
    surface_height: float = 0.125  # 腕の上面の高さの目安 [m] (ここの少し上で減速する)
    retreat_speed: float = 1.0  # 離脱速度 [m/s]
    # --- インピーダンス ---
    stiffness: tuple[float, float, float] = (900.0, 900.0, 600.0)  # [N/m]
    damping: tuple[float, float, float] = (100.0, 100.0, 90.0)  # [N·s/m]
    rot_stiffness: float = 60.0  # [N·m/rad]
    rot_damping: float = 8.0  # [N·m·s/rad]
    posture_stiffness: float = 8.0  # 冗長自由度を初期姿勢へ戻す剛性 [N·m/rad]
    posture_damping: float = 2.0
    force_gain: float = 0.03  # アドミタンスゲイン [(m/s)/N]
    contact_threshold: float = 0.15  # これを超えたら接触中とみなす [N]
    # --- 振動 ---
    vibration_force: float = 0.5  # PWM=255 時の力振幅 [N]
    vibration_freq: float = 40.0  # [Hz]
    # --- 相手役 ---
    elbow_speed: float = 3.0  # 腕を引くときの肘の角速度 [rad/s]


@dataclass(frozen=True, slots=True)
class TouchSample:
    """ある時刻のシミュレーション状態 (評価・動画の字幕に使う)。"""

    t: float
    force: float  # 接触力 [N]
    force_raw: float
    x: float  # パッド位置 world x [m]
    z: float  # パッド先端高さ [m]
    speed: float  # 前腕方向の速度 [m/s]
    phase: Phase
    mode: str
    vibration: int
    target_position: float


class TouchCareSim:
    """I/O を持たない物理シミュレーション本体。command() で指令を与え step() で時間を進める。"""

    def __init__(self, config: TouchCareConfig | None = None, model_path: Path = MODEL_PATH) -> None:
        self.cfg = config or TouchCareConfig()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._site = self.model.site("pad_site").id
        self._sensor = self.model.sensor("pad_force").adr[0]
        self._deflection = self.model.sensor("pad_deflection").adr[0]
        pad_joint = self.model.joint("pad_slide")
        self._pad_dof = int(pad_joint.dofadr[0])
        self._pad_stiffness = float(self.model.jnt_stiffness[pad_joint.id])
        self._pad_damping = float(self.model.dof_damping[self._pad_dof])
        self._elbow_act = self.model.actuator("elbow_servo").id
        self._pad_mat = self.model.material("pad_mat").id
        self._pad_rgba = self.model.mat_rgba[self._pad_mat].copy()
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self._kp = np.asarray(self.cfg.stiffness)
        self._kd = np.asarray(self.cfg.damping)
        self.reset()

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.model.key("home").qpos[:7]
        mujoco.mj_forward(self.model, self.data)
        self._q_home = self.data.qpos[:7].copy()
        self._r_target = self.data.site_xmat[self._site].reshape(3, 3).copy()
        self._ref = self.data.site_xpos[self._site].copy()
        self._ref_vel = np.zeros(3)
        self._vel_x = 0.0
        self._elbow = 0.0
        self._force_slow = 0.0
        self._force_fast = 0.0
        self._speed = 0.0
        self._cmd: ActuatorCommand | None = None
        self._cmd_time = -1e9
        self._reflex_until = -1e9
        self._phase = Phase.HOVER
        self.reflex_count = 0
        self.elbow_target = 0.0

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def phase(self) -> Phase:
        return self._phase

    def command(self, cmd: ActuatorCommand) -> None:
        self._cmd = cmd
        self._cmd_time = self.time

    def step(self, duration: float | None = None) -> None:
        """duration 秒 (省略時は 1 ステップ) 進める。"""
        steps = 1 if duration is None else max(1, round(duration / self.model.opt.timestep))
        for _ in range(steps):
            self._control()
            mujoco.mj_step(self.model, self.data)

    def sample(self) -> TouchSample:
        cmd = self._cmd
        return TouchSample(
            t=self.time,
            force=self._force_slow,
            force_raw=float(self.data.sensordata[self._sensor]),
            x=float(self.data.site_xpos[self._site][0]),
            z=float(self.data.site_xpos[self._site][2]),
            speed=self._speed,
            phase=self._phase,
            mode=cmd.mode if cmd else "IDLE",
            vibration=cmd.vibration_intensity if cmd else 0,
            target_position=cmd.linear_position if cmd else 0.0,
        )

    def sync_visuals(self) -> None:
        """描画前に呼ぶ。振動の強さに応じてパッドを発光させる。"""
        level = (self._cmd.vibration_intensity if self._cmd else 0) / 255.0
        if self._phase is not Phase.CONTACT:
            level = 0.0
        glow = np.array([1.0, 0.42, 0.42, 1.0])
        self.model.mat_rgba[self._pad_mat] = self._pad_rgba + (glow - self._pad_rgba) * level

    # ----------------------------------------------------------------- control
    def _position_to_x(self, position: float) -> float:
        c = self.cfg
        return c.stroke_start + (c.stroke_end - c.stroke_start) * position * 0.01

    def _want_contact(self, t: float) -> bool:
        cmd = self._cmd
        if cmd is None or cmd.mode in LIFTED_MODES:
            return False
        if t - self._cmd_time > self.cfg.watchdog:
            return False
        return t >= self._reflex_until

    def _update_reference(self, pos: np.ndarray, dt: float, t: float) -> None:
        c = self.cfg
        if self._force_fast > c.reflex_force and t >= self._reflex_until:
            self._reflex_until = t + c.reflex_hold
            self.reflex_count += 1

        want = self._want_contact(t)
        in_contact = self._force_slow > c.contact_threshold
        target_x = self._position_to_x(self._cmd.linear_position) if self._cmd else self._ref[0]
        prev_ref = self._ref.copy()

        # 接触中は腕に沿ってゆっくり、離れていれば素早く移動する。離脱中は水平に動かさない。
        # 速度だけでなく加速度も制限する (参照速度の段差は追従の行き過ぎになり速度上限を破る)
        limit = c.max_contact_speed if in_contact else c.max_hover_speed
        moving = (want or not in_contact) and (want or pos[2] > c.hover_height - 0.1)
        desired = np.clip((target_x - self._ref[0]) / c.tracking_tau, -limit, limit) if moving else 0.0
        self._vel_x += np.clip(desired - self._vel_x, -c.max_accel * dt, c.max_accel * dt)
        self._ref[0] += self._vel_x * dt
        self._ref[1] = 0.0

        if want and abs(target_x - self._ref[0]) < 0.02 or want and in_contact:
            if in_contact:
                dz = np.clip(c.force_gain * (self._force_slow - c.contact_force), -c.touchdown_speed, c.retreat_speed)
                self._phase = Phase.CONTACT
            else:
                near = pos[2] < c.surface_height + 0.025
                dz = -c.touchdown_speed if near else -c.approach_speed
                self._phase = Phase.APPROACH
            self._ref[2] += dz * dt
        else:
            step = c.retreat_speed * dt
            self._ref[2] += np.clip(c.hover_height - self._ref[2], -step, step)
            self._phase = Phase.RETREAT if pos[2] < c.hover_height - 0.02 else Phase.HOVER
        self._ref_vel = (self._ref - prev_ref) / dt

    def _update_elbow(self, dt: float) -> None:
        step = self.cfg.elbow_speed * dt
        self._elbow += float(np.clip(self.elbow_target - self._elbow, -step, step))
        self.data.ctrl[self._elbow_act] = self._elbow

    def _control(self) -> None:
        m, d, c = self.model, self.data, self.cfg
        dt = m.opt.timestep
        t = float(d.time)

        # 接触力はパッドのバネのたわみから測る (実機ならバネ + 変位センサで同じ構成にできる)
        # ばねだけだと腕が急に動いたとき実際の接触力より遅れるので、ダンパ分も足す
        deflection = float(d.sensordata[self._deflection])
        rate = float(d.qvel[self._pad_dof])
        force = max(0.0, -deflection * self._pad_stiffness - rate * self._pad_damping)
        self._force_fast += (force - self._force_fast) * dt / (0.005 + dt)
        self._force_slow += (force - self._force_slow) * dt / (0.03 + dt)

        pos = d.site_xpos[self._site]
        self._update_reference(pos, dt, t)
        self._update_elbow(dt)

        mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self._site)
        jac = np.vstack((self._jacp[:, :7], self._jacr[:, :7]))
        qvel = d.qvel[:7]
        twist = jac @ qvel
        self._speed = float(twist[0])

        # 減衰は参照軌道との相対速度に掛ける (絶対速度に掛けると離脱の動きそのものに抵抗してしまう)
        force_cmd = self._kp * (self._ref - pos) + self._kd * (self._ref_vel - twist[:3])
        if self._phase is Phase.CONTACT and self._cmd is not None:
            amp = c.vibration_force * self._cmd.vibration_intensity / 255.0
            force_cmd[2] += amp * np.sin(2 * np.pi * c.vibration_freq * t)

        quat_err = np.zeros(4)
        quat_cur = np.zeros(4)
        quat_ref = np.zeros(4)
        rot_err = np.zeros(3)
        mujoco.mju_mat2Quat(quat_cur, d.site_xmat[self._site])
        mujoco.mju_mat2Quat(quat_ref, self._r_target.ravel())
        mujoco.mju_negQuat(quat_cur, quat_cur)
        mujoco.mju_mulQuat(quat_err, quat_ref, quat_cur)
        mujoco.mju_quat2Vel(rot_err, quat_err, 1.0)
        torque_cmd = c.rot_stiffness * rot_err - c.rot_damping * twist[3:]

        wrench = np.concatenate((force_cmd, torque_cmd))
        tau = jac.T @ wrench

        # 冗長自由度は作業に影響しない零空間で初期姿勢へ戻す
        posture = c.posture_stiffness * (self._q_home - d.qpos[:7]) - c.posture_damping * qvel
        tau += posture - jac.T @ (np.linalg.pinv(jac.T) @ posture)

        tau += d.qfrc_bias[:7]  # 重力・コリオリ補償
        d.ctrl[:7] = np.clip(tau, m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1])
