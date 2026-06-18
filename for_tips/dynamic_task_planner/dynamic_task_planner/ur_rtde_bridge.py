from dataclasses import dataclass
from typing import Optional
import numpy as np

from .ur_rtde_bridge import RTDESettings  # if split, otherwise remove this import

@dataclass
class RTDESettings:
    robot_ip: str
    use_mock: bool
    command_mode: str
    servo_dt_sec: float
    lookahead_time_sec: float
    gain: int
    speed: float
    accel: float
    poll_dt_sec: float
    require_robot_ready: bool
    require_normal_safety: bool
    allowed_safety_status: List[int]
    planner_timeout_sec: float
    scene_timeout_sec: float
    rtde_state_timeout_sec: float
    stop_on_watchdog_trip: bool
    use_servo_stop: bool
    use_speed_stop: bool
    hold_position_after_stop: bool

    @classmethod
    def from_yaml(cls, cfg: Dict[str, Any]) -> "RTDESettings":
        r = cfg["rtde"]
        control = r["control"]
        receive = r["receive"]
        watchdogs = r["watchdogs"]
        stop_behavior = r["stop_behavior"]
        return cls(
            robot_ip=r["robot_ip"],
            use_mock=bool(r["use_mock"]),
            command_mode=control["command_mode"],
            servo_dt_sec=float(control["servo_dt_sec"]),
            lookahead_time_sec=float(control["lookahead_time_sec"]),
            gain=int(control["gain"]),
            speed=float(control["speed"]),
            accel=float(control["accel"]),
            poll_dt_sec=float(receive["poll_dt_sec"]),
            require_robot_ready=bool(receive["require_robot_ready"]),
            require_normal_safety=bool(receive["require_normal_safety"]),
            allowed_safety_status=list(receive["allowed_safety_status"]),
            planner_timeout_sec=float(watchdogs["planner_timeout_sec"]),
            scene_timeout_sec=float(watchdogs["scene_timeout_sec"]),
            rtde_state_timeout_sec=float(watchdogs["rtde_state_timeout_sec"]),
            stop_on_watchdog_trip=bool(watchdogs["stop_on_watchdog_trip"]),
            use_servo_stop=bool(stop_behavior["use_servo_stop"]),
            use_speed_stop=bool(stop_behavior["use_speed_stop"]),
            hold_position_after_stop=bool(stop_behavior["hold_position_after_stop"]),
        )

@dataclass
class RobotState:
    actual_q: Optional[np.ndarray] = None
    actual_tcp_pose: Optional[np.ndarray] = None
    robot_mode: Optional[int] = None
    safety_status: Optional[int] = None
    speed_scaling: Optional[float] = None
    runtime_state: Optional[int] = None
    is_ready: bool = False


class URRTDEBridge:
    def __init__(self, settings: RTDESettings):
        self.settings = settings
        self.robot_ip = settings.robot_ip
        self.use_mock = settings.use_mock
        self.rtde_c = None
        self.rtde_r = None
        self.state = RobotState()

    @classmethod
    def from_yaml(cls, cfg):
        return cls(RTDESettings.from_yaml(cfg))

    def connect(self):
        if self.use_mock:
            return
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
        self.rtde_c = RTDEControlInterface(self.robot_ip)
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)

    def read_state(self) -> RobotState:
        if self.use_mock:
            self.state.is_ready = True
            return self.state

        self.state.actual_q = np.array(self.rtde_r.getActualQ(), dtype=float)
        self.state.actual_tcp_pose = np.array(self.rtde_r.getActualTCPPose(), dtype=float)
        self.state.robot_mode = self.rtde_r.getRobotMode()
        self.state.safety_status = self.rtde_r.getSafetyStatus()
        self.state.speed_scaling = self.rtde_r.getSpeedScaling()
        self.state.runtime_state = self.rtde_r.getRuntimeState()
        self.state.is_ready = True
        return self.state

    def servo_joint(self, q_cmd: np.ndarray):
        if self.use_mock:
            return
        self.rtde_c.servoJ(
            q_cmd.tolist(),
            self.settings.speed,
            self.settings.accel,
            self.settings.servo_dt_sec,
            self.settings.lookahead_time_sec,
            self.settings.gain,
        )

    def stop_motion(self):
        if self.use_mock:
            return
        if self.settings.use_speed_stop:
            self.rtde_c.speedStop()
        if self.settings.use_servo_stop:
            self.rtde_c.servoStop()

    def disconnect(self):
        if self.use_mock:
            return
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()