"""Central low-force configuration for LEAP Hand controllers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyProfile:
    """Validated parameters shared by every safe-drive consumer."""

    kp: int = 300
    ki: int = 0
    kd: int = 100
    goal_current: int = 150
    max_speed_rad_s: float = 1.0
    startup_seconds: float = 2.0
    shutdown_seconds: float = 2.0
    tracking_loss_seconds: float = 0.5

    def __post_init__(self):
        if min(self.kp, self.ki, self.kd) < 0:
            raise ValueError("PID gains cannot be negative")
        if self.goal_current <= 0:
            raise ValueError("goal_current must be positive")
        if self.max_speed_rad_s <= 0:
            raise ValueError("max_speed_rad_s must be positive")
        if min(self.startup_seconds, self.shutdown_seconds) <= 0:
            raise ValueError("startup and shutdown durations must be positive")
        if self.tracking_loss_seconds < 0:
            raise ValueError("tracking_loss_seconds cannot be negative")


SAFE_PROFILE = SafetyProfile()
