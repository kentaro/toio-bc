from __future__ import annotations


class Mixer:
    def __init__(
        self,
        max_speed: int = 120,
        deadzone: float = 0.08,
        expo: float = 0.3,
        slew_rate: float = 300.0,
        rate_hz: float = 60.0,
        invert_x: bool = False,
        invert_y: bool = False,
    ) -> None:
        self.max_speed = float(max_speed)
        self.deadzone = float(deadzone)
        self.expo = float(expo)
        self.slew_rate = float(slew_rate)
        self.dt = 1.0 / float(rate_hz)
        self.invx = -1.0 if invert_x else 1.0
        self.invy = -1.0 if invert_y else 1.0
        self.prev_left = 0.0
        self.prev_right = 0.0

    def _shape(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        e = self.expo
        return (1.0 - e) * value + e * (value ** 3)

    def _slew(self, target: float, previous: float) -> float:
        max_step = self.slew_rate * self.dt
        return max(min(target, previous + max_step), previous - max_step)

    def mix(self, x: float, y: float) -> tuple[int, int]:
        x = max(-1.0, min(1.0, x * self.invx))
        y = max(-1.0, min(1.0, y * self.invy))

        x = self._shape(x)
        y = self._shape(y)

        left = y + x
        right = y - x

        magnitude = max(1.0, abs(left), abs(right))
        left /= magnitude
        right /= magnitude

        left *= self.max_speed
        right *= self.max_speed

        left = self._slew(left, self.prev_left)
        right = self._slew(right, self.prev_right)

        self.prev_left = left
        self.prev_right = right

        return int(round(left)), int(round(right))

    def reset(self) -> None:
        self.prev_left = 0.0
        self.prev_right = 0.0
