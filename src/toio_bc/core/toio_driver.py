from __future__ import annotations

import asyncio
import contextlib
import inspect
from dataclasses import dataclass
from typing import Any, Optional

from bleak import BleakClient, BleakScanner

LOG_PREFIX = "[toio]"

# Core Cube BLE UUIDs
DEFAULT_SERVICE_UUID = "10b20100-5b3b-4571-9508-cf3efcd7bbae"
DEFAULT_MOTOR_CHAR_UUID = "10b20102-5b3b-4571-9508-cf3efcd7bbae"
DEFAULT_SENSOR_CHAR_UUID = "10b20106-5b3b-4571-9508-cf3efcd7bbae"
DEFAULT_CONFIG_CHAR_UUID = "10b201ff-5b3b-4571-9508-cf3efcd7bbae"


@dataclass
class ToioDriverConfig:
    mac_address: Optional[str] = None
    name_prefix: str = "toio Core Cube"
    scan_timeout_sec: float = 10.0
    scan_retry: int = 3
    collision_threshold: int = 3  # 1-10: lower=more sensitive, higher=less sensitive
    service_uuid: str = DEFAULT_SERVICE_UUID
    motor_characteristic_uuid: str = DEFAULT_MOTOR_CHAR_UUID
    sensor_characteristic_uuid: str = DEFAULT_SENSOR_CHAR_UUID
    config_characteristic_uuid: str = DEFAULT_CONFIG_CHAR_UUID


class ToioDriver:
    """BLE driver that speaks directly to a toio Core Cube."""

    def __init__(self, config: ToioDriverConfig):
        self.cfg = config
        self.mac: Optional[str] = config.mac_address.upper() if config.mac_address else None
        self._client: Optional[BleakClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._collision_event = asyncio.Event()

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()

        device_address = self.mac or await self._discover_cube()
        if device_address is None:
            raise RuntimeError(
                "Unable to locate a toio Core Cube. Make sure the cube is on and advertising."
            )

        client = BleakClient(device_address)
        print(f"{LOG_PREFIX} Connecting to cube {device_address}...")
        await client.connect()
        print(f"{LOG_PREFIX} Connected")
        await asyncio.sleep(0.1)

        await self._enable_motion_detection(client)
        await client.start_notify(self.cfg.sensor_characteristic_uuid, self._sensor_callback)
        print(f"{LOG_PREFIX} Sensor notifications enabled")

        self._client = client

    async def move(self, left: int, right: int, duration_ms: int = 100) -> None:
        """Send motor control command to toio cube."""
        client = self._ensure_client()
        payload = self._build_motor_payload(left, right, duration_ms)
        await client.write_gatt_char(self.cfg.motor_characteristic_uuid, payload, response=False)

    async def stop(self) -> None:
        await self.move(0, 0, 100)

    def consume_collision(self) -> bool:
        if self._collision_event.is_set():
            self._collision_event.clear()
            return True
        return False

    async def wait_for_collision(self, timeout: Optional[float] = None) -> bool:
        try:
            await asyncio.wait_for(self._collision_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        else:
            self._collision_event.clear()
            return True

    async def close(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            print(f"{LOG_PREFIX} Disconnecting cube")
            with contextlib.suppress(Exception):
                await client.stop_notify(self.cfg.sensor_characteristic_uuid)
            await client.disconnect()
        finally:
            self._client = None

    async def _discover_cube(self) -> Optional[str]:
        service_uuid = self.cfg.service_uuid.lower()
        name_prefix = self.cfg.name_prefix

        def _filter(device: Any, adv: Any) -> bool:
            name = (getattr(device, "name", "") or "").strip()
            if name_prefix and name.startswith(name_prefix):
                return True
            uuids = {(uuid or "").lower() for uuid in getattr(adv, "service_uuids", []) or []}
            return service_uuid in uuids

        for attempt in range(max(1, int(self.cfg.scan_retry))):
            print(f"{LOG_PREFIX} Scanning for cube (attempt {attempt + 1})")
            device = await BleakScanner.find_device_by_filter(
                filterfunc=_filter, timeout=self.cfg.scan_timeout_sec
            )
            if device and getattr(device, "address", None):
                print(f"{LOG_PREFIX} Found device {device.address}")
                return str(device.address).upper()
        return None

    async def _enable_motion_detection(self, client: BleakClient) -> None:
        # Set collision detection threshold
        # https://toio.github.io/toio-spec/docs/ble_configuration
        # Format: [0x06, 0x00, threshold]
        # Threshold: 1-10 (default 7, lower = more sensitive, higher = less sensitive)
        threshold = max(1, min(10, self.cfg.collision_threshold))  # Clamp to valid range
        collision_threshold_cmd = bytearray([0x06, 0x00, threshold])

        try:
            await client.write_gatt_char(self.cfg.config_characteristic_uuid, collision_threshold_cmd, response=True)
            print(f"{LOG_PREFIX} Collision detection threshold set to level {threshold}")
        except Exception as e:
            print(f"{LOG_PREFIX} Warning: Failed to set collision threshold: {e}")

    def _sensor_callback(self, _: int, data: bytearray) -> None:
        if not data:
            return
        # Motion sensor data format (6 bytes):
        # [0]: 0x01 (detection type)
        # [1]: horizontal detection (0x00/0x01)
        # [2]: collision detection (0x00/0x01) <-- THIS IS WHAT WE NEED
        # [3]: double tap detection (0x00/0x01)
        # [4]: posture detection (1-6)
        # [5]: shake detection (0x00-0x0a)
        # https://toio.github.io/toio-spec/docs/ble_sensor

        if len(data) >= 3 and data[0] == 0x01:
            # Only check collision detection (data[2])
            # data[1] (horizontal) detects tilt/movement, not actual collisions
            collision_detected = data[2] == 0x01

            if collision_detected:
                print(f"{LOG_PREFIX} [SENSOR] Collision detected! data={list(data)}")
                if self._loop:
                    self._loop.call_soon_threadsafe(self._collision_event.set)

    def _build_motor_payload(self, left: int, right: int, duration_ms: int) -> bytearray:
        """
        Build motor control payload according to toio Core Cube specification.

        Official spec: https://toio.github.io/toio-spec/en/docs/ble_motor/
        Command: Motor control with specified duration (0x02)

        Byte format:
        [0]: Control type (0x02 = motor control with time)
        [1]: Left motor ID (0x01)
        [2]: Left motor direction (0x01 = forward, 0x02 = backward)
        [3]: Left motor speed (0-100)
        [4]: Right motor ID (0x02)
        [5]: Right motor direction (0x01 = forward, 0x02 = backward)
        [6]: Right motor speed (0-100)
        [7]: Duration (0-255 in 10ms units, 0 = continuous)
        """
        # Encode direction and speed for each motor
        # Direction: 1 = forward (positive value), 2 = backward (negative value)
        # Speed: 0-100 (toio accepts 0-255 but we limit to 100 for safety)
        left_dir = 0x01 if left >= 0 else 0x02
        left_speed = max(0, min(100, abs(left)))

        right_dir = 0x01 if right >= 0 else 0x02
        right_speed = max(0, min(100, abs(right)))

        # Convert duration from milliseconds to 10ms units
        # Duration range: 0-255 (0 = no time limit, 1-255 = value * 10ms)
        duration_10ms = max(0, min(255, int(round(duration_ms / 10.0))))

        return bytearray([
            0x02,          # Control type: motor control with specified duration
            0x01,          # Left motor ID
            left_dir,      # Left motor direction
            left_speed,    # Left motor speed (0-100)
            0x02,          # Right motor ID
            right_dir,     # Right motor direction
            right_speed,   # Right motor speed (0-100)
            duration_10ms, # Duration in 10ms units
        ])

    def _ensure_client(self) -> BleakClient:
        if self._client is None:
            raise RuntimeError("Driver not connected")
        return self._client

    async def _maybe_await(self, result: Any) -> None:
        if inspect.isawaitable(result):
            await result
