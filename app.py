
from __future__ import annotations

import asyncio
import csv
import json
import os
import sqlite3
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from bleak import BleakClient, BleakScanner

try:
    from fit_tool.fit_file_builder import FitFileBuilder
    from fit_tool.profile.messages.activity_message import ActivityMessage
    from fit_tool.profile.messages.event_message import EventMessage
    from fit_tool.profile.messages.file_id_message import FileIdMessage
    from fit_tool.profile.messages.lap_message import LapMessage
    from fit_tool.profile.messages.record_message import RecordMessage
    from fit_tool.profile.messages.session_message import SessionMessage
    from fit_tool.profile.profile_type import Event, EventType, FileType, Manufacturer, Sport, SubSport

    FIT_EXPORT_AVAILABLE = True
except Exception:
    FIT_EXPORT_AVAILABLE = False

FTMS_SERVICE_UUID = "00001826-0000-1000-8000-00805f9b34fb"
FTMS_CONTROL_POINT_UUID = "00002ad9-0000-1000-8000-00805f9b34fb"
FTMS_INDOOR_BIKE_DATA_UUID = "00002ad2-0000-1000-8000-00805f9b34fb"
FTMS_STATUS_UUID = "00002ada-0000-1000-8000-00805f9b34fb"

KICKR_FILTER = "kickr core"
MIN_TARGET_WATTS = 80
SAFE_PAUSE_WATTS = 100

SCAN_TIMEOUT_SECONDS = 6.0
RECONNECT_INTERVAL_SECONDS = 2.0
RECONNECT_WINDOW_SECONDS = 20.0
CONTROL_LOOP_SECONDS = 0.5
KEEPALIVE_SECONDS = 5.0
SAMPLE_SECONDS = 1.0
DISPLAY_POWER_AVG_SECONDS = 3
CADENCE_STALL_SECONDS = 5.0
CONTROL_ERROR_LIMIT = 3


class EngineState(str, Enum):
    IDLE = "IDLE"
    READY = "READY"
    COUNTDOWN = "COUNTDOWN"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STEP_TRANSITION = "STEP_TRANSITION"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    ERROR = "ERROR"


ALLOWED_TRANSITIONS: dict[EngineState, set[EngineState]] = {
    EngineState.IDLE: {EngineState.READY, EngineState.ERROR},
    EngineState.READY: {EngineState.COUNTDOWN, EngineState.IDLE, EngineState.ERROR, EngineState.ABORTED},
    EngineState.COUNTDOWN: {EngineState.RUNNING, EngineState.PAUSED, EngineState.ABORTED, EngineState.ERROR},
    EngineState.RUNNING: {EngineState.PAUSED, EngineState.STEP_TRANSITION, EngineState.COMPLETED, EngineState.ABORTED, EngineState.ERROR},
    EngineState.PAUSED: {EngineState.RUNNING, EngineState.STEP_TRANSITION, EngineState.COMPLETED, EngineState.ABORTED, EngineState.ERROR},
    EngineState.STEP_TRANSITION: {EngineState.RUNNING, EngineState.PAUSED, EngineState.COMPLETED, EngineState.ERROR},
    EngineState.COMPLETED: {EngineState.READY, EngineState.IDLE},
    EngineState.ABORTED: {EngineState.READY, EngineState.IDLE},
    EngineState.ERROR: {EngineState.IDLE, EngineState.READY},
}


@dataclass(frozen=True)
class WorkoutStep:
    type: str
    duration_s: int
    target_pct_ftp: float


@dataclass(frozen=True)
class Workout:
    id: str
    name: str
    ftp: int
    steps: list[WorkoutStep]

    @property
    def total_duration_s(self) -> int:
        return sum(step.duration_s for step in self.steps)

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "name": self.name,
                "ftp": self.ftp,
                "steps": [asdict(s) for s in self.steps],
            }
        )


@dataclass(frozen=True)
class DeviceChoice:
    name: str
    address: str
    rssi: int


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fmt_clock(seconds: float) -> str:
    seconds_i = int(max(seconds, 0.0))
    mins, secs = divmod(seconds_i, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def slugify(text: str) -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw or f"workout_{uuid.uuid4().hex[:8]}"


def parse_workout_dict(payload: dict[str, Any], fallback_name: str = "Imported Workout") -> Workout:
    workout_id = str(payload.get("id") or slugify(str(payload.get("name") or fallback_name)))
    name = str(payload.get("name") or fallback_name).strip() or fallback_name
    ftp = int(payload.get("ftp") or 250)
    ftp = int(clamp(float(ftp), 80.0, 600.0))

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Workout must contain a non-empty 'steps' list.")

    steps: list[WorkoutStep] = []
    for index, row in enumerate(raw_steps, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Step {index} is invalid.")
        step_type = str(row.get("type") or "steady")
        duration_s = int(row.get("duration_s") or 0)
        target_pct_ftp = float(row.get("target_pct_ftp") or 0.0)
        if duration_s <= 0:
            raise ValueError(f"Step {index} duration must be > 0.")
        if target_pct_ftp <= 0:
            raise ValueError(f"Step {index} target_pct_ftp must be > 0.")
        steps.append(WorkoutStep(type=step_type, duration_s=duration_s, target_pct_ftp=target_pct_ftp))

    return Workout(id=workout_id, name=name, ftp=ftp, steps=steps)


def load_json_workout(path: Path) -> Workout:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Workout JSON root must be an object.")
    return parse_workout_dict(payload, fallback_name=path.stem)


def _duration_chunks(total_s: int, chunk_s: int) -> list[int]:
    remaining = max(total_s, 1)
    out: list[int] = []
    while remaining > chunk_s:
        out.append(chunk_s)
        remaining -= chunk_s
    out.append(remaining)
    return out


def load_zwo_workout(path: Path) -> Workout:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    name = root.findtext("name") or root.findtext("workout_file/name") or path.stem
    workout_node = root.find("workout")
    if workout_node is None:
        wf = root.find("workout_file")
        if wf is not None:
            workout_node = wf.find("workout")
    if workout_node is None:
        raise ValueError("ZWO file is missing a <workout> section.")

    steps: list[WorkoutStep] = []

    def read_float(node: ET.Element, attr: str, fallback: float) -> float:
        raw = node.attrib.get(attr) or node.attrib.get(attr.lower()) or node.attrib.get(attr.upper())
        return float(raw) if raw is not None else fallback

    def read_int(node: ET.Element, attr: str, fallback: int) -> int:
        raw = node.attrib.get(attr) or node.attrib.get(attr.lower()) or node.attrib.get(attr.upper())
        return int(float(raw)) if raw is not None else fallback

    for child in workout_node:
        tag = child.tag.lower()
        if tag in ("warmup", "cooldown", "ramp"):
            duration_s = read_int(child, "Duration", 0)
            if duration_s <= 0:
                continue
            p_low = read_float(child, "PowerLow", read_float(child, "Power", 0.5))
            p_high = read_float(child, "PowerHigh", p_low)
            chunks = _duration_chunks(duration_s, 30)
            progressed = 0
            for seg in chunks:
                frac = progressed / max(duration_s, 1)
                pct = p_low + ((p_high - p_low) * frac)
                steps.append(WorkoutStep(type=tag, duration_s=seg, target_pct_ftp=float(pct)))
                progressed += seg
        elif tag in ("steadystate", "steadystate"):
            duration_s = read_int(child, "Duration", 0)
            power = read_float(child, "Power", 0.7)
            if duration_s > 0:
                steps.append(WorkoutStep(type="steady", duration_s=duration_s, target_pct_ftp=power))
        elif tag == "intervalst":
            repeat = read_int(child, "Repeat", 0)
            on_dur = read_int(child, "OnDuration", 0)
            off_dur = read_int(child, "OffDuration", 0)
            on_power = read_float(child, "OnPower", 1.0)
            off_power = read_float(child, "OffPower", 0.6)
            for _ in range(max(repeat, 0)):
                if on_dur > 0:
                    steps.append(WorkoutStep(type="interval_on", duration_s=on_dur, target_pct_ftp=on_power))
                if off_dur > 0:
                    steps.append(WorkoutStep(type="interval_off", duration_s=off_dur, target_pct_ftp=off_power))
        else:
            duration_s = read_int(child, "Duration", 0)
            power = read_float(child, "Power", 0.7)
            if duration_s > 0:
                steps.append(WorkoutStep(type=tag, duration_s=duration_s, target_pct_ftp=power))

    if not steps:
        raise ValueError("ZWO import produced no usable workout steps.")

    payload = {
        "id": slugify(path.stem),
        "name": str(name).strip() or path.stem,
        "ftp": 250,
        "steps": [asdict(s) for s in steps],
    }
    return parse_workout_dict(payload, fallback_name=path.stem)


def default_workouts() -> dict[str, Workout]:
    raw = [
        {
            "id": "tempo_builder_28",
            "name": "Tempo Builder 28m",
            "ftp": 250,
            "steps": [
                {"type": "warmup", "duration_s": 300, "target_pct_ftp": 0.55},
                {"type": "steady", "duration_s": 480, "target_pct_ftp": 0.75},
                {"type": "steady", "duration_s": 120, "target_pct_ftp": 0.60},
                {"type": "steady", "duration_s": 480, "target_pct_ftp": 0.80},
                {"type": "cooldown", "duration_s": 300, "target_pct_ftp": 0.50},
            ],
        },
        {
            "id": "vo2_6x2",
            "name": "VO2 6x2",
            "ftp": 250,
            "steps": (
                [{"type": "warmup", "duration_s": 480, "target_pct_ftp": 0.55}]
                + [
                    {"type": "vo2_on", "duration_s": 120, "target_pct_ftp": 1.15},
                    {"type": "vo2_off", "duration_s": 120, "target_pct_ftp": 0.55},
                ]
                * 6
                + [{"type": "cooldown", "duration_s": 360, "target_pct_ftp": 0.50}]
            ),
        },
    ]
    out: dict[str, Workout] = {}
    for item in raw:
        workout = parse_workout_dict(item)
        out[workout.id] = workout
    return out

class LocalStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS workouts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                json_blob TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                workout_id TEXT NOT NULL,
                start_ts TEXT NOT NULL,
                end_ts TEXT NOT NULL,
                ftp INTEGER NOT NULL,
                status TEXT NOT NULL,
                summary_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                t_ms INTEGER NOT NULL,
                power REAL,
                cadence REAL,
                speed REAL,
                target_power INTEGER,
                step_index INTEGER
            );
            """
        )
        self.conn.commit()

    def save_workout(self, workout: Workout) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO workouts (id, name, json_blob) VALUES (?, ?, ?)",
            (workout.id, workout.name, workout.to_json()),
        )
        self.conn.commit()

    def list_workouts(self) -> dict[str, Workout]:
        out: dict[str, Workout] = {}
        rows = self.conn.execute("SELECT json_blob FROM workouts").fetchall()
        for row in rows:
            data = json.loads(row["json_blob"])
            workout = parse_workout_dict(data)
            out[workout.id] = workout
        return out

    def save_session(
        self,
        session_id: str,
        workout_id: str,
        start_ts: str,
        end_ts: str,
        ftp: int,
        status: str,
        summary: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (id, workout_id, start_ts, end_ts, ftp, status, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, workout_id, start_ts, end_ts, ftp, status, json.dumps(summary)),
            )
            self.conn.execute("DELETE FROM samples WHERE session_id = ?", (session_id,))
            self.conn.executemany(
                """
                INSERT INTO samples
                (session_id, t_ms, power, cadence, speed, target_power, step_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        int(s["t_ms"]),
                        float(s.get("power", 0.0)),
                        s.get("cadence"),
                        s.get("speed"),
                        int(s.get("target_power", 0)),
                        int(s.get("step_index", 0)),
                    )
                    for s in samples
                ],
            )

    def close(self) -> None:
        self.conn.close()


class FTMSBridge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "connected": False,
            "trainer_name": "-",
            "address": None,
            "rssi": None,
            "status": "Disconnected",
            "last_error": None,
            "power_w": None,
            "cadence_rpm": None,
            "speed_kph": None,
            "control_status": "-",
            "machine_status": "-",
        }
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

        self._client: BleakClient | None = None
        self._cp_uuid: str | None = None
        self._ibd_uuid: str | None = None
        self._status_uuid: str | None = None
        self._cp_lock: asyncio.Lock | None = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._cp_lock = asyncio.Lock()
        self._ready.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()

    def _set_state(self, **updates: Any) -> None:
        with self._lock:
            self._state.update(updates)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _submit(self, coro: Any):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def scan_kickr(self, timeout: float = SCAN_TIMEOUT_SECONDS):
        return self._submit(self._scan_kickr(timeout))

    def connect(self, address: str):
        return self._submit(self._connect(address))

    def disconnect(self):
        return self._submit(self._disconnect())

    def request_control(self):
        return self._submit(self._request_control())

    def start_resume(self):
        return self._submit(self._start_resume())

    def pause(self):
        return self._submit(self._pause())

    def reset(self):
        return self._submit(self._reset())

    def set_target_power(self, watts: int):
        return self._submit(self._set_target_power(watts))

    def shutdown(self) -> None:
        try:
            self.disconnect().result(timeout=8.0)
        except Exception:
            pass
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    async def _scan_kickr(self, timeout: float) -> list[DeviceChoice]:
        self._set_state(status=f"Scanning BLE for KICKR CORE ({int(timeout)}s)...")
        found = await BleakScanner.discover(timeout=timeout, return_adv=True)
        entries = found.values() if isinstance(found, dict) else [(d, None) for d in found]
        devices: list[DeviceChoice] = []
        for device, adv in entries:
            name = (getattr(adv, "local_name", "") or getattr(device, "name", "") or "Unknown Device").strip()
            if KICKR_FILTER not in name.lower():
                continue
            addr = getattr(device, "address", "")
            rssi = int(getattr(adv, "rssi", -999) or -999)
            devices.append(DeviceChoice(name=name, address=addr, rssi=rssi))
        devices.sort(key=lambda d: (-d.rssi, d.name))
        self._set_state(status=f"Scan complete: {len(devices)} KICKR CORE found")
        return devices

    async def _connect(self, address: str) -> None:
        await self._disconnect()
        self._set_state(status="Connecting trainer...", last_error=None)
        try:
            device = await BleakScanner.find_device_by_address(address, timeout=12.0)
            if device is None:
                raise RuntimeError("Trainer not found.")

            client = BleakClient(device, disconnected_callback=self._on_disconnected)
            await client.connect()
            await client.get_services()

            cp = client.services.get_characteristic(FTMS_CONTROL_POINT_UUID)
            ibd = client.services.get_characteristic(FTMS_INDOOR_BIKE_DATA_UUID)
            status = client.services.get_characteristic(FTMS_STATUS_UUID)
            if cp is None or ibd is None:
                raise RuntimeError("FTMS control/data characteristics missing.")

            self._client = client
            self._cp_uuid = cp.uuid
            self._ibd_uuid = ibd.uuid
            self._status_uuid = status.uuid if status else None

            if "notify" in cp.properties or "indicate" in cp.properties:
                await client.start_notify(cp.uuid, self._on_control_notify)
            if "notify" in ibd.properties:
                await client.start_notify(ibd.uuid, self._on_indoor_bike_data)
            if status is not None and "notify" in status.properties:
                await client.start_notify(status.uuid, self._on_status_notify)

            name = getattr(device, "name", None) or address
            self._set_state(
                connected=True,
                trainer_name=name,
                address=address,
                status=f"Connected: {name}",
                last_error=None,
            )
            await self._request_control()
            await asyncio.sleep(0.1)
            await self._start_resume()
        except Exception as exc:
            await self._disconnect()
            self._set_state(last_error=str(exc), status=f"Connect failed: {exc}")
            raise

    async def _disconnect(self) -> None:
        client = self._client
        cp = self._cp_uuid
        ibd = self._ibd_uuid
        status = self._status_uuid
        self._client = None
        self._cp_uuid = None
        self._ibd_uuid = None
        self._status_uuid = None
        if client is not None:
            for uuid_value in (cp, ibd, status):
                if not uuid_value:
                    continue
                try:
                    await client.stop_notify(uuid_value)
                except Exception:
                    pass
            try:
                await client.disconnect()
            except Exception:
                pass
        self._set_state(
            connected=False,
            trainer_name="-",
            address=None,
            status="Disconnected",
            power_w=None,
            cadence_rpm=None,
            speed_kph=None,
            control_status="-",
            machine_status="-",
        )

    async def _request_control(self) -> None:
        await self._write_cp(bytes([0x00]))

    async def _start_resume(self) -> None:
        await self._write_cp(bytes([0x07]))

    async def _pause(self) -> None:
        try:
            await self._write_cp(bytes([0x08, 0x02]))
        except Exception:
            await self._write_cp(bytes([0x08]))

    async def _reset(self) -> None:
        await self._write_cp(bytes([0x01]))

    async def _set_target_power(self, watts: int) -> None:
        target = int(clamp(float(watts), 30.0, 2000.0))
        payload = bytes([0x05, target & 0xFF, (target >> 8) & 0xFF])
        await self._write_cp(payload)

    async def _write_cp(self, payload: bytes) -> None:
        if self._client is None or self._cp_uuid is None or self._cp_lock is None:
            raise RuntimeError("Trainer not connected.")
        async with self._cp_lock:
            try:
                await self._client.write_gatt_char(self._cp_uuid, payload, response=True)
            except Exception:
                await self._client.write_gatt_char(self._cp_uuid, payload, response=False)

    def _on_disconnected(self, _client: BleakClient) -> None:
        self._set_state(
            connected=False,
            status="Connection lost",
            power_w=None,
            cadence_rpm=None,
            speed_kph=None,
        )

    def _on_control_notify(self, _sender: Any, data: bytearray) -> None:
        if len(data) >= 3 and data[0] == 0x80:
            result = data[2]
            self._set_state(control_status=f"CP 0x{data[1]:02X} -> {result}")

    def _on_status_notify(self, _sender: Any, data: bytearray) -> None:
        if data:
            self._set_state(machine_status=f"0x{data[0]:02X}")

    def _on_indoor_bike_data(self, _sender: Any, data: bytearray) -> None:
        if len(data) < 4:
            return
        flags = int.from_bytes(data[0:2], byteorder="little", signed=False)
        idx = 2
        speed = None
        cadence = None
        power = None

        if not (flags & 0x0001) and len(data) >= idx + 2:
            speed_raw = int.from_bytes(data[idx : idx + 2], byteorder="little", signed=False)
            speed = speed_raw / 100.0
            idx += 2
        if flags & 0x0002:
            idx += 2
        if flags & 0x0004 and len(data) >= idx + 2:
            cadence_raw = int.from_bytes(data[idx : idx + 2], byteorder="little", signed=False)
            cadence = cadence_raw / 2.0
            idx += 2
        if flags & 0x0008:
            idx += 2
        if flags & 0x0010:
            idx += 3
        if flags & 0x0020:
            idx += 2
        if flags & 0x0040 and len(data) >= idx + 2:
            power = int.from_bytes(data[idx : idx + 2], byteorder="little", signed=True)

        updates: dict[str, Any] = {}
        if speed is not None:
            updates["speed_kph"] = speed
        if cadence is not None:
            updates["cadence_rpm"] = cadence
        if power is not None:
            updates["power_w"] = power
        if updates:
            self._set_state(**updates)

class LocalWattApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Local Watt v1")
        self.geometry("1320x860")
        self.minsize(1160, 760)
        self.configure(bg="#0b1320")

        self.data_dir = self._resolve_data_dir()
        self.exports_dir = self.data_dir / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        self.store = LocalStore(self.data_dir / "localwatt.db")
        self.bridge = FTMSBridge()

        self.workouts = default_workouts()
        stored = self.store.list_workouts()
        self.workouts.update(stored)
        for workout in self.workouts.values():
            self.store.save_workout(workout)

        self.device_choices: list[DeviceChoice] = []
        self.selected_device_index: int | None = None
        self.connected_address: str | None = None

        self.state = EngineState.IDLE
        self.state_message = "Connect KICKR CORE and load a workout."
        self.last_connected_flag = False

        self.pending_scan = None
        self.pending_connect = None
        self.pending_disconnect = None

        self.reconnect_until = 0.0
        self.next_reconnect_attempt = 0.0
        self.auto_paused_for_disconnect = False
        self.user_disconnect_requested = False

        self.selected_workout_id = next(iter(self.workouts.keys()), None)
        self.step_index = 0
        self.step_elapsed_s = 0.0
        self.workout_elapsed_s = 0.0
        self.countdown_s = 3.0

        self.intensity_bias = 1.0
        self.target_dirty = True
        self.last_target_sent = None
        self.last_target_sent_ts = 0.0
        self.pending_target_write = False
        self.control_failures = 0
        self.last_control_eval_ts = 0.0
        self.ramp: tuple[float, float, float, float] | None = None

        self.zero_cadence_since: float | None = None
        self.cadence_relief_active = False

        self.session_id: str | None = None
        self.session_start_iso: str | None = None
        self.session_start_perf = 0.0
        self.samples: list[dict[str, Any]] = []
        self.next_sample_ts = 0.0

        self.display_power_window: deque[float] = deque(maxlen=DISPLAY_POWER_AVG_SECONDS)
        self.chart_power: deque[float] = deque(maxlen=300)
        self.chart_target: deque[float] = deque(maxlen=300)
        self.chart_last_draw_ts = 0.0

        self.latest_summary: dict[str, Any] | None = None
        self.latest_status: str | None = None

        self.last_tick = time.perf_counter()

        self._build_ui()
        self._refresh_workout_selector()
        self._refresh_workout_details()
        self._refresh_ui()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._tick)

    def _resolve_data_dir(self) -> Path:
        if os.name == "nt":
            base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            out = base / "LocalWatt"
        else:
            out = Path.home() / ".local" / "share" / "localwatt"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#0b1320")
        style.configure("TLabelframe", background="#132033", foreground="#dfe8f3")
        style.configure("TLabelframe.Label", background="#132033", foreground="#eaf2ff")
        style.configure("TLabel", background="#0b1320", foreground="#e4eefc")
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=6)
        style.configure("TCombobox", padding=4)
        style.configure("Horizontal.TProgressbar", troughcolor="#1f2e44", bordercolor="#1f2e44", background="#3ecb79")

        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        main = ttk.Frame(root)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        self.badge_state = tk.Label(
            left,
            text="IDLE",
            bg="#374151",
            fg="#f8fafc",
            font=("Segoe UI", 12, "bold"),
            padx=12,
            pady=6,
            width=20,
        )
        self.badge_state.pack(fill="x", pady=(0, 8))

        device_box = ttk.LabelFrame(left, text="Device", padding=8)
        device_box.pack(fill="x", pady=(0, 8))
        ttk.Button(device_box, text="Scan BLE", command=self._scan_devices).pack(fill="x", pady=(0, 4))
        self.device_list = tk.Listbox(
            device_box,
            height=5,
            bg="#0f1725",
            fg="#e7f0ff",
            selectbackground="#3b82f6",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 10),
        )
        self.device_list.pack(fill="x", pady=(0, 4))
        self.device_list.bind("<<ListboxSelect>>", self._on_device_selected)
        btn_row = ttk.Frame(device_box)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Connect", command=self._connect_selected).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(btn_row, text="Disconnect", command=self._disconnect_trainer).pack(side="left", fill="x", expand=True)
        self.device_status = tk.Label(device_box, text="Disconnected", bg="#132033", fg="#c8d7ee", anchor="w", justify="left")
        self.device_status.pack(fill="x", pady=(6, 0))

        workout_box = ttk.LabelFrame(left, text="Workout", padding=8)
        workout_box.pack(fill="x", pady=(0, 8))
        self.workout_combo = ttk.Combobox(workout_box, state="readonly")
        self.workout_combo.pack(fill="x")
        self.workout_combo.bind("<<ComboboxSelected>>", self._on_workout_pick)
        import_row = ttk.Frame(workout_box)
        import_row.pack(fill="x", pady=(6, 0))
        ttk.Button(import_row, text="Import JSON", command=self._import_json).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(import_row, text="Import ZWO", command=self._import_zwo).pack(side="left", fill="x", expand=True)
        self.workout_details = tk.Text(
            workout_box,
            height=10,
            bg="#0f1725",
            fg="#e7f0ff",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 10),
            wrap="none",
        )
        self.workout_details.pack(fill="both", expand=True, pady=(6, 0))
        self.workout_details.configure(state="disabled")

        control_box = ttk.LabelFrame(left, text="Controls", padding=8)
        control_box.pack(fill="x")
        row1 = ttk.Frame(control_box)
        row1.pack(fill="x")
        ttk.Button(row1, text="Start", command=self._start).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row1, text="Pause", command=self._pause).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row1, text="Resume", command=self._resume).pack(side="left", fill="x", expand=True)
        row2 = ttk.Frame(control_box)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Button(row2, text="Back", command=self._back_step).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row2, text="Skip", command=self._skip_step).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row2, text="End", command=self._end_session).pack(side="left", fill="x", expand=True)

        ttk.Label(control_box, text="Intensity Bias").pack(anchor="w", pady=(8, 0))
        self.bias_var = tk.DoubleVar(value=100.0)
        self.bias_scale = ttk.Scale(control_box, from_=90.0, to=110.0, variable=self.bias_var, command=self._on_bias_change)
        self.bias_scale.pack(fill="x")
        self.bias_label = ttk.Label(control_box, text="100%")
        self.bias_label.pack(anchor="w", pady=(2, 0))

        self.stage_label = ttk.Label(main, text="Device Screen", font=("Segoe UI", 15, "bold"))
        self.stage_label.grid(row=0, column=0, sticky="w")

        hero = ttk.Frame(main)
        hero.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        hero.columnconfigure(0, weight=1)
        hero.columnconfigure(1, weight=1)

        target_box = tk.Frame(hero, bg="#1f3557", bd=0, relief="flat", padx=20, pady=12)
        target_box.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Label(target_box, text="TARGET", bg="#1f3557", fg="#bfdbfe", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        self.target_big = tk.Label(target_box, text="0 W", bg="#1f3557", fg="#f8fafc", font=("Segoe UI", 46, "bold"))
        self.target_big.pack(anchor="w")

        power_box = tk.Frame(hero, bg="#193248", bd=0, relief="flat", padx=20, pady=12)
        power_box.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        tk.Label(power_box, text="CURRENT POWER", bg="#193248", fg="#bfdbfe", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        self.power_big = tk.Label(power_box, text="0 W", bg="#193248", fg="#f8fafc", font=("Segoe UI", 46, "bold"))
        self.power_big.pack(anchor="w")

        meta = ttk.Frame(main)
        meta.grid(row=2, column=0, sticky="nsew")
        meta.columnconfigure(0, weight=1)
        meta.rowconfigure(4, weight=1)

        self.metric_vars = {
            "elapsed": tk.StringVar(value="00:00"),
            "remaining": tk.StringVar(value="00:00"),
            "step": tk.StringVar(value="-"),
            "step_countdown": tk.StringVar(value="00:00"),
            "cadence": tk.StringVar(value="--"),
            "speed": tk.StringVar(value="--"),
            "connection": tk.StringVar(value="Disconnected"),
        }
        grid = ttk.Frame(meta)
        grid.grid(row=0, column=0, sticky="ew")
        labels = [
            ("Elapsed", "elapsed"),
            ("Remaining", "remaining"),
            ("Step", "step"),
            ("Step Left", "step_countdown"),
            ("Cadence", "cadence"),
            ("Speed", "speed"),
            ("Connection", "connection"),
        ]
        for idx, (label, key) in enumerate(labels):
            col = idx % 4
            row = idx // 4
            block = ttk.Frame(grid)
            block.grid(row=row, column=col, sticky="ew", padx=6, pady=4)
            ttk.Label(block, text=label, foreground="#9fb4d0").pack(anchor="w")
            ttk.Label(block, textvariable=self.metric_vars[key], font=("Segoe UI", 14, "bold")).pack(anchor="w")
            grid.columnconfigure(col, weight=1)

        self.step_progress = ttk.Progressbar(meta, mode="determinate", maximum=100)
        self.step_progress.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        self.workout_progress = ttk.Progressbar(meta, mode="determinate", maximum=100)
        self.workout_progress.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        self.chart = tk.Canvas(meta, bg="#0f1725", highlightthickness=0, height=220)
        self.chart.grid(row=4, column=0, sticky="nsew")

        self.summary_box = ttk.LabelFrame(meta, text="Post-Ride Summary", padding=8)
        self.summary_box.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        self.summary_text = tk.Text(
            self.summary_box,
            height=6,
            bg="#0f1725",
            fg="#e7f0ff",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 10),
            wrap="word",
        )
        self.summary_text.pack(fill="x")
        self.summary_text.configure(state="disabled")
        export_row = ttk.Frame(self.summary_box)
        export_row.pack(fill="x", pady=(6, 0))
        self.export_csv_btn = ttk.Button(export_row, text="Export CSV", command=self._export_csv, state="disabled")
        self.export_csv_btn.pack(side="left", padx=(0, 4))
        self.export_fit_btn = ttk.Button(export_row, text="Export FIT", command=self._export_fit, state="disabled")
        self.export_fit_btn.pack(side="left")

    def _refresh_workout_selector(self) -> None:
        names = [self.workouts[k].name for k in self.workouts]
        self.workout_combo["values"] = names
        if not self.selected_workout_id and self.workouts:
            self.selected_workout_id = next(iter(self.workouts.keys()))
        if self.selected_workout_id:
            current_name = self.workouts[self.selected_workout_id].name
            self.workout_combo.set(current_name)

    def _refresh_workout_details(self) -> None:
        workout = self._current_workout()
        lines: list[str] = []
        if workout is None:
            lines.append("No workout selected.")
        else:
            lines.append(f"{workout.name}")
            lines.append(f"FTP: {workout.ftp} W")
            lines.append(f"Total: {fmt_clock(workout.total_duration_s)}")
            lines.append("")
            for idx, step in enumerate(workout.steps, start=1):
                lines.append(f"{idx:02d} {step.type:12} {fmt_clock(step.duration_s):>8}  {step.target_pct_ftp * 100:6.1f}% FTP")
        self.workout_details.configure(state="normal")
        self.workout_details.delete("1.0", tk.END)
        self.workout_details.insert("1.0", "\n".join(lines))
        self.workout_details.configure(state="disabled")

    def _set_state(self, new_state: EngineState, message: str | None = None, force: bool = False) -> None:
        if not force and new_state not in ALLOWED_TRANSITIONS.get(self.state, set()):
            return
        self.state = new_state
        if message:
            self.state_message = message
        colors = {
            EngineState.IDLE: "#4b5563",
            EngineState.READY: "#0ea5e9",
            EngineState.COUNTDOWN: "#f59e0b",
            EngineState.RUNNING: "#22c55e",
            EngineState.PAUSED: "#f97316",
            EngineState.STEP_TRANSITION: "#eab308",
            EngineState.COMPLETED: "#16a34a",
            EngineState.ABORTED: "#ef4444",
            EngineState.ERROR: "#dc2626",
        }
        self.badge_state.configure(text=self.state.value, bg=colors[self.state])

    def _current_workout(self) -> Workout | None:
        if not self.selected_workout_id:
            return None
        return self.workouts.get(self.selected_workout_id)

    def _current_step(self) -> WorkoutStep | None:
        workout = self._current_workout()
        if workout is None:
            return None
        if not (0 <= self.step_index < len(workout.steps)):
            return None
        return workout.steps[self.step_index]

    def _target_watts_for_step(self, step: WorkoutStep | None = None) -> int:
        workout = self._current_workout()
        step = step or self._current_step()
        if workout is None or step is None:
            return SAFE_PAUSE_WATTS
        ceiling = max(MIN_TARGET_WATTS, int(round(workout.ftp * 1.5)))
        target = round(workout.ftp * step.target_pct_ftp * self.intensity_bias)
        return int(clamp(float(target), float(MIN_TARGET_WATTS), float(ceiling)))

    def _effective_target(self, now_ts: float) -> int:
        base_target = self._target_watts_for_step()
        if self.state in {EngineState.IDLE, EngineState.READY, EngineState.COUNTDOWN, EngineState.COMPLETED, EngineState.ABORTED}:
            return SAFE_PAUSE_WATTS
        if self.state == EngineState.PAUSED and not self.auto_paused_for_disconnect:
            return SAFE_PAUSE_WATTS
        if self.state == EngineState.RUNNING and self.ramp is not None:
            start, end, started, duration = self.ramp
            delta = now_ts - started
            if delta >= duration:
                self.ramp = None
                base_target = int(round(end))
            else:
                ratio = clamp(delta / duration, 0.0, 1.0)
                base_target = int(round(start + ((end - start) * ratio)))
        if self.cadence_relief_active and self.state == EngineState.RUNNING:
            base_target = max(MIN_TARGET_WATTS, int(round(base_target * 0.75)))
        return int(clamp(float(base_target), float(MIN_TARGET_WATTS), 2000.0))

    def _set_stage(self) -> None:
        if self.state in {EngineState.IDLE, EngineState.READY}:
            self.stage_label.configure(text="Device / Workout Setup")
        elif self.state in {EngineState.COUNTDOWN, EngineState.RUNNING, EngineState.PAUSED, EngineState.STEP_TRANSITION}:
            self.stage_label.configure(text="Ride Screen (ERG)")
        elif self.state in {EngineState.COMPLETED, EngineState.ABORTED}:
            self.stage_label.configure(text="Summary Screen")
        else:
            self.stage_label.configure(text="Error")

    def _on_bias_change(self, _value: str = "") -> None:
        self.intensity_bias = float(self.bias_var.get()) / 100.0
        self.bias_label.configure(text=f"{self.intensity_bias * 100:.0f}%")
        self.target_dirty = True
        self._configure_ramp(self.last_target_sent or self._target_watts_for_step(), self._target_watts_for_step())

    def _on_workout_pick(self, _event: Any = None) -> None:
        selected_name = self.workout_combo.get()
        for workout_id, workout in self.workouts.items():
            if workout.name == selected_name:
                self.selected_workout_id = workout_id
                break
        self.step_index = 0
        self.step_elapsed_s = 0.0
        self.workout_elapsed_s = 0.0
        self.target_dirty = True
        self._refresh_workout_details()
        self._enter_ready_if_possible()

    def _on_device_selected(self, _event: Any = None) -> None:
        sel = self.device_list.curselection()
        if not sel:
            return
        self.selected_device_index = int(sel[0])

    def _scan_devices(self) -> None:
        if self.pending_scan and not self.pending_scan.done():
            return
        self.state_message = "Scanning BLE for KICKR CORE..."
        self.pending_scan = self.bridge.scan_kickr(SCAN_TIMEOUT_SECONDS)

    def _connect_selected(self) -> None:
        if self.pending_connect and not self.pending_connect.done():
            return
        if self.selected_device_index is None:
            messagebox.showinfo("Trainer", "Select a KICKR CORE from the list first.")
            return
        if self.selected_device_index >= len(self.device_choices):
            return
        choice = self.device_choices[self.selected_device_index]
        self.connected_address = choice.address
        self.state_message = f"Connecting {choice.name}..."
        self.pending_connect = self.bridge.connect(choice.address)

    def _disconnect_trainer(self) -> None:
        if self.pending_disconnect and not self.pending_disconnect.done():
            return
        self.user_disconnect_requested = True
        self.pending_disconnect = self.bridge.disconnect()
        self.connected_address = None
        self.reconnect_until = 0.0
        self.next_reconnect_attempt = 0.0
        self.auto_paused_for_disconnect = False
        self._set_state(EngineState.IDLE, "Disconnected.")

    def _import_json(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Workout JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            workout = load_json_workout(Path(path))
            self.workouts[workout.id] = workout
            self.store.save_workout(workout)
            self.selected_workout_id = workout.id
            self._refresh_workout_selector()
            self._refresh_workout_details()
            self._enter_ready_if_possible()
        except Exception as exc:
            messagebox.showerror("Import JSON", f"Failed to import workout:\n{exc}")

    def _import_zwo(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Zwift Workout", "*.zwo"), ("All files", "*.*")])
        if not path:
            return
        try:
            workout = load_zwo_workout(Path(path))
            self.workouts[workout.id] = workout
            self.store.save_workout(workout)
            self.selected_workout_id = workout.id
            self._refresh_workout_selector()
            self._refresh_workout_details()
            self._enter_ready_if_possible()
        except Exception as exc:
            messagebox.showerror("Import ZWO", f"Failed to import ZWO:\n{exc}")

    def _enter_ready_if_possible(self) -> None:
        workout = self._current_workout()
        connected = self.bridge.snapshot().get("connected")
        if workout is not None and connected and self.state in {EngineState.IDLE, EngineState.ERROR, EngineState.ABORTED, EngineState.COMPLETED}:
            self.step_index = 0
            self.step_elapsed_s = 0.0
            self.workout_elapsed_s = 0.0
            self._set_state(EngineState.READY, "Trainer connected and workout loaded.")

    def _start(self) -> None:
        if self.state != EngineState.READY:
            return
        if self.session_id is None:
            self._begin_session()
        self.countdown_s = 3.0
        self._set_state(EngineState.COUNTDOWN, "Starting in 3...")
        self.target_dirty = True
        self._future_noop(self.bridge.request_control())
        self._future_noop(self.bridge.start_resume())

    def _pause(self) -> None:
        if self.state not in {EngineState.RUNNING, EngineState.COUNTDOWN, EngineState.STEP_TRANSITION}:
            return
        self.auto_paused_for_disconnect = False
        self._set_state(EngineState.PAUSED, "Paused.")
        self.target_dirty = True
        self._future_noop(self.bridge.pause())

    def _resume(self) -> None:
        if self.state != EngineState.PAUSED:
            return
        self.auto_paused_for_disconnect = False
        self._set_state(EngineState.RUNNING, "Resumed.")
        self.target_dirty = True
        self._future_noop(self.bridge.start_resume())

    def _skip_step(self) -> None:
        workout = self._current_workout()
        if workout is None:
            return
        if self.state not in {EngineState.RUNNING, EngineState.PAUSED, EngineState.COUNTDOWN, EngineState.READY, EngineState.STEP_TRANSITION}:
            return
        if self.step_index >= len(workout.steps) - 1:
            self._complete_session()
            return
        old_target = self._target_watts_for_step()
        self.step_index += 1
        self.step_elapsed_s = 0.0
        self._configure_ramp(old_target, self._target_watts_for_step())
        if self.state == EngineState.RUNNING:
            self._set_state(EngineState.STEP_TRANSITION, "Step skipped.")
            self._set_state(EngineState.RUNNING, "Running.", force=True)
        self.target_dirty = True

    def _back_step(self) -> None:
        if self.step_index <= 0:
            return
        if self.state not in {EngineState.RUNNING, EngineState.PAUSED, EngineState.COUNTDOWN, EngineState.READY, EngineState.STEP_TRANSITION}:
            return
        old_target = self._target_watts_for_step()
        self.step_index -= 1
        self.step_elapsed_s = 0.0
        self._configure_ramp(old_target, self._target_watts_for_step())
        if self.state == EngineState.RUNNING:
            self._set_state(EngineState.STEP_TRANSITION, "Step moved back.")
            self._set_state(EngineState.RUNNING, "Running.", force=True)
        self.target_dirty = True

    def _end_session(self) -> None:
        if self.state not in {EngineState.RUNNING, EngineState.PAUSED, EngineState.COUNTDOWN, EngineState.STEP_TRANSITION, EngineState.READY}:
            return
        self._set_state(EngineState.ABORTED, "Session ended by user.")
        self._finalize_session("aborted")
        self._future_noop(self.bridge.pause())
        self._future_noop(self.bridge.reset())
        self.target_dirty = True

    def _begin_session(self) -> None:
        workout = self._current_workout()
        if workout is None:
            return
        self.session_id = uuid.uuid4().hex
        self.session_start_iso = now_iso()
        self.session_start_perf = time.perf_counter()
        self.samples = []
        self.next_sample_ts = self.session_start_perf
        self.display_power_window.clear()
        self.chart_power.clear()
        self.chart_target.clear()
        self.latest_summary = None
        self.latest_status = None
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.configure(state="disabled")
        self.export_csv_btn.configure(state="disabled")
        self.export_fit_btn.configure(state="disabled")

    def _complete_session(self) -> None:
        self._set_state(EngineState.COMPLETED, "Workout complete.")
        self._finalize_session("completed")
        self._future_noop(self.bridge.pause())
        self._future_noop(self.bridge.reset())
        self.target_dirty = True

    def _configure_ramp(self, start_target: int, end_target: int) -> None:
        delta = abs(end_target - start_target)
        if delta < 50:
            self.ramp = None
            return
        duration = clamp(delta / 30.0, 3.0, 5.0)
        self.ramp = (float(start_target), float(end_target), time.perf_counter(), float(duration))

    def _future_noop(self, future: Any) -> None:
        if future is None:
            return
        future.add_done_callback(lambda f: None)

    def _handle_target_write_done(self, future: Any, target: int, reason: str) -> None:
        self.pending_target_write = False
        exc = future.exception()
        if exc:
            self.control_failures += 1
            self.state_message = f"Control write failed ({self.control_failures}/{CONTROL_ERROR_LIMIT}): {exc}"
            if self.control_failures >= CONTROL_ERROR_LIMIT:
                self._set_state(EngineState.ERROR, "Trainer control failed repeatedly. Reconnect required.")
            return
        self.control_failures = 0
        self.last_target_sent = target
        self.last_target_sent_ts = time.perf_counter()
        if reason:
            self.state_message = reason

    def _send_target_if_needed(self, now_ts: float) -> None:
        telemetry = self.bridge.snapshot()
        if not telemetry.get("connected"):
            return
        if self.pending_target_write:
            return
        desired = self._effective_target(now_ts)
        should_send = (
            self.target_dirty
            or self.last_target_sent is None
            or desired != self.last_target_sent
            or (now_ts - self.last_target_sent_ts) >= KEEPALIVE_SECONDS
        )
        if not should_send:
            return
        self.pending_target_write = True
        self.target_dirty = False
        reason = f"Set target {desired}W"
        fut = self.bridge.set_target_power(desired)
        fut.add_done_callback(lambda f: self.after(0, self._handle_target_write_done, f, desired, reason))

    def _sample(self, now_ts: float, telemetry: dict[str, Any], target: int) -> None:
        if self.session_id is None:
            return
        if now_ts < self.next_sample_ts:
            return
        while now_ts >= self.next_sample_ts:
            t_ms = int((self.next_sample_ts - self.session_start_perf) * 1000.0)
            raw_power = float(telemetry.get("power_w") or 0.0)
            cadence = telemetry.get("cadence_rpm")
            speed = telemetry.get("speed_kph")
            row = {
                "t_ms": t_ms,
                "power": raw_power,
                "cadence": float(cadence) if cadence is not None else None,
                "speed": float(speed) if speed is not None else None,
                "target_power": int(target),
                "step_index": int(self.step_index),
            }
            self.samples.append(row)
            self.chart_power.append(raw_power)
            self.chart_target.append(float(target))
            self.display_power_window.append(raw_power)
            self.next_sample_ts += SAMPLE_SECONDS

    def _compute_summary(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "duration_s": int(self.workout_elapsed_s),
                "avg_power_w": 0,
                "max_power_w": 0,
                "time_in_target_band_s": 0,
                "time_in_target_band_pct": 0.0,
                "interval_compliance_pct": 0.0,
                "total_work_kj": 0.0,
            }
        powers = [max(0.0, float(s.get("power") or 0.0)) for s in self.samples]
        target_rows = [s for s in self.samples if (s.get("target_power") or 0) > 0]
        in_band = 0
        for row in target_rows:
            target = float(row["target_power"])
            power = float(row.get("power") or 0.0)
            if abs(power - target) <= (target * 0.05):
                in_band += 1
        compliance = (in_band / max(len(target_rows), 1)) * 100.0
        total_work_kj = sum(powers) / 1000.0
        return {
            "duration_s": int(self.workout_elapsed_s),
            "avg_power_w": int(round(sum(powers) / max(len(powers), 1))),
            "max_power_w": int(round(max(powers))),
            "time_in_target_band_s": int(in_band),
            "time_in_target_band_pct": round(compliance, 1),
            "interval_compliance_pct": round(compliance, 1),
            "total_work_kj": round(total_work_kj, 1),
        }

    def _finalize_session(self, status: str) -> None:
        if self.session_id is None or self.session_start_iso is None:
            return
        workout = self._current_workout()
        if workout is None:
            return
        summary = self._compute_summary()
        end_ts = now_iso()
        self.store.save_session(
            session_id=self.session_id,
            workout_id=workout.id,
            start_ts=self.session_start_iso,
            end_ts=end_ts,
            ftp=workout.ftp,
            status=status,
            summary=summary,
            samples=self.samples,
        )
        self.latest_summary = summary
        self.latest_status = status
        self._render_summary()
        self.export_csv_btn.configure(state="normal")
        self.export_fit_btn.configure(state="normal")

    def _render_summary(self) -> None:
        if self.latest_summary is None:
            return
        summary = self.latest_summary
        lines = [
            f"Status: {self.latest_status}",
            f"Duration: {fmt_clock(summary['duration_s'])}",
            f"Avg Power: {summary['avg_power_w']} W",
            f"Max Power: {summary['max_power_w']} W",
            f"Time In Band (+/-5%): {summary['time_in_target_band_s']} s ({summary['time_in_target_band_pct']}%)",
            f"Interval Compliance: {summary['interval_compliance_pct']}%",
            f"Total Work: {summary['total_work_kj']} kJ",
        ]
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state="disabled")

    def _export_csv(self) -> None:
        if self.session_id is None or not self.samples:
            messagebox.showinfo("Export CSV", "No recorded session to export.")
            return
        workout = self._current_workout()
        if workout is None:
            return
        default_name = f"{workout.id}_{self.session_id[:8]}.csv"
        path = filedialog.asksaveasfilename(
            title="Export Session CSV",
            initialdir=str(self.exports_dir),
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["session_id", self.session_id])
            writer.writerow(["workout_id", workout.id])
            writer.writerow(["status", self.latest_status or "unknown"])
            if self.latest_summary:
                for key, value in self.latest_summary.items():
                    writer.writerow([key, value])
            writer.writerow([])
            writer.writerow(["t_ms", "power", "cadence", "speed", "target_power", "step_index"])
            for row in self.samples:
                writer.writerow(
                    [
                        int(row["t_ms"]),
                        float(row["power"]),
                        row["cadence"] if row["cadence"] is not None else "",
                        row["speed"] if row["speed"] is not None else "",
                        int(row["target_power"]),
                        int(row["step_index"]),
                    ]
                )
        messagebox.showinfo("Export CSV", f"Saved:\n{path}")

    def _export_fit(self) -> None:
        if not FIT_EXPORT_AVAILABLE:
            messagebox.showwarning("Export FIT", "FIT export dependency is unavailable in this build.")
            return
        if self.session_id is None or not self.samples or self.session_start_iso is None:
            messagebox.showinfo("Export FIT", "No recorded session to export.")
            return
        workout = self._current_workout()
        if workout is None:
            return
        default_name = f"{workout.id}_{self.session_id[:8]}.fit"
        path = filedialog.asksaveasfilename(
            title="Export Session FIT",
            initialdir=str(self.exports_dir),
            initialfile=default_name,
            defaultextension=".fit",
            filetypes=[("FIT", "*.fit")],
        )
        if not path:
            return
        try:
            start_ts_ms = int(datetime.fromisoformat(self.session_start_iso).timestamp() * 1000)
            end_ts_ms = start_ts_ms + int(self.workout_elapsed_s * 1000.0)
            summary = self.latest_summary or self._compute_summary()
            builder = FitFileBuilder(auto_define=True)

            file_id = FileIdMessage()
            file_id.type = FileType.ACTIVITY
            file_id.manufacturer = Manufacturer.DEVELOPMENT
            file_id.product = 1
            file_id.serial_number = 1
            file_id.time_created = start_ts_ms
            builder.add(file_id)

            start_event = EventMessage()
            start_event.event = Event.TIMER
            start_event.event_type = EventType.START
            start_event.timestamp = start_ts_ms
            builder.add(start_event)

            for sample in self.samples:
                record = RecordMessage()
                record.timestamp = start_ts_ms + int(sample["t_ms"])
                record.power = int(round(float(sample["power"])))
                if sample.get("cadence") is not None:
                    record.cadence = int(round(float(sample["cadence"])))
                if sample.get("speed") is not None:
                    ms = float(sample["speed"]) * (1000.0 / 3600.0)
                    record.speed = int(round(ms * 1000.0))
                builder.add(record)

            stop_event = EventMessage()
            stop_event.event = Event.TIMER
            stop_event.event_type = EventType.STOP
            stop_event.timestamp = end_ts_ms
            builder.add(stop_event)

            lap = LapMessage()
            lap.timestamp = end_ts_ms
            lap.total_elapsed_time = float(self.workout_elapsed_s)
            lap.total_timer_time = float(self.workout_elapsed_s)
            lap.avg_power = int(summary["avg_power_w"])
            lap.max_power = int(summary["max_power_w"])
            lap.total_work = int(round(float(summary["total_work_kj"]) * 1000.0))
            builder.add(lap)

            session = SessionMessage()
            session.timestamp = end_ts_ms
            session.total_elapsed_time = float(self.workout_elapsed_s)
            session.total_timer_time = float(self.workout_elapsed_s)
            session.avg_power = int(summary["avg_power_w"])
            session.max_power = int(summary["max_power_w"])
            session.total_work = int(round(float(summary["total_work_kj"]) * 1000.0))
            session.sport = Sport.CYCLING
            session.sub_sport = SubSport.INDOOR_CYCLING
            builder.add(session)

            activity = ActivityMessage()
            activity.timestamp = end_ts_ms
            activity.total_timer_time = float(self.workout_elapsed_s)
            activity.num_sessions = 1
            builder.add(activity)

            fit_file = builder.build()
            fit_file.to_file(path)
            messagebox.showinfo("Export FIT", f"Saved:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export FIT", f"Failed to export FIT:\n{exc}")

    def _poll_futures(self) -> None:
        if self.pending_scan is not None and self.pending_scan.done():
            try:
                self.device_choices = self.pending_scan.result()
                self.device_list.delete(0, tk.END)
                for index, choice in enumerate(self.device_choices):
                    self.device_list.insert(tk.END, f"{choice.name} ({choice.rssi:+d} dBm)")
                    if index == 0:
                        self.selected_device_index = 0
                if self.device_choices:
                    self.device_list.selection_set(0)
                else:
                    self.selected_device_index = None
                self.state_message = f"Scan complete: {len(self.device_choices)} trainer(s)."
            except Exception as exc:
                self.state_message = f"Scan failed: {exc}"
            self.pending_scan = None

        if self.pending_connect is not None and self.pending_connect.done():
            try:
                self.pending_connect.result()
                self.state_message = "Trainer connected."
                self._enter_ready_if_possible()
            except Exception as exc:
                self.state_message = f"Connect failed: {exc}"
                self._set_state(EngineState.ERROR, self.state_message)
            self.pending_connect = None

        if self.pending_disconnect is not None and self.pending_disconnect.done():
            try:
                self.pending_disconnect.result()
                self.state_message = "Trainer disconnected."
            except Exception as exc:
                self.state_message = f"Disconnect failed: {exc}"
            self.pending_disconnect = None
            self.user_disconnect_requested = False

    def _handle_connection_events(self, now_ts: float, telemetry: dict[str, Any]) -> None:
        connected = bool(telemetry.get("connected"))
        if self.last_connected_flag and not connected:
            if self.user_disconnect_requested:
                self.user_disconnect_requested = False
            else:
                self.state_message = "Trainer disconnected. Auto-reconnect started."
                if self.state in {EngineState.RUNNING, EngineState.COUNTDOWN, EngineState.STEP_TRANSITION}:
                    self.auto_paused_for_disconnect = True
                    self._set_state(EngineState.PAUSED, "Connection lost. Timer paused.")
                    self.target_dirty = True
                self.reconnect_until = now_ts + RECONNECT_WINDOW_SECONDS
                self.next_reconnect_attempt = now_ts
        if (not self.last_connected_flag) and connected and self.reconnect_until > 0.0:
            self.reconnect_until = 0.0
            self.next_reconnect_attempt = 0.0
            self.state_message = "Trainer reconnected. ERG target reapplied. Timer remains paused."
            self._future_noop(self.bridge.request_control())
            self._future_noop(self.bridge.start_resume())
            self.target_dirty = True
        self.last_connected_flag = connected

        if self.reconnect_until > 0.0 and not connected:
            if now_ts > self.reconnect_until:
                self.reconnect_until = 0.0
                self.next_reconnect_attempt = 0.0
                self._set_state(EngineState.ERROR, "Reconnect window expired.")
            elif now_ts >= self.next_reconnect_attempt and self.connected_address and self.pending_connect is None:
                self.next_reconnect_attempt = now_ts + RECONNECT_INTERVAL_SECONDS
                self.pending_connect = self.bridge.connect(self.connected_address)

    def _update_engine(self, dt: float, now_ts: float, telemetry: dict[str, Any]) -> None:
        workout = self._current_workout()
        if workout is None:
            return

        if self.state == EngineState.COUNTDOWN:
            self.countdown_s -= dt
            if self.countdown_s <= 0:
                self._set_state(EngineState.RUNNING, "Ride started.")
                self.target_dirty = True
                self._future_noop(self.bridge.start_resume())

        if self.state == EngineState.RUNNING:
            self.workout_elapsed_s += dt
            self.step_elapsed_s += dt
            step = self._current_step()
            if step and self.step_elapsed_s >= step.duration_s:
                old_target = self._target_watts_for_step(step)
                self.step_elapsed_s = 0.0
                self.step_index += 1
                if self.step_index >= len(workout.steps):
                    self.step_index = len(workout.steps) - 1
                    self._complete_session()
                else:
                    self._set_state(EngineState.STEP_TRANSITION, "Step changed.")
                    new_target = self._target_watts_for_step()
                    self._configure_ramp(old_target, new_target)
                    self.target_dirty = True
                    self._set_state(EngineState.RUNNING, "Running.", force=True)

        cadence = telemetry.get("cadence_rpm")
        if self.state == EngineState.RUNNING:
            if cadence is not None and float(cadence) <= 5.0:
                if self.zero_cadence_since is None:
                    self.zero_cadence_since = now_ts
                elif (now_ts - self.zero_cadence_since) > CADENCE_STALL_SECONDS and not self.cadence_relief_active:
                    self.cadence_relief_active = True
                    self.state_message = "Cadence stalled. Pedal to resume full target."
                    self.target_dirty = True
            else:
                self.zero_cadence_since = None
                if self.cadence_relief_active and cadence is not None and float(cadence) > 12.0:
                    self.cadence_relief_active = False
                    self.target_dirty = True
        else:
            self.zero_cadence_since = None
            if self.cadence_relief_active:
                self.cadence_relief_active = False
                self.target_dirty = True

        if (now_ts - self.last_control_eval_ts) >= CONTROL_LOOP_SECONDS:
            self.last_control_eval_ts = now_ts
            self._send_target_if_needed(now_ts)

        self._sample(now_ts, telemetry, self._target_watts_for_step())

    def _draw_chart(self) -> None:
        if (time.perf_counter() - self.chart_last_draw_ts) < 0.20:
            return
        self.chart_last_draw_ts = time.perf_counter()
        c = self.chart
        c.delete("all")
        width = max(c.winfo_width(), 10)
        height = max(c.winfo_height(), 10)
        c.create_rectangle(0, 0, width, height, fill="#0f1725", outline="")

        for i in range(1, 5):
            y = int((height * i) / 5)
            c.create_line(0, y, width, y, fill="#1f2a3a")

        powers = list(self.chart_power)
        targets = list(self.chart_target)
        if len(powers) < 2 or len(targets) < 2:
            c.create_text(12, 12, anchor="nw", text="Power vs Target (last 5 min)", fill="#93a8c7", font=("Segoe UI", 10))
            return

        max_value = max(200.0, max(max(powers), max(targets)) * 1.15)
        n = min(len(powers), len(targets))
        powers = powers[-n:]
        targets = targets[-n:]

        def to_xy(index: int, value: float) -> tuple[float, float]:
            x = (index / max(n - 1, 1)) * (width - 20) + 10
            y = height - 10 - ((value / max_value) * (height - 20))
            return x, y

        p_points: list[float] = []
        t_points: list[float] = []
        for i in range(n):
            px, py = to_xy(i, powers[i])
            tx, ty = to_xy(i, targets[i])
            p_points.extend([px, py])
            t_points.extend([tx, ty])

        c.create_line(*t_points, fill="#fbbf24", width=2)
        c.create_line(*p_points, fill="#38bdf8", width=2)
        c.create_text(12, 12, anchor="nw", text="Power (blue) vs Target (amber) - last 5 min", fill="#93a8c7", font=("Segoe UI", 10))

    def _refresh_ui(self) -> None:
        telemetry = self.bridge.snapshot()
        connected = bool(telemetry.get("connected"))
        connection_label = "Connected" if connected else ("Reconnecting" if self.reconnect_until > 0 else "Disconnected")
        self.metric_vars["connection"].set(connection_label)
        self.device_status.configure(
            text=f"{telemetry.get('status','-')}\nControl: {telemetry.get('control_status','-')}\nFTMS Status: {telemetry.get('machine_status','-')}"
        )

        workout = self._current_workout()
        step = self._current_step()
        target = self._target_watts_for_step(step) if step else 0
        raw_power = float(telemetry.get("power_w") or 0.0)
        if self.display_power_window:
            shown_power = sum(self.display_power_window) / len(self.display_power_window)
        else:
            shown_power = raw_power
        self.target_big.configure(text=f"{target} W")
        self.power_big.configure(text=f"{int(round(shown_power))} W")

        self.metric_vars["elapsed"].set(fmt_clock(self.workout_elapsed_s))
        total = float(workout.total_duration_s) if workout else 0.0
        remaining = max(total - self.workout_elapsed_s, 0.0)
        self.metric_vars["remaining"].set(fmt_clock(remaining))
        self.metric_vars["step"].set(step.type if step else "-")
        if step:
            step_remaining = max(step.duration_s - self.step_elapsed_s, 0.0)
            self.metric_vars["step_countdown"].set(fmt_clock(step_remaining))
            self.step_progress["value"] = (self.step_elapsed_s / max(step.duration_s, 1)) * 100.0
        else:
            self.metric_vars["step_countdown"].set("00:00")
            self.step_progress["value"] = 0.0
        self.workout_progress["value"] = (self.workout_elapsed_s / max(total, 1.0)) * 100.0 if total else 0.0

        cadence = telemetry.get("cadence_rpm")
        speed = telemetry.get("speed_kph")
        self.metric_vars["cadence"].set(f"{float(cadence):.0f} rpm" if cadence is not None else "--")
        self.metric_vars["speed"].set(f"{float(speed):.1f} km/h" if speed is not None else "--")

        self._set_stage()
        self._draw_chart()

    def _tick(self) -> None:
        now_ts = time.perf_counter()
        dt = min(now_ts - self.last_tick, 0.2)
        self.last_tick = now_ts

        self._poll_futures()
        telemetry = self.bridge.snapshot()
        self._handle_connection_events(now_ts, telemetry)
        self._update_engine(dt, now_ts, telemetry)
        self._refresh_ui()

        self.after(80, self._tick)

    def _on_close(self) -> None:
        if self.session_id and self.state in {EngineState.RUNNING, EngineState.PAUSED, EngineState.COUNTDOWN, EngineState.STEP_TRANSITION, EngineState.READY}:
            should_end = messagebox.askyesno("Exit", "End current session and exit?")
            if not should_end:
                return
            self._set_state(EngineState.ABORTED, "Session ended on exit.")
            self._finalize_session("aborted")
        try:
            self.bridge.shutdown()
        finally:
            self.store.close()
            self.destroy()


def main() -> None:
    app = LocalWattApp()
    app.mainloop()


if __name__ == "__main__":
    main()
