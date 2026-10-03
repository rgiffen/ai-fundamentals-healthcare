"""A simulated bedside-monitor MQTT feed.

Replaces the former MAS400 autonomous-ship AIS feed. That notebook subscribed to a
public marine broker; the healthcare equivalent cannot, because there is no public
stream of real patient vitals and there should not be. So this module simulates the
broker instead.

The point of the lesson is the streaming pattern - connect, subscribe to topics,
receive messages asynchronously, buffer them, turn the buffer into a DataFrame - and
that pattern is preserved exactly. `SimulatedMonitorClient` mirrors the parts of the
`paho.mqtt.client.Client` API the notebook uses, so the code a student writes here is
the code they would write against a real broker.

Topics
------
    ward/bed/<bed_id>/vitals    continuous monitor channels, every `interval` seconds
    ward/bed/<bed_id>/status    slower device and alarm state

Payloads are JSON, the way a real HL7-over-MQTT or device gateway feed would be.

Usage
-----
    from bedside_monitor_simulator import SimulatedMonitorClient

    client = SimulatedMonitorClient(beds=4, speedup=120)
    client.on_connect = lambda c, u, f, rc: print("connected", rc)
    client.on_message = lambda c, u, msg: print(msg.topic, msg.payload[:60])
    client.connect()
    client.subscribe("ward/bed/+/vitals")
    client.loop_start()
    ...
    client.loop_stop(); client.disconnect()

To point the same notebook at a real broker, swap this class for
`paho.mqtt.client.Client` and change the topic strings. Nothing else changes.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import numpy as np

from synthetic_vitals_generator import CHANNELS, VitalsGenerator

# --------------------------------------------------------------------------
# Message object: the same attributes paho gives a subscriber
# --------------------------------------------------------------------------


@dataclass
class SimulatedMessage:
    topic: str
    payload: bytes
    qos: int = 0
    retain: bool = False


def _topic_matches(pattern: str, topic: str) -> bool:
    """MQTT topic matching, supporting the + and # wildcards."""
    p = pattern.split("/")
    t = topic.split("/")
    for i, seg in enumerate(p):
        if seg == "#":
            return True
        if i >= len(t):
            return False
        if seg == "+":
            continue
        if seg != t[i]:
            return False
    return len(p) == len(t)


# --------------------------------------------------------------------------
# The simulated client
# --------------------------------------------------------------------------


class SimulatedMonitorClient:
    """Stands in for `paho.mqtt.client.Client` against a simulated ward.

    Parameters
    ----------
    beds : int
        How many bedside monitors publish to the ward.
    speedup : float
        How much faster than real time to run. The underlying patient traces are
        sampled every 5 minutes; a speedup of 120 delivers one message every
        2.5 seconds of wall clock. Raise it to collect a long stay quickly.
    hours : float
        How many hours of each patient's stay to simulate before looping.
    seed : int
        Reproducibility.
    """

    def __init__(
        self,
        beds: int = 4,
        speedup: float = 300.0,
        hours: float = 24.0,
        seed: int = 42,
    ):
        self.beds = beds
        self.speedup = speedup
        self.hours = hours

        self.on_connect = None
        self.on_message = None
        self.on_disconnect = None

        self._subscriptions: list[str] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._connected = False

        # Build the ward up front so every run is reproducible.
        gen = VitalsGenerator(seed=seed, sample_minutes=5)
        self._patients = []
        for i in range(beds):
            patient = gen.make_patient(i)
            frame = gen.generate_patient(patient, hours=hours)
            self._patients.append(
                {
                    "bed_id": f"ICU-{i + 1:02d}",
                    "patient": patient,
                    "frame": frame,
                    "cursor": 0,
                }
            )
        self._rng = np.random.default_rng(seed)

    # -- connection --------------------------------------------------------
    def connect(self, host: str = "simulated", port: int = 1883, keepalive: int = 60):
        """Connect to the simulated ward. Signature mirrors paho's."""
        self._connected = True
        if self.on_connect:
            self.on_connect(self, None, {}, 0)  # rc=0 means success, as in MQTT
        return 0

    def subscribe(self, topic: str, qos: int = 0):
        self._subscriptions.append(topic)
        return (0, 1)

    def unsubscribe(self, topic: str):
        self._subscriptions = [t for t in self._subscriptions if t != topic]
        return (0, 1)

    def disconnect(self):
        self.loop_stop()
        self._connected = False
        if self.on_disconnect:
            self.on_disconnect(self, None, 0)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- the publishing loop ----------------------------------------------
    def loop_start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def loop_stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _deliver(self, topic: str, payload: dict):
        if not any(_topic_matches(s, topic) for s in self._subscriptions):
            return
        if self.on_message:
            self.on_message(
                self, None, SimulatedMessage(topic, json.dumps(payload).encode("utf-8"))
            )

    def _run(self):
        # Each tick advances every bed by one 5-minute sample.
        interval = (5 * 60) / self.speedup
        while not self._stop.is_set():
            for bed in self._patients:
                frame = bed["frame"]
                i = bed["cursor"] % len(frame)
                row = frame.iloc[i]
                bed["cursor"] += 1

                vitals = {
                    "bed_id": bed["bed_id"],
                    "timestamp": row["timestamp"].isoformat(),
                    "minutes_since_admission": int(row["minutes_since_admission"]),
                    **{ch: round(float(row[ch]), 2) for ch in CHANNELS},
                }
                self._deliver(f"ward/bed/{bed['bed_id']}/vitals", vitals)

                # Device and alarm status is published less often, as on a real ward.
                if bed["cursor"] % 6 == 0:
                    alarms = []
                    if row["spo2"] < 92:
                        alarms.append("SPO2_LOW")
                    if row["heart_rate"] > 120:
                        alarms.append("HR_HIGH")
                    if row["map_mmhg"] < 65:
                        alarms.append("MAP_LOW")
                    if row["resp_rate"] > 25:
                        alarms.append("RR_HIGH")
                    status = {
                        "bed_id": bed["bed_id"],
                        "timestamp": row["timestamp"].isoformat(),
                        "acuity": int(row["acuity"]),
                        "age": int(row["age"]),
                        "sex": str(row["sex"]),
                        "battery_pct": round(
                            float(100 - (bed["cursor"] % 400) * 0.15), 1
                        ),
                        "lead_quality": round(
                            float(np.clip(self._rng.normal(0.93, 0.05), 0.4, 1.0)), 2
                        ),
                        "alarms": alarms,
                    }
                    self._deliver(f"ward/bed/{bed['bed_id']}/status", status)

            self._stop.wait(interval)


# --------------------------------------------------------------------------
# Offline fallback, mirroring the notebook's original sample-data path
# --------------------------------------------------------------------------


def load_sample_vitals(beds: int = 4, hours: float = 6.0, seed: int = 7):
    """Return a ready-made vitals DataFrame without running the stream at all."""
    gen = VitalsGenerator(seed=seed, sample_minutes=5)
    frames = []
    for i in range(beds):
        patient = gen.make_patient(i)
        frame = gen.generate_patient(patient, hours=hours)
        frame = frame.assign(bed_id=f"ICU-{i + 1:02d}")
        frames.append(frame)
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    received = []
    client = SimulatedMonitorClient(beds=3, speedup=6000)
    client.on_connect = lambda c, u, f, rc: print(f"connected (rc={rc})")
    client.on_message = lambda c, u, msg: received.append(
        (msg.topic, json.loads(msg.payload))
    )
    client.connect()
    client.subscribe("ward/bed/+/vitals")
    client.subscribe("ward/bed/+/status")
    client.loop_start()
    time.sleep(2.0)
    client.loop_stop()
    client.disconnect()

    print(f"received {len(received)} messages")
    for topic, payload in received[:3]:
        print(" ", topic, json.dumps(payload)[:110])
