"""
Synthetic bedside-monitor vitals generator.

Produces realistic multi-channel ICU/ward vital-sign trajectories for the
machine-learning course. Replaces the former synthetic AIS vessel-track
generator: where that produced (latitude, longitude, speed, course) along a
shipping route, this produces (heart rate, mean arterial pressure, SpO2,
respiratory rate, temperature) along a patient's stay.

No real patient data is used or required. Everything here is simulated from
published physiological ranges, so the notebooks carry no PHI.

Typical use
-----------
    from synthetic_vitals_generator import VitalsGenerator

    gen = VitalsGenerator(seed=42)
    df = gen.generate_cohort(n_patients=60, hours=24)
    df.head()

Channels
--------
heart_rate      beats per minute
map_mmhg        mean arterial pressure, mmHg
spo2            peripheral oxygen saturation, percent
resp_rate       breaths per minute
temp_c          core temperature, degrees Celsius

Each patient follows one clinical archetype. The archetype drives how the
channels move over the stay, the same way a vessel's route drove its track.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Clinical archetypes
# --------------------------------------------------------------------------
# Each archetype names a plausible course of a stay. `prevalence` is the share
# of a generated cohort; `acuity` is the triage level a clinician would assign
# at admission (1 = least acute, 4 = most acute).

ARCHETYPES = {
    "stable": dict(
        prevalence=0.40,
        acuity=1,
        label="Stable",
        description="Vitals hold near the patient's own baseline for the whole stay.",
    ),
    "post_op_recovery": dict(
        prevalence=0.25,
        acuity=2,
        label="Post-operative recovery",
        description="Starts perturbed after surgery, then settles back toward baseline.",
    ),
    "sepsis_deterioration": dict(
        prevalence=0.20,
        acuity=3,
        label="Sepsis deterioration",
        description="A slow early drift becomes a sharp decline: fever, tachycardia, falling pressure.",
    ),
    "cardiac_event": dict(
        prevalence=0.15,
        acuity=4,
        label="Cardiac event",
        description="Long quiet period, then an abrupt, short-lived crisis.",
    ),
}

CHANNELS = ["heart_rate", "map_mmhg", "spo2", "resp_rate", "temp_c"]

# Population baselines: (mean, between-patient sd, within-patient minute-to-minute sd)
_BASELINE = {
    "heart_rate": (78.0, 9.0, 2.2),
    "map_mmhg": (88.0, 8.0, 2.0),
    "spo2": (97.0, 1.2, 0.45),
    "resp_rate": (16.0, 2.2, 0.9),
    "temp_c": (36.8, 0.30, 0.06),
}

# Physiological limits used to clip the simulated signal.
_LIMITS = {
    "heart_rate": (30.0, 200.0),
    "map_mmhg": (35.0, 140.0),
    "spo2": (70.0, 100.0),
    "resp_rate": (6.0, 45.0),
    "temp_c": (34.0, 41.5),
}


@dataclass
class Patient:
    """One simulated patient: their baseline, archetype and event timing."""

    patient_id: str
    archetype: str
    age: int
    sex: str
    acuity: int
    baseline: dict = field(default_factory=dict)
    onset_frac: float = 0.5  # where in the stay the clinical event begins


class VitalsGenerator:
    """Generates bedside-monitor trajectories for a cohort of patients.

    Parameters
    ----------
    seed : int
        Random seed, so a notebook run is reproducible.
    sample_minutes : int
        Minutes between monitor samples. Bedside monitors chart every
        5 minutes in most units, which is the default.
    """

    def __init__(self, seed: int = 42, sample_minutes: int = 5):
        self.rng = np.random.default_rng(seed)
        self.sample_minutes = sample_minutes

    # -- patients ----------------------------------------------------------
    def _draw_baseline(self) -> dict:
        """Draw one patient's personal resting values."""
        out = {}
        for ch, (mu, between_sd, _) in _BASELINE.items():
            lo, hi = _LIMITS[ch]
            out[ch] = float(np.clip(self.rng.normal(mu, between_sd), lo, hi))
        return out

    def make_patient(self, index: int, archetype: str | None = None) -> Patient:
        """Create one patient, optionally forcing a clinical archetype."""
        if archetype is None:
            names = list(ARCHETYPES)
            weights = np.array([ARCHETYPES[a]["prevalence"] for a in names])
            archetype = str(self.rng.choice(names, p=weights / weights.sum()))

        age = int(np.clip(self.rng.normal(64, 16), 18, 98))
        sex = str(self.rng.choice(["F", "M"]))
        base = self._draw_baseline()

        # Older patients sit a little lower on pressure and saturation.
        base["map_mmhg"] -= (age - 64) * 0.12
        base["spo2"] -= max(0.0, (age - 70)) * 0.02

        return Patient(
            patient_id=f"P{index:04d}",
            archetype=archetype,
            age=age,
            sex=sex,
            acuity=int(ARCHETYPES[archetype]["acuity"]),
            baseline=base,
            onset_frac=float(self.rng.uniform(0.35, 0.70)),
        )

    # -- event shapes ------------------------------------------------------
    @staticmethod
    def _ramp(t: np.ndarray, start: float, width: float) -> np.ndarray:
        """A smooth 0 -> 1 ramp beginning at `start`, reaching 1 after `width`."""
        x = np.clip((t - start) / max(width, 1e-6), 0.0, 1.0)
        return x * x * (3 - 2 * x)  # smoothstep

    @staticmethod
    def _pulse(t: np.ndarray, centre: float, width: float) -> np.ndarray:
        """A short symmetric excursion centred on `centre`."""
        return np.exp(-0.5 * ((t - centre) / max(width, 1e-6)) ** 2)

    def _effect(self, archetype: str, t: np.ndarray, onset: float) -> dict:
        """Per-channel offset from baseline over normalized stay time `t` (0-1)."""
        z = {ch: np.zeros_like(t) for ch in CHANNELS}

        if archetype == "stable":
            return z

        if archetype == "post_op_recovery":
            # Perturbed at admission, decaying back to baseline.
            decay = np.exp(-t / 0.28)
            z["heart_rate"] += 20.0 * decay
            z["map_mmhg"] -= 8.0 * decay
            z["resp_rate"] += 4.0 * decay
            z["temp_c"] += 0.45 * decay
            z["spo2"] -= 1.6 * decay

        elif archetype == "sepsis_deterioration":
            # A gentle early drift that steepens after onset.
            drift = self._ramp(t, 0.10, 0.55)
            crisis = self._ramp(t, onset, 0.22)
            z["heart_rate"] += 16.0 * drift + 30.0 * crisis
            z["map_mmhg"] -= 6.0 * drift + 24.0 * crisis
            z["temp_c"] += 0.8 * drift + 0.9 * crisis
            z["resp_rate"] += 4.0 * drift + 9.0 * crisis
            z["spo2"] -= 1.0 * drift + 4.0 * crisis

        elif archetype == "cardiac_event":
            # Quiet, then an abrupt short crisis with a partial recovery.
            spike = self._pulse(t, onset, 0.045)
            after = self._ramp(t, onset, 0.10)
            z["heart_rate"] += 48.0 * spike + 10.0 * after
            z["map_mmhg"] -= 30.0 * spike + 6.0 * after
            z["spo2"] -= 7.0 * spike + 1.2 * after
            z["resp_rate"] += 11.0 * spike + 2.5 * after
            z["temp_c"] += 0.15 * spike

        return z

    # -- trajectories ------------------------------------------------------
    def generate_patient(self, patient: Patient, hours: float = 24.0) -> pd.DataFrame:
        """Generate one patient's monitor trace."""
        n = max(int(hours * 60 / self.sample_minutes), 8)
        minutes = np.arange(n) * self.sample_minutes
        t = minutes / minutes[-1]  # normalized stay time, 0 to 1

        effects = self._effect(patient.archetype, t, patient.onset_frac)
        # Circadian component: a gentle 24-hour rhythm on rate and temperature.
        clock = 2 * np.pi * (minutes / (24 * 60))
        circadian = {
            "heart_rate": 3.5 * np.sin(clock - 0.6),
            "map_mmhg": 2.5 * np.sin(clock - 0.3),
            "spo2": 0.25 * np.sin(clock),
            "resp_rate": 0.8 * np.sin(clock - 0.4),
            "temp_c": 0.22 * np.sin(clock - 1.1),
        }

        data = {"minutes_since_admission": minutes}
        for ch in CHANNELS:
            _, _, noise_sd = _BASELINE[ch]
            # Correlated (red) noise: monitors drift, they do not jitter.
            white = self.rng.normal(0.0, noise_sd, n)
            red = np.zeros(n)
            for i in range(1, n):
                red[i] = 0.72 * red[i - 1] + white[i]
            series = patient.baseline[ch] + effects[ch] + circadian[ch] + red
            lo, hi = _LIMITS[ch]
            data[ch] = np.clip(series, lo, hi)

        df = pd.DataFrame(data)
        df.insert(0, "patient_id", patient.patient_id)
        df["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(
            df["minutes_since_admission"], unit="m"
        )
        df["archetype"] = patient.archetype
        df["acuity"] = patient.acuity
        df["age"] = patient.age
        df["sex"] = patient.sex
        # Label the deteriorating window, for notebooks that need a target.
        onset_minute = patient.onset_frac * minutes[-1]
        df["deteriorating"] = (
            (df["minutes_since_admission"] >= onset_minute)
            & (patient.archetype in ("sepsis_deterioration", "cardiac_event"))
        ).astype(int)
        return df

    def generate_cohort(
        self,
        n_patients: int = 60,
        hours: float = 24.0,
        archetypes: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate a cohort of patients as one long (tidy) DataFrame."""
        frames = []
        for i in range(n_patients):
            arch = None
            if archetypes:
                arch = archetypes[i % len(archetypes)]
            patient = self.make_patient(i, archetype=arch)
            frames.append(self.generate_patient(patient, hours=hours))
        return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------
def generate_vitals_cohort(
    n_patients: int = 60, hours: float = 24.0, seed: int = 42
) -> pd.DataFrame:
    """One-call cohort generation, for notebooks that want a single import."""
    return VitalsGenerator(seed=seed).generate_cohort(n_patients=n_patients, hours=hours)


def to_sequences(
    df: pd.DataFrame,
    context: int = 128,
    horizon: int = 32,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a tidy cohort frame into supervised (context, horizon) windows.

    Returns arrays shaped (n_windows, context, n_channels) and
    (n_windows, horizon, n_channels). Windows never straddle two patients.
    """
    channels = channels or CHANNELS
    xs, ys = [], []
    for _, g in df.groupby("patient_id", sort=False):
        arr = g[channels].to_numpy(dtype=float)
        limit = len(arr) - context - horizon + 1
        for start in range(0, max(limit, 0)):
            xs.append(arr[start : start + context])
            ys.append(arr[start + context : start + context + horizon])
    if not xs:
        return np.empty((0, context, len(channels))), np.empty((0, horizon, len(channels)))
    return np.asarray(xs), np.asarray(ys)


if __name__ == "__main__":
    gen = VitalsGenerator(seed=7)
    cohort = gen.generate_cohort(n_patients=8, hours=12)
    print(cohort.head(3).to_string(index=False))
    print()
    print("rows:", len(cohort), " patients:", cohort['patient_id'].nunique())
    print(cohort.groupby("archetype")[CHANNELS].mean().round(1).to_string())
