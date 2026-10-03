"""Generate the synthetic hospital encounters dataset used by the data-exploration
notebook.

This replaces the AIS vessel-movement file the course previously explored. The point of
that notebook is data-quality assessment, so this generator deliberately injects the
problems a real extract from a hospital system actually has:

  * missing values, at different rates in different columns
  * a sentinel string ("Unknown") standing in for a null in a categorical column
  * physiologically impossible values (a heart rate of 0, a weight of 0)
  * inconsistent category casing from different feeder systems
  * exact duplicate rows from a botched extract join
  * a handful of extreme outliers

None of this is accidental. A learner is supposed to find it.

Run:
    .venv.nosync/bin/python generate_encounters_dataset.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_PATIENTS = 4000
OBS_PER_PATIENT = (4, 30)  # each patient contributes several observation rows
SEED = 42
OUT = "data/encounters/hospital_encounters.csv"

ADMISSION_TYPES = {
    "Emergency": 0.46,
    "Elective": 0.24,
    "Day Case": 0.15,
    "Maternity": 0.08,
    "Transfer": 0.07,
}

ENCOUNTER_STATUS = {
    "Under Observation": 0.38,
    "Admitted": 0.28,
    "Discharged": 0.21,
    "Transferred": 0.09,
    "Deceased": 0.04,
}


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    rows = []
    for pid in range(N_PATIENTS):
        patient_id = 100000 + pid * 7 + int(rng.integers(0, 6))
        n_obs = int(rng.integers(*OBS_PER_PATIENT))

        admission_type = rng.choice(
            list(ADMISSION_TYPES), p=list(ADMISSION_TYPES.values())
        )
        status = rng.choice(list(ENCOUNTER_STATUS), p=list(ENCOUNTER_STATUS.values()))

        age = int(np.clip(rng.normal(62, 19), 0, 104))
        sex = rng.choice(["F", "M"])

        # Anthropometrics are per-patient, not per-observation
        height_cm = float(np.clip(rng.normal(170 if sex == "M" else 162, 8), 140, 201))
        weight_kg = float(np.clip(rng.normal(82 if sex == "M" else 71, 16), 38, 180))
        bmi = weight_kg / (height_cm / 100) ** 2

        # Baselines vary by patient; sicker statuses run higher heart rates
        sickness = {"Under Observation": 0.4, "Admitted": 0.8, "Discharged": 0.1,
                    "Transferred": 1.0, "Deceased": 1.4}[status]
        hr_base = rng.normal(76 + 12 * sickness, 9)
        sbp_base = rng.normal(130 - 8 * sickness, 15)

        for _ in range(n_obs):
            heart_rate = float(np.clip(rng.normal(hr_base, 6), 35, 190))
            systolic_bp = float(np.clip(rng.normal(sbp_base, 9), 70, 210))
            # Diastolic tracks systolic, which gives the notebook a genuine
            # two-related-measures relationship to explore.
            diastolic_bp = float(
                np.clip(systolic_bp * 0.63 + rng.normal(0, 6), 40, 130)
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "encounter_status": status,
                    "admission_type": admission_type,
                    "age": age,
                    "sex": sex,
                    "heart_rate": round(heart_rate, 1),
                    "systolic_bp": round(systolic_bp, 1),
                    "diastolic_bp": round(diastolic_bp, 1),
                    "height_cm": round(height_cm, 1),
                    "weight_kg": round(weight_kg, 1),
                    "bmi": round(bmi, 1),
                }
            )

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"clean rows: {n}")

    # ---------------------------------------------------------------- defects
    # 1. Missing values at different rates per column, as a real extract has.
    for col, rate in [("bmi", 0.11), ("weight_kg", 0.07), ("height_cm", 0.07),
                      ("diastolic_bp", 0.04), ("age", 0.01)]:
        idx = rng.choice(n, int(n * rate), replace=False)
        df.loc[idx, col] = np.nan

    # 2. A sentinel string instead of a null, the classic categorical trap.
    idx = rng.choice(n, int(n * 0.06), replace=False)
    df.loc[idx, "encounter_status"] = "Unknown"

    # 3. Physiologically impossible values: a disconnected monitor writes 0,
    #    and an unrecorded weight is stored as 0 rather than left blank.
    idx = rng.choice(n, int(n * 0.015), replace=False)
    df.loc[idx, "heart_rate"] = 0.0
    idx = rng.choice(n, int(n * 0.012), replace=False)
    df.loc[idx, ["systolic_bp", "diastolic_bp"]] = 0.0
    idx = rng.choice(n, int(n * 0.008), replace=False)
    df.loc[idx, "weight_kg"] = 0.0

    # 4. Inconsistent casing from different feeder systems.
    idx = rng.choice(n, int(n * 0.05), replace=False)
    df.loc[idx, "admission_type"] = df.loc[idx, "admission_type"].str.upper()
    idx = rng.choice(n, int(n * 0.03), replace=False)
    df.loc[idx, "admission_type"] = df.loc[idx, "admission_type"].str.lower()

    # 5. Extreme outliers: transcription slips.
    idx = rng.choice(n, 40, replace=False)
    df.loc[idx, "heart_rate"] = rng.uniform(250, 400, 40).round(1)
    idx = rng.choice(n, 25, replace=False)
    df.loc[idx, "weight_kg"] = rng.uniform(350, 600, 25).round(1)
    idx = rng.choice(n, 30, replace=False)
    df.loc[idx, "age"] = rng.integers(150, 999, 30)

    # 6. Exact duplicate rows from a bad join in the extract.
    dupes = df.sample(int(n * 0.02), random_state=SEED)
    df = pd.concat([df, dupes], ignore_index=True)

    # Shuffle so the defects are not clustered at the end.
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import os

    os.makedirs("data/encounters", exist_ok=True)
    df = build()
    df.to_csv(OUT, index=False)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT}  rows={len(df)}  cols={len(df.columns)}  {size_mb:.1f} MB")
    print()
    print("Injected defects a learner should find:")
    print(f"  missing bmi            {df['bmi'].isna().mean():.1%}")
    print(f"  'Unknown' status       {(df['encounter_status'] == 'Unknown').mean():.1%}")
    print(f"  heart_rate == 0        {(df['heart_rate'] == 0).mean():.1%}")
    print(f"  heart_rate > 250       {(df['heart_rate'] > 250).sum()} rows")
    print(f"  weight_kg == 0         {(df['weight_kg'] == 0).mean():.1%}")
    print(f"  age > 120              {(df['age'] > 120).sum()} rows")
    print(f"  duplicate rows         {df.duplicated().sum()}")
    print(f"  admission_type variants {sorted(df['admission_type'].unique())[:8]}")
