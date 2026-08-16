"""
AquaGuard AI — Synthetic Pipeline Data Generator
--------------------------------------------------
Simulates an underground water pipeline network with 5 sensor stations
(S1 -> S2 -> S3 -> S4 -> S5), giving 4 monitored segments:
    SEG1: S1-S2, SEG2: S2-S3, SEG3: S3-S4, SEG4: S4-S5

Physics logic (simplified, but internally consistent):
  - Baseline flow enters at S1 and, under NORMAL conditions, decays only
    slightly segment-to-segment due to friction/consumption.
  - Pressure drops along the pipe due to friction losses (Darcy-Weisbach
    style approximation): dP = k * Q^1.85 (Hazen-Williams-like exponent).
  - A LEAK at a segment causes:
      * An extra, abrupt pressure drop across that segment (proportional
        to leak severity).
      * A flow imbalance: flow_out < flow_in for that segment (the
        difference IS the water loss, in L/min).
  - Severity classes: none, small, medium, severe -> drive the magnitude
    of the extra pressure drop and flow loss.

Output: data/pipeline_dataset.csv
Each row = one reading, for one segment, at one timestamp.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

SEGMENTS = ["SEG1", "SEG2", "SEG3", "SEG4"]
STATIONS = {"SEG1": ("S1", "S2"), "SEG2": ("S2", "S3"),
            "SEG3": ("S3", "S4"), "SEG4": ("S4", "S5")}

# Pipe physical properties per segment (diameter in inches, length in meters)
PIPE_PROPS = {
    "SEG1": {"diameter": 8.0, "length": 250},
    "SEG2": {"diameter": 6.0, "length": 300},
    "SEG3": {"diameter": 6.0, "length": 275},
    "SEG4": {"diameter": 4.0, "length": 200},
}

SEVERITY_LEVELS = ["none", "small", "medium", "severe"]
# extra pressure drop (psi) and flow loss (L/min) ranges per severity
SEVERITY_PROFILE = {
    "none":   {"dp_extra": (0.0, 0.3),  "loss": (0.0, 0.15)},
    "small":  {"dp_extra": (1.0, 3.0),  "loss": (0.5, 2.5)},
    "medium": {"dp_extra": (3.0, 7.0),  "loss": (2.5, 6.0)},
    "severe": {"dp_extra": (7.0, 15.0), "loss": (6.0, 15.0)},
}

N_TIMESTAMPS = 4000        # readings per segment
LEAK_EVENT_PROB = 0.22     # probability a given timestamp/segment has a leak


def friction_pressure_drop(flow, diameter, length, noise=True):
    """Simplified Hazen-Williams-like friction loss."""
    k = 4.5 / (diameter ** 1.5)
    dp = k * (flow ** 1.85) * (length / 250.0)
    if noise:
        dp += RNG.normal(0, 0.15)
    return max(dp, 0.05)


def generate_segment_data(segment, n=N_TIMESTAMPS):
    props = PIPE_PROPS[segment]
    rows = []
    base_flow = RNG.uniform(4.5, 6.5)  # L/s baseline for this segment's inlet

    for t in range(n):
        # gentle diurnal-style demand variation
        time_factor = 1 + 0.15 * np.sin(2 * np.pi * (t % 288) / 288)
        flow_upstream = max(base_flow * time_factor + RNG.normal(0, 0.2), 0.5)

        is_leak = RNG.random() < LEAK_EVENT_PROB
        severity = "none"
        if is_leak:
            severity = RNG.choice(["small", "medium", "severe"], p=[0.55, 0.3, 0.15])

        prof = SEVERITY_PROFILE[severity]
        water_loss = RNG.uniform(*prof["loss"])
        flow_downstream = max(flow_upstream - water_loss / 60.0, 0.05)  # loss L/min -> L/s

        pressure_upstream = 55 + RNG.normal(0, 1.0)
        dp_friction = friction_pressure_drop(flow_upstream, props["diameter"], props["length"])
        dp_extra = RNG.uniform(*prof["dp_extra"])
        pressure_downstream = pressure_upstream - dp_friction - dp_extra

        vibration = RNG.normal(0.2, 0.05) + (dp_extra * 0.02 if is_leak else 0)
        acoustic_level = RNG.normal(30, 2) + (dp_extra * 1.2 if is_leak else 0)  # dB-like proxy

        rows.append({
            "timestamp": t,
            "segment": segment,
            "station_upstream": STATIONS[segment][0],
            "station_downstream": STATIONS[segment][1],
            "pipe_diameter_in": props["diameter"],
            "pipe_length_m": props["length"],
            "pressure_upstream_psi": round(pressure_upstream, 2),
            "pressure_downstream_psi": round(pressure_downstream, 2),
            "pressure_drop_psi": round(pressure_upstream - pressure_downstream, 2),
            "flow_upstream_lps": round(flow_upstream, 3),
            "flow_downstream_lps": round(flow_downstream, 3),
            "flow_diff_lps": round(flow_upstream - flow_downstream, 3),
            "vibration_g": round(max(vibration, 0), 3),
            "acoustic_db": round(max(acoustic_level, 0), 2),
            "is_leak": int(is_leak),
            "severity": severity,
            "water_loss_lpm": round(water_loss, 3),
        })
    return pd.DataFrame(rows)


def generate_full_dataset():
    dfs = [generate_segment_data(seg) for seg in SEGMENTS]
    full = pd.concat(dfs, ignore_index=True)
    full = full.sample(frac=1, random_state=42).reset_index(drop=True)
    return full


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    df = generate_full_dataset()
    out_path = out_dir / "pipeline_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["severity"].value_counts())
    print(df["segment"].value_counts())
