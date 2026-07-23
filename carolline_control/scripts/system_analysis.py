"""Analyze upright recovery and flight thrust/tilt behavior from flight log."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from carolline_control.utils.so3 import body_z_world, quat_to_rot

LOG = Path(__file__).resolve().parent.parent / "logs" / "flight_log.csv"


def body_z_from_quat(qw, qx, qy, qz):
    return float(body_z_world(quat_to_rot(np.array([qw, qx, qy, qz])))[2])


def main() -> None:
    rows = list(csv.DictReader(open(LOG, encoding="utf-8")))
    print("Modes:", Counter(r["mode"] for r in rows))
    print()

    # Mode transitions
    prev = None
    for r in rows:
        if r["mode"] != prev:
            bz = body_z_from_quat(float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"]))
            print(
                f"t={float(r['time']):6.2f} {prev} -> {r['mode']:10s} "
                f"pos=({float(r['px']):.2f},{float(r['py']):.2f},{float(r['pz']):.2f}) "
                f"bz={bz:.3f} thrust={float(r['thrust']):.1f}"
            )
            prev = r["mode"]

    # Upright recovery analysis
    for mode in ("PRETAKEOFF", "UPRIGHT", "TAKEOFF"):
        seg = [r for r in rows if r["mode"] == mode]
        if not seg:
            continue
        bz = [body_z_from_quat(float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])) for r in seg]
        print(f"\n=== {mode} n={len(seg)} duration={float(seg[-1]['time'])-float(seg[0]['time']):.2f}s ===")
        print(f"  body_z: min={min(bz):.3f} max={max(bz):.3f} final={bz[-1]:.3f}")
        print(f"  pz: {min(float(r['pz']) for r in seg):.3f} .. {max(float(r['pz']) for r in seg):.3f}")
        if mode == "TAKEOFF" and seg:
            first = seg[0]
            bz0 = body_z_from_quat(
                float(first["qw"]), float(first["qx"]), float(first["qy"]), float(first["qz"])
            )
            print(f"  takeoff start bz={bz0:.3f} thrust={float(first['thrust']):.1f} N")

    # FLIGHT thrust/tilt during active legs (exclude near-hover segments)
    flight = [r for r in rows if r["mode"] == "FLIGHT"]
    thrust = np.array([float(r["thrust"]) for r in flight])
    motors = np.array([[float(r[f"m{i}"]) for i in range(1, 5)] for r in flight])
    bz = np.array(
        [body_z_from_quat(float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])) for r in flight]
    )
    des_v = np.array(
        [
            np.sqrt(
                (float(r["des_px"]) - float(rows[max(0, i - 1)]["des_px"])) ** 2
                + (float(r["des_py"]) - float(rows[max(0, i - 1)]["des_py"])) ** 2
            )
            / 0.004
            for i, r in enumerate(flight)
        ]
    )
    print(f"\n=== FLIGHT thrust/tilt ===")
    print(f"  collective thrust: mean={thrust.mean():.2f} std={thrust.std():.3f} range={thrust.min():.2f}..{thrust.max():.2f}")
    print(f"  motor spread (max-min): mean={(motors.max(axis=1)-motors.min(axis=1)).mean():.3f} max={(motors.max(axis=1)-motors.min(axis=1)).max():.3f}")
    print(f"  body_z: mean={bz.mean():.3f} min={bz.min():.3f}")

    # Per 6s segment during flight
    t0 = float(flight[0]["time"])
    seg_dur = 6.0
    idx = 0
    seg_i = 0
    while idx < len(flight):
        seg = []
        t_start = float(flight[idx]["time"])
        while idx < len(flight) and float(flight[idx]["time"]) - t_start < seg_dur:
            seg.append(flight[idx])
            idx += 1
        if len(seg) < 50:
            continue
        th = np.array([float(r["thrust"]) for r in seg])
        m = np.array([[float(r[f"m{i}"]) for i in range(1, 5)] for r in seg])
        px = np.array([float(r["px"]) for r in seg])
        py = np.array([float(r["py"]) for r in seg])
        des_px = float(seg[-1]["des_px"])
        des_py = float(seg[-1]["des_py"])
        travel = np.sqrt((px[-1] - px[0]) ** 2 + (py[-1] - py[0]) ** 2)
        print(
            f"\n  Seg {seg_i} t={t_start:.1f}s goal=({des_px:.1f},{des_py:.1f}) "
            f"travel={travel:.2f}m thrust_std={th.std():.3f} motor_spread_mean={(m.max(axis=1)-m.min(axis=1)).mean():.2f}"
        )
        seg_i += 1


if __name__ == "__main__":
    main()
