"""Analyze hover and flight oscillations from flight log."""
import csv
from pathlib import Path

log = Path(r"E:\CAROLLINE_PROJECT\carolline_control\logs\flight_log.csv")
rows = list(csv.DictReader(open(log)))

for mode in ("HOVER", "FLIGHT", "ROLLING"):
    seg = [r for r in rows if r["mode"] == mode]
    if not seg:
        continue
    px = [float(r["px"]) for r in seg]
    py = [float(r["py"]) for r in seg]
    pz = [float(r["pz"]) for r in seg]
    vx = [float(r["vx"]) for r in seg]
    vy = [float(r["vy"]) for r in seg]
    vz = [float(r["vz"]) for r in seg]
    print(f"\n=== {mode} n={len(seg)} t={seg[0]['time'][:6]}-{seg[-1]['time'][:6]} ===")
    print(f"  px range {min(px):.3f} .. {max(px):.3f}  std {(__import__('numpy').std(px)):.3f}")
    print(f"  py range {min(py):.3f} .. {max(py):.3f}  std {(__import__('numpy').std(py)):.3f}")
    print(f"  pz range {min(pz):.3f} .. {max(pz):.3f}  std {(__import__('numpy').std(pz)):.3f}")
    print(f"  |v| max {max((vx[i]**2+vy[i]**2+vz[i]**2)**0.5 for i in range(len(seg))):.3f}")
    if mode == "ROLLING":
        qw = [float(r.get("qw", 1)) for r in seg if "qw" in r]

# min z overall
all_z = [float(r["pz"]) for r in rows]
print(f"\nGlobal pz min {min(all_z):.3f} (cage bottom ~ {min(all_z)-0.4:.3f})")
