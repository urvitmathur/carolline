"""Quick flight log analysis."""
import csv
import sys
from pathlib import Path

log = Path(__file__).resolve().parents[1] / "logs" / "flight_log.csv"
if len(sys.argv) > 1:
    log = Path(sys.argv[1])

rows = list(csv.DictReader(open(log)))
roll = [r for r in rows if r["mode"] == "ROLLING"]
print("rolling duration", roll[-1]["time"], "n", len(roll))
print("start", roll[0]["px"], roll[0]["py"])
print("end", roll[-1]["px"], roll[-1]["py"])
for i in range(0, len(roll), max(1, len(roll) // 10)):
    r = roll[i]
    m = [float(r["m1"]), float(r["m2"]), float(r["m3"]), float(r["m4"])]
    print(
        "t", r["time"][:6],
        "pos", f"{float(r['px']):.3f}", f"{float(r['py']):.3f}",
        "v", f"{float(r['vx']):.3f}", f"{float(r['vy']):.3f}",
        "T", r["thrust"][:6],
        "m", [f"{x:.2f}" for x in m],
        "sum", sum(m),
    )
jumps = 0
prev = None
for r in roll:
    s = sum(float(r[f"m{i}"]) for i in range(1, 5))
    if prev is not None and abs(s - prev) > 3:
        jumps += 1
    prev = s
print("collective jumps >3N", jumps)
dmax = 0.0
for i in range(1, len(roll)):
    for j in range(1, 5):
        d = abs(float(roll[i][f"m{j}"]) - float(roll[i - 1][f"m{j}"]))
        dmax = max(dmax, d)
print("max single motor step", dmax)
modes = {}
for r in rows:
    modes[r["mode"]] = modes.get(r["mode"], 0) + 1
print("mode counts", modes)
print("final mode", rows[-1]["mode"])
