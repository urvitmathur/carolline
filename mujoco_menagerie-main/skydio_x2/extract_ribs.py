"""Flatten rib capsule geoms into a single attachable body."""
from pathlib import Path
import re

root = Path(__file__).resolve().parent
text = (root / "scene_cage_legacy.xml").read_text(encoding="utf-16")
geoms = re.findall(r'<geom class="cage" fromto="[^"]+"\s*/>', text)
geoms = [g.replace('class="cage"', 'class="cage_rib"') for g in geoms]
out = """<mujoco>
  <body name="cage_ribs">
"""
out += "\n".join(f"    {g}" for g in geoms)
out += """
  </body>
</mujoco>
"""
(root / "cage_ribs.xml").write_text(out, encoding="utf-8")
print(f"Wrote {len(geoms)} rib geoms")
