# Copyright 2026 The Physics-Next Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compare cumulative SDF scheduling switches in isolated temporary package copies."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--worlds", type=int, default=4096)
  parser.add_argument("--steps", type=int, default=100)
  parser.add_argument("--variant", choices=("baseline", "compact", "grid", "all"), help=argparse.SUPPRESS)
  parser.add_argument("--empty", action="store_true", help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.worlds < 1 or args.steps < 1:
    parser.error("worlds and steps must be positive")
  if args.variant is None:
    root = Path(__file__).resolve().parents[1]
    print("Cumulative switches: baseline -> compact pairs -> fixed grid -> skip empty")
    print("CUDA graph collision only; fixed state; 20 warmups; synchronized timing; compile/capture excluded")
    for empty in (False, True):
      print(f"\n{'No SDF candidates' if empty else 'Mixed primitive/SDF contacts'}; worlds={args.worlds}; steps={args.steps}")
      previous = None
      expected_contacts = None
      for variant in ("baseline", "compact", "grid", "all"):
        with tempfile.TemporaryDirectory(prefix="mjwarp-sdf-") as directory:
          package = Path(directory) / "mujoco_warp"
          shutil.copytree(root / "mujoco_warp", package, ignore=shutil.ignore_patterns("__pycache__"))
          path = package / "_src/collision_sdf.py"
          source = path.read_text()
          if variant != "all":
            source = source.replace("if m.opt.graph_conditional:", "if False:")
          if variant in ("baseline", "compact"):
            start = source.index("  workid = wp.tid()")
            body_start = source.index("    geoms = collision_pair_in[contact_tid]", start)
            end = source.index("\n\n@event_scope", body_start)
            body = "\n".join(line[2:] if line.startswith("  ") else line for line in source[body_start:end].split("\n"))
            body = body.replace("    continue", "    return")
            prefix = "  i, contact_tid = wp.tid()\n  if contact_tid >= nsdfcollision_in[0]:\n    return\n"
            if variant == "compact":
              prefix += "  contact_tid = sdf_collision_index_in[contact_tid]\n"
            else:
              body = body.replace("  type2 = geom_type[g2]", "  type2 = geom_type[g2]\n  if type2 != GeomType.SDF:\n    return")
              source = source.replace("ctx.nsdfcollision,", "d.ncollision,")
            source = source[:start] + prefix + body + source[end:]
            source = source.replace("dim=_SDF_NARROWPHASE_THREADS", "dim=(m.opt.sdf_initpoints, d.naconmax)")
          path.write_text(source)
          if variant == "baseline":
            path = package / "_src/collision_driver.py"
            path.write_text(path.read_text().replace("if type1 == GeomType.SDF or type2 == GeomType.SDF:", "if False:"))
          environment = os.environ.copy()
          environment["PYTHONPATH"] = directory
          command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--variant",
            variant,
            "--worlds",
            str(args.worlds),
            "--steps",
            str(args.steps),
          ]
          if empty:
            command.append("--empty")
          result = subprocess.run(command, env=environment, text=True, capture_output=True)
          if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
          metrics = json.loads(result.stdout.splitlines()[-1])
          if expected_contacts is None:
            expected_contacts = metrics["contacts"]
            print(f"{metrics['device']}; MuJoCo {metrics['mujoco']}; Warp {metrics['warp']}")
          assert metrics["contacts"] == expected_contacts, "Scheduling changed the contact count"
          milliseconds = metrics["milliseconds"]
          speedup = "-" if previous is None else f"{previous / milliseconds:.2f}x"
          print(
            f"{variant:8s} {milliseconds:10.3f} ms total; {milliseconds / args.steps:8.3f} ms/call; vs previous {speedup}; contacts={expected_contacts}",
            flush=True,
          )
          previous = milliseconds
    return

  import mujoco
  import warp as wp

  import mujoco_warp as mjw

  print(f"package={mjw.__file__}")
  spec = mujoco.MjSpec()
  spec.option.gravity = [0, 0, 0]
  spec.option.sdf_initpoints = 40
  vertices = [[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (-0.02, 0.02)]
  spec.add_mesh(name="box", uservert=[coordinate for vertex in vertices for coordinate in vertex])
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SDF, meshname="box", pos=[10 if args.empty else 0.3, 0, 0.03])
  for index in range(17):
    position = [0.3, 0, 0.055] if index == 16 else [index % 4 * 0.03, index // 4 * 0.03, 0.03]
    body = spec.worldbody.add_body(pos=position)
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=0.1)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(model)
    d = mjw.put_data(model, data, nworld=args.worlds, nconmax=128, njmax=512)
    m.opt.graph_conditional = True
    mjw.collision(m, d)
    with wp.ScopedCapture() as capture:
      mjw.collision(m, d)
    for _ in range(20):
      wp.capture_launch(capture.graph)
    wp.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
      wp.capture_launch(capture.graph)
    wp.synchronize()
    milliseconds = (time.perf_counter() - started) * 1000
    contacts = int(d.nacon.numpy()[0])
    assert contacts < d.naconmax, "Increase contact capacity"
    print(
      json.dumps(
        {
          "milliseconds": milliseconds,
          "contacts": contacts,
          "device": wp.get_device().name,
          "mujoco": mujoco.__version__,
          "warp": wp.__version__,
        }
      )
    )


if __name__ == "__main__":
  main()
