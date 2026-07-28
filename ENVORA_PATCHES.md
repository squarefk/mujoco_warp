# Envora patches

This checkout contains a small integration patch maintained by the parent
`envora_sim` repository. Keep this log current when rebasing the submodule or
changing the patch.

## Newton texture SDF collision

- Added: 2026-07-28
- Upstream base: `eb35c2d8f56563a60bc09bb24472722706709332`
- Changed file: `mujoco_warp/_src/collision_sdf.py`
- Consumer: `envora_sim/environment/so101_warp`

### Reason

The SO-101 cube-pick task combines all visual mesh parts belonging to each
robot link and bakes the result with `newton.SDF.create_from_mesh`. Newton
stores these SDFs as `TextureSDFData`.

MJWarp's SDF narrow phase originally supported analytic primitives, MuJoCo SDF
plugins, and MuJoCo mesh octrees. It had no input for an external Newton texture
table, so an unmodified kernel could not sample the baked link SDFs. Compiling
the meshes as MuJoCo SDF geoms would instead create octrees, which is not the
collision representation required by this task.

### Implementation

The patch extends the existing SDF narrow phase without replacing its optimizer:

1. `VolumeData` can reference a Newton texture table and one entry in that
   table.
2. `sdf` and `sdf_grad` use Newton's hardware texture samplers when that entry
   is present, then fall back to the original MJWarp implementations.
3. `_sdf_narrowphase` accepts a geom-to-texture index and the texture table,
   then attaches the matching entry to each geom's `VolumeData`.
4. `get_sdf_params` reads a MuJoCo octree only when `mesh_octadr` is valid. The
   SO-101 companion meshes intentionally have `mesh_octadr == -1`.
5. Models without the Envora attributes receive cached empty defaults, so the
   original MJWarp SDF paths remain unchanged.

The parent environment creates the two model attributes after `mjw.put_model`:

- `geom_newton_sdf_index`: one `int32` entry per geom; `-1` means that the geom
  does not use a Newton texture.
- `newton_sdf_data`: the `TextureSDFData` table sampled by the collision kernel.

The parent model retains the `newton.SDF` owners for the lifetime of the Warp
model so their CUDA textures remain valid.

### Dependency contract

The patch imports Newton's internal texture sampler API. The parent project pins
Newton to the tested `1.4.0` release and supplies it when installing this
editable submodule.

### Verification

Run from the parent `envora_sim` repository:

```bash
CUDA_LAUNCH_BLOCKING=1 uv run pytest -q tests/environment/so101_warp
uv run pytest -q
```

The focused tests verify that all seven link SDFs are cache hits from
`assets/newton_sdf`, the absence of MuJoCo octrees, the geom-to-texture mapping,
finite penetration distance, and an actual cube-to-link contact. The full suite
also covers ordinary MuJoCo and MJWarp models that do not provide Newton
textures.

### Rebase checklist

1. Check whether upstream MJWarp now accepts an external texture SDF table. If
   it does, remove this patch and use the upstream interface.
2. Reapply only the changes described above to the new
   `collision_sdf.py`; do not copy the complete file from the old revision.
3. Preserve the explicit `newton_sdf_index = -1` initialization because Warp
   otherwise zero-initializes the struct.
4. Preserve the `mesh_octadr != -1` guard to avoid invalid CUDA memory access
   for texture-backed SDF geoms without a MuJoCo octree.
5. Run both verification commands and update the upstream base hash in this
   log.
