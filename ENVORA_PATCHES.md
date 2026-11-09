# Envora patches

Base: upstream v4.0.0 (`4d1c64bc077d725247e9bf5e2ac3f357749ff804`).
The fork's existing `main` branch remains on the previous Envora revision;
this port lives on `envora/mujoco4-newton-sdf`.

## Native SDF scheduling

Only scheduling changes remain: compact SDF candidate indices, a fixed
65,536-thread strided launch, and a graph conditional for empty candidate lists.
Disjoint bounds skip the current work item rather than terminating its thread.
Native distance/gradient sampling and upstream contact handling are unchanged.

The former Newton texture sampler, custom model attributes and Newton dependency
have been removed. Envora now obtains Newton contacts through its official
CollisionPipeline, outside MuJoCo Warp. The fork remains one commit above the
upstream v4.0.0 base; the benchmark below covers native octree SDF scheduling.

## One-command scheduling comparison

From the parent `envora_sim` checkout, using its pinned MuJoCo/Warp versions:

```sh
uv run python deps/mujoco_warp/envora_tests/benchmark_sdf.py
```

Runs 4096 worlds, 20 warmups and 100 measured collision calls per variant.
`--worlds` and `--steps` adjust the workload. Each variant runs in a fresh
process using a temporary source copy, leaving the checkout unchanged:

| Variant | Compact pairs | Fixed grid | Skip empty |
|---|---|---|---|
| baseline | off | off | off |
| compact | on | off | off |
| grid | on | on | off |
| all | on | on | on |

Prints total/per-call milliseconds and speedup over the preceding variant for
mixed primitive/SDF contacts and zero SDF candidates. Contact counts must match.
This is a cumulative ablation, not isolated end-to-end environment throughput:
state is fixed, collision runs inside a CUDA graph, and compilation, capture,
rendering and host transfers are excluded from timing. Run on an otherwise idle
GPU; an optimization need not help every workload or batch size.
