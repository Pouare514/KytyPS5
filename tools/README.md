# RDNA2 ISA tooling

Scripts for tracking and extending shader recompiler coverage against the AMD RDNA2 reference guide.

## Workflow

```powershell
# Regenerate supplemental decode tables from the AMD doc
python tools/generate_rdna2_tables.py

# Audit decode-table coverage vs Ch. 13 opcode tables
python tools/rdna2_opcode_matrix.py

# Fail if coverage regresses below baseline
python tools/check_coverage.py

# Optional: run compute shader tests (requires MSVC SDK + Vulkan GPU)
scripts\build-windows.cmd
build\windows\install\shader_recompiler_compute_tests.exe
build\windows\install\shader_cfg_tests.exe
```

Current decode baseline: **100%** (`tools/rdna2_coverage_baseline.json`).

`check_coverage.py` tracks IR lowering against **doc opcode entries** (`lower_ops_doc_coverage_pct`),
not decoder enum cardinality. Extra decoder→IR alias rows are reported separately as
`lower_ops_alias_extra`.

## SPIR-V notes

- `ImageBvhIntersectRay` / `ImageBvh64IntersectRay` currently emit a decode-safe stub (zero-filled
  destinations). Ray-query lowering is deferred until the host exposes RT resources.
- Prefer typed storage image formats (`Rgba8Unorm`, `R32ui`, …) over
  `*WithoutFormat` when game descriptors provide a known format — this improves GPU portability on
  iGPUs and software stacks.

## Optional GPU CI

The `gpu-shader-tests` workflow job (`.github/workflows/ci.yml`) rebuilds with
`scripts/build-windows.cmd` and runs `shader_recompiler_compute_tests` on a Windows runner. It is
`continue-on-error: true` so CPU-only PR validation stays authoritative.

## Outputs

| File | Purpose |
|------|---------|
| `tools/rdna2_coverage.json` | Latest per-family coverage report |
| `tools/rdna2_coverage_baseline.json` | CI baseline (auto-updated on improvement) |
| `src/graphics/shader/recompiler/generated/*.inc` | Generated enum entries, decode tables, IR lower maps |

## Notes

- `GLOBAL` / `SCRATCH` share the FLAT microcode format (`ENCODING=0x37`) via the `SEG` field; encodings `0x39` / `0x3B` are also routed as aliases.
- MIMG coverage in the audit includes `ImageOps.cpp` sample/gather/atomic/base tables plus `generated/Rdna2ExtraMIMG_*.inc`.
- Generated opcodes are appended to `Decoder::Opcode` via `Rdna2ExtraOpcodes.inc`.
