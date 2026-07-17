# Pull Request — copy/paste for GitHub

> Temporary file — delete after use.

---

## Title

```
feat: Windows CI, compatibility database, and roadmap emulation improvements
```

**Alternatives:**
- `Add Windows CI, compatibility DB, and roadmap emulation improvements`
- `Windows roadmap: CI, compat DB, SaveData, PlayGo, GPU/shaders/audio`

---

## PR body

```markdown
## Summary

This PR implements the Windows roadmap batch for KytyPS5: continuous integration, a community compatibility database, and several functional emulator improvements (SaveData, PlayGo, import telemetry, GPU/shaders/audio).

It also adds a Windows build script (`scripts/build-windows.cmd`), validated locally with clang-cl, Ninja, Vulkan SDK 1.4, and Qt 6.8.2.

## Changes

### CI / DevOps
- **`.github/workflows/ci.yml`** — Windows build (cmake + clang-cl + ninja), tests via `ctest`, launcher disabled (`KYTY_BUILD_LAUNCHER=OFF`)
- **`.github/workflows/compat-db.yml`** — automatic reminder on issues labeled `compatibility` to contribute to `compat-db/compatibility_db.json`
- **`.github/ISSUE_TEMPLATE/compatibility-report.yaml`** — structured template for compatibility reports
- **`compat-db/compatibility_db.json`** — compatibility database skeleton
- **`scripts/build-windows.cmd`** — configure, build (emulator + launcher + tests), install, and `ctest` locally

### Import telemetry
- Per-stub call counter in `runtimeLinker`
- `GetStubbedImportReport()` / `WriteStubbedImportReport()` API
- Writes `_ImportReport.txt` on emulator shutdown

### PlayGo
- Per-chunk state (locus)
- Functional stubs for `GetToDoList`, `SetToDoList`, `Prefetch`, `GetProgress`

### SaveData
- Configurable `save_data_folder` (`--save-data-folder`, launcher)
- On-disk persistence for param / icon / memory in `libSaveData`

### GPU / Shaders
- PM4 fallbacks: AA sample control, depth render target
- `kGsFront` support (geometry Type 4) in `agc.cpp`
- **Fetch** shader stage (GS) in the SPIR-V recompiler
- Multi-GPU scoring (AMD / NVIDIA / Intel) in `vulkanWindow.cpp`
- **`docs/GPU_VALIDATION.md`** — GPU validation notes

### Audio
- LPCM and CELP/HE-VAG decoders (stub) in `ajm.cpp`
- Audio3D → SDL stereo in `audio.cpp`

### Tests
- `enable_testing()` + registration of 10 unit tests in `src/CMakeLists.txt`

## Test plan

- [x] Local Release Windows build (`clang-cl` + Ninja + Vulkan SDK + Qt 6.8.2)
- [x] `build/windows/install/kyty_emulator.exe` produced
- [x] `build/windows/install/launcher.exe` produced
- [x] `ctest`: **9/10 tests pass**
  - Known failure: `shader_recompiler_compute_tests` — `shaderStorageImageReadWithoutFormat` not supported on local test GPU (hardware limitation, not a build regression)
- [ ] GitHub Actions CI green on `windows-latest` (to confirm after merge)
- [ ] Manual launcher launch and test game run
- [ ] SaveData persistence verified after restart
- [ ] `_ImportReport.txt` generation verified after a session

## Local build (reminder)

```powershell
cmd /c C:\codes\KytyPS5-main\scripts\build-windows.cmd
```

Prerequisites: VS 2022 Build Tools, LLVM/clang-cl, Ninja, Vulkan SDK 1.3+, Qt 6 MSVC 2022 64-bit.

## AI disclosure

This PR was prepared with **AI tooling** assistance (Cursor Agent, Composer model).

| Area | AI role |
|------|---------|
| Planning | Windows setup/build plan and roadmap definition |
| Implementation | Code generation and edits (CI, stubs, shaders, SaveData, etc.) |
| Machine setup | Automated installation (winget, aqt, LLVM, manual submodules) |
| Build debugging | Fixes for `ExecutionModelGeometry`, `file.GetSize()` → `file.Size()` |
| Documentation | PR text, `GPU_VALIDATION.md`, build script |

**Human review**: the contributor validated the push to `main` (`8966ea4`). All changes were reviewed before commit; submodules were realigned to the commits recorded in the parent repo before push.

## Notes

- The launcher requires Qt 6; CI disables it to keep runs fast.
- `build/` and local install logs are not tracked in git.
- Range: `1349950` → `8966ea4` (32 files, +691 / −48 lines).
```
