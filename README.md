# KytyPS5

[![Platform](https://img.shields.io/badge/platform-Windows%20x64-0078D4.svg)](#system-requirements)
[![Status](https://img.shields.io/badge/status-early%20development-orange.svg)](#current-status)
[![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](LICENSE)

KytyPS5 is a free and open-source PlayStation 5 emulator written in C++ for Windows. It is based on
a heavily modified version of [Kyty](https://github.com/InoriRus/Kyty). The project is in an early
stage of development, so compatibility is limited and behavior may change significantly between
builds.

> [!IMPORTANT]
> KytyPS5 is not affiliated with Sony Interactive Entertainment or PlayStation. The project does
> not distribute games or copyrighted system software. Use only game files that you have obtained
> legally.

## Current Status

KytyPS5 can boot 2D games and a selection of 3D games, including titles built with Unreal Engine
4/5, Unity, and custom engines. No external low-level emulation modules are currently required.

Development is focused on compatibility and boot reliability.

Linux support is planned, but Windows is the only supported platform at this time.

## Bugs and Issues

The project is in an early stage, so please be mindful when opening new issues. Expect crashes,
graphical glitches, low compatibility, and poor performance.

## Screenshots

<table align="center">
  <tr>
    <td align="center">
      <strong>Disgaea 6</strong><br>
      <img src="docs/screenshots/ps5-01.png" width="300" alt="Disgaea 6 running in KytyPS5">
    </td>
    <td align="center">
      <strong>Dreaming Sarah</strong><br>
      <img src="docs/screenshots/ps5-03.png" width="300" alt="Dreaming Sarah running in KytyPS5">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Minecraft Legends</strong><br>
      <img src="docs/screenshots/ps5-04.png" width="300" alt="Minecraft Legends running in KytyPS5">
    </td>
    <td align="center">
      <strong>SILENT HILL: The Short Message</strong><br>
      <img src="docs/screenshots/ps5-05.png" width="300" alt="SILENT HILL: The Short Message running in KytyPS5">
    </td>
  </tr>
</table>

## Contributing

Testing games and submitting detailed bug reports are useful ways to contribute. Search existing
issues first, then use the **Game Emulation Bug Report** template and attach the complete log file.

Code contributions should be focused, build successfully on Windows, and include relevant tests
where practical. Because KytyPS5 is still evolving quickly, consider opening an issue before
starting a large change.

## Developer Information

The PS5 graphics architecture is based on AMD RDNA 2. Use AMD's
[RDNA 2 Instruction Set Architecture Reference Guide (document 70648)](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)
as the primary instruction-encoding reference when working on shader decoding and recompilation.

Important areas of the codebase:

- [`src/graphics/shader/recompiler`](src/graphics/shader/recompiler) — instruction decoding,
  intermediate representation, control flow, resource tracking, and SPIR-V emission
- [`src/graphics/guest_gpu`](src/graphics/guest_gpu) — PS5 (Prospero) GPU formats and command processing
- [`src/graphics/host_gpu`](src/graphics/host_gpu) — Vulkan host backend and resource management
- [`tests`](tests) — focused memory, shader, and resource-tracking regression tests

The renderer targets Vulkan 1.3. Keep shader changes aligned with both the RDNA 2 ISA semantics and
the Vulkan/SPIR-V validation rules.

## Building

### System requirements

- Windows 10 version 1803
- A 64-bit x86 processor
- A Vulkan 1.3-capable GPU with current drivers

### Build requirements

- Git
- CMake 3.12 or newer
- Ninja
- Visual Studio 2022 or Build Tools 2022 with the **Desktop development with C++** workload and
  **C++ Clang tools for Windows** component
- Qt 6 for MSVC 2022 64-bit, including Concurrent, Network, and Widgets
- Vulkan SDK 1.4.350 (same version as CI and `scripts/build-windows.cmd`). Set `VULKAN_SDK` to the SDK
  root (e.g. `C:\VulkanSDK\1.4.350.0`); CMake expects `Lib\vulkan-1.lib` under that path. CI
  verifies this before configuring.

The Microsoft C++ compiler (`cl.exe`) is not supported; use `clang-cl`.

Open an **x64 Native Tools Command Prompt for Visual Studio 2022** (or the equivalent Developer
PowerShell), change to the repository root, and initialize the dependencies:

```powershell
git submodule update --init --recursive
```

Configure the project. Replace the Qt path with the version installed on your system:

```powershell
cmake -S src -B build/windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/msvc2022_64"
```

Build the launcher and stage a runnable installation:

```powershell
cmake --build build/windows --target launcher
cmake --install build/windows --prefix build/windows/install
```

The finished application and its runtime dependencies will be placed in
`build/windows/install`.

### Visual Studio Code

A ready-made Visual Studio Code setup is included in [`.vscode`](.vscode). It configures CMake
Tools to build the project with Ninja and `clang-cl` and provides launch profiles for both
`launcher.exe` and `kyty_emulator.exe`.

Before using it:

1. Install the **CMake Tools** and **C/C++** extensions in Visual Studio Code.
2. Update `CMAKE_PREFIX_PATH` in [`.vscode/settings.json`](.vscode/settings.json) to point to your
   Qt 6 MSVC installation.
3. Update the `--game` path in [`.vscode/launch.json`](.vscode/launch.json) for the
   **Debug kyty_emulator** profile.
4. Open the repository in an x64 Visual Studio developer environment, configure the CMake project,
   and select a launch profile from **Run and Debug**.

## Running

Update your graphics driver before reporting rendering problems.

To use the graphical launcher:

```powershell
.\build\windows\install\launcher.exe
```

On first launch, add one or more game folders in the global settings. The launcher searches those
folders recursively for game directories containing `eboot.bin`. Select a detected game and run it
from the game list.

The emulator can also be started directly with a legally obtained game directory or ELF file:

```powershell
.\build\windows\install\kyty_emulator.exe --game "D:\Games\ExampleGame"
```

Run `kyty_emulator.exe --help` to see the available graphics, logging, validation, profiling, and
debugging options.

### Troubleshooting

#### Crash at startup with `loader_get_json: Failed to open JSON file ... medal-vulkan64.json`

This happens when a third-party Vulkan layer (commonly [Medal](https://medal.tv/)) is registered in
Windows but its manifest JSON file is missing, often after an update or partial uninstall. Orphaned
registry entries can remain even after Medal is removed.

**Automatic fix (kyty_emulator / launcher):** at startup the process sets `VK_LOADER_LAYERS_DISABLE`
to skip known third-party layers (Medal, NVIDIA capture `*nvspcap*`, Steam overlay, Discord, OBS).
Other applications are unaffected; only this process skips them.

Boot logs include `VulkanLayerTrace:` lines listing available layers and the effective disable list.

**Option A — Disable Vulkan validation** (fastest if issues persist):

In the launcher, uncheck **Enable Vulkan validation layers**, or run:

```powershell
.\build\windows\install\kyty_emulator.exe --game "D:\Games\ExampleGame" --vulkan-validation false
```

**Option B — Fix Medal**:

Reinstall or update Medal, disable Vulkan capture in Medal settings, or uninstall Medal if unused.

**Option C — Manual layer disable** (only if automatic disable is insufficient):

```powershell
$env:VK_LOADER_LAYERS_DISABLE = "VK_LAYER_MEDAL_capture,VK_LAYER_MEDAL_HOOK,*nvspcap*,~implicit~"
.\build\windows\install\kyty_emulator.exe --game "D:\Games\ExampleGame"
```

#### NVIDIA / Steam overlay and crash `0xC0000409`

GeForce Experience / NVIDIA App in-game overlay and other implicit Vulkan layers can hook
`vkQueuePresentKHR` and trigger native crashes (`-1073740791` / `0xC0000409`).

**Quick test:**

1. Disable **In-Game Overlay** in NVIDIA App / GeForce Experience.
2. Or run with all implicit layers disabled:

```powershell
$env:VK_LOADER_LAYERS_DISABLE = "~implicit~"
.\build\windows\install\kyty_emulator.exe --game "D:\Games\ExampleGame" --vulkan-validation false
```

Fatal crash details are appended to `_kyty_fatal.txt` (and the `--printf-output-file` if set):
`Fatal win exception: rip=..., module=...` or `Security failure: retaddr=..., module=...`.

**Registry cleanup:** if Medal errors still appear, remove stale keys under
`HKCU\Software\Khronos\Vulkan\ImplicitLayers` that point to `%LocalAppData%\Medal\`.

Recent builds log general loader errors without exiting, but disabling the broken layer or
validation avoids the noise entirely.

#### `C++ exception (0xe06d7363)` or game closes without `GameMainLoop: exit...`

`0xe06d7363` is the Windows code for MSVC C++ `throw`. KytyPS5 no longer logs these on first
chance — they are usually caught internally (Vulkan validation, shader compiler, Qt) and are not
fatal by themselves.

If the game window closes without a clean exit message, capture a diagnostic log:

```powershell
.\build\windows\install\kyty_emulator.exe --game "D:\Games\ExampleGame" `
  --vulkan-validation false --printf-direction File --printf-output-file _kyty.txt 2> _kyty_stderr.txt
echo $LASTEXITCODE
```

| Exit code | Meaning |
|-----------|---------|
| `0` | Clean exit |
| `-1073740791` (`0xC0000409`) | Stack buffer overrun — check `_kyty_fatal.txt`, `_kyty_stderr.txt` for `Security failure:` (retaddr) or `_kyty.txt` for `Fatal win exception:` |

Compare with `--vulkan-validation true` to see whether the Vulkan validation layer is involved.
Recent builds print `Fatal win exception: code=..., rip=..., module=...` before an unhandled crash,
and `Security failure: code=..., retaddr=...` when the MSVC `/GS` cookie trips before `__fastfail`.

#### Hybrid laptop GPU (Intel + NVIDIA / AMD)

On startup, KytyPS5 prints `Vulkan selected device: ...`. If you have a discrete GPU (e.g. RTX 4080)
but see Intel integrated graphics instead, Windows may be running the emulator on the wrong GPU.

**Automatic fix:** recent builds export `NvOptimusEnablement` / `AmdPowerXpressRequestHighPerformance`
so the discrete GPU is preferred.

**Manual fix:** Windows Settings → System → Display → Graphics → add `kyty_emulator.exe` →
**High performance** (NVIDIA / AMD).

### AI Use

AI tools may be used for research, reverse engineering, and development assistance. Contributors
must fully understand, review, and test all code they submit and remain responsible for its
correctness. Repository communication, including pull-request descriptions, code comments, and
issue comments, must come from the human contributor rather than an autonomous AI agent.

Pull requests that include AI-assisted or AI-generated work should disclose the scope of the AI
involvement and describe the human review and testing performed before submission. Unverified or
untested generated changes may be closed without review.

## License

KytyPS5 is licensed under the [GNU General Public License version 2](LICENSE)
(`GPL-2.0-only`).

This project is based on the original [Kyty](https://github.com/InoriRus/Kyty), which was released
under the MIT License. Kyty's original copyright and license notice are preserved in
[`LICENSES/Kyty-MIT.txt`](LICENSES/Kyty-MIT.txt). Third-party components remain subject to the
licenses included with those components.

## Special Thanks

- [InoriRus/Kyty](https://github.com/InoriRus/Kyty) — KytyPS5 is based on a heavily modified version
  of the original Kyty project.
- [shadps4-emu/shadPS4](https://github.com/shadps4-emu/shadPS4) — reference for memory-model
  understanding and the AVPlayer implementation.
