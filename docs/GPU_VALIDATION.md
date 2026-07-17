# GPU validation notes (Windows)

KytyPS5 selects a Vulkan 1.3 device using a score that prefers:

1. Discrete GPUs over integrated/virtual devices
2. AMD (`0x1002`) — PS5 RDNA2 reference architecture
3. NVIDIA (`0x10de`) and Intel (`0x8086`) discrete GPUs

At startup the log prints each enumerated device and the selected device:

```
Vulkan device: <name> (vendor=0x.... type=... score candidate=...)
Vulkan selected device: <name> (vendor=0x.... score=...)
```

If rendering fails on a non-NVIDIA GPU, capture the full log and enable `--vulkan-validation true`.

Recommended manual checks:

- AMD RDNA2: verify depth (`D32_SFLOAT`) and BC3 textures
- Intel Arc: verify storage image fallback paths for `R8G8B8A8_SRGB`
