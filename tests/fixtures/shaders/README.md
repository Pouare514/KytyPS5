# PS5 shader corpus (fixtures)

Place extracted PS5 shader blobs here for regression testing.

## Expected layout

```
tests/fixtures/shaders/
  README.md
  <game-or-suite>/
    shader_<hash>.bin
    shader_<hash>.meta.json   # optional: stage, wave size, user data
```

## Pipeline

1. Decode with `ShaderDecoder`
2. Lower to IR and validate SPIR-V emission (compile-only by default)
3. Dispatch through `shader_recompiler_compute_tests` harness when GPU features are available

New fixtures should start as compile-only coverage; promote to GPU dispatch once descriptors and
expected outputs are captured.
