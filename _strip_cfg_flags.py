#!/usr/bin/env python3
"""Clear IMAGE_GUARD_CF_INSTRUMENTED from kyty_emulator.exe load config."""
import struct
import sys

paths = sys.argv[1:] or [
    r"build\windows\install\kyty_emulator.exe",
    r"build\windows\kyty_emulator.exe",
]

for path in paths:
    try:
        data = bytearray(open(path, "rb").read())
    except OSError as e:
        print(f"skip {path}: {e}")
        continue
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    numsec = struct.unpack_from("<H", data, e_lfanew + 6)[0]
    opt_size = struct.unpack_from("<H", data, e_lfanew + 20)[0]
    sec = e_lfanew + 24 + opt_size
    dd = e_lfanew + 24 + 112
    rva = struct.unpack_from("<I", data, dd + 10 * 8)[0]

    def rva_to_off(rva: int) -> int:
        for i in range(numsec):
            off = sec + i * 40
            vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
            if va <= rva < va + max(vsize, rawsize):
                return rawptr + (rva - va)
        raise SystemExit(f"rva {rva:#x} not in sections")

    gf = rva_to_off(rva) + 0x90
    flags = struct.unpack_from("<I", data, gf)[0]
    new_flags = flags & ~0x100
    struct.pack_into("<I", data, gf, new_flags)
    open(path, "wb").write(data)
    print(f"{path}: GuardFlags {flags:#x} -> {new_flags:#x}")
