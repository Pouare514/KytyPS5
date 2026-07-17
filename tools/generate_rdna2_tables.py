#!/usr/bin/env python3
"""Generate supplemental RDNA2 decode tables, enum entries, and IR lower mappings."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "src" / "graphics" / "shader" / "recompiler" / "generated"
MATRIX = ROOT / "tools" / "rdna2_opcode_matrix.py"

FAMILY_TABLE = {
    "SOP2": "SOP2_OPS",
    "SOPK": "SOPK_OPS",
    "SOP1": "SOP1_OPS",
    "SOPC": "SOPC_OPS",
    "SOPP": "SOPP_OPS",
    "SMEM": "SMEM_OPS",
    "VOP2": "VOP2_OPS",
    "VOP1": "VOP1_OPS",
    "VOPC": "VOPC_OPS",
    "VOP3A": "VOP3_OPS",
    "VOP3B": "VOP3_OPS",
    "VOP3P": "VOP3P_OPS",
    "DS": "DS_OPS",
    "MTBUF": "MTBUF_OPS",
    "MUBUF": "MUBUF_OPS",
    "FLAT": "FLAT_OPS",
    "GLOBAL": "FLAT_OPS",
    "SCRATCH": "FLAT_OPS",
}

FLAT_OPCODE_MAP = {
    "LOAD_UBYTE": ("FlatLoadUbyte", 1, 8, False),
    "LOAD_SBYTE": ("FlatLoadSbyte", 1, 8, True),
    "LOAD_USHORT": ("FlatLoadUshort", 1, 16, False),
    "LOAD_SSHORT": ("FlatLoadSshort", 1, 16, True),
    "LOAD_DWORD": ("FlatLoadDword", 1, 32, False),
    "LOAD_DWORDX2": ("FlatLoadDwordx2", 2, 32, False),
    "LOAD_DWORDX3": ("FlatLoadDwordx3", 3, 32, False),
    "LOAD_DWORDX4": ("FlatLoadDwordx4", 4, 32, False),
    "LOAD_DWORD_ADDTID": ("FlatLoadDwordAddtid", 1, 32, False),
    "LOAD_UBYTE_D16": ("FlatLoadUbyteD16", 1, 8, False),
    "LOAD_UBYTE_D16_HI": ("FlatLoadUbyteD16Hi", 1, 8, False),
    "LOAD_SBYTE_D16": ("FlatLoadSbyteD16", 1, 8, True),
    "LOAD_SBYTE_D16_HI": ("FlatLoadSbyteD16Hi", 1, 8, True),
    "LOAD_USHORT_D16": ("FlatLoadUshortD16", 1, 16, False),
    "LOAD_USHORT_D16_HI": ("FlatLoadUshortD16Hi", 1, 16, False),
    "LOAD_SSHORT_D16": ("FlatLoadSshortD16", 1, 16, True),
    "LOAD_SSHORT_D16_HI": ("FlatLoadSshortD16Hi", 1, 16, True),
    "LOAD_SHORT_D16": ("FlatLoadShortD16", 1, 16, False),
    "LOAD_SHORT_D16_HI": ("FlatLoadShortD16Hi", 1, 16, False),
    "STORE_BYTE": ("FlatStoreByte", 1, 8, False),
    "STORE_SHORT": ("FlatStoreShort", 1, 16, False),
    "STORE_DWORD": ("FlatStoreDword", 1, 32, False),
    "STORE_DWORDX2": ("FlatStoreDwordx2", 2, 32, False),
    "STORE_DWORDX3": ("FlatStoreDwordx3", 3, 32, False),
    "STORE_DWORDX4": ("FlatStoreDwordx4", 4, 32, False),
    "STORE_DWORD_ADDTID": ("FlatStoreDwordAddtid", 1, 32, False),
    "STORE_BYTE_D16_HI": ("FlatStoreByteD16Hi", 1, 8, False),
    "STORE_SHORT_D16_HI": ("FlatStoreShortD16Hi", 1, 16, False),
    "ATOMIC_SWAP": ("BufferAtomicSwap", 1, 32, False),
    "ATOMIC_CMPSWAP": ("BufferAtomicSwap", 1, 32, False),
    "ATOMIC_ADD": ("BufferAtomicAdd", 1, 32, False),
    "ATOMIC_SUB": ("BufferAtomicSub", 1, 32, False),
    "ATOMIC_CSUB": ("BufferAtomicCsub", 1, 32, False),
    "ATOMIC_SMIN": ("BufferAtomicSmin", 1, 32, False),
    "ATOMIC_UMIN": ("BufferAtomicUmin", 1, 32, False),
    "ATOMIC_SMAX": ("BufferAtomicSmax", 1, 32, False),
    "ATOMIC_UMAX": ("BufferAtomicUmax", 1, 32, False),
    "ATOMIC_AND": ("BufferAtomicAnd", 1, 32, False),
    "ATOMIC_OR": ("BufferAtomicOr", 1, 32, False),
    "ATOMIC_XOR": ("BufferAtomicXor", 1, 32, False),
    "ATOMIC_INC": ("BufferAtomicInc", 1, 32, False),
    "ATOMIC_DEC": ("BufferAtomicDec", 1, 32, False),
    "ATOMIC_FCMPSWAP": ("BufferAtomicFcmpswap", 1, 32, False),
    "ATOMIC_FMIN": ("BufferAtomicFmin", 1, 32, False),
    "ATOMIC_FMAX": ("BufferAtomicFmax", 1, 32, False),
    "ATOMIC_SWAP_X2": ("BufferAtomicSwapX2", 2, 32, False),
    "ATOMIC_CMPSWAP_X2": ("BufferAtomicSwapX2", 2, 32, False),
    "ATOMIC_ADD_X2": ("BufferAtomicAddX2", 2, 32, False),
    "ATOMIC_SUB_X2": ("BufferAtomicSubX2", 2, 32, False),
    "ATOMIC_SMIN_X2": ("BufferAtomicSminX2", 2, 32, False),
    "ATOMIC_UMIN_X2": ("BufferAtomicUminX2", 2, 32, False),
    "ATOMIC_SMAX_X2": ("BufferAtomicSmaxX2", 2, 32, False),
    "ATOMIC_UMAX_X2": ("BufferAtomicUmaxX2", 2, 32, False),
    "ATOMIC_AND_X2": ("BufferAtomicAndX2", 2, 32, False),
    "ATOMIC_OR_X2": ("BufferAtomicOrX2", 2, 32, False),
    "ATOMIC_XOR_X2": ("BufferAtomicXorX2", 2, 32, False),
    "ATOMIC_INC_X2": ("BufferAtomicIncX2", 2, 32, False),
    "ATOMIC_DEC_X2": ("BufferAtomicDecX2", 2, 32, False),
    "ATOMIC_FCMPSWAP_X2": ("BufferAtomicFcmpswapX2", 2, 32, False),
    "ATOMIC_FMIN_X2": ("BufferAtomicFminX2", 2, 32, False),
    "ATOMIC_FMAX_X2": ("BufferAtomicFmaxX2", 2, 32, False),
}

FLAT_OPCODE_KEYS = sorted(FLAT_OPCODE_MAP.keys(), key=len, reverse=True)

MIMG_BASE_SUFFIX = {
    "LOAD": "ImageLoad",
    "LOAD_MIP": "ImageLoadMip",
    "LOAD_PCK": "ImageLoadPck",
    "LOAD_PCK_SGN": "ImageLoadPckSgn",
    "LOAD_MIP_PCK": "ImageLoadMipPck",
    "LOAD_MIP_PCK_SGN": "ImageLoadMipPckSgn",
    "STORE": "ImageStore",
    "STORE_MIP": "ImageStoreMip",
    "STORE_PCK": "ImageStorePck",
    "STORE_MIP_PCK": "ImageStoreMipPck",
    "GET_RESINFO": "ImageGetResinfo",
    "GET_LOD": "ImageGetLod",
    "MSAA_LOAD": "ImageMsaaLoad",
    "BVH_INTERSECT_RAY": "ImageBvhIntersectRay",
    "BVH64_INTERSECT_RAY": "ImageBvh64IntersectRay",
}

MIMG_GATHER_SUFFIX = {
    "GATHER4": ("ImageGather4", 0, 2),
    "GATHER4_CL": ("ImageGather4Cl", 64, 2),
    "GATHER4_L": ("ImageGather4L", 1, 2),
    "GATHER4_B": ("ImageGather4B", 2, 2),
    "GATHER4_B_CL": ("ImageGather4BCl", 66, 2),
    "GATHER4_LZ": ("ImageGather4Lz", 32, 2),
    "GATHER4_C": ("ImageGather4C", 8, 3),
    "GATHER4_C_CL": ("ImageGather4CCl", 72, 3),
    "GATHER4_C_L": ("ImageGather4CL", 9, 3),
    "GATHER4_C_B": ("ImageGather4CB", 10, 3),
    "GATHER4_C_B_CL": ("ImageGather4CBCl", 74, 3),
    "GATHER4_C_LZ": ("ImageGather4CLz", 40, 3),
    "GATHER4_O": ("ImageGather4O", 16, 3),
    "GATHER4_CL_O": ("ImageGather4ClO", 80, 3),
    "GATHER4_L_O": ("ImageGather4LO", 17, 3),
    "GATHER4_B_O": ("ImageGather4BO", 18, 3),
    "GATHER4_B_CL_O": ("ImageGather4BClO", 82, 3),
    "GATHER4_LZ_O": ("ImageGather4LzO", 48, 3),
    "GATHER4_C_O": ("ImageGather4CO", 24, 4),
    "GATHER4_C_CL_O": ("ImageGather4CClO", 88, 4),
    "GATHER4_C_L_O": ("ImageGather4CLO", 25, 4),
    "GATHER4_C_B_O": ("ImageGather4CBO", 26, 4),
    "GATHER4_C_B_CL_O": ("ImageGather4CBClO", 90, 4),
    "GATHER4_C_LZ_O": ("ImageGather4CLzO", 56, 4),
    "GATHER4H": ("ImageGather4h", 512, 2),
}

MIMG_ATOMIC_SUFFIX = {
    "ATOMIC_SWAP": "ImageAtomicSwap",
    "ATOMIC_CMPSWAP": "ImageAtomicCmpswap",
    "ATOMIC_ADD": "ImageAtomicAdd",
    "ATOMIC_SUB": "ImageAtomicSub",
    "ATOMIC_SMIN": "ImageAtomicSmin",
    "ATOMIC_UMIN": "ImageAtomicUmin",
    "ATOMIC_SMAX": "ImageAtomicSmax",
    "ATOMIC_UMAX": "ImageAtomicUmax",
    "ATOMIC_AND": "ImageAtomicAnd",
    "ATOMIC_OR": "ImageAtomicOr",
    "ATOMIC_XOR": "ImageAtomicXor",
    "ATOMIC_INC": "ImageAtomicInc",
    "ATOMIC_DEC": "ImageAtomicDec",
    "ATOMIC_FCMPSWAP": "ImageAtomicFcmpswap",
    "ATOMIC_FMIN": "ImageAtomicFmin",
    "ATOMIC_FMAX": "ImageAtomicFmax",
}

MIMG_SAMPLE_SUFFIX = {
    "SAMPLE_D_G16": (4, 64),
    "SAMPLE_D_CL_G16": (4 | 64, 64),
    "SAMPLE_C_D_G16": (12, 64),
    "SAMPLE_C_D_CL_G16": (12 | 64, 64),
    "SAMPLE_D_O_G16": (20, 64),
    "SAMPLE_D_CL_O_G16": (20 | 64, 64),
    "SAMPLE_C_D_O_G16": (28, 64),
    "SAMPLE_C_D_CL_O_G16": (28 | 64, 64),
}


def load_matrix_module():
    import sys
    spec = importlib.util.spec_from_file_location("rdna2_opcode_matrix", MATRIX)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rdna2_opcode_matrix"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def doc_to_enum_name(doc_name: str) -> str:
    raw = doc_name
    for prefix in (
        "S_",
        "V_",
        "BUFFER_",
        "TBUFFER_",
        "IMAGE_",
        "FLAT_",
        "GLOBAL_",
        "SCRATCH_",
        "DS_",
    ):
        if raw.startswith(prefix):
            suffix = raw[len(prefix) :]
            head = prefix[0]
            if prefix.startswith("BUFFER"):
                head = "Buffer"
            elif prefix.startswith("TBUFFER"):
                head = "TBuffer"
            elif prefix.startswith("IMAGE"):
                head = "Image"
            elif prefix.startswith("FLAT"):
                head = "Flat"
            elif prefix.startswith("GLOBAL"):
                head = "Flat"
            elif prefix.startswith("SCRATCH"):
                head = "Flat"
            elif prefix.startswith("DS"):
                head = "Ds"
            else:
                head = prefix[0]
            parts = suffix.lower().split("_")
            return head + "".join(piece.capitalize() for piece in parts)
    return raw


def flat_memory_entry(doc_name: str, opcode_id: int) -> str | None:
    for prefix in ("FLAT_", "GLOBAL_", "SCRATCH_"):
        if not doc_name.startswith(prefix):
            continue
        suffix = doc_name[len(prefix) :]
        for key in FLAT_OPCODE_KEYS:
            enum_name, dwords, bits, signed = FLAT_OPCODE_MAP[key]
            if suffix == key or suffix.startswith(key + "_"):
                signed_flag = "true" if signed else "false"
                return (
                    f"    {{0x{opcode_id:02x}u, Opcode::{enum_name}, {dwords}, {bits}, "
                    f"{signed_flag}}},"
                )
    return None


def mimg_base_entry(doc_name: str, opcode_id: int) -> str | None:
    if not doc_name.startswith("IMAGE_"):
        return None
    suffix = doc_name[len("IMAGE_") :]
    enum_name = MIMG_BASE_SUFFIX.get(suffix)
    if enum_name is None:
        return None
    return f"    {{0x{opcode_id:02x}u, Opcode::{enum_name}}},"


def mimg_gather_entry(doc_name: str, opcode_id: int) -> str | None:
    if not doc_name.startswith("IMAGE_"):
        return None
    suffix = doc_name[len("IMAGE_") :]
    info = MIMG_GATHER_SUFFIX.get(suffix)
    if info is None:
        return None
    enum_name, flags, addr = info
    name = suffix.lower()
    return (
        f"    {{0x{opcode_id:02x}u, \"image_{name}\", Opcode::{enum_name}, {flags}u, {addr}u}},"
    )


def mimg_atomic_entry(doc_name: str, opcode_id: int) -> str | None:
    if not doc_name.startswith("IMAGE_"):
        return None
    suffix = doc_name[len("IMAGE_") :]
    enum_name = MIMG_ATOMIC_SUFFIX.get(suffix)
    if enum_name is None:
        return None
    name = suffix.lower()
    return f"    {{0x{opcode_id:02x}u, \"image_{name}\", Opcode::{enum_name}}},"


def mimg_sample_entry(doc_name: str, opcode_id: int) -> str | None:
    if not doc_name.startswith("IMAGE_"):
        return None
    suffix = doc_name[len("IMAGE_") :]
    info = MIMG_SAMPLE_SUFFIX.get(suffix)
    if info is None:
        return None
    flags, _ = info
    name = suffix.lower()
    return f"    SampleInfo(0x{opcode_id:02x}u, \"image_{name}\", {flags}u),"


def buffer_memory_entry(doc_name: str, opcode_id: int) -> str | None:
    if not doc_name.startswith("BUFFER_"):
        return None
    enum_name = doc_to_enum_name(doc_name)
    suffix = doc_name[len("BUFFER_") :]
    if suffix.startswith("LOAD_FORMAT"):
        return f"    {{0x{opcode_id:02x}u, Opcode::{enum_name}, 1, 32, false, false, true}},"
    if suffix.startswith("STORE_FORMAT"):
        return f"    {{0x{opcode_id:02x}u, Opcode::{enum_name}, 1, 32, false, false, true}},"
    if "LOAD" in suffix or "STORE" in suffix or "ATOMIC" in suffix:
        dwords = 1
        bits = 32
        signed = "SBYTE" in suffix or "SSHORT" in suffix
        if "X4" in suffix:
            dwords = 4
        elif "X3" in suffix:
            dwords = 3
        elif "X2" in suffix:
            dwords = 2
        elif "BYTE" in suffix and "DWORD" not in suffix:
            bits = 8
        elif "SHORT" in suffix and "DWORD" not in suffix:
            bits = 16
        signed_flag = "true" if signed else "false"
        return (
            f"    {{0x{opcode_id:02x}u, Opcode::{enum_name}, {dwords}, {bits}, "
            f"{signed_flag}}},"
        )
    return f"    {{0x{opcode_id:02x}u, Opcode::{enum_name}, 1, 32}},"


def scalar_entry(opcode_id: int, enum_name: str) -> str:
    return f"    {{0x{opcode_id:02x}u, Opcode::{enum_name}}},"


def parse_enum(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    block = re.search(r"enum class Opcode \{([^}]+)\}", text, re.DOTALL)
    names = set()
    if block:
        for line in block.group(1).splitlines():
            line = line.strip().rstrip(",")
            if line and line not in ("Unknown", "Unsupported"):
                names.add(line)
    return names


def parse_existing_lower_ops() -> set[str]:
    path = ROOT / "src/graphics/shader/recompiler/shaderIR/ShaderIROpcodes.cpp"
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"Decoder::Opcode::([A-Za-z0-9_]+)", text))


def explicit_lower_mappings() -> dict[str, str]:
    """Decoder opcodes lowered outside the static LOWER_OPS table in ShaderIROpcodes.cpp."""
    m: dict[str, str] = {}

    atomic_u32 = {
        "Swap": "AtomicSwapU32",
        "Cmpswap": "AtomicSwapU32",
        "Add": "AtomicAddU32",
        "Sub": "AtomicSubU32",
        "Smin": "AtomicSMinI32",
        "Umin": "AtomicUMinU32",
        "Smax": "AtomicSMaxI32",
        "Umax": "AtomicUMaxU32",
        "And": "AtomicAndU32",
        "Or": "AtomicOrU32",
        "Xor": "AtomicXorU32",
        "Inc": "AtomicIncU32",
        "Dec": "AtomicDecU32",
    }
    atomic_f32 = {
        "Fmin": "AtomicFMinF32",
        "Fmax": "AtomicFMaxF32",
        "Fcmpswap": "AtomicFcmpswapF32",
    }
    for prefix in ("Buffer", "Flat", "Global", "Scratch"):
        for name, ir in atomic_u32.items():
            m[f"{prefix}Atomic{name}"] = ir
            m[f"{prefix}Atomic{name}X2"] = ir
        for name, ir in atomic_f32.items():
            m[f"{prefix}Atomic{name}"] = ir
            m[f"{prefix}Atomic{name}X2"] = ir
    m.update(
        {
            "ImageAtomicSwap": "AtomicSwapU32",
            "ImageAtomicCmpswap": "AtomicSwapU32",
            "ImageAtomicAdd": "AtomicAddU32",
            "ImageAtomicSub": "AtomicSubU32",
            "ImageAtomicSmin": "AtomicSMinI32",
            "ImageAtomicSmax": "AtomicSMaxI32",
            "ImageAtomicUMin": "AtomicUMinU32",
            "ImageAtomicUMax": "AtomicUMaxU32",
            "ImageAtomicAnd": "AtomicAndU32",
            "ImageAtomicOr": "AtomicOrU32",
            "ImageAtomicXor": "AtomicXorU32",
            "ImageAtomicInc": "AtomicIncU32",
            "ImageAtomicDec": "AtomicDecU32",
            "ImageAtomicFcmpswap": "AtomicFcmpswapF32",
            "ImageAtomicFmin": "AtomicFMinF32",
            "ImageAtomicFmax": "AtomicFMaxF32",
        }
    )
    m["BufferAtomicCsub"] = "AtomicCsubU32"

    buffer_load = {
        "Ubyte": "BufferLoadUbyte",
        "Sbyte": "BufferLoadSbyte",
        "Ushort": "BufferLoadUshort",
        "Sshort": "BufferLoadSshort",
        "Dword": "BufferLoadDword",
        "Dwordx2": "BufferLoadDword",
        "Dwordx3": "BufferLoadDword",
        "Dwordx4": "BufferLoadDword",
    }
    flat_load = {
        "Ubyte": "FlatLoadUbyte",
        "Sbyte": "FlatLoadSbyte",
        "Ushort": "FlatLoadUshort",
        "Sshort": "FlatLoadSshort",
        "Dword": "FlatLoadDword",
        "Dwordx2": "FlatLoadDword",
        "Dwordx3": "FlatLoadDword",
        "Dwordx4": "FlatLoadDword",
        "UbyteD16": "FlatLoadUbyte",
        "UbyteD16Hi": "FlatLoadUbyte",
        "SbyteD16": "FlatLoadSbyte",
        "SbyteD16Hi": "FlatLoadSbyte",
        "UshortD16": "FlatLoadUshort",
        "UshortD16Hi": "FlatLoadUshort",
        "SshortD16": "FlatLoadSshort",
        "SshortD16Hi": "FlatLoadSshort",
        "ShortD16": "FlatLoadUshort",
        "ShortD16Hi": "FlatLoadUshort",
        "DwordAddtid": "FlatLoadDword",
    }
    for suffix, ir in buffer_load.items():
        m[f"BufferLoad{suffix}"] = ir
    for suffix, ir in flat_load.items():
        m[f"FlatLoad{suffix}"] = ir

    buffer_store = {
        "Byte": "BufferStoreByte",
        "Short": "BufferStoreShort",
        "Dword": "BufferStoreDword",
        "Dwordx2": "BufferStoreDword",
        "Dwordx3": "BufferStoreDword",
        "Dwordx4": "BufferStoreDword",
    }
    flat_store = {
        "Byte": "FlatStoreByte",
        "Short": "FlatStoreShort",
        "Dword": "FlatStoreDword",
        "Dwordx2": "FlatStoreDword",
        "Dwordx3": "FlatStoreDword",
        "Dwordx4": "FlatStoreDword",
        "ByteD16Hi": "FlatStoreByte",
        "ShortD16Hi": "FlatStoreShort",
        "DwordAddtid": "FlatStoreDword",
    }
    for suffix, ir in buffer_store.items():
        m[f"BufferStore{suffix}"] = ir
    for suffix, ir in flat_store.items():
        m[f"FlatStore{suffix}"] = ir

    for stem in ("Buffer", "TBuffer"):
        for comp in ("X", "Xy", "Xyz", "Xyzw"):
            m[f"{stem}LoadFormat{comp}"] = "BufferLoadDword"
            m[f"{stem}StoreFormat{comp}"] = "BufferStoreDword"

    image_base = {
        "ImageLoadPck": "ImageLoadPck",
        "ImageLoadPckSgn": "ImageLoadPckSgn",
        "ImageLoadMipPck": "ImageLoadMipPck",
        "ImageLoadMipPckSgn": "ImageLoadMipPckSgn",
        "ImageMsaaLoad": "ImageMsaaLoad",
        "ImageBvhIntersectRay": "ImageBvhIntersectRay",
        "ImageBvh64IntersectRay": "ImageBvh64IntersectRay",
        "ImageStorePck": "ImageStorePck",
        "ImageStoreMipPck": "ImageStoreMipPck",
    }
    m.update(image_base)

    gather_names = [
        "ImageGather4",
        "ImageGather4B",
        "ImageGather4BCl",
        "ImageGather4BClO",
        "ImageGather4BO",
        "ImageGather4CB",
        "ImageGather4CBCl",
        "ImageGather4CBClO",
        "ImageGather4CBO",
        "ImageGather4CCl",
        "ImageGather4CClO",
        "ImageGather4CL",
        "ImageGather4CLO",
        "ImageGather4Cl",
        "ImageGather4ClO",
        "ImageGather4L",
        "ImageGather4LO",
        "ImageGather4O",
    ]
    for name in gather_names:
        m[name] = "ImageGather4"

    m.update(
        {
            "VDivScaleF32": "DivScaleF32",
            "VDivScaleF64": "DivScaleF64",
            "VMadI64I32": "IMadI64I32",
            "VMadU64U32": "UMadU64U32",
            "VAddcU32": "IAddCarryU32",
            "VMovB32": "MoveU32",
            "VMovreldB32": "MoveRelDestU32",
            "VMovrelsB32": "MoveRelSourceU32",
            "VNop": "ControlNop",
            "VCndmaskB32": "SelectF32Bits",
            "VDot2cF32F16": "Dot2AccF32F16",
            "VInterpP1F32": "LoadInputF32",
            "VInterpP2F32": "LoadInputF32",
            "VInterpMovF32": "LoadInputF32",
            "VCvtF32Ubyte0": "ConvertU32ToF32",
            "VCvtF32Ubyte1": "ConvertU32ToF32",
            "VCvtF32Ubyte2": "ConvertU32ToF32",
            "VCvtF32Ubyte3": "ConvertU32ToF32",
            "DsSwizzleB32": "DsSwizzleB32",
            "Exp": "Export",
            "SAddcU32": "ScalarAddCarryU32",
            "SSubbU32": "ScalarSubBorrowCarryU32",
            "SAndSaveexecB32": "SaveexecB32",
            "SAndSaveexecB64": "SaveexecB64",
            "SAndn1SaveexecB32": "SaveexecB32",
            "SAndn1SaveexecB64": "SaveexecB64",
            "SOrn2SaveexecB64": "SaveexecB64",
            "SCselectB32": "SelectU32",
            "SCselectB64": "SelectU64",
            "SGetpcB64": "MoveU64",
            "SLshl1AddU32": "ShiftLeftAddU32",
            "SLshl2AddU32": "ShiftLeftAddU32",
            "SLshl3AddU32": "ShiftLeftAddU32",
            "SLshl4AddU32": "ShiftLeftAddU32",
            "SBitset0B32": "BitClearU32",
            "SBitset1B32": "BitSetU32",
            "SLoadDwordx2": "SLoadDword",
            "SLoadDwordx4": "SLoadDword",
            "SLoadDwordx8": "SLoadDword",
            "SLoadDwordx16": "SLoadDword",
            "SBufferLoadDwordx2": "SBufferLoadDword",
            "SBufferLoadDwordx4": "SBufferLoadDword",
            "SBufferLoadDwordx8": "SBufferLoadDword",
            "SBufferLoadDwordx16": "SBufferLoadDword",
        }
    )
    return m


def parse_base_decoder_enum() -> set[str]:
    text = (ROOT / "src/graphics/shader/recompiler/ShaderDecoder.h").read_text(encoding="utf-8")
    before, _, _ = text.partition("#define RDNA2_EXTRA_OPCODE")
    return set(re.findall(r"^\s+([A-Z][A-Za-z0-9_]+),", before, re.M))


def referenced_decoder_opcodes(extra_by_table: dict[str, list[str]]) -> set[str]:
    names: set[str] = set()
    for rows in extra_by_table.values():
        for row in rows:
            match = re.search(r"Opcode::([A-Za-z0-9_]+)", row)
            if match:
                names.add(match.group(1))
    for path in GENERATED.glob("Rdna2Extra*.inc"):
        if any(
            token in path.name
            for token in ("Opcodes", "LowerOps", "ExplicitLower", "ScalarOps")
        ):
            continue
        for match in re.finditer(
            r"Opcode::([A-Za-z0-9_]+)", path.read_text(encoding="utf-8")
        ):
            names.add(match.group(1))
    return names


def guess_ir_opcode(decoder_name: str) -> str | None:
    explicit = explicit_lower_mappings()
    if decoder_name in explicit:
        return explicit[decoder_name]
    aliases = {
        "SMulHiI32": "UMulHighU32",
        "SAbsdiffI32": "AbsI32",
        "SAshrI64": "ShiftRightLogicalU64",
        "SBfeI32": "BitFieldExtractU32",
        "SBfeI64": "BitFieldExtractU64",
        "BufferAtomicCmpswap": "AtomicSwapU32",
        "FlatAtomicCmpswap": "AtomicSwapU32",
    }
    if decoder_name in aliases:
        return aliases[decoder_name]
    if decoder_name.startswith("VCmp") and decoder_name.endswith("F32"):
        return "CompareEqF32"
    if decoder_name.startswith("VCmp") and decoder_name.endswith("F16"):
        return "CompareEqF16"
    if decoder_name.startswith("VCmp") and decoder_name.endswith("I32"):
        return "CompareEqI32"
    if decoder_name.startswith("VCmp") and decoder_name.endswith("U32"):
        return "CompareEqU32"
    if decoder_name.startswith("VCmpx") and decoder_name.endswith("F32"):
        return "CompareEqF32"
    if decoder_name.startswith("VCmpx") and decoder_name.endswith("F16"):
        return "CompareEqF16"
    if decoder_name.startswith("VCmpx") and decoder_name.endswith("I32"):
        return "CompareEqI32"
    if decoder_name.startswith("VCmpx") and decoder_name.endswith("U32"):
        return "CompareEqU32"
    if decoder_name.startswith("ImageGather4"):
        return "ImageGather4"
    if decoder_name.startswith("ImageAtomic"):
        if "Fmin" in decoder_name:
            return "AtomicFMinF32"
        if "Fmax" in decoder_name:
            return "AtomicFMaxF32"
        if "Fcmpswap" in decoder_name:
            return "AtomicFcmpswapF32"
        if "Inc" in decoder_name:
            return "AtomicIncU32"
        if "Dec" in decoder_name:
            return "AtomicDecU32"
        if "Cmpswap" in decoder_name or "Swap" in decoder_name:
            return "AtomicSwapU32"
        if "Sub" in decoder_name:
            return "AtomicSubU32"
        if "Smax" in decoder_name:
            return "AtomicSMaxI32"
        if "Smin" in decoder_name:
            return "AtomicSMinI32"
        if "Umax" in decoder_name:
            return "AtomicUMaxU32"
        if "Umin" in decoder_name:
            return "AtomicUMinU32"
    if decoder_name.startswith("FlatLoad") or decoder_name.startswith("BufferLoad"):
        return "BufferLoadDword"
    if decoder_name.startswith("FlatStore") or decoder_name.startswith("BufferStore"):
        return "BufferStoreDword"
    if decoder_name.startswith("S") and decoder_name.endswith("I32"):
        if "Add" in decoder_name:
            return "IAddU32"
        if "Sub" in decoder_name:
            return "ISubU32"
        if "Mul" in decoder_name:
            return "IMulU32"
    if decoder_name.startswith("V") and decoder_name.endswith("F32"):
        if "Add" in decoder_name:
            return "FAddF32"
        if "Mul" in decoder_name:
            return "FMulF32"
        if "Min" in decoder_name:
            return "FMinF32"
        if "Max" in decoder_name:
            return "FMaxF32"
        if "Mad" in decoder_name or "Mac" in decoder_name:
            return "FMadF32"
    if decoder_name.startswith("V") and decoder_name.endswith("F16"):
        if "Add" in decoder_name:
            return "AddF16"
        if "Mul" in decoder_name:
            return "MulF16"
        if "Fma" in decoder_name or "Mac" in decoder_name:
            return "FmaF16"
    return None


def main() -> int:
    matrix = load_matrix_module()
    doc_ops = matrix.parse_doc_tables(matrix.DOC.read_text(encoding="utf-8"))
    code_ops = matrix.parse_cpp_opcode_tables(include_generated=False)
    enum_names = parse_enum(ROOT / "src/graphics/shader/recompiler/ShaderDecoder.h")

    code_keys = {(matrix.family_for_table(e.table), e.opcode_id) for e in code_ops}
    extra_enum: list[str] = []
    extra_by_table: dict[str, list[str]] = {}
    extra_lower: list[str] = []

    for doc_op in doc_ops:
        family = doc_op.family
        lookup_family = "FLAT" if family in ("GLOBAL", "SCRATCH") else family
        if (lookup_family, doc_op.opcode_id) in code_keys:
            continue
        enum_name = doc_to_enum_name(doc_op.name)
        if enum_name not in enum_names:
            extra_enum.append(enum_name)
            enum_names.add(enum_name)

        row: str | None = None
        if family in ("FLAT", "GLOBAL", "SCRATCH"):
            row = flat_memory_entry(doc_op.name, doc_op.opcode_id)
            table = "FLAT_OPS"
        elif family == "MUBUF":
            row = buffer_memory_entry(doc_op.name, doc_op.opcode_id)
            table = "MUBUF_OPS"
        elif family == "MIMG":
            row = mimg_sample_entry(doc_op.name, doc_op.opcode_id)
            if row is not None:
                table = "MIMG_SAMPLE_OPS"
            else:
                row = mimg_gather_entry(doc_op.name, doc_op.opcode_id)
                if row is not None:
                    table = "MIMG_GATHER_OPS"
                else:
                    row = mimg_atomic_entry(doc_op.name, doc_op.opcode_id)
                    if row is not None:
                        table = "MIMG_ATOMIC_OPS"
                    else:
                        row = mimg_base_entry(doc_op.name, doc_op.opcode_id)
                        table = "MIMG_BASE_OPS" if row is not None else None
        elif family in FAMILY_TABLE:
            table = FAMILY_TABLE[family]
            if table not in ("FLAT_OPS", "MUBUF_OPS"):
                row = scalar_entry(doc_op.opcode_id, enum_name)
        else:
            table = None

        if row and table:
            extra_by_table.setdefault(table, []).append(row)

    base_enum = parse_base_decoder_enum()
    needed_enum = referenced_decoder_opcodes(extra_by_table)
    extra_enum = sorted(needed_enum - base_enum)
    enum_names = base_enum | needed_enum

    extra_lower = []
    for decoder_name in sorted(needed_enum - base_enum):
        ir = guess_ir_opcode(decoder_name)
        if ir is None:
            continue
        extra_lower.append(f"    {{Decoder::Opcode::{decoder_name}, Opcode::{ir}}},")

    GENERATED.mkdir(parents=True, exist_ok=True)

    enum_path = GENERATED / "Rdna2ExtraOpcodes.inc"
    enum_path.write_text(
        "// Generated by tools/generate_rdna2_tables.py\n"
        + "\n".join(f"RDNA2_EXTRA_OPCODE({name})" for name in sorted(set(extra_enum)))
        + "\n",
        encoding="utf-8",
    )

    memory_path = GENERATED / "Rdna2ExtraMemoryOps.inc"
    memory_path.write_text(
        "// Generated by tools/generate_rdna2_tables.py\n"
        + "\n".join(dict.fromkeys(extra_by_table.get("FLAT_OPS", [])))
        + "\n",
        encoding="utf-8",
    )

    for table_name, rows in extra_by_table.items():
        if table_name in ("FLAT_OPS", "MUBUF_OPS"):
            continue
        table_path = GENERATED / f"Rdna2Extra{table_name.replace('_OPS', '')}.inc"
        table_path.write_text(
            "// Generated by tools/generate_rdna2_tables.py\n"
            + "\n".join(dict.fromkeys(rows))
            + "\n",
            encoding="utf-8",
        )

    scalar_path = GENERATED / "Rdna2ExtraScalarOps.inc"
    scalar_rows: list[str] = []
    for table_name, rows in extra_by_table.items():
        if table_name not in ("FLAT_OPS", "MUBUF_OPS"):
            scalar_rows.extend(rows)
    scalar_path.write_text("// Legacy aggregate file\n", encoding="utf-8")

    mubuf_path = GENERATED / "Rdna2ExtraMubufOps.inc"
    mubuf_path.write_text(
        "// Generated by tools/generate_rdna2_tables.py\n"
        + "\n".join(dict.fromkeys(extra_by_table.get("MUBUF_OPS", [])))
        + "\n",
        encoding="utf-8",
    )

    for mimg_table in ("MIMG_BASE_OPS", "MIMG_GATHER_OPS", "MIMG_ATOMIC_OPS", "MIMG_SAMPLE_OPS"):
        rows = dict.fromkeys(extra_by_table.get(mimg_table, []))
        if not rows:
            continue
        stem = mimg_table.replace("_OPS", "")
        mimg_path = GENERATED / f"Rdna2Extra{stem}.inc"
        mimg_path.write_text(
            "// Generated by tools/generate_rdna2_tables.py\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    lower_path = GENERATED / "Rdna2ExtraLowerOps.inc"
    lower_path.write_text(
        "// Generated by tools/generate_rdna2_tables.py\n"
        + "\n".join(dict.fromkeys(extra_lower))
        + "\n",
        encoding="utf-8",
    )

    existing_lower = parse_existing_lower_ops()
    explicit_rows: list[str] = []
    for decoder_name, ir_name in sorted(explicit_lower_mappings().items()):
        if decoder_name not in enum_names or decoder_name in existing_lower:
            continue
        explicit_rows.append(f"    {{Decoder::Opcode::{decoder_name}, Opcode::{ir_name}}},")
    explicit_path = GENERATED / "Rdna2ExplicitLowerOps.inc"
    explicit_path.write_text(
        "// Generated by tools/generate_rdna2_tables.py\n"
        + "\n".join(explicit_rows)
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "extra_enum_count": len(set(extra_enum)),
        "extra_memory_rows": len(dict.fromkeys(extra_by_table.get("FLAT_OPS", []))),
        "extra_mubuf_rows": len(dict.fromkeys(extra_by_table.get("MUBUF_OPS", []))),
        "extra_scalar_rows": len(dict.fromkeys(scalar_rows)),
        "extra_lower_rows": len(dict.fromkeys(extra_lower)),
        "explicit_lower_rows": len(explicit_rows),
        "extra_mimg_base_rows": len(dict.fromkeys(extra_by_table.get("MIMG_BASE_OPS", []))),
        "extra_mimg_gather_rows": len(dict.fromkeys(extra_by_table.get("MIMG_GATHER_OPS", []))),
        "extra_mimg_atomic_rows": len(dict.fromkeys(extra_by_table.get("MIMG_ATOMIC_OPS", []))),
        "extra_mimg_sample_rows": len(dict.fromkeys(extra_by_table.get("MIMG_SAMPLE_OPS", []))),
        "tables": {k: len(dict.fromkeys(v)) for k, v in extra_by_table.items()},
    }
    (GENERATED / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
