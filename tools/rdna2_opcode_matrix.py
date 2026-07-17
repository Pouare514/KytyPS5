#!/usr/bin/env python3
"""Parse AMD RDNA2 ISA reference (Ch. 13 opcode tables) and compare with KytyPS5 decoder."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "RDNA 2 Instruction Set Architecture Reference Guide.md"
RECOMPILER = ROOT / "src" / "graphics" / "shader" / "recompiler"
OUTPUT = ROOT / "tools" / "rdna2_coverage.json"

TABLE_FAMILY = {
    "Table 64.": "SOP2",
    "Table 66.": "SOPK",
    "Table 68.": "SOP1",
    "Table 70.": "SOPC",
    "Table 72.": "SOPP",
    "Table 74.": "SMEM",
    "Table 76.": "VOP2",
    "Table 78.": "VOP1",
    "Table 81.": "VOPC",
    "Table 83.": "VOP3A",
    "Table 85.": "VOP3B",
    "Table 87.": "VOP3P",
    "Table 95.": "DS",
    "Table 97.": "MTBUF",
    "Table 99.": "MUBUF",
    "Table 101.": "MIMG",
    "Table 103.": "FLAT",
    "Table 104.": "GLOBAL",
    "Table 105.": "SCRATCH",
}

ENCODING_FAMILIES = {
    0x32: "VINTRP",
    0x33: "VOP3P",
    0x35: "VOP3",
    0x36: "DS",
    0x37: "FLAT",
    0x38: "MUBUF",
    0x39: "GLOBAL",
    0x3A: "MTBUF",
    0x3B: "SCRATCH",
    0x3C: "MIMG",
    0x3D: "SMEM",
    0x3E: "EXP",
}


@dataclass
class DocOpcode:
    family: str
    opcode_id: int
    name: str


@dataclass
class CodeOpcode:
    table: str
    opcode_id: int
    decoder_name: str


def snake_to_pascal(snake: str) -> str:
    parts = snake.split("_")
    out = ""
    for part in parts:
        if not part:
            continue
        if part in ("b", "u", "i", "f", "x"):
            out += part.upper()
        elif len(part) <= 3 and part.isalpha() and part.islower():
            out += part[0].upper() + part[1:]
        else:
            out += part[0].upper() + part[1:]
    return out


def normalize_name(name: str) -> str:
    name = name.strip().upper()
    for prefix in ("S_", "V_", "BUFFER_", "TBUFFER_", "IMAGE_", "FLAT_", "GLOBAL_", "SCRATCH_", "DS_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", "")


OPCODE_LINE = re.compile(
    r"^(\d+)\s+([A-Z][A-Z0-9_]+)(?:\s+(\d+)\s+([A-Z][A-Z0-9_]+))?\s*$"
)
TABLE_HEADER = re.compile(r"^Table (\d+)\.\s+(\S+)\s+Opcodes")


SECTION_BREAK = re.compile(r"^\*\*13\.\d")
PAGE_FOOTER = re.compile(r"\bof \d+\s*$")


def parse_doc_tables(text: str) -> list[DocOpcode]:
    opcodes: list[DocOpcode] = []
    current_family: str | None = None

    for line in text.splitlines():
        stripped = line.strip().strip("`")
        if PAGE_FOOTER.search(stripped):
            continue
        if SECTION_BREAK.match(stripped):
            current_family = None
            continue
        header = TABLE_HEADER.match(stripped)
        if header:
            table_key = f"Table {header.group(1)}."
            current_family = TABLE_FAMILY.get(table_key)
            continue
        if current_family is None:
            continue
        if stripped.startswith("Table "):
            if TABLE_HEADER.match(stripped) is None:
                current_family = None
            continue
        match = OPCODE_LINE.match(stripped)
        if not match:
            continue
        opcodes.append(DocOpcode(current_family, int(match.group(1)), match.group(2)))
        if match.group(3) and match.group(4):
            opcodes.append(DocOpcode(current_family, int(match.group(3)), match.group(4)))
    return opcodes


def parse_cpp_opcode_tables(include_generated: bool = True) -> list[CodeOpcode]:
    entries: list[CodeOpcode] = []
    table_re = re.compile(r"\{(\s*0x[0-9a-fA-F]+u?\s*,\s*Opcode::([A-Za-z0-9_]+))")
    table_name_re = re.compile(r"constexpr\s+\w+\s+(\w+_OPS)\[\]")

    for path in sorted(RECOMPILER.rglob("*.cpp")):
        text = path.read_text(encoding="utf-8", errors="replace")
        current_table = "unknown"
        for line in text.splitlines():
            m = table_name_re.search(line)
            if m:
                current_table = m.group(1)
            for match in table_re.finditer(line):
                opcode_hex = match.group(1).split(",")[0].strip()
                opcode_id = int(opcode_hex.replace("u", ""), 16)
                decoder_name = match.group(2)
                entries.append(CodeOpcode(current_table, opcode_id, decoder_name))

    generated = ROOT / "src" / "graphics" / "shader" / "recompiler" / "generated"
    if include_generated and generated.exists():
        table_aliases = {
            "MubufOps": "MUBUF_OPS",
            "MemoryOps": "FLAT_OPS",
        }
        for path in sorted(generated.glob("Rdna2Extra*.inc")):
            if path.name == "Rdna2ExtraOpcodes.inc" or path.name == "Rdna2ExtraLowerOps.inc":
                continue
            stem = path.stem.replace("Rdna2Extra", "")
            table = table_aliases.get(stem, f"{stem}_OPS")
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in table_re.finditer(text):
                opcode_hex = match.group(1).split(",")[0].strip()
                opcode_id = int(opcode_hex.replace("u", ""), 16)
                decoder_name = match.group(2)
                entries.append(CodeOpcode(table, opcode_id, decoder_name))
    image_path = RECOMPILER / "ImageOps.cpp"
    if image_path.exists():
        text = image_path.read_text(encoding="utf-8", errors="replace")
        sample_re = re.compile(r"SampleInfo\(0x([0-9a-fA-F]+)u?,")
        for match in sample_re.finditer(text):
            entries.append(CodeOpcode("MIMG_SAMPLE_OPS", int(match.group(1), 16), "ImageSample"))
        gather_re = re.compile(
            r"\{0x([0-9a-fA-F]+)u?,\s*\"[^\"]+\",\s*Opcode::([A-Za-z0-9_]+)"
        )
        in_gather = False
        for line in text.splitlines():
            if "MIMG_GATHER_OPS[]" in line:
                in_gather = True
                continue
            if in_gather:
                if line.strip().startswith("};"):
                    in_gather = False
                    continue
                match = gather_re.search(line)
                if match:
                    entries.append(
                        CodeOpcode(
                            "MIMG_GATHER_OPS", int(match.group(1), 16), match.group(2)
                        )
                    )
        atomic_re = re.compile(
            r"\{0x([0-9a-fA-F]+)u?,\s*\"[^\"]+\",\s*Opcode::([A-Za-z0-9_]+)\}"
        )
        in_atomic = False
        for line in text.splitlines():
            if "MIMG_ATOMIC_OPS[]" in line:
                in_atomic = True
                continue
            if in_atomic:
                if line.strip().startswith("};"):
                    in_atomic = False
                    continue
                match = atomic_re.search(line)
                if match:
                    entries.append(
                        CodeOpcode(
                            "MIMG_ATOMIC_OPS", int(match.group(1), 16), match.group(2)
                        )
                    )
        base_re = re.compile(
            r"\{0x([0-9a-fA-F]+)u?,\s*Opcode::([A-Za-z0-9_]+)\}"
        )
        in_base = False
        for line in text.splitlines():
            if "MIMG_BASE_OPS[]" in line:
                in_base = True
                continue
            if in_base:
                if line.strip().startswith("};"):
                    in_base = False
                    continue
                match = base_re.search(line)
                if match:
                    entries.append(
                        CodeOpcode("MIMG_BASE_OPS", int(match.group(1), 16), match.group(2))
                    )
        inline_re = re.compile(
            r"opcode\s*==\s*0x([0-9a-fA-F]+)u?\s*\?\s*Opcode::([A-Za-z0-9_]+)"
        )
        for match in inline_re.finditer(text):
            entries.append(
                CodeOpcode("MIMG_INLINE_OPS", int(match.group(1), 16), match.group(2))
            )
        if generated.exists():
            for path in sorted(generated.glob("Rdna2ExtraMIMG*.inc")):
                inc_text = path.read_text(encoding="utf-8", errors="replace")
                table = "MIMG_BASE_OPS"
                if "GATHER" in path.name:
                    table = "MIMG_GATHER_OPS"
                elif "ATOMIC" in path.name:
                    table = "MIMG_ATOMIC_OPS"
                elif "SAMPLE" in path.name:
                    table = "MIMG_SAMPLE_OPS"
                if table == "MIMG_GATHER_OPS":
                    for match in gather_re.finditer(inc_text):
                        entries.append(
                            CodeOpcode(
                                table, int(match.group(1), 16), match.group(2)
                            )
                        )
                elif table == "MIMG_ATOMIC_OPS":
                    for match in atomic_re.finditer(inc_text):
                        entries.append(
                            CodeOpcode(
                                table, int(match.group(1), 16), match.group(2)
                            )
                        )
                elif table == "MIMG_SAMPLE_OPS":
                    for match in sample_re.finditer(inc_text):
                        entries.append(
                            CodeOpcode(table, int(match.group(1), 16), "ImageSample")
                        )
                else:
                    for match in base_re.finditer(inc_text):
                        entries.append(
                            CodeOpcode(
                                table, int(match.group(1), 16), match.group(2)
                            )
                        )

    return entries


def parse_decoder_enum() -> list[str]:
    header = RECOMPILER / "ShaderDecoder.h"
    text = header.read_text(encoding="utf-8", errors="replace")
    enum_block = re.search(r"enum class Opcode \{([^}]+)\}", text, re.DOTALL)
    if not enum_block:
        return []
    names = []
    for line in enum_block.group(1).splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("//") or line in ("Unknown", "Unsupported"):
            continue
        names.append(line)
    return names


def parse_lower_ops() -> set[str]:
    path = RECOMPILER / "shaderIR" / "ShaderIROpcodes.cpp"
    text = path.read_text(encoding="utf-8", errors="replace")
    for inc in re.findall(
        r'#include\s+"graphics/shader/recompiler/generated/([^"]+)"', text
    ):
        inc_path = RECOMPILER / "generated" / inc
        if inc_path.exists():
            text += inc_path.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"Decoder::Opcode::([A-Za-z0-9_]+)", text))


def family_for_table(table: str) -> str:
    mapping = {
        "SOP2_OPS": "SOP2",
        "SOPK_OPS": "SOPK",
        "SOP1_OPS": "SOP1",
        "SOPC_OPS": "SOPC",
        "SOPP_OPS": "SOPP",
        "SMEM_OPS": "SMEM",
        "VOP2_OPS": "VOP2",
        "VOP1_OPS": "VOP1",
        "VOPC_OPS": "VOPC",
        "VOP3_OPS": "VOP3A",
        "VOP3P_OPS": "VOP3P",
        "MUBUF_OPS": "MUBUF",
        "MTBUF_OPS": "MTBUF",
        "FLAT_OPS": "FLAT",
        "DS_OPS": "DS",
        "MIMG_SAMPLE_OPS": "MIMG",
        "MIMG_GATHER_OPS": "MIMG",
        "MIMG_ATOMIC_OPS": "MIMG",
        "MIMG_BASE_OPS": "MIMG",
        "MIMG_INLINE_OPS": "MIMG",
    }
    return mapping.get(table, table)


def vop3b_opcode_ids(doc_ops: list[DocOpcode]) -> set[int]:
    return {op.opcode_id for op in doc_ops if op.family == "VOP3B"}


def guess_decoder_name(doc: DocOpcode) -> str:
    family_prefix = {
        "SOP2": "S",
        "SOPK": "S",
        "SOP1": "S",
        "SOPC": "S",
        "SOPP": "S",
        "SMEM": "S",
        "VOP2": "V",
        "VOP1": "V",
        "VOPC": "V",
        "VOP3A": "V",
        "VOP3B": "V",
        "VOP3P": "V",
        "DS": "Ds",
        "MUBUF": "Buffer",
        "MTBUF": "TBuffer",
        "MIMG": "Image",
        "FLAT": "Flat",
        "GLOBAL": "Flat",
        "SCRATCH": "Flat",
    }
    raw = doc.name
    for prefix in ("S_", "V_", "BUFFER_", "TBUFFER_", "IMAGE_", "FLAT_", "GLOBAL_", "SCRATCH_", "DS_"):
        if raw.startswith(prefix):
            suffix = raw[len(prefix) :]
            p = family_prefix.get(doc.family, "")
            parts = suffix.lower().split("_")
            camel = p + "".join(piece.capitalize() for piece in parts)
            return camel
    return raw


def build_report() -> dict:
    doc_text = DOC.read_text(encoding="utf-8", errors="replace")
    doc_ops = parse_doc_tables(doc_text)
    code_ops = parse_cpp_opcode_tables()
    enum_ops = parse_decoder_enum()
    lower_ops = parse_lower_ops()

    vop3b_ids = vop3b_opcode_ids(doc_ops)
    code_by_family: dict[str, dict[int, str]] = {}
    for entry in code_ops:
        family = family_for_table(entry.table)
        code_by_family.setdefault(family, {})[entry.opcode_id] = entry.decoder_name
        if entry.table == "VOP3_OPS" and entry.opcode_id in vop3b_ids:
            code_by_family.setdefault("VOP3B", {})[entry.opcode_id] = entry.decoder_name

    enum_set = set(enum_ops)
    missing_decode: list[dict] = []
    covered_decode: list[dict] = []

    for doc_op in doc_ops:
        family = doc_op.family
        if family in ("GLOBAL", "SCRATCH"):
            lookup_family = "FLAT"
        elif family == "VOP3B":
            lookup_family = "VOP3B"
        else:
            lookup_family = family
        decoded = code_by_family.get(lookup_family, {}).get(doc_op.opcode_id)
        if decoded is None and family == "VOP3B":
            decoded = code_by_family.get("VOP3A", {}).get(doc_op.opcode_id)
        guess = guess_decoder_name(doc_op)
        item = {
            "family": family,
            "opcode_id": doc_op.opcode_id,
            "doc_name": doc_op.name,
            "decoded_name": decoded,
            "guessed_name": guess,
            "in_enum": guess in enum_set or (decoded in enum_set if decoded else False),
            "in_lower_ops": (decoded in lower_ops if decoded else False) or guess in lower_ops,
        }
        if decoded is None:
            missing_decode.append(item)
        else:
            covered_decode.append(item)

    lower_hits = sum(1 for item in covered_decode if item["in_lower_ops"])

    families: dict[str, dict] = {}
    for family in sorted({op.family for op in doc_ops}):
        fam_doc = [op for op in doc_ops if op.family == family]
        fam_missing = [op for op in missing_decode if op["family"] == family]
        families[family] = {
            "doc_count": len(fam_doc),
            "decoded_count": len(fam_doc) - len(fam_missing),
            "missing_count": len(fam_missing),
            "coverage_pct": round(100.0 * (len(fam_doc) - len(fam_missing)) / max(len(fam_doc), 1), 1),
        }

    return {
        "summary": {
            "doc_opcode_entries": len(doc_ops),
            "decoded_entries": len(covered_decode),
            "missing_decode_entries": len(missing_decode),
            "decoder_enum_count": len(enum_ops),
            "lower_ops_count": len(lower_ops),
            "lower_ops_doc_hits": lower_hits,
            "lower_ops_doc_coverage_pct": round(
                100.0 * lower_hits / max(len(doc_ops), 1), 1
            ),
            "lower_ops_alias_extra": max(0, len(lower_ops) - lower_hits),
            "decode_coverage_pct": round(100.0 * len(covered_decode) / max(len(doc_ops), 1), 1),
            "encoding_families_routed": ENCODING_FAMILIES,
        },
        "families": families,
        "missing_decode": missing_decode,
    }


def main() -> int:
    if not DOC.exists():
        print(f"error: missing doc at {DOC}", file=sys.stderr)
        return 1
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    print(
        f"RDNA2 coverage: {summary['decoded_entries']}/{summary['doc_opcode_entries']} "
        f"({summary['decode_coverage_pct']}%) decode table entries"
    )
    print(f"Wrote {OUTPUT}")
    for family, stats in sorted(report["families"].items()):
        if stats["missing_count"]:
            print(
                f"  {family}: {stats['decoded_count']}/{stats['doc_count']} "
                f"({stats['coverage_pct']}%) missing={stats['missing_count']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
