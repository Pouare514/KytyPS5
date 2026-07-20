#include "common/assert.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ShaderIR.h"
#include "graphics/shader/recompiler/shaderIR/ShaderIRInternal.h"

#include <algorithm>
#include <fmt/format.h>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

bool LowerScalarBufferLoadDword(const Decoder::Instruction& decoded, BasicBlock* block, Opcode op,
                                std::string* error) {
	for (uint32_t i = 0; i < decoded.data_dwords; i++) {
		Instruction inst;
		inst.pc        = decoded.pc;
		inst.op        = op;
		inst.src_count = 1;
		inst.memory    = OffsetMemoryInfo(decoded, ResourceKind::ScalarBuffer, i);
		// Raw s_load uses a two-SGPR pointer base; s_buffer_load uses a four-SGPR
		// descriptor resource index.
		inst.memory.resource = op == Opcode::SLoadDword ? RawScalarLoadBase(decoded.src0)
		                                                : ResourceIndexFromOperand(decoded.src0);
		if (!LowerRegisterOperand(OffsetDecodedRegister(decoded.dst, i), &inst.dst, error) ||
		    !LowerSourceOperand(decoded.src1, &inst.src[0], error)) {
			return false;
		}
		block->instructions.push_back(inst);
	}
	return true;
}

bool LowerBufferAddressSources(const Decoder::Instruction& decoded, Instruction* inst,
                               uint32_t first_src, std::string* error) {
	uint32_t src = first_src;
	if (decoded.idxen) {
		if (!LowerSourceOperand(decoded.src0, &inst->src[src++], error)) {
			return false;
		}
	}
	if (decoded.offen) {
		const auto offset_source =
		    decoded.idxen ? OffsetDecodedRegister(decoded.src0, 1) : decoded.src0;
		if (!LowerSourceOperand(offset_source, &inst->src[src++], error)) {
			return false;
		}
	}
	if (!LowerSourceOperand(decoded.src2, &inst->src[src++], error)) {
		return false;
	}
	inst->src_count = src;
	return true;
}

bool LowerBufferLoad(const Decoder::Instruction& decoded, BasicBlock* block, std::string* error) {
	const auto ir_op =
	    decoded.data_bits == 8u
	        ? (decoded.data_signed ? Opcode::BufferLoadSbyte : Opcode::BufferLoadUbyte)
	    : decoded.data_bits == 16u
	        ? (decoded.data_signed ? Opcode::BufferLoadSshort : Opcode::BufferLoadUshort)
	        : Opcode::BufferLoadDword;
	const auto count                   = decoded.data_bits == 32u ? decoded.data_dwords : 1u;
	const auto typed_format_components = TypedBufferFormatComponentCount(decoded);
	for (uint32_t i = 0; i < count; i++) {
		if (i >= typed_format_components) {
			if (!LowerMoveImmediateU32(decoded.pc, OffsetDecodedRegister(decoded.dst, i), 0, block,
			                           error)) {
				return false;
			}
			continue;
		}

		Instruction inst;
		inst.pc     = decoded.pc;
		inst.op     = ir_op;
		inst.memory = OffsetBufferMemoryInfo(decoded, i);
		if (!LowerRegisterOperand(OffsetDecodedRegister(decoded.dst, i), &inst.dst, error) ||
		    !LowerBufferAddressSources(decoded, &inst, 0, error)) {
			return false;
		}
		block->instructions.push_back(inst);
	}
	return true;
}

bool LowerBufferStore(const Decoder::Instruction& decoded, BasicBlock* block, std::string* error) {
	const auto ir_op = decoded.data_bits == 8u    ? Opcode::BufferStoreByte
	                   : decoded.data_bits == 16u ? Opcode::BufferStoreShort
	                                              : Opcode::BufferStoreDword;
	auto       count = decoded.data_bits == 32u ? decoded.data_dwords : 1u;
	if (decoded.data_bits == 32u) {
		const auto typed_format_components = TypedBufferFormatComponentCount(decoded);
		if (typed_format_components < count) {
			count = typed_format_components;
		}
	}
	for (uint32_t i = 0; i < count; i++) {
		Instruction inst;
		inst.pc       = decoded.pc;
		inst.op       = ir_op;
		inst.memory   = OffsetBufferMemoryInfo(decoded, i);
		inst.dst.kind = OperandKind::Null;
		if (!LowerSourceOperand(OffsetDecodedRegister(decoded.dst, i), &inst.src[0], error) ||
		    !LowerBufferAddressSources(decoded, &inst, 1, error)) {
			return false;
		}
		block->instructions.push_back(inst);
	}
	return true;
}

Opcode BufferAtomicIrOpcode(Decoder::Opcode opcode) {
	switch (opcode) {
		case Decoder::Opcode::BufferAtomicSwap: return Opcode::AtomicSwapU32;
		case Decoder::Opcode::BufferAtomicAdd: return Opcode::AtomicAddU32;
		case Decoder::Opcode::BufferAtomicSub: return Opcode::AtomicSubU32;
		case Decoder::Opcode::BufferAtomicSMin: return Opcode::AtomicSMinI32;
		case Decoder::Opcode::BufferAtomicUMin: return Opcode::AtomicUMinU32;
		case Decoder::Opcode::BufferAtomicSMax: return Opcode::AtomicSMaxI32;
		case Decoder::Opcode::BufferAtomicUMax: return Opcode::AtomicUMaxU32;
		case Decoder::Opcode::BufferAtomicAnd: return Opcode::AtomicAndU32;
		case Decoder::Opcode::BufferAtomicOr: return Opcode::AtomicOrU32;
		case Decoder::Opcode::BufferAtomicXor: return Opcode::AtomicXorU32;
		case Decoder::Opcode::BufferAtomicInc: return Opcode::AtomicIncU32;
		case Decoder::Opcode::BufferAtomicDec: return Opcode::AtomicDecU32;
		case Decoder::Opcode::BufferAtomicFmin: return Opcode::AtomicFMinF32;
		case Decoder::Opcode::BufferAtomicFmax: return Opcode::AtomicFMaxF32;
		case Decoder::Opcode::BufferAtomicFcmpswap: return Opcode::AtomicFcmpswapF32;
		case Decoder::Opcode::BufferAtomicCsub: return Opcode::AtomicCsubU32;
		case Decoder::Opcode::BufferAtomicIncX2: return Opcode::AtomicIncU32;
		case Decoder::Opcode::BufferAtomicDecX2: return Opcode::AtomicDecU32;
		case Decoder::Opcode::BufferAtomicFminX2: return Opcode::AtomicFMinF32;
		case Decoder::Opcode::BufferAtomicFmaxX2: return Opcode::AtomicFMaxF32;
		case Decoder::Opcode::BufferAtomicFcmpswapX2: return Opcode::AtomicFcmpswapF32;
		default: break;
	}
	const auto name = Decoder::OpcodeToString(opcode);
	if (name.find("atomic_inc") != std::string::npos) {
		return Opcode::AtomicIncU32;
	}
	if (name.find("atomic_dec") != std::string::npos) {
		return Opcode::AtomicDecU32;
	}
	if (name.find("atomic_cmpswap") != std::string::npos ||
	    name.find("atomic_swap") != std::string::npos) {
		return Opcode::AtomicSwapU32;
	}
	if (name.find("atomic_add") != std::string::npos) {
		return Opcode::AtomicAddU32;
	}
	if (name.find("atomic_sub") != std::string::npos) {
		return Opcode::AtomicSubU32;
	}
	if (name.find("atomic_smin") != std::string::npos) {
		return Opcode::AtomicSMinI32;
	}
	if (name.find("atomic_umin") != std::string::npos) {
		return Opcode::AtomicUMinU32;
	}
	if (name.find("atomic_smax") != std::string::npos) {
		return Opcode::AtomicSMaxI32;
	}
	if (name.find("atomic_umax") != std::string::npos) {
		return Opcode::AtomicUMaxU32;
	}
	if (name.find("atomic_and") != std::string::npos) {
		return Opcode::AtomicAndU32;
	}
	if (name.find("atomic_or") != std::string::npos) {
		return Opcode::AtomicOrU32;
	}
	if (name.find("atomic_xor") != std::string::npos) {
		return Opcode::AtomicXorU32;
	}
	return Opcode::AtomicAddU32;
}

bool LowerBufferAtomicDword(const Decoder::Instruction& decoded, BasicBlock* block, Opcode op,
                            std::string* error) {
	Instruction inst;
	inst.pc     = decoded.pc;
	inst.op     = op;
	const bool  is_flat = decoded.family == Decoder::Family::FLAT;
	const bool  implicit_inc_dec = op == Opcode::AtomicIncU32 || op == Opcode::AtomicDecU32;
	const auto  kind =
	    is_flat ? FlatSegmentResourceKind(decoded.memory_segment) : ResourceKind::Buffer;
	inst.memory = MemoryInfoFromDecoded(decoded, kind);
	if (is_flat) {
		inst.memory.resource = 0;
	}
	inst.dst.kind = OperandKind::Null;
	const bool returns_old = decoded.glc || (is_flat && implicit_inc_dec);
	if (returns_old && !LowerRegisterOperand(decoded.dst, &inst.dst, error)) {
		return false;
	}
	if (implicit_inc_dec) {
		inst.src[0].kind = OperandKind::ImmediateU32;
		inst.src[0].imm  = 1;
	} else if ((decoded.glc && !LowerRegisterOperand(decoded.dst, &inst.dst, error)) ||
	           !LowerSourceOperand(decoded.dst, &inst.src[0], error)) {
		return false;
	}
	if (is_flat) {
		inst.src_count = decoded.src_count + 1u;
		for (uint32_t i = 0; i < decoded.src_count && i + 1u < 3u; i++) {
			if (!LowerSourceOperand(DecodedSourceAt(decoded, i), &inst.src[i + 1u], error)) {
				return false;
			}
		}
	} else if (!LowerBufferAddressSources(decoded, &inst, 1, error)) {
		return false;
	}
	block->instructions.push_back(inst);
	return true;
}

bool LowerBufferAtomicOpcode(const Decoder::Instruction& decoded, BasicBlock* block,
                             std::string* error) {
	return LowerBufferAtomicDword(decoded, block, BufferAtomicIrOpcode(decoded.opcode), error);
}

ResourceKind DsMemoryKind(const Decoder::Instruction& decoded) {
	return decoded.gds ? ResourceKind::Gds : ResourceKind::Lds;
}

bool LowerDsRead(const Decoder::Instruction& decoded, BasicBlock* block, std::string* error) {
	const auto ir_op = decoded.data_bits == 8u
	                       ? (decoded.data_signed ? Opcode::DsReadSbyte : Opcode::DsReadUbyte)
	                   : decoded.data_bits == 16u
	                       ? (decoded.data_signed ? Opcode::DsReadSshort : Opcode::DsReadUshort)
	                       : Opcode::DsReadB32;
	const auto count = decoded.data_bits == 32u ? decoded.data_dwords : 1u;
	for (uint32_t i = 0; i < count; i++) {
		Instruction inst;
		inst.pc        = decoded.pc;
		inst.op        = ir_op;
		inst.src_count = 1;
		inst.memory    = OffsetMemoryInfo(decoded, DsMemoryKind(decoded), i);
		if (!LowerRegisterOperand(OffsetDecodedRegister(decoded.dst, i), &inst.dst, error) ||
		    !LowerSourceOperand(decoded.src0, &inst.src[0], error)) {
			return false;
		}
		block->instructions.push_back(inst);
	}
	return true;
}

bool LowerDsRead2(const Decoder::Instruction& decoded, BasicBlock* block, uint32_t dwords_per_read,
                  std::string* error) {
	const uint32_t offsets[] = {decoded.offset, decoded.secondary_offset};
	for (uint32_t read = 0; read < 2u; read++) {
		for (uint32_t dword = 0; dword < dwords_per_read; dword++) {
			const auto  index = read * dwords_per_read + dword;
			Instruction inst;
			inst.pc        = decoded.pc;
			inst.op        = Opcode::DsReadB32;
			inst.src_count = 1;
			inst.memory =
			    ByteOffsetMemoryInfo(decoded, DsMemoryKind(decoded), offsets[read] + dword * 4u);
			if (!LowerRegisterOperand(OffsetDecodedRegister(decoded.dst, index), &inst.dst,
			                          error) ||
			    !LowerSourceOperand(decoded.src0, &inst.src[0], error)) {
				return false;
			}
			block->instructions.push_back(inst);
		}
	}
	return true;
}

bool LowerDsAtomicU32(const Decoder::Instruction& decoded, BasicBlock* block, Opcode op,
                      bool return_old, std::string* error) {
	Instruction inst;
	inst.pc        = decoded.pc;
	inst.op        = op;
	inst.src_count = 2;
	inst.memory    = MemoryInfoFromDecoded(decoded, DsMemoryKind(decoded));
	inst.dst.kind  = return_old ? OperandKind::Register : OperandKind::Null;
	if ((return_old && !LowerRegisterOperand(decoded.dst, &inst.dst, error)) ||
	    !LowerSourceOperand(decoded.src1, &inst.src[0], error) ||
	    !LowerSourceOperand(decoded.src0, &inst.src[1], error)) {
		return false;
	}
	block->instructions.push_back(inst);
	return true;
}

bool LowerDsSwizzleB32(const Decoder::Instruction& decoded, BasicBlock* block, std::string* error) {
	Instruction inst;
	inst.pc          = decoded.pc;
	inst.op          = Opcode::DsSwizzleB32;
	inst.src_count   = 2;
	inst.src[1].kind = OperandKind::ImmediateU32;
	inst.src[1].imm  = decoded.offset & 0xffffu;
	if (!LowerRegisterOperand(decoded.dst, &inst.dst, error) ||
	    !LowerSourceOperand(decoded.src0, &inst.src[0], error)) {
		return false;
	}
	block->instructions.push_back(inst);
	return true;
}

bool LowerDsFloatMinMaxF32(const Decoder::Instruction& decoded, BasicBlock* block, Opcode op,
                           std::string* error) {
	Instruction inst;
	inst.pc              = decoded.pc;
	inst.op              = op;
	inst.src_count       = 2;
	inst.memory          = MemoryInfoFromDecoded(decoded, DsMemoryKind(decoded));
	inst.memory.resource = 0;
	inst.dst.kind        = OperandKind::Null;
	if (!LowerSourceOperand(decoded.src1, &inst.src[0], error) ||
	    !LowerSourceOperand(decoded.src0, &inst.src[1], error) ||
	    !LowerSourceOperand(decoded.src2, &inst.src[2], error)) {
		return false;
	}
	inst.src_count = 3;
	block->instructions.push_back(inst);
	return true;
}

bool LowerDsWriteAddtidB32(const Decoder::Instruction& decoded, BasicBlock* block,
                           std::string* error) {
	Instruction inst;
	inst.pc              = decoded.pc;
	inst.op              = Opcode::DsWriteAddtidB32;
	inst.src_count       = 2;
	inst.memory          = MemoryInfoFromDecoded(decoded, DsMemoryKind(decoded));
	inst.memory.resource = 0;
	inst.dst.kind        = OperandKind::Null;
	const auto m0        = M0Operand();
	if (!LowerSourceOperand(decoded.src1, &inst.src[0], error) ||
	    !LowerSourceOperand(m0, &inst.src[1], error)) {
		return false;
	}
	block->instructions.push_back(inst);
	return true;
}

bool LowerDsReadAddtidB32(const Decoder::Instruction& decoded, BasicBlock* block,
                          std::string* error) {
	Instruction inst;
	inst.pc              = decoded.pc;
	inst.op              = Opcode::DsReadAddtidB32;
	inst.src_count       = 1;
	inst.memory          = MemoryInfoFromDecoded(decoded, DsMemoryKind(decoded));
	inst.memory.resource = 0;
	const auto m0        = M0Operand();
	if (!LowerRegisterOperand(decoded.dst, &inst.dst, error) ||
	    !LowerSourceOperand(m0, &inst.src[0], error)) {
		return false;
	}
	block->instructions.push_back(inst);
	return true;
}

bool LowerDsAppendConsume(const Decoder::Instruction& decoded, BasicBlock* block, Opcode op,
                          std::string* error) {
	Instruction inst;
	inst.pc        = decoded.pc;
	inst.op        = op;
	inst.src_count = 1;
	inst.memory    = MemoryInfoFromDecoded(decoded, DsMemoryKind(decoded));
	const auto m0  = M0Operand();
	if (!LowerRegisterOperand(decoded.dst, &inst.dst, error) ||
	    !LowerSourceOperand(m0, &inst.src[0], error)) {
		return false;
	}
	block->instructions.push_back(inst);
	return true;
}

Opcode ImageAtomicIrOpcode(Decoder::Opcode opcode) {
	switch (opcode) {
		case Decoder::Opcode::ImageAtomicAdd: return Opcode::AtomicAddU32;
		case Decoder::Opcode::ImageAtomicUMin: return Opcode::AtomicUMinU32;
		case Decoder::Opcode::ImageAtomicUMax: return Opcode::AtomicUMaxU32;
		case Decoder::Opcode::ImageAtomicAnd: return Opcode::AtomicAndU32;
		case Decoder::Opcode::ImageAtomicOr: return Opcode::AtomicOrU32;
		case Decoder::Opcode::ImageAtomicXor: return Opcode::AtomicXorU32;
		default: return Opcode::AtomicAddU32;
	}
}

bool LowerImageAtomicU32(const Decoder::Instruction& decoded, BasicBlock* block,
                         std::string* error) {
	Instruction inst;
	inst.pc             = decoded.pc;
	inst.op             = ImageAtomicIrOpcode(decoded.opcode);
	inst.src_count      = 2;
	inst.memory         = MemoryInfoFromDecoded(decoded, ResourceKind::StorageImageUint);
	inst.memory.sampler = 0;
	inst.dst.kind       = OperandKind::Null;
	if ((decoded.glc && !LowerRegisterOperand(decoded.dst, &inst.dst, error)) ||
	    !LowerSourceOperand(decoded.dst, &inst.src[0], error) ||
	    !LowerSourceOperand(decoded.src0, &inst.src[1], error)) {
		return false;
	}
	block->instructions.push_back(inst);
	return true;
}

bool LowerFlatLoad(const Decoder::Instruction& decoded, BasicBlock* block, std::string* error) {
	const auto kind  = FlatSegmentResourceKind(decoded.memory_segment);
	const auto ir_op = decoded.data_bits == 8u
	                       ? (decoded.data_signed ? Opcode::FlatLoadSbyte : Opcode::FlatLoadUbyte)
	                   : decoded.data_bits == 16u
	                       ? (decoded.data_signed ? Opcode::FlatLoadSshort : Opcode::FlatLoadUshort)
	                       : Opcode::FlatLoadDword;
	const auto count = decoded.data_bits == 32u ? decoded.data_dwords : 1u;
	if (decoded.data_bits == 32u && count > 1u) {
		Instruction inst;
		inst.pc              = decoded.pc;
		inst.op              = ir_op;
		inst.src_count       = decoded.src_count;
		inst.memory          = MemoryInfoFromDecoded(decoded, kind);
		inst.memory.resource = 0;
		if (!LowerRegisterOperand(decoded.dst, &inst.dst, error)) {
			return false;
		}
		for (uint32_t i = 0; i < decoded.src_count && i < 3u; i++) {
			if (!LowerSourceOperand(DecodedSourceAt(decoded, i), &inst.src[i], error)) {
				return false;
			}
		}
		block->instructions.push_back(inst);
		return true;
	}

	for (uint32_t dword = 0; dword < count; dword++) {
		Instruction inst;
		inst.pc              = decoded.pc;
		inst.op              = ir_op;
		inst.src_count       = decoded.src_count;
		inst.memory          = OffsetMemoryInfo(decoded, kind, dword);
		inst.memory.resource = 0;
		if (!LowerRegisterOperand(OffsetDecodedRegister(decoded.dst, dword), &inst.dst, error)) {
			return false;
		}
		for (uint32_t i = 0; i < decoded.src_count && i < 3u; i++) {
			if (!LowerSourceOperand(DecodedSourceAt(decoded, i), &inst.src[i], error)) {
				return false;
			}
		}
		block->instructions.push_back(inst);
	}
	return true;
}

bool LowerFlatStore(const Decoder::Instruction& decoded, BasicBlock* block, std::string* error) {
	const auto kind  = FlatSegmentResourceKind(decoded.memory_segment);
	const auto ir_op = decoded.data_bits == 8u    ? Opcode::FlatStoreByte
	                   : decoded.data_bits == 16u ? Opcode::FlatStoreShort
	                                              : Opcode::FlatStoreDword;
	const auto count = decoded.data_bits == 32u ? decoded.data_dwords : 1u;
	for (uint32_t dword = 0; dword < count; dword++) {
		Instruction inst;
		inst.pc              = decoded.pc;
		inst.op              = ir_op;
		inst.src_count       = decoded.src_count + 1u;
		inst.memory          = OffsetMemoryInfo(decoded, kind, dword);
		inst.memory.resource = 0;
		inst.dst.kind        = OperandKind::Null;
		if (!LowerSourceOperand(OffsetDecodedRegister(decoded.dst, dword), &inst.src[0], error)) {
			return false;
		}
		for (uint32_t i = 0; i < decoded.src_count && i + 1u < 3u; i++) {
			if (!LowerSourceOperand(DecodedSourceAt(decoded, i), &inst.src[i + 1u], error)) {
				return false;
			}
		}
		block->instructions.push_back(inst);
	}
	return true;
}

bool LowerDsWrite(const Decoder::Instruction& decoded, BasicBlock* block, std::string* error) {
	const auto ir_op = decoded.data_bits == 8u    ? Opcode::DsWriteByte
	                   : decoded.data_bits == 16u ? Opcode::DsWriteShort
	                                              : Opcode::DsWriteB32;
	const auto count = decoded.data_bits == 32u ? decoded.data_dwords : 1u;
	for (uint32_t i = 0; i < count; i++) {
		Instruction inst;
		inst.pc        = decoded.pc;
		inst.op        = ir_op;
		inst.src_count = 2;
		inst.memory    = OffsetMemoryInfo(decoded, DsMemoryKind(decoded), i);
		inst.dst.kind  = OperandKind::Null;
		if (!LowerSourceOperand(OffsetDecodedRegister(decoded.src1, i), &inst.src[0], error) ||
		    !LowerSourceOperand(decoded.src0, &inst.src[1], error)) {
			return false;
		}
		block->instructions.push_back(inst);
	}
	return true;
}

bool LowerDsWrite2(const Decoder::Instruction& decoded, BasicBlock* block,
                   uint32_t dwords_per_write, std::string* error) {
	const Decoder::Operand* data[]    = {&decoded.src1, &decoded.src2};
	const uint32_t          offsets[] = {decoded.offset, decoded.secondary_offset};
	for (uint32_t write = 0; write < 2u; write++) {
		for (uint32_t dword = 0; dword < dwords_per_write; dword++) {
			Instruction inst;
			inst.pc        = decoded.pc;
			inst.op        = Opcode::DsWriteB32;
			inst.src_count = 2;
			inst.memory =
			    ByteOffsetMemoryInfo(decoded, DsMemoryKind(decoded), offsets[write] + dword * 4u);
			inst.dst.kind = OperandKind::Null;
			if (!LowerSourceOperand(OffsetDecodedRegister(*data[write], dword), &inst.src[0],
			                        error) ||
			    !LowerSourceOperand(decoded.src0, &inst.src[1], error)) {
				return false;
			}
			block->instructions.push_back(inst);
		}
	}
	return true;
}

bool LowerImageOperation(const Decoder::Instruction& decoded, BasicBlock* block,
                         std::string* error) {
	Instruction inst;
	inst.pc = decoded.pc;
	if (decoded.opcode == Decoder::Opcode::ImageStore ||
	    decoded.opcode == Decoder::Opcode::ImageStoreMip ||
	    decoded.opcode == Decoder::Opcode::ImageStorePck ||
	    decoded.opcode == Decoder::Opcode::ImageStoreMipPck) {
		inst.op = decoded.opcode == Decoder::Opcode::ImageStorePck       ? Opcode::ImageStorePck
		        : decoded.opcode == Decoder::Opcode::ImageStoreMipPck ? Opcode::ImageStoreMipPck
		                                                              : Opcode::ImageStore;
		inst.src_count      = 2;
		inst.memory         = MemoryInfoFromDecoded(decoded, ResourceKind::StorageImage);
		inst.memory.sampler = 0;
		inst.dst.kind       = OperandKind::Null;
		if (!LowerSourceOperand(decoded.dst, &inst.src[0], error) ||
		    !LowerSourceOperand(decoded.src0, &inst.src[1], error)) {
			return false;
		}
		block->instructions.push_back(inst);
		return true;
	}

	inst.op = [&]() -> Opcode {
		switch (decoded.opcode) {
			case Decoder::Opcode::ImageGetResinfo: return Opcode::ImageGetResinfo;
			case Decoder::Opcode::ImageGetLod: return Opcode::ImageGetLod;
			case Decoder::Opcode::ImageLoadPck: return Opcode::ImageLoadPck;
			case Decoder::Opcode::ImageLoadPckSgn: return Opcode::ImageLoadPckSgn;
			case Decoder::Opcode::ImageLoadMipPck: return Opcode::ImageLoadMipPck;
			case Decoder::Opcode::ImageLoadMipPckSgn: return Opcode::ImageLoadMipPckSgn;
			case Decoder::Opcode::ImageMsaaLoad: return Opcode::ImageMsaaLoad;
			case Decoder::Opcode::ImageBvhIntersectRay: return Opcode::ImageBvhIntersectRay;
			case Decoder::Opcode::ImageBvh64IntersectRay: return Opcode::ImageBvh64IntersectRay;
			case Decoder::Opcode::ImageLoad:
			case Decoder::Opcode::ImageLoadMip: return Opcode::ImageLoad;
			default: break;
		}
		if (decoded.opcode == Decoder::Opcode::ImageGather4Lz ||
		    decoded.opcode == Decoder::Opcode::ImageGather4C ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CLz ||
		    decoded.opcode == Decoder::Opcode::ImageGather4LzO ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CO ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CLzO ||
		    decoded.opcode == Decoder::Opcode::ImageGather4H ||
		    decoded.opcode == Decoder::Opcode::ImageGather4 ||
		    decoded.opcode == Decoder::Opcode::ImageGather4Cl ||
		    decoded.opcode == Decoder::Opcode::ImageGather4L ||
		    decoded.opcode == Decoder::Opcode::ImageGather4B ||
		    decoded.opcode == Decoder::Opcode::ImageGather4BCl ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CCl ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CL ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CB ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CBCl ||
		    decoded.opcode == Decoder::Opcode::ImageGather4O ||
		    decoded.opcode == Decoder::Opcode::ImageGather4ClO ||
		    decoded.opcode == Decoder::Opcode::ImageGather4LO ||
		    decoded.opcode == Decoder::Opcode::ImageGather4BO ||
		    decoded.opcode == Decoder::Opcode::ImageGather4BClO ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CClO ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CLO ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CBO ||
		    decoded.opcode == Decoder::Opcode::ImageGather4CBClO) {
			return Opcode::ImageGather4;
		}
		return Opcode::ImageSample;
	}();
	inst.src_count = 1;
	inst.memory    = MemoryInfoFromDecoded(decoded, ResourceKind::Image);
	if (!LowerRegisterOperand(decoded.dst, &inst.dst, error) ||
	    !LowerSourceOperand(decoded.src0, &inst.src[0], error)) {
		return false;
	}
	block->instructions.push_back(inst);
	return true;
}

} // namespace

bool LowerMemoryInstruction(const Decoder::Instruction& decoded, BasicBlock* block,
                            std::string* error) {
	switch (decoded.opcode) {
		case Decoder::Opcode::SLoadDword:
		case Decoder::Opcode::SLoadDwordx2:
		case Decoder::Opcode::SLoadDwordx4:
		case Decoder::Opcode::SLoadDwordx8:
		case Decoder::Opcode::SLoadDwordx16:
			return LowerScalarBufferLoadDword(decoded, block, Opcode::SLoadDword, error);
		case Decoder::Opcode::SBufferLoadDword:
		case Decoder::Opcode::SBufferLoadDwordx2:
		case Decoder::Opcode::SBufferLoadDwordx4:
		case Decoder::Opcode::SBufferLoadDwordx8:
		case Decoder::Opcode::SBufferLoadDwordx16:
			return LowerScalarBufferLoadDword(decoded, block, Opcode::SBufferLoadDword, error);
		case Decoder::Opcode::BufferLoadUbyte:
		case Decoder::Opcode::BufferLoadSbyte:
		case Decoder::Opcode::BufferLoadUshort:
		case Decoder::Opcode::BufferLoadSshort:
		case Decoder::Opcode::BufferLoadDword:
		case Decoder::Opcode::BufferLoadDwordx2:
		case Decoder::Opcode::BufferLoadDwordx3:
		case Decoder::Opcode::BufferLoadDwordx4:
		case Decoder::Opcode::BufferLoadFormatX:
		case Decoder::Opcode::BufferLoadFormatXy:
		case Decoder::Opcode::BufferLoadFormatXyz:
		case Decoder::Opcode::BufferLoadFormatXyzw:
		case Decoder::Opcode::TBufferLoadFormatX:
		case Decoder::Opcode::TBufferLoadFormatXy:
		case Decoder::Opcode::TBufferLoadFormatXyz:
		case Decoder::Opcode::TBufferLoadFormatXyzw: return LowerBufferLoad(decoded, block, error);
		case Decoder::Opcode::BufferStoreDword:
		case Decoder::Opcode::BufferStoreDwordx2:
		case Decoder::Opcode::BufferStoreDwordx3:
		case Decoder::Opcode::BufferStoreDwordx4:
		case Decoder::Opcode::BufferStoreByte:
		case Decoder::Opcode::BufferStoreShort:
		case Decoder::Opcode::BufferStoreFormatX:
		case Decoder::Opcode::BufferStoreFormatXy:
		case Decoder::Opcode::BufferStoreFormatXyz:
		case Decoder::Opcode::BufferStoreFormatXyzw:
		case Decoder::Opcode::TBufferStoreFormatX:
		case Decoder::Opcode::TBufferStoreFormatXy:
		case Decoder::Opcode::TBufferStoreFormatXyz:
		case Decoder::Opcode::TBufferStoreFormatXyzw:
			return LowerBufferStore(decoded, block, error);
		case Decoder::Opcode::BufferAtomicSwap:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicSwapU32, error);
		case Decoder::Opcode::BufferAtomicAdd:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicAddU32, error);
		case Decoder::Opcode::BufferAtomicSub:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicSubU32, error);
		case Decoder::Opcode::BufferAtomicSMin:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicSMinI32, error);
		case Decoder::Opcode::BufferAtomicUMin:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicUMinU32, error);
		case Decoder::Opcode::BufferAtomicSMax:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicSMaxI32, error);
		case Decoder::Opcode::BufferAtomicUMax:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicUMaxU32, error);
		case Decoder::Opcode::BufferAtomicAnd:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicAndU32, error);
		case Decoder::Opcode::BufferAtomicOr:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicOrU32, error);
		case Decoder::Opcode::BufferAtomicXor:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicXorU32, error);
		case Decoder::Opcode::BufferAtomicSwapX2:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicSwapU32, error);
		case Decoder::Opcode::BufferAtomicAddX2:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicAddU32, error);
		case Decoder::Opcode::BufferAtomicSubX2:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicSubU32, error);
		case Decoder::Opcode::BufferAtomicSminX2:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicSMinI32, error);
		case Decoder::Opcode::BufferAtomicUminX2:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicUMinU32, error);
		case Decoder::Opcode::BufferAtomicSmaxX2:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicSMaxI32, error);
		case Decoder::Opcode::BufferAtomicUmaxX2:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicUMaxU32, error);
		case Decoder::Opcode::BufferAtomicAndX2:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicAndU32, error);
		case Decoder::Opcode::BufferAtomicOrX2:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicOrU32, error);
		case Decoder::Opcode::BufferAtomicXorX2:
			return LowerBufferAtomicDword(decoded, block, Opcode::AtomicXorU32, error);
		case Decoder::Opcode::BufferAtomicCmpswap:
		case Decoder::Opcode::BufferAtomicCmpswapX2:
		case Decoder::Opcode::BufferAtomicInc:
		case Decoder::Opcode::BufferAtomicDec:
		case Decoder::Opcode::BufferAtomicIncX2:
		case Decoder::Opcode::BufferAtomicDecX2:
		case Decoder::Opcode::BufferAtomicFcmpswap:
		case Decoder::Opcode::BufferAtomicFcmpswapX2:
		case Decoder::Opcode::BufferAtomicFmin:
		case Decoder::Opcode::BufferAtomicFmax:
		case Decoder::Opcode::BufferAtomicFminX2:
		case Decoder::Opcode::BufferAtomicFmaxX2:
		case Decoder::Opcode::BufferAtomicCsub:
			return LowerBufferAtomicOpcode(decoded, block, error);
		case Decoder::Opcode::FlatLoadUbyte:
		case Decoder::Opcode::FlatLoadSbyte:
		case Decoder::Opcode::FlatLoadUshort:
		case Decoder::Opcode::FlatLoadSshort:
		case Decoder::Opcode::FlatLoadDword:
		case Decoder::Opcode::FlatLoadDwordx2:
		case Decoder::Opcode::FlatLoadDwordx3:
		case Decoder::Opcode::FlatLoadDwordx4:
		case Decoder::Opcode::FlatLoadShortD16:
		case Decoder::Opcode::FlatLoadShortD16Hi:
		case Decoder::Opcode::FlatLoadUbyteD16:
		case Decoder::Opcode::FlatLoadUbyteD16Hi:
		case Decoder::Opcode::FlatLoadSbyteD16:
		case Decoder::Opcode::FlatLoadSbyteD16Hi:
		case Decoder::Opcode::FlatLoadDwordAddtid: return LowerFlatLoad(decoded, block, error);
		case Decoder::Opcode::FlatStoreByte:
		case Decoder::Opcode::FlatStoreShort:
		case Decoder::Opcode::FlatStoreDword:
		case Decoder::Opcode::FlatStoreDwordx2:
		case Decoder::Opcode::FlatStoreDwordx3:
		case Decoder::Opcode::FlatStoreDwordx4:
		case Decoder::Opcode::FlatStoreByteD16Hi:
		case Decoder::Opcode::FlatStoreShortD16Hi:
		case Decoder::Opcode::FlatStoreDwordAddtid: return LowerFlatStore(decoded, block, error);
		case Decoder::Opcode::DsAddU32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicAddU32, false, error);
		case Decoder::Opcode::DsAddRtnU32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicAddU32, true, error);
		case Decoder::Opcode::DsSubU32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicSubU32, false, error);
		case Decoder::Opcode::DsSubRtnU32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicSubU32, true, error);
		case Decoder::Opcode::DsMinI32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicSMinI32, false, error);
		case Decoder::Opcode::DsMinRtnI32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicSMinI32, true, error);
		case Decoder::Opcode::DsMaxI32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicSMaxI32, false, error);
		case Decoder::Opcode::DsMaxRtnI32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicSMaxI32, true, error);
		case Decoder::Opcode::DsMinU32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicUMinU32, false, error);
		case Decoder::Opcode::DsMinRtnU32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicUMinU32, true, error);
		case Decoder::Opcode::DsMaxU32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicUMaxU32, false, error);
		case Decoder::Opcode::DsMaxRtnU32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicUMaxU32, true, error);
		case Decoder::Opcode::DsAndB32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicAndU32, false, error);
		case Decoder::Opcode::DsAndRtnB32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicAndU32, true, error);
		case Decoder::Opcode::DsOrB32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicOrU32, false, error);
		case Decoder::Opcode::DsOrRtnB32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicOrU32, true, error);
		case Decoder::Opcode::DsXorB32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicXorU32, false, error);
		case Decoder::Opcode::DsXorRtnB32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicXorU32, true, error);
		case Decoder::Opcode::DsWrxchgRtnB32:
			return LowerDsAtomicU32(decoded, block, Opcode::AtomicSwapU32, true, error);
		case Decoder::Opcode::DsMinF32:
			return LowerDsFloatMinMaxF32(decoded, block, Opcode::DsMinF32, error);
		case Decoder::Opcode::DsMaxF32:
			return LowerDsFloatMinMaxF32(decoded, block, Opcode::DsMaxF32, error);
		case Decoder::Opcode::DsSwizzleB32: return LowerDsSwizzleB32(decoded, block, error);
		case Decoder::Opcode::DsConsume:
			return LowerDsAppendConsume(decoded, block, Opcode::DsConsume, error);
		case Decoder::Opcode::DsAppend:
			return LowerDsAppendConsume(decoded, block, Opcode::DsAppend, error);
		case Decoder::Opcode::DsWriteAddtidB32: return LowerDsWriteAddtidB32(decoded, block, error);
		case Decoder::Opcode::DsReadAddtidB32: return LowerDsReadAddtidB32(decoded, block, error);
		case Decoder::Opcode::DsRead2B32: return LowerDsRead2(decoded, block, 1, error);
		case Decoder::Opcode::DsRead2B64:
		case Decoder::Opcode::DsRead2St64B64: return LowerDsRead2(decoded, block, 2, error);
		case Decoder::Opcode::DsReadSbyte:
		case Decoder::Opcode::DsReadUbyte:
		case Decoder::Opcode::DsReadSshort:
		case Decoder::Opcode::DsReadUshort:
		case Decoder::Opcode::DsReadB32:
		case Decoder::Opcode::DsReadB64:
		case Decoder::Opcode::DsReadB96:
		case Decoder::Opcode::DsReadB128: return LowerDsRead(decoded, block, error);
		case Decoder::Opcode::DsWrite2B32:
		case Decoder::Opcode::DsWrite2St64B32: return LowerDsWrite2(decoded, block, 1, error);
		case Decoder::Opcode::DsWrite2B64:
		case Decoder::Opcode::DsWrite2St64B64: return LowerDsWrite2(decoded, block, 2, error);
		case Decoder::Opcode::DsWriteByte:
		case Decoder::Opcode::DsWriteShort:
		case Decoder::Opcode::DsWriteB32:
		case Decoder::Opcode::DsWriteB64:
		case Decoder::Opcode::DsWriteB96:
		case Decoder::Opcode::DsWriteB128: return LowerDsWrite(decoded, block, error);
		case Decoder::Opcode::ImageGetResinfo:
		case Decoder::Opcode::ImageGetLod:
		case Decoder::Opcode::ImageLoad:
		case Decoder::Opcode::ImageLoadMip:
		case Decoder::Opcode::ImageLoadPck:
		case Decoder::Opcode::ImageLoadPckSgn:
		case Decoder::Opcode::ImageLoadMipPck:
		case Decoder::Opcode::ImageLoadMipPckSgn:
		case Decoder::Opcode::ImageMsaaLoad:
		case Decoder::Opcode::ImageBvhIntersectRay:
		case Decoder::Opcode::ImageBvh64IntersectRay:
		case Decoder::Opcode::ImageStore:
		case Decoder::Opcode::ImageStoreMip:
		case Decoder::Opcode::ImageStorePck:
		case Decoder::Opcode::ImageStoreMipPck:
		case Decoder::Opcode::ImageGather4:
		case Decoder::Opcode::ImageGather4Cl:
		case Decoder::Opcode::ImageGather4L:
		case Decoder::Opcode::ImageGather4B:
		case Decoder::Opcode::ImageGather4BCl:
		case Decoder::Opcode::ImageGather4Lz:
		case Decoder::Opcode::ImageGather4C:
		case Decoder::Opcode::ImageGather4CCl:
		case Decoder::Opcode::ImageGather4CL:
		case Decoder::Opcode::ImageGather4CB:
		case Decoder::Opcode::ImageGather4CBCl:
		case Decoder::Opcode::ImageGather4CLz:
		case Decoder::Opcode::ImageGather4O:
		case Decoder::Opcode::ImageGather4ClO:
		case Decoder::Opcode::ImageGather4LO:
		case Decoder::Opcode::ImageGather4BO:
		case Decoder::Opcode::ImageGather4BClO:
		case Decoder::Opcode::ImageGather4LzO:
		case Decoder::Opcode::ImageGather4CO:
		case Decoder::Opcode::ImageGather4CClO:
		case Decoder::Opcode::ImageGather4CLO:
		case Decoder::Opcode::ImageGather4CBO:
		case Decoder::Opcode::ImageGather4CBClO:
		case Decoder::Opcode::ImageGather4CLzO:
		case Decoder::Opcode::ImageGather4H:
		case Decoder::Opcode::ImageSample: return LowerImageOperation(decoded, block, error);
		case Decoder::Opcode::ImageAtomicAdd:
		case Decoder::Opcode::ImageAtomicUMin:
		case Decoder::Opcode::ImageAtomicUMax:
		case Decoder::Opcode::ImageAtomicAnd:
		case Decoder::Opcode::ImageAtomicOr:
		case Decoder::Opcode::ImageAtomicXor:
		case Decoder::Opcode::ImageAtomicSwap:
		case Decoder::Opcode::ImageAtomicCmpswap:
		case Decoder::Opcode::ImageAtomicSub:
		case Decoder::Opcode::ImageAtomicSmin:
		case Decoder::Opcode::ImageAtomicSmax:
		case Decoder::Opcode::ImageAtomicInc:
		case Decoder::Opcode::ImageAtomicDec:
		case Decoder::Opcode::ImageAtomicFcmpswap:
		case Decoder::Opcode::ImageAtomicFmin:
		case Decoder::Opcode::ImageAtomicFmax: return LowerImageAtomicU32(decoded, block, error);
		default: return false;
	}
}

bool IsMemoryOpcode(Decoder::Opcode opcode) {
	switch (opcode) {
		case Decoder::Opcode::SLoadDword:
		case Decoder::Opcode::SLoadDwordx2:
		case Decoder::Opcode::SLoadDwordx4:
		case Decoder::Opcode::SLoadDwordx8:
		case Decoder::Opcode::SLoadDwordx16:
		case Decoder::Opcode::SBufferLoadDword:
		case Decoder::Opcode::SBufferLoadDwordx2:
		case Decoder::Opcode::SBufferLoadDwordx4:
		case Decoder::Opcode::SBufferLoadDwordx8:
		case Decoder::Opcode::SBufferLoadDwordx16:
		case Decoder::Opcode::BufferLoadUbyte:
		case Decoder::Opcode::BufferLoadSbyte:
		case Decoder::Opcode::BufferLoadUshort:
		case Decoder::Opcode::BufferLoadSshort:
		case Decoder::Opcode::BufferLoadDword:
		case Decoder::Opcode::BufferLoadDwordx2:
		case Decoder::Opcode::BufferLoadDwordx3:
		case Decoder::Opcode::BufferLoadDwordx4:
		case Decoder::Opcode::BufferLoadFormatX:
		case Decoder::Opcode::BufferLoadFormatXy:
		case Decoder::Opcode::BufferLoadFormatXyz:
		case Decoder::Opcode::BufferLoadFormatXyzw:
		case Decoder::Opcode::TBufferLoadFormatX:
		case Decoder::Opcode::TBufferLoadFormatXy:
		case Decoder::Opcode::TBufferLoadFormatXyz:
		case Decoder::Opcode::TBufferLoadFormatXyzw:
		case Decoder::Opcode::BufferStoreDword:
		case Decoder::Opcode::BufferStoreDwordx2:
		case Decoder::Opcode::BufferStoreDwordx3:
		case Decoder::Opcode::BufferStoreDwordx4:
		case Decoder::Opcode::BufferStoreByte:
		case Decoder::Opcode::BufferStoreShort:
		case Decoder::Opcode::BufferStoreFormatX:
		case Decoder::Opcode::BufferStoreFormatXy:
		case Decoder::Opcode::BufferStoreFormatXyz:
		case Decoder::Opcode::BufferStoreFormatXyzw:
		case Decoder::Opcode::TBufferStoreFormatX:
		case Decoder::Opcode::TBufferStoreFormatXy:
		case Decoder::Opcode::TBufferStoreFormatXyz:
		case Decoder::Opcode::TBufferStoreFormatXyzw:
		case Decoder::Opcode::BufferAtomicSwap:
		case Decoder::Opcode::BufferAtomicAdd:
		case Decoder::Opcode::BufferAtomicSub:
		case Decoder::Opcode::BufferAtomicSMin:
		case Decoder::Opcode::BufferAtomicUMin:
		case Decoder::Opcode::BufferAtomicSMax:
		case Decoder::Opcode::BufferAtomicUMax:
		case Decoder::Opcode::BufferAtomicAnd:
		case Decoder::Opcode::BufferAtomicOr:
		case Decoder::Opcode::BufferAtomicXor:
		case Decoder::Opcode::BufferAtomicInc:
		case Decoder::Opcode::BufferAtomicDec:
		case Decoder::Opcode::BufferAtomicIncX2:
		case Decoder::Opcode::BufferAtomicDecX2:
		case Decoder::Opcode::BufferAtomicSwapX2:
		case Decoder::Opcode::BufferAtomicAddX2:
		case Decoder::Opcode::BufferAtomicSubX2:
		case Decoder::Opcode::BufferAtomicSminX2:
		case Decoder::Opcode::BufferAtomicUminX2:
		case Decoder::Opcode::BufferAtomicSmaxX2:
		case Decoder::Opcode::BufferAtomicUmaxX2:
		case Decoder::Opcode::BufferAtomicAndX2:
		case Decoder::Opcode::BufferAtomicOrX2:
		case Decoder::Opcode::BufferAtomicXorX2:
		case Decoder::Opcode::BufferAtomicFcmpswap:
		case Decoder::Opcode::BufferAtomicFcmpswapX2:
		case Decoder::Opcode::BufferAtomicFmin:
		case Decoder::Opcode::BufferAtomicFmax:
		case Decoder::Opcode::BufferAtomicFminX2:
		case Decoder::Opcode::BufferAtomicFmaxX2:
		case Decoder::Opcode::BufferAtomicCsub:
		case Decoder::Opcode::BufferAtomicCmpswap:
		case Decoder::Opcode::BufferAtomicCmpswapX2:
		case Decoder::Opcode::FlatLoadUbyte:
		case Decoder::Opcode::FlatLoadSbyte:
		case Decoder::Opcode::FlatLoadUshort:
		case Decoder::Opcode::FlatLoadSshort:
		case Decoder::Opcode::FlatLoadDword:
		case Decoder::Opcode::FlatLoadDwordx2:
		case Decoder::Opcode::FlatLoadDwordx3:
		case Decoder::Opcode::FlatLoadDwordx4:
		case Decoder::Opcode::FlatLoadShortD16:
		case Decoder::Opcode::FlatLoadShortD16Hi:
		case Decoder::Opcode::FlatLoadUbyteD16:
		case Decoder::Opcode::FlatLoadUbyteD16Hi:
		case Decoder::Opcode::FlatLoadSbyteD16:
		case Decoder::Opcode::FlatLoadSbyteD16Hi:
		case Decoder::Opcode::FlatLoadDwordAddtid:
		case Decoder::Opcode::FlatStoreByte:
		case Decoder::Opcode::FlatStoreShort:
		case Decoder::Opcode::FlatStoreDword:
		case Decoder::Opcode::FlatStoreDwordx2:
		case Decoder::Opcode::FlatStoreDwordx3:
		case Decoder::Opcode::FlatStoreDwordx4:
		case Decoder::Opcode::FlatStoreByteD16Hi:
		case Decoder::Opcode::FlatStoreShortD16Hi:
		case Decoder::Opcode::FlatStoreDwordAddtid:
		case Decoder::Opcode::DsAddU32:
		case Decoder::Opcode::DsAddRtnU32:
		case Decoder::Opcode::DsSubU32:
		case Decoder::Opcode::DsSubRtnU32:
		case Decoder::Opcode::DsMinI32:
		case Decoder::Opcode::DsMinRtnI32:
		case Decoder::Opcode::DsMaxI32:
		case Decoder::Opcode::DsMaxRtnI32:
		case Decoder::Opcode::DsMinU32:
		case Decoder::Opcode::DsMinRtnU32:
		case Decoder::Opcode::DsMaxU32:
		case Decoder::Opcode::DsMaxRtnU32:
		case Decoder::Opcode::DsAndB32:
		case Decoder::Opcode::DsAndRtnB32:
		case Decoder::Opcode::DsOrB32:
		case Decoder::Opcode::DsOrRtnB32:
		case Decoder::Opcode::DsXorB32:
		case Decoder::Opcode::DsXorRtnB32:
		case Decoder::Opcode::DsWrxchgRtnB32:
		case Decoder::Opcode::DsMinF32:
		case Decoder::Opcode::DsMaxF32:
		case Decoder::Opcode::DsSwizzleB32:
		case Decoder::Opcode::DsConsume:
		case Decoder::Opcode::DsAppend:
		case Decoder::Opcode::DsReadSbyte:
		case Decoder::Opcode::DsReadUbyte:
		case Decoder::Opcode::DsReadSshort:
		case Decoder::Opcode::DsReadUshort:
		case Decoder::Opcode::DsRead2B32:
		case Decoder::Opcode::DsReadB32:
		case Decoder::Opcode::DsReadB64:
		case Decoder::Opcode::DsRead2B64:
		case Decoder::Opcode::DsRead2St64B64:
		case Decoder::Opcode::DsReadB96:
		case Decoder::Opcode::DsReadB128:
		case Decoder::Opcode::DsWriteByte:
		case Decoder::Opcode::DsWriteShort:
		case Decoder::Opcode::DsWrite2B32:
		case Decoder::Opcode::DsWrite2St64B32:
		case Decoder::Opcode::DsWrite2B64:
		case Decoder::Opcode::DsWrite2St64B64:
		case Decoder::Opcode::DsWriteB32:
		case Decoder::Opcode::DsWriteB64:
		case Decoder::Opcode::DsWriteB96:
		case Decoder::Opcode::DsWriteB128:
		case Decoder::Opcode::DsWriteAddtidB32:
		case Decoder::Opcode::DsReadAddtidB32:
		case Decoder::Opcode::ImageGetResinfo:
		case Decoder::Opcode::ImageGetLod:
		case Decoder::Opcode::ImageLoad:
		case Decoder::Opcode::ImageLoadMip:
		case Decoder::Opcode::ImageLoadPck:
		case Decoder::Opcode::ImageLoadPckSgn:
		case Decoder::Opcode::ImageLoadMipPck:
		case Decoder::Opcode::ImageLoadMipPckSgn:
		case Decoder::Opcode::ImageMsaaLoad:
		case Decoder::Opcode::ImageBvhIntersectRay:
		case Decoder::Opcode::ImageBvh64IntersectRay:
		case Decoder::Opcode::ImageStore:
		case Decoder::Opcode::ImageStoreMip:
		case Decoder::Opcode::ImageStorePck:
		case Decoder::Opcode::ImageStoreMipPck:
		case Decoder::Opcode::ImageAtomicAdd:
		case Decoder::Opcode::ImageAtomicUMin:
		case Decoder::Opcode::ImageAtomicUMax:
		case Decoder::Opcode::ImageAtomicAnd:
		case Decoder::Opcode::ImageAtomicOr:
		case Decoder::Opcode::ImageAtomicXor:
		case Decoder::Opcode::ImageAtomicSwap:
		case Decoder::Opcode::ImageAtomicCmpswap:
		case Decoder::Opcode::ImageAtomicSub:
		case Decoder::Opcode::ImageAtomicSmin:
		case Decoder::Opcode::ImageAtomicSmax:
		case Decoder::Opcode::ImageAtomicInc:
		case Decoder::Opcode::ImageAtomicDec:
		case Decoder::Opcode::ImageAtomicFcmpswap:
		case Decoder::Opcode::ImageAtomicFmin:
		case Decoder::Opcode::ImageAtomicFmax:
		case Decoder::Opcode::ImageGather4:
		case Decoder::Opcode::ImageGather4Cl:
		case Decoder::Opcode::ImageGather4L:
		case Decoder::Opcode::ImageGather4B:
		case Decoder::Opcode::ImageGather4BCl:
		case Decoder::Opcode::ImageGather4Lz:
		case Decoder::Opcode::ImageGather4C:
		case Decoder::Opcode::ImageGather4CCl:
		case Decoder::Opcode::ImageGather4CL:
		case Decoder::Opcode::ImageGather4CB:
		case Decoder::Opcode::ImageGather4CBCl:
		case Decoder::Opcode::ImageGather4CLz:
		case Decoder::Opcode::ImageGather4O:
		case Decoder::Opcode::ImageGather4ClO:
		case Decoder::Opcode::ImageGather4LO:
		case Decoder::Opcode::ImageGather4BO:
		case Decoder::Opcode::ImageGather4BClO:
		case Decoder::Opcode::ImageGather4LzO:
		case Decoder::Opcode::ImageGather4CO:
		case Decoder::Opcode::ImageGather4CClO:
		case Decoder::Opcode::ImageGather4CLO:
		case Decoder::Opcode::ImageGather4CBO:
		case Decoder::Opcode::ImageGather4CBClO:
		case Decoder::Opcode::ImageGather4CLzO:
		case Decoder::Opcode::ImageGather4H:
		case Decoder::Opcode::ImageSample: return true;
		default: return false;
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
