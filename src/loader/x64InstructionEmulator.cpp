#include "loader/x64InstructionEmulator.h"

#include "common/common.h"
#include "common/crashDiagnostics.h"
#include "common/fatalLog.h"

#include <atomic>
#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstring>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#endif

namespace Loader::X64InstructionEmulator {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

extern "C" void KytyCfgSafeCallTarget();
extern "C" void KytyCfgAllowAll();

static M128A* GetContextXmm(PCONTEXT context, uint8_t index) {
	if (context == nullptr || index >= 16) {
		return nullptr;
	}

	return &context->Xmm0 + index;
}

static uint64_t* GetGpReg(PCONTEXT context, uint8_t index) {
	if (context == nullptr) {
		return nullptr;
	}
	switch (index & 0x0fu) {
		case 0: return &context->Rax;
		case 1: return &context->Rcx;
		case 2: return &context->Rdx;
		case 3: return &context->Rbx;
		case 4: return &context->Rsp;
		case 5: return &context->Rbp;
		case 6: return &context->Rsi;
		case 7: return &context->Rdi;
		case 8: return &context->R8;
		case 9: return &context->R9;
		case 10: return &context->R10;
		case 11: return &context->R11;
		case 12: return &context->R12;
		case 13: return &context->R13;
		case 14: return &context->R14;
		case 15: return &context->R15;
		default: return nullptr;
	}
}

static uint64_t ExtractBitField(uint64_t value, uint32_t length, uint32_t index) {
	length &= 0x3fu;
	index &= 0x3fu;

	if (length == 0) {
		length = 64;
	}

	if (index >= 64) {
		return 0;
	}

	auto available = 64u - index;
	if (length > available) {
		length = available;
	}

	const uint64_t mask = (length == 64 ? UINT64_MAX : ((uint64_t {1} << length) - 1u));
	return (value >> index) & mask;
}

static uint64_t InsertBitField(uint64_t dst, uint64_t src, uint32_t length, uint32_t index) {
	length &= 0x3fu;
	index &= 0x3fu;

	if (length == 0) {
		length = 64;
	}

	if (index >= 64) {
		return dst;
	}

	auto available = 64u - index;
	if (length > available) {
		length = available;
	}

	const uint64_t mask        = (length == 64 ? UINT64_MAX : ((uint64_t {1} << length) - 1u));
	const uint64_t shifted     = (index == 0 ? mask : (mask << index));
	const uint64_t src_shifted = (src & mask) << index;

	return (dst & ~shifted) | src_shifted;
}

static bool TryEmulateSse4a(PCONTEXT context) {
	if (context == nullptr) {
		return false;
	}

	const auto* rip = reinterpret_cast<const uint8_t*>(context->Rip);

	const uint8_t prefix = rip[0];
	if (prefix != 0x66 && prefix != 0xf2) {
		return false;
	}

	size_t  offset = 1;
	uint8_t rex    = 0;
	if ((rip[offset] & 0xf0u) == 0x40u) {
		rex = rip[offset];
		offset++;
	}

	if (rip[offset] != 0x0f || rip[offset + 1] != 0x78) {
		return false;
	}

	auto modrm = rip[offset + 2];
	if ((modrm & 0xc0u) != 0xc0u) {
		return false;
	}

	const uint8_t reg    = ((modrm >> 3u) & 0x07u) | ((rex & 0x04u) << 1u);
	const uint8_t rm     = (modrm & 0x07u) | ((rex & 0x01u) << 3u);
	const uint8_t length = rip[offset + 3];
	const uint8_t index  = rip[offset + 4];

	// AMD SSE4a immediate-form EXTRQ/INSERTQ. PS5 code can execute these natively on AMD hardware,
	// while Intel hosts raise an illegal-instruction exception.
	if (prefix == 0x66) {
		auto* dst = GetContextXmm(context, rm);
		if (dst == nullptr) {
			return false;
		}

		dst->Low  = ExtractBitField(dst->Low, length, index);
		dst->High = 0;
		context->Rip += offset + 5;
		return true;
	}

	auto* dst = GetContextXmm(context, reg);
	auto* src = GetContextXmm(context, rm);
	if (dst == nullptr || src == nullptr) {
		return false;
	}

	dst->Low = InsertBitField(dst->Low, src->Low, length, index);
	context->Rip += offset + 5;
	return true;
}

static bool TryEmulateMonitorxMwaitx(PCONTEXT context) {
	if (context == nullptr) {
		return false;
	}

	const auto* rip = reinterpret_cast<const uint8_t*>(context->Rip);
	if (rip[0] != 0x0f || rip[1] != 0x01 || (rip[2] != 0xfa && rip[2] != 0xfb)) {
		return false;
	}

	// AMD MONITORX/MWAITX are used by PS5 code in wait loops. Intel hosts can raise an illegal-
	// instruction exception, so approximate them as a no-op/yield pair.
	if (rip[2] == 0xfb) {
		SwitchToThread();
	}
	context->Rip += 3;
	return true;
}

struct DecodedMemOp {
	size_t  length      = 0;
	uint8_t reg         = 0;
	bool    is_xmm      = false;
	bool    is_write    = false;
	bool    has_modrm   = false;
	bool    is_memory   = false;
	uint8_t operand_bits = 64; // 8/16/32/64 for GP; ignored for XMM zeroing
};

static bool DecodeMemOp(const uint8_t* rip, DecodedMemOp* out) {
	if (rip == nullptr || out == nullptr) {
		return false;
	}

	size_t  i           = 0;
	uint8_t rex         = 0;
	bool    op_size_16  = false;
	bool    repnz       = false;
	bool    repz        = false;
	bool    addr_size32 = false;

	// Legacy prefixes (cap to avoid runaway).
	for (int guard = 0; guard < 14; ++guard) {
		const uint8_t b = rip[i];
		if (b == 0x66) {
			op_size_16 = true;
			++i;
			continue;
		}
		if (b == 0x67) {
			addr_size32 = true;
			++i;
			continue;
		}
		if (b == 0xf2) {
			repnz = true;
			++i;
			continue;
		}
		if (b == 0xf3) {
			repz = true;
			++i;
			continue;
		}
		if (b == 0x2e || b == 0x36 || b == 0x3e || b == 0x26 || b == 0x64 || b == 0x65) {
			++i;
			continue;
		}
		break;
	}
	(void)addr_size32;

	if ((rip[i] & 0xf0u) == 0x40u) {
		rex = rip[i++];
	}

	DecodedMemOp decoded {};
	decoded.is_write = false;

	auto parse_modrm = [&](size_t at) -> bool {
		const uint8_t modrm = rip[at++];
		const uint8_t mod   = static_cast<uint8_t>((modrm >> 6u) & 0x3u);
		const uint8_t rm    = static_cast<uint8_t>(modrm & 0x7u);
		decoded.reg         = static_cast<uint8_t>(((modrm >> 3u) & 0x7u) | ((rex & 0x04u) << 1u));
		decoded.has_modrm   = true;
		if (mod == 3) {
			decoded.is_memory = false;
			decoded.length    = at;
			return true;
		}
		decoded.is_memory = true;
		if (rm == 4) {
			const uint8_t sib = rip[at++];
			const uint8_t base = static_cast<uint8_t>(sib & 0x7u);
			if (mod == 0 && base == 5) {
				at += 4; // disp32 with no base
			} else if (mod == 1) {
				at += 1;
			} else if (mod == 2) {
				at += 4;
			}
		} else if (mod == 0 && rm == 5) {
			at += 4; // RIP-relative or abs disp32
		} else if (mod == 1) {
			at += 1;
		} else if (mod == 2) {
			at += 4;
		}
		decoded.length = at;
		return true;
	};

	const uint8_t op0 = rip[i++];

	// Two-byte opcodes.
	if (op0 == 0x0f) {
		const uint8_t op1 = rip[i++];
		// movzx / movsx
		if (op1 == 0xb6 || op1 == 0xb7 || op1 == 0xbe || op1 == 0xbf) {
			decoded.is_write     = false;
			decoded.operand_bits = (rex & 0x08u) != 0 ? 64 : (op_size_16 ? 16 : 32);
			if (!parse_modrm(i)) {
				return false;
			}
			*out = decoded;
			return decoded.is_memory;
		}
		// movups/movaps/movdqa/movdqu loads: 0F 10/28/6F (reg <- mem)
		if (op1 == 0x10 || op1 == 0x28 || op1 == 0x6f) {
			decoded.is_write = false;
			decoded.is_xmm   = true;
			if (!parse_modrm(i)) {
				return false;
			}
			*out = decoded;
			return decoded.is_memory;
		}
		// stores 0F 11/29/7F
		if (op1 == 0x11 || op1 == 0x29 || op1 == 0x7f) {
			decoded.is_write = true;
			decoded.is_xmm   = true;
			if (!parse_modrm(i)) {
				return false;
			}
			*out = decoded;
			return decoded.is_memory;
		}
		return false;
	}

	// GP moves.
	if (op0 == 0x8b) { // mov r, r/m
		decoded.is_write     = false;
		decoded.operand_bits = (rex & 0x08u) != 0 ? 64 : (op_size_16 ? 16 : 32);
		if (!parse_modrm(i)) {
			return false;
		}
		*out = decoded;
		return decoded.is_memory;
	}
	if (op0 == 0x8a) { // mov r8, r/m8
		decoded.is_write     = false;
		decoded.operand_bits = 8;
		if (!parse_modrm(i)) {
			return false;
		}
		*out = decoded;
		return decoded.is_memory;
	}
	if (op0 == 0x89) { // mov r/m, r
		decoded.is_write = true;
		if (!parse_modrm(i)) {
			return false;
		}
		*out = decoded;
		return decoded.is_memory;
	}
	if (op0 == 0x88) { // mov r/m8, r8
		decoded.is_write = true;
		if (!parse_modrm(i)) {
			return false;
		}
		*out = decoded;
		return decoded.is_memory;
	}
	if (op0 == 0x63) { // movsxd
		decoded.is_write     = false;
		decoded.operand_bits = 64;
		if (!parse_modrm(i)) {
			return false;
		}
		*out = decoded;
		return decoded.is_memory;
	}
	if (op0 == 0xc7) { // mov r/m, imm32
		decoded.is_write = true;
		if (!parse_modrm(i)) {
			return false;
		}
		decoded.length += (rex & 0x08u) != 0 ? 4 : (op_size_16 ? 2 : 4);
		*out = decoded;
		return decoded.is_memory;
	}
	if (op0 == 0xc6) { // mov r/m8, imm8
		decoded.is_write = true;
		if (!parse_modrm(i)) {
			return false;
		}
		decoded.length += 1;
		*out = decoded;
		return decoded.is_memory;
	}

	// SSE scalar load/store with F2/F3: movsd/movss
	if ((repz || repnz) && op0 == 0x0f) {
		// unreachable — 0F handled above; keep for clarity
	}
	(void)repz;
	(void)repnz;

	return false;
}

static void ZeroGpDest(PCONTEXT context, uint8_t reg, uint8_t bits) {
	uint64_t* gp = GetGpReg(context, reg);
	if (gp == nullptr) {
		return;
	}
	if (bits >= 64) {
		*gp = 0;
	} else if (bits == 32) {
		*gp = 0; // 32-bit writes zero-extend
	} else if (bits == 16) {
		*gp = (*gp & ~uint64_t {0xffff}) | 0;
	} else {
		// 8-bit: low byte; ignore AH complexity for poison soft-continue
		*gp = (*gp & ~uint64_t {0xff}) | 0;
	}
}

static void ZeroXmmDest(PCONTEXT context, uint8_t reg) {
	M128A* xmm = GetContextXmm(context, reg);
	if (xmm == nullptr) {
		return;
	}
	xmm->Low  = 0;
	xmm->High = 0;
}

bool TrySoftContinuePoisonAccess(void* native_context, uint64_t fault_vaddr, bool is_write,
                                 bool force, bool allow_system_module) {
	auto* context = static_cast<PCONTEXT>(native_context);
	if (context == nullptr) {
		return false;
	}

	// Never soft-continue inside foreign modules (ntdll / kernel32 / …) unless the caller
	// explicitly opts in after orphan-commit failed (otherwise stack walks loop forever).
	{
		if (!allow_system_module && context->Rip >= 0x00007FF000000000ull) {
			return false;
		}
		HMODULE self  = GetModuleHandleA(nullptr);
		HMODULE owner = nullptr;
		if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                       reinterpret_cast<LPCSTR>(context->Rip), &owner) != 0 &&
		    owner != nullptr && self != nullptr && owner != self) {
			if (!allow_system_module) {
				return false;
			}
		}
	}

	const bool poisonish = fault_vaddr == 0 || fault_vaddr == UINT64_MAX ||
	                       fault_vaddr >= (1ull << 47);
	if (!poisonish && !force) {
		return false;
	}

	MEMORY_BASIC_INFORMATION mbi {};
	if (VirtualQuery(reinterpret_cast<const void*>(context->Rip), &mbi, sizeof(mbi)) == 0 ||
	    mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
		return false;
	}

	const auto* rip = reinterpret_cast<const uint8_t*>(context->Rip);
	DecodedMemOp op {};
	if (!DecodeMemOp(rip, &op) || op.length == 0 || op.length > 15) {
		return false;
	}
	if (op.is_write != is_write) {
		// Still allow skip if decode says write but VEH said read (or vice versa) for poison.
		// Prefer VEH classification for zeroing vs nop-store.
	}

	if (!is_write) {
		if (op.is_xmm) {
			ZeroXmmDest(context, op.reg);
		} else {
			ZeroGpDest(context, op.reg, op.operand_bits);
		}
	}

	context->Rip += op.length;

	static std::atomic<uint32_t> soft_n {0};
	const uint32_t               n = soft_n.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 48 || (n % 64) == 0) {
		char line[256];
		std::snprintf(line, sizeof(line),
		              "MemoryTrace: poison soft-continue n=%u %s vaddr=0x%016" PRIx64
		              " rip=0x%016" PRIx64 " len=%zu reg=%u xmm=%d",
		              n, is_write ? "Write" : "Read", fault_vaddr,
		              static_cast<uint64_t>(context->Rip - op.length), op.length, op.reg,
		              op.is_xmm ? 1 : 0);
		Common::LogFatalToFile(line);
		std::fprintf(stderr, "%s\n", line);
		std::fflush(stderr);
		if (n == 1) {
			Common::NoteHaltReason("poison_soft_continue", "first AV -1/non-canonical skipped");
			Common::FlushHleRingToFatal("poison_soft_continue");
		}
	}
	return true;
}

static bool LooksLikeAlignedSseMemOp(const uint8_t* rip, size_t* out_len) {
	if (rip == nullptr) {
		return false;
	}
	size_t i = 0;
	// Skip legacy prefixes (not F2/F3 — those select movsd/movss which are unaligned-OK).
	for (int guard = 0; guard < 8; ++guard) {
		const uint8_t b = rip[i];
		if (b == 0x66 || b == 0x67 || b == 0x2e || b == 0x36 || b == 0x3e || b == 0x26 ||
		    b == 0x64 || b == 0x65) {
			++i;
			continue;
		}
		break;
	}
	if ((rip[i] & 0xf0u) == 0x40u) {
		++i;
	}
	if (rip[i] == 0xf2 || rip[i] == 0xf3) {
		return false; // scalar SSE — not alignment-sensitive the same way
	}
	if (rip[i++] != 0x0f) {
		return false;
	}
	const uint8_t op = rip[i++];
	// movaps load/store 28/29, movdqa 6F/7F (with 66 already skipped as prefix)
	if (op != 0x28 && op != 0x29 && op != 0x6f && op != 0x7f) {
		return false;
	}
	DecodedMemOp decoded {};
	// Re-decode from original rip for length (includes prefixes).
	if (!DecodeMemOp(rip, &decoded) || !decoded.is_memory || !decoded.is_xmm) {
		return false;
	}
	if (out_len != nullptr) {
		*out_len = decoded.length;
	}
	return true;
}

bool TryFixMisalignedSseAccess(void* native_context, uint64_t fault_vaddr) {
	auto* context = static_cast<PCONTEXT>(native_context);
	if (context == nullptr) {
		return false;
	}

	const bool poisonish = fault_vaddr == 0 || fault_vaddr == UINT64_MAX ||
	                       fault_vaddr >= (1ull << 47);
	// Also accept real AVs to [rsp+disp] when rsp is misaligned (info1 may be the EA).
	const bool rsp_misaligned = (context->Rsp & 0xfu) != 0;
	if (!rsp_misaligned) {
		return false;
	}
	if (!poisonish) {
		// Only intervene for poisonish EA OR EA near RSP (spill slot).
		const uint64_t ea = fault_vaddr;
		if (ea < context->Rsp || ea > context->Rsp + 0x400) {
			return false;
		}
	}

	MEMORY_BASIC_INFORMATION mbi {};
	if (VirtualQuery(reinterpret_cast<const void*>(context->Rip), &mbi, sizeof(mbi)) == 0 ||
	    mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
		return false;
	}

	const auto* rip = reinterpret_cast<const uint8_t*>(context->Rip);
	size_t      len = 0;
	if (!LooksLikeAlignedSseMemOp(rip, &len)) {
		return false;
	}

	const uint64_t old_rsp = context->Rsp;
	context->Rsp &= ~uint64_t {0xfu}; // align down — retry same instruction

	static std::atomic<uint32_t> fix_n {0};
	const uint32_t               n = fix_n.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 64 || (n % 64) == 0) {
		char line[256];
		std::snprintf(line, sizeof(line),
		              "MemoryTrace: sse misalign fix n=%u rsp=0x%016" PRIx64 "->0x%016" PRIx64
		              " rip=0x%016" PRIx64 " vaddr=0x%016" PRIx64 " len=%zu",
		              n, old_rsp, context->Rsp, static_cast<uint64_t>(context->Rip), fault_vaddr,
		              len);
		Common::LogFatalToFile(line);
		std::fprintf(stderr, "%s\n", line);
		std::fflush(stderr);
		if (n == 1) {
			Common::NoteHaltReason("sse_misalign", "aligned RSP and retried movaps/movdqa");
		}
	}
	return true;
}

// KytyCfgSafeCallTarget lives in main.cpp (CFG sanitize stubs).

static bool IsHostExecutableCode(uint64_t addr) {
	if (addr < 0x10000) {
		return false;
	}
	MEMORY_BASIC_INFORMATION mbi {};
	if (VirtualQuery(reinterpret_cast<const void*>(addr), &mbi, sizeof(mbi)) == 0 ||
	    mbi.State != MEM_COMMIT) {
		return false;
	}
	const DWORD prot = mbi.Protect & 0xffu;
	return prot == PAGE_EXECUTE || prot == PAGE_EXECUTE_READ || prot == PAGE_EXECUTE_READWRITE ||
	       prot == PAGE_EXECUTE_WRITECOPY;
}

bool TrySoftContinueCfgBitmap(void* native_context, uint64_t fault_vaddr) {
	auto* context = static_cast<PCONTEXT>(native_context);
	if (context == nullptr) {
		return false;
	}

	// ntdll!LdrpValidateUserCallTarget bitmap probe — sparse MEM_MAPPED at high VA.
	MEMORY_BASIC_INFORMATION fault_mbi {};
	if (VirtualQuery(reinterpret_cast<const void*>(fault_vaddr), &fault_mbi, sizeof(fault_mbi)) ==
	        0 ||
	    fault_mbi.Type != MEM_MAPPED || fault_mbi.AllocationBase == nullptr) {
		return false;
	}
	const uint64_t alloc_base = reinterpret_cast<uint64_t>(fault_mbi.AllocationBase);
	if (alloc_base < 0x0000700000000000ull) {
		return false;
	}
	if (!(fault_mbi.State == MEM_RESERVE ||
	      (fault_mbi.State == MEM_COMMIT && (fault_mbi.Protect & PAGE_NOACCESS) != 0))) {
		return false;
	}

	HMODULE ntdll = GetModuleHandleA("ntdll.dll");
	HMODULE owner = nullptr;
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
	                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCSTR>(context->Rip), &owner) == 0 ||
	    owner == nullptr || owner != ntdll) {
		return false;
	}

	// Do NOT blindly retarget RCX: stack walkers validate return addresses via CFG without
	// calling them. Only retarget when the CFG caller will call/jmp rcx.
	MEMORY_BASIC_INFORMATION stack_mbi {};
	if (VirtualQuery(reinterpret_cast<const void*>(context->Rsp), &stack_mbi, sizeof(stack_mbi)) ==
	        0 ||
	    stack_mbi.State != MEM_COMMIT || (stack_mbi.Protect & PAGE_NOACCESS) != 0) {
		return false;
	}
	const uint64_t call_target = context->Rcx;
	const uint64_t stacked_ret = *reinterpret_cast<const uint64_t*>(context->Rsp);
	if (stacked_ret < 0x10000 || !IsHostExecutableCode(stacked_ret)) {
		return false;
	}

	// First hit: redirect LdrpValidateUserCallTarget → KytyCfgAllowAll (abs jmp).
	{
		static std::atomic<bool> patched {false};
		bool                     expect = false;
		if (patched.compare_exchange_strong(expect, true, std::memory_order_relaxed)) {
			DWORD64           image_base = 0;
			PRUNTIME_FUNCTION rf = RtlLookupFunctionEntry(context->Rip, &image_base, nullptr);
			if (rf != nullptr && image_base != 0) {
				auto*            start = reinterpret_cast<uint8_t*>(image_base + rf->BeginAddress);
				constexpr size_t kLen  = 14;
				uint8_t          buf[kLen] = {0xff, 0x25, 0x00, 0x00, 0x00, 0x00};
				const uint64_t   dest      = reinterpret_cast<uint64_t>(&KytyCfgAllowAll);
				std::memcpy(buf + 6, &dest, sizeof(dest));
				DWORD old = 0;
				if (VirtualProtect(start, kLen, PAGE_EXECUTE_READWRITE, &old) != 0) {
					std::memcpy(start, buf, kLen);
					FlushInstructionCache(GetCurrentProcess(), start, kLen);
					VirtualProtect(start, kLen, old, &old);
					char line[192];
					std::snprintf(line, sizeof(line),
					              "MemoryTrace: CFG ntdll check patched to KytyCfgAllowAll at %p "
					              "(was mid-fn rip=0x%016" PRIx64 ")",
					              static_cast<void*>(start),
					              static_cast<uint64_t>(context->Rip));
					Common::LogFatalToFile(line);
					std::fprintf(stderr, "%s\n", line);
				}
			}
		}
	}

	{
		const auto* p = reinterpret_cast<const uint8_t*>(stacked_ret);
		const bool  will_invoke = p[0] == 0xff && (p[1] == 0xd1 || p[1] == 0xe1);
		if (will_invoke && !IsHostExecutableCode(call_target)) {
			context->Rcx = reinterpret_cast<uint64_t>(&KytyCfgSafeCallTarget);
		} else if (!will_invoke && !IsHostExecutableCode(call_target)) {
			// Stack walker validating a guest-stack / non-X "return address". Allowing
			// the check lets the walk dive into guest frames → 0xC0000005. Soft-halt.
			char line[320];
			std::snprintf(line, sizeof(line),
			              "MemoryTrace: CFG-bitmap walker soft-halt target=0x%016" PRIx64
			              " ret=0x%016" PRIx64,
			              call_target, stacked_ret);
			Common::LogFatalToFile(line);
			std::fprintf(stderr, "%s\n", line);
			return false;
		}
	}

	context->Rip = stacked_ret;
	context->Rsp += 8;

	static std::atomic<uint32_t> cfg_n {0};
	const uint32_t               n = cfg_n.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 32 || (n % 128) == 0) {
		char line[384];
		std::snprintf(line, sizeof(line),
		              "MemoryTrace: CFG-bitmap soft-continue n=%u fault=0x%016" PRIx64
		              " target=0x%016" PRIx64 " ret=0x%016" PRIx64 " exec=%d rcx_now=0x%016" PRIx64,
		              n, fault_vaddr, call_target, stacked_ret,
		              IsHostExecutableCode(call_target) ? 1 : 0,
		              static_cast<uint64_t>(context->Rcx));
		Common::LogFatalToFile(line);
		std::fprintf(stderr, "%s\n", line);
		std::fflush(stderr);
	}
	return true;
}

#endif

bool TryEmulate(void* native_context) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	auto* context = static_cast<PCONTEXT>(native_context);
	return TryEmulateMonitorxMwaitx(context) || TryEmulateSse4a(context);
#else
	(void)native_context;
	return false;
#endif
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
bool TrySoftContinuePoisonAccess(void* native_context, uint64_t fault_vaddr, bool is_write,
                                 bool force, bool allow_system_module) {
	(void)native_context;
	(void)fault_vaddr;
	(void)is_write;
	(void)force;
	(void)allow_system_module;
	return false;
}

bool TryFixMisalignedSseAccess(void* native_context, uint64_t fault_vaddr) {
	(void)native_context;
	(void)fault_vaddr;
	return false;
}

bool TrySoftContinueCfgBitmap(void* native_context, uint64_t fault_vaddr) {
	(void)native_context;
	(void)fault_vaddr;
	return false;
}
#endif

} // namespace Loader::X64InstructionEmulator
