#include "loader/runtimeLinker.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/crashDiagnostics.h"
#include "common/emulatorConfig.h"
#include "common/fatalLog.h"
#include "common/file.h"
#include "common/hostException.h"
#include "common/logging/log.h"
#include "common/magicEnum.h"
#include "common/platform/sysDbg.h"
#include "common/profiler.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "loader/elf.h"
#include "loader/jit.h"
#include "loader/symbolDatabase.h"
#include "loader/x64InstructionEmulator.h"
#include "graphics/presentation/videoOut.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <memory>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::LibKernel {
void SetProgName(const std::string& name);
} // namespace Libs::LibKernel

namespace Loader {

namespace {
void SynthesizeProgramProcParam(Program* program);
} // namespace

Program::Program() = default;

Program::~Program() = default;

static void FreeTlsBlock(ThreadLocalStorage::Block* block) {
	if (block == nullptr || block->ptr == nullptr) {
		return;
	}

	if (block->free_func != nullptr) {
		block->free_func(block->ptr);
	} else if (block->vm_alloc) {
		Common::VirtualMemory::Free(reinterpret_cast<uint64_t>(block->ptr));
	} else {
		// Legacy/corrupt block: never delete[] guest-heap pointers.
		LOGF("FreeTlsBlock: dropping TLS ptr=0x%016" PRIx64 " without free_func/vm_alloc\n",
		     reinterpret_cast<uint64_t>(block->ptr));
	}

	block->ptr       = nullptr;
	block->free_func = nullptr;
	block->vm_alloc  = false;
}

static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
	return alignment != 0 ? (value + alignment - 1) & ~(alignment - 1) : value;
}

ThreadLocalStorage::~ThreadLocalStorage() {
	for (auto& [_, block]: tlss) {
		FreeTlsBlock(&block);
	}
}

#pragma pack(1)

struct EntryParams {
	int         argc;
	uint32_t    pad;
	const char* argv[3];
};

#pragma pack()

using atexit_func_t = KYTY_SYSV_ABI void (*)();
using entry_func_t  = KYTY_SYSV_ABI void (*)(EntryParams* params, atexit_func_t atexit_func);
using module_ini_fini_func_t = KYTY_SYSV_ABI int (*)(size_t args, const void* argp,
                                                     module_func_t func);

enum class BindType { Unknown, Local, Global, Weak };

struct RelocationInfo {
	bool        resolved   = false;
	BindType    bind       = BindType::Unknown;
	SymbolType  type       = SymbolType::Unknown;
	uint64_t    value      = 0;
	uint64_t    vaddr      = 0;
	uint64_t    base_vaddr = 0;
	std::string name;
	std::string dbg_name;
	bool        bind_self = false;
};

struct StubbedImportRecord {
	uint32_t    index       = 0;
	uint64_t    patch_vaddr = 0;
	uint64_t    thunk_vaddr = 0;
	std::string name;
	SymbolType  type = SymbolType::Unknown;
	BindType    bind = BindType::Unknown;
	std::string program;
	uint64_t    call_count              = 0;
	bool        called_during_nd_window = false;
};

// The structure will be passed via the stack
// since the size of an object is larger than 16 bytes
struct RelocateHandlerStack {
	uint64_t stack[3];
};

static std::vector<StubbedImportRecord> g_stubbed_imports;
static std::atomic_uint32_t             g_unresolved_stub_call_log_count {0};
static std::vector<uint64_t>            g_unresolved_stub_thunk_pages;
static uint64_t                         g_unresolved_stub_thunk_offset = 0;

static KYTY_SYSV_ABI uint64_t ResolveImportStubWithId(uint64_t record_id);

static uint64_t AllocateUnresolvedImportThunk(uint64_t record_id) {
	constexpr uint64_t page_size  = 4096;
	constexpr uint64_t thunk_size = 162;

	if (g_unresolved_stub_thunk_pages.empty() ||
	    g_unresolved_stub_thunk_offset + thunk_size > page_size) {
		auto page = Common::VirtualMemory::Alloc(0, page_size,
		                                         Common::VirtualMemory::Mode::ExecuteReadWrite);
		EXIT_NOT_IMPLEMENTED(page == 0);
		g_unresolved_stub_thunk_pages.push_back(page);
		g_unresolved_stub_thunk_offset = 0;
	}

	auto* code = reinterpret_cast<uint8_t*>(g_unresolved_stub_thunk_pages.back() +
	                                        g_unresolved_stub_thunk_offset);
	g_unresolved_stub_thunk_offset += thunk_size;

	const auto target = reinterpret_cast<uint64_t>(ResolveImportStubWithId);
	uint8_t    bytes[thunk_size] {};
	size_t     i      = 0;
	const auto emit   = [&](uint8_t b) { bytes[i++] = b; };
	const auto emit64 = [&](uint64_t v) {
		std::memcpy(bytes + i, &v, sizeof(v));
		i += sizeof(v);
	};
	const auto emit32 = [&](uint32_t v) {
		std::memcpy(bytes + i, &v, sizeof(v));
		i += sizeof(v);
	};
	const auto save_xmm = [&](uint8_t reg, uint8_t offset) {
		emit(0xf3);
		emit(0x0f);
		emit(0x7f);
		if (offset == 0) {
			emit(static_cast<uint8_t>(0x04u | (reg << 3u)));
			emit(0x24);
		} else {
			emit(static_cast<uint8_t>(0x44u | (reg << 3u)));
			emit(0x24);
			emit(offset);
		}
	};
	const auto load_xmm = [&](uint8_t reg, uint8_t offset) {
		emit(0xf3);
		emit(0x0f);
		emit(0x6f);
		if (offset == 0) {
			emit(static_cast<uint8_t>(0x04u | (reg << 3u)));
			emit(0x24);
		} else {
			emit(static_cast<uint8_t>(0x44u | (reg << 3u)));
			emit(0x24);
			emit(offset);
		}
	};

	emit(0x50); // push rax; preserve AL for variadic SysV calls
	emit(0x57); // push rdi
	emit(0x56); // push rsi
	emit(0x52); // push rdx
	emit(0x51); // push rcx
	emit(0x41);
	emit(0x50); // push r8
	emit(0x41);
	emit(0x51); // push r9
	emit(0x48);
	emit(0x81);
	emit(0xec);
	emit32(0x80); // sub rsp, 0x80
	for (uint8_t reg = 0; reg < 8; reg++) {
		save_xmm(reg, static_cast<uint8_t>(reg * 0x10u));
	}
	emit(0x48);
	emit(0xbf);
	emit64(record_id); // mov rdi, record_id
	emit(0x48);
	emit(0xb8);
	emit64(target); // mov rax, ResolveImportStubWithId
	emit(0xff);
	emit(0xd0); // call rax
	emit(0x49);
	emit(0x89);
	emit(0xc3); // mov r11, rax
	for (uint8_t reg = 0; reg < 8; reg++) {
		load_xmm(reg, static_cast<uint8_t>(reg * 0x10u));
	}
	emit(0x48);
	emit(0x81);
	emit(0xc4);
	emit32(0x80); // add rsp, 0x80
	emit(0x41);
	emit(0x59); // pop r9
	emit(0x41);
	emit(0x58); // pop r8
	emit(0x59); // pop rcx
	emit(0x5a); // pop rdx
	emit(0x5e); // pop rsi
	emit(0x5f); // pop rdi
	emit(0x58); // pop rax
	emit(0x4d);
	emit(0x85);
	emit(0xdb); // test r11, r11
	emit(0x74);
	emit(0x03); // jz +3
	emit(0x41);
	emit(0xff);
	emit(0xe3); // jmp r11
	emit(0x31);
	emit(0xc0); // xor eax, eax
	emit(0xc3); // ret

	EXIT_NOT_IMPLEMENTED(i != thunk_size);
	std::memcpy(code, bytes, sizeof(bytes));
	Common::VirtualMemory::FlushInstructionCache(reinterpret_cast<uint64_t>(code), thunk_size);
	return reinterpret_cast<uint64_t>(code);
}

static uint64_t RegisterStubbedImport(uint32_t index, const Program* program,
                                      const RelocationInfo& ri) {
	const auto program_name = program != nullptr ? Common::PathToString(program->file_name) : "";

	for (auto& record: g_stubbed_imports) {
		if (record.patch_vaddr == ri.vaddr) {
			record.index   = index;
			record.name    = ri.name;
			record.type    = ri.type;
			record.bind    = ri.bind;
			record.program = program_name;
			return record.thunk_vaddr;
		}
	}

	StubbedImportRecord record {};
	record.index       = index;
	record.patch_vaddr = ri.vaddr;
	record.name        = ri.name;
	record.type        = ri.type;
	record.bind        = ri.bind;
	record.program     = program_name;
	g_stubbed_imports.push_back(record);
	const auto record_id                     = g_stubbed_imports.size() - 1;
	const auto thunk                         = AllocateUnresolvedImportThunk(record_id);
	g_stubbed_imports[record_id].thunk_vaddr = thunk;
	return thunk;
}

static KYTY_SYSV_ABI uint64_t ResolveImportStubWithId(uint64_t record_id) {
	if (record_id < g_stubbed_imports.size()) {
		auto& record = g_stubbed_imports[record_id];
		auto  nid    = record.name;
		auto  pos    = Common::FindIndex(nid, "[");
		if (Common::IndexValid(nid, pos)) {
			nid = Common::Left(nid, pos);
		}

		SymbolRecord resolved {};
		if (!nid.empty() &&
		    Common::Singleton<RuntimeLinker>::Instance()->ResolveLoadedSymbolByNid(nid, record.type,
		                                                                           &resolved) &&
		    resolved.vaddr != 0 && resolved.vaddr != record.thunk_vaddr) {
			LOGF("Late-resolved import: %s -> %s [0x%016" PRIx64 "]\n", record.name.c_str(),
			     resolved.name.c_str(), resolved.vaddr);

			if (record.patch_vaddr != 0) {
				*reinterpret_cast<uint64_t*>(record.patch_vaddr) = resolved.vaddr;
			}

			return resolved.vaddr;
		}
	}

	const auto log_index = g_unresolved_stub_call_log_count.fetch_add(1);
	if (record_id < g_stubbed_imports.size()) {
		g_stubbed_imports[record_id].call_count++;
		if (Libs::LibKernel::Memory::IsNdMemoryValidationTraceActive()) {
			g_stubbed_imports[record_id].called_during_nd_window = true;
		}
	}
	const bool force_agc_log =
	    record_id < g_stubbed_imports.size() &&
	    (g_stubbed_imports[record_id].name.find("AgcDriver") != std::string::npos ||
	     g_stubbed_imports[record_id].name.find("[Agc_v1]") != std::string::npos ||
	     g_stubbed_imports[record_id].name.find("[Agc]") != std::string::npos);
	if (log_index < 1024 || force_agc_log) {
		if (record_id < g_stubbed_imports.size()) {
			const auto& record = g_stubbed_imports[record_id];
			printf("Unresolved import stub called: %s\n", record.name.c_str());
			LOGF("Unresolved import stub called [%u]: patch_vaddr=0x%016" PRIx64
			     " jmprela_index=%" PRIu32 " symbol=%s type=%s bind=%s program=%s\n",
			     log_index, record.patch_vaddr, record.index, record.name.c_str(),
			     Common::EnumName(record.type).c_str(), Common::EnumName(record.bind).c_str(),
			     record.program.c_str());
			if (force_agc_log && log_index >= 1024) {
				fprintf(stderr, "Unresolved Agc stub called: %s\n", record.name.c_str());
			}
		} else {
			printf("Unresolved import stub called: <bad-record>\n");
			LOGF("Unresolved import stub called [%u]: record_id=%" PRIu64 " symbol=<bad-record>\n",
			     log_index, record_id);
		}
	}
	return 0;
}

constexpr uint64_t SYSTEM_RESERVED  = 0x800000000u;
constexpr uint64_t CODE_BASE_INCR   = 0x010000000u;
constexpr uint64_t INVALID_OFFSET   = 0x040000000u;
constexpr uint64_t CODE_BASE_OFFSET = 0x100000000u;
constexpr uint64_t INVALID_MEMORY   = SYSTEM_RESERVED + INVALID_OFFSET;

static uint64_t g_desired_base_addr = SYSTEM_RESERVED + CODE_BASE_OFFSET;
static uint64_t g_invalid_memory    = 0;

static Program*              g_tls_main_program        = nullptr;
static thread_local Program* g_tls_cached_main_program = nullptr;
static thread_local uint8_t* g_tls_cached_main_tcb     = nullptr;

static KYTY_SYSV_ABI void RunEntry(uint64_t addr, EntryParams* params, atexit_func_t atexit_func,
                                   void* stack_top) {
#if defined(__x86_64__) || defined(_M_X64)
	auto* func = reinterpret_cast<entry_func_t>(addr);

	if (stack_top != nullptr) {
		// Same trampoline/jmp path as pthread workers (Phase 18) — no callq across stacks.
		(void)Libs::LibKernel::PthreadRunGuestOnStack(stack_top, reinterpret_cast<void*>(func),
		                                              reinterpret_cast<uint64_t>(params),
		                                              reinterpret_cast<uint64_t>(atexit_func));
		return;
	}

	uintptr_t guest_root_frame[2] = {};

	asm volatile("pushq %%r12\n\t"
	             "pushq %%r13\n\t"
	             "movq %%rbp, %%r12\n\t"
	             "movq %[guest_rbp], %%rbp\n\t"
	             "callq *%[func]\n\t"
	             "movq %%r12, %%rbp\n\t"
	             "popq %%r13\n\t"
	             "popq %%r12\n\t"
	             :
	             : [func] "r"(func), "D"(params),
	               "S"(atexit_func), [guest_rbp] "r"(guest_root_frame)
	             : "cc", "memory", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "xmm0", "xmm1",
	               "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11",
	               "xmm12", "xmm13", "xmm14", "xmm15");
#else
	(void)stack_top;
	reinterpret_cast<entry_func_t>(addr)(params, atexit_func);
#endif
}

static uint64_t GetAlignedSize(const Elf64_Phdr* p) {
	return (p->p_align != 0 ? (p->p_memsz + (p->p_align - 1)) & ~(p->p_align - 1) : p->p_memsz);
}

static void DbgDumpSymbols(const std::string& folder, Elf64_Sym* symbols, uint64_t size,
                           const char* names) {
	auto folder_str = Common::FixDirectorySlash(folder);

	Common::File::CreateDirectories(folder_str);

	Common::File f;
	f.Create(folder_str + "symbols.txt");

	for (auto* sym = symbols;
	     reinterpret_cast<uint8_t*>(sym) < reinterpret_cast<uint8_t*>(symbols) + size; sym++) {
		f.Printf("----\n");
		f.Printf("st_name = %" PRIu32 ", %s\n", sym->st_name, names + sym->st_name);
		f.Printf("st_info = 0x%02" PRIx8 "\n", sym->st_info);
		f.Printf("st_other = 0x%02" PRIx8 "\n", sym->st_other);
		f.Printf("st_shndx = 0x%04" PRIx16 "\n", sym->st_shndx);
		f.Printf("st_value = 0x%016" PRIx64 "\n", sym->st_value);
		f.Printf("st_size = %" PRIu64 "\n", sym->st_size);
	}

	f.Close();
}

static void DbgDumpRela(const std::string& folder, Elf64_Rela* records, uint64_t size,
                        const char* /*names*/, const char* file_name) {
	auto folder_str = Common::FixDirectorySlash(folder);

	Common::File::CreateDirectories(folder_str);

	Common::File f;
	f.Create(folder_str + file_name);

	for (auto* r = records;
	     reinterpret_cast<uint8_t*>(r) < reinterpret_cast<uint8_t*>(records) + size; r++) {
		f.Printf("----\n"
		         "r_offset = 0x%016" PRIx64 "\n"
		         "r_info = 0x%016" PRIx64 "\n"
		         "r_addend = %" PRId64 "\n",
		         r->r_offset, r->r_info, r->r_addend);
	}

	f.Close();
}

static Common::VirtualMemory::Mode GetMode(Elf64_Word flags) {
	switch (flags) {
		case PF_R: return Common::VirtualMemory::Mode::Read;
		case PF_W: return Common::VirtualMemory::Mode::Write;
		case PF_R | PF_W: return Common::VirtualMemory::Mode::ReadWrite;
		// Windows PAGE_EXECUTE is not readable; ND InitializeMemory reads the image to size
		// CODE_AND_DATA. Map PF_X as RX (hardware code pages are effectively readable).
		case PF_X: return Common::VirtualMemory::Mode::ExecuteRead;
		case PF_X | PF_R: return Common::VirtualMemory::Mode::ExecuteRead;
		case PF_X | PF_W: return Common::VirtualMemory::Mode::ExecuteReadWrite;
		case PF_X | PF_W | PF_R: return Common::VirtualMemory::Mode::ExecuteReadWrite;

		default: return Common::VirtualMemory::Mode::NoAccess;
	}
}

struct FrameS {
	FrameS*   next;
	uintptr_t ret_addr;
};

static void KYTY_SYSV_ABI StackwalkX86(uint64_t rbp, void** stack, int* depth, uintptr_t stack_addr,
                                       size_t stack_size, uintptr_t code_addr, size_t code_size) {
	auto* frame = reinterpret_cast<FrameS*>(rbp);

	int d = *depth;
	int i = 0;

	for (; i < d; i++) {
		if (!(reinterpret_cast<uintptr_t>(frame) >= stack_addr &&
		      reinterpret_cast<uintptr_t>(frame) < stack_addr + stack_size)) {
			break;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		{
			MEMORY_BASIC_INFORMATION mbi {};
			if (VirtualQuery(frame, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT ||
			    (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
				break;
			}
			const auto region_end =
			    reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize;
			if (reinterpret_cast<uint64_t>(frame) + sizeof(FrameS) > region_end) {
				break;
			}
		}
#endif

		if (!(frame->ret_addr >= code_addr && frame->ret_addr < code_addr + code_size)) {
			break;
		}

		stack[i] = reinterpret_cast<void*>(frame->ret_addr);

		frame = frame->next;
	}

	*depth = i;
}

static void KYTY_SYSV_ABI SysStackWalkX86(uint64_t rbp, uint64_t rsp, void** stack, int* depth) {
	if (rsp == 0 || rbp < rsp) {
		*depth = 0;
		return;
	}

	StackwalkX86(rbp, stack, depth, rsp, 1024u * 1024u, SYSTEM_RESERVED + CODE_BASE_OFFSET,
	             g_desired_base_addr - (SYSTEM_RESERVED + CODE_BASE_OFFSET));
}

void KYTY_SYSV_ABI SysStackWalkX86(uint64_t rbp, void** stack, int* depth) {
	SysStackWalkX86(rbp, rbp, stack, depth);
}

static bool KytyExceptionHandler(const Common::HostException::ExceptionInfo& exception_info) {
	const auto* info = &exception_info;

	if (info->type == Common::HostException::ExceptionType::IllegalInstruction &&
	    Loader::X64InstructionEmulator::TryEmulate(info->native_context)) {
		return true;
	}

	if (info->type == Common::HostException::ExceptionType::AccessViolation) {
		using CoreAccess  = Common::HostException::AccessViolationType;
		using GpuAccess   = Libs::Graphics::PageFaultAccess;

		// SSE #GP on misaligned guest RSP is often reported as AV[-1] (see poison-rsp logs).
		if (Loader::X64InstructionEmulator::TryFixMisalignedSseAccess(
		        info->native_context, info->access_violation_vaddr)) {
			return true;
		}

		// Poisoned / non-canonical guest pointers (e.g. empty NdJob head) fault at ~0.
		// They are never GPU-tracked or demand-committed; skip recovery before diagnostics.
		const bool unresolvable_vaddr =
		    info->access_violation_vaddr == 0 ||
		    info->access_violation_vaddr == UINT64_MAX ||
		    info->access_violation_vaddr >= (1ull << 47);
		if (!unresolvable_vaddr) {
		const auto access = [&]() {
			switch (info->access_violation_type) {
				case CoreAccess::Read: return GpuAccess::Read;
				case CoreAccess::Write: return GpuAccess::Write;
				case CoreAccess::Execute: return GpuAccess::Execute;
				case CoreAccess::Unknown:
					EXIT("unknown access type for page fault at 0x%016" PRIx64 "\n",
					     info->access_violation_vaddr);
			}
			EXIT("invalid access type for page fault at 0x%016" PRIx64 "\n",
			     info->access_violation_vaddr);
		}();
		if (Libs::Graphics::GetRenderContext().GetGpuResources().HandleFault(
		        access, info->access_violation_vaddr)) {
			return true;
		}

		if (Libs::LibKernel::Memory::KernelHandleReservedRangeAccessViolation(
		        info->access_violation_vaddr)) {
			return true;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// Transient protect races: fault fires, then the page is already accessible again.
		// Cap retries per address — a permanent fault must not spin the VEH forever.
		{
			MEMORY_BASIC_INFORMATION mbi {};
			const auto               fault_addr = info->access_violation_vaddr;
			if (VirtualQuery(reinterpret_cast<const void*>(fault_addr), &mbi, sizeof(mbi)) != 0 &&
			    mbi.State == MEM_COMMIT && (mbi.Protect & PAGE_GUARD) == 0 &&
			    (mbi.Protect & PAGE_NOACCESS) == 0) {
				const DWORD prot = mbi.Protect;
				const bool readable =
				    (prot & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ |
				             PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY)) != 0;
				const bool writable =
				    (prot & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY |
				             PAGE_EXECUTE_WRITECOPY)) != 0;
				const bool executable =
				    (prot & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
				             PAGE_EXECUTE_WRITECOPY)) != 0;
				const bool ok =
				    (info->access_violation_type == CoreAccess::Read && readable) ||
				    (info->access_violation_type == CoreAccess::Write && writable) ||
				    (info->access_violation_type == CoreAccess::Execute && executable);
				if (ok) {
					static std::atomic<uint64_t> last_retry_addr {0};
					static std::atomic<uint32_t> last_retry_count {0};
					const uint64_t               prev = last_retry_addr.load(std::memory_order_relaxed);
					uint32_t                     count = 1;
					if (prev == fault_addr) {
						count = last_retry_count.fetch_add(1, std::memory_order_relaxed) + 1;
					} else {
						last_retry_addr.store(fault_addr, std::memory_order_relaxed);
						last_retry_count.store(1, std::memory_order_relaxed);
					}
					LOGF("MemoryTrace: AV resolved-retry addr=0x%016" PRIx64 " prot=0x%08" PRIx32
					     " count=%u\n",
					     fault_addr, static_cast<uint32_t>(prot), count);
					fprintf(stderr,
					        "MemoryTrace: AV resolved-retry addr=0x%016" PRIx64 " count=%u\n",
					        fault_addr, count);
					std::fflush(stderr);
					if (count <= 8) {
						return true;
					}
					LOGF("MemoryTrace: AV resolved-retry exhausted addr=0x%016" PRIx64 "\n",
					     fault_addr);
				}
			}
		}
#endif
		// Canonical guest AV not handled by GPU/reserved/retry — last-chance soft-continue
		// (misaligned SSE / decode-skip) instead of falling through to EXIT+LOGF (nests AVs).
		{
			const bool is_write = info->access_violation_type == CoreAccess::Write;
			if (Loader::X64InstructionEmulator::TryFixMisalignedSseAccess(
			        info->native_context, info->access_violation_vaddr)) {
				return true;
			}
			if (Loader::X64InstructionEmulator::TrySoftContinuePoisonAccess(
			        info->native_context, info->access_violation_vaddr, is_write, true)) {
				return true;
			}
			// GPU UMD null-page AV (e.g. nvwgf2umx Read[8] after soft-continued Read[0]).
			if (info->access_violation_vaddr < 0x10000ull) {
				bool gpu_umd = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
				HMODULE owner = nullptr;
				if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
				                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				                       reinterpret_cast<LPCSTR>(info->exception_address),
				                       &owner) != 0 &&
				    owner != nullptr) {
					char module_name[MAX_PATH] = {};
					if (GetModuleFileNameA(owner, module_name, MAX_PATH) != 0) {
						for (char* p = module_name; *p != '\0'; ++p) {
							if (*p >= 'A' && *p <= 'Z') {
								*p = static_cast<char>(*p - 'A' + 'a');
							}
						}
						gpu_umd = std::strstr(module_name, "nvwgf2um") != nullptr ||
						          std::strstr(module_name, "nvd3dum") != nullptr ||
						          std::strstr(module_name, "nvgpucomp") != nullptr ||
						          std::strstr(module_name, "nvoglv") != nullptr ||
						          std::strstr(module_name, "amdvlk") != nullptr ||
						          std::strstr(module_name, "amdxc64") != nullptr ||
						          std::strstr(module_name, "igc64") != nullptr ||
						          std::strstr(module_name, "igd10") != nullptr;
					}
				}
#endif
				if (gpu_umd && Loader::X64InstructionEmulator::TrySoftContinuePoisonAccess(
				                   info->native_context, info->access_violation_vaddr, is_write,
				                   true, true)) {
					static std::atomic<uint32_t> umd_canon_n {0};
					const uint32_t n = umd_canon_n.fetch_add(1, std::memory_order_relaxed) + 1;
					char line[192];
					std::snprintf(line, sizeof(line),
					              "MemoryTrace: GPU UMD null-page soft-continue addr=0x%016" PRIx64
					              " rip=0x%016" PRIx64 " n=%u",
					              info->access_violation_vaddr, info->exception_address, n);
					Common::LogFatalToFile(line);
					std::fprintf(stderr, "%s\n", line);
					// Do not park — can freeze GraphicsRing / flip producer threads.
					return true;
				}
			}

			// System-DLL RIP: only CONTINUE_SEARCH (defer-to-OS) for true Windows thread-stack
			// growth (same AllocationBase as GetCurrentThreadStackLimits). Guest-stack / orphan
			// RESERVE pages must be demand-committed by us — the OS will not.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			{
				const uint64_t fault_va = info->access_violation_vaddr;
				if (Libs::LibKernel::Memory::KernelHandleReservedRangeAccessViolation(fault_va)) {
					return true;
				}

				HMODULE owner = nullptr;
				if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
				                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				                       reinterpret_cast<LPCSTR>(info->exception_address),
				                       &owner) != 0 &&
				    owner != nullptr) {
					char module_name[MAX_PATH] = {};
					if (GetModuleFileNameA(owner, module_name, MAX_PATH) != 0) {
						for (char* p = module_name; *p != '\0'; ++p) {
							if (*p >= 'A' && *p <= 'Z') {
								*p = static_cast<char>(*p - 'A' + 'a');
							}
						}
						if (std::strstr(module_name, "\\ntdll.dll") != nullptr ||
						    std::strstr(module_name, "\\kernel32.dll") != nullptr ||
						    std::strstr(module_name, "\\kernelbase.dll") != nullptr) {
							bool host_stack_growth = false;
							ULONG_PTR stack_low    = 0;
							ULONG_PTR stack_high   = 0;
							GetCurrentThreadStackLimits(&stack_low, &stack_high);
							MEMORY_BASIC_INFORMATION fault_mbi {};
							MEMORY_BASIC_INFORMATION stack_mbi {};
							if (VirtualQuery(reinterpret_cast<const void*>(fault_va), &fault_mbi,
							                 sizeof(fault_mbi)) != 0 &&
							    VirtualQuery(reinterpret_cast<const void*>(stack_low), &stack_mbi,
							                 sizeof(stack_mbi)) != 0 &&
							    fault_mbi.AllocationBase != nullptr &&
							    fault_mbi.AllocationBase == stack_mbi.AllocationBase &&
							    (fault_mbi.State == MEM_RESERVE ||
							     (fault_mbi.Protect & PAGE_GUARD) != 0)) {
								host_stack_growth = true;
							}
							char line[320];
							if (host_stack_growth) {
								std::snprintf(line, sizeof(line),
								              "MemoryTrace: AV defer-to-OS rip=0x%016" PRIx64
								              " addr=0x%016" PRIx64 " mod=%s",
								              info->exception_address, fault_va, module_name);
								Common::LogFatalToFile(line);
								std::fprintf(stderr, "%s\n", line);
								return false;
							}
							// CFG bitmap (ntdll!LdrpValidateUserCallTarget): high MEM_MAPPED
							// sparse section. Soft-continue via TrySoftContinueCfgBitmap.
							// Main may return false → CONTINUE_SEARCH for the OS pager
							// (SharpEmu contract). Do not Sleep-park — that freezes producers.
							MEMORY_BASIC_INFORMATION cfg_mbi {};
							if (VirtualQuery(reinterpret_cast<const void*>(fault_va), &cfg_mbi,
							                 sizeof(cfg_mbi)) != 0 &&
							    cfg_mbi.Type == MEM_MAPPED && cfg_mbi.AllocationBase != nullptr &&
							    reinterpret_cast<uint64_t>(cfg_mbi.AllocationBase) >=
							        0x0000700000000000ull &&
							    (cfg_mbi.State == MEM_RESERVE ||
							     (cfg_mbi.State == MEM_COMMIT &&
							      (cfg_mbi.Protect & PAGE_NOACCESS) != 0))) {
								if (Loader::X64InstructionEmulator::TrySoftContinueCfgBitmap(
								        info->native_context, fault_va)) {
									std::snprintf(line, sizeof(line),
									              "MemoryTrace: AV CFG-bitmap handled "
									              "rip=0x%016" PRIx64 " addr=0x%016" PRIx64
									              " alloc=0x%016" PRIx64 " rcx=0x%016" PRIx64,
									              info->exception_address, fault_va,
									              reinterpret_cast<uint64_t>(cfg_mbi.AllocationBase),
									              info->rcx);
									Common::LogFatalToFile(line);
									std::fprintf(stderr, "%s\n", line);
									return true;
								}
								std::snprintf(line, sizeof(line),
								              "MemoryTrace: AV CFG-bitmap defer-OS "
								              "rip=0x%016" PRIx64 " addr=0x%016" PRIx64
								              " alloc=0x%016" PRIx64 " rcx=0x%016" PRIx64,
								              info->exception_address, fault_va,
								              reinterpret_cast<uint64_t>(cfg_mbi.AllocationBase),
								              info->rcx);
								Common::LogFatalToFile(line);
								std::fprintf(stderr, "%s\n", line);
								return false;
							}
							std::snprintf(line, sizeof(line),
							              "MemoryTrace: AV system-dll non-host-stack "
							              "rip=0x%016" PRIx64 " addr=0x%016" PRIx64 " mod=%s",
							              info->exception_address, fault_va, module_name);
							Common::LogFatalToFile(line);
							std::fprintf(stderr, "%s\n", line);
							// RtlEnterCriticalSection(NULL) → Write[8]; unwind one frame instead
							// of soft-halt (blanket ntdll soft-continue still unsafe).
							if (Loader::X64InstructionEmulator::TrySoftContinueNullCriticalSection(
							        info->native_context, fault_va)) {
								return true;
							}
							// Fall through to soft-halt — OS cannot commit this MEM_MAPPED RESERVE
							// (VirtualAlloc/Unmap/VirtualFree all return ERROR_INVALID_PARAMETER).
							// Soft-continuing ntdll here corrupts unwind and dies with 0xC0000005.
						}
					}
				}
			}
#endif

			if (Loader::X64InstructionEmulator::TrySoftContinueNullCriticalSection(
			        info->native_context, info->access_violation_vaddr)) {
				return true;
			}

			char        line[448];
			const char* rip_mod = "?";
			char        rip_mod_buf[260] = {};
			uint64_t    rip_mod_off     = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			{
				HMODULE owner = nullptr;
				if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
				                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				                       reinterpret_cast<LPCSTR>(info->exception_address),
				                       &owner) != 0 &&
				    owner != nullptr &&
				    GetModuleFileNameA(owner, rip_mod_buf, static_cast<DWORD>(sizeof(rip_mod_buf))) !=
				        0) {
					rip_mod = rip_mod_buf;
					for (char* p = rip_mod_buf; *p != '\0'; ++p) {
						if (*p == '\\' || *p == '/') {
							rip_mod = p + 1;
						}
					}
					rip_mod_off = info->exception_address - reinterpret_cast<uint64_t>(owner);
				}
			}
#endif
			std::snprintf(line, sizeof(line),
			              "MemoryTrace: AV unhandled canonical addr=0x%016" PRIx64
			              " rip=0x%016" PRIx64 " av=%u mod=%s+0x%llX — soft-halt 321",
			              info->access_violation_vaddr, info->exception_address,
			              static_cast<unsigned>(info->access_violation_type), rip_mod,
			              static_cast<unsigned long long>(rip_mod_off));
			Common::LogFatalToFile(line);
			std::fprintf(stderr, "%s\n", line);
			{
				char regs[384];
				std::snprintf(regs, sizeof(regs),
				              "MemoryTrace: AV soft-halt regs rax=%016" PRIx64 " rcx=%016" PRIx64
				              " rdx=%016" PRIx64 " rbx=%016" PRIx64 " rsi=%016" PRIx64
				              " rdi=%016" PRIx64 " rsp=%016" PRIx64 " rbp=%016" PRIx64,
				              info->rax, info->rcx, info->rdx, info->rbx, info->rsi, info->rdi,
				              info->rsp, info->rbp);
				Common::LogFatalToFile(regs);
				std::fprintf(stderr, "%s\n", regs);
				std::snprintf(regs, sizeof(regs),
				              "MemoryTrace: AV soft-halt regs r8=%016" PRIx64 " r9=%016" PRIx64
				              " r10=%016" PRIx64 " r11=%016" PRIx64 " r12=%016" PRIx64
				              " r13=%016" PRIx64 " r14=%016" PRIx64 " r15=%016" PRIx64,
				              info->r8, info->r9, info->r10, info->r11, info->r12, info->r13,
				              info->r14, info->r15);
				Common::LogFatalToFile(regs);
				std::fprintf(stderr, "%s\n", regs);
			}
			Common::FlushHleRingToFatal("av_unhandled_canonical");
			Common::NoteHaltReason("av_unhandled", line);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			TerminateProcess(GetCurrentProcess(), 321);
#endif
			std::_Exit(321);
		}
		} else {
			// VEH-safe fatal path: no LOGF/EnumName/EXIT/stack walk (those can nest another AV).
			const char* av_type = "Unknown";
			switch (info->access_violation_type) {
				case CoreAccess::Read: av_type = "Read"; break;
				case CoreAccess::Write: av_type = "Write"; break;
				case CoreAccess::Execute: av_type = "Execute"; break;
				case CoreAccess::Unknown: break;
			}
			char line[384];
			std::snprintf(line, sizeof(line),
			              "MemoryTrace: AV unresolvable addr=0x%016" PRIx64 " rip=0x%016" PRIx64
			              " av=%s",
			              info->access_violation_vaddr, info->exception_address, av_type);
			Common::LogFatalToFile(line);
			std::fprintf(stderr, "%s\n", line);
			std::snprintf(line, sizeof(line),
			              "regs: rax=%016" PRIx64 " rbx=%016" PRIx64 " rcx=%016" PRIx64
			              " rdx=%016" PRIx64,
			              info->rax, info->rbx, info->rcx, info->rdx);
			Common::LogFatalToFile(line);
			std::snprintf(line, sizeof(line),
			              "regs: rsi=%016" PRIx64 " rdi=%016" PRIx64 " rbp=%016" PRIx64
			              " rsp=%016" PRIx64,
			              info->rsi, info->rdi, info->rbp, info->rsp);
			Common::LogFatalToFile(line);
			std::snprintf(line, sizeof(line),
			              "regs: r8 =%016" PRIx64 " r9 =%016" PRIx64 " r10=%016" PRIx64
			              " r11=%016" PRIx64,
			              info->r8, info->r9, info->r10, info->r11);
			Common::LogFatalToFile(line);
			std::snprintf(line, sizeof(line),
			              "regs: r12=%016" PRIx64 " r13=%016" PRIx64 " r14=%016" PRIx64
			              " r15=%016" PRIx64,
			              info->r12, info->r13, info->r14, info->r15);
			Common::LogFatalToFile(line);
			if (info->access_violation_vaddr == UINT64_MAX) {
				auto log_if_neg1 = [&](const char* name, uint64_t value) {
					if (value == UINT64_MAX) {
						char neg[64];
						std::snprintf(neg, sizeof(neg), "guest reg -1: %s", name);
						Common::LogFatalToFile(neg);
						std::fprintf(stderr, "%s\n", neg);
					}
				};
				log_if_neg1("rax", info->rax);
				log_if_neg1("rbx", info->rbx);
				log_if_neg1("rcx", info->rcx);
				log_if_neg1("rdx", info->rdx);
				log_if_neg1("rsi", info->rsi);
				log_if_neg1("rdi", info->rdi);
				log_if_neg1("r8", info->r8);
				log_if_neg1("r9", info->r9);
				log_if_neg1("r10", info->r10);
				log_if_neg1("r11", info->r11);
				log_if_neg1("r12", info->r12);
				log_if_neg1("r13", info->r13);
				log_if_neg1("r14", info->r14);
				log_if_neg1("r15", info->r15);
			}
			std::snprintf(line, sizeof(line),
			              "Access violation: %s [%016" PRIx64 "] (non-canonical/poisoned)", av_type,
			              info->access_violation_vaddr);
			Common::LogFatalToFile(line);
			Common::FlushHleRingToFatal("av_unresolvable");
			// Host-only diagnostics: match RIP to unresolved-import thunk pages / hot stubs.
			{
				constexpr uint64_t kThunkSize = 162;
				constexpr uint64_t kPageSize  = 4096;
				bool              matched    = false;
				for (uint64_t page: g_unresolved_stub_thunk_pages) {
					if (info->exception_address < page ||
					    info->exception_address >= page + kPageSize) {
						continue;
					}
					std::snprintf(line, sizeof(line),
					              "poison-rip in stub-thunk page=0x%016" PRIx64 " off=0x%llx", page,
					              static_cast<unsigned long long>(info->exception_address - page));
					Common::LogFatalToFile(line);
					for (const auto& record: g_stubbed_imports) {
						if (record.thunk_vaddr == 0) {
							continue;
						}
						if (info->exception_address >= record.thunk_vaddr &&
						    info->exception_address < record.thunk_vaddr + kThunkSize) {
							std::snprintf(line, sizeof(line),
							              "poison-rip stub: %s program=%s patch=0x%016" PRIx64
							              " calls=%" PRIu64,
							              record.name.c_str(), record.program.c_str(),
							              record.patch_vaddr, record.call_count);
							Common::LogFatalToFile(line);
							matched = true;
							break;
						}
					}
					break;
				}
				if (!matched) {
					Common::LogFatalToFile(
					    "poison-rip not inside an unresolved-import thunk (likely guest caller "
					    "after stub returned 0 / null)");
				}
				Libs::VideoOut::PhaseDescribeFaultRip(info->exception_address, info->rsp);
				const StubbedImportRecord* top[12] = {};
				for (const auto& record: g_stubbed_imports) {
					if (record.call_count == 0) {
						continue;
					}
					for (int i = 0; i < 12; i++) {
						if (top[i] == nullptr || record.call_count > top[i]->call_count) {
							for (int j = 11; j > i; j--) {
								top[j] = top[j - 1];
							}
							top[i] = &record;
							break;
						}
					}
				}
				for (const auto* record: top) {
					if (record == nullptr) {
						break;
					}
					std::snprintf(line, sizeof(line), "stub-hot: n=%" PRIu64 " %s prog=%s",
					              record->call_count, record->name.c_str(),
					              record->program.c_str());
					Common::LogFatalToFile(line);
				}
			}

			const bool is_write = info->access_violation_type == CoreAccess::Write;
			char       abort_detail[160] {};
			const bool guest_abort_trap = Loader::X64InstructionEmulator::DescribeGuestAbortTrap(
			    info->exception_address, abort_detail, sizeof(abort_detail));
			if (!guest_abort_trap) {
				if (Loader::X64InstructionEmulator::TryFixMisalignedSseAccess(
				        info->native_context, info->access_violation_vaddr)) {
					return true;
				}
				if (Loader::X64InstructionEmulator::TrySoftContinuePoisonAccess(
				        info->native_context, info->access_violation_vaddr, is_write)) {
					Common::NoteHaltReason("poison", "soft-continued AV -1/non-canonical");
					std::fflush(stderr);
					return true;
				}
				// NVIDIA/AMD/Intel UMD null-page deref during VideoOut create/upload: skip the
				// faulting memop instead of soft-halting (PPSA21564 RegisterBuffers2).
				{
					bool gpu_umd = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
					HMODULE owner = nullptr;
					if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
					                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
					                       reinterpret_cast<LPCSTR>(info->exception_address),
					                       &owner) != 0 &&
					    owner != nullptr) {
						char module_name[MAX_PATH] = {};
						if (GetModuleFileNameA(owner, module_name, MAX_PATH) != 0) {
							const char* base = module_name;
							for (const char* p = module_name; *p != '\0'; ++p) {
								if (*p == '\\' || *p == '/') {
									base = p + 1;
								}
							}
							char lower[MAX_PATH] = {};
							size_t n = 0;
							for (; base[n] != '\0' && n + 1 < sizeof(lower); ++n) {
								const unsigned char c = static_cast<unsigned char>(base[n]);
								lower[n] = static_cast<char>(
								    (c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c);
							}
							lower[n] = '\0';
							gpu_umd = std::strstr(lower, "nvwgf2um") != nullptr ||
							          std::strstr(lower, "nvd3dum") != nullptr ||
							          std::strstr(lower, "nvgpucomp") != nullptr ||
							          std::strstr(lower, "nvoglv") != nullptr ||
							          std::strstr(lower, "amdvlk") != nullptr ||
							          std::strstr(lower, "amdxc64") != nullptr ||
							          std::strstr(lower, "atiogl") != nullptr ||
							          std::strstr(lower, "igc64") != nullptr ||
							          std::strstr(lower, "igd10") != nullptr ||
							          std::strstr(lower, "intelocl") != nullptr;
							if (gpu_umd) {
								std::snprintf(line, sizeof(line),
								              "MemoryTrace: GPU UMD poison AV mod=%s — "
								              "soft-continue allow_system_module",
								              base);
								Common::LogFatalToFile(line);
								std::fprintf(stderr, "%s\n", line);
							}
						}
					}
#endif
					const bool null_page = info->access_violation_vaddr < 0x10000ull;
					if (gpu_umd && null_page) {
						static std::atomic<uint32_t> umd_soft_n {0};
						const uint32_t n =
						    umd_soft_n.fetch_add(1, std::memory_order_relaxed) + 1;
						// Never park: GraphicsRing PrepareCpuFlip / present can fault in the
						// UMD; parking that thread leaves VO stuck at Recording forever.
						if (n > 64 && (n % 256u) == 0u) {
							std::snprintf(line, sizeof(line),
							              "MemoryTrace: GPU UMD null-page soft-continue "
							              "n=%u (no park — keep producer alive)",
							              n);
							Common::LogFatalToFile(line);
							std::fprintf(stderr, "%s\n", line);
						}
						if (Loader::X64InstructionEmulator::TrySoftContinuePoisonAccess(
						        info->native_context, info->access_violation_vaddr, is_write,
						        /*force=*/true, /*allow_system_module=*/true)) {
							Common::NoteHaltReason("poison",
							                       "soft-continued GPU UMD null-page AV");
							std::fflush(stderr);
							return true;
						}
					}
				}
			} else {
				// Prefer guest C-strings often left in SysV arg regs after puts/assert.
				auto log_guest_cstr = [](const char* reg, uint64_t addr) {
					if (addr == 0 || addr == UINT64_MAX || addr >= (1ull << 47)) {
						return;
					}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
					MEMORY_BASIC_INFORMATION mbi {};
					if (VirtualQuery(reinterpret_cast<const void*>(addr), &mbi, sizeof(mbi)) == 0 ||
					    mbi.State != MEM_COMMIT ||
					    (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
						return;
					}
					const auto*        p   = reinterpret_cast<const char*>(addr);
					const auto         end = reinterpret_cast<uint64_t>(mbi.BaseAddress) +
					                 mbi.RegionSize;
					char               buf[240];
					size_t             n = 0;
					while (n + 1 < sizeof(buf) && addr + n < end) {
						const char c = p[n];
						if (c == '\0') {
							break;
						}
						if (c < 0x20 && c != '\t' && c != '\n' && c != '\r') {
							return; // not a C string
						}
						buf[n++] = c;
					}
					if (n == 0) {
						return;
					}
					buf[n] = '\0';
					char line[320];
					std::snprintf(line, sizeof(line), "guest_abort_trap %s=\"%s\"", reg, buf);
					Common::LogFatalToFile(line);
					std::fprintf(stderr, "%s\n", line);
#else
					(void)reg;
#endif
				};
				log_guest_cstr("rdi", info->rdi);
				log_guest_cstr("rsi", info->rsi);
				log_guest_cstr("rcx", info->rcx);
				log_guest_cstr("r8", info->r8);
				Common::LogFatalToFile(abort_detail);
				std::fprintf(stderr, "guest_abort_trap: %s\n", abort_detail);
				Common::NoteHaltReason("guest_abort_trap", abort_detail);
				Common::FlushHleRingToFatal("guest_abort_trap");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
				TerminateProcess(GetCurrentProcess(), 321);
#endif
				std::_Exit(321);
			}

			Common::NoteHaltReason("poison", "unresolvable AV — halt 321");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			TerminateProcess(GetCurrentProcess(), 321);
#endif
			std::_Exit(321);
		}
	}

	LOGF("kyty_exception_handler: %016" PRIx64 "\n", info->exception_address);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HMODULE owner_module = nullptr;
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
	                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCSTR>(info->exception_address), &owner_module) != 0 &&
	    owner_module != nullptr) {
		char module_name[MAX_PATH] = {};
		if (GetModuleFileNameA(owner_module, module_name, MAX_PATH) != 0) {
			LOGF("exception module: %s\n", module_name);
		}
	}
#endif
	if (info->exception_address != 0) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		MEMORY_BASIC_INFORMATION mem_info = {};
		auto* dump_ptr = reinterpret_cast<const uint8_t*>(info->exception_address - 32);
		if (VirtualQuery(dump_ptr, &mem_info, sizeof(mem_info)) != 0 &&
		    mem_info.State == MEM_COMMIT && mem_info.Protect != PAGE_NOACCESS &&
		    (mem_info.Protect & PAGE_GUARD) == 0) {
			const auto dump_start = reinterpret_cast<uint64_t>(dump_ptr);
			const auto region_end =
			    reinterpret_cast<uint64_t>(mem_info.BaseAddress) + mem_info.RegionSize;
			const auto dump_size =
			    (dump_start + 64 <= region_end ? 64u
			                                   : static_cast<uint32_t>(region_end - dump_start));
			LOGF("code-32:");
			for (uint32_t i = 0; i < dump_size; i++) {
				LOGF(" %02" PRIx32, static_cast<uint32_t>(dump_ptr[i]));
			}
			LOGF("\n");
		} else {
			LOGF("code-32: unavailable\n");
		}
#else
		LOGF("code-32:");
		for (uint64_t i = 0; i < 64; i++) {
			LOGF(" %02" PRIx32, static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(
			                        info->exception_address + i - 32)));
		}
		LOGF("\n");
#endif
	} else {
		LOGF("code: unavailable\n");
	}
	LOGF("exception: type=%s, av_type=%s, av_addr=%016" PRIx64 ", native_code=%08" PRIx32 "\n",
	     Common::EnumName(info->type).c_str(),
	     Common::EnumName(info->access_violation_type).c_str(), info->access_violation_vaddr,
	     info->native_code);
	LOGF("regs: rax=%016" PRIx64 " rbx=%016" PRIx64 " rcx=%016" PRIx64 " rdx=%016" PRIx64 "\n",
	     info->rax, info->rbx, info->rcx, info->rdx);
	LOGF("regs: rsi=%016" PRIx64 " rdi=%016" PRIx64 " rbp=%016" PRIx64 " rsp=%016" PRIx64 "\n",
	     info->rsi, info->rdi, info->rbp, info->rsp);
	LOGF("regs: r8 =%016" PRIx64 " r9 =%016" PRIx64 " r10=%016" PRIx64 " r11=%016" PRIx64 "\n",
	     info->r8, info->r9, info->r10, info->r11);
	LOGF("regs: r12=%016" PRIx64 " r13=%016" PRIx64 " r14=%016" PRIx64 " r15=%016" PRIx64 "\n",
	     info->r12, info->r13, info->r14, info->r15);

	auto is_readable_range = [](uint64_t addr, uint64_t size) {
		if (addr == 0 || size == 0) {
			return false;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		uint64_t current = addr;
		uint64_t end     = addr + size;
		if (end < addr) {
			return false;
		}
		while (current < end) {
			MEMORY_BASIC_INFORMATION mbi {};
			if (VirtualQuery(reinterpret_cast<const void*>(current), &mbi, sizeof(mbi)) == 0 ||
			    mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
				return false;
			}
			const auto region_end = reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize;
			if (region_end <= current) {
				return false;
			}
			current = std::min(region_end, end);
		}
#else
		(void)addr;
		(void)size;
#endif
		return true;
	};

	if (is_readable_range(info->rsp, 16u * sizeof(uint64_t))) {
		auto* stack = reinterpret_cast<const uint64_t*>(info->rsp);
		LOGF("stack:");
		for (uint64_t i = 0; i < 16; i++) {
			LOGF(" [%02" PRIu64 "]=%016" PRIx64, i, stack[i]);
		}
		LOGF("\n");
	} else {
		LOGF("stack: unavailable\n");
	}

	auto dump_guest_code = [](const char* name, uint64_t addr) {
		auto* p = Common::Singleton<Loader::RuntimeLinker>::Instance()->FindProgramByAddr(addr);
		if (p == nullptr || addr < p->base_vaddr) {
			return;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		MEMORY_BASIC_INFORMATION mbi {};
		auto* dump_ptr = reinterpret_cast<const uint8_t*>(addr >= 16 ? addr - 16 : addr);
		if (VirtualQuery(dump_ptr, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT ||
		    (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
			return;
		}
		const auto dump_start = reinterpret_cast<uint64_t>(dump_ptr);
		const auto region_end = reinterpret_cast<uint64_t>(mbi.BaseAddress) + mbi.RegionSize;
		const auto dump_size =
		    (dump_start + 32 <= region_end ? 32u : static_cast<uint32_t>(region_end - dump_start));
#else
		auto*      dump_ptr  = reinterpret_cast<const uint8_t*>(addr >= 16 ? addr - 16 : addr);
		const auto dump_size = 32u;
#endif

		LOGF("%s code: addr=%016" PRIx64 ", off=%016" PRIx64 ", module=%s:", name, addr,
		     addr - p->base_vaddr,
		     Common::FilenameWithoutDirectory(Common::PathToGenericString(p->file_name)).c_str());
		for (uint32_t i = 0; i < dump_size; i++) {
			LOGF(" %02" PRIx32, static_cast<uint32_t>(dump_ptr[i]));
		}
		LOGF("\n");
	};

	dump_guest_code("guest rax[0]", info->rax);
	dump_guest_code("guest rbx[0]", info->rbx);
	dump_guest_code("guest rcx[0]", info->rcx);
	dump_guest_code("guest rsi[0]", info->rsi);
	if (is_readable_range(info->rsp, 16u * sizeof(uint64_t))) {
		auto* stack = reinterpret_cast<const uint64_t*>(info->rsp);
		for (uint64_t i = 0; i < 16; i++) {
			char name[32] {};
			std::snprintf(name, sizeof(name), "stack[%" PRIu64 "]", i);
			dump_guest_code(name, stack[i]);
		}
	}

	if (info->type == Common::HostException::ExceptionType::AccessViolation) {
		if (info->rbp != 0) {
			void* stack[20];
			int   depth = 20;
			SysStackWalkX86(info->rbp, info->rsp, stack, &depth);

			LOGF("Stack trace [thread = %d]:\n", Common::Thread::GetThreadIdUnique());
			for (int i = 0; i < depth; i++) {
				auto  vaddr = reinterpret_cast<uint64_t>(stack[i]);
				auto* p =
				    Common::Singleton<Loader::RuntimeLinker>::Instance()->FindProgramByAddr(vaddr);
				LOGF("[%d] %016" PRIx64 ", off=%016" PRIx64 ", %s\n", i, vaddr,
				     (p == nullptr ? 0 : vaddr - p->base_vaddr),
				     (p == nullptr ? "???"
				                   : Common::FilenameWithoutDirectory(
				                         Common::PathToGenericString(p->file_name))
				                         .c_str()));
			}
		}

		auto dump_guest_qwords = [&is_readable_range](const char* name, uint64_t addr) {
			if (addr == 0) {
				LOGF("%s = 0\n", name);
				return;
			}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			MEMORY_BASIC_INFORMATION mbi {};
			if (VirtualQuery(reinterpret_cast<const void*>(addr), &mbi, sizeof(mbi)) == 0 ||
			    mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
				LOGF("%s = %016" PRIx64 " (unmapped)\n", name, addr);
				return;
			}
#endif

			if (!is_readable_range(addr, 8u * sizeof(uint64_t))) {
				LOGF("%s = %016" PRIx64 " (unmapped)\n", name, addr);
				return;
			}

			auto* q = reinterpret_cast<const uint64_t*>(addr);
			LOGF("%s = %016" PRIx64 ": %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64
			     " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
			     name, addr, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
		};

		dump_guest_qwords("guest rbx", info->rbx);
		dump_guest_qwords("guest rax", info->rax);
		dump_guest_qwords("guest rcx", info->rcx);
		dump_guest_qwords("guest rsi", info->rsi);
		dump_guest_qwords("guest rdi", info->rdi);
		dump_guest_qwords("guest r8 ", info->r8);
		dump_guest_qwords("guest r9 ", info->r9);
		dump_guest_qwords("guest r10", info->r10);
		dump_guest_qwords("guest r11", info->r11);
		dump_guest_qwords("guest r12", info->r12);
		dump_guest_qwords("guest r13", info->r13);
		dump_guest_qwords("guest r14", info->r14);
		dump_guest_qwords("guest r15", info->r15);

		if (info->exception_address == 0x000000090064364e && info->rbx != 0 &&
		    is_readable_range(info->rbx, sizeof(uint64_t))) {
			auto* local = reinterpret_cast<const uint64_t*>(info->rbx);
			dump_guest_qwords("vorbis obj", local[0]);
			dump_guest_qwords("vorbis len", info->rcx);
		}

		if (info->access_violation_vaddr == UINT64_MAX) {
			LOGF("guest registers equal to -1:");
			if (info->rax == UINT64_MAX) {
				LOGF("\trax\n");
			}
			if (info->rbx == UINT64_MAX) {
				LOGF("\trbx\n");
			}
			if (info->rcx == UINT64_MAX) {
				LOGF("\trcx\n");
			}
			if (info->rdx == UINT64_MAX) {
				LOGF("\trdx\n");
			}
			if (info->rsi == UINT64_MAX) {
				LOGF("\trsi\n");
			}
			if (info->rdi == UINT64_MAX) {
				LOGF("\trdi\n");
			}
			if (info->r8 == UINT64_MAX) {
				LOGF("\tr8\n");
			}
			if (info->r9 == UINT64_MAX) {
				LOGF("\tr9\n");
			}
			if (info->r10 == UINT64_MAX) {
				LOGF("\tr10\n");
			}
			if (info->r11 == UINT64_MAX) {
				LOGF("\tr11\n");
			}
			if (info->r12 == UINT64_MAX) {
				LOGF("\tr12\n");
			}
			if (info->r13 == UINT64_MAX) {
				LOGF("\tr13\n");
			}
			if (info->r14 == UINT64_MAX) {
				LOGF("\tr14\n");
			}
			if (info->r15 == UINT64_MAX) {
				LOGF("\tr15\n");
			}
		}

		WriteStubbedImportReport("_ImportReport.txt");

		EXIT("Access violation: %s [%016" PRIx64 "] %s\n",
		     Common::EnumName(info->access_violation_type).c_str(), info->access_violation_vaddr,
		     (info->access_violation_vaddr == g_invalid_memory ? "(Unpatched object)" : ""));
		return false;
	}

	EXIT("Unknown exception!!! (%08" PRIx32 ")", info->native_code);
	return false;
}

static void EncodeId64(uint16_t in_id, std::string* out_id) {
	static const char* str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
	if (in_id < 0x40u) {
		*out_id += str[in_id];
	} else {
		if (in_id < 0x1000u) {
			*out_id += str[static_cast<uint16_t>(in_id >> 6u) & 0x3fu];
			*out_id += str[in_id & 0x3fu];
		} else {
			*out_id += str[static_cast<uint16_t>(in_id >> 12u) & 0x3fu];
			*out_id += str[static_cast<uint16_t>(in_id >> 6u) & 0x3fu];
			*out_id += str[in_id & 0x3fu];
		}
	}
}

template <class T>
static void GetDynDataOs(Elf64* elf, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = elf->GetDynamicData<T>(dyn->d_un.d_ptr);
	}
}

template <class T>
static void GetDynData(Elf64* elf, uint64_t base_vaddr, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = reinterpret_cast<T>(base_vaddr + dyn->d_un.d_ptr);
	}
}

template <class T>
static void GetDynValue(Elf64* elf, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = dyn->d_un.d_val;
	}
}

template <class T>
static void GetDynValues(Elf64* elf, T* out, Elf64_Sxword tag) {
	for (const auto* dyn: elf->GetDynList(tag)) {
		out->push_back(dyn->d_un.d_val);
	}
}

template <class T>
static void GetDynPtr(Elf64* elf, T* out, Elf64_Sxword tag) {
	if (const auto* dyn = elf->GetDynValue(tag); dyn != nullptr) {
		*out = dyn->d_un.d_ptr;
	}
}

static void KYTY_SYSV_ABI ProgramExitHandler() {
	Common::Singleton<RuntimeLinker>::Instance()->StopAllModules();

	LOGF("exit!!!\n");
}

template <class T>
static void GetDynModules(Elf64* elf, T* out, const char* names, Elf64_Sxword tag) {
	std::vector<uint64_t> needed_modules;
	GetDynValues(elf, &needed_modules, tag);
	for (auto need: needed_modules) {
		ModuleId id {};
		// id.id            = static_cast<int>((need >> 48u) & 0xffffu);
		EncodeId64(static_cast<uint16_t>((need >> 48u) & 0xffffu), &id.id);
		id.version_major = static_cast<int>((need >> 40u) & 0xffu);
		id.version_minor = static_cast<int>((need >> 32u) & 0xffu);
		id.name          = names + (need & 0xffffffff);
		out->push_back(id);
	}
}

template <class T>
static void GetDynLibs(Elf64* elf, T* out, const char* names, Elf64_Sxword tag) {
	std::vector<uint64_t> needed_modules;
	GetDynValues(elf, &needed_modules, tag);
	for (auto need: needed_modules) {
		LibraryId id {};
		// id.id      = static_cast<int>((need >> 48u) & 0xffffu);
		EncodeId64(static_cast<uint16_t>((need >> 48u) & 0xffffu), &id.id);
		id.version = static_cast<int>((need >> 32u) & 0xffffu);
		id.name    = names + (need & 0xffffffff);
		out->push_back(id);
	}
}

static RelocationInfo GetRelocationInfo(Elf64_Rela* r, Program* program) {
	KYTY_PROFILER_FUNCTION();

	// KYTY_PROFILER_BLOCK("1");

	RelocationInfo ret;
	// SymbolRecord   sr {};

	// KYTY_PROFILER_END_BLOCK;

	// KYTY_PROFILER_BLOCK("2");

	auto         type    = r->GetType();
	auto         symbol  = r->GetSymbol();
	Elf64_Sxword addend  = r->r_addend;
	auto*        symbols = program->dynamic_info->symbol_table;
	auto*        names   = program->dynamic_info->str_table;
	ret.base_vaddr       = program->base_vaddr;
	ret.vaddr            = ret.base_vaddr + r->r_offset;
	ret.bind_self        = false;

	// KYTY_PROFILER_END_BLOCK;

	// KYTY_PROFILER_BLOCK("3");

	switch (type) {
		case R_X86_64_GLOB_DAT:
		case R_X86_64_JUMP_SLOT: addend = 0; [[fallthrough]];
		case R_X86_64_64: {
			auto         sym          = symbols[symbol];
			auto         bind         = sym.GetBind();
			auto         sym_type     = sym.GetType();
			uint64_t     symbol_vaddr = 0;
			SymbolRecord sr {};
			switch (sym_type) {
				case STT_NOTYPE: ret.type = SymbolType::NoType; break;
				case STT_FUNC: ret.type = SymbolType::Func; break;
				case STT_OBJECT: ret.type = SymbolType::Object; break;
				default: EXIT("unknown symbol type: %d\n", (int)sym_type);
			}
			switch (bind) {
				case STB_LOCAL:
					symbol_vaddr = ret.base_vaddr + sym.st_value;
					ret.bind     = BindType::Local;
					break;
				case STB_GLOBAL: ret.bind = BindType::Global; [[fallthrough]];
				case STB_WEAK: {
					ret.bind = (ret.bind == BindType::Unknown ? BindType::Weak : ret.bind);
					ret.name = names + sym.st_name;
					program->rt->Resolve(ret.name, ret.type, program, &sr, &ret.bind_self);
					symbol_vaddr = sr.vaddr;
				} break;
				default: EXIT("unknown bind: %d\n", (int)bind);
			}
			ret.resolved = (symbol_vaddr != 0);
			ret.value    = (ret.resolved ? symbol_vaddr + addend : 0);
			ret.name     = sr.name;
			ret.dbg_name = sr.dbg_name;
		} break;
		case R_X86_64_RELATIVE:
			ret.value    = ret.base_vaddr + addend;
			ret.resolved = true;
			break;
		case R_X86_64_DTPMOD64:
			ret.value    = reinterpret_cast<uint64_t>(program);
			ret.resolved = true;
			ret.type     = SymbolType::TlsModule;
			ret.bind     = BindType::Local;
			ret.dbg_name = Common::PathToString(program->file_name);
			break;
		default: EXIT("unknown type: %d\n", (int)type);
	}

	// KYTY_PROFILER_END_BLOCK;

	return ret;
}

static void RelocateRecord(uint32_t index, Elf64_Rela* r, Program* program, bool jmprela_table,
                           bool imports_only, std::vector<std::string>* unresolved) {
	KYTY_PROFILER_FUNCTION();

	auto ri = GetRelocationInfo(r, program);

	if (imports_only &&
	    (ri.bind_self || (ri.bind != BindType::Global && ri.bind != BindType::Weak))) {
		return;
	}

	[[maybe_unused]] bool patched        = false;
	bool                  stubbed_import = false;
	bool                  stubbed_func   = false;

	// KYTY_PROFILER_BLOCK("patch");

	if (ri.resolved) {
		patched = Common::VirtualMemory::PatchReplace(ri.vaddr, ri.value);
	} else {
		uint64_t value = 0;
		bool     weak  = (ri.bind == BindType::Weak || !program->fail_if_global_not_resolved);
		if (ri.type == SymbolType::Object && weak) {
			value = g_invalid_memory;
		} else if (ri.type == SymbolType::Func && jmprela_table && weak) {
			value          = RegisterStubbedImport(index, program, ri);
			stubbed_import = true;
			stubbed_func   = true;
		} else if (ri.type == SymbolType::Func && !jmprela_table && weak) {
			value        = RegisterStubbedImport(index, program, ri);
			stubbed_func = true;
		} else if (ri.type == SymbolType::NoType && weak) {
			value = RuntimeLinker::ReadFromElf(program, ri.vaddr) + ri.base_vaddr;
		}

		if (value != 0) {
			patched = Common::VirtualMemory::PatchReplace(ri.vaddr, value);
		} else {
			auto dbg_str = fmt::format("[{:016x}] <- {:016x}, {}, {}, {}, {}", ri.vaddr, ri.value,
			                           ri.name.c_str(), Common::EnumName(ri.type).c_str(),
			                           Common::EnumName(ri.bind).c_str(), ri.dbg_name.c_str());

			if (unresolved != nullptr) {
				unresolved->push_back(dbg_str);
			} else {
				EXIT("Can't resolve: %s\n", dbg_str.c_str());
			}

			if (ri.type == SymbolType::Object) {
				value = g_invalid_memory;
			} else if (ri.type == SymbolType::Func || ri.type == SymbolType::NoType) {
				value        = RegisterStubbedImport(index, program, ri);
				stubbed_func = true;
				if (jmprela_table) {
					stubbed_import = true;
				}
			}

			if (value != 0) {
				patched = Common::VirtualMemory::PatchReplace(ri.vaddr, value);
			}
		}
	}

	// KYTY_PROFILER_END_BLOCK;

	if (patched && stubbed_import) {
		const auto thunk = RegisterStubbedImport(index, program, ri);
		LOGF("Relocate: unresolved PLT import patched to stub [%u] [%016" PRIx64 "] <- %016" PRIx64
		     ", %s, %s, %s, %s\n",
		     index, ri.vaddr, thunk, ri.name.c_str(), Common::EnumName(ri.type).c_str(),
		     Common::EnumName(ri.bind).c_str(), Common::PathToString(program->file_name).c_str());
	} else if (patched && stubbed_func) {
		const auto thunk = RegisterStubbedImport(index, program, ri);
		LOGF("Relocate: unresolved non-PLT function patched to stub [%u] [%016" PRIx64
		     "] <- %016" PRIx64 ", %s, %s, %s, %s\n",
		     index, ri.vaddr, thunk, ri.name.c_str(), Common::EnumName(ri.type).c_str(),
		     Common::EnumName(ri.bind).c_str(), Common::PathToString(program->file_name).c_str());
	}

	if (program->dbg_print_reloc) {
		if (/* !dbg_str.ContainsStr("libc_") && */ patched && !ri.bind_self &&
		    (ri.bind == BindType::Global || ri.bind == BindType::Weak ||
		     ri.type == SymbolType::TlsModule)) {
			auto dbg_str = fmt::format("[{:016x}] <- {:016x}, {}, {}, {}, {}", ri.vaddr, ri.value,
			                           ri.name.c_str(), Common::EnumName(ri.type).c_str(),
			                           Common::EnumName(ri.bind).c_str(), ri.dbg_name.c_str());

			LOGF("Relocate: %s\n", dbg_str.c_str());
		}
	}
}

static void RelocateRecords(Elf64_Rela* records, uint64_t size, Program* program,
                            bool jmprela_table, bool imports_only,
                            std::vector<std::string>* unresolved) {
	KYTY_PROFILER_FUNCTION();

	uint32_t index = 0;
	for (auto* r = records;
	     reinterpret_cast<uint8_t*>(r) < reinterpret_cast<uint8_t*>(records) + size; r++, index++) {
		RelocateRecord(index, r, program, jmprela_table, imports_only, unresolved);
	}
}

__attribute__((naked)) static KYTY_SYSV_ABI void RelocateHandlerReturnStub() {
	asm volatile("addq $8, %rsp\n\t"
	             "retq\n");
}

static KYTY_SYSV_ABI uint64_t RelocateHandler(RelocateHandlerStack s) {
	auto*       stack     = s.stack;
	auto*       program   = reinterpret_cast<Program*>(stack[-1]);
	auto        rel_index = stack[0];
	std::string name      = "<unknown function>";

	if (program != nullptr && program->dynamic_info != nullptr &&
	    program->dynamic_info->jmprela_table != nullptr) {
		auto ri = GetRelocationInfo(program->dynamic_info->jmprela_table + rel_index, program);

		name = ri.name.c_str();
	}

	// Restore return address (for stack trace)
	stack[-1] = reinterpret_cast<uint64_t>(RelocateHandlerReturnStub);

	LOGF("=== Stubbed function, returning OK ===\n[%d]\t%s\n", Common::Thread::GetThreadIdUnique(),
	     name.c_str());
	return 0;
}

static KYTY_MS_ABI uint8_t* TlsMainGetAddr() {
	EXIT_IF(g_tls_main_program == nullptr);

	if (g_tls_cached_main_program == g_tls_main_program && g_tls_cached_main_tcb != nullptr) {
		return g_tls_cached_main_tcb;
	}

	g_tls_cached_main_program = g_tls_main_program;
	g_tls_cached_main_tcb =
	    RuntimeLinker::TlsGetAddr(g_tls_main_program) + g_tls_main_program->tls.tcb_offset;
	return g_tls_cached_main_tcb;
}

static void PatchProgram(Program* program, uint64_t address, uint64_t size) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->elf == nullptr);

	if (size >= 12) {
		// Replace guest stack-canary/errno stores through fs:[0x28] with nops.
		// Windows x64 cannot host guest FS directly, and an unpatched shared-library access faults
		// at address 0x28.
		const uint8_t fs_store_pattern[8] = {0x64, 0xc7, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00};
		auto*         start_ptr           = reinterpret_cast<uint8_t*>(address);
		auto*         end_ptr             = start_ptr + size - 12;

		for (auto* ptr = start_ptr; ptr <= end_ptr; ptr++) {
			if (memcmp(ptr, fs_store_pattern, sizeof(fs_store_pattern)) == 0) {
				LOGF("Patch fs:[0x28] store at addr: [%016" PRIx64 "]\n",
				     reinterpret_cast<uint64_t>(ptr));
				if (ptr + 16 < start_ptr + size && ptr[12] == 0xcd && ptr[13] == 0x45 &&
				    ptr[14] == 0x90 && ptr[15] == 0x0f && ptr[16] == 0x0b) {
					ptr[0] = 0x5d; // pop rbp
					ptr[1] = 0xc3; // ret
					std::memset(ptr + 2, 0x90, 15);
				} else {
					std::memset(ptr, 0x90, 12);
				}
			}
		}
	}

	if (!program->elf->IsShared() && program->tls.handler_vaddr != 0) {
		// Replace:
		//   66 66 66
		//   mov <reg>, qword ptr fs:[0x00]
		// with:
		//   call <handler>
		//   mov <reg>,rax
		//   nop ...
		const uint8_t tls_pattern[5] = {0x64, 0x48, 0x8B, 0x00, 0x25};

		EXIT_IF(Jit::Call9::GetSize() != 9);

		auto* start_ptr = reinterpret_cast<uint8_t*>(address);
		auto* end_ptr   = start_ptr + size - Jit::Call9::GetSize();

		for (auto* ptr = start_ptr; ptr <= end_ptr; ptr++) {
			auto*  inst_ptr     = ptr;
			size_t prefix_count = 0;
			while (prefix_count < 3 && inst_ptr < start_ptr + size && *inst_ptr == 0x66) {
				inst_ptr++;
				prefix_count++;
			}

			const size_t inst_size = prefix_count + Jit::Call9::GetSize();
			if (inst_ptr + Jit::Call9::GetSize() > start_ptr + size) {
				break;
			}

			const uint8_t modrm = inst_ptr[3];
			if (memcmp(inst_ptr, tls_pattern, 3) == 0 && (modrm & 0xc7u) == 0x04u &&
			    inst_ptr[4] == tls_pattern[4] &&
			    *reinterpret_cast<const uint32_t*>(inst_ptr + 5) == 0) {
				LOGF("Patch tls at addr: [%016" PRIx64 "]\n", reinterpret_cast<uint64_t>(ptr));

				const auto reg = (modrm >> 3u) & 7u;
				EXIT_NOT_IMPLEMENTED(reg == 4u);

				auto* code = new (ptr) Jit::Call9;
				code->SetFunc(reg == 0
				                  ? program->tls.handler_vaddr
				                  : program->tls.handler_vaddr + Jit::TlsRegStub::GetOffset(reg));
				if (inst_size > Jit::Call9::GetSize()) {
					std::memset(ptr + Jit::Call9::GetSize(), 0x90,
					            inst_size - Jit::Call9::GetSize());
				}
				ptr += inst_size - 1;
			}
		}
	}
}

uint64_t RuntimeLinker::GetEntry() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		if (p->elf != nullptr && !p->elf->IsShared()) {
			return p->elf->GetEntry() + p->base_vaddr;
		}
	}
	return 0;
}

uint64_t RuntimeLinker::GetProcParam() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		if (p->elf != nullptr && !p->elf->IsShared()) {
			return p->proc_param_vaddr;
		}
	}
	return 0;
}

void RuntimeLinker::DbgDump(const std::string& folder) {
	KYTY_PROFILER_FUNCTION();

	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		auto folder_str = Common::FixDirectorySlash(folder);
		folder_str += Common::FilenameWithoutDirectory(Common::PathToGenericString(p->file_name));

		EXIT_IF(p->elf == nullptr);

		p->elf->DbgDump(folder_str);

		if (p->dynamic_info != nullptr) {
			EXIT_NOT_IMPLEMENTED(p->dynamic_info->symbol_table_entry_size != 0 &&
			                     p->dynamic_info->symbol_table_entry_size != sizeof(Elf64_Sym));
			EXIT_NOT_IMPLEMENTED(p->dynamic_info->rela_table_entry_size != 0 &&
			                     p->dynamic_info->rela_table_entry_size != sizeof(Elf64_Rela));
			// EXIT_NOT_IMPLEMENTED(p->dynamic_info->jmprela_table == nullptr);
			// EXIT_NOT_IMPLEMENTED(p->dynamic_info->rela_table == nullptr);
			// EXIT_NOT_IMPLEMENTED(p->dynamic_info->symbol_table == nullptr);

			if (p->dynamic_info->symbol_table != nullptr) {
				DbgDumpSymbols(folder_str, p->dynamic_info->symbol_table,
				               p->dynamic_info->symbol_table_total_size,
				               p->dynamic_info->str_table);
			}
			if (p->dynamic_info->jmprela_table != nullptr) {
				DbgDumpRela(folder_str, p->dynamic_info->jmprela_table,
				            p->dynamic_info->jmprela_table_size, p->dynamic_info->str_table,
				            "jmprela_table.txt");
			}
			if (p->dynamic_info->rela_table != nullptr) {
				DbgDumpRela(folder_str, p->dynamic_info->rela_table,
				            p->dynamic_info->rela_table_total_size, p->dynamic_info->str_table,
				            "rela_table.txt");
			}
		}

		if (p->export_symbols != nullptr) {
			p->export_symbols->DbgDump(folder_str, "export_symbols.txt");
		}
		if (p->import_symbols != nullptr) {
			p->import_symbols->DbgDump(folder_str, "import_symbols.txt");
		}
	}
}

void RuntimeLinker::RelocateAll() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		Relocate(p);
	}

	m_relocated = true;
}

void RuntimeLinker::RelocateProgram(Program* program) {
	Common::LockGuard lock(m_mutex);

	EXIT_IF(program == nullptr);
	EXIT_IF(std::find(m_programs.begin(), m_programs.end(), program) == m_programs.end());

	Relocate(program);
}

void RuntimeLinker::UnloadProgram(Program* program) {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	if (auto it = std::find(m_programs.begin(), m_programs.end(), program);
	    it != m_programs.end()) {
		DeleteProgram(*it);
		m_programs.erase(it);
	} else {
		EXIT("program not found");
	}

	if (m_relocated) {
		RelocateAll();
	}
}

RuntimeLinker::RuntimeLinker(): m_symbols(std::make_unique<SymbolDatabase>()) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RuntimeLinker::~RuntimeLinker() {
	Clear();
}

Program* RuntimeLinker::LoadProgram(const std::filesystem::path& elf_name) {
	KYTY_PROFILER_FUNCTION();

	Common::LockGuard lock(m_mutex);

	static int32_t id_seq = 0;

	LOGF("Loading: %s\n", Common::PathToString(elf_name).c_str());

	auto  program_owner = std::make_unique<Program>();
	auto* program       = program_owner.get();

	program->rt        = this;
	program->file_name = elf_name;
	program->unique_id = ++id_seq;

	program->elf = std::make_unique<Elf64>();
	program->elf->Open(elf_name);

	if (program->elf->IsValid()) {
		LoadProgramToMemory(program);
		ParseProgramDynamicInfo(program);
		CreateSymbolDatabase(program);
	} else {
		EXIT("elf is not valid: %s\n", Common::PathToString(elf_name).c_str());
	}

	m_programs.push_back(program_owner.release());

	if (!program->elf->IsShared()) {
		program->fail_if_global_not_resolved = false;
		Libs::LibKernel::SetProgName(elf_name.filename().string());
	}

	if (Common::EndsWith(Common::ToLower(Common::DirectoryWithoutFilename(
	                         Common::PathToGenericString(elf_name))),
	                     "_module/")) {
		program->fail_if_global_not_resolved = false;
	}

	return program;
}

void RuntimeLinker::SaveMainProgram(const std::filesystem::path& elf_name) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (const auto* p: m_programs) {
		EXIT_IF(p->elf == nullptr);

		if (!p->elf->IsShared()) {
			p->elf->Save(elf_name);
			break;
		}
	}
}

void RuntimeLinker::SaveProgram(Program* program, const std::filesystem::path& elf_name) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	if (auto it = std::find(m_programs.begin(), m_programs.end(), program);
	    it != m_programs.end()) {
		EXIT_IF((*it)->elf == nullptr);

		(*it)->elf->Save(elf_name);
	} else {
		EXIT("program not found");
	}
}

static void SimulateGnmDriverDirectMemoryInit() {
	constexpr size_t k_gnm_size     = 0x10000;
	constexpr size_t k_gnm_align    = 0x10000;
	constexpr int    k_memory_type  = 3;
	constexpr int    k_prot         = 0x13;

	int64_t phys_addr = 0;
	const auto direct_size =
	    static_cast<int64_t>(Libs::LibKernel::Memory::KernelGetDirectMemorySize());
	const auto alloc_result = Libs::LibKernel::Memory::KernelAllocateDirectMemory(
	    0, direct_size, k_gnm_size, k_gnm_align, k_memory_type, &phys_addr);
	if (alloc_result != 0) {
		LOGF("SimulateGnmDriverDirectMemoryInit: allocate failed (%d)\n", alloc_result);
		return;
	}

	void* vaddr = reinterpret_cast<void*>(0xfe0000000ull);
	const auto map_result = Libs::LibKernel::Memory::KernelMapNamedDirectMemory(
	    &vaddr, k_gnm_size, k_prot, 0, phys_addr, k_gnm_align, "SceGnmDriver");
	LOGF("SimulateGnmDriverDirectMemoryInit: phys=0x%016" PRIx64 " vaddr=0x%016" PRIx64
	     " result=%d\n",
	     static_cast<uint64_t>(phys_addr), reinterpret_cast<uint64_t>(vaddr), map_result);
}

void RuntimeLinker::Execute() {
	KYTY_PROFILER_THREAD("Thread_Main");

	Libs::LibKernel::PthreadInitSelfForMainThread();

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Guest code has no Windows stack probes and may jump over the guard page. Module
	// initializers execute on the host stack too, so grow it before calling any guest code.
	size_t expanded_size = 0;
	while (expanded_size < static_cast<size_t>(768) * 1024) {
		sys_dbg_stack_info_t stack {};
		SysStackUsage(stack);
		*reinterpret_cast<uint32_t*>(stack.guard_addr) = 0;
		expanded_size += stack.guard_size;
	}
#endif

	PreloadAdjacentPrograms();
	RelocateAll();
	for (auto* program: m_programs) {
		if (program->elf != nullptr && !program->elf->IsShared()) {
			SynthesizeProgramProcParam(program);
		}
	}
	StartAllModules();
	EnsureApplicationHeapApi();
	SimulateGnmDriverDirectMemoryInit();

	LOGF_COLOR(Log::Color::BrightYellow, "---\n--- Execute: %s\n---\n", "Main");

	if (auto entry = GetEntry(); entry != 0) {
		auto* main_stack_top = Libs::LibKernel::PthreadCreateMainGuestStack();
		auto* params = reinterpret_cast<EntryParams*>(
		    (reinterpret_cast<uintptr_t>(main_stack_top) - 0x100u) & ~static_cast<uintptr_t>(0x0f));
		std::memset(params, 0, sizeof(EntryParams));
		params->argc    = 1;
		params->argv[0] = "KytyEmu";

		LOGF("stack_addr = %" PRIx64 "\n", reinterpret_cast<uint64_t>(params));

		RunEntry(entry, params, ProgramExitHandler,
		         reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(params) - 0x1000u));
	}
}

void RuntimeLinker::Clear() {
	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());

	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		DeleteProgram(p);
	}
	m_programs.clear();
	m_symbols.reset();
	m_relocated = false;
}

void RuntimeLinker::Resolve(const std::string& name, SymbolType type, Program* program,
                            SymbolRecord* out_info, bool* bind_self) {
	KYTY_PROFILER_FUNCTION();

	Common::LockGuard lock(m_mutex);

	EXIT_IF(out_info == nullptr);

	auto ids = Common::Split(name, '#');

	if (bind_self != nullptr) {
		*bind_self = false;
	}

	if (ids.size() == 3) {
		const LibraryId* l = FindLibrary(*program, ids.at(1));
		const ModuleId*  m = FindModule(*program, ids.at(2));

		auto resolve_by_nid = [this, type](const std::string& nid, SymbolRecord* out) -> bool {
			EXIT_IF(out == nullptr);

			if (m_symbols != nullptr) {
				if (const auto* rec = m_symbols->FindByNid(nid, type); rec != nullptr) {
					*out = *rec;
					return true;
				}
			}

			for (auto* p: m_programs) {
				if (p != nullptr && p->export_symbols != nullptr) {
					if (const auto* rec = p->export_symbols->FindByNid(nid, type); rec != nullptr) {
						*out = *rec;
						return true;
					}
				}
			}

			return false;
		};

		if (l != nullptr && m != nullptr) {
			SymbolResolve sr {};
			sr.name                 = ids.at(0);
			sr.library              = l->name;
			sr.library_version      = l->version;
			sr.module               = m->name;
			sr.module_version_major = m->version_major;
			sr.module_version_minor = m->version_minor;
			sr.type                 = type;

			const SymbolRecord* rec = nullptr;

			if (m_symbols != nullptr) {
				rec = m_symbols->Find(sr);
			}

			if (rec == nullptr) {
				if (auto* p = FindProgram(*m, *l); p != nullptr && p->export_symbols != nullptr) {
					rec = p->export_symbols->Find(sr);
					if (bind_self != nullptr) {
						*bind_self = (p == program);
					}
				}
			}

			if (rec == nullptr) {
				if (resolve_by_nid(sr.name, out_info)) {
					LOGF("PS5 NID fallback: %s -> %s\n", sr.name.c_str(), out_info->name.c_str());
					return;
				}
			}

			if (rec != nullptr) {
				//*out_vaddr = rec->vaddr;
				*out_info = *rec;
			} else {
				out_info->vaddr    = 0;
				out_info->name     = SymbolDatabase::GenerateName(sr);
				out_info->dbg_name = "";
			}
		} else {
			if (resolve_by_nid(ids.at(0), out_info)) {
				LOGF("PS5 NID fallback: %s -> %s (missing lib/module metadata)\n",
				     ids.at(0).c_str(), out_info->name.c_str());
				return;
			}

			EXIT("l == nullptr || m == nullptr");
		}
	} else {
		out_info->vaddr    = 0;
		out_info->name     = name;
		out_info->dbg_name = "";
	}
}

bool RuntimeLinker::ResolveLoadedSymbolByNid(const std::string& nid, SymbolType type,
                                             SymbolRecord* out_info) {
	KYTY_PROFILER_FUNCTION();

	Common::LockGuard lock(m_mutex);

	EXIT_IF(out_info == nullptr);

	for (auto* p: m_programs) {
		if (p != nullptr && p->export_symbols != nullptr) {
			if (const auto* rec = p->export_symbols->FindByNid(nid, type); rec != nullptr) {
				*out_info = *rec;
				return true;
			}
		}
	}

	if (m_symbols != nullptr) {
		if (const auto* rec = m_symbols->FindByNid(nid, type); rec != nullptr) {
			*out_info = *rec;
			return true;
		}
	}

	return false;
}

uint64_t RuntimeLinker::ReadFromElf(Program* program, uint64_t vaddr) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->base_vaddr == 0 || program->base_size == 0);
	EXIT_IF(program->elf == nullptr);

	uint64_t ret = 0;

	const auto* ehdr = program->elf->GetEhdr();
	const auto* phdr = program->elf->GetPhdr();

	EXIT_IF(phdr == nullptr || ehdr == nullptr);

	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_memsz != 0 && (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
			uint64_t segment_addr      = phdr[i].p_vaddr + program->base_vaddr;
			uint64_t segment_file_size = phdr[i].p_filesz;

			if (vaddr >= segment_addr && vaddr < segment_addr + segment_file_size) {
				program->elf->LoadSegment(reinterpret_cast<uint64_t>(&ret),
				                          phdr[i].p_offset + vaddr - segment_addr, sizeof(ret));
				break;
			}
		}
	}

	return ret;
}

Program* RuntimeLinker::FindProgramById(int32_t id) {
	Common::LockGuard lock(m_mutex);

	// Id 0 is reserved for main program
	if (id == 0 && !m_programs.empty()) {
		return m_programs.front();
	}

	for (auto* p: m_programs) {
		if (p->unique_id == id) {
			return p;
		}
	}

	return nullptr;
}

Program* RuntimeLinker::FindProgramByFileName(const std::filesystem::path& elf_name) {
	Common::LockGuard lock(m_mutex);

	auto fixed_name = Common::FixFilenameSlash(Common::PathToGenericString(elf_name));
	for (auto* p: m_programs) {
		if (Common::EqualNoCase(Common::FixFilenameSlash(Common::PathToGenericString(p->file_name)),
		                        fixed_name)) {
			return p;
		}
	}

	return nullptr;
}

Program* RuntimeLinker::FindProgramByAddr(uint64_t vaddr) {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		const auto* ehdr = p->elf->GetEhdr();
		const auto* phdr = p->elf->GetPhdr();

		EXIT_IF(phdr == nullptr || ehdr == nullptr);

		for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
			if (phdr[i].p_memsz != 0 &&
			    (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
				uint64_t segment_addr = phdr[i].p_vaddr + p->base_vaddr;
				uint64_t segment_size = GetAlignedSize(phdr + i);

				if (vaddr >= segment_addr && vaddr < segment_addr + segment_size) {
					return p;
				}
			}
		}
	}

	return nullptr;
}

void RuntimeLinker::StackTrace(uint64_t frame_ptr) {
	void* stack[20];
	int   depth = 20;

	SysStackWalkX86(frame_ptr, stack, &depth);

	LOGF("Stack trace [thread = %d]:\n", Common::Thread::GetThreadIdUnique());

	for (int i = 0; i < depth; i++) {
		auto  vaddr = reinterpret_cast<uint64_t>(stack[i]);
		auto* p     = FindProgramByAddr(vaddr);
		LOGF("[%d] %016" PRIx64 ", off=%016" PRIx64 ", %s\n", i, vaddr,
		     (p == nullptr ? 0 : vaddr - p->base_vaddr),
		     (p == nullptr
		          ? "???"
		          : Common::FilenameWithoutDirectory(Common::PathToGenericString(p->file_name))
		                .c_str()));
	}
}

static std::string GetProgramModuleName(const Program* program) {
	EXIT_IF(program == nullptr);

	if (program->dynamic_info != nullptr && program->dynamic_info->so_name != nullptr &&
	    program->dynamic_info->so_name[0] != '\0') {
		return std::string(program->dynamic_info->so_name);
	}

	return Common::FilenameWithoutDirectory(Common::PathToGenericString(program->file_name));
}

static bool ModuleStartDependenciesSatisfied(const Program*               program,
                                             const std::vector<Program*>& programs,
                                             const std::vector<Program*>& started) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->dynamic_info == nullptr);

	for (const auto* needed: program->dynamic_info->needed) {
		if (needed == nullptr || needed[0] == '\0') {
			continue;
		}

		const auto needed_name = std::string(needed);

		for (auto* dependency: programs) {
			if (dependency == nullptr || dependency == program || dependency->elf == nullptr ||
			    !dependency->elf->IsShared()) {
				continue;
			}

			const auto dependency_name = GetProgramModuleName(dependency);
			if (Common::EqualNoCase(dependency_name, needed_name) ||
			    Common::EqualNoCase(Common::FilenameWithoutDirectory(
			                            Common::PathToGenericString(dependency->file_name)),
			                        needed_name)) {
				if (std::find(started.begin(), started.end(), dependency) == started.end()) {
					return false;
				}
				break;
			}
		}
	}

	return true;
}

void RuntimeLinker::StartAllModules() {
	Common::LockGuard lock(m_mutex);

	std::vector<Program*> started;

	for (;;) {
		bool progressed = false;

		for (auto* p: m_programs) {
			if (p->elf->IsShared() && p->dynamic_info->init_vaddr != 0 &&
			    std::find(started.begin(), started.end(), p) == started.end() &&
			    ModuleStartDependenciesSatisfied(p, m_programs, started)) {
				StartModule(p, 0, nullptr, nullptr);
				started.push_back(p);
				progressed = true;
			}
		}

		if (!progressed) {
			break;
		}
	}

	for (auto* p: m_programs) {
		if (p->elf->IsShared() && p->dynamic_info->init_vaddr != 0 &&
		    std::find(started.begin(), started.end(), p) == started.end()) {
			StartModule(p, 0, nullptr, nullptr);
			started.push_back(p);
		}
	}
}

void RuntimeLinker::StopAllModules() {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		if (p->elf->IsShared() && p->dynamic_info->fini_vaddr != 0) {
			StopModule(p, 0, nullptr, nullptr);
		}
	}
}

static bool IsAdjacentModuleFile(const std::string& name) {
	auto lower = Common::ToLower(name);
	return Common::EndsWith(lower, ".prx") || Common::EndsWith(lower, ".sprx");
}

static bool SkipAdjacentModuleFile(const std::string& name) {
	auto lower = Common::ToLower(name);
	return lower == "eboot.bin" || lower == "libkernel.prx" || lower == "libkernel_sys.prx";
}

void RuntimeLinker::PreloadAdjacentPrograms() {
	if (m_programs.empty()) {
		return;
	}

	std::vector<std::filesystem::path> module_paths;

	auto is_loaded = [this](const std::filesystem::path& path) {
		auto fixed_path = Common::FixFilenameSlash(Common::PathToGenericString(path));
		for (auto* program: m_programs) {
			if (Common::EqualNoCase(
			        Common::FixFilenameSlash(Common::PathToGenericString(program->file_name)),
			        fixed_path)) {
				return true;
			}
		}
		return false;
	};

	auto add_path = [&module_paths, &is_loaded](const std::filesystem::path& path) {
		if (is_loaded(path)) {
			return;
		}
		for (const auto& p: module_paths) {
			if (Common::EqualNoCase(Common::PathToGenericString(p),
			                        Common::PathToGenericString(path))) {
				return;
			}
		}
		module_paths.push_back(path);
	};

	auto add_dir = [&add_path](const std::filesystem::path& dir) {
		if (!Common::File::IsDirectoryExisting(dir)) {
			return;
		}
		for (const auto& entry: Common::File::GetDirEntries(dir)) {
			if (entry.is_file && IsAdjacentModuleFile(entry.name) &&
			    !SkipAdjacentModuleFile(entry.name)) {
				add_path(dir / entry.name);
			}
		}
	};

	auto root = m_programs.at(0)->file_name.parent_path();
	if (root.empty()) {
		return;
	}

	add_dir(root);
	add_dir(root / "sce_module");
	add_dir(root / "sce_modules");

	for (const auto& path: module_paths) {
		auto* program                        = LoadProgram(path);
		program->fail_if_global_not_resolved = false;
	}
}

int RuntimeLinker::StartModule(Program* program, size_t args, const void* argp,
                               module_func_t func) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->dynamic_info == nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(!program->elf->IsShared());

	EXIT_IF(std::find(m_programs.begin(), m_programs.end(), program) == m_programs.end());

	LOGF_COLOR(Log::Color::BrightYellow, "---\n--- Start module: %s\n---\n",
	           Common::PathToString(program->file_name).c_str());

	return reinterpret_cast<module_ini_fini_func_t>(program->dynamic_info->init_vaddr +
	                                                program->base_vaddr)(args, argp, func);
}

int RuntimeLinker::StopModule(Program* program, size_t args, const void* argp, module_func_t func) {
	EXIT_IF(program == nullptr);
	EXIT_IF(program->dynamic_info == nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(!program->elf->IsShared());

	EXIT_IF(std::find(m_programs.begin(), m_programs.end(), program) == m_programs.end());

	LOGF_COLOR(Log::Color::BrightYellow, "---\n--- Stop module: %s\n---\n",
	           Common::PathToString(program->file_name).c_str());

	int result = reinterpret_cast<module_ini_fini_func_t>(program->dynamic_info->fini_vaddr +
	                                                      program->base_vaddr)(args, argp, func);

	Libs::LibKernel::PthreadDeleteStaticObjects(program);

	return result;
}

uint8_t* RuntimeLinker::TlsGetAddr(Program* program) {
	EXIT_IF(program == nullptr);

	Common::LockGuard lock(program->tls.mutex);

	auto& tls = program->tls.tlss[Common::Thread::GetThreadIdUnique()];

	if (tls.ptr == nullptr) {
		constexpr uint64_t TCB_SIZE  = 0x40;
		constexpr uint64_t TCB_ALIGN = 0x20;

		const auto tcb_offset =
		    program->tls.tcb_offset != 0 ? program->tls.tcb_offset : program->tls.image_size;
		const auto alloc_size = AlignUp(tcb_offset, TCB_ALIGN) + TCB_SIZE;

		// Always allocate TLS with host VirtualMemory. Using the guest application heap here
		// forced CleanupThread to call guest free() from a host pthread teardown context, which
		// is unsafe; the previous delete[] fallback caused STATUS_HEAP_CORRUPTION (0xc0000374).
		void* allocated = reinterpret_cast<void*>(
		    Common::VirtualMemory::Alloc(0, alloc_size, Common::VirtualMemory::Mode::ReadWrite));
		tls.vm_alloc  = true;
		tls.free_func = nullptr;
		tls.ptr       = reinterpret_cast<uint8_t*>(allocated);

		EXIT_IF(tls.ptr == nullptr);

		std::memset(tls.ptr, 0, alloc_size);

		if (!program->tls.init_image.empty()) {
			std::memcpy(tls.ptr, program->tls.init_image.data(), program->tls.init_image.size());
		} else {
			std::memcpy(tls.ptr, reinterpret_cast<void*>(program->tls.image_vaddr),
			            program->tls.init_size);
		}

		auto* tcb = reinterpret_cast<uint64_t*>(tls.ptr + tcb_offset);
		tcb[0]    = reinterpret_cast<uint64_t>(tcb);
	}

	return tls.ptr;
}

void RuntimeLinker::DeleteTls(Program* program, int thread_id) {
	EXIT_IF(program == nullptr);

	if (thread_id == Common::Thread::GetThreadIdUnique() && g_tls_cached_main_program == program) {
		g_tls_cached_main_program = nullptr;
		g_tls_cached_main_tcb     = nullptr;
	}

	Common::LockGuard lock(program->tls.mutex);

	if (auto it = program->tls.tlss.find(thread_id); it != program->tls.tlss.end()) {
		FreeTlsBlock(&it->second);
		program->tls.tlss.erase(it);
	}
}

static uint64_t CalcBaseSize(const Elf64_Ehdr* ehdr, const Elf64_Phdr* phdr) {
	uint64_t base_size = 0;
	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_memsz != 0 && (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
			uint64_t last_addr = phdr[i].p_vaddr + GetAlignedSize(phdr + i);
			if (last_addr > base_size) {
				base_size = last_addr;
			}
		}
	}
	return base_size;
}

namespace {

constexpr uint64_t FLEXIBLE_MEMORY_BASE        = 64ull * 1024ull * 1024ull;
constexpr uint32_t ORBIS_PROC_PARAM_MAGIC       = 0x13bccf52u;
constexpr uint32_t ORBIS_PROC_PARAM_FILE_MAGIC  = 0x4942524fu; // "ORBI"
constexpr uint64_t ORBIS_PROC_PARAM_STRUCT_SIZE = 0x60;
constexpr uint64_t ORBIS_LIBC_PARAM_STRUCT_SIZE = 0x50;
constexpr uint64_t PROC_PARAM_SYNTH_SIZE        = 0x120;

#pragma pack(push, 1)
struct GuestKernelMemParam {
	uint64_t size;
	uint64_t extended_page_table;
	uint64_t flexible_memory_size;
	uint64_t extended_memory_1;
	uint64_t extended_gpu_page_table;
	uint64_t extended_memory_2;
	uint64_t extended_cpu_page_table;
};
static_assert(sizeof(GuestKernelMemParam) == 0x38);

struct GuestProcParam {
	uint64_t size;
	uint32_t magic;
	uint32_t entry_count;
	uint64_t sdk_version;
	uint64_t process_name;
	uint64_t main_thread_name;
	uint64_t main_thread_prio;
	uint64_t main_thread_stack_size;
	uint64_t libc_param;
	uint64_t mem_param;
	uint64_t fs_param;
	uint64_t process_preload_enable;
	uint64_t unknown1;
};
static_assert(sizeof(GuestProcParam) == 0x60);

struct GuestLibcParam {
	uint64_t size;
	uint64_t unknown_08;
	uint64_t heap_size;
	uint64_t unknown_18;
	uint64_t unknown_20;
	uint64_t unknown_28;
	uint64_t unknown_30;
	uint64_t unknown_38;
	uint64_t unknown_40;
	uint64_t need_sce_libc;
};
static_assert(sizeof(GuestLibcParam) == 0x50);
#pragma pack(pop)

static bool IsValidProcParamMagic(uint32_t magic) {
	return magic == ORBIS_PROC_PARAM_FILE_MAGIC || magic == ORBIS_PROC_PARAM_MAGIC;
}

static bool GuestAddrIsReadable(uint64_t addr, uint64_t size, const Program* program) {
	if (addr == 0 || size == 0 || size > UINT64_MAX - addr) {
		return false;
	}

	if (program != nullptr && addr >= program->base_vaddr &&
	    addr + size <= program->base_vaddr + program->mapped_size) {
		return true;
	}

	auto* loaded = Common::Singleton<RuntimeLinker>::Instance()->FindProgramByAddr(addr);
	return loaded != nullptr && addr + size <= loaded->base_vaddr + loaded->mapped_size;
}

static bool MemParamHasValidExtendedMemory(const GuestKernelMemParam* mem) {
	return mem != nullptr && mem->extended_memory_1 != 0 && mem->extended_memory_2 != 0;
}

static void LogProcParamPostRelocate(const Program* program) {
	if (program == nullptr || program->proc_param_vaddr == 0) {
		return;
	}

	const auto* proc = reinterpret_cast<const GuestProcParam*>(program->proc_param_vaddr);
	LOGF("ProcParam post-reloc: proc=0x%016" PRIx64 " magic=0x%08" PRIx32
	     " mem_param=0x%016" PRIx64 " libc_param=0x%016" PRIx64 "\n",
	     program->proc_param_vaddr, proc->magic, proc->mem_param, proc->libc_param);

	if (proc->mem_param == 0) {
		return;
	}

	const auto* dump = reinterpret_cast<const uint8_t*>(proc->mem_param);
	for (size_t row = 0; row < 0x40; row += 16) {
		LOGF("  mem_param+%04zx: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
		     row, dump[row + 0], dump[row + 1], dump[row + 2], dump[row + 3], dump[row + 4],
		     dump[row + 5], dump[row + 6], dump[row + 7], dump[row + 8], dump[row + 9],
		     dump[row + 10], dump[row + 11], dump[row + 12], dump[row + 13], dump[row + 14],
		     dump[row + 15]);
	}
}

static void LogProcParamStructures(const char* tag, const GuestProcParam* proc) {
	if (proc == nullptr) {
		return;
	}

	const auto* raw = reinterpret_cast<const uint8_t*>(proc);
	LOGF("ProcParam %s: size=0x%016" PRIx64 " magic=0x%08" PRIx32 " entry_count=%" PRIu32
	     " sdk=0x%016" PRIx64 " libc_param=0x%016" PRIx64 " mem_param=0x%016" PRIx64
	     " stack_size=0x%016" PRIx64 "\n",
	     tag, proc->size, proc->magic, proc->entry_count, proc->sdk_version, proc->libc_param,
	     proc->mem_param, proc->main_thread_stack_size);
	for (size_t row = 0; row < 0x60; row += 16) {
		LOGF("  proc+%04zx: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
		     row, raw[row + 0], raw[row + 1], raw[row + 2], raw[row + 3], raw[row + 4],
		     raw[row + 5], raw[row + 6], raw[row + 7], raw[row + 8], raw[row + 9], raw[row + 10],
		     raw[row + 11], raw[row + 12], raw[row + 13], raw[row + 14], raw[row + 15]);
	}

	if (proc->libc_param != 0) {
		const auto* libc = reinterpret_cast<const GuestLibcParam*>(proc->libc_param);
		const auto* dump = reinterpret_cast<const uint8_t*>(proc->libc_param);
		LOGF("ProcParam %s libc: size=0x%016" PRIx64 " heap_size_ptr=0x%016" PRIx64
		     " need_sce_libc=0x%016" PRIx64 "\n",
		     tag, libc->size, libc->heap_size, libc->need_sce_libc);
		for (size_t row = 0; row < 0x50; row += 16) {
			LOGF("  libc+%04zx: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
			     row, dump[row + 0], dump[row + 1], dump[row + 2], dump[row + 3], dump[row + 4],
			     dump[row + 5], dump[row + 6], dump[row + 7], dump[row + 8], dump[row + 9],
			     dump[row + 10], dump[row + 11], dump[row + 12], dump[row + 13], dump[row + 14],
			     dump[row + 15]);
		}
		if (libc->heap_size != 0) {
			LOGF("ProcParam %s *heap_size_ptr=0x%016" PRIx64 "\n", tag,
			     *reinterpret_cast<const uint64_t*>(libc->heap_size));
		}
	}

	if (proc->mem_param != 0) {
		const auto* mem  = reinterpret_cast<const GuestKernelMemParam*>(proc->mem_param);
		const auto* dump = reinterpret_cast<const uint8_t*>(proc->mem_param);
		LOGF("ProcParam %s mem: size=0x%016" PRIx64 " flex_ptr=0x%016" PRIx64
		     " ext1=0x%016" PRIx64 " ext2=0x%016" PRIx64 "\n",
		     tag, mem->size, mem->flexible_memory_size, mem->extended_memory_1,
		     mem->extended_memory_2);
		for (size_t row = 0; row < 0x40; row += 16) {
			LOGF("  mem+%04zx: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
			     row, dump[row + 0], dump[row + 1], dump[row + 2], dump[row + 3], dump[row + 4],
			     dump[row + 5], dump[row + 6], dump[row + 7], dump[row + 8], dump[row + 9],
			     dump[row + 10], dump[row + 11], dump[row + 12], dump[row + 13], dump[row + 14],
			     dump[row + 15]);
		}
		if (mem->flexible_memory_size != 0) {
			LOGF("ProcParam %s *flex_ptr=0x%016" PRIx64 "\n", tag,
			     *reinterpret_cast<const uint64_t*>(mem->flexible_memory_size));
		}
	}
}

static bool IsSystemModuleFilename(const std::string& name) {
	const auto lower = Common::ToLower(name);
	return Common::EndsWith(lower, "libc.prx") || Common::EndsWith(lower, "libkernel.prx") ||
	       Common::EndsWith(lower, "libkernel_sys.prx");
}

static void RegisterLoadedProgramMemory(Program* program) {
	const auto module_name = Common::PathToString(program->file_name.filename());
	Libs::LibKernel::Memory::RegisterProgramFlexibleQuota(program->base_size_aligned,
	                                                     module_name.c_str());
}

static bool MakeGuestRegionWritable(uint64_t vaddr, uint64_t size);

static void WireGuestMemParamExtended(GuestKernelMemParam* mem, uint64_t proc_vaddr,
                                      uint64_t synth_size) {
	if (mem == nullptr ||
	    mem->size < offsetof(GuestKernelMemParam, extended_memory_2) + sizeof(uint64_t)) {
		return;
	}

	const auto ext1_vaddr = proc_vaddr + 0x100;
	const auto ext2_vaddr = proc_vaddr + 0x108;
	if (ext2_vaddr + sizeof(uint32_t) > proc_vaddr + synth_size) {
		LOGF("WireGuestMemParamExtended: proc_param segment too small\n");
		return;
	}

	auto* ext1 = reinterpret_cast<uint32_t*>(ext1_vaddr);
	auto* ext2 = reinterpret_cast<uint32_t*>(ext2_vaddr);
	*ext1      = 1;
	*ext2      = 1;
	mem->extended_memory_1 = ext1_vaddr;
	mem->extended_memory_2 = ext2_vaddr;

	LOGF("WireGuestMemParamExtended: ext1=0x%016" PRIx64 " ext2=0x%016" PRIx64 "\n", ext1_vaddr,
	     ext2_vaddr);
}

static bool PatchFlexibleScalarInProcParam(GuestProcParam* proc, uint64_t flex_scalar,
                                           uint64_t proc_vaddr, uint64_t synth_size,
                                           const Program* program) {
	if (proc == nullptr || proc->mem_param == 0) {
		return false;
	}

	if (!GuestAddrIsReadable(proc->mem_param, sizeof(GuestKernelMemParam), program)) {
		LOGF("PatchFlexibleScalarInProcParam: mem_param 0x%016" PRIx64 " not readable\n",
		     proc->mem_param);
		return false;
	}

	auto* mem = reinterpret_cast<GuestKernelMemParam*>(proc->mem_param);
	if (mem->size < offsetof(GuestKernelMemParam, flexible_memory_size) + sizeof(uint64_t)) {
		return false;
	}

	uint64_t flex_ptr_addr = mem->flexible_memory_size;
	if (flex_ptr_addr == 0) {
		// mem_param+sizeof(mem) is fs_param on this title — never store the scalar there.
		// Use padding inside the writable proc_param synth region instead.
		flex_ptr_addr = AlignUp(proc_vaddr + sizeof(GuestProcParam), 8ull);
		if (flex_ptr_addr + sizeof(uint64_t) > proc_vaddr + synth_size) {
			LOGF("PatchFlexibleScalarInProcParam: no room for flex scalar in proc_param synth\n");
			return false;
		}
		if (!MakeGuestRegionWritable(proc_vaddr, synth_size)) {
			LOGF("PatchFlexibleScalarInProcParam: failed to make proc_param writable\n");
			return false;
		}
		if (!MakeGuestRegionWritable(proc->mem_param, sizeof(GuestKernelMemParam))) {
			LOGF("PatchFlexibleScalarInProcParam: failed to make mem_param writable at 0x%016"
			     PRIx64 "\n",
			     proc->mem_param);
			return false;
		}
		mem->flexible_memory_size = flex_ptr_addr;
		LOGF("PatchFlexibleScalarInProcParam: wired flex_ptr into proc_param pad at 0x%016" PRIx64
		     " (avoided fs_param overlap)\n",
		     flex_ptr_addr);
	} else if (!MakeGuestRegionWritable(proc->mem_param, sizeof(GuestKernelMemParam))) {
		LOGF("PatchFlexibleScalarInProcParam: failed to make mem_param writable at 0x%016" PRIx64
		     "\n",
		     proc->mem_param);
	}

	if (!GuestAddrIsReadable(flex_ptr_addr, sizeof(uint64_t), program)) {
		LOGF("PatchFlexibleScalarInProcParam: flex_ptr 0x%016" PRIx64 " not readable\n",
		     flex_ptr_addr);
		return false;
	}

	if (!MakeGuestRegionWritable(flex_ptr_addr, sizeof(uint64_t))) {
		LOGF("PatchFlexibleScalarInProcParam: failed to make flex_ptr writable at 0x%016" PRIx64 "\n",
		     flex_ptr_addr);
		return false;
	}

	auto* flex_ptr     = reinterpret_cast<uint64_t*>(flex_ptr_addr);
	const auto flex_before = *flex_ptr;
	LOGF("PatchFlexibleScalarInProcParam: mem_param=0x%016" PRIx64 " flex_ptr=0x%016" PRIx64
	     " *flex_ptr=0x%016" PRIx64 "\n",
	     proc->mem_param, flex_ptr_addr, flex_before);
	*flex_ptr = flex_scalar;
	LOGF("PatchFlexibleScalarInProcParam: *flex_ptr after=0x%016" PRIx64 "\n", *flex_ptr);

	if (!MemParamHasValidExtendedMemory(mem)) {
		WireGuestMemParamExtended(mem, proc_vaddr, synth_size);
	} else {
		LOGF("PatchFlexibleScalarInProcParam: preserved ELF extended_memory\n");
	}

	Libs::LibKernel::Memory::ApplyMemoryRegionsFromProcParam(flex_scalar);
	return true;
}

static uint64_t GetProgramSdkVersion(const Program* program) {
	if (program == nullptr || program->elf == nullptr) {
		return 0x05000000ull;
	}

	uint64_t module_attr = 0;
	if (const auto* dyn = program->elf->GetDynValue(DT_OS_MODULE_ATTR); dyn != nullptr) {
		module_attr = static_cast<uint64_t>(dyn->d_un.d_val);
	}
	if (module_attr != 0) {
		return module_attr;
	}

	return 0x05000000ull;
}

static bool MakeGuestRegionWritable(uint64_t vaddr, uint64_t size) {
	Common::VirtualMemory::Mode old_mode {};
	if (!Common::VirtualMemory::Protect(vaddr, size, Common::VirtualMemory::Mode::ReadWrite,
	                                    &old_mode)) {
		return false;
	}
	Libs::LibKernel::Memory::UpdateProgramMemoryProtection(vaddr, size,
	                                                       Common::VirtualMemory::Mode::ReadWrite);
	return true;
}

static void WireGuestProcParamPointers(GuestProcParam* proc, uint64_t proc_vaddr,
                                       uint64_t synth_size) {
	// Synth layout after the 0x60 header:
	//   +0x60: flex scalar (u64) — owned by PatchFlexibleScalarInProcParam
	//   +0x68: main_thread_stack_size value (u32)
	//   +0x70: heap_size scalar (u64) for ELF libc_param when heap_size ptr is null
	const auto stack_val_vaddr = AlignUp(proc_vaddr + sizeof(GuestProcParam) + sizeof(uint64_t), 8ull);
	const auto heap_val_vaddr  = AlignUp(stack_val_vaddr + sizeof(uint32_t), 8ull);

	if (heap_val_vaddr + sizeof(uint64_t) > proc_vaddr + synth_size) {
		LOGF("WireGuestProcParamPointers: proc_param synth too small\n");
		return;
	}

	if (proc->main_thread_stack_size == 0) {
		auto* stack_val = reinterpret_cast<uint32_t*>(stack_val_vaddr);
		*stack_val                    = 0x200000u;
		proc->main_thread_stack_size  = stack_val_vaddr;
		LOGF("WireGuestProcParamPointers: stack_size_ptr=0x%016" PRIx64 " *val=0x%x\n",
		     stack_val_vaddr, *stack_val);
	}

	if (proc->libc_param == 0) {
		return;
	}

	// PS5 libc_param is 0xa8; heap_size pointer remains at offset 0x10 like PS4.
	auto* libc_words = reinterpret_cast<uint64_t*>(proc->libc_param);
	const auto libc_size = libc_words[0];
	if (libc_size < 0x18) {
		return;
	}

	if (libc_words[2] == 0) {
		if (!MakeGuestRegionWritable(proc->libc_param, libc_size)) {
			LOGF("WireGuestProcParamPointers: failed to make libc_param writable\n");
			return;
		}
		auto* heap_val = reinterpret_cast<uint64_t*>(heap_val_vaddr);
		*heap_val      = UINT64_MAX;
		libc_words[2]  = heap_val_vaddr;
		LOGF("WireGuestProcParamPointers: libc heap_size_ptr=0x%016" PRIx64 " *val=0x%016" PRIx64
		     "\n",
		     heap_val_vaddr, *heap_val);
	}
}

static void WireGuestLibcParam(GuestProcParam* proc, uint64_t proc_vaddr, uint64_t synth_size) {
	const auto libc_vaddr =
	    AlignUp(proc_vaddr + sizeof(GuestProcParam) + sizeof(GuestKernelMemParam) + sizeof(uint64_t),
	            8ull);
	const auto heap_scalar_vaddr = libc_vaddr + sizeof(GuestLibcParam);

	if (heap_scalar_vaddr + sizeof(uint64_t) > proc_vaddr + synth_size) {
		LOGF("WireGuestLibcParam: proc_param segment too small for libc_param\n");
		return;
	}

	auto* libc         = reinterpret_cast<GuestLibcParam*>(libc_vaddr);
	auto* heap_scalar  = reinterpret_cast<uint64_t*>(heap_scalar_vaddr);

	libc->size          = ORBIS_LIBC_PARAM_STRUCT_SIZE;
	libc->heap_size     = heap_scalar_vaddr;
	libc->need_sce_libc = 1;
	*heap_scalar        = UINT64_MAX;

	proc->libc_param = libc_vaddr;

	LOGF("WireGuestLibcParam: libc_param=0x%016" PRIx64 " heap_size_ptr=0x%016" PRIx64 "\n",
	     libc_vaddr, heap_scalar_vaddr);
}

void SynthesizeProgramProcParam(Program* program) {
	if (program == nullptr || program->proc_param_vaddr == 0) {
		return;
	}

	const auto configured = Libs::LibKernel::Memory::GetFlexibleMemorySize();
	const auto flex_scalar =
	    (configured > FLEXIBLE_MEMORY_BASE ? configured - FLEXIBLE_MEMORY_BASE : 0ull);

	const auto proc_vaddr = program->proc_param_vaddr;
	const auto synth_size = std::max(program->proc_param_size, PROC_PARAM_SYNTH_SIZE);

	LogProcParamPostRelocate(program);

	std::vector<uint8_t> elf_backup(synth_size);
	std::memcpy(elf_backup.data(), reinterpret_cast<void*>(proc_vaddr), synth_size);
	const auto* elf_proc = reinterpret_cast<const GuestProcParam*>(elf_backup.data());
	const bool  elf_template =
	    IsValidProcParamMagic(elf_proc->magic) && elf_proc->size >= sizeof(GuestProcParam);
	LOGF("SynthesizeProgramProcParam: backup magic=0x%08" PRIx32 " size=0x%016" PRIx64
	     " elf_template=%d\n",
	     elf_proc->magic, elf_proc->size, elf_template ? 1 : 0);

	if (!MakeGuestRegionWritable(proc_vaddr, synth_size)) {
		LOGF("SynthesizeProgramProcParam: failed to make proc_param writable at 0x%016" PRIx64 "\n",
		     proc_vaddr);
		return;
	}

	auto* proc = reinterpret_cast<GuestProcParam*>(proc_vaddr);

	if (elf_template) {
		std::memcpy(proc, elf_backup.data(), synth_size);
		LOGF("SynthesizeProgramProcParam: merging ELF proc_param template at 0x%016" PRIx64
		     " sdk=0x%016" PRIx64 " magic=0x%08" PRIx32 "\n",
		     proc_vaddr, proc->sdk_version, proc->magic);
		// ELF sdk_version 0x0500003309590001 looks like a relocated/packed value; ND budget
		// init may key off a normal 0xMMmmmpp00 SDK word (e.g. 0x05000000).
		const auto elf_sdk = proc->sdk_version;
		proc->sdk_version  = GetProgramSdkVersion(program);
		LOGF("SynthesizeProgramProcParam: sdk normalized 0x%016" PRIx64 " -> 0x%016" PRIx64 "\n",
		     elf_sdk, proc->sdk_version);
		// Runtime/kernel consumers expect 0x13bccf52; keep ELF body but normalize magic.
		if (proc->magic == ORBIS_PROC_PARAM_FILE_MAGIC) {
			proc->magic = ORBIS_PROC_PARAM_MAGIC;
			LOGF("SynthesizeProgramProcParam: normalized magic ORBI -> 0x%08" PRIx32 "\n",
			     proc->magic);
		}
		// main_thread_stack_size / libc heap_size are guest pointers — wire into synth pad.
		WireGuestProcParamPointers(proc, proc_vaddr, synth_size);
		if (PatchFlexibleScalarInProcParam(proc, flex_scalar, proc_vaddr, synth_size, program)) {
			if (proc->libc_param == 0) {
				WireGuestLibcParam(proc, proc_vaddr, synth_size);
			} else {
				LOGF("WireGuestLibcParam: preserved ELF libc_param=0x%016" PRIx64 "\n",
				     proc->libc_param);
			}
			LogProcParamStructures("post-merge-patched", proc);
			return;
		}
		LOGF("SynthesizeProgramProcParam: ELF template preserved without flex scalar patch "
		     "(mem_param=0x%016" PRIx64 ")\n",
		     proc->mem_param);
		Libs::LibKernel::Memory::ApplyMemoryRegionsFromProcParam(flex_scalar);
		LogProcParamStructures("post-merge-nopatch", proc);
		return;
	}

	if (PatchFlexibleScalarInProcParam(proc, flex_scalar, proc_vaddr, synth_size, program)) {
		if (proc->libc_param == 0) {
			WireGuestLibcParam(proc, proc_vaddr, synth_size);
		}
		return;
	}

	const auto mem_vaddr    = proc_vaddr + sizeof(GuestProcParam);
	const auto scalar_vaddr = mem_vaddr + sizeof(GuestKernelMemParam);

	auto* mem    = reinterpret_cast<GuestKernelMemParam*>(mem_vaddr);
	auto* scalar = reinterpret_cast<uint64_t*>(scalar_vaddr);

	std::memset(proc, 0, synth_size);
	proc->size                  = ORBIS_PROC_PARAM_STRUCT_SIZE;
	proc->magic                 = ORBIS_PROC_PARAM_MAGIC;
	proc->entry_count           = 1;
	proc->sdk_version           = GetProgramSdkVersion(program);
	proc->main_thread_stack_size = 0x200000ull;
	proc->mem_param             = mem_vaddr;

	mem->size                 = sizeof(GuestKernelMemParam);
	mem->flexible_memory_size = scalar_vaddr;
	*scalar                   = flex_scalar;
	WireGuestMemParamExtended(mem, proc_vaddr, synth_size);

	Libs::LibKernel::Memory::ApplyMemoryRegionsFromProcParam(flex_scalar);
	WireGuestLibcParam(proc, proc_vaddr, synth_size);

	LOGF("SynthesizeProgramProcParam: proc=0x%016" PRIx64 " mem_param=0x%016" PRIx64
	     " flex_scalar=0x%016" PRIx64 " configured=0x%016" PRIx64 " sdk=0x%016" PRIx64 "\n",
	     proc_vaddr, mem_vaddr, flex_scalar, configured, proc->sdk_version);

	const auto* dump = reinterpret_cast<const uint8_t*>(proc_vaddr);
	for (size_t row = 0; row < 0x100; row += 16) {
		LOGF("  proc_param+%04zx: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
		     row, dump[row + 0], dump[row + 1], dump[row + 2], dump[row + 3], dump[row + 4],
		     dump[row + 5], dump[row + 6], dump[row + 7], dump[row + 8], dump[row + 9],
		     dump[row + 10], dump[row + 11], dump[row + 12], dump[row + 13], dump[row + 14],
		     dump[row + 15]);
	}
}

} // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RuntimeLinker::LoadProgramToMemory(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr || program->base_vaddr != 0 || program->base_size != 0 ||
	        program->elf == nullptr);

	// static uint64_t desired_base_addr = DESIRED_BASE_ADDR;

	bool is_shared   = program->elf->IsShared();
	bool is_next_gen = program->elf->IsNextGen();

	EXIT_NOT_IMPLEMENTED(!is_shared && !is_next_gen);

	const auto* ehdr = program->elf->GetEhdr();
	const auto* phdr = program->elf->GetPhdr();

	EXIT_IF(phdr == nullptr || ehdr == nullptr);

	program->base_size                 = CalcBaseSize(ehdr, phdr);
	constexpr uint64_t GUEST_PAGE_SIZE = 0x4000;
	EXIT_IF(program->base_size > UINT64_MAX - (GUEST_PAGE_SIZE - 1));
	program->base_size_aligned = AlignUp(program->base_size, GUEST_PAGE_SIZE);

	uint64_t tls_handler_size = is_shared ? 0 : Jit::SafeCall::GetSize();
	EXIT_IF(tls_handler_size > UINT64_MAX - program->base_size_aligned);
	program->mapped_size = program->base_size_aligned + tls_handler_size;

	program->base_vaddr = Common::VirtualMemory::Alloc(
	    g_desired_base_addr, program->mapped_size, Common::VirtualMemory::Mode::ExecuteReadWrite);

	if (!is_shared) {
		program->tls.handler_vaddr = program->base_vaddr + program->base_size_aligned;
	}

	g_desired_base_addr += CODE_BASE_INCR * (1 + program->mapped_size / CODE_BASE_INCR);

	EXIT_IF(program->base_vaddr == 0);
	EXIT_IF(program->base_size_aligned < program->base_size);
	RegisterLoadedProgramMemory(program);

	LOGF("base_vaddr             = 0x%016" PRIx64 "\n"
	     "base_size              = 0x%016" PRIx64 "\n"
	     "base_size_aligned      = 0x%016" PRIx64 "\n"
	     "mapped_size            = 0x%016" PRIx64 "\n",
	     program->base_vaddr, program->base_size, program->base_size_aligned, program->mapped_size);
	if (!is_shared) {
		LOGF("tls_handler_size       = 0x%016" PRIx64 "\n", tls_handler_size);
	}

	if (!Common::HostException::InstallHandler(KytyExceptionHandler)) {
		EXIT("Failed to install the required vectored exception handler\n");
	}

	// program->elf->SetBaseVAddr(program->base_vaddr);

	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_memsz != 0 && (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_OS_RELRO)) {
			uint64_t segment_addr        = phdr[i].p_vaddr + program->base_vaddr;
			uint64_t segment_file_size   = phdr[i].p_filesz;
			uint64_t segment_memory_size = GetAlignedSize(phdr + i);
			auto     mode                = GetMode(phdr[i].p_flags);
			const auto module_name       = Common::PathToString(program->file_name.filename());

			LOGF("[%d] addr        = 0x%016" PRIx64 "\n"
			     "[%d] file_size   = %" PRIu64 "\n"
			     "[%d] memory_size = %" PRIu64 "\n"
			     "[%d] mode        = %s\n",
			     i, segment_addr, i, segment_file_size, i, segment_memory_size, i,
			     Common::EnumName(mode).c_str());

			Libs::LibKernel::Memory::RegisterProgramSegmentMemory(
			    segment_addr, segment_memory_size, mode, module_name.c_str(),
			    IsSystemModuleFilename(module_name));

			program->elf->LoadSegment(segment_addr, phdr[i].p_offset, segment_file_size);

			bool skip_protect = (phdr[i].p_type == PT_LOAD && is_next_gen &&
			                     mode == Common::VirtualMemory::Mode::NoAccess);

			if (Common::VirtualMemory::IsExecute(mode)) {
				PatchProgram(program, segment_addr, segment_memory_size);
			}

			if (!skip_protect) {
				if (!Common::VirtualMemory::Protect(segment_addr, segment_memory_size, mode)) {
					EXIT("failed to protect ELF segment %u\n", static_cast<unsigned>(i));
				}
				Libs::LibKernel::Memory::UpdateProgramMemoryProtection(segment_addr,
				                                                       segment_memory_size, mode);

				if (Common::VirtualMemory::IsExecute(mode)) {
					Common::VirtualMemory::FlushInstructionCache(segment_addr, segment_memory_size);
				}
			}
		}

		if (phdr[i].p_type == PT_TLS) {
			EXIT_IF(phdr[i].p_vaddr >= program->base_size);

			program->tls.image_vaddr = phdr[i].p_vaddr + program->base_vaddr;
			program->tls.init_size   = std::min(phdr[i].p_filesz, GetAlignedSize(phdr + i));
			program->tls.image_size  = GetAlignedSize(phdr + i);
			program->tls.tcb_offset  = program->tls.image_size;

			LOGF("tls addr = 0x%016" PRIx64 "\n"
			     "tls init   = %" PRIu64 "\n"
			     "tls size   = %" PRIu64 "\n"
			     "tls offset = %" PRIu64 "\n",
			     program->tls.image_vaddr, program->tls.init_size, program->tls.image_size,
			     program->tls.tcb_offset);
		}

		if (phdr[i].p_type == PT_OS_PROCPARAM) {
			EXIT_IF(program->proc_param_vaddr != 0);
			EXIT_IF(phdr[i].p_vaddr >= program->base_size);

			program->proc_param_vaddr = phdr[i].p_vaddr + program->base_vaddr;
			program->proc_param_size  = phdr[i].p_memsz;
			LOGF("proc_param_vaddr = 0x%016" PRIx64 " size = 0x%016" PRIx64 "\n",
			     program->proc_param_vaddr, program->proc_param_size);

			if (phdr[i].p_filesz > 0) {
				program->elf->LoadSegment(program->proc_param_vaddr, phdr[i].p_offset,
				                          phdr[i].p_filesz);
				LOGF("proc_param loaded from ELF: file_size = 0x%016" PRIx64 "\n", phdr[i].p_filesz);
			}
		}
	}

	{
		const auto module_name = Common::PathToString(program->file_name.filename());
		Libs::LibKernel::Memory::FillProgramMemoryGaps(program->base_vaddr,
		                                               program->base_size_aligned,
		                                               module_name.c_str());
	}

	if (!is_shared) {
		SetupTlsHandler(program);
	}

	LOGF("entry = 0x%016" PRIx64 "\n", program->elf->GetEntry() + program->base_vaddr);
}

void RuntimeLinker::DeleteProgram(Program* p) {
	auto program = std::unique_ptr<Program>(p);

	if (program->base_vaddr != 0 || program->mapped_size != 0) {
		EXIT_IF(program->base_vaddr == 0 || program->mapped_size == 0);
		Libs::LibKernel::Memory::UnregisterProgramMemory(program->base_vaddr,
		                                                 program->base_size_aligned);
		Libs::LibKernel::Memory::UnregisterProgramFlexibleQuota(
		    program->base_size_aligned, Common::PathToString(program->file_name.filename()).c_str());
		EXIT_IF(!Common::VirtualMemory::Free(program->base_vaddr));
	}

	if (program->custom_call_plt_vaddr != 0 || program->custom_call_plt_num != 0) {
		Common::VirtualMemory::Free(program->custom_call_plt_vaddr);
	}
}

void RuntimeLinker::ParseProgramDynamicInfo(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(program->dynamic_info != nullptr);

	program->dynamic_info = std::make_unique<DynamicInfo>();

	auto* elf = program->elf.get();

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_HASH) && elf->HasDynValue(DT_HASH));
	GetDynDataOs(elf, &program->dynamic_info->hash_table, DT_OS_HASH);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->hash_table, DT_HASH);
	GetDynValue(elf, &program->dynamic_info->hash_table_size, DT_OS_HASHSZ);

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_STRTAB) && elf->HasDynValue(DT_STRTAB));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_STRSZ) && elf->HasDynValue(DT_STRSZ));
	GetDynDataOs(elf, &program->dynamic_info->str_table, DT_OS_STRTAB);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->str_table, DT_STRTAB);
	GetDynValue(elf, &program->dynamic_info->str_table_size, DT_OS_STRSZ);
	GetDynValue(elf, &program->dynamic_info->str_table_size, DT_STRSZ);

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_SYMTAB) && elf->HasDynValue(DT_SYMTAB));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_SYMENT) && elf->HasDynValue(DT_SYMENT));
	GetDynDataOs(elf, &program->dynamic_info->symbol_table, DT_OS_SYMTAB);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->symbol_table, DT_SYMTAB);
	GetDynValue(elf, &program->dynamic_info->symbol_table_total_size, DT_OS_SYMTABSZ);
	GetDynValue(elf, &program->dynamic_info->symbol_table_entry_size, DT_OS_SYMENT);
	GetDynValue(elf, &program->dynamic_info->symbol_table_entry_size, DT_SYMENT);

	GetDynPtr(elf, &program->dynamic_info->init_vaddr, DT_INIT);
	GetDynPtr(elf, &program->dynamic_info->fini_vaddr, DT_FINI);
	GetDynPtr(elf, &program->dynamic_info->init_array_vaddr, DT_INIT_ARRAY);
	GetDynPtr(elf, &program->dynamic_info->fini_array_vaddr, DT_FINI_ARRAY);
	GetDynPtr(elf, &program->dynamic_info->preinit_array_vaddr, DT_PREINIT_ARRAY);
	GetDynValue(elf, &program->dynamic_info->init_array_size, DT_INIT_ARRAYSZ);
	GetDynValue(elf, &program->dynamic_info->fini_array_size, DT_FINI_ARRAYSZ);
	GetDynValue(elf, &program->dynamic_info->preinit_array_size, DT_PREINIT_ARRAYSZ);

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_PLTGOT) && elf->HasDynValue(DT_PLTGOT));
	GetDynPtr(elf, &program->dynamic_info->pltgot_vaddr, DT_OS_PLTGOT);
	GetDynPtr(elf, &program->dynamic_info->pltgot_vaddr, DT_PLTGOT);

	Elf64_Sxword jmprel_type = 0;
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_PLTREL) && elf->HasDynValue(DT_PLTREL));
	GetDynValue(elf, &jmprel_type, DT_OS_PLTREL);
	GetDynValue(elf, &jmprel_type, DT_PLTREL);

	EXIT_NOT_IMPLEMENTED(jmprel_type != DT_RELA);
	if (jmprel_type == DT_RELA) {
		EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_JMPREL) && elf->HasDynValue(DT_JMPREL));
		EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_PLTRELSZ) && elf->HasDynValue(DT_PLTRELSZ));
		GetDynDataOs(elf, &program->dynamic_info->jmprela_table, DT_OS_JMPREL);
		GetDynData(elf, program->base_vaddr, &program->dynamic_info->jmprela_table, DT_JMPREL);
		GetDynValue(elf, &program->dynamic_info->jmprela_table_size, DT_OS_PLTRELSZ);
		GetDynValue(elf, &program->dynamic_info->jmprela_table_size, DT_PLTRELSZ);
	}

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_RELA) && elf->HasDynValue(DT_RELA));
	GetDynDataOs(elf, &program->dynamic_info->rela_table, DT_OS_RELA);
	GetDynData(elf, program->base_vaddr, &program->dynamic_info->rela_table, DT_RELA);
	GetDynValue(elf, &program->dynamic_info->rela_table_total_size, DT_OS_RELASZ);
	GetDynValue(elf, &program->dynamic_info->rela_table_total_size, DT_RELASZ);
	GetDynValue(elf, &program->dynamic_info->rela_table_entry_size, DT_OS_RELAENT);
	GetDynValue(elf, &program->dynamic_info->rela_table_entry_size, DT_RELAENT);

	GetDynValue(elf, &program->dynamic_info->relative_count, DT_RELACOUNT);

	GetDynValue(elf, &program->dynamic_info->debug, DT_DEBUG);
	GetDynValue(elf, &program->dynamic_info->flags, DT_FLAGS);
	GetDynValue(elf, &program->dynamic_info->textrel, DT_TEXTREL);

	EXIT_NOT_IMPLEMENTED(program->dynamic_info->debug != 0);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->textrel != 0);

	std::vector<uint64_t> needed;
	GetDynValues(elf, &needed, DT_NEEDED);
	for (auto need: needed) {
		program->dynamic_info->needed.push_back(program->dynamic_info->str_table + need);
	}

	uint64_t so_name = 0;
	GetDynValue(elf, &so_name, DT_SONAME);
	program->dynamic_info->so_name = program->dynamic_info->str_table + so_name;

	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_NEEDED_MODULE) &&
	                     elf->HasDynValue(DT_OS_NEEDED_MODULE_1));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_MODULE_INFO) &&
	                     elf->HasDynValue(DT_OS_MODULE_INFO_1));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_IMPORT_LIB) &&
	                     elf->HasDynValue(DT_OS_IMPORT_LIB_1));
	EXIT_NOT_IMPLEMENTED(elf->HasDynValue(DT_OS_EXPORT_LIB) &&
	                     elf->HasDynValue(DT_OS_EXPORT_LIB_1));
	GetDynModules(elf, &program->dynamic_info->import_modules, program->dynamic_info->str_table,
	              DT_OS_NEEDED_MODULE);
	GetDynModules(elf, &program->dynamic_info->import_modules, program->dynamic_info->str_table,
	              DT_OS_NEEDED_MODULE_1);
	GetDynModules(elf, &program->dynamic_info->export_modules, program->dynamic_info->str_table,
	              DT_OS_MODULE_INFO);
	GetDynModules(elf, &program->dynamic_info->export_modules, program->dynamic_info->str_table,
	              DT_OS_MODULE_INFO_1);
	GetDynLibs(elf, &program->dynamic_info->import_libs, program->dynamic_info->str_table,
	           DT_OS_IMPORT_LIB);
	GetDynLibs(elf, &program->dynamic_info->import_libs, program->dynamic_info->str_table,
	           DT_OS_IMPORT_LIB_1);
	GetDynLibs(elf, &program->dynamic_info->export_libs, program->dynamic_info->str_table,
	           DT_OS_EXPORT_LIB);
	GetDynLibs(elf, &program->dynamic_info->export_libs, program->dynamic_info->str_table,
	           DT_OS_EXPORT_LIB_1);
}

static void InstallRelocateHandler(Program* program) {
	KYTY_PROFILER_FUNCTION();

	uint64_t pltgot_vaddr = program->dynamic_info->pltgot_vaddr + program->base_vaddr;
	uint64_t pltgot_size  = static_cast<uint64_t>(3) * 8;
	void**   pltgot       = reinterpret_cast<void**>(pltgot_vaddr);

	Common::VirtualMemory::Mode old_mode {};
	Common::VirtualMemory::Protect(pltgot_vaddr, pltgot_size, Common::VirtualMemory::Mode::Write,
	                               &old_mode);

	pltgot[1] = program;
	pltgot[2] = reinterpret_cast<void*>(RelocateHandler);

	Common::VirtualMemory::Protect(pltgot_vaddr, pltgot_size, old_mode);

	if (Common::VirtualMemory::IsExecute(old_mode)) {
		Common::VirtualMemory::FlushInstructionCache(pltgot_vaddr, pltgot_size);
	}

	// TODO(): check if this table already generated by compiler (sometimes it is missing)
	if (program->custom_call_plt_vaddr == 0) {
		program->custom_call_plt_num =
		    program->dynamic_info->jmprela_table_size / sizeof(Elf64_Rela);
		auto size = Jit::CallPlt::GetSize(program->custom_call_plt_num);
		program->custom_call_plt_vaddr =
		    Common::VirtualMemory::Alloc(SYSTEM_RESERVED, size, Common::VirtualMemory::Mode::Write);
		EXIT_NOT_IMPLEMENTED(program->custom_call_plt_vaddr == 0);
		auto* code = new (reinterpret_cast<void*>(program->custom_call_plt_vaddr))
		    Jit::CallPlt(program->custom_call_plt_num);
		code->SetPltGot(pltgot_vaddr);
		Common::VirtualMemory::Protect(program->custom_call_plt_vaddr, size,
		                               Common::VirtualMemory::Mode::ExecuteRead);
		Common::VirtualMemory::FlushInstructionCache(program->custom_call_plt_vaddr, size);
	}
}

void RuntimeLinker::Relocate(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr);

	if (g_invalid_memory == 0) {
		g_invalid_memory = Common::VirtualMemory::Alloc(INVALID_MEMORY, 4096,
		                                                Common::VirtualMemory::Mode::NoAccess);
		EXIT_NOT_IMPLEMENTED(g_invalid_memory == 0);
	}

	LOGF_COLOR(Log::Color::White, "--- Relocate program: %s ---\n",
	           Common::PathToString(program->file_name).c_str());

	EXIT_NOT_IMPLEMENTED(program->dynamic_info->symbol_table_entry_size != sizeof(Elf64_Sym));
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->rela_table_entry_size != sizeof(Elf64_Rela));
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->jmprela_table == nullptr);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->rela_table == nullptr);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->symbol_table == nullptr);
	EXIT_NOT_IMPLEMENTED(program->dynamic_info->pltgot_vaddr == 0);

	InstallRelocateHandler(program);

	std::vector<std::string> unresolved;
	const bool               imports_only = program->relocated;

	RelocateRecords(program->dynamic_info->rela_table, program->dynamic_info->rela_table_total_size,
	                program, false, imports_only, &unresolved);
	RelocateRecords(program->dynamic_info->jmprela_table, program->dynamic_info->jmprela_table_size,
	                program, true, imports_only, &unresolved);
	program->relocated = true;

	if (program->tls.image_vaddr != 0 && program->tls.init_size != 0 &&
	    program->tls.init_image.empty()) {
		const auto* src = reinterpret_cast<const uint8_t*>(program->tls.image_vaddr);
		program->tls.init_image.assign(src, src + program->tls.init_size);
	}

	if (!unresolved.empty()) {
		LOGF("--- Stubbed unresolved imports: %zu ---\n", unresolved.size());
		for (const auto& symbol: unresolved) {
			LOGF("Stubbed: %s\n", symbol.c_str());
		}
	}
}

Program* RuntimeLinker::FindProgram(const ModuleId& m, const LibraryId& l) {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		const auto& export_libs    = p->dynamic_info->export_libs;
		const auto& export_modules = p->dynamic_info->export_modules;

		if (std::find(export_libs.begin(), export_libs.end(), l) != export_libs.end() &&
		    std::find(export_modules.begin(), export_modules.end(), m) != export_modules.end()) {
			return p;
		}
	}
	return nullptr;
}

const ModuleId* RuntimeLinker::FindModule(const Program& program, const std::string& id) {
	const auto& import_modules = program.dynamic_info->import_modules;

	if (auto it = std::find_if(import_modules.begin(), import_modules.end(),
	                           [&id](const auto& module) { return module.id == id; });
	    it != import_modules.end()) {
		return &(*it);
	}

	const auto& export_modules = program.dynamic_info->export_modules;

	if (auto it = std::find_if(export_modules.begin(), export_modules.end(),
	                           [&id](const auto& module) { return module.id == id; });
	    it != export_modules.end()) {
		return &(*it);
	}

	return nullptr;
}

const LibraryId* RuntimeLinker::FindLibrary(const Program& program, const std::string& id) {
	const auto& import_libs = program.dynamic_info->import_libs;

	if (auto it = std::find_if(import_libs.begin(), import_libs.end(),
	                           [&id](const auto& lib) { return lib.id == id; });
	    it != import_libs.end()) {
		return &(*it);
	}

	const auto& export_libs = program.dynamic_info->export_libs;

	if (auto it = std::find_if(export_libs.begin(), export_libs.end(),
	                           [&id](const auto& lib) { return lib.id == id; });
	    it != export_libs.end()) {
		return &(*it);
	}

	return nullptr;
}

void RuntimeLinker::CreateSymbolDatabase(Program* program) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(program == nullptr);
	EXIT_IF(program->export_symbols != nullptr);
	EXIT_IF(program->import_symbols != nullptr);

	program->export_symbols = std::make_unique<SymbolDatabase>();
	program->import_symbols = std::make_unique<SymbolDatabase>();

	auto syms = [](Program* program, SymbolDatabase* symbols, bool is_export) {
		if (program->dynamic_info->symbol_table == nullptr ||
		    program->dynamic_info->str_table == nullptr) {
			return;
		}

		for (auto* sym = program->dynamic_info->symbol_table;
		     reinterpret_cast<uint8_t*>(sym) <
		     reinterpret_cast<uint8_t*>(program->dynamic_info->symbol_table) +
		         program->dynamic_info->symbol_table_total_size;
		     sym++) {
			std::string id   = std::string(program->dynamic_info->str_table + sym->st_name);
			auto        bind = sym->GetBind();
			auto        type = sym->GetType();
			auto        ids  = Common::Split(id, '#');

			if (ids.size() == 3) {
				const auto* l = FindLibrary(*program, ids.at(1));
				const auto* m = FindModule(*program, ids.at(2));

				if (l != nullptr && m != nullptr && (bind == STB_GLOBAL || bind == STB_WEAK) &&
				    (type == STT_FUNC || type == STT_OBJECT || type == STT_NOTYPE) &&
				    is_export == (sym->st_value != 0)) {
					SymbolResolve sr {};
					sr.name                 = ids.at(0);
					sr.library              = l->name;
					sr.library_version      = l->version;
					sr.module               = m->name;
					sr.module_version_major = m->version_major;
					sr.module_version_minor = m->version_minor;
					switch (type) {
						case STT_NOTYPE: sr.type = SymbolType::NoType; break;
						case STT_FUNC: sr.type = SymbolType::Func; break;
						case STT_OBJECT: sr.type = SymbolType::Object; break;
						default: sr.type = SymbolType::Unknown; break;
					}
					symbols->Add(sr, (is_export ? sym->st_value + program->base_vaddr : 0));
				}
			}
		}
	};

	syms(program, program->export_symbols.get(), true);
	syms(program, program->import_symbols.get(), false);
}

void RuntimeLinker::SetupTlsHandler(Program* program) {
	EXIT_IF(program == nullptr);
	EXIT_IF(g_tls_main_program != nullptr);
	EXIT_IF(program->elf == nullptr);
	EXIT_IF(program->elf->IsShared());
	EXIT_IF(program->tls.handler_vaddr == 0);

	g_tls_main_program = program;

	auto* code = new (reinterpret_cast<void*>(program->tls.handler_vaddr)) Jit::SafeCall;

	code->SetFunc(TlsMainGetAddr);

	for (uint8_t reg = 1; reg < 8; reg++) {
		if (reg == 4) {
			continue;
		}

		auto* stub = new (reinterpret_cast<void*>(program->tls.handler_vaddr +
		                                          Jit::TlsRegStub::GetOffset(reg))) Jit::TlsRegStub;
		stub->SetFunc(program->tls.handler_vaddr);
		stub->SetOutputReg(reg);
	}

	if (!Common::VirtualMemory::Protect(program->tls.handler_vaddr, Jit::SafeCall::GetSize(),
	                                    Common::VirtualMemory::Mode::ExecuteRead)) {
		EXIT("failed to protect program TLS handler\n");
	}
	Common::VirtualMemory::FlushInstructionCache(program->tls.handler_vaddr,
	                                             Jit::SafeCall::GetSize());
}

void RuntimeLinker::DeleteTlss(int thread_id) {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_programs) {
		DeleteTls(p, thread_id);
	}
}

void RuntimeLinker::SetApplicationHeapApi(void* const api[10]) {
	Common::LockGuard lock(m_mutex);

	if (api == nullptr || api[0] == nullptr || api[1] == nullptr) {
		m_application_heap_malloc         = nullptr;
		m_application_heap_free           = nullptr;
		m_application_heap_posix_memalign = nullptr;
		return;
	}

	m_application_heap_malloc = reinterpret_cast<application_heap_malloc_func_t>(api[0]);
	m_application_heap_free   = reinterpret_cast<application_heap_free_func_t>(api[1]);
	m_application_heap_posix_memalign =
	    reinterpret_cast<application_heap_posix_memalign_func_t>(api[6]);
}

bool RuntimeLinker::IsApplicationHeapApiSet() const {
	return m_application_heap_malloc != nullptr;
}

void RuntimeLinker::EnsureApplicationHeapApi() {
	{
		Common::LockGuard lock(m_mutex);

		if (m_application_heap_malloc != nullptr) {
			LOGF("EnsureApplicationHeapApi: set (malloc=0x%016" PRIx64 " free=0x%016" PRIx64
			     " memalign=0x%016" PRIx64 ")\n",
			     reinterpret_cast<uint64_t>(reinterpret_cast<void*>(m_application_heap_malloc)),
			     reinterpret_cast<uint64_t>(reinterpret_cast<void*>(m_application_heap_free)),
			     reinterpret_cast<uint64_t>(
			         reinterpret_cast<void*>(m_application_heap_posix_memalign)));
			return;
		}
	}

	LOGF_COLOR(Log::Color::Yellow,
	           "EnsureApplicationHeapApi: NOT set after StartAllModules, probing libc.prx\n");

	using malloc_init_func_t = KYTY_SYSV_ABI int (*)();

	std::vector<uint64_t> init_funcs;
	{
		Common::LockGuard lock(m_mutex);

		for (auto* program: m_programs) {
			if (program == nullptr || program->elf == nullptr || !program->elf->IsShared() ||
			    program->export_symbols == nullptr) {
				continue;
			}

			const auto file = Common::ToLower(Common::PathToString(program->file_name.filename()));
			if (!Common::EndsWith(file, "libc.prx")) {
				continue;
			}

			static constexpr const char* kMallocInitNids[] = {"EHsF2i9FXPM", "_malloc_init"};
			for (const char* nid: kMallocInitNids) {
				const auto* rec = program->export_symbols->FindByNid(nid, SymbolType::Func);
				if (rec != nullptr && rec->vaddr != 0) {
					init_funcs.push_back(rec->vaddr);
				}
			}
		}
	}

	for (const auto vaddr: init_funcs) {
		LOGF("EnsureApplicationHeapApi: calling malloc_init at 0x%016" PRIx64 "\n", vaddr);
		const auto ret = reinterpret_cast<malloc_init_func_t>(vaddr)();
		LOGF("EnsureApplicationHeapApi: malloc_init returned %d, heap_api %s\n", ret,
		     IsApplicationHeapApiSet() ? "set" : "still unset");
		if (IsApplicationHeapApiSet()) {
			return;
		}
	}

	if (!IsApplicationHeapApiSet()) {
		LOGF_COLOR(Log::Color::Red, "EnsureApplicationHeapApi: heap API still unset\n");
	}
}

void* RuntimeLinker::ApplicationHeapMemalign(uint64_t alignment, uint64_t size) {
	Common::LockGuard lock(m_mutex);

	if (m_application_heap_posix_memalign != nullptr) {
		void* ptr = nullptr;
		return m_application_heap_posix_memalign(&ptr, alignment, size) == 0 ? ptr : nullptr;
	}

	return nullptr;
}

void* RuntimeLinker::ApplicationHeapMalloc(uint64_t size) {
	Common::LockGuard lock(m_mutex);

	return m_application_heap_malloc != nullptr ? m_application_heap_malloc(size) : nullptr;
}

application_heap_free_func_t RuntimeLinker::ApplicationHeapFreeFunc() const {
	return m_application_heap_free;
}

std::vector<StubbedImportSummary> GetStubbedImportReport() {
	std::vector<StubbedImportSummary> report;
	report.reserve(g_stubbed_imports.size());

	for (const auto& record: g_stubbed_imports) {
		StubbedImportSummary entry {};
		entry.name               = record.name;
		entry.program            = record.program;
		entry.registration_index = record.index;
		entry.call_count         = record.call_count;
		entry.was_called         = record.call_count > 0;
		report.push_back(std::move(entry));
	}

	std::sort(report.begin(), report.end(), [](const StubbedImportSummary& a,
	                                           const StubbedImportSummary& b) {
		if (a.call_count != b.call_count) {
			return a.call_count > b.call_count;
		}
		return a.name < b.name;
	});

	return report;
}

bool WriteStubbedImportReport(const std::filesystem::path& file_path) {
	const auto report = GetStubbedImportReport();

	Common::File file;
	file.Create(file_path);
	if (file.IsInvalid()) {
		LOGF("Can't create stub import report: %s\n", Common::PathToString(file_path).c_str());
		return false;
	}

	uint64_t registered = report.size();
	uint64_t called     = 0;
	for (const auto& entry: report) {
		if (entry.was_called) {
			called++;
		}
	}

	file.Printf("# KytyPS5 unresolved import report\n");
	file.Printf("# registered=%" PRIu64 " called=%" PRIu64 "\n\n", registered, called);

	for (const auto& entry: report) {
		file.Printf("[%s] calls=%" PRIu64 " index=%" PRIu32 " program=%s\n", entry.name.c_str(),
		            entry.call_count, entry.registration_index, entry.program.c_str());
	}

	file.Printf("\n# called_during_nd_window\n");
	uint64_t nd_window_called = 0;
	for (const auto& record: g_stubbed_imports) {
		if (!record.called_during_nd_window) {
			continue;
		}
		nd_window_called++;
		file.Printf("[%s] calls=%" PRIu64 " index=%" PRIu32 " program=%s\n", record.name.c_str(),
		            record.call_count, record.index, record.program.c_str());
	}
	file.Printf("# nd_window_called=%" PRIu64 "\n", nd_window_called);

	file.Close();
	LOGF("Wrote stub import report (%" PRIu64 " entries, %" PRIu64 " called): %s\n", registered,
	     called, Common::PathToString(file_path).c_str());
	return true;
}

} // namespace Loader
