#include "common/crashDiagnostics.h"

#include "common/common.h"
#include "common/fatalLog.h"
#include "common/threads.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstring>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#endif

namespace Common {

namespace {

constexpr size_t kHleRingSize = 64;
constexpr size_t kHleNameMax  = 96;

struct HleRingEntry {
	uint32_t tid = 0;
	char     library[32] {};
	char     module[32] {};
	char     func[kHleNameMax] {};
};

std::atomic<uint32_t> g_hle_seq {0};
HleRingEntry          g_hle_ring[kHleRingSize] {};
std::atomic<bool>     g_log_guest_av {false};
std::atomic<uint32_t> g_guest_av_logs {0};

void CopyCapped(char* dst, size_t dst_size, const char* src) {
	if (dst == nullptr || dst_size == 0) {
		return;
	}
	if (src == nullptr) {
		dst[0] = '\0';
		return;
	}
	std::snprintf(dst, dst_size, "%s", src);
}

} // namespace

void EnableGuestAccessViolationLogging(bool enable) {
	g_log_guest_av.store(enable, std::memory_order_release);
}

void NoteHleCall(const char* library, const char* module, const char* func) {
	const uint32_t seq = g_hle_seq.fetch_add(1, std::memory_order_relaxed);
	auto&          e   = g_hle_ring[seq % kHleRingSize];
	e.tid              = static_cast<uint32_t>(Thread::GetThreadIdUnique());
	CopyCapped(e.library, sizeof(e.library), library);
	CopyCapped(e.module, sizeof(e.module), module);
	CopyCapped(e.func, sizeof(e.func), func);
}

void FlushHleRingToFatal(const char* reason) {
	char header[192];
	std::snprintf(header, sizeof(header), "HLE ring flush reason=%s seq=%" PRIu32,
	              reason != nullptr ? reason : "(null)", g_hle_seq.load(std::memory_order_relaxed));
	LogFatalToFile(header);

	const uint32_t seq   = g_hle_seq.load(std::memory_order_relaxed);
	const size_t   count = seq < kHleRingSize ? seq : kHleRingSize;
	for (size_t i = 0; i < count; i++) {
		const size_t idx =
		    count < kHleRingSize ? i : static_cast<size_t>((seq - count + i) % kHleRingSize);
		const auto& e = g_hle_ring[idx];
		char        line[256];
		std::snprintf(line, sizeof(line), "  hle[%zu] tid=%u %s::%s::%s", i, e.tid, e.library,
		              e.module, e.func);
		LogFatalToFile(line);
	}
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	FlushFileBuffers(GetStdHandle(STD_ERROR_HANDLE));
#endif
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

using SecerrHandlerFunc    = void(__cdecl*)(int, void*);
using SetSecerrHandlerFunc = SecerrHandlerFunc(__cdecl*)(SecerrHandlerFunc);

static void SecurityFailureHandler(int code, void* return_address) {
	char module_name[MAX_PATH] = {};
	if (return_address != nullptr) {
		HMODULE owner_module = nullptr;
		if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                       static_cast<LPCSTR>(return_address), &owner_module) != 0 &&
		    owner_module != nullptr) {
			(void)GetModuleFileNameA(owner_module, module_name, MAX_PATH);
		}
	}

	char message[512];
	std::snprintf(message, sizeof(message),
	              "Security failure: code=%d, retaddr=%p, module=%s", code, return_address,
	              module_name[0] != '\0' ? module_name : "(unknown)");
	LogFatalToFile(message);
	FlushHleRingToFatal("security_failure");
}

static LONG CALLBACK FatalFailfastVectoredHandler(PEXCEPTION_POINTERS exception) {
	if (exception == nullptr || exception->ExceptionRecord == nullptr) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	const auto code = static_cast<uint32_t>(exception->ExceptionRecord->ExceptionCode);
	// ACCESS_VIOLATION / any exception after pending0 — log then continue search.
	if (g_log_guest_av.load(std::memory_order_acquire) &&
	    g_guest_av_logs.fetch_add(1, std::memory_order_relaxed) < 32) {
		auto* ctx = exception->ContextRecord;
		const auto rip = ctx != nullptr ? ctx->Rip : 0ull;
		const auto addr =
		    exception->ExceptionRecord->NumberParameters > 1
		        ? exception->ExceptionRecord->ExceptionInformation[1]
		        : 0ull;
		char message[384];
		std::snprintf(message, sizeof(message),
		              "GuestExc VEH: code=0x%08" PRIx32 " rip=0x%016" PRIx64
		              " addr=0x%016" PRIx64 " write=%u",
		              code, rip, static_cast<uint64_t>(addr),
		              exception->ExceptionRecord->NumberParameters > 0
		                  ? static_cast<unsigned>(
		                        exception->ExceptionRecord->ExceptionInformation[0])
		                  : 0u);
		LogFatalToFile(message);
		FlushHleRingToFatal("guest_exc");
		FlushFileBuffers(GetStdHandle(STD_ERROR_HANDLE));
	}
	if (code == 0xC0000005u) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	// STATUS_STACK_BUFFER_OVERRUN / HEAP_CORRUPTION / STACK_OVERFLOW — often from __fastfail.
	if (code != 0xC0000409u && code != 0xC0000374u && code != 0xC00000FDu) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	auto* ctx = exception->ContextRecord;
	char  module_name[MAX_PATH] = {};
	const auto rip = ctx != nullptr ? ctx->Rip : 0ull;
	HMODULE    owner_module = nullptr;
	if (rip != 0 &&
	    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
	                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCSTR>(rip), &owner_module) != 0 &&
	    owner_module != nullptr) {
		(void)GetModuleFileNameA(owner_module, module_name, MAX_PATH);
	}

	char message[768];
	const auto info0 =
	    exception->ExceptionRecord->NumberParameters > 0
	        ? exception->ExceptionRecord->ExceptionInformation[0]
	        : 0ull;
	const auto info1 =
	    exception->ExceptionRecord->NumberParameters > 1
	        ? exception->ExceptionRecord->ExceptionInformation[1]
	        : 0ull;
	std::snprintf(message, sizeof(message),
	              "FatalFailfast VEH: code=0x%08" PRIx32 ", addr=0x%016" PRIx64
	              ", rip=0x%016" PRIx64 ", rsp=0x%016" PRIx64
	              ", info0=0x%016" PRIx64 ", info1=0x%016" PRIx64 ", module=%s",
	              code, reinterpret_cast<uint64_t>(exception->ExceptionRecord->ExceptionAddress),
	              rip, ctx != nullptr ? ctx->Rsp : 0ull, static_cast<uint64_t>(info0),
	              static_cast<uint64_t>(info1),
	              module_name[0] != '\0' ? module_name : "(unknown)");
	LogFatalToFile(message);
	if (ctx != nullptr) {
		char regs[384];
		std::snprintf(regs, sizeof(regs),
		              "  failfast regs rax=0x%016" PRIx64 " rcx=0x%016" PRIx64
		              " rdx=0x%016" PRIx64 " rbx=0x%016" PRIx64 " rsi=0x%016" PRIx64
		              " rdi=0x%016" PRIx64,
		              ctx->Rax, ctx->Rcx, ctx->Rdx, ctx->Rbx, ctx->Rsi, ctx->Rdi);
		LogFatalToFile(regs);
	}
	FlushHleRingToFatal("veh_failfast");
	FlushFileBuffers(GetStdHandle(STD_ERROR_HANDLE));
	return EXCEPTION_CONTINUE_SEARCH;
}

#endif

void InstallCrashDiagnostics() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef ProcessUserShadowStackPolicy
#define ProcessUserShadowStackPolicy static_cast<PROCESS_MITIGATION_POLICY>(9)
#endif
	PROCESS_MITIGATION_USER_SHADOW_STACK_POLICY shadow_stack_policy {};
	if (SetProcessMitigationPolicy(ProcessUserShadowStackPolicy, &shadow_stack_policy,
	                               sizeof(shadow_stack_policy)) == 0) {
		::printf("Warning: SetProcessMitigationPolicy(UserShadowStack) failed: 0x%08" PRIx32 "\n",
		         static_cast<uint32_t>(GetLastError()));
	}

	PROCESS_MITIGATION_USER_SHADOW_STACK_POLICY shadow_stack_state {};
	if (GetProcessMitigationPolicy(GetCurrentProcess(), ProcessUserShadowStackPolicy,
	                               &shadow_stack_state, sizeof(shadow_stack_state)) != 0) {
		char message[256];
		std::snprintf(
		    message, sizeof(message),
		    "UserShadowStack policy: Flags=0x%08" PRIx32
		    " Enable=%u Audit=%u SetContextIpValidation=%u Strict=%u",
		    static_cast<uint32_t>(shadow_stack_state.Flags),
		    static_cast<unsigned>(shadow_stack_state.EnableUserShadowStack),
		    static_cast<unsigned>(shadow_stack_state.AuditUserShadowStack),
		    static_cast<unsigned>(shadow_stack_state.SetContextIpValidation),
		    static_cast<unsigned>(shadow_stack_state.EnableUserShadowStackStrictMode));
		::printf("%s\n", message);
		LogFatalToFile(message);
	} else {
		::printf("Warning: GetProcessMitigationPolicy(UserShadowStack) failed: 0x%08" PRIx32 "\n",
		         static_cast<uint32_t>(GetLastError()));
	}

	if (AddVectoredExceptionHandler(1, FatalFailfastVectoredHandler) == nullptr) {
		::printf("Warning: AddVectoredExceptionHandler(FatalFailfast) failed\n");
	} else {
		::printf("CrashTrace: FatalFailfast VEH installed\n");
	}

	const char* const modules[] = {"ucrtbase.dll", "vcruntime140.dll", "msvcrt.dll"};
	for (const char* module_name : modules) {
		const HMODULE module = GetModuleHandleA(module_name);
		if (module == nullptr) {
			continue;
		}
		const auto set_handler = reinterpret_cast<SetSecerrHandlerFunc>(
		    GetProcAddress(module, "_set_security_error_handler"));
		if (set_handler != nullptr) {
			(void)set_handler(&SecurityFailureHandler);
			return;
		}
	}
#endif
}

} // namespace Common
