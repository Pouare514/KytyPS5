#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/fatalLog.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "libs/agc.h"
#include "libs/errno.h"
#include "libs/guestPrintf.h"
#include "libs/libs.h"
#include "libs/vaContext.h"
#include "loader/runtimeLinker.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fmt/format.h>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs {

namespace LibKernel {
void KernelDispatchPendingSignalForCurrentThread();
} // namespace LibKernel

namespace LibC {

LIB_VERSION("libc", 1, "libc", 1, 1);

static uint32_t g_need_flag = 1;

using cxa_destructor_func_t = KYTY_SYSV_ABI void (*)(void*);
using atexit_func_t         = KYTY_SYSV_ABI void (*)();

struct CxaDestructor {
	cxa_destructor_func_t destructor_func;
	void*                 destructor_object;
	void*                 module_id;
};

struct CContext {
	std::list<CxaDestructor> cxa;
	std::list<atexit_func_t> atexit;
};

struct InitEnvParams {
	int         argc;
	uint32_t    pad;
	const char* argv[3];
};

static int                g_argc = 0;
static const char* const* g_argv = nullptr;
static const char* const* g_envp = nullptr;

int GetArgc() {
	return g_argc;
}

const char** GetArgv() {
	return const_cast<const char**>(g_argv);
}

static KYTY_SYSV_ABI void exit(int code) {
	// Avoid PRINT_NAME/LOGF here: may run on a misaligned guest stack (fmt movaps → silent death).
	fprintf(stderr, "GuestExit: libC::exit code=%d\n", code);
	std::fflush(stderr);
	char msg[96];
	std::snprintf(msg, sizeof(msg), "GuestExit: libC::exit code=%d", code);
	Common::LogFatalToFile(msg);

	// Boot/AGC: guest exit must not tear down the emulator. Opt-in real exit only.
	const char* allow = std::getenv("KYTY_ALLOW_GUEST_EXIT");
	if (allow != nullptr && allow[0] == '1') {
		::exit(code);
	}

	fprintf(stderr, "GuestExit: suppressing exit(%d) — soft-idle (set KYTY_ALLOW_GUEST_EXIT=1 to honor)\n",
	        code);
	std::fflush(stderr);
	Common::LogFatalToFile("GuestExit: libC::exit suppressed soft-idle");
	Libs::Graphics::WindowArmIgnoreQuit(120);
	for (int tick = 0;; ++tick) {
		std::this_thread::sleep_for(std::chrono::milliseconds(16));
		(void)Libs::LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
		if ((tick % 30) == 0) {
			(void)Libs::LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
			Libs::Graphics::WindowArmIgnoreQuit(60);
		}
	}
}

static void PrintAbortStringCandidate(const char* name, uint64_t addr) {
	if (!Graphics::HostMemoryIsReadable(addr)) {
		return;
	}

	const auto* ptr = reinterpret_cast<const char*>(static_cast<uintptr_t>(addr));
	char        buf[257] {};
	size_t      len = 0;

	for (; len < sizeof(buf) - 1 && Graphics::HostMemoryIsReadable(addr + len); len++) {
		const auto c = static_cast<unsigned char>(ptr[len]);
		if (c == 0) {
			break;
		}
		if (!std::isprint(c) && c != '\n' && c != '\r' && c != '\t') {
			return;
		}
		buf[len] = static_cast<char>(c);
	}

	if (len >= 4) {
		LOGF("\t %s string = \"%s\"\n", name, buf);
	}
}

static void PrintAbortWideStringCandidate(const char* name, uint64_t addr) {
	if (!Graphics::HostMemoryIsReadable(addr)) {
		return;
	}

	const auto* ptr = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(addr));
	char        buf[257] {};
	size_t      len = 0;

	for (;
	     len < sizeof(buf) - 1 && Graphics::HostMemoryIsReadable(addr + len * sizeof(uint16_t) + 1);
	     len++) {
		const auto c = ptr[len];
		if (c == 0) {
			break;
		}
		if (c > 0x7f ||
		    (!std::isprint(static_cast<unsigned char>(c)) && c != '\n' && c != '\r' && c != '\t')) {
			return;
		}
		buf[len] = static_cast<char>(c);
	}

	if (len >= 4) {
		LOGF("\t %s u16 = \"%s\"\n", name, buf);
	}
}

static void PrintAbortBytesCandidate(const char* name, uint64_t addr) {
	if (!Graphics::HostMemoryIsReadable(addr)) {
		return;
	}

	uint8_t bytes[64] {};
	size_t  len = 0;

	for (; len < sizeof(bytes) && Graphics::HostMemoryIsReadable(addr + len); len++) {
		bytes[len] = *reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(addr + len));
	}

	if (len == 0) {
		return;
	}

	LOGF("\t %s bytes =", name);
	for (size_t i = 0; i < len; i++) {
		LOGF(" %02" PRIx8, bytes[i]);
	}
	LOGF("\n");
}

static void PrintAbortPointerCandidate(const char* name, uint64_t addr) {
	PrintAbortStringCandidate(name, addr);
	PrintAbortWideStringCandidate(name, addr);
	PrintAbortBytesCandidate(name, addr);
}

static bool LooksLikeAbortPointer(uint64_t addr) {
	return Graphics::HostMemoryIsReadable(addr) && addr >= 0x10000;
}

static void PrintAbortPointerArrayCandidate(const char* name, uint64_t addr) {
	if (!LooksLikeAbortPointer(addr)) {
		return;
	}

	for (int i = 0; i < 8; i++) {
		const auto slot_addr = addr + static_cast<uint64_t>(i) * sizeof(uint64_t);
		if (!Graphics::HostMemoryIsReadable(slot_addr)) {
			break;
		}

		const auto value = *reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(slot_addr));
		if (!LooksLikeAbortPointer(value)) {
			continue;
		}

		LOGF("\t %s[%d] = 0x%016" PRIx64 "\n", name, i, value);
		const auto child_name = fmt::format("{}[{}]", name, i);
		PrintAbortPointerCandidate(child_name.c_str(), value);
	}
}

[[noreturn]] static KYTY_SYSV_ABI void abort(uint64_t arg0, uint64_t arg1, uint64_t arg2,
                                             uint64_t arg3, uint64_t arg4, uint64_t arg5) {
	PRINT_NAME();

	uint64_t rbp = 0;
	uint64_t rsp = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	asm volatile("movq %%rbp, %0" : "=r"(rbp));
	asm volatile("movq %%rsp, %0" : "=r"(rsp));
#endif

	const auto ret = reinterpret_cast<uint64_t>(__builtin_return_address(0));

	LOGF("Guest abort diagnostics:\n"
	     "\t return = 0x%016" PRIx64 "\n"
	     "\t rbp    = 0x%016" PRIx64 "\n"
	     "\t rsp    = 0x%016" PRIx64 "\n"
	     "\t arg0   = 0x%016" PRIx64 "\n"
	     "\t arg1   = 0x%016" PRIx64 "\n"
	     "\t arg2   = 0x%016" PRIx64 "\n"
	     "\t arg3   = 0x%016" PRIx64 "\n"
	     "\t arg4   = 0x%016" PRIx64 "\n"
	     "\t arg5   = 0x%016" PRIx64 "\n",
	     ret, rbp, rsp, arg0, arg1, arg2, arg3, arg4, arg5);

	PrintAbortPointerCandidate("arg0", arg0);
	PrintAbortPointerCandidate("arg1", arg1);
	PrintAbortPointerCandidate("arg2", arg2);
	PrintAbortPointerCandidate("arg3", arg3);
	PrintAbortPointerCandidate("arg4", arg4);
	PrintAbortPointerCandidate("arg5", arg5);
	PrintAbortPointerArrayCandidate("arg0", arg0);
	PrintAbortPointerArrayCandidate("arg1", arg1);
	PrintAbortPointerArrayCandidate("arg2", arg2);
	PrintAbortPointerArrayCandidate("arg3", arg3);
	PrintAbortPointerArrayCandidate("arg4", arg4);
	PrintAbortPointerArrayCandidate("arg5", arg5);

	if (rbp != 0) {
		Common::Singleton<Loader::RuntimeLinker>::Instance()->StackTrace(rbp);
	}

	if (rsp != 0) {
		LOGF("Guest abort stack words:\n");
		auto* linker = Common::Singleton<Loader::RuntimeLinker>::Instance();
		for (int i = 0; i < 16; i++) {
			const auto addr = rsp + static_cast<uint64_t>(i) * sizeof(uint64_t);
			if (!Graphics::HostMemoryIsReadable(addr)) {
				break;
			}
			const auto value   = *reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(addr));
			auto*      program = linker->FindProgramByAddr(value);
			if (program != nullptr) {
				auto module_name = Common::PathToString(program->file_name.filename());
				LOGF("\t [%02d] 0x%016" PRIx64 " %s+0x%016" PRIx64 "\n", i, value,
				     module_name.c_str(), value - program->base_vaddr);
			} else {
				LOGF("\t [%02d] 0x%016" PRIx64 "\n", i, value);
			}
			PrintAbortPointerCandidate(fmt::format("stack[{:02d}]", i).c_str(), value);
			PrintAbortPointerArrayCandidate(fmt::format("stack[{:02d}]", i).c_str(), value);
		}
	}

	EXIT("Guest abort()\n");
	std::abort();
}

static KYTY_SYSV_ABI int* libc_error() {
	PRINT_NAME();

	return Posix::GetErrorAddr();
}

static KYTY_SYSV_ABI void init_env(const InitEnvParams* params) {
	PRINT_NAME();

	if (params == nullptr) {
		g_argc = 0;
		g_argv = nullptr;
		g_envp = nullptr;
		return;
	}

	constexpr int argv_capacity = static_cast<int>(sizeof(params->argv) / sizeof(params->argv[0]));

	EXIT_NOT_IMPLEMENTED(params->argc < 0 || params->argc >= argv_capacity);

	g_argc = params->argc;
	g_argv = params->argv;
	g_envp = params->argv + params->argc + 1;

	LOGF("\t argc = %d\n"
	     "\t argv = 0x%016" PRIx64 "\n"
	     "\t envp = 0x%016" PRIx64 "\n",
	     g_argc, reinterpret_cast<uint64_t>(g_argv), reinterpret_cast<uint64_t>(g_envp));

	for (int i = 0; i < g_argc; i++) {
		LOGF("\t argv[%d] = %s\n", i, g_argv[i] != nullptr ? g_argv[i] : "<null>");
	}
}

static KYTY_SYSV_ABI int atexit(atexit_func_t func) {
	PRINT_NAME();

	if (func != nullptr) {
		Common::Singleton<CContext>::Instance()->atexit.push_front(func);
	}

	return 0;
}

static KYTY_SYSV_ABI int libc_printf(VA_ARGS) {
	VA_CONTEXT(ctx); // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)

	PRINT_NAME();

	const auto* format = reinterpret_cast<const char*>(rdi);
	if (format != nullptr &&
	    (std::strstr(format, "Code and Data section is too large") != nullptr ||
	     std::strstr(format, "RAM Cache is being disabled") != nullptr)) {
		double d0 = 0.0;
		double d1 = 0.0;
		std::memcpy(&d0, &xmm0, sizeof(d0));
		std::memcpy(&d1, &xmm1, sizeof(d1));
		uint64_t d0_bits = 0;
		uint64_t d1_bits = 0;
		std::memcpy(&d0_bits, &d0, sizeof(d0_bits));
		std::memcpy(&d1_bits, &d1, sizeof(d1_bits));
		LOGF("NdMemoryPrintf libc_printf regs: rdi=0x%016" PRIx64 " rsi=0x%016" PRIx64
		     " rdx=0x%016" PRIx64 " rcx=0x%016" PRIx64 " r8=0x%016" PRIx64 " r9=0x%016" PRIx64
		     " xmm0=%.6g/0x%016" PRIx64 " xmm1=%.6g/0x%016" PRIx64 "\n",
		     rdi, rsi, rdx, rcx, r8, r9, d0, d0_bits, d1, d1_bits);
		// TLOU InitializeMemory guest globals (eboot PPSA03396 @ base 0x900000000).
		constexpr uint64_t kNdTablePtrGlobal = 0x0000000904b5a520ull;
		constexpr uint64_t kNdCountGlobal    = 0x0000000904b9a8e0ull;
		constexpr uint64_t kNdEntry6Global   = 0x0000000904b1a3e8ull;
		constexpr uint64_t kNdMeasuredGlobal = 0x0000000904b1a3f0ull;
		const auto* table_ptr_cell = reinterpret_cast<const uint64_t*>(kNdTablePtrGlobal);
		const auto* count_cell     = reinterpret_cast<const uint32_t*>(kNdCountGlobal);
		const auto* entry6_cell    = reinterpret_cast<const uint64_t*>(kNdEntry6Global);
		const auto* measured_cell  = reinterpret_cast<const uint64_t*>(kNdMeasuredGlobal);
		const uint64_t table_ptr   = *table_ptr_cell;
		const uint32_t count       = *count_cell;
		const uint64_t entry6      = *entry6_cell;
		const uint64_t measured    = *measured_cell;
		LOGF("NdMemoryPrintf ND globals: table_ptr=0x%016" PRIx64 " count=%u entry6=0x%016" PRIx64
		     " measured=0x%016" PRIx64 " (%.3f MiB raw/2^20)\n",
		     table_ptr, count, entry6, measured,
		     static_cast<double>(measured) / (1024.0 * 1024.0));
		{
			constexpr uint64_t kNdObjectGlobal = 0x0000000904ada0e0ull;
			constexpr uint64_t kNdFnPtrGlobal  = 0x000000090493c478ull;
			const uint64_t     object          = *reinterpret_cast<const uint64_t*>(kNdObjectGlobal);
			const uint64_t     fn_cell         = *reinterpret_cast<const uint64_t*>(kNdFnPtrGlobal);
			const uint64_t     fn              =
			    fn_cell != 0 ? *reinterpret_cast<const uint64_t*>(fn_cell) : 0;
			const uint64_t second =
			    (measured + entry6 + 0x98e60000ull); // approx; refined below if avail known
			LOGF("NdMemoryPrintf ND call: object=0x%016" PRIx64 " fn_cell=0x%016" PRIx64
			     " fn=0x%016" PRIx64 "\n",
			     object, fn_cell, fn);
			if (object != 0) {
				const auto* o = reinterpret_cast<const uint64_t*>(object);
				LOGF("NdMemoryPrintf ND object[0..7]: %016" PRIx64 " %016" PRIx64 " %016" PRIx64
				     " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 " %016" PRIx64 "\n",
				     o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
			}
			(void)second;
		}
		if (table_ptr != 0) {
			// Entries start at table+0x20; walk until id==0 (guest terminator), max 16.
			const auto* obj = reinterpret_cast<const uint64_t*>(table_ptr + 0x20);
			uint64_t    total = 0;
			for (uint32_t i = 0; i < 16; i++) {
				const auto* e  = obj + static_cast<size_t>(i) * 4;
				const auto  id = static_cast<uint32_t>(e[0] & 0xffffffffu);
				if (id == 0) {
					LOGF("NdMemoryPrintf desc end at [%u]\n", i);
					break;
				}
				const auto size = e[1];
				total += size;
				const char* name = reinterpret_cast<const char*>(e[3]);
				LOGF("NdMemoryPrintf desc[%u]: id=%u hdr=0x%016" PRIx64 " size=0x%016" PRIx64
				     " field2=0x%016" PRIx64 " name=%s\n",
				     i, id, e[0], size, e[2], name != nullptr ? name : "<null>");
			}
			LOGF("NdMemoryPrintf desc_total_size=0x%016" PRIx64 " (%.3f MiB) limit_rsi=0x%016" PRIx64
			     "\n",
			     total, static_cast<double>(total) / (1024.0 * 1024.0), rsi);
		}
	}

	return GetGuestPrintfCtxFunc()(&ctx);
}

static KYTY_SYSV_ABI int puts(const char* s) {
	PRINT_NAME();

	return GetGuestPrintfStdFunc()("%s\n", s);
}

static KYTY_SYSV_ABI int setenv(const char* name, const char* value, int overwrite) {
	PRINT_NAME();

	LOGF("\t name      = %s\n"
	     "\t value     = %s\n"
	     "\t overwrite = %d\n",
	     (name != nullptr ? name : "<null>"), (value != nullptr ? value : "<null>"), overwrite);

	if (name == nullptr || value == nullptr || name[0] == '\0' ||
	    std::strchr(name, '=') != nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}

	if (overwrite == 0 && std::getenv(name) != nullptr) {
		return 0;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto ret = ::_putenv_s(name, value);
	if (ret != 0) {
		*Posix::GetErrorAddr() = ret;
		return -1;
	}
	return 0;
#else
	const auto ret = ::setenv(name, value, overwrite);
	if (ret != 0) {
		*Posix::GetErrorAddr() = errno;
		return -1;
	}
	return 0;
#endif
}

static KYTY_SYSV_ABI int64_t libc_time(int64_t* timer) {
	const auto now = static_cast<int64_t>(std::time(nullptr));
	if (timer != nullptr) {
		*timer = now;
	}
	return now;
}

static KYTY_SYSV_ABI double libc_difftime(int64_t time1, int64_t time0) {
	return std::difftime(static_cast<std::time_t>(time1), static_cast<std::time_t>(time0));
}

static KYTY_SYSV_ABI std::tm* libc_gmtime(const int64_t* timer) {
	if (timer == nullptr) {
		return nullptr;
	}

	thread_local std::tm result {};
	const auto           t = static_cast<std::time_t>(*timer);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (_gmtime64_s(&result, &t) != 0) {
		return nullptr;
	}
#else
	if (gmtime_r(&t, &result) == nullptr) {
		return nullptr;
	}
#endif

	return &result;
}

static KYTY_SYSV_ABI std::tm* libc_localtime(const int64_t* timer) {
	if (timer == nullptr) {
		return nullptr;
	}

	thread_local std::tm result {};
	const auto           t = static_cast<std::time_t>(*timer);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (_localtime64_s(&result, &t) != 0) {
		return nullptr;
	}
#else
	if (localtime_r(&t, &result) == nullptr) {
		return nullptr;
	}
#endif

	return &result;
}

static KYTY_SYSV_ABI int64_t libc_mktime(std::tm* timeptr) {
	if (timeptr == nullptr) {
		return -1;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return static_cast<int64_t>(_mktime64(timeptr));
#else
	return static_cast<int64_t>(std::mktime(timeptr));
#endif
}

static KYTY_SYSV_ABI size_t libc_strftime(char* str, size_t count, const char* format,
                                          const std::tm* timeptr) {
	if (str == nullptr || format == nullptr || timeptr == nullptr) {
		return 0;
	}

	return std::strftime(str, count, format, timeptr);
}

// Phase 33 survival: divert MainThread forever (opt-in via KYTY_PHASE33_DIVERT=1).
static void Phase33DivertPumpForever() {
	for (int tick = 0;; ++tick) {
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		(void)Libs::LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
		if ((tick % 20) == 0) {
			(void)Libs::LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
		}
	}
}

// Phase 34/47: MainThread soft-idle — keep menu flips alive after guest_draw park.
static void Phase34MainThreadSoftIdle() {
	LOGF("GuestExit: catchReturnFromMain — phase47 soft-idle (sustain flips forever)\n");
	fprintf(stderr, "GuestExit: catchReturnFromMain — phase47 soft-idle\n");
	std::fflush(stderr);
	Common::LogFatalToFile("catchReturnFromMain phase47 soft-idle");
	for (int tick = 0;; ++tick) {
		std::this_thread::sleep_for(std::chrono::milliseconds(33));
		// Always handoff/sustain — stopping after guest_draw froze presented≈60.
		Libs::VideoOut::Phase41MenuHandoffAttempt();
		Libs::VideoOut::Phase38NudgeBootWorkers();
		if (Libs::VideoOut::Phase46ReadyForSoftIdle() &&
		    !Libs::VideoOut::Phase47GuestDrawSeen()) {
			Libs::VideoOut::Phase47PostSeedWake("mainthread-softidle");
		}
		(void)Libs::LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
		if ((tick % 30) == 0) {
			(void)Libs::LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
			Libs::Graphics::WindowArmIgnoreQuit(60);
			if (!Libs::VideoOut::Phase47ReadyForSoftIdle() && (tick % 300) == 0) {
				LOGF("GuestExit: phase47 soft-idle still waiting for guest_draw (%ds)\n",
				     tick / 30);
				fprintf(stderr, "GuestExit: phase47 still waiting guest_draw (%ds)\n", tick / 30);
			} else if (Libs::VideoOut::Phase47ReadyForSoftIdle() && (tick % 300) == 0) {
				LOGF("GuestExit: phase47 soft-idle sustain tick=%d\n", tick);
				fprintf(stderr, "GuestExit: phase47 soft-idle sustain tick=%d\n", tick);
			}
		}
	}
}

// Phase 47: anti-CRT divert — handoff until menu frames + guest DRAW/DISPATCH.
static void Phase38DeferredSoftIdle() {
	LOGF("GuestExit: phase47 anti-CRT divert — until menu_frames_ok + guest_draw\n");
	fprintf(stderr, "GuestExit: phase47 anti-CRT divert — until menu_frames + guest_draw\n");
	std::fflush(stderr);
	Common::LogFatalToFile("phase47 anti-CRT divert");
	for (int i = 0;; ++i) {
		const bool frames = Libs::VideoOut::Phase38GuestBootProgressSeen();
		const bool p47    = Libs::VideoOut::Phase47ReadyForSoftIdle();
		if (frames && p47) {
			LOGF("GuestExit: phase47 menu_frames+guest_draw at %dms — soft-idle park\n",
			     i * 100);
			fprintf(stderr, "GuestExit: phase47 ready — soft-idle park\n");
			break;
		}
		if (i == 20) {
			Libs::LibKernel::PthreadDumpAllGuestThreads("phase47 anti-CRT +2s");
		}
		if (i > 0 && (i % 300) == 0) {
			LOGF("GuestExit: phase47 still waiting frames=%d guest_draw=%d (%ds) — NOT soft-idle\n",
			     frames ? 1 : 0, p47 ? 1 : 0, i / 10);
			fprintf(stderr, "GuestExit: phase47 still waiting frames=%d guest_draw=%d (%ds)\n",
			        frames ? 1 : 0, p47 ? 1 : 0, i / 10);
			Common::LogFatalToFile("phase47 still waiting for frames+guest_draw");
		}
		Libs::VideoOut::Phase41MenuHandoffAttempt();
		Libs::VideoOut::Phase38NudgeBootWorkers();
		if (Libs::VideoOut::Phase46ReadyForSoftIdle() && !p47) {
			Libs::VideoOut::Phase47PostSeedWake("anti-crt-divert");
		}
		Libs::Graphics::WindowArmIgnoreQuit(60);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	Phase34MainThreadSoftIdle();
}

// Kick a non-empty Submit + SubmitFlip(0) from MainThread (opt-in KYTY_PHASE33_KICK=1).
static void Phase33KickGuestSubmitAndFlip() {
	static std::atomic<bool> once {false};
	if (once.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	// Two type-2 padding words — graphicsRun requires dw>=2 for type-3 packets.
	static uint32_t dcb_storage[8] = {0x80000000u, 0x80000000u, 0, 0, 0, 0, 0, 0};
	const uint32_t  dcb_size       = 2;
	LOGF("SubmitTrace: phase33 KickSubmit (catchReturn) dcb=%p size=0x%x\n",
	     static_cast<void*>(dcb_storage), dcb_size);
	fprintf(stderr, "SubmitTrace: phase33 KickSubmit (catchReturn) size=0x%x\n", dcb_size);
	(void)Libs::Graphics::Gen5Driver::GraphicsDriverSubmitCommandBuffer(0, dcb_storage, dcb_size);
	const int flip_result =
	    Libs::VideoOut::VideoOutSubmitFlip(1, 0, /*VIDEO_OUT_FLIP_MODE_VSYNC*/ 1, 0);
	LOGF("FlipTrace: phase33 KickFlip0 result=%d (no PENDING0 bridge)\n", flip_result);
	fprintf(stderr, "FlipTrace: phase33 KickFlip0 result=%d\n", flip_result);
	(void)Libs::LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
}

static KYTY_SYSV_ABI void catchReturnFromMain(int status) {
	PRINT_NAME();

	uint64_t ret_addr = reinterpret_cast<uint64_t>(__builtin_return_address(0));
	uint64_t rsp      = 0;
	uint64_t rbp      = 0;
#if defined(__x86_64__) || defined(_M_X64)
	asm volatile("movq %%rsp, %0\n\t"
	             "movq %%rbp, %1\n\t"
	             : "=r"(rsp), "=r"(rbp));
#endif

	LOGF("\t status     = %d\n"
	     "\t ret_addr   = 0x%016" PRIx64 "\n"
	     "\t rsp        = 0x%016" PRIx64 "\n"
	     "\t rbp        = 0x%016" PRIx64 "\n",
	     status, ret_addr, rsp, rbp);
	if (rsp != 0) {
		auto* stack = reinterpret_cast<const uint64_t*>(rsp);
		for (uint32_t i = 0; i < 12; i++) {
			LOGF("\t stack[%02" PRIu32 "] = 0x%016" PRIx64 "\n", i, stack[i]);
		}
	}

	::printf("return from main = %d\n", status);
	std::fflush(stdout);

	// CRT calls exit() then ud2 after this returns — never return (TLOU boot soft-idle).
	const char* kick   = std::getenv("KYTY_PHASE33_KICK");
	const char* divert = std::getenv("KYTY_PHASE33_DIVERT");
	if (kick != nullptr && kick[0] == '1') {
		Phase33KickGuestSubmitAndFlip();
	}
	if (divert != nullptr && divert[0] == '1') {
		LOGF("GuestExit: catchReturnFromMain status=%d — phase33 divert\n", status);
		fprintf(stderr, "GuestExit: catchReturnFromMain status=%d — phase33 divert\n", status);
		std::fflush(stderr);
		Common::LogFatalToFile("catchReturnFromMain phase33 divert");
		Phase33DivertPumpForever();
	}
	fprintf(stderr, "GuestExit: catchReturnFromMain status=%d — soft-idle\n", status);
	std::fflush(stderr);
	Common::LogFatalToFile("catchReturnFromMain soft-idle");
	Libs::Graphics::WindowArmIgnoreQuit(120);
	Phase38DeferredSoftIdle();
}

enum class ThrdResult : int {
	Success  = 0,
	NoMem    = 1,
	TimedOut = 2,
	Busy     = 3,
	Error    = 4,
	Perm     = 5,
};

static int mtx_result(int result) {
	return (result == OK ? static_cast<int>(ThrdResult::Success)
	                     : static_cast<int>(ThrdResult::Error));
}

static KYTY_SYSV_ABI int MtxInit(LibKernel::PthreadMutex* mtx, int type) {
	PRINT_NAME();

	if (mtx == nullptr) {
		return static_cast<int>(ThrdResult::Error);
	}

	constexpr int mtx_recursive = 0x100;
	if ((type & mtx_recursive) == 0) {
		return mtx_result(LibKernel::PthreadMutexInit(mtx, nullptr));
	}

	LibKernel::PthreadMutexattr attr   = nullptr;
	int                         result = LibKernel::PthreadMutexattrInit(&attr);
	result = (result == OK ? LibKernel::PthreadMutexattrSettype(&attr, 2) : result);
	result = (result == OK ? LibKernel::PthreadMutexInit(mtx, &attr) : result);
	(void)LibKernel::PthreadMutexattrDestroy(&attr);

	return mtx_result(result);
}

static KYTY_SYSV_ABI int MtxInitWithName(LibKernel::PthreadMutex* mtx, int type,
                                         const char* /*name*/) {
	PRINT_NAME();

	return MtxInit(mtx, type);
}

static KYTY_SYSV_ABI int MtxInitWithDefaultNameOverride(LibKernel::PthreadMutex* mtx, int type,
                                                        const char* /*name*/) {
	PRINT_NAME();

	return MtxInit(mtx, type);
}

static KYTY_SYSV_ABI void MtxDestroy(LibKernel::PthreadMutex* mtx) {
	PRINT_NAME();

	if (mtx != nullptr) {
		(void)LibKernel::PthreadMutexDestroy(mtx);
	}
}

static KYTY_SYSV_ABI int MtxLock(LibKernel::PthreadMutex* mtx) {
	PRINT_NAME();

	return mtx_result(LibKernel::PthreadMutexLock(mtx));
}

static KYTY_SYSV_ABI int MtxTrylock(LibKernel::PthreadMutex* mtx) {
	PRINT_NAME();

	const auto result = LibKernel::PthreadMutexTrylock(mtx);
	if (result == OK) {
		return static_cast<int>(ThrdResult::Success);
	}
	return static_cast<int>(ThrdResult::Busy);
}

static KYTY_SYSV_ABI int MtxTimedlock(LibKernel::PthreadMutex* mtx, const void* /*xtime*/) {
	PRINT_NAME();

	return mtx_result(LibKernel::PthreadMutexLock(mtx));
}

static KYTY_SYSV_ABI int MtxUnlock(LibKernel::PthreadMutex* mtx) {
	PRINT_NAME();

	return mtx_result(LibKernel::PthreadMutexUnlock(mtx));
}

static KYTY_SYSV_ABI int MtxCurrentOwns(LibKernel::PthreadMutex* /*mtx*/) {
	PRINT_NAME();

	return 0;
}

using execute_once_func_t = KYTY_SYSV_ABI int (*)(void*, void*, void**);

static std::mutex              g_execute_once_mutex;
static std::condition_variable g_execute_once_cv;

static KYTY_SYSV_ABI int std_execute_once(int* flag, execute_once_func_t func, void* arg) {
	PRINT_NAME();

	if (flag == nullptr || func == nullptr) {
		return 0;
	}

	constexpr int once_init    = 0;
	constexpr int once_done    = 1;
	constexpr int once_running = 2;

	{
		std::unique_lock lock(g_execute_once_mutex);
		while (*flag == once_running) {
			g_execute_once_cv.wait_for(lock, std::chrono::microseconds(10000));
			if (*flag == once_running) {
				lock.unlock();
				LibKernel::KernelDispatchPendingSignalForCurrentThread();
				lock.lock();
			}
		}
		if (*flag == once_done) {
			return 1;
		}
		*flag = once_running;
	}

	void* callback_context = nullptr;
	int   result           = func(nullptr, arg, &callback_context);

	{
		std::lock_guard lock(g_execute_once_mutex);
		*flag = (result != 0 ? once_done : once_init);
	}
	g_execute_once_cv.notify_all();

	return result;
}

static KYTY_SYSV_ABI int cxa_atexit(cxa_destructor_func_t func, void* arg, void* d) {
	PRINT_NAME();

	auto* cc = Common::Singleton<CContext>::Instance();

	CxaDestructor c {};
	c.destructor_func   = func;
	c.destructor_object = arg;
	c.module_id         = d;

	cc->cxa.push_back(c);

	return 0;
}

void KYTY_SYSV_ABI cxa_finalize(void* d) {
	PRINT_NAME();

	auto* cc = Common::Singleton<CContext>::Instance();

	for (auto i = cc->cxa.rbegin(); i != cc->cxa.rend(); ++i) {
		auto& c = *i;
		if ((d == nullptr || c.module_id == d) && c.destructor_func != nullptr) {
			auto* func        = c.destructor_func;
			auto* object      = c.destructor_object;
			c.destructor_func = nullptr;
			func(object);
		}
	}
}

} // namespace LibC

namespace LibcInternalExt {

LIB_VERSION("LibcInternalExt", 1, "LibcInternal", 1, 1);

static uint64_t g_mspace_atomic_id_mask = 0;
static uint64_t g_mstate_table[64]      = {0};

struct Info {
	uint64_t  size;
	uint32_t  unknown1;
	uint32_t  unknown2;
	uint64_t* mspace_atomic_id_mask;
	uint64_t* mstate_table;
};

void KYTY_SYSV_ABI LibcHeapGetTraceInfo(Info* info) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(info->size != 32);

	info->mspace_atomic_id_mask = &g_mspace_atomic_id_mask;
	info->mstate_table          = g_mstate_table;
}

uint64_t KYTY_SYSV_ABI LibcInternalExtUnknownQBS714Jr3g(uint64_t arg0, uint64_t arg1, uint64_t arg2,
                                                        uint64_t arg3, uint64_t arg4,
                                                        uint64_t arg5) {
	PRINT_NAME();

	LOGF("\t arg0 = 0x%016" PRIx64 "\n"
	     "\t arg1 = 0x%016" PRIx64 "\n"
	     "\t arg2 = 0x%016" PRIx64 "\n"
	     "\t arg3 = 0x%016" PRIx64 "\n"
	     "\t arg4 = 0x%016" PRIx64 "\n"
	     "\t arg5 = 0x%016" PRIx64 "\n",
	     arg0, arg1, arg2, arg3, arg4, arg5);

	return 0;
}

int KYTY_SYSV_ABI LibcInternalExtMallocInit() {
	PRINT_NAME();
	return 0;
}

LIB_DEFINE(InitLibcInternalExt_1) {
	LIB_FUNC("NWtTN10cJzE", LibcInternalExt::LibcHeapGetTraceInfo);
	LIB_FUNC("qBS714-Jr3g", LibcInternalExt::LibcInternalExtUnknownQBS714Jr3g);
	LIB_FUNC("EHsF2i9FXPM", LibcInternalExt::LibcInternalExtMallocInit);
}

} // namespace LibcInternalExt

namespace LibcInternal {

LIB_VERSION("LibcInternal", 1, "LibcInternal", 1, 1);

static uint32_t g_need_flag = 1;

int KYTY_SYSV_ABI vprintf(const char* str, VaList* c) {
	PRINT_NAME();

	return GetGuestVprintfFunc()(str, c);
}

static KYTY_SYSV_ABI int snprintf(VA_ARGS) {
	VA_CONTEXT(ctx); // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)

	PRINT_NAME();

	return GetGuestSnprintfCtxFunc()(&ctx);
}

int KYTY_SYSV_ABI fflush(FILE* stream) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(stream != stdout);

	return ::fflush(stream);
}

void* KYTY_SYSV_ABI memset(void* s, int c, size_t n) {
	PRINT_NAME();

	return ::memset(s, c, n);
}

void* KYTY_SYSV_ABI memcpy(void* dest, const void* src, size_t n) {
	return ::memcpy(dest, src, n);
}

void* KYTY_SYSV_ABI memmove(void* dest, const void* src, size_t n) {
	return ::memmove(dest, src, n);
}

int KYTY_SYSV_ABI memcmp(const void* s1, const void* s2, size_t n) {
	return ::memcmp(s1, s2, n);
}

int KYTY_SYSV_ABI strcmp(const char* s1, const char* s2) {
	return ::strcmp(s1, s2);
}

int KYTY_SYSV_ABI strncmp(const char* s1, const char* s2, size_t n) {
	return ::strncmp(s1, s2, n);
}

size_t KYTY_SYSV_ABI strlen(const char* s) {
	return ::strlen(s);
}

char* KYTY_SYSV_ABI strcpy(char* dest, const char* src) {
	return ::strcpy(dest, src);
}

char* KYTY_SYSV_ABI strncpy(char* dest, const char* src, size_t count) {
	return ::strncpy(dest, src, count);
}

char* KYTY_SYSV_ABI strcat(char* dest, const char* src) {
	return ::strcat(dest, src);
}

const char* KYTY_SYSV_ABI strchr(const char* s, int c) {
	return ::strchr(s, c);
}

char* KYTY_SYSV_ABI strrchr(const char* s, int c) {
	return const_cast<char*>(::strrchr(s, c));
}

char* KYTY_SYSV_ABI strstr(const char* haystack, const char* needle) {
	static int log_count = 0;
	if (log_count++ < 8) {
		LOGF("LibcInternal::strstr(\"%s\", \"%s\")\n", (haystack != nullptr ? haystack : "<null>"),
		     (needle != nullptr ? needle : "<null>"));
	}
	return const_cast<char*>(::strstr(haystack, needle));
}

long KYTY_SYSV_ABI strtol(const char* str, char** endptr, int base) {
	static int log_count = 0;
	if (log_count++ < 16) {
		LOGF("LibcInternal::strtol(\"%s\", base=%d)\n", (str != nullptr ? str : "<null>"), base);
	}
	return ::strtol(str, endptr, base);
}

unsigned long KYTY_SYSV_ABI strtoul(const char* str, char** endptr, int base) {
	static int log_count = 0;
	if (log_count++ < 16) {
		LOGF("LibcInternal::strtoul(\"%s\", base=%d)\n", (str != nullptr ? str : "<null>"), base);
	}
	return ::strtoul(str, endptr, base);
}

int KYTY_SYSV_ABI atoi(const char* str) {
	return ::atoi(str);
}

float KYTY_SYSV_ABI sinf(float x) {
	return std::sinf(x);
}

float KYTY_SYSV_ABI cosf(float x) {
	return std::cosf(x);
}

void KYTY_SYSV_ABI sincosf(float x, float* sinp, float* cosp) {
	if (sinp != nullptr) {
		*sinp = std::sinf(x);
	}
	if (cosp != nullptr) {
		*cosp = std::cosf(x);
	}
}

double KYTY_SYSV_ABI sin(double x) {
	return std::sin(x);
}

double KYTY_SYSV_ABI cos(double x) {
	return std::cos(x);
}

void KYTY_SYSV_ABI sincos(double x, double* sinp, double* cosp) {
	if (sinp != nullptr) {
		*sinp = std::sin(x);
	}
	if (cosp != nullptr) {
		*cosp = std::cos(x);
	}
}

int KYTY_SYSV_ABI LibcHeapErrorReportForGame(uint64_t msp, uint64_t ptr, uint64_t error,
                                             uint64_t arg3, uint64_t arg4, uint64_t arg5) {
	PRINT_NAME();

	LOGF("\t temporary: heap error report ignored, msp=0x%016" PRIx64 ", ptr=0x%016" PRIx64
	     ", error=0x%016" PRIx64 ", args=(0x%016" PRIx64 ",0x%016" PRIx64 ",0x%016" PRIx64 ")\n",
	     msp, ptr, error, arg3, arg4, arg5);

	return 0;
}

LIB_DEFINE(InitLibcInternal_1) {
	LibcInternalExt::InitLibcInternalExt_1(s);

	LIB_OBJECT("ZT4ODD2Ts9o", &LibcInternal::g_need_flag);
	LIB_OBJECT("2sWzhYqFH4E", stdout);

	LIB_FUNC("GMpvxPFW924", LibcInternal::vprintf);
	LIB_FUNC("MUjC4lbHrK4", LibcInternal::fflush);
	LIB_FUNC("8zTFvBIAIN8", LibcInternal::memset);
	LIB_FUNC("Q3VBxCXhUHs", LibcInternal::memcpy);
	LIB_FUNC("+P6FRGH4LfA", LibcInternal::memmove);
	LIB_FUNC("DfivPArhucg", LibcInternal::memcmp);
	LIB_FUNC("aesyjrHVWy4", LibcInternal::strcmp);
	LIB_FUNC("Ovb2dSJOAuE", LibcInternal::strncmp);
	LIB_FUNC("j4ViWNHEgww", LibcInternal::strlen);
	LIB_FUNC("5Xa2ACNECdo", LibcInternal::strcpy);
	LIB_FUNC("6sJWiWSRuqk", LibcInternal::strncpy);
	LIB_FUNC("Ls4tzzhimqQ", LibcInternal::strcat);
	LIB_FUNC("ob5xAW4ln-0", LibcInternal::strchr);
	LIB_FUNC("9yDWMxEFdJU", LibcInternal::strrchr);
	LIB_FUNC("viiwFMaNamA", LibcInternal::strstr);
	LIB_FUNC("mXlxhmLNMPg", LibcInternal::strtol);
	LIB_FUNC("QxmSHBCuKTk", LibcInternal::strtoul);
	LIB_FUNC("zlfEH8FmyUA", LibcInternal::strtoul);
	LIB_FUNC("fPxypibz2MY", LibcInternal::atoi);
	LIB_FUNC("Q4rRL34CEeE", LibcInternal::sinf);
	LIB_FUNC("-P6FNMzk2Kc", LibcInternal::cosf);
	LIB_FUNC("pztV4AF18iI", LibcInternal::sincosf);
	LIB_FUNC("H8ya2H00jbI", LibcInternal::sin);
	LIB_FUNC("2WE3BTYVwKM", LibcInternal::cos);
	LIB_FUNC("jMB7EFyu30Y", LibcInternal::sincos);
	LIB_FUNC("eLdDw6l0-bU", LibcInternal::snprintf);

	LIB_FUNC("L1SBTkC+Cvw", LibC::abort);
	LIB_FUNC("tsvEmnenz48", LibC::cxa_atexit);
	LIB_FUNC("H2e8t5ScQGc", LibC::cxa_finalize);
	LIB_FUNC("DiGVep5yB5w", LibC::std_execute_once);

	LIB_FUNC("al3JzFI9MQ0", LibcInternal::LibcHeapErrorReportForGame);
}

} // namespace LibcInternal

LIB_USING(LibC);

LIB_DEFINE(InitLibC_1) {
	LibcInternal::InitLibcInternal_1(s);

	LIB_OBJECT("P330P3dFF68", &LibC::g_need_flag);

	LIB_FUNC("uMei1W9uyNo", LibC::exit);
	LIB_FUNC("L1SBTkC+Cvw", LibC::abort);
	LIB_FUNC("9BcDykPmo1I", LibC::libc_error);
	LIB_FUNC("bzQExy189ZI", LibC::init_env);
	LIB_FUNC("8G2LB+A3rzg", LibC::atexit);
	LIB_FUNC("hcuQgD53UxM", LibC::libc_printf);
	LIB_FUNC("YQ0navp+YIc", LibC::puts);
	LIB_FUNC("M4YYbSFfJ8g", LibC::setenv);
	LIB_FUNC("wLlFkwG9UcQ", LibC::libc_time);
	LIB_FUNC("-VVn74ZyhEs", LibC::libc_difftime);
	LIB_FUNC("1mecP7RgI2A", LibC::libc_gmtime);
	LIB_FUNC("efhK-YSUYYQ", LibC::libc_localtime);
	LIB_FUNC("n7AepwR0s34", LibC::libc_mktime);
	LIB_FUNC("Av3zjWi64Kw", LibC::libc_strftime);
	LIB_FUNC("XKRegsFpEpk", LibC::catchReturnFromMain);
	LIB_FUNC("tsvEmnenz48", LibC::cxa_atexit);
	LIB_FUNC("H2e8t5ScQGc", LibC::cxa_finalize);
	LIB_FUNC("DiGVep5yB5w", LibC::std_execute_once);
	LIB_FUNC("YaHc3GS7y7g", LibC::MtxInit);
	LIB_FUNC("tgioGpKtmbE", LibC::MtxInitWithName);
	LIB_FUNC("JHp7ogc1+HY", LibC::MtxInitWithDefaultNameOverride);
	LIB_FUNC("5Lf51jvohTQ", LibC::MtxDestroy);
	LIB_FUNC("iS4aWbUonl0", LibC::MtxLock);
	LIB_FUNC("k6pGNMwJB08", LibC::MtxTrylock);
	LIB_FUNC("hPzYSd5Nasc", LibC::MtxTimedlock);
	LIB_FUNC("gTuXQwP9rrs", LibC::MtxUnlock);
	LIB_FUNC("VYQwFs4CC4Y", LibC::MtxCurrentOwns);
}

} // namespace Libs
