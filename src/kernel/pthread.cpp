#include "kernel/pthread.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/dateTime.h"
#include "common/emulatorConfig.h"
#include "common/fatalLog.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "loader/runtimeLinker.h"
#include "loader/timer.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#if defined(_M_X64) || defined(__x86_64__)
#include <intrin.h>
#endif
#include <windows.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <pthread.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <fmt/format.h>
#include <pthread_time.h>
#endif

#ifdef pthread_attr_getguardsize
#undef pthread_attr_getguardsize
#endif

namespace Libs {

namespace LibKernel {

LIB_NAME("libkernel", "libkernel");

constexpr int      KEYS_MAX                  = 256;
constexpr int      DESTRUCTOR_ITERATIONS     = 4;
constexpr size_t   PTHREAD_STACK_DEFAULT     = 0x100000;
constexpr size_t   PTHREAD_STACK_MIN         = 0x4000;
constexpr size_t   PTHREAD_STACK_PAGE        = 0x4000;
constexpr size_t   PTHREAD_STACK_GRANULARITY = 0x10000;
constexpr size_t   PTHREAD_STACK_INITIAL     = 0x200000;
constexpr size_t   PTHREAD_STACK_EXTRA       = 0x100000;
constexpr uint64_t PTHREAD_STACK_TOP         = 0x7efff8000ull;
constexpr uint32_t SIGNAL_APC_POLL_MICROS    = 10000;

static constexpr KernelClockid KERNEL_CLOCK_REALTIME          = 0;
static constexpr KernelClockid KERNEL_CLOCK_VIRTUAL           = 1;
static constexpr KernelClockid KERNEL_CLOCK_PROF              = 2;
static constexpr KernelClockid KERNEL_CLOCK_MONOTONIC         = 4;
static constexpr KernelClockid KERNEL_CLOCK_UPTIME            = 5;
static constexpr KernelClockid KERNEL_CLOCK_UPTIME_PRECISE    = 7;
static constexpr KernelClockid KERNEL_CLOCK_UPTIME_FAST       = 8;
static constexpr KernelClockid KERNEL_CLOCK_REALTIME_PRECISE  = 9;
static constexpr KernelClockid KERNEL_CLOCK_REALTIME_FAST     = 10;
static constexpr KernelClockid KERNEL_CLOCK_MONOTONIC_PRECISE = 11;
static constexpr KernelClockid KERNEL_CLOCK_MONOTONIC_FAST    = 12;
static constexpr KernelClockid KERNEL_CLOCK_SECOND            = 13;
static constexpr KernelClockid KERNEL_CLOCK_THREAD_CPUTIME_ID = 14;
static constexpr KernelClockid KERNEL_CLOCK_PROCTIME          = 15;
static constexpr KernelClockid KERNEL_CLOCK_EXT_NETWORK       = 16;
static constexpr KernelClockid KERNEL_CLOCK_EXT_DEBUG_NETWORK = 17;
static constexpr KernelClockid KERNEL_CLOCK_EXT_AD_NETWORK    = 18;
static constexpr KernelClockid KERNEL_CLOCK_EXT_RAW_NETWORK   = 19;

static constexpr int KERNEL_PTHREAD_MUTEX_ERRORCHECK = 1;
static constexpr int KERNEL_PTHREAD_MUTEX_RECURSIVE  = 2;
static constexpr int KERNEL_PTHREAD_MUTEX_NORMAL     = 3;
static constexpr int KERNEL_PTHREAD_MUTEX_ADAPTIVE   = 4;

static uint64_t KernelReadTscNative() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS && (defined(_M_X64) || defined(__x86_64__))
	return __rdtsc();
#else
	return Common::Timer::QueryPerformanceCounter();
#endif
}

static uint64_t KernelGetTscFrequencyNative() {
	static const uint64_t frequency = [] {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS && (defined(_M_X64) || defined(__x86_64__))
		const auto host_frequency = Common::Timer::QueryPerformanceFrequency();
		if (host_frequency == 0) {
			return uint64_t {1000000000};
		}

		KernelReadTscNative();
		Common::Thread::Sleep(1);
		KernelReadTscNative();

		const auto host_start = Common::Timer::QueryPerformanceCounter();
		const auto tsc_start  = KernelReadTscNative();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		const auto host_end = Common::Timer::QueryPerformanceCounter();
		const auto tsc_end  = KernelReadTscNative();

		const auto host_delta = host_end - host_start;
		const auto tsc_delta  = tsc_end - tsc_start;
		if (host_delta == 0 || tsc_delta == 0) {
			return host_frequency;
		}

		auto freq = static_cast<uint64_t>(
		    (static_cast<long double>(tsc_delta) * static_cast<long double>(host_frequency)) /
		    static_cast<long double>(host_delta));
		static constexpr uint64_t ROUND_TO = 100000;
		const auto                mod      = freq % ROUND_TO;
		freq = (mod >= ROUND_TO / 2 ? freq - mod + ROUND_TO : freq - mod);

		return freq != 0 ? freq : host_frequency;
#else
		return Common::Timer::QueryPerformanceFrequency();
#endif
	}();

	return frequency;
}

static uint64_t KernelGetInitialTsc() {
	static const uint64_t initial_tsc = KernelReadTscNative();
	return initial_tsc;
}

static uint64_t KernelGetElapsedTsc() {
	const auto initial_tsc = KernelGetInitialTsc();
	const auto current_tsc = KernelReadTscNative();
	return current_tsc >= initial_tsc ? current_tsc - initial_tsc : 0;
}

static uint64_t KernelGetProcessTimeUsNative() {
	const auto frequency = KernelGetTscFrequencyNative();
	if (frequency == 0) {
		return static_cast<uint64_t>(Loader::Timer::GetTimeMs() * 1000.0);
	}

	const auto elapsed = KernelGetElapsedTsc();
	return static_cast<uint64_t>((static_cast<long double>(elapsed) * 1000000.0L) /
	                             static_cast<long double>(frequency));
}

static void KernelUsToTimespec(uint64_t us, KernelTimespec* tp) {
	EXIT_IF(tp == nullptr);
	tp->tv_sec  = static_cast<int64_t>(us / 1000000);
	tp->tv_nsec = static_cast<int64_t>((us % 1000000) * 1000);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static uint64_t KernelFiletimeTo100ns(FILETIME ft) {
	ULARGE_INTEGER v {};
	v.LowPart  = ft.dwLowDateTime;
	v.HighPart = ft.dwHighDateTime;
	return v.QuadPart;
}

static void Kernel100nsToTimespec(uint64_t value, KernelTimespec* tp) {
	EXIT_IF(tp == nullptr);
	tp->tv_sec  = static_cast<int64_t>(value / 10000000);
	tp->tv_nsec = static_cast<int64_t>((value % 10000000) * 100);
}

static bool KernelRealtimeToTimespec(bool precise, KernelTimespec* tp) {
	EXIT_IF(tp == nullptr);

	FILETIME ft {};
	if (precise) {
		GetSystemTimePreciseAsFileTime(&ft);
	} else {
		GetSystemTimeAsFileTime(&ft);
	}

	static constexpr uint64_t WINDOWS_UNIX_EPOCH_DELTA_100NS = 116444736000000000ULL;
	const auto                value                          = KernelFiletimeTo100ns(ft);
	if (value < WINDOWS_UNIX_EPOCH_DELTA_100NS) {
		return false;
	}

	Kernel100nsToTimespec(value - WINDOWS_UNIX_EPOCH_DELTA_100NS, tp);
	return true;
}

static bool KernelMonotonicToTimespec(KernelTimespec* tp) {
	EXIT_IF(tp == nullptr);

	const auto frequency = Common::Timer::QueryPerformanceFrequency();
	if (frequency == 0) {
		return false;
	}

	const auto counter = Common::Timer::QueryPerformanceCounter();
	tp->tv_sec         = static_cast<int64_t>(counter / frequency);
	tp->tv_nsec        = static_cast<int64_t>(((counter % frequency) * 1000000000ull) / frequency);
	return true;
}

static bool KernelClockGettimeSpecial(KernelClockid clock_id, KernelTimespec* tp, int* error) {
	EXIT_IF(tp == nullptr);
	EXIT_IF(error == nullptr);

	if (clock_id == KERNEL_CLOCK_PROCTIME) {
		KernelUsToTimespec(KernelGetProcessTimeUsNative(), tp);
		return true;
	}

	FILETIME create_time {};
	FILETIME exit_time {};
	FILETIME kernel_time {};
	FILETIME user_time {};

	switch (clock_id) {
		case KERNEL_CLOCK_REALTIME:
		case KERNEL_CLOCK_REALTIME_PRECISE:
			if (!KernelRealtimeToTimespec(true, tp)) {
				*error = KERNEL_ERROR_EFAULT;
			}
			return true;
		case KERNEL_CLOCK_REALTIME_FAST:
		case KERNEL_CLOCK_SECOND:
			if (!KernelRealtimeToTimespec(false, tp)) {
				*error = KERNEL_ERROR_EFAULT;
			}
			if (clock_id == KERNEL_CLOCK_SECOND) {
				tp->tv_nsec = 0;
			}
			return true;
		case KERNEL_CLOCK_MONOTONIC:
		case KERNEL_CLOCK_UPTIME:
		case KERNEL_CLOCK_UPTIME_PRECISE:
		case KERNEL_CLOCK_UPTIME_FAST:
		case KERNEL_CLOCK_MONOTONIC_PRECISE:
		case KERNEL_CLOCK_MONOTONIC_FAST:
		case KERNEL_CLOCK_EXT_NETWORK:
		case KERNEL_CLOCK_EXT_DEBUG_NETWORK:
		case KERNEL_CLOCK_EXT_AD_NETWORK:
		case KERNEL_CLOCK_EXT_RAW_NETWORK:
			if (!KernelMonotonicToTimespec(tp)) {
				*error = KERNEL_ERROR_EFAULT;
			}
			return true;
		case KERNEL_CLOCK_THREAD_CPUTIME_ID:
			if (!GetThreadTimes(GetCurrentThread(), &create_time, &exit_time, &kernel_time,
			                    &user_time)) {
				*error = KERNEL_ERROR_EFAULT;
				return true;
			}
			Kernel100nsToTimespec(
			    KernelFiletimeTo100ns(kernel_time) + KernelFiletimeTo100ns(user_time), tp);
			return true;
		case KERNEL_CLOCK_VIRTUAL:
			if (!GetProcessTimes(GetCurrentProcess(), &create_time, &exit_time, &kernel_time,
			                     &user_time)) {
				*error = KERNEL_ERROR_EFAULT;
				return true;
			}
			Kernel100nsToTimespec(KernelFiletimeTo100ns(user_time), tp);
			return true;
		case KERNEL_CLOCK_PROF:
			if (!GetProcessTimes(GetCurrentProcess(), &create_time, &exit_time, &kernel_time,
			                     &user_time)) {
				*error = KERNEL_ERROR_EFAULT;
				return true;
			}
			Kernel100nsToTimespec(KernelFiletimeTo100ns(kernel_time), tp);
			return true;
		default: return false;
	}
}
#else
static bool kernel_clock_gettime_special(KernelClockid clock_id, KernelTimespec* tp, int* error) {
	EXIT_IF(tp == nullptr);
	EXIT_IF(error == nullptr);

	if (clock_id == KERNEL_CLOCK_PROCTIME) {
		kernel_us_to_timespec(kernel_get_process_time_us_native(), tp);
		return true;
	}

	return false;
}
#endif

struct PthreadMutexPrivate {
	uint8_t                 reserved[256];
	std::string             name;
	std::mutex              m;
	std::condition_variable cv;
	Pthread                 owner     = nullptr;
	uint32_t                count     = 0;
	int                     type      = 1;
	int                     pprotocol = PTHREAD_PRIO_NONE;
};

struct PthreadMutexattrPrivate {
	uint8_t             reserved[64];
	pthread_mutexattr_t p;
	int                 pprotocol;
	int                 type;
};

struct PthreadAttrPrivate {
	uint8_t        reserved[64];
	KernelCpumask  affinity;
	size_t         guard_size;
	void*          stack_addr;
	size_t         stack_size;
	bool           stack_user;
	uint64_t       stack_map_addr;
	size_t         stack_map_size;
	int            policy;
	int            inherit_sched;
	int            solosched;
	bool           detached;
	pthread_attr_t p;
};

struct PthreadCondPrivate;

struct PthreadGuestData {
	int32_t thread_id;
	uint8_t reserved[4092];
};

static_assert(sizeof(PthreadGuestData) == 4096);

// Mirrors FiberCpuContext: non-local return from guest stack without retq on guest RSP.
struct HostCpuContext {
	uint64_t rbx;
	uint64_t rbp;
	uint64_t rdi;
	uint64_t rsi;
	uint64_t r12;
	uint64_t r13;
	uint64_t r14;
	uint64_t r15;
	uint64_t rsp;
	uint64_t rip;
};

static_assert(sizeof(HostCpuContext) == 80);

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((noinline, returns_twice)) static uint64_t HostSaveContext(HostCpuContext* ctx) {
	uint64_t ret = 0;
	asm volatile("movq %[ctx], %%r10\n\t"
	             "movq %%rbx, 0(%%r10)\n\t"
	             "movq %%rbp, 8(%%r10)\n\t"
	             "movq %%rdi, 16(%%r10)\n\t"
	             "movq %%rsi, 24(%%r10)\n\t"
	             "movq %%r12, 32(%%r10)\n\t"
	             "movq %%r13, 40(%%r10)\n\t"
	             "movq %%r14, 48(%%r10)\n\t"
	             "movq %%r15, 56(%%r10)\n\t"
	             "leaq 8(%%rsp), %%r11\n\t"
	             "movq %%r11, 64(%%r10)\n\t"
	             "movq (%%rsp), %%r11\n\t"
	             "movq %%r11, 72(%%r10)\n\t"
	             "xorq %%rax, %%rax\n\t"
	             : "=a"(ret)
	             : [ctx] "r"(ctx)
	             : "memory", "r10", "r11");
	return ret;
}

__attribute__((noreturn, noinline)) static void HostRestoreContext(HostCpuContext* ctx,
                                                                   uint64_t         ret) {
	asm volatile("movq %[ctx], %%r10\n\t"
	             "movq 72(%%r10), %%r11\n\t"
	             "movq 0(%%r10), %%rbx\n\t"
	             "movq 8(%%r10), %%rbp\n\t"
	             "movq 16(%%r10), %%rdi\n\t"
	             "movq 24(%%r10), %%rsi\n\t"
	             "movq 32(%%r10), %%r12\n\t"
	             "movq 40(%%r10), %%r13\n\t"
	             "movq 48(%%r10), %%r14\n\t"
	             "movq 56(%%r10), %%r15\n\t"
	             "movq 64(%%r10), %%rsp\n\t"
	             "movq %[ret], %%rax\n\t"
	             "jmp *%%r11\n\t"
	             :
	             : [ctx] "r"(ctx), [ret] "r"(ret)
	             : "memory", "rax", "r10", "r11");
	__builtin_unreachable();
}
#else
static uint64_t HostSaveContext(HostCpuContext* ctx) {
	(void)ctx;
	return 0;
}

[[noreturn]] static void HostRestoreContext(HostCpuContext* ctx, uint64_t ret) {
	(void)ctx;
	(void)ret;
	EXIT("Host context switching is only implemented on x86_64\n");
}
#endif

struct PthreadPrivate {
	PthreadGuestData      guest;
	std::string           name;
	pthread_t             p;
	PthreadAttr           attr;
	pthread_entry_func_t  entry;
	void*                 arg;
	int                   unique_id;
	std::atomic_bool      detached;
	std::atomic_bool      almost_done;
	std::atomic_bool      free;
	std::atomic_bool      joining;
	// False while recycled/creating or after join — Setprio must not touch stale pthread_t.
	std::atomic_bool      p_valid;
	std::mutex            life_mutex;
	uint64_t              host_thread_id;
	HostCpuContext        host_return_ctx {};
	bool                  host_return_valid = false;
	uint64_t              host_return_value = 0;
	uint64_t              cond_sequence = 0;
	PthreadCondPrivate*   waiting_cond  = nullptr;
	std::atomic<uint64_t> pending_signal_mask {0};
};

static PthreadAttrPrivate* GetPthreadAttrValue(const PthreadAttr* attr, const char* func_name) {
	if (attr == nullptr) {
		return nullptr;
	}

	auto* attr_value = *attr;
	if (attr_value == nullptr || reinterpret_cast<uint64_t>(attr_value) < 0x10000u) {
		(void)func_name;
		return nullptr;
	}

	return attr_value;
}

struct PthreadRwlockPrivate {
	struct Reader {
		Pthread  thread = nullptr;
		uint32_t count  = 0;
	};

	uint8_t                 reserved[256];
	std::string             name;
	std::mutex              m;
	std::condition_variable cv;
	Pthread                 writer          = nullptr;
	uint32_t                writer_count    = 0;
	uint32_t                reader_count    = 0;
	uint32_t                waiting_writers = 0;
	std::vector<Reader>     readers;
};

struct PthreadRwlockattrPrivate {
	uint8_t              reserved[64];
	int                  type;
	pthread_rwlockattr_t p;
};

struct PthreadCondattrPrivate {
	uint8_t            reserved[64];
	pthread_condattr_t p;
	KernelClockid      clock_id = KERNEL_CLOCK_REALTIME;
};

struct PthreadCondPrivate {
	uint8_t                 reserved[256];
	std::string             name;
	std::mutex              m;
	std::condition_variable cv;
	uint64_t                sequence = 0;
	KernelClockid           clock_id = KERNEL_CLOCK_REALTIME;
	std::vector<Pthread>    waiters;
};

static bool IsSubmissionRelatedName(const std::string& name) {
	return name.find("Mixed") != std::string::npos || name.find("Compute") != std::string::npos ||
	       name.find("Submit") != std::string::npos;
}

// PPSA21564: Draw*/Havok/Gfx sit on job-queue conds that never get Mixed/Compute wakes.
// Without them the render pipeline stays idle and Main never reaches guest SubmitFlip.
static bool IsProducerPipelineName(const std::string& name) {
	if (IsSubmissionRelatedName(name)) {
		return true;
	}
	return name.find("Draw") != std::string::npos || name.find("Havok") != std::string::npos ||
	       name.find("Gfx") != std::string::npos || name.find("Render") != std::string::npos ||
	       name.find("Geometry") != std::string::npos || name.find("Streamer") != std::string::npos;
}

static bool IsMixedName(const std::string& name) {
	return name.find("Mixed") != std::string::npos;
}

static bool IsComputeName(const std::string& name) {
	return name.find("Compute") != std::string::npos;
}

static bool IsMainRelatedThread(const PthreadPrivate* thread) {
	if (thread == nullptr) {
		return false;
	}
	return thread->unique_id == 8 || thread->name == "MainThread" ||
	       thread->name.find("Main") != std::string::npos ||
	       thread->name.find("BootCards") != std::string::npos;
}

static const char* Phase54RoleOf(const PthreadPrivate* thread) {
	if (thread == nullptr) {
		return "?";
	}
	if (IsMixedName(thread->name)) {
		return "Mixed";
	}
	if (IsComputeName(thread->name)) {
		return "Compute";
	}
	if (IsMainRelatedThread(thread)) {
		return "Main";
	}
	if (IsSubmissionRelatedName(thread->name)) {
		return "Submit";
	}
	return "other";
}

struct Phase54CondTrace {
	uintptr_t cond_ptr              = 0;
	uint32_t  wait_n                = 0;
	uint32_t  signal_n              = 0;
	uint32_t  signals_from_main     = 0;
	uint32_t  signals_from_other    = 0;
	uint32_t  wake_mixed            = 0;
	uint32_t  wake_compute          = 0;
	int       last_tid              = 0;
	char      last_by[40]           = {};
	uint8_t   has_mixed_waiter : 1;
	uint8_t   has_compute_waiter : 1;
	uint8_t   role_logged_mixed : 1;
	uint8_t   role_logged_compute : 1;
};

static constexpr size_t kPhase54CondSlots = 48;
static Phase54CondTrace g_phase54_conds[kPhase54CondSlots] {};
static std::mutex       g_phase54_cond_mu;

static Phase54CondTrace* Phase54FindOrAddCond(uintptr_t cond_ptr) {
	if (cond_ptr == 0) {
		return nullptr;
	}
	Phase54CondTrace* free_slot = nullptr;
	for (size_t i = 0; i < kPhase54CondSlots; i++) {
		if (g_phase54_conds[i].cond_ptr == cond_ptr) {
			return &g_phase54_conds[i];
		}
		if (free_slot == nullptr && g_phase54_conds[i].cond_ptr == 0) {
			free_slot = &g_phase54_conds[i];
		}
	}
	if (free_slot != nullptr) {
		free_slot->cond_ptr            = cond_ptr;
		free_slot->has_mixed_waiter    = 0;
		free_slot->has_compute_waiter  = 0;
		free_slot->role_logged_mixed   = 0;
		free_slot->role_logged_compute = 0;
		return free_slot;
	}
	return nullptr;
}

static void Phase54TagRoleOnce(PthreadPrivate* thread) {
	if (thread == nullptr || !Libs::VideoOut::Phase37PostUnregisterSeen()) {
		return;
	}
	static std::atomic<uint32_t> mixed_logged {0};
	static std::atomic<uint32_t> compute_logged {0};
	if (IsMixedName(thread->name) && mixed_logged.fetch_add(1, std::memory_order_relaxed) == 0) {
		LOGF("SubmitTrace: phase54 role=Mixed tid=%d name=%s\n", thread->unique_id,
		     thread->name.c_str());
		fprintf(stderr, "SubmitTrace: phase54 role=Mixed tid=%d name=%s\n", thread->unique_id,
		        thread->name.c_str());
	}
	if (IsComputeName(thread->name) &&
	    compute_logged.fetch_add(1, std::memory_order_relaxed) == 0) {
		LOGF("SubmitTrace: phase54 role=Compute tid=%d name=%s\n", thread->unique_id,
		     thread->name.c_str());
		fprintf(stderr, "SubmitTrace: phase54 role=Compute tid=%d name=%s\n", thread->unique_id,
		        thread->name.c_str());
	}
}

static thread_local bool     g_phase54_tls_just_woken = false;
static thread_local uint64_t g_phase70_tls_guest_rip  = 0;

static void Phase70CaptureGuestRipFromAbi() {
	g_phase70_tls_guest_rip = reinterpret_cast<uint64_t>(__builtin_return_address(0));
}

static void Phase54OnCondWaitExit(PthreadPrivate* thread, PthreadCondPrivate* cond_value,
                                  int wait_result) {
	if (wait_result != OK || thread == nullptr || cond_value == nullptr) {
		return;
	}
	if (!IsSubmissionRelatedName(thread->name) ||
	    !Libs::VideoOut::Phase37PostUnregisterSeen()) {
		return;
	}
	g_phase54_tls_just_woken = true;
	Libs::VideoOut::Phase54NoteMixedWake(Phase54RoleOf(thread),
	                                     reinterpret_cast<uintptr_t>(cond_value), "woken");
}

static void Phase54NoteCondWait(PthreadCondPrivate* cond, PthreadPrivate* thread) {
	if (cond == nullptr || thread == nullptr) {
		return;
	}
	Phase54TagRoleOnce(thread);
	if (g_phase54_tls_just_woken && IsSubmissionRelatedName(thread->name) &&
	    Libs::VideoOut::Phase37PostUnregisterSeen()) {
		g_phase54_tls_just_woken = false;
		Libs::VideoOut::Phase54NoteMixedWake(Phase54RoleOf(thread),
		                                     reinterpret_cast<uintptr_t>(cond), "rewait");
	}
	const bool interesting = IsSubmissionRelatedName(thread->name) || IsMainRelatedThread(thread);
	if (!interesting && !Libs::VideoOut::Phase37PostUnregisterSeen()) {
		return;
	}
	uint64_t cycle = Libs::VideoOut::Phase54CurrentCycleId();
	{
		std::lock_guard lock(g_phase54_cond_mu);
		auto*           tr = Phase54FindOrAddCond(reinterpret_cast<uintptr_t>(cond));
		if (tr != nullptr) {
			tr->wait_n++;
			if (IsMixedName(thread->name)) {
				tr->has_mixed_waiter = 1;
			}
			if (IsComputeName(thread->name)) {
				tr->has_compute_waiter = 1;
			}
		}
	}
	if (!Libs::VideoOut::Phase37PostUnregisterSeen() && !interesting) {
		return;
	}
	if (Libs::VideoOut::Phase37PostUnregisterSeen() && IsMainRelatedThread(thread)) {
		Libs::VideoOut::Phase64NoteMainCondWait(reinterpret_cast<uint64_t>(cond));
	}
	if (Libs::VideoOut::Phase37PostUnregisterSeen()) {
		// Prefer ABI-entry guest RIP (tls set in PthreadCondWait*); else RA(1).
		uint64_t guest_rip = g_phase70_tls_guest_rip;
		if (guest_rip < 0x0000000900000000ULL || guest_rip >= 0x0000000A00000000ULL) {
			guest_rip = reinterpret_cast<uint64_t>(__builtin_return_address(1));
		}
		Libs::VideoOut::Phase70NoteGuestRip(guest_rip, Phase54RoleOf(thread));
		Libs::VideoOut::Phase65NoteCondWait(Phase54RoleOf(thread), thread->name.c_str(),
		                                    thread->unique_id, guest_rip);
	}
	static std::atomic<uint32_t> logs {0};
	if (logs.fetch_add(1, std::memory_order_relaxed) < 128) {
		// fprintf only on guest-stack HLE (avoid fmt /GS fastfail).
		fprintf(stderr,
		        "SubmitTrace: phase54 cond_wait role=%s cond=0x%016" PRIx64 " cycle=%" PRIu64 "\n",
		        Phase54RoleOf(thread), reinterpret_cast<uint64_t>(cond), cycle);
		std::fflush(stderr);
	}
}

static uint64_t Phase54NoteCondSignal(PthreadCondPrivate* cond, PthreadPrivate* by, bool notify,
                                      bool broadcast, const char* waiters_summary) {
	if (cond == nullptr) {
		return Libs::VideoOut::Phase54CurrentCycleId();
	}
	const bool from_main = IsMainRelatedThread(by);
	bool       has_mixed = false;
	bool       has_comp  = false;
	uint32_t   sig_main  = 0;
	uint32_t   wake_m    = 0;
	uint32_t   wake_c    = 0;
	uint64_t   cycle     = Libs::VideoOut::Phase54CurrentCycleId();
	{
		std::lock_guard lock(g_phase54_cond_mu);
		auto*           tr = Phase54FindOrAddCond(reinterpret_cast<uintptr_t>(cond));
		if (tr != nullptr) {
			tr->signal_n++;
			if (from_main) {
				tr->signals_from_main++;
			} else {
				tr->signals_from_other++;
			}
			tr->last_tid = by != nullptr ? by->unique_id : 0;
			if (by != nullptr) {
				std::snprintf(tr->last_by, sizeof(tr->last_by), "%s", by->name.c_str());
			}
			has_mixed = tr->has_mixed_waiter != 0;
			has_comp  = tr->has_compute_waiter != 0;
			if (notify) {
				if (has_mixed) {
					tr->wake_mixed++;
				}
				if (has_comp) {
					tr->wake_compute++;
				}
			}
			sig_main = tr->signals_from_main;
			wake_m   = tr->wake_mixed;
			wake_c   = tr->wake_compute;
		}
	}
	if ((has_mixed || has_comp) && notify) {
		cycle = Libs::VideoOut::Phase54BumpCycle(broadcast ? "cond_broadcast" : "cond_signal");
	}
	const bool log_it =
	    Libs::VideoOut::Phase37PostUnregisterSeen() || from_main || has_mixed || has_comp;
	if (Libs::VideoOut::Phase37PostUnregisterSeen()) {
		Libs::VideoOut::Phase64NoteMainCondSignal(reinterpret_cast<uint64_t>(cond), false);
	}
	if (!log_it) {
		return cycle;
	}
	static std::atomic<uint32_t> logs {0};
	if (logs.fetch_add(1, std::memory_order_relaxed) < 128) {
		const uint64_t ra = reinterpret_cast<uint64_t>(__builtin_return_address(0));
		LOGF("SubmitTrace: phase54 cond_%s by=%s role=%s cond=0x%016" PRIx64
		     " notify=%d waiters=%s cycle=%" PRIu64 " rip=0x%016" PRIx64
		     " sig_main=%u wake_m=%u wake_c=%u tid=%d\n",
		     broadcast ? "broadcast" : "signal", by != nullptr ? by->name.c_str() : "?",
		     Phase54RoleOf(by), reinterpret_cast<uint64_t>(cond), notify ? 1 : 0,
		     waiters_summary != nullptr ? waiters_summary : "-", cycle, ra, sig_main, wake_m,
		     wake_c, by != nullptr ? by->unique_id : 0);
		fprintf(stderr,
		        "SubmitTrace: phase54 cond_%s by=%s cond=0x%016" PRIx64 " notify=%d cycle=%" PRIu64
		        "\n",
		        broadcast ? "broadcast" : "signal", by != nullptr ? by->name.c_str() : "?",
		        reinterpret_cast<uint64_t>(cond), notify ? 1 : 0, cycle);
	}
	return cycle;
}

static void CondAddWaiter(PthreadCondPrivate* cond, Pthread thread) {
	EXIT_IF(cond == nullptr);
	EXIT_IF(thread == nullptr);

	thread->waiting_cond = cond;
	cond->waiters.push_back(thread);

	const bool main_related = IsMainRelatedThread(thread);
	if (IsSubmissionRelatedName(thread->name) || main_related) {
		static std::atomic<uint32_t> logs {0};
		if (logs.fetch_add(1, std::memory_order_relaxed) < 96) {
			// fprintf only — LOGF/fmt on guest-stack HLE → /GS 0xC0000409.
			fprintf(stderr, "SubmitTrace: CondWait name=%s tid=%d cond=%s\n", thread->name.c_str(),
			        thread->unique_id, cond->name.c_str());
			std::fflush(stderr);
			char beat[160];
			std::snprintf(beat, sizeof(beat), "heartbeat CondWait name=%s tid=%d",
			              thread->name.c_str(), thread->unique_id);
			Common::HeartbeatLog(beat);
		}
	}
	Common::HeartbeatLog("heartbeat CondWait before_phase54");
	Phase54NoteCondWait(cond, thread);
	Common::HeartbeatLog("heartbeat CondWait after_phase54");
}

static bool CondRemoveWaiter(PthreadCondPrivate* cond, Pthread thread) {
	EXIT_IF(cond == nullptr);
	EXIT_IF(thread == nullptr);

	auto it = std::find(cond->waiters.begin(), cond->waiters.end(), thread);
	if (it == cond->waiters.end()) {
		return false;
	}

	cond->waiters.erase(it);
	if (thread->waiting_cond == cond) {
		thread->waiting_cond = nullptr;
	}
	return true;
}

static bool CondWakeWaiter(PthreadCondPrivate* cond, Pthread thread) {
	EXIT_IF(cond == nullptr);

	if (thread == nullptr) {
		if (cond->waiters.empty()) {
			return false;
		}
		thread = cond->waiters.front();
	}

	auto it = std::find(cond->waiters.begin(), cond->waiters.end(), thread);
	if (it == cond->waiters.end()) {
		return false;
	}

	cond->waiters.erase(it);
	if (thread->waiting_cond == cond) {
		thread->waiting_cond = nullptr;
	}
	thread->cond_sequence++;
	return true;
}

static void CondClearWaiters(PthreadCondPrivate* cond) {
	EXIT_IF(cond == nullptr);

	for (auto* thread: cond->waiters) {
		if (thread != nullptr && thread->waiting_cond == cond) {
			thread->waiting_cond = nullptr;
		}
	}
	cond->waiters.clear();
}

void PthreadWakeForSignal(Pthread thread) {
	if (thread == nullptr) {
		return;
	}

	auto* cond = thread->waiting_cond;
	if (cond == nullptr) {
		return;
	}

	bool notify = false;
	{
		std::lock_guard lock(cond->m);
		notify = (thread->waiting_cond == cond);
	}
	if (notify) {
		cond->cv.notify_all();
	}
}

void KernelDispatchPendingSignalForCurrentThread();

static void SchedulerBackoffOnce() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (SwitchToThread() == 0) {
		Sleep(0);
	}
#else
	std::this_thread::yield();
#endif
}

static bool SleepMicroSchedulerBackoff(uint64_t microseconds) {
	if (microseconds > 1) {
		return false;
	}

	SchedulerBackoffOnce();
	return true;
}

static void SleepMicroWithSignalPoll(uint64_t microseconds) {
	if (microseconds == 0) {
		KernelDispatchPendingSignalForCurrentThread();
		return;
	}

	while (microseconds > 0) {
		const auto step = std::min<uint64_t>(microseconds, SIGNAL_APC_POLL_MICROS);
		if (!SleepMicroSchedulerBackoff(step)) {
			Common::Thread::SleepMicro(step);
		}
		microseconds -= step;
		KernelDispatchPendingSignalForCurrentThread();
	}
}

static void SleepNanoWithSignalPoll(uint64_t nanoseconds) {
	if (nanoseconds == 0) {
		KernelDispatchPendingSignalForCurrentThread();
		return;
	}

	constexpr uint64_t poll_nanos = static_cast<uint64_t>(SIGNAL_APC_POLL_MICROS) * 1000ull;
	while (nanoseconds > 0) {
		const auto step = std::min<uint64_t>(nanoseconds, poll_nanos);
		Common::Thread::SleepNano(step);
		nanoseconds -= step;
		KernelDispatchPendingSignalForCurrentThread();
	}
}

struct PthreadStaticObject {
	enum class Type { Mutex, Cond, Rwlock };

	Type             type;
	uint64_t         vaddr;
	Loader::Program* program;
};

class PthreadStaticObjects {
public:
	PthreadStaticObjects() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	virtual ~PthreadStaticObjects() { KYTY_NOT_IMPLEMENTED; }

	KYTY_CLASS_NO_COPY(PthreadStaticObjects);

	void* CreateObject(void* addr, PthreadStaticObject::Type type);
	void  DeleteObjects(Loader::Program* program);

private:
	std::vector<PthreadStaticObject*> m_objects;
	Common::Mutex                     m_mutex;
};

class PthreadKeys {
public:
	PthreadKeys() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	virtual ~PthreadKeys() { KYTY_NOT_IMPLEMENTED; }

	KYTY_CLASS_NO_COPY(PthreadKeys);

	bool Create(int* key, pthread_key_destructor_func_t destructor);
	bool Delete(int key);
	void Destruct(int thread_id);
	bool Set(int key, int thread_id, void* data);
	bool Get(int key, int thread_id, void** data);

private:
	struct Map {
		int   thread_id = -1;
		void* data      = nullptr;
	};

	struct Key {
		bool                          used       = false;
		pthread_key_destructor_func_t destructor = nullptr;
		std::vector<Map>              specific_values;
	};

	Common::Mutex m_mutex;
	Key           m_keys[KEYS_MAX];
};

class PthreadPool {
public:
	PthreadPool() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	virtual ~PthreadPool() { KYTY_NOT_IMPLEMENTED; }

	KYTY_CLASS_NO_COPY(PthreadPool);

	Pthread Create();

	void   FreeDetachedThreads();
	void   DumpSubmissionThreads(const char* reason);
	void   DumpAllGuestThreads(const char* reason);
	void   SnapshotGuestThread(int unique_id, uint32_t host_tid_hint, const char* reason);
	size_t WakeSubmissionCondWaiters();
	size_t SuspendAllGuests();
	size_t ResumeAllGuests();
	size_t SuspendMainRelatedGuests();
	size_t ResumeMainRelatedGuests();
	[[nodiscard]] bool HasAliveMainThread();

private:
	std::vector<Pthread> m_threads;
	Common::Mutex        m_mutex;
};

class PThreadContext {
public:
	PThreadContext() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	virtual ~PThreadContext() { KYTY_NOT_IMPLEMENTED; }

	KYTY_CLASS_NO_COPY(PThreadContext);

	PthreadAttr*       GetDefaultAttr() { return &m_default_attr; }
	void               SetDefaultAttr(PthreadAttr attr) { m_default_attr = attr; }
	PthreadCondattr*   GetDefaultCondattr() { return &m_default_condattr; }
	void               SetDefaultCondattr(PthreadCondattr attr) { m_default_condattr = attr; }
	PthreadMutexattr*  GetDefaultMutexattr() { return &m_default_mutexattr; }
	void               SetDefaultMutexattr(PthreadMutexattr attr) { m_default_mutexattr = attr; }
	PthreadRwlockattr* GetDefaultRwlockattr() { return &m_default_rwlockattr; }
	void               SetDefaultRwlockattr(PthreadRwlockattr attr) { m_default_rwlockattr = attr; }
	PthreadPool*       GetPthreadPool() { return m_pthread_pool; }
	void               SetPthreadPool(PthreadPool* pool) { m_pthread_pool = pool; }
	PthreadStaticObjects* GetPthreadStaticObjects() { return m_pthread_static_objects; }
	void SetPthreadStaticObjects(PthreadStaticObjects* objs) { m_pthread_static_objects = objs; }
	PthreadKeys* GetPthreadKeys() { return m_pthread_keys; }
	void         SetPthreadKeys(PthreadKeys* keys) { m_pthread_keys = keys; }

	[[nodiscard]] thread_dtors_func_t GetThreadDtors() const { return m_thread_dtors; }
	void SetThreadDtors(thread_dtors_func_t dtors) { m_thread_dtors = dtors; }

private:
	// Common::Mutex           m_mutex;
	PthreadMutexattr      m_default_mutexattr      = nullptr;
	PthreadRwlockattr     m_default_rwlockattr     = nullptr;
	PthreadCondattr       m_default_condattr       = nullptr;
	PthreadAttr           m_default_attr           = nullptr;
	PthreadPool*          m_pthread_pool           = nullptr;
	PthreadStaticObjects* m_pthread_static_objects = nullptr;
	PthreadKeys*          m_pthread_keys           = nullptr;

	std::atomic<thread_dtors_func_t> m_thread_dtors = nullptr;
};

thread_local Pthread        g_pthread_self      = nullptr;
static Pthread              g_pthread_main      = nullptr;
PThreadContext*             g_pthread_context   = nullptr;
static std::atomic<int32_t> g_pthread_thread_id = 0;

static Common::Mutex g_guest_stack_mutex;
static uint64_t      g_guest_stack_last = 0;

static size_t RoundStackSize(size_t size) {
	return ((size + PTHREAD_STACK_PAGE - 1) / PTHREAD_STACK_PAGE) * PTHREAD_STACK_PAGE;
}

static size_t RoundStackMappingSize(size_t size) {
	return ((size + PTHREAD_STACK_GRANULARITY - 1) / PTHREAD_STACK_GRANULARITY) *
	       PTHREAD_STACK_GRANULARITY;
}

static int CreateGuestStack(PthreadAttr attr) {
	if (attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	if (attr->stack_addr != nullptr) {
		attr->guard_size     = 0;
		attr->stack_user     = true;
		attr->stack_map_addr = 0;
		attr->stack_map_size = 0;
		return OK;
	}

	const auto stack_size = RoundStackSize(attr->stack_size);
	const auto guard_size = RoundStackSize(attr->guard_size);
	const auto map_size   = RoundStackMappingSize(stack_size + guard_size);

	uint64_t stack_addr = 0;
	{
		Common::LockGuard lock(g_guest_stack_mutex);

		if (g_guest_stack_last == 0) {
			g_guest_stack_last = (PTHREAD_STACK_TOP - PTHREAD_STACK_INITIAL - PTHREAD_STACK_PAGE) &
			                     ~(static_cast<uint64_t>(PTHREAD_STACK_GRANULARITY) - 1);
		}

		stack_addr = g_guest_stack_last - map_size;
		g_guest_stack_last -= map_size;
	}

	void* mapped_addr = reinterpret_cast<void*>(stack_addr);

	constexpr int PROT_READ_WRITE = 0x03;
	constexpr int MAP_PRIVATE     = 0x02;
	constexpr int MAP_FIXED       = 0x10;
	constexpr int MAP_STACK       = 0x400;
	constexpr int MAP_ANON        = 0x1000;

	int result = Memory::KernelMapNamedFlexibleMemory(
	    &mapped_addr, map_size, PROT_READ_WRITE, MAP_PRIVATE | MAP_FIXED | MAP_STACK | MAP_ANON,
	    "stack");
	if (result != OK) {
		return KERNEL_ERROR_EAGAIN;
	}

	if (guard_size != 0) {
		result = Memory::KernelMprotect(reinterpret_cast<void*>(stack_addr), guard_size, 0);
		if (result != OK) {
			Memory::KernelMunmap(stack_addr, map_size);
			return KERNEL_ERROR_EAGAIN;
		}
	}

	attr->stack_addr     = reinterpret_cast<void*>(stack_addr + guard_size);
	attr->stack_size     = map_size - guard_size;
	attr->stack_user     = false;
	attr->stack_map_addr = stack_addr;
	attr->stack_map_size = map_size;

	std::memset(attr->stack_addr, 0, stack_size);

	return OK;
}

static void FreeGuestStack(PthreadAttr attr) {
	if (attr == nullptr || attr->stack_user || attr->stack_map_addr == 0 ||
	    attr->stack_map_size == 0) {
		return;
	}

	Memory::KernelMunmap(attr->stack_map_addr, attr->stack_map_size);

	attr->stack_addr     = nullptr;
	attr->stack_map_addr = 0;
	attr->stack_map_size = 0;
}

extern "C" __attribute__((noreturn, noinline, no_stack_protector)) void GuestThreadFinishOnHost(
    uint64_t value) {
	// Still on the guest stack here (arrived from GuestThreadReturnTrampoline).
	// Do not LOGF/fmt/EXIT — those use movaps and require a 16-byte-aligned host stack.
	// HostRestoreContext must use a non-zero sentinel: guest may legitimately return 0, and
	// HostSaveContext also returns 0 on first entry.
	auto* self = g_pthread_self;
	if (self == nullptr || !self->host_return_valid) {
		char line[160];
		std::snprintf(line, sizeof(line),
		              "GuestThreadFinishOnHost abort: self=%p valid=%d value=0x%016" PRIx64,
		              static_cast<void*>(self),
		              self != nullptr && self->host_return_valid ? 1 : 0, value);
		Common::LogFatalToFile(line);
		fprintf(stderr, "%s\n", line);
		std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		TerminateProcess(GetCurrentProcess(), 321);
#endif
		std::_Exit(321);
	}
	self->host_return_value = value;
	self->host_return_valid = false;
	HostRestoreContext(&self->host_return_ctx, 1);
}

// Entered by guest `ret` with return value still in rax — must not clobber rax in a C prologue.
#if defined(__x86_64__) || defined(_M_X64)
extern "C" __attribute__((naked, noinline)) void GuestThreadReturnTrampoline() {
	// Realign then call: guest rsp is often ≡8 mod 16 after ret; Finish/C++ must not see that.
	asm volatile(
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	    "movq %rax, %rcx\n\t"
	    "andq $-16, %rsp\n\t"
	    "subq $0x28, %rsp\n\t"
	    "call GuestThreadFinishOnHost\n\t"
#else
	    "movq %rax, %rdi\n\t"
	    "andq $-16, %rsp\n\t"
	    "subq $0x8, %rsp\n\t"
	    "call GuestThreadFinishOnHost\n\t"
#endif
	    "ud2\n\t");
}
#else
extern "C" __attribute__((noreturn, noinline)) void GuestThreadReturnTrampoline() {
	EXIT("GuestThreadReturnTrampoline is only implemented on x86_64\n");
}
#endif

uint64_t PthreadRunGuestOnStack(void* stack_top, void* entry, uint64_t arg0, uint64_t arg1) {
#if defined(__x86_64__) || defined(_M_X64)
	EXIT_IF(stack_top == nullptr);
	EXIT_IF(entry == nullptr);
	EXIT_IF(g_pthread_self == nullptr);

	const uint64_t rc = HostSaveContext(&g_pthread_self->host_return_ctx);
	if (rc != 0) {
		// Resumed on the host stack from GuestThreadFinishOnHost → HostRestoreContext.
		const uint64_t value = g_pthread_self->host_return_value;
		g_pthread_self->host_return_valid = false;
		// fprintf only — avoid fmt/LOGF on this path (historically hit after bad stack restores).
		fprintf(stderr, "GuestThreadFinishOnHost tid=%d value=0x%016" PRIx64 "\n",
		        g_pthread_self->unique_id, value);
		return value;
	}
	g_pthread_self->host_return_valid = true;

	auto guest_top = reinterpret_cast<uintptr_t>(stack_top) & ~static_cast<uintptr_t>(0x0f);
	const auto guest_rbp = guest_top - 4u * sizeof(uint64_t);

	auto* guest_root_frame = reinterpret_cast<uintptr_t*>(guest_rbp);
	guest_root_frame[0]    = 0;
	guest_root_frame[1]    = 0;

	// Synthetic call frame: [rsp]=trampoline, rsp ≡ 8 (mod 16) as after a real call.
	guest_top -= sizeof(uint64_t);
	*reinterpret_cast<uint64_t*>(guest_top) =
	    reinterpret_cast<uint64_t>(&GuestThreadReturnTrampoline);

	const auto guest_rsp = guest_top;
	auto*      entry_fn  = entry;

	// Guest SysV: rdi=arg0, rsi=arg1. Never callq — jmp only; return via trampoline.
	asm volatile("movq %[guest_rsp], %%rsp\n\t"
	             "movq %[guest_rbp], %%rbp\n\t"
	             "movq %[arg0], %%rdi\n\t"
	             "movq %[arg1], %%rsi\n\t"
	             "jmpq *%[entry]\n\t"
	             :
	             : [guest_rsp] "r"(guest_rsp), [guest_rbp] "r"(guest_rbp), [arg0] "r"(arg0),
	               [arg1] "r"(arg1), [entry] "r"(entry_fn)
	             : "memory", "cc", "rax", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11");
	__builtin_unreachable();
#else
	(void)stack_top;
	using Fn = KYTY_SYSV_ABI uint64_t (*)(uint64_t, uint64_t);
	return reinterpret_cast<Fn>(entry)(arg0, arg1);
#endif
}

static KYTY_SYSV_ABI void* RunOnGuestStack(void* arg, pthread_entry_func_t func, void* stack_top) {
	const uint64_t rc = PthreadRunGuestOnStack(stack_top, reinterpret_cast<void*>(func),
	                                           reinterpret_cast<uint64_t>(arg), 0);
	return reinterpret_cast<void*>(static_cast<uintptr_t>(rc));
}

static void UpdateCurrentThreadStackAttr(PthreadAttr* attr) {
	if (attr == nullptr || *attr == nullptr) {
		return;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	ULONG_PTR low  = 0;
	ULONG_PTR high = 0;
	GetCurrentThreadStackLimits(&low, &high);

	if (low != 0 && high > low) {
		(*attr)->stack_addr = reinterpret_cast<void*>(low);
		(*attr)->stack_size = high - low;
		(*attr)->stack_user = true;
	}
#endif
}

static void FreeDetachedThreads(void* /*arg*/) {
	PRINT_NAME_ENABLE(false);

	auto* pthread_pool = g_pthread_context->GetPthreadPool();

	while (true) {
		Common::Thread::Sleep(10000);
		pthread_pool->FreeDetachedThreads();
	}
}

void PthreadDeleteStaticObjects(Loader::Program* program) {

	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	pthread_static_objects->DeleteObjects(program);
}

void PthreadInitSelfForMainThread() {
	EXIT_IF(g_pthread_self != nullptr);

	g_pthread_self = new PthreadPrivate {};
	PthreadAttrInit(&g_pthread_self->attr);
	UpdateCurrentThreadStackAttr(&g_pthread_self->attr);
	g_pthread_self->p               = pthread_self();
	g_pthread_self->name            = "MainThread";
	g_pthread_self->guest.thread_id = ++g_pthread_thread_id;
	g_pthread_self->unique_id       = Common::Thread::GetThreadIdUnique();
	g_pthread_self->free            = false;
	g_pthread_self->joining         = false;
	g_pthread_self->p_valid         = true;
	g_pthread_self->detached        = false;
	g_pthread_self->almost_done     = false;
	g_pthread_self->entry           = nullptr;
	g_pthread_self->arg             = nullptr;

	uint64_t os_thread_id = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	os_thread_id = static_cast<uint64_t>(GetCurrentThreadId());
#endif
	g_pthread_self->host_thread_id = os_thread_id;
	g_pthread_main                 = g_pthread_self;

	LOGF("\tPthread main self: id = %d, os_thread_id = %" PRIu64 ", stack_addr = 0x%016" PRIx64
	     ", stack_size = %" PRIu64 "\n",
	     g_pthread_self->unique_id, os_thread_id,
	     reinterpret_cast<uint64_t>(g_pthread_self->attr->stack_addr),
	     static_cast<uint64_t>(g_pthread_self->attr->stack_size));
}

void* PthreadCreateMainGuestStack() {
	EXIT_IF(g_pthread_self == nullptr);
	EXIT_IF(g_pthread_self->attr == nullptr);

	if (g_pthread_self->attr->stack_map_addr != 0) {
		return reinterpret_cast<void*>(
		    (reinterpret_cast<uintptr_t>(g_pthread_self->attr->stack_addr) +
		     g_pthread_self->attr->stack_size) &
		    ~static_cast<uintptr_t>(0x0f));
	}

	g_pthread_self->attr->stack_addr     = nullptr;
	g_pthread_self->attr->stack_size     = PTHREAD_STACK_DEFAULT + PTHREAD_STACK_EXTRA;
	g_pthread_self->attr->stack_user     = false;
	g_pthread_self->attr->stack_map_addr = 0;
	g_pthread_self->attr->stack_map_size = 0;

	const auto result = CreateGuestStack(g_pthread_self->attr);
	EXIT_NOT_IMPLEMENTED(result != OK);

	auto* stack_top =
	    reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(g_pthread_self->attr->stack_addr) +
	                             g_pthread_self->attr->stack_size) &
	                            ~static_cast<uintptr_t>(0x0f));

	LOGF("\tPthread main guest stack: stack_addr = 0x%016" PRIx64 ", stack_size = %" PRIu64
	     ", stack_top = 0x%016" PRIx64 "\n",
	     reinterpret_cast<uint64_t>(g_pthread_self->attr->stack_addr),
	     static_cast<uint64_t>(g_pthread_self->attr->stack_size),
	     reinterpret_cast<uint64_t>(stack_top));

	return stack_top;
}

KYTY_SUBSYSTEM_INIT(Pthread) {
	PRINT_NAME_ENABLE(false);

	EXIT_IF(g_pthread_context != nullptr);

	g_pthread_context = new PThreadContext;

	g_pthread_context->SetPthreadStaticObjects(new PthreadStaticObjects);
	g_pthread_context->SetPthreadPool(new PthreadPool);
	g_pthread_context->SetPthreadKeys(new PthreadKeys);
	Common::CondVar::SetWaitPollCallback(KernelDispatchPendingSignalForCurrentThread);

	PthreadMutexattr  default_mutexattr  = nullptr;
	PthreadRwlockattr default_rwlockattr = nullptr;
	PthreadCondattr   default_condattr   = nullptr;
	PthreadAttr       default_attr       = nullptr;

	PthreadAttrInit(&default_attr);
	PthreadMutexattrInit(&default_mutexattr);
	PthreadRwlockattrInit(&default_rwlockattr);
	PthreadCondattrInit(&default_condattr);

	g_pthread_context->SetDefaultMutexattr(default_mutexattr);
	g_pthread_context->SetDefaultRwlockattr(default_rwlockattr);
	g_pthread_context->SetDefaultCondattr(default_condattr);
	g_pthread_context->SetDefaultAttr(default_attr);

	PRINT_NAME_ENABLE(true);

	Common::Thread thread(FreeDetachedThreads, nullptr);
	thread.Detach();
}

KYTY_SUBSYSTEM_UNEXPECTED_SHUTDOWN(Pthread) {}

KYTY_SUBSYSTEM_DESTROY(Pthread) {}

static int PthreadAttrCopy(PthreadAttr* dst, const PthreadAttr* src) {
	if (dst == nullptr || *dst == nullptr || src == nullptr || *src == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	KernelCpumask    mask          = 0;
	int              state         = 0;
	size_t           guard_size    = 0;
	int              inherit_sched = 0;
	KernelSchedParam param         = {};
	int              policy        = 0;
	int              solosched     = 0;
	void*            stack_addr    = nullptr;
	size_t           stack_size    = 0;

	int result = 0;

	result = (result == 0 ? PthreadAttrGetaffinity(src, &mask) : result);
	result = (result == 0 ? PthreadAttrGetdetachstate(src, &state) : result);
	result = (result == 0 ? PthreadAttrGetguardsize(src, &guard_size) : result);
	result = (result == 0 ? PthreadAttrGetinheritsched(src, &inherit_sched) : result);
	result = (result == 0 ? PthreadAttrGetschedparam(src, &param) : result);
	result = (result == 0 ? PthreadAttrGetschedpolicy(src, &policy) : result);
	result = (result == 0 ? PthreadAttrGetsolosched(src, &solosched) : result);
	result = (result == 0 ? PthreadAttrGetstackaddr(src, &stack_addr) : result);
	result = (result == 0 ? PthreadAttrGetstacksize(src, &stack_size) : result);

	result = (result == 0 ? PthreadAttrSetaffinity(dst, mask) : result);
	result = (result == 0 ? PthreadAttrSetdetachstate(dst, state) : result);
	result = (result == 0 ? PthreadAttrSetguardsize(dst, guard_size) : result);
	result = (result == 0 ? PthreadAttrSetinheritsched(dst, inherit_sched) : result);
	result = (result == 0 ? PthreadAttrSetschedparam(dst, &param) : result);
	result = (result == 0 ? PthreadAttrSetschedpolicy(dst, policy) : result);
	result = (result == 0 ? PthreadAttrSetsolosched(dst, solosched) : result);
	if (stack_addr != nullptr) {
		result = (result == 0 ? PthreadAttrSetstackaddr(dst, stack_addr) : result);
	}
	if (stack_size != 0) {
		result = (result == 0 ? PthreadAttrSetstacksize(dst, stack_size) : result);
	}
	return result;
}

static void PthreadAttrDbgPrint(const PthreadAttr* src) {
	KernelCpumask    mask          = 0;
	int              state         = 0;
	size_t           guard_size    = 0;
	int              inherit_sched = 0;
	KernelSchedParam param         = {};
	int              policy        = 0;
	int              solosched     = 0;
	void*            stack_addr    = nullptr;
	size_t           stack_size    = 0;

	PthreadAttrGetaffinity(src, &mask);
	PthreadAttrGetdetachstate(src, &state);
	PthreadAttrGetguardsize(src, &guard_size);
	PthreadAttrGetinheritsched(src, &inherit_sched);
	PthreadAttrGetschedparam(src, &param);
	PthreadAttrGetschedpolicy(src, &policy);
	PthreadAttrGetsolosched(src, &solosched);
	PthreadAttrGetstackaddr(src, &stack_addr);
	PthreadAttrGetstacksize(src, &stack_size);

	LOGF("\tcpu_mask       = 0x%" PRIx64 "\n"
	     "\tdetach_state   = %d\n"
	     "\tguard_size     = %" PRIu64 "\n"
	     "\tinherit_sched  = %d\n"
	     "\tsched_priority = %d\n"
	     "\tpolicy         = %d\n"
	     "\tsolosched      = %d\n"
	     "\tstack_addr     = 0x%016" PRIx64 "\n"
	     "\tstack_size    = %" PRIu64 "\n",
	     mask, state, guard_size, inherit_sched, param.sched_priority, policy, solosched,
	     reinterpret_cast<uint64_t>(stack_addr), reinterpret_cast<uint64_t>(stack_size));
}

static constexpr int32_t DST_NONE = 0;
static constexpr int32_t DST_MET  = 4;

static int32_t GetDstSeconds() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TIME_ZONE_INFORMATION tzi {};
	const DWORD           result = GetTimeZoneInformation(&tzi);
	return (result == TIME_ZONE_ID_DAYLIGHT ? -tzi.DaylightBias * 60 : 0);
#else
	const std::time_t now = std::time(nullptr);
	std::tm           local_tm {};
	localtime_r(&now, &local_tm);
	return (local_tm.tm_isdst > 0 ? 3600 : 0);
#endif
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
static void sec_to_timeval(KernelTimeval* ts, double sec) {
	ts->tv_sec  = static_cast<int64_t>(sec);
	ts->tv_usec = static_cast<int64_t>((sec - static_cast<double>(ts->tv_sec)) * 1000000.0);
}
#endif

static bool GetPosixClockId(KernelClockid clock_id, clockid_t* out) {
	EXIT_IF(out == nullptr);

	switch (clock_id) {
		case KERNEL_CLOCK_REALTIME:
		case KERNEL_CLOCK_REALTIME_PRECISE:
		case KERNEL_CLOCK_REALTIME_FAST:
		case KERNEL_CLOCK_SECOND: *out = CLOCK_REALTIME; return true;
		case KERNEL_CLOCK_MONOTONIC:
		case KERNEL_CLOCK_UPTIME:
		case KERNEL_CLOCK_UPTIME_PRECISE:
		case KERNEL_CLOCK_UPTIME_FAST:
		case KERNEL_CLOCK_MONOTONIC_PRECISE:
		case KERNEL_CLOCK_MONOTONIC_FAST:
		case KERNEL_CLOCK_EXT_NETWORK:
		case KERNEL_CLOCK_EXT_DEBUG_NETWORK:
		case KERNEL_CLOCK_EXT_AD_NETWORK:
		case KERNEL_CLOCK_EXT_RAW_NETWORK: *out = CLOCK_MONOTONIC; return true;
		default: return false;
	}
}

static bool NativeClockGettime(KernelClockid clock_id, KernelTimespec* tp) {
	EXIT_IF(tp == nullptr);

	int error = OK;
	if (KernelClockGettimeSpecial(clock_id, tp, &error)) {
		return error == OK;
	}

	clockid_t native_clock_id {};
	if (!GetPosixClockId(clock_id, &native_clock_id)) {
		return false;
	}

	timespec native_time {};
	if (::clock_gettime(native_clock_id, &native_time) != 0) {
		return false;
	}

	tp->tv_sec  = native_time.tv_sec;
	tp->tv_nsec = native_time.tv_nsec;
	return true;
}

static int PthreadMutexInitNamed(PthreadMutex* mutex, const PthreadMutexattr* attr,
                                 const char* name, int static_type = 0);
static int PthreadRwlockInitNamed(PthreadRwlock* rwlock, const PthreadRwlockattr* attr,
                                  const char* name);
static int PthreadCondInitNamed(PthreadCond* cond, const PthreadCondattr* attr, const char* name);

static int NativeMutexLock(PthreadMutexPrivate* mutex, KernelUseconds* timeout_us) {
	EXIT_IF(mutex == nullptr);

	auto* self = g_pthread_self;
	EXIT_NOT_IMPLEMENTED(self == nullptr);

	std::unique_lock lock(mutex->m);

	if (mutex->owner == self) {
		if (mutex->type == 2) {
			if (mutex->count == UINT32_MAX) {
				return EAGAIN;
			}
			mutex->count++;
			return OK;
		}
		if (timeout_us != nullptr) {
			if (*timeout_us > 0) {
				mutex->cv.wait_for(lock, std::chrono::microseconds(*timeout_us));
			}
			return ETIMEDOUT;
		}
		return EDEADLK;
	}

	if (timeout_us == nullptr) {
		while (mutex->owner != nullptr) {
			mutex->cv.wait_for(lock, std::chrono::microseconds(SIGNAL_APC_POLL_MICROS));
			if (mutex->owner != nullptr) {
				lock.unlock();
				KernelDispatchPendingSignalForCurrentThread();
				lock.lock();
			}
		}
	} else if (*timeout_us == 0) {
		if (mutex->owner != nullptr) {
			return ETIMEDOUT;
		}
	} else {
		const auto deadline =
		    std::chrono::steady_clock::now() + std::chrono::microseconds(*timeout_us);
		while (mutex->owner != nullptr) {
			const auto now = std::chrono::steady_clock::now();
			if (now >= deadline) {
				return ETIMEDOUT;
			}

			const auto remaining = deadline - now;
			const auto poll      = (remaining < std::chrono::microseconds(SIGNAL_APC_POLL_MICROS)
			                            ? remaining
			                            : std::chrono::steady_clock::duration(
			                                  std::chrono::microseconds(SIGNAL_APC_POLL_MICROS)));
			mutex->cv.wait_for(lock, poll);
			if (mutex->owner != nullptr) {
				lock.unlock();
				KernelDispatchPendingSignalForCurrentThread();
				lock.lock();
			}
		}
	}

	mutex->owner = self;
	mutex->count = 1;
	return OK;
}

static int NativeMutexTrylock(PthreadMutexPrivate* mutex) {
	EXIT_IF(mutex == nullptr);

	auto* self = g_pthread_self;
	EXIT_NOT_IMPLEMENTED(self == nullptr);

	std::unique_lock lock(mutex->m);

	if (mutex->owner == self) {
		if (mutex->type == 2) {
			if (mutex->count == UINT32_MAX) {
				return EAGAIN;
			}
			mutex->count++;
			return OK;
		}
		return EBUSY;
	}

	if (mutex->owner != nullptr) {
		return EBUSY;
	}

	mutex->owner = self;
	mutex->count = 1;
	return OK;
}

static int NativeMutexUnlock(PthreadMutexPrivate* mutex, uint32_t* recurse = nullptr) {
	EXIT_IF(mutex == nullptr);

	auto* self = g_pthread_self;
	EXIT_NOT_IMPLEMENTED(self == nullptr);

	std::unique_lock lock(mutex->m);

	if (mutex->owner != self) {
		return EPERM;
	}

	if (recurse != nullptr) {
		*recurse     = mutex->count;
		mutex->count = 1;
	}

	if (mutex->type == 2 && mutex->count > 1) {
		mutex->count--;
		return OK;
	}

	mutex->owner = nullptr;
	mutex->count = 0;
	lock.unlock();
	mutex->cv.notify_one();

	return OK;
}

static int NativeMutexLockRecurse(PthreadMutexPrivate* mutex, uint32_t recurse) {
	EXIT_IF(mutex == nullptr);

	int result = NativeMutexLock(mutex, nullptr);
	if (result == OK) {
		std::lock_guard lock(mutex->m);
		mutex->count = std::max<uint32_t>(recurse, 1);
	}
	return result;
}

static bool NativeCondDeadlineFromAbs(KernelClockid clock_id, const KernelTimespec* abstime,
                                      std::chrono::steady_clock::time_point* out) {
	if (abstime == nullptr || out == nullptr || abstime->tv_sec < 0 || abstime->tv_nsec < 0 ||
	    abstime->tv_nsec >= 1000000000) {
		return false;
	}

	KernelTimespec now {};
	if (!NativeClockGettime(clock_id, &now)) {
		return false;
	}

	auto delta = std::chrono::seconds(abstime->tv_sec - now.tv_sec) +
	             std::chrono::nanoseconds(abstime->tv_nsec - now.tv_nsec);
	*out       = std::chrono::steady_clock::now() +
	             std::chrono::duration_cast<std::chrono::steady_clock::duration>(delta);
	return true;
}

void* PthreadStaticObjects::CreateObject(void* addr, PthreadStaticObject::Type type) {
	if (addr == nullptr) {
		return addr;
	}

	const uint64_t guest_va = reinterpret_cast<uint64_t>(addr);
	if (Libs::VideoOut::Phase37PostUnregisterSeen() && PthreadCurrentIsMainRelated() &&
	    guest_va >= 0x0000000900000000ULL && guest_va < 0x0000000a00000000ULL) {
		Libs::VideoOut::Phase57NoteMainObjectWrite(guest_va, "CreateObject");
	}

	auto*      current       = static_cast<void**>(addr);
	auto       current_ref   = std::atomic_ref<void*>(*current);
	auto*      current_value = current_ref.load(std::memory_order_acquire);
	const bool adaptive_static_mutex =
	    (type == PthreadStaticObject::Type::Mutex &&
	     current_value == reinterpret_cast<void*>(static_cast<uintptr_t>(1)));

	if (current_value != nullptr && !adaptive_static_mutex) {
		return addr;
	}

	Common::LockGuard lock(m_mutex);

	current_value = current_ref.load(std::memory_order_acquire);
	const bool adaptive_static_mutex_after_lock =
	    (type == PthreadStaticObject::Type::Mutex &&
	     current_value == reinterpret_cast<void*>(static_cast<uintptr_t>(1)));

	if (current_value != nullptr && !adaptive_static_mutex_after_lock) {
		return addr;
	}

	auto* rt      = Common::Singleton<Loader::RuntimeLinker>::Instance();
	auto  vaddr   = reinterpret_cast<uint64_t>(addr);
	auto* program = rt->FindProgramByAddr(vaddr);

	std::string name = fmt::format("Static{:016x}", vaddr);

	int result = OK;
	switch (type) {
		case PthreadStaticObject::Type::Mutex:
			result = PthreadMutexInitNamed(static_cast<PthreadMutex*>(addr), nullptr, name.c_str(),
			                               adaptive_static_mutex_after_lock ? 4 : 0);
			break;
		case PthreadStaticObject::Type::Cond:
			result = PthreadCondInitNamed(static_cast<PthreadCond*>(addr), nullptr, name.c_str());
			break;
		case PthreadStaticObject::Type::Rwlock:
			result =
			    PthreadRwlockInitNamed(static_cast<PthreadRwlock*>(addr), nullptr, name.c_str());
			break;
		default: EXIT("unknown type: %d\n", static_cast<int>(type));
	}

	EXIT_NOT_IMPLEMENTED(result != OK);

	// Heap-backed lazy pthread objects are valid. Initialize them without requiring
	// an owning ELF segment; only segment-backed objects need module-unload cleanup bookkeeping.
	if (program == nullptr) {
		return addr;
	}

	auto* obj    = new PthreadStaticObject;
	obj->program = program;
	obj->type    = type;
	obj->vaddr   = vaddr;

	const auto it = std::find(m_objects.begin(), m_objects.end(), nullptr);
	if (it != m_objects.end()) {
		*it = obj;
	} else {
		m_objects.push_back(obj);
	}

	return addr;
}

void PthreadStaticObjects::DeleteObjects(Loader::Program* program) {
	Common::LockGuard lock(m_mutex);

	for (auto& obj: m_objects) {
		if (obj != nullptr && obj->program == program) {
			int result = OK;
			switch (obj->type) {
				case PthreadStaticObject::Type::Mutex:
					result = PthreadMutexDestroy(reinterpret_cast<PthreadMutex*>(obj->vaddr));
					break;
				case PthreadStaticObject::Type::Cond:
					result = PthreadCondDestroy(reinterpret_cast<PthreadCond*>(obj->vaddr));
					break;
				case PthreadStaticObject::Type::Rwlock:
					result = PthreadRwlockDestroy(reinterpret_cast<PthreadRwlock*>(obj->vaddr));
					break;
				default: EXIT("unknown type: %d\n", static_cast<int>(obj->type));
			}

			EXIT_NOT_IMPLEMENTED(result != OK);

			delete obj;
			obj = nullptr;
		}
	}
}

Pthread PthreadPool::Create() {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_threads) {
		if (p->free) {
			std::lock_guard life(p->life_mutex);
			p->free     = false;
			p->joining  = false;
			p->p_valid  = false;
			p->p        = {};
			p->detached = false;
			p->almost_done = false;
			return p;
		}
	}

	auto* ret = new PthreadPrivate {};

	ret->free        = false;
	ret->joining     = false;
	ret->p_valid     = false;
	ret->detached    = false;
	ret->almost_done = false;
	ret->attr        = nullptr;

	m_threads.push_back(ret);

	return ret;
}

void PthreadPool::FreeDetachedThreads() {
	Common::LockGuard lock(m_mutex);

	for (auto* p: m_threads) {
		if (p->detached && p->almost_done && !p->free) {
			PthreadJoin(p, nullptr);
		}
	}
}

static bool IsSubmissionRelatedName(const std::string& name);

void PthreadPool::DumpSubmissionThreads(const char* reason) {
	Common::LockGuard lock(m_mutex);
	const char*       why = reason != nullptr ? reason : "?";
	for (auto* p: m_threads) {
		if (p == nullptr || p->free || !IsSubmissionRelatedName(p->name)) {
			continue;
		}
		LOGF("SubmitTrace: SubmissionThreadDump reason=%s name=%s tid=%d guest_tid=%d "
		     "entry=0x%016" PRIx64 " arg=0x%016" PRIx64 " waiting_cond=%d cond=%s cond_ptr=0x%016"
		     PRIx64 " almost_done=%d\n",
		     why, p->name.c_str(), p->unique_id, p->guest.thread_id,
		     reinterpret_cast<uint64_t>(p->entry), reinterpret_cast<uint64_t>(p->arg),
		     p->waiting_cond != nullptr ? 1 : 0,
		     p->waiting_cond != nullptr ? p->waiting_cond->name.c_str() : "-",
		     reinterpret_cast<uint64_t>(p->waiting_cond), p->almost_done.load() ? 1 : 0);
		fprintf(stderr,
		        "SubmitTrace: SubmissionThreadDump name=%s tid=%d cond_ptr=0x%016" PRIx64 "\n",
		        p->name.c_str(), p->unique_id, reinterpret_cast<uint64_t>(p->waiting_cond));
	}
}

void PthreadPool::DumpAllGuestThreads(const char* reason) {
	Common::LockGuard lock(m_mutex);
	const char*       why = reason != nullptr ? reason : "?";
	uint32_t          n   = 0;
	for (auto* p: m_threads) {
		if (p == nullptr || p->free) {
			continue;
		}
		LOGF("GuestExit: phase40 threadSnap reason=%s name=%s tid=%d guest_tid=%d "
		     "entry=0x%016" PRIx64 " waiting_cond=%d cond=%s almost_done=%d\n",
		     why, p->name.c_str(), p->unique_id, p->guest.thread_id,
		     reinterpret_cast<uint64_t>(p->entry), p->waiting_cond != nullptr ? 1 : 0,
		     p->waiting_cond != nullptr ? p->waiting_cond->name.c_str() : "-",
		     p->almost_done.load() ? 1 : 0);
		fprintf(stderr, "GuestExit: phase40 threadSnap name=%s tid=%d cond=%s\n", p->name.c_str(),
		        p->unique_id, p->waiting_cond != nullptr ? p->waiting_cond->name.c_str() : "-");
		if (++n >= 64) {
			break;
		}
	}
	LOGF("GuestExit: phase40 threadSnap done count=%u reason=%s\n", n, why);
	fprintf(stderr, "GuestExit: phase40 threadSnap done count=%u\n", n);
}

void PthreadPool::SnapshotGuestThread(int unique_id, uint32_t host_tid_hint, const char* reason) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const char* why = reason != nullptr ? reason : "cfg_post_rip";
	PthreadPrivate* target = nullptr;
	{
		Common::LockGuard lock(m_mutex);
		for (auto* p: m_threads) {
			if (p == nullptr || p->free.load(std::memory_order_acquire) ||
			    p->host_thread_id == 0) {
				continue;
			}
			if (unique_id >= 0 && p->unique_id == unique_id) {
				target = p;
				break;
			}
			if (unique_id < 0 && host_tid_hint != 0 &&
			    p->host_thread_id == static_cast<uint64_t>(host_tid_hint)) {
				target = p;
				break;
			}
		}
		// Fallback: prefer MainThread by name, then any "main", never prefer tid=8 alone
		// (TaskGraphThreadNP 0 is tid=8 on GTA/UE and was a false snapshot target).
		if (target == nullptr) {
			for (auto* p: m_threads) {
				if (p == nullptr || p->free.load(std::memory_order_acquire) ||
				    p->host_thread_id == 0) {
					continue;
				}
				if (p->name == "MainThread") {
					target = p;
					break;
				}
			}
		}
		if (target == nullptr) {
			for (auto* p: m_threads) {
				if (p == nullptr || p->free.load(std::memory_order_acquire) ||
				    p->host_thread_id == 0) {
					continue;
				}
				if (p->name.find("Main") != std::string::npos ||
				    p->name.find("main") != std::string::npos) {
					target = p;
					break;
				}
			}
		}
		if (target == nullptr) {
			char line[192];
			std::snprintf(line, sizeof(line),
			              "SubmitTrace: cfg_post_rip miss reason=%s unique_id=%d host_tid=%u", why,
			              unique_id, host_tid_hint);
			Common::LogFatalToFile(line);
			fprintf(stderr, "%s\n", line);
			return;
		}
	}

	const DWORD host_tid = static_cast<DWORD>(target->host_thread_id);
	if (host_tid == GetCurrentThreadId()) {
		char line[256];
		std::snprintf(line, sizeof(line),
		              "SubmitTrace: cfg_post_rip skip_self reason=%s name=%s tid=%d host_tid=%lu",
		              why, target->name.c_str(), target->unique_id,
		              static_cast<unsigned long>(host_tid));
		Common::LogFatalToFile(line);
		fprintf(stderr, "%s\n", line);
		return;
	}

	HANDLE handle =
	    OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
	               host_tid);
	if (handle == nullptr) {
		char line[256];
		std::snprintf(line, sizeof(line),
		              "SubmitTrace: cfg_post_rip open_fail reason=%s name=%s tid=%d host_tid=%lu "
		              "err=%lu",
		              why, target->name.c_str(), target->unique_id,
		              static_cast<unsigned long>(host_tid),
		              static_cast<unsigned long>(GetLastError()));
		Common::LogFatalToFile(line);
		fprintf(stderr, "%s\n", line);
		return;
	}

	if (SuspendThread(handle) == static_cast<DWORD>(-1)) {
		char line[256];
		std::snprintf(line, sizeof(line),
		              "SubmitTrace: cfg_post_rip suspend_fail reason=%s name=%s tid=%d err=%lu",
		              why, target->name.c_str(), target->unique_id,
		              static_cast<unsigned long>(GetLastError()));
		Common::LogFatalToFile(line);
		fprintf(stderr, "%s\n", line);
		CloseHandle(handle);
		return;
	}

	CONTEXT ctx {};
	ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
	const BOOL got = GetThreadContext(handle, &ctx);
	uint64_t   stack_q[4] = {};
	if (got != FALSE) {
		MEMORY_BASIC_INFORMATION mbi {};
		if (VirtualQuery(reinterpret_cast<const void*>(ctx.Rsp), &mbi, sizeof(mbi)) != 0 &&
		    mbi.State == MEM_COMMIT && (mbi.Protect & PAGE_NOACCESS) == 0) {
			const auto* sp = reinterpret_cast<const uint64_t*>(ctx.Rsp);
			for (int i = 0; i < 4; ++i) {
				stack_q[i] = sp[i];
			}
		}
	}

	char module_name[MAX_PATH] = {};
	uint64_t rva = 0;
	const char* mod_base_name = "(none)";
	if (got != FALSE) {
		HMODULE owner = nullptr;
		if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                       reinterpret_cast<LPCSTR>(ctx.Rip), &owner) != 0 &&
		    owner != nullptr) {
			GetModuleFileNameA(owner, module_name, MAX_PATH);
			rva = static_cast<uint64_t>(ctx.Rip) - reinterpret_cast<uint64_t>(owner);
			mod_base_name = module_name;
			const char* slash = std::strrchr(module_name, '\\');
			if (slash != nullptr && slash[1] != '\0') {
				mod_base_name = slash + 1;
			}
		} else {
			mod_base_name = "(guest_or_anon)";
		}
	}

	const char* cond_name =
	    target->waiting_cond != nullptr ? target->waiting_cond->name.c_str() : "-";
	char line[768];
	if (got != FALSE) {
		std::snprintf(line, sizeof(line),
		              "SubmitTrace: cfg_post_rip reason=%s name=%s tid=%d host_tid=%lu "
		              "almost_done=%d cond=%s rip=0x%016" PRIx64 " rsp=0x%016" PRIx64
		              " rcx=0x%016" PRIx64 " rax=0x%016" PRIx64 " mod=%s rva=0x%016" PRIx64
		              " stack0=0x%016" PRIx64 " stack1=0x%016" PRIx64 " stack2=0x%016" PRIx64
		              " stack3=0x%016" PRIx64,
		              why, target->name.c_str(), target->unique_id,
		              static_cast<unsigned long>(host_tid), target->almost_done.load() ? 1 : 0,
		              cond_name, static_cast<uint64_t>(ctx.Rip), static_cast<uint64_t>(ctx.Rsp),
		              static_cast<uint64_t>(ctx.Rcx), static_cast<uint64_t>(ctx.Rax), mod_base_name,
		              rva, stack_q[0], stack_q[1], stack_q[2], stack_q[3]);
		// #region agent log
		{
			FILE* f = nullptr;
#if defined(_MSC_VER)
			if (fopen_s(&f, "c:\\codes\\KytyPS5-main\\debug-bacc56.log", "a") == 0 && f != nullptr)
#else
			if ((f = fopen("c:\\codes\\KytyPS5-main\\debug-bacc56.log", "a")) != nullptr)
#endif
			{
				std::fprintf(f,
				             "{\"sessionId\":\"bacc56\",\"hypothesisId\":\"E\",\"location\":"
				             "\"pthread.cpp:cfg_post_rip\",\"message\":\"post-CFG Main RIP\","
				             "\"data\":{\"reason\":\"%s\",\"name\":\"%s\",\"rip\":\"0x%016" PRIx64
				             "\",\"rsp\":\"0x%016" PRIx64 "\",\"mod\":\"%s\",\"rva\":\"0x%016" PRIx64
				             "\"},\"timestamp\":%llu}\n",
				             why, target->name.c_str(), static_cast<uint64_t>(ctx.Rip),
				             static_cast<uint64_t>(ctx.Rsp), mod_base_name, rva,
				             static_cast<unsigned long long>(GetTickCount64()));
				std::fclose(f);
			}
		}
		// #endregion
	} else {
		std::snprintf(line, sizeof(line),
		              "SubmitTrace: cfg_post_rip ctx_fail reason=%s name=%s tid=%d host_tid=%lu "
		              "err=%lu cond=%s almost_done=%d",
		              why, target->name.c_str(), target->unique_id,
		              static_cast<unsigned long>(host_tid),
		              static_cast<unsigned long>(GetLastError()), cond_name,
		              target->almost_done.load() ? 1 : 0);
	}
	Common::LogFatalToFile(line);
	fprintf(stderr, "%s\n", line);
	std::fflush(stderr);

	ResumeThread(handle);
	CloseHandle(handle);
#else
	(void)unique_id;
	(void)host_tid_hint;
	(void)reason;
#endif
}

void PthreadSnapshotGuestThread(int unique_id, uint32_t host_tid_hint, const char* reason) {
	if (g_pthread_context == nullptr || g_pthread_context->GetPthreadPool() == nullptr) {
		return;
	}
	g_pthread_context->GetPthreadPool()->SnapshotGuestThread(unique_id, host_tid_hint, reason);
}

// Phase 26: Mixed/Compute sit on an anonymous job-queue cond that is never signaled pre-submit.
// Wake those waiters once when synthetic EOP / AGC user interrupt runs.
// Also wake Main-related (incl. unique_id==8 / "VMem" on MCL) — otherwise queue_empty stall
// kicks only Mixed/Compute (woken=0) while Main stays CondWait and submit_gpu freezes ~60–70.
// PPSA21564: also wake Draw*/Havok/Gfx producer-pipeline waiters (see IsProducerPipelineName).
size_t PthreadPool::WakeSubmissionCondWaiters() {
	Common::LockGuard lock(m_mutex);
	size_t            woken = 0;
	for (auto* p: m_threads) {
		if (p == nullptr || p->free || p->waiting_cond == nullptr) {
			continue;
		}
		if (!IsProducerPipelineName(p->name) && !IsMainRelatedThread(p)) {
			continue;
		}
		auto* cond = p->waiting_cond;
		bool  notify = false;
		{
			std::lock_guard cond_lock(cond->m);
			notify = CondWakeWaiter(cond, p);
		}
		if (notify) {
			cond->cv.notify_all();
			++woken;
			LOGF("SubmitTrace: WakeSubmissionCond name=%s tid=%d role=%s cond_ptr=0x%016" PRIx64
			     "\n",
			     p->name.c_str(), p->unique_id, Phase54RoleOf(p),
			     reinterpret_cast<uint64_t>(cond));
			fprintf(stderr, "SubmitTrace: WakeSubmissionCond name=%s tid=%d role=%s\n",
			        p->name.c_str(), p->unique_id, Phase54RoleOf(p));
			Libs::VideoOut::Phase50NoteWake(p->name.c_str(), 1);
		}
	}
	return woken;
}

bool PthreadCurrentIsSubmissionRelated() {
	auto* self = g_pthread_self;
	return self != nullptr && IsSubmissionRelatedName(self->name);
}

bool PthreadCurrentIsMainRelated() {
	auto* self = g_pthread_self;
	if (self == nullptr) {
		return false;
	}
	// MainThread or unique_id==8 (BootCards / producer in TLOU traces).
	return self->unique_id == 8 || self->name == "MainThread" ||
	       self->name.find("Main") != std::string::npos ||
	       self->name.find("BootCards") != std::string::npos;
}

bool PthreadMainThreadAlive() {
	auto alive_one = [](const PthreadPrivate* p) -> bool {
		if (p == nullptr || p->free.load(std::memory_order_acquire) ||
		    p->almost_done.load(std::memory_order_acquire) || p->host_thread_id == 0) {
			return false;
		}
		return p->unique_id == 8 || p->name == "MainThread";
	};
	if (alive_one(g_pthread_main)) {
		return true;
	}
	if (g_pthread_context == nullptr || g_pthread_context->GetPthreadPool() == nullptr) {
		return false;
	}
	return g_pthread_context->GetPthreadPool()->HasAliveMainThread();
}

bool PthreadPool::HasAliveMainThread() {
	Common::LockGuard lock(m_mutex);
	for (auto* p: m_threads) {
		if (p == nullptr || p->free.load(std::memory_order_acquire) ||
		    p->almost_done.load(std::memory_order_acquire) || p->host_thread_id == 0) {
			continue;
		}
		if (p->unique_id == 8 || p->name == "MainThread") {
			return true;
		}
	}
	return false;
}

void PthreadDumpSubmissionThreads(const char* reason) {
	static std::atomic<uint32_t> dumps {0};
	if (dumps.fetch_add(1, std::memory_order_relaxed) >= 4) {
		return;
	}
	if (g_pthread_context == nullptr || g_pthread_context->GetPthreadPool() == nullptr) {
		return;
	}
	g_pthread_context->GetPthreadPool()->DumpSubmissionThreads(reason);
}

void PthreadDumpAllGuestThreads(const char* reason) {
	if (g_pthread_context == nullptr || g_pthread_context->GetPthreadPool() == nullptr) {
		return;
	}
	g_pthread_context->GetPthreadPool()->DumpAllGuestThreads(reason);
}

void PthreadFormatCurrentGuest(char* out, size_t out_size) {
	if (out == nullptr || out_size == 0) {
		return;
	}
	auto* self = g_pthread_self;
	if (self == nullptr) {
		std::snprintf(out, out_size, "name=- tid=- cond=-");
		return;
	}
	std::snprintf(out, out_size, "name=%s tid=%d cond=%s", self->name.c_str(), self->unique_id,
	              self->waiting_cond != nullptr ? self->waiting_cond->name.c_str() : "-");
}

size_t PthreadWakeSubmissionCondWaiters() {
	static std::atomic<uint32_t> wakes {0};
	// Empty-queue re-wait is expected; a few kicks are enough to prove / attempt progress.
	if (wakes.fetch_add(1, std::memory_order_relaxed) >= 4) {
		return 0;
	}
	if (g_pthread_context == nullptr || g_pthread_context->GetPthreadPool() == nullptr) {
		return 0;
	}
	return g_pthread_context->GetPthreadPool()->WakeSubmissionCondWaiters();
}

size_t PthreadWakeSubmissionCondWaitersAfterFlip() {
	static std::atomic<uint32_t> wakes {0};
	// Separate budget from bootstrap IRQ wakes — post-blank handshake needs its own kicks.
	if (wakes.fetch_add(1, std::memory_order_relaxed) >= 8) {
		return 0;
	}
	if (g_pthread_context == nullptr || g_pthread_context->GetPthreadPool() == nullptr) {
		return 0;
	}
	return g_pthread_context->GetPthreadPool()->WakeSubmissionCondWaiters();
}

size_t PthreadWakeSubmissionCondWaitersUnlimited() {
	if (!Libs::VideoOut::Phase54AllowHostWake()) {
		static std::atomic<uint32_t> skips {0};
		const uint32_t               sn = skips.fetch_add(1, std::memory_order_relaxed);
		if (sn < 8 || (sn % 64) == 0) {
			LOGF("SubmitTrace: phase54 wake_budget_skip unlimited cycle=%" PRIu64 " n=%u\n",
			     Libs::VideoOut::Phase54CurrentCycleId(), sn);
			fprintf(stderr, "SubmitTrace: phase54 wake_budget_skip unlimited n=%u\n", sn);
		}
		return 0;
	}
	Libs::VideoOut::Phase54NoteHostWake("unlimited");
	if (g_pthread_context == nullptr || g_pthread_context->GetPthreadPool() == nullptr) {
		return 0;
	}
	return g_pthread_context->GetPthreadPool()->WakeSubmissionCondWaiters();
}

size_t PthreadPool::SuspendAllGuests() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Common::LockGuard lock(m_mutex);
	size_t            suspended = 0;
	const DWORD       self_tid  = GetCurrentThreadId();
	for (auto* p: m_threads) {
		if (p == nullptr || p->free || p->host_thread_id == 0 ||
		    p->host_thread_id == self_tid) {
			continue;
		}
		HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME, FALSE,
		                           static_cast<DWORD>(p->host_thread_id));
		if (handle == nullptr) {
			continue;
		}
		if (SuspendThread(handle) != static_cast<DWORD>(-1)) {
			++suspended;
		}
		CloseHandle(handle);
	}
	return suspended;
#else
	return 0;
#endif
}

size_t PthreadPool::ResumeAllGuests() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Common::LockGuard lock(m_mutex);
	size_t            resumed = 0;
	const DWORD       self_tid = GetCurrentThreadId();
	for (auto* p: m_threads) {
		if (p == nullptr || p->free || p->host_thread_id == 0 ||
		    p->host_thread_id == self_tid) {
			continue;
		}
		HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME, FALSE,
		                           static_cast<DWORD>(p->host_thread_id));
		if (handle == nullptr) {
			continue;
		}
		// Resume until the suspend count reaches 0 (best-effort).
		for (;;) {
			const DWORD prev = ResumeThread(handle);
			if (prev == static_cast<DWORD>(-1) || prev <= 1) {
				if (prev != static_cast<DWORD>(-1) && prev == 1) {
					++resumed;
				}
				break;
			}
		}
		CloseHandle(handle);
	}
	return resumed;
#else
	return 0;
#endif
}

size_t PthreadSuspendAllGuests() {
	if (g_pthread_context == nullptr || g_pthread_context->GetPthreadPool() == nullptr) {
		return 0;
	}
	return g_pthread_context->GetPthreadPool()->SuspendAllGuests();
}

size_t PthreadResumeAllGuests() {
	if (g_pthread_context == nullptr || g_pthread_context->GetPthreadPool() == nullptr) {
		return 0;
	}
	return g_pthread_context->GetPthreadPool()->ResumeAllGuests();
}

static bool IsMainRelatedPthread(const PthreadPrivate* p) {
	if (p == nullptr) {
		return false;
	}
	return p->unique_id == 8 || p->name == "MainThread" ||
	       p->name.find("Main") != std::string::npos ||
	       p->name.find("BootCards") != std::string::npos;
}

size_t PthreadPool::SuspendMainRelatedGuests() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Common::LockGuard lock(m_mutex);
	size_t            suspended = 0;
	const DWORD       self_tid  = GetCurrentThreadId();
	for (auto* p: m_threads) {
		if (p == nullptr || p->free || p->host_thread_id == 0 ||
		    p->host_thread_id == self_tid || !IsMainRelatedPthread(p)) {
			continue;
		}
		HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME, FALSE,
		                           static_cast<DWORD>(p->host_thread_id));
		if (handle == nullptr) {
			continue;
		}
		if (SuspendThread(handle) != static_cast<DWORD>(-1)) {
			++suspended;
			LOGF("FlipTrace: suspended main-related name=%s tid=%d\n", p->name.c_str(),
			     p->unique_id);
			fprintf(stderr, "FlipTrace: suspended main-related name=%s tid=%d\n", p->name.c_str(),
			        p->unique_id);
		}
		CloseHandle(handle);
	}
	return suspended;
#else
	return 0;
#endif
}

size_t PthreadPool::ResumeMainRelatedGuests() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Common::LockGuard lock(m_mutex);
	size_t            resumed  = 0;
	const DWORD       self_tid = GetCurrentThreadId();
	for (auto* p: m_threads) {
		if (p == nullptr || p->free || p->host_thread_id == 0 ||
		    p->host_thread_id == self_tid || !IsMainRelatedPthread(p)) {
			continue;
		}
		HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME, FALSE,
		                           static_cast<DWORD>(p->host_thread_id));
		if (handle == nullptr) {
			continue;
		}
		for (;;) {
			const DWORD prev = ResumeThread(handle);
			if (prev == static_cast<DWORD>(-1) || prev <= 1) {
				if (prev != static_cast<DWORD>(-1) && prev == 1) {
					++resumed;
				}
				break;
			}
		}
		CloseHandle(handle);
	}
	return resumed;
#else
	return 0;
#endif
}

size_t PthreadSuspendMainRelatedGuests() {
	size_t suspended = 0;
	if (g_pthread_context != nullptr && g_pthread_context->GetPthreadPool() != nullptr) {
		suspended += g_pthread_context->GetPthreadPool()->SuspendMainRelatedGuests();
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// MainThread is not in the pthread pool — suspend it explicitly.
	if (g_pthread_main != nullptr && g_pthread_main->host_thread_id != 0 &&
	    g_pthread_main->host_thread_id != GetCurrentThreadId()) {
		HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME, FALSE,
		                           static_cast<DWORD>(g_pthread_main->host_thread_id));
		if (handle != nullptr) {
			if (SuspendThread(handle) != static_cast<DWORD>(-1)) {
				++suspended;
				LOGF("FlipTrace: suspended MainThread tid=%d\n", g_pthread_main->unique_id);
				fprintf(stderr, "FlipTrace: suspended MainThread tid=%d\n",
				        g_pthread_main->unique_id);
			}
			CloseHandle(handle);
		}
	}
#endif
	return suspended;
}

size_t PthreadResumeMainRelatedGuests() {
	size_t resumed = 0;
	if (g_pthread_context != nullptr && g_pthread_context->GetPthreadPool() != nullptr) {
		resumed += g_pthread_context->GetPthreadPool()->ResumeMainRelatedGuests();
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (g_pthread_main != nullptr && g_pthread_main->host_thread_id != 0 &&
	    g_pthread_main->host_thread_id != GetCurrentThreadId()) {
		HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME, FALSE,
		                           static_cast<DWORD>(g_pthread_main->host_thread_id));
		if (handle != nullptr) {
			for (;;) {
				const DWORD prev = ResumeThread(handle);
				if (prev == static_cast<DWORD>(-1) || prev <= 1) {
					if (prev != static_cast<DWORD>(-1) && prev == 1) {
						++resumed;
					}
					break;
				}
			}
			CloseHandle(handle);
		}
	}
#endif
	return resumed;
}

bool PthreadKeys::Create(int* key, pthread_key_destructor_func_t destructor) {
	EXIT_IF(key == nullptr);

	Common::LockGuard lock(m_mutex);

	for (int index = 0; index < KEYS_MAX; index++) {
		if (!m_keys[index].used) {
			*key                     = index;
			m_keys[index].used       = true;
			m_keys[index].destructor = destructor;
			m_keys[index].specific_values.clear();
			return true;
		}
	}

	return false;
}

bool PthreadKeys::Delete(int key) {
	Common::LockGuard lock(m_mutex);

	if (key < 0 || key >= KEYS_MAX || !m_keys[key].used) {
		return false;
	}

	m_keys[key].used       = false;
	m_keys[key].destructor = nullptr;
	m_keys[key].specific_values.clear();

	return true;
}

void PthreadKeys::Destruct(int thread_id) {
	struct CallInfo {
		pthread_key_destructor_func_t destructor;
		void*                         data;
	};

	for (int iter = 0; iter < DESTRUCTOR_ITERATIONS; iter++) {
		std::vector<CallInfo> delete_list;

		{
			Common::LockGuard lock(m_mutex);

			for (auto& key: m_keys) {
				if (key.used && key.destructor != nullptr) {
					for (auto& v: key.specific_values) {
						if (v.thread_id == thread_id && v.data != nullptr) {
							delete_list.push_back(CallInfo {key.destructor, v.data});
							v.data = nullptr;
						}
					}
				}
			}
		}

		if (delete_list.empty()) {
			return;
		}

		for (auto& d: delete_list) {
			d.destructor(d.data);
		}
	}

	Common::LockGuard lock(m_mutex);

	for (auto& key: m_keys) {
		auto& values = key.specific_values;
		values.erase(std::remove_if(values.begin(), values.end(),
		                            [thread_id](const Map& v) { return v.thread_id == thread_id; }),
		             values.end());
	}
}

bool PthreadKeys::Set(int key, int thread_id, void* data) {
	Common::LockGuard lock(m_mutex);

	if (key < 0 || key >= KEYS_MAX || !m_keys[key].used) {
		return false;
	}

	for (auto& v: m_keys[key].specific_values) {
		if (v.thread_id == thread_id) {
			v.data = data;
			return true;
		}
	}

	m_keys[key].specific_values.push_back(Map({thread_id, data}));

	return true;
}

bool PthreadKeys::Get(int key, int thread_id, void** data) {
	EXIT_IF(data == nullptr);

	Common::LockGuard lock(m_mutex);

	if (key < 0 || key >= KEYS_MAX || !m_keys[key].used) {
		return false;
	}

	for (auto& v: m_keys[key].specific_values) {
		if (v.thread_id == thread_id) {
			*data = v.data;
			return true;
		}
	}

	*data = nullptr;

	return true;
}

int KYTY_SYSV_ABI PthreadMutexattrInit(PthreadMutexattr* attr) {
	// PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(attr == nullptr);

	*attr = new PthreadMutexattrPrivate {};

	int result = pthread_mutexattr_init(&(*attr)->p);

	result = (result == 0 ? PthreadMutexattrSettype(attr, 1) : result);
	result = (result == 0 ? PthreadMutexattrSetprotocol(attr, 0) : result);

	switch (result) {
		case 0: return OK;
		case ENOMEM: return KERNEL_ERROR_ENOMEM;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadMutexattrDestroy(PthreadMutexattr* attr) {
	// PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(attr == nullptr || *attr == nullptr);

	int result = pthread_mutexattr_destroy(&(*attr)->p);

	delete *attr;
	*attr = nullptr;

	switch (result) {
		case 0: return OK;
		case ENOMEM: return KERNEL_ERROR_ENOMEM;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadMutexattrSettype(PthreadMutexattr* attr, int type) {
	// PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(attr == nullptr || *attr == nullptr);

	int ptype = PTHREAD_MUTEX_DEFAULT;
	switch (type) {
		case KERNEL_PTHREAD_MUTEX_ERRORCHECK: ptype = PTHREAD_MUTEX_ERRORCHECK; break;
		case KERNEL_PTHREAD_MUTEX_RECURSIVE: ptype = PTHREAD_MUTEX_RECURSIVE; break;
		case KERNEL_PTHREAD_MUTEX_NORMAL:
		case KERNEL_PTHREAD_MUTEX_ADAPTIVE: ptype = PTHREAD_MUTEX_NORMAL; break;
		default: return KERNEL_ERROR_EINVAL;
	}

	int result = pthread_mutexattr_settype(&(*attr)->p, ptype);

	if (result == 0) {
		(*attr)->type = type;
		return OK;
	}

	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadMutexattrSetprotocol([[maybe_unused]] PthreadMutexattr* attr,
                                              int                                protocol) {
	// PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(attr == nullptr || *attr == nullptr);

	[[maybe_unused]] int pprotocol = PTHREAD_PRIO_NONE;
	switch (protocol) {
		case 0: pprotocol = PTHREAD_PRIO_NONE; break;
		case 1: pprotocol = PTHREAD_PRIO_INHERIT; break;
		case 2: pprotocol = PTHREAD_PRIO_PROTECT; break;
		default: EXIT("invalid protocol: %d\n", protocol);
	}

	// protocol doesn't work in winpthreads
	int result         = 0; // pthread_mutexattr_setprotocol(&(*attr)->p, pprotocol);
	(*attr)->pprotocol = pprotocol;

	if (result == 0) {
		return OK;
	}

	return KERNEL_ERROR_EINVAL;
}

static int PthreadMutexInitNamed(PthreadMutex* mutex, const PthreadMutexattr* attr,
                                 const char* name, int static_type) {
	if (name != nullptr && name[0] != '\0') {
		PRINT_NAME();
	}

	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	if (mutex == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	PthreadMutexattrPrivate static_attr {};
	PthreadMutexattr        static_attr_ptr = nullptr;

	if (static_type != 0) {
		static_attr_ptr = &static_attr;
		pthread_mutexattr_init(&static_attr.p);
		PthreadMutexattrSettype(&static_attr_ptr, static_type);
		PthreadMutexattrSetprotocol(&static_attr_ptr, 0);
		attr = &static_attr_ptr;
	} else if (attr == nullptr) {

		attr = g_pthread_context->GetDefaultMutexattr();
	}

	auto* new_mutex = new PthreadMutexPrivate {};

	new_mutex->name      = (name != nullptr ? name : "");
	new_mutex->type      = (*attr)->type;
	new_mutex->pprotocol = (*attr)->pprotocol;

	int result = 0;

	if (static_attr_ptr != nullptr) {
		pthread_mutexattr_destroy(&static_attr.p);
	}

	if (name != nullptr && name[0] != '\0') {
		LOGF("\tmutex init: %s, %d\n", new_mutex->name.c_str(), result);
	}

	std::atomic_ref<PthreadMutexPrivate*>(*mutex).store(new_mutex, std::memory_order_release);

	switch (result) {
		case 0: return OK;
		case EAGAIN: return KERNEL_ERROR_EAGAIN;
		case EINVAL: return KERNEL_ERROR_EINVAL;
		case ENOMEM: return KERNEL_ERROR_ENOMEM;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadMutexInit(PthreadMutex* mutex, const PthreadMutexattr* attr) {
	return PthreadMutexInitNamed(mutex, attr, "");
}

int KYTY_SYSV_ABI PthreadMutexDestroy(PthreadMutex* mutex) {
	// Hot path for Python/Ren'Py startup; keep this quiet unless it fails.

	if (mutex == nullptr || *mutex == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	int result = ((*mutex)->owner == nullptr ? 0 : EBUSY);

	if (result != 0) {
		LOGF("\tmutex destroy: %s, %d\n", (*mutex)->name.c_str(), result);
		switch (result) {
			case EBUSY: return KERNEL_ERROR_EBUSY;
			case EINVAL:
			default: return KERNEL_ERROR_EINVAL;
		}
	}

	delete *mutex;
	*mutex = nullptr;

	switch (result) {
		case 0: return OK;
		case EBUSY: return KERNEL_ERROR_EBUSY;
		case EINVAL:
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadMutexLock(PthreadMutex* mutex) {
	// PRINT_NAME();

	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	mutex = static_cast<PthreadMutex*>(
	    pthread_static_objects->CreateObject(mutex, PthreadStaticObject::Type::Mutex));

	if (mutex == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*mutex == nullptr);

	int result = NativeMutexLock(*mutex, nullptr);

	// LOGF("\tmutex lock: %s, %d\n", (*mutex)->name.c_str(), result);

	switch (result) {
		case 0: return OK;
		case EAGAIN: return KERNEL_ERROR_EAGAIN;
		case EINVAL: return KERNEL_ERROR_EINVAL;
		case EDEADLK: return KERNEL_ERROR_EDEADLK;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadMutexTrylock(PthreadMutex* mutex) {
	// PRINT_NAME();

	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	mutex = static_cast<PthreadMutex*>(
	    pthread_static_objects->CreateObject(mutex, PthreadStaticObject::Type::Mutex));

	if (mutex == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*mutex == nullptr);

	int result = NativeMutexTrylock(*mutex);

	// LOGF("\tmutex trylock: %s, %d\n", (*mutex)->name.c_str(), result);

	switch (result) {
		case 0: return OK;
		case EAGAIN: return KERNEL_ERROR_EAGAIN;
		case EBUSY: return KERNEL_ERROR_EBUSY;
		case EINVAL:
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadMutexTimedlock(PthreadMutex* mutex, KernelUseconds usec) {
	// PRINT_NAME();

	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	mutex = static_cast<PthreadMutex*>(
	    pthread_static_objects->CreateObject(mutex, PthreadStaticObject::Type::Mutex));

	if (mutex == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*mutex == nullptr);

	int result = NativeMutexLock(*mutex, &usec);

	switch (result) {
		case 0: return OK;
		case ETIMEDOUT: return KERNEL_ERROR_ETIMEDOUT;
		case EAGAIN: return KERNEL_ERROR_EAGAIN;
		case EDEADLK: return KERNEL_ERROR_EDEADLK;
		case EINVAL:
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadMutexUnlock(PthreadMutex* mutex) {
	// PRINT_NAME();

	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	mutex = static_cast<PthreadMutex*>(
	    pthread_static_objects->CreateObject(mutex, PthreadStaticObject::Type::Mutex));

	if (mutex == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*mutex == nullptr);

	int result = NativeMutexUnlock(*mutex);

	if (result != 0) {
		LOGF("\tmutex unlock: %s, %d, thread_id = %d\n", (*mutex)->name.c_str(), result,
		     Common::Thread::GetThreadIdUnique());
	}

	switch (result) {
		case 0: return OK;

		case EINVAL: return KERNEL_ERROR_EINVAL;
		case EPERM: return KERNEL_ERROR_EPERM;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadAttrInit(PthreadAttr* attr) {
	// PRINT_NAME();

	*attr = new PthreadAttrPrivate {};

	int result = pthread_attr_init(&(*attr)->p);

	(*attr)->affinity       = 0x7f;
	(*attr)->guard_size     = 0x1000;
	(*attr)->stack_addr     = nullptr;
	(*attr)->stack_size     = PTHREAD_STACK_DEFAULT;
	(*attr)->stack_user     = false;
	(*attr)->stack_map_addr = 0;
	(*attr)->stack_map_size = 0;

	KernelSchedParam param;
	param.sched_priority = 700;

	result = (result == 0 ? PthreadAttrSetinheritsched(attr, 4) : result);
	result = (result == 0 ? PthreadAttrSetschedparam(attr, &param) : result);
	result = (result == 0 ? PthreadAttrSetschedpolicy(attr, 1) : result);
	result = (result == 0 ? PthreadAttrSetdetachstate(attr, 0) : result);
	result = (result == 0 ? PthreadAttrSetstacksize(attr, PTHREAD_STACK_DEFAULT) : result);

	if (false && PRINT_NAME_ENABLED) {
		PthreadAttrDbgPrint(attr);
	}

	switch (result) {
		case 0: return OK;
		case ENOMEM: return KERNEL_ERROR_ENOMEM;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadAttrDestroy(PthreadAttr* attr) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrDestroy");
	if (attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	int result = pthread_attr_destroy(&attr_value->p);

	delete *attr;
	*attr = nullptr;

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadAttrGet(Pthread thread, PthreadAttr* attr) {
	// PRINT_NAME();

	if (thread == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	return PthreadAttrCopy(attr, &thread->attr);
}

int KYTY_SYSV_ABI PthreadAttrGetaffinity(const PthreadAttr* attr, KernelCpumask* mask) {
	// PRINT_NAME();

	if (mask == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*mask = (*attr)->affinity;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrGetdetachstate(const PthreadAttr* attr, int* state) {
	// PRINT_NAME();

	if (state == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	// int result = pthread_attr_getdetachstate(&(*attr)->p, state);
	int result = 0;

	*state = ((*attr)->detached ? PTHREAD_CREATE_DETACHED : PTHREAD_CREATE_JOINABLE);

	switch (*state) {
		case PTHREAD_CREATE_JOINABLE: *state = 0; break;
		case PTHREAD_CREATE_DETACHED: *state = 1; break;
		default: EXIT("unknown state: %d\n", *state);
	}

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadAttrGetguardsize(const PthreadAttr* attr, size_t* guard_size) {
	// PRINT_NAME();

	if (guard_size == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*guard_size = (*attr)->guard_size;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrGetinheritsched(const PthreadAttr* attr, int* inherit_sched) {
	// PRINT_NAME();

	if (inherit_sched == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*inherit_sched = (*attr)->inherit_sched;

	switch (*inherit_sched) {
		case 0:
		case 4: break;
		default: EXIT("unknown inherit_sched: %d\n", *inherit_sched);
	}

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrGetschedparam(const PthreadAttr* attr, KernelSchedParam* param) {
	// PRINT_NAME();

	if (param == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	int result = pthread_attr_getschedparam(&(*attr)->p, param);

	if (param->sched_priority <= -2) {
		param->sched_priority = 767;
	} else if (param->sched_priority >= +2) {
		param->sched_priority = 256;
	} else {
		param->sched_priority = 700;
	}

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadAttrGetschedpolicy(const PthreadAttr* attr, int* policy) {
	// PRINT_NAME();

	if (policy == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	int result = pthread_attr_getschedpolicy(&(*attr)->p, policy);

	switch (*policy) {
		case SCHED_OTHER: *policy = (*attr)->policy; break;
		case SCHED_FIFO: *policy = 1; break;
		case SCHED_RR: *policy = 3; break;
		default: EXIT("unknown policy: %d\n", *policy);
	}

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadAttrGetsolosched(const PthreadAttr* attr, int* solosched) {
	// PRINT_NAME();

	if (solosched == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*solosched = (*attr)->solosched;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrGetstack(const PthreadAttr* __restrict attr,
                                      void** __restrict stack_addr, size_t* __restrict stack_size) {
	// PRINT_NAME();

	if (stack_size == nullptr || stack_addr == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*stack_addr = (*attr)->stack_addr;
	*stack_size = (*attr)->stack_size;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrGetstackaddr(const PthreadAttr* attr, void** stack_addr) {
	// PRINT_NAME();

	if (stack_addr == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*stack_addr = (*attr)->stack_addr;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrGetstacksize(const PthreadAttr* attr, size_t* stack_size) {
	// PRINT_NAME();

	if (stack_size == nullptr || attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*stack_size = (*attr)->stack_size;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrSetaffinity(PthreadAttr* attr, KernelCpumask mask) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrSetaffinity");
	if (attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	attr_value->affinity = mask;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrSetdetachstate(PthreadAttr* attr, int state) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrSetdetachstate");
	if (attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	int pstate = PTHREAD_CREATE_JOINABLE;
	switch (state) {
		case 0: pstate = PTHREAD_CREATE_JOINABLE; break;
		case 1: pstate = PTHREAD_CREATE_DETACHED; break;
		default: EXIT("unknown state: %d\n", state);
	}

	// int result = pthread_attr_setdetachstate(&(*attr)->p, pstate);
	int result = 0;

	attr_value->detached = (pstate == PTHREAD_CREATE_DETACHED);

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadAttrSetguardsize(PthreadAttr* attr, size_t guard_size) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrSetguardsize");
	if (attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	attr_value->guard_size = guard_size;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrSetinheritsched(PthreadAttr* attr, int inherit_sched) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrSetinheritsched");
	if (attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	int pinherit_sched = PTHREAD_INHERIT_SCHED;
	switch (inherit_sched) {
		case 0: pinherit_sched = PTHREAD_EXPLICIT_SCHED; break;
		case 4: pinherit_sched = PTHREAD_INHERIT_SCHED; break;
		default: EXIT("unknown inherit_sched: %d\n", inherit_sched);
	}

	// Keep this in Kyty state. winpthreads' inheritsched support is not needed for guest-visible
	// behavior, and some guest attr flows can leave the native attr in a shape that crashes inside
	// winpthreads.
	attr_value->inherit_sched = inherit_sched;
	(void)pinherit_sched;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrSetschedparam(PthreadAttr* attr, const KernelSchedParam* param) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrSetschedparam");
	if (param == nullptr || attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	KernelSchedParam pparam {};
	if (param->sched_priority <= 478) {
		pparam.sched_priority = +2;
	} else if (param->sched_priority >= 733) {
		pparam.sched_priority = -2;
	} else {
		pparam.sched_priority = 0;
	}

	int result = pthread_attr_setschedparam(&attr_value->p, &pparam);

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadAttrSetschedpolicy(PthreadAttr* attr, int policy) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrSetschedpolicy");
	if (attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	// winpthreads supports only SCHED_OTHER policy
	int ppolicy = SCHED_OTHER;

	attr_value->policy = policy;

	int result = pthread_attr_setschedpolicy(&attr_value->p, ppolicy);

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadAttrSetsolosched(PthreadAttr* attr, int solosched) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrSetsolosched");
	if (attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	attr_value->solosched = solosched;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrSetstack(PthreadAttr* attr, void* addr, size_t size) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrSetstack");
	if (addr == nullptr || size < PTHREAD_STACK_MIN || attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	attr_value->stack_addr     = addr;
	attr_value->stack_size     = size;
	attr_value->stack_user     = true;
	attr_value->stack_map_addr = 0;
	attr_value->stack_map_size = 0;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrSetstackaddr(PthreadAttr* attr, void* addr) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrSetstackaddr");
	if (addr == nullptr || attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	attr_value->stack_addr     = addr;
	attr_value->stack_user     = true;
	attr_value->stack_map_addr = 0;
	attr_value->stack_map_size = 0;

	return OK;
}

int KYTY_SYSV_ABI PthreadAttrSetstacksize(PthreadAttr* attr, size_t stack_size) {
	// PRINT_NAME();

	auto* attr_value = GetPthreadAttrValue(attr, "PthreadAttrSetstacksize");
	if (stack_size < PTHREAD_STACK_MIN || attr_value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	attr_value->stack_size = stack_size;

	return OK;
}

int KYTY_SYSV_ABI PthreadRwlockDestroy(PthreadRwlock* rwlock) {
	PRINT_NAME();

	if (rwlock == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*rwlock == nullptr);

	{
		std::lock_guard lock((*rwlock)->m);
		if ((*rwlock)->writer != nullptr || (*rwlock)->reader_count != 0) {
			return KERNEL_ERROR_EBUSY;
		}
	}

	LOGF("\trwlock destroy: %s, 0\n", (*rwlock)->name.c_str());

	delete *rwlock;
	*rwlock = nullptr;

	return OK;
}

static int PthreadRwlockInitNamed(PthreadRwlock* rwlock, const PthreadRwlockattr* attr,
                                  const char* name) {
	PRINT_NAME();

	if (rwlock == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	if (attr == nullptr) {

		attr = g_pthread_context->GetDefaultRwlockattr();
	}

	auto* new_rwlock = new PthreadRwlockPrivate {};

	new_rwlock->name = (name != nullptr ? name : "");

	LOGF("\trwlock init: %s, 0\n", new_rwlock->name.c_str());

	std::atomic_ref<PthreadRwlockPrivate*>(*rwlock).store(new_rwlock, std::memory_order_release);

	return OK;
}

int KYTY_SYSV_ABI PthreadRwlockInit(PthreadRwlock* rwlock, const PthreadRwlockattr* attr) {
	return PthreadRwlockInitNamed(rwlock, attr, "");
}

static PthreadRwlockPrivate::Reader* RwlockFindReader(PthreadRwlock rwlock, Pthread thread) {
	EXIT_IF(rwlock == nullptr);

	for (auto& reader: rwlock->readers) {
		if (reader.thread == thread) {
			return &reader;
		}
	}

	return nullptr;
}

static void RwlockAddReader(PthreadRwlock rwlock, Pthread thread) {
	if (auto* reader = RwlockFindReader(rwlock, thread); reader != nullptr) {
		reader->count++;
	} else {
		rwlock->readers.push_back({thread, 1});
	}
	rwlock->reader_count++;
}

static bool RwlockRemoveReader(PthreadRwlock rwlock, Pthread thread) {
	for (auto it = rwlock->readers.begin(); it != rwlock->readers.end(); ++it) {
		if (it->thread == thread) {
			it->count--;
			rwlock->reader_count--;
			if (it->count == 0) {
				rwlock->readers.erase(it);
			}
			return true;
		}
	}

	return false;
}

static int RwlockLockCooperative(PthreadRwlock rwlock, bool write, KernelUseconds* timeout_us) {
	EXIT_IF(rwlock == nullptr);

	auto* self = g_pthread_self;
	if (self == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	const auto has_timeout = (timeout_us != nullptr);
	const auto deadline =
	    (has_timeout ? std::chrono::steady_clock::now() + std::chrono::microseconds(*timeout_us)
	                 : std::chrono::steady_clock::time_point {});
	bool writer_wait_registered = false;

	for (;;) {
		{
			std::unique_lock lock(rwlock->m);

			auto unregister_writer_wait = [&rwlock, &writer_wait_registered] {
				if (writer_wait_registered) {
					rwlock->waiting_writers--;
					writer_wait_registered = false;
				}
			};

			if (write) {
				if (rwlock->writer == self || RwlockFindReader(rwlock, self) != nullptr) {
					unregister_writer_wait();
					return KERNEL_ERROR_EDEADLK;
				}
				if (rwlock->writer == nullptr && rwlock->reader_count == 0) {
					unregister_writer_wait();
					rwlock->writer       = self;
					rwlock->writer_count = 1;
					return OK;
				}
				if (!writer_wait_registered) {
					rwlock->waiting_writers++;
					writer_wait_registered = true;
				}
			} else {
				const bool already_reader = (RwlockFindReader(rwlock, self) != nullptr);
				if ((rwlock->writer == nullptr &&
				     (rwlock->waiting_writers == 0 || already_reader)) ||
				    rwlock->writer == self) {
					RwlockAddReader(rwlock, self);
					return OK;
				}
			}

			if (has_timeout) {
				const auto now = std::chrono::steady_clock::now();
				if (*timeout_us == 0 || now >= deadline) {
					unregister_writer_wait();
					return KERNEL_ERROR_ETIMEDOUT;
				}

				const auto remaining = deadline - now;
				const auto poll = (remaining < std::chrono::microseconds(SIGNAL_APC_POLL_MICROS)
				                       ? remaining
				                       : std::chrono::steady_clock::duration(
				                             std::chrono::microseconds(SIGNAL_APC_POLL_MICROS)));
				rwlock->cv.wait_for(lock, poll);
			} else {
				rwlock->cv.wait_for(lock, std::chrono::microseconds(SIGNAL_APC_POLL_MICROS));
			}
		}

		KernelDispatchPendingSignalForCurrentThread();
	}
}

int KYTY_SYSV_ABI PthreadRwlockRdlock(PthreadRwlock* rwlock) {
	// Hot path for some PS5 titles; per-call name logging can dominate runtime.

	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	rwlock = static_cast<PthreadRwlock*>(
	    pthread_static_objects->CreateObject(rwlock, PthreadStaticObject::Type::Rwlock));

	if (rwlock == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*rwlock == nullptr);

	return RwlockLockCooperative(*rwlock, false, nullptr);
}

int KYTY_SYSV_ABI PthreadRwlockTimedrdlock(PthreadRwlock* rwlock, KernelUseconds usec) {
	PRINT_NAME();

	if (rwlock == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*rwlock == nullptr);

	return RwlockLockCooperative(*rwlock, false, &usec);
}

int KYTY_SYSV_ABI PthreadRwlockTimedwrlock(PthreadRwlock* rwlock, KernelUseconds usec) {
	PRINT_NAME();

	if (rwlock == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*rwlock == nullptr);

	return RwlockLockCooperative(*rwlock, true, &usec);
}

int KYTY_SYSV_ABI PthreadRwlockTryrdlock(PthreadRwlock* rwlock) {
	PRINT_NAME();

	if (rwlock == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*rwlock == nullptr);

	auto* self = g_pthread_self;
	if (self == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	{
		std::lock_guard lock((*rwlock)->m);
		const bool      already_reader = (RwlockFindReader(*rwlock, self) != nullptr);
		if (((*rwlock)->writer == nullptr && ((*rwlock)->waiting_writers == 0 || already_reader)) ||
		    (*rwlock)->writer == self) {
			RwlockAddReader(*rwlock, self);
			return OK;
		}
	}

	return KERNEL_ERROR_EBUSY;
}

int KYTY_SYSV_ABI PthreadRwlockTrywrlock(PthreadRwlock* rwlock) {
	PRINT_NAME();

	if (rwlock == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*rwlock == nullptr);

	auto* self = g_pthread_self;
	if (self == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	{
		std::lock_guard lock((*rwlock)->m);
		if ((*rwlock)->writer == self || RwlockFindReader(*rwlock, self) != nullptr) {
			return KERNEL_ERROR_EDEADLK;
		}
		if ((*rwlock)->writer == nullptr && (*rwlock)->reader_count == 0) {
			(*rwlock)->writer       = self;
			(*rwlock)->writer_count = 1;
			return OK;
		}
	}

	return KERNEL_ERROR_EBUSY;
}

int KYTY_SYSV_ABI PthreadRwlockUnlock(PthreadRwlock* rwlock) {
	// PRINT_NAME();

	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	rwlock = static_cast<PthreadRwlock*>(
	    pthread_static_objects->CreateObject(rwlock, PthreadStaticObject::Type::Rwlock));

	if (rwlock == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*rwlock == nullptr);

	auto* self = g_pthread_self;
	if (self == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	{
		std::lock_guard lock((*rwlock)->m);
		if (RwlockRemoveReader(*rwlock, self)) {
			(*rwlock)->cv.notify_all();
			return OK;
		}
		if ((*rwlock)->writer == self) {
			EXIT_IF((*rwlock)->writer_count == 0);
			(*rwlock)->writer_count--;
			if ((*rwlock)->writer_count == 0) {
				(*rwlock)->writer = nullptr;
				(*rwlock)->cv.notify_all();
			}
			return OK;
		}

		return KERNEL_ERROR_EPERM;
	}
}

int KYTY_SYSV_ABI PthreadRwlockWrlock(PthreadRwlock* rwlock) {
	// PRINT_NAME();

	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	rwlock = static_cast<PthreadRwlock*>(
	    pthread_static_objects->CreateObject(rwlock, PthreadStaticObject::Type::Rwlock));

	if (rwlock == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*rwlock == nullptr);

	return RwlockLockCooperative(*rwlock, true, nullptr);
}

int KYTY_SYSV_ABI PthreadRwlockattrDestroy(PthreadRwlockattr* attr) {
	PRINT_NAME();

	if (attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*attr == nullptr);

	int result = pthread_rwlockattr_destroy(&(*attr)->p);

	delete *attr;
	*attr = nullptr;

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadRwlockattrInit(PthreadRwlockattr* attr) {
	PRINT_NAME();

	*attr = new PthreadRwlockattrPrivate {};

	int result = pthread_rwlockattr_init(&(*attr)->p);

	result = (result == 0 ? PthreadRwlockattrSettype(attr, 1) : result);

	switch (result) {
		case 0: return OK;
		case ENOMEM: return KERNEL_ERROR_ENOMEM;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadRwlockattrGettype(PthreadRwlockattr* attr, int* type) {
	PRINT_NAME();

	if (type == nullptr || attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*type = (*attr)->type;

	return OK;
}

int KYTY_SYSV_ABI PthreadRwlockattrSettype(PthreadRwlockattr* attr, int type) {
	PRINT_NAME();

	if (attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	(*attr)->type = type;

	return OK;
}

int KYTY_SYSV_ABI PthreadCondattrDestroy(PthreadCondattr* attr) {
	PRINT_NAME();

	if (attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*attr == nullptr);

	int result = pthread_condattr_destroy(&(*attr)->p);

	delete *attr;
	*attr = nullptr;

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadCondattrInit(PthreadCondattr* attr) {
	PRINT_NAME();

	*attr = new PthreadCondattrPrivate {};

	int result        = pthread_condattr_init(&(*attr)->p);
	(*attr)->clock_id = KERNEL_CLOCK_REALTIME;

	switch (result) {
		case 0: return OK;
		case ENOMEM: return KERNEL_ERROR_ENOMEM;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadCondattrSetclock(PthreadCondattr* attr, KernelClockid clock_id) {
	PRINT_NAME();

	if (attr == nullptr || *attr == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	clockid_t pclock_id = {};
	if (!GetPosixClockId(clock_id, &pclock_id)) {
		LOGF("\tunknown clock_id: %d\n", clock_id);
		return KERNEL_ERROR_EINVAL;
	}

	int result        = pthread_condattr_setclock(&(*attr)->p, pclock_id);
	(*attr)->clock_id = clock_id;

	LOGF("\tcondattr setclock: clock_id = %d, native = %d, result = %d\n", clock_id,
	     static_cast<int>(pclock_id), result);

	if (result == EINVAL && pclock_id == CLOCK_MONOTONIC) {
		LOGF("\tcondattr setclock: host rejected CLOCK_MONOTONIC, keeping emulated attribute\n");
		return OK;
	}

	switch (result) {
		case 0: return OK;
		case EINVAL:
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadCondBroadcast(PthreadCond* cond) {
	// PRINT_NAME();

	const uint64_t guest_cond_va = reinterpret_cast<uint64_t>(cond);
	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	cond = static_cast<PthreadCond*>(
	    pthread_static_objects->CreateObject(cond, PthreadStaticObject::Type::Cond));

	if (cond == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*cond == nullptr);

	int  result = 0;
	bool notify = false;
	bool mixed_waiter = false;
	char waiters_buf[96] = "-";
	{
		std::lock_guard lock((*cond)->m);
		size_t          wpos = 0;
		waiters_buf[0]       = '\0';
		for (auto* waiter: (*cond)->waiters) {
			if (waiter != nullptr && IsSubmissionRelatedName(waiter->name)) {
				mixed_waiter = true;
			}
			if (waiter != nullptr && wpos + 24 < sizeof(waiters_buf)) {
				wpos += static_cast<size_t>(std::snprintf(
				    waiters_buf + wpos, sizeof(waiters_buf) - wpos, "%s%s",
				    wpos == 0 ? "" : ",", Phase54RoleOf(waiter)));
			}
		}
		if (waiters_buf[0] == '\0') {
			std::snprintf(waiters_buf, sizeof(waiters_buf), "-");
		}
		if (!(*cond)->waiters.empty()) {
			(*cond)->sequence++;
			CondClearWaiters(*cond);
			notify = true;
		}
	}
	if (mixed_waiter) {
		static std::atomic<uint32_t> logs {0};
		if (logs.fetch_add(1, std::memory_order_relaxed) < 32) {
			auto* self = g_pthread_self;
			LOGF("SubmitTrace: MixedCondBroadcast cond_ptr=0x%016" PRIx64 " cond=%s by=%s "
			     "notify=%d\n",
			     reinterpret_cast<uint64_t>(*cond), (*cond)->name.c_str(),
			     self != nullptr ? self->name.c_str() : "?", notify ? 1 : 0);
			fprintf(stderr, "SubmitTrace: MixedCondBroadcast cond_ptr=0x%016" PRIx64 " by=%s\n",
			        reinterpret_cast<uint64_t>(*cond), self != nullptr ? self->name.c_str() : "?");
		}
	}
	(void)Phase54NoteCondSignal(*cond, g_pthread_self, notify, true, waiters_buf);
	Libs::VideoOut::Phase56NoteMainSignal(guest_cond_va, Phase54RoleOf(g_pthread_self));
	if (notify) {
		(*cond)->cv.notify_all();
	}

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadCondDestroy(PthreadCond* cond) {
	PRINT_NAME();

	if (cond == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*cond == nullptr);

	int result = 0;

	LOGF("\tcond destroy: %s, %d\n", (*cond)->name.c_str(), result);

	delete *cond;
	*cond = nullptr;

	switch (result) {
		case 0: return OK;
		case EINVAL: return KERNEL_ERROR_EINVAL;
		case EBUSY: return KERNEL_ERROR_EBUSY;
		default: return KERNEL_ERROR_EINVAL;
	}
}

static int PthreadCondInitNamed(PthreadCond* cond, const PthreadCondattr* attr, const char* name) {
	if (name != nullptr && name[0] != '\0') {
		PRINT_NAME();
	}

	if (cond == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	if (attr == nullptr) {

		attr = g_pthread_context->GetDefaultCondattr();
	}

	auto* new_cond = new PthreadCondPrivate {};

	new_cond->name     = (name != nullptr ? name : "");
	new_cond->clock_id = (*attr)->clock_id;

	int result = 0;

	if (name != nullptr && name[0] != '\0') {
		LOGF("\tcond init: %s, cond=0x%016" PRIx64 ", caller=0x%016" PRIx64 ", %d\n",
		     new_cond->name.c_str(), reinterpret_cast<uint64_t>(cond),
		     reinterpret_cast<uint64_t>(__builtin_return_address(0)), result);
	}

	std::atomic_ref<PthreadCondPrivate*>(*cond).store(new_cond, std::memory_order_release);

	switch (result) {
		case 0: return OK;
		case EAGAIN: return KERNEL_ERROR_EAGAIN;
		case EINVAL: return KERNEL_ERROR_EINVAL;
		case ENOMEM: return KERNEL_ERROR_ENOMEM;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadCondInit(PthreadCond* cond, const PthreadCondattr* attr) {
	return PthreadCondInitNamed(cond, attr, "");
}

int KYTY_SYSV_ABI PthreadCondSignal(PthreadCond* cond) {
	// PRINT_NAME();

	const uint64_t guest_cond_va = reinterpret_cast<uint64_t>(cond);
	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	cond = static_cast<PthreadCond*>(
	    pthread_static_objects->CreateObject(cond, PthreadStaticObject::Type::Cond));

	if (cond == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*cond == nullptr);

	int  result = 0;
	bool notify = false;
	bool mixed_waiter = false;
	char waiters_buf[96] = "-";
	{
		std::lock_guard lock((*cond)->m);
		size_t          wpos = 0;
		waiters_buf[0]       = '\0';
		for (auto* waiter: (*cond)->waiters) {
			if (waiter != nullptr && IsSubmissionRelatedName(waiter->name)) {
				mixed_waiter = true;
			}
			if (waiter != nullptr && wpos + 24 < sizeof(waiters_buf)) {
				wpos += static_cast<size_t>(std::snprintf(
				    waiters_buf + wpos, sizeof(waiters_buf) - wpos, "%s%s",
				    wpos == 0 ? "" : ",", Phase54RoleOf(waiter)));
			}
		}
		if (waiters_buf[0] == '\0') {
			std::snprintf(waiters_buf, sizeof(waiters_buf), "-");
		}
		notify = CondWakeWaiter(*cond, nullptr);
	}
	if (mixed_waiter) {
		static std::atomic<uint32_t> logs {0};
		if (logs.fetch_add(1, std::memory_order_relaxed) < 32) {
			auto* self = g_pthread_self;
			LOGF("SubmitTrace: MixedCondSignal cond_ptr=0x%016" PRIx64 " cond=%s by=%s notify=%d\n",
			     reinterpret_cast<uint64_t>(*cond), (*cond)->name.c_str(),
			     self != nullptr ? self->name.c_str() : "?", notify ? 1 : 0);
			fprintf(stderr,
			        "SubmitTrace: MixedCondSignal cond_ptr=0x%016" PRIx64 " by=%s notify=%d\n",
			        reinterpret_cast<uint64_t>(*cond), self != nullptr ? self->name.c_str() : "?",
			        notify ? 1 : 0);
		}
	}
	(void)Phase54NoteCondSignal(*cond, g_pthread_self, notify, false, waiters_buf);
	Libs::VideoOut::Phase56NoteMainSignal(guest_cond_va, Phase54RoleOf(g_pthread_self));
	if (notify) {
		(*cond)->cv.notify_all();
	}

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadCondSignalto(PthreadCond* cond, Pthread thread) {
	// PRINT_NAME();

	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	cond = static_cast<PthreadCond*>(
	    pthread_static_objects->CreateObject(cond, PthreadStaticObject::Type::Cond));

	if (cond == nullptr || thread == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*cond == nullptr);

	int  result = 0;
	bool notify = false;
	{
		std::lock_guard lock((*cond)->m);
		notify = CondWakeWaiter(*cond, thread);
	}
	if (notify) {
		(*cond)->cv.notify_all();
	}

	// LOGF("\tcond signalto: %s(0x%016" PRIx64 "), %d\n", (*cond)->name.c_str(),
	// reinterpret_cast<uint64_t>(cond), result);

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

static void Phase55MaybeNoteGuestSync(PthreadCond* guest_cond_before, PthreadMutex* guest_mutex_before,
                                      Pthread thread) {
	if (thread == nullptr || !IsSubmissionRelatedName(thread->name)) {
		return;
	}
	Libs::VideoOut::Phase56NoteGuestSync(reinterpret_cast<uint64_t>(guest_cond_before),
	                                     reinterpret_cast<uint64_t>(guest_mutex_before),
	                                     reinterpret_cast<uint64_t>(thread->arg),
	                                     Phase54RoleOf(thread));
}

static void Phase55MaybeNoteGuestCond(PthreadCond* guest_cond_before_create, Pthread thread) {
	if (guest_cond_before_create == nullptr || thread == nullptr ||
	    !IsSubmissionRelatedName(thread->name)) {
		return;
	}
	Libs::VideoOut::Phase55NoteGuestCond(reinterpret_cast<uint64_t>(guest_cond_before_create),
	                                     reinterpret_cast<uint64_t>(thread->arg),
	                                     Phase54RoleOf(thread));
}

int KYTY_SYSV_ABI PthreadCondTimedwait(PthreadCond* cond, PthreadMutex* mutex,
                                       KernelUseconds usec) {
	// PRINT_NAME();
	Phase70CaptureGuestRipFromAbi();

	const uint64_t guest_cond_va  = reinterpret_cast<uint64_t>(cond);
	const uint64_t guest_mutex_va = reinterpret_cast<uint64_t>(mutex);
	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	cond = static_cast<PthreadCond*>(
	    pthread_static_objects->CreateObject(cond, PthreadStaticObject::Type::Cond));
	mutex = static_cast<PthreadMutex*>(
	    pthread_static_objects->CreateObject(mutex, PthreadStaticObject::Type::Mutex));

	if (cond == nullptr || mutex == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*cond == nullptr);
	EXIT_NOT_IMPLEMENTED(*mutex == nullptr);

	auto* cond_value  = *cond;
	auto* mutex_value = *mutex;

	if (mutex_value->owner != g_pthread_self) {
		return KERNEL_ERROR_EPERM;
	}

	Phase55MaybeNoteGuestSync(reinterpret_cast<PthreadCond*>(guest_cond_va),
	                          reinterpret_cast<PthreadMutex*>(guest_mutex_va), g_pthread_self);
	Phase55MaybeNoteGuestCond(reinterpret_cast<PthreadCond*>(guest_cond_va), g_pthread_self);

	std::unique_lock cond_lock(cond_value->m);
	const auto       sequence        = cond_value->sequence;
	auto*            thread          = g_pthread_self;
	const auto       thread_sequence = thread->cond_sequence;
	CondAddWaiter(cond_value, thread);

	uint32_t recurse = 0;
	int      result  = NativeMutexUnlock(mutex_value, &recurse);
	if (result != OK) {
		CondRemoveWaiter(cond_value, thread);
		return (result == EPERM ? KERNEL_ERROR_EPERM : KERNEL_ERROR_EINVAL);
	}

	const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(usec);
	auto       ready    = [cond_value, thread, sequence, thread_sequence] {
		return cond_value->sequence != sequence || thread->cond_sequence != thread_sequence;
	};

	if (usec == 0) {
		result = ETIMEDOUT;
	} else {
		while (!ready()) {
			const auto now = std::chrono::steady_clock::now();
			if (now >= deadline) {
				break;
			}

			const auto remaining = deadline - now;
			const auto poll      = (remaining < std::chrono::microseconds(SIGNAL_APC_POLL_MICROS)
			                            ? remaining
			                            : std::chrono::steady_clock::duration(
			                                  std::chrono::microseconds(SIGNAL_APC_POLL_MICROS)));
			cond_value->cv.wait_for(cond_lock, poll);

			if (!ready()) {
				cond_lock.unlock();
				KernelDispatchPendingSignalForCurrentThread();
				cond_lock.lock();
			}
		}

		result = (ready() ? OK : ETIMEDOUT);
	}
	CondRemoveWaiter(cond_value, thread);
	cond_lock.unlock();

	if (result == OK && thread != nullptr && !thread->name.empty()) {
		Libs::VideoOut::Phase35TryGuestMenuFromSubmissionThread(thread->name.c_str());
	}
	Phase54OnCondWaitExit(thread, cond_value, result == OK ? OK : result);

	int lock_result = NativeMutexLockRecurse(mutex_value, recurse);
	if (result == OK) {
		result = lock_result;
	}

	// LOGF("\tcond timedwait: %s, %d\n", (*cond)->name.c_str(), result);

	switch (result) {
		case 0: return OK;
		case ETIMEDOUT: return KERNEL_ERROR_ETIMEDOUT;
		case EPERM: return KERNEL_ERROR_EPERM;
		case EINVAL:
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadCondTimedwaitAbs(PthreadCond* cond, PthreadMutex* mutex,
                                          const KernelTimespec* abstime) {
	// PRINT_NAME();
	Phase70CaptureGuestRipFromAbi();

	const uint64_t guest_cond_va  = reinterpret_cast<uint64_t>(cond);
	const uint64_t guest_mutex_va = reinterpret_cast<uint64_t>(mutex);
	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	cond = static_cast<PthreadCond*>(
	    pthread_static_objects->CreateObject(cond, PthreadStaticObject::Type::Cond));
	mutex = static_cast<PthreadMutex*>(
	    pthread_static_objects->CreateObject(mutex, PthreadStaticObject::Type::Mutex));

	if (cond == nullptr || mutex == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*cond == nullptr);
	EXIT_NOT_IMPLEMENTED(*mutex == nullptr);

	auto* cond_value  = *cond;
	auto* mutex_value = *mutex;

	std::chrono::steady_clock::time_point deadline {};
	if (!NativeCondDeadlineFromAbs(cond_value->clock_id, abstime, &deadline)) {
		return KERNEL_ERROR_EINVAL;
	}

	if (mutex_value->owner != g_pthread_self) {
		return KERNEL_ERROR_EPERM;
	}

	Phase55MaybeNoteGuestSync(reinterpret_cast<PthreadCond*>(guest_cond_va),
	                          reinterpret_cast<PthreadMutex*>(guest_mutex_va), g_pthread_self);
	Phase55MaybeNoteGuestCond(reinterpret_cast<PthreadCond*>(guest_cond_va), g_pthread_self);

	std::unique_lock cond_lock(cond_value->m);
	const auto       sequence        = cond_value->sequence;
	auto*            thread          = g_pthread_self;
	const auto       thread_sequence = thread->cond_sequence;
	CondAddWaiter(cond_value, thread);

	uint32_t recurse = 0;
	int      result  = NativeMutexUnlock(mutex_value, &recurse);
	if (result != OK) {
		CondRemoveWaiter(cond_value, thread);
		return (result == EPERM ? KERNEL_ERROR_EPERM : KERNEL_ERROR_EINVAL);
	}

	auto ready = [cond_value, thread, sequence, thread_sequence] {
		return cond_value->sequence != sequence || thread->cond_sequence != thread_sequence;
	};

	while (!ready()) {
		const auto now = std::chrono::steady_clock::now();
		if (now >= deadline) {
			break;
		}

		const auto remaining = deadline - now;
		const auto poll      = (remaining < std::chrono::microseconds(SIGNAL_APC_POLL_MICROS)
		                            ? remaining
		                            : std::chrono::steady_clock::duration(
		                                  std::chrono::microseconds(SIGNAL_APC_POLL_MICROS)));
		cond_value->cv.wait_for(cond_lock, poll);

		if (!ready()) {
			cond_lock.unlock();
			KernelDispatchPendingSignalForCurrentThread();
			cond_lock.lock();
		}
	}

	result = (ready() ? OK : ETIMEDOUT);
	CondRemoveWaiter(cond_value, thread);
	cond_lock.unlock();

	if (result == OK && thread != nullptr && !thread->name.empty()) {
		Libs::VideoOut::Phase35TryGuestMenuFromSubmissionThread(thread->name.c_str());
	}
	Phase54OnCondWaitExit(thread, cond_value, result);

	int lock_result = NativeMutexLockRecurse(mutex_value, recurse);
	if (result == OK) {
		result = lock_result;
	}

	switch (result) {
		case 0: return OK;
		case ETIMEDOUT: return KERNEL_ERROR_ETIMEDOUT;
		case EPERM: return KERNEL_ERROR_EPERM;
		case EINVAL:
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadCondWait(PthreadCond* cond, PthreadMutex* mutex) {
	PRINT_NAME();
	Phase70CaptureGuestRipFromAbi();

	const uint64_t guest_cond_va  = reinterpret_cast<uint64_t>(cond);
	const uint64_t guest_mutex_va = reinterpret_cast<uint64_t>(mutex);
	auto* pthread_static_objects = g_pthread_context->GetPthreadStaticObjects();

	cond = static_cast<PthreadCond*>(
	    pthread_static_objects->CreateObject(cond, PthreadStaticObject::Type::Cond));
	mutex = static_cast<PthreadMutex*>(
	    pthread_static_objects->CreateObject(mutex, PthreadStaticObject::Type::Mutex));

	if (cond == nullptr || mutex == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(*cond == nullptr);
	EXIT_NOT_IMPLEMENTED(*mutex == nullptr);

	auto* cond_value  = *cond;
	auto* mutex_value = *mutex;

	if (mutex_value->owner != g_pthread_self) {
		return KERNEL_ERROR_EPERM;
	}

	std::unique_lock cond_lock(cond_value->m);
	const auto       sequence        = cond_value->sequence;
	auto*            thread          = g_pthread_self;
	const auto       thread_sequence = thread->cond_sequence;
	if (thread != nullptr && IsSubmissionRelatedName(thread->name)) {
		Phase55MaybeNoteGuestSync(reinterpret_cast<PthreadCond*>(guest_cond_va),
		                          reinterpret_cast<PthreadMutex*>(guest_mutex_va), thread);
		Libs::VideoOut::Phase55NoteGuestCond(guest_cond_va,
		                                     reinterpret_cast<uint64_t>(thread->arg),
		                                     Phase54RoleOf(thread));
	}
	CondAddWaiter(cond_value, thread);

	uint32_t recurse = 0;
	int      result  = NativeMutexUnlock(mutex_value, &recurse);
	if (result != OK) {
		CondRemoveWaiter(cond_value, thread);
		return (result == EPERM ? KERNEL_ERROR_EPERM : KERNEL_ERROR_EINVAL);
	}

	auto ready = [cond_value, thread, sequence, thread_sequence] {
		return cond_value->sequence != sequence || thread->cond_sequence != thread_sequence;
	};

	while (!ready()) {
		cond_value->cv.wait_for(cond_lock, std::chrono::microseconds(SIGNAL_APC_POLL_MICROS));
		if (!ready()) {
			cond_lock.unlock();
			KernelDispatchPendingSignalForCurrentThread();
			cond_lock.lock();
		}
	}
	CondRemoveWaiter(cond_value, thread);
	cond_lock.unlock();

	// Phase 35: after Mixed Submission wakes post-Unregister, run guest menu path once.
	if (thread != nullptr && !thread->name.empty()) {
		Libs::VideoOut::Phase35TryGuestMenuFromSubmissionThread(thread->name.c_str());
	}
	Phase54OnCondWaitExit(thread, cond_value, OK);

	result = NativeMutexLockRecurse(mutex_value, recurse);

	switch (result) {
		case 0: return OK;
		case EPERM: return KERNEL_ERROR_EPERM;
		case EINVAL:
		default: return KERNEL_ERROR_EINVAL;
	}
}

Pthread KYTY_SYSV_ABI PthreadSelf() {
	// PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(g_pthread_self == nullptr);

	return g_pthread_self;
}

Pthread PthreadSelfOrNull() {
	return g_pthread_self;
}

Pthread PthreadSwapSelfForSignal(Pthread thread) {
	auto* previous = g_pthread_self;
	g_pthread_self = thread;
	return previous;
}

int PthreadGetUniqueId(Pthread thread) {
	return thread != nullptr ? thread->unique_id : 0;
}

uint64_t PthreadGetHostThreadId(Pthread thread) {
	return thread != nullptr ? thread->host_thread_id : 0;
}

void PthreadQueuePendingSignal(Pthread thread, int signum) {
	if (thread == nullptr || signum < 0 || signum >= 64) {
		return;
	}

	thread->pending_signal_mask.fetch_or(1ull << static_cast<uint32_t>(signum),
	                                     std::memory_order_release);
}

bool PthreadHasPendingSignal(Pthread thread, int signum) {
	if (thread == nullptr || signum < 0 || signum >= 64) {
		return false;
	}

	const auto mask = 1ull << static_cast<uint32_t>(signum);
	return (thread->pending_signal_mask.load(std::memory_order_acquire) & mask) != 0;
}

bool PthreadTakePendingSignal(Pthread thread, int signum) {
	if (thread == nullptr || signum < 0 || signum >= 64) {
		return false;
	}

	const auto mask = 1ull << static_cast<uint32_t>(signum);
	return (thread->pending_signal_mask.fetch_and(~mask, std::memory_order_acq_rel) & mask) != 0;
}

bool PthreadGetGuestStack(Pthread thread, uint64_t* stack_addr, uint64_t* stack_size) {
	if (thread == nullptr || thread->attr == nullptr || stack_addr == nullptr ||
	    stack_size == nullptr) {
		return false;
	}

	auto* addr = thread->attr->stack_addr;
	auto  size = thread->attr->stack_size;
	if (addr == nullptr || size == 0) {
		return false;
	}

	*stack_addr = reinterpret_cast<uint64_t>(addr);
	*stack_size = static_cast<uint64_t>(size);
	return true;
}

int PthreadGetPriorityForKernel(Pthread thread) {
	if (thread == nullptr) {
		return 700;
	}

	std::lock_guard lock(thread->life_mutex);
	if (thread->free || thread->joining || !thread->p_valid) {
		return 700;
	}
	return 700;
}

int PthreadGetCurrentPriorityForKernel() {
	return PthreadGetPriorityForKernel(g_pthread_self);
}

static void CleanupThread(void* arg) {
	auto* thread = static_cast<Pthread>(arg);

	auto thread_dtors = g_pthread_context->GetThreadDtors();

	if (thread_dtors != nullptr) {
		thread_dtors();
	}

	g_pthread_context->GetPthreadKeys()->Destruct(thread->unique_id);

	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
	rt->DeleteTlss(thread->unique_id);

	thread->almost_done = true;
}

static void* RunThread(void* arg) {
	auto* thread = static_cast<Pthread>(arg);
	void* ret    = nullptr;

	thread->unique_id = Common::Thread::GetThreadIdUnique();

	g_pthread_self = thread;

	uint64_t os_thread_id = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	os_thread_id = static_cast<uint64_t>(GetCurrentThreadId());
#endif
	thread->host_thread_id = os_thread_id;

	LOGF("\tPthread run begin: %s, id = %d, os_thread_id = %" PRIu64 ", entry = 0x%016" PRIx64
	     ", arg = 0x%016" PRIx64 ", stack_addr = 0x%016" PRIx64 ", stack_size = %" PRIu64 "\n",
	     thread->name.c_str(), thread->unique_id, os_thread_id,
	     reinterpret_cast<uint64_t>(thread->entry), reinterpret_cast<uint64_t>(thread->arg),
	     reinterpret_cast<uint64_t>(thread->attr->stack_addr),
	     static_cast<uint64_t>(thread->attr->stack_size));

	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
	pthread_cleanup_push(CleanupThread, thread);

	auto* stack_top = reinterpret_cast<void*>(
	    (reinterpret_cast<uintptr_t>(thread->attr->stack_addr) + thread->attr->stack_size) &
	    ~static_cast<uintptr_t>(0x0f));
	ret = RunOnGuestStack(thread->arg, thread->entry, stack_top);

	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
	pthread_cleanup_pop(1);

	return ret;
}

int KYTY_SYSV_ABI PthreadCreate(Pthread* thread, const PthreadAttr* attr,
                                pthread_entry_func_t entry, void* arg, const char* name) {
	PRINT_NAME();

	if (thread == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	auto* pthread_pool = g_pthread_context->GetPthreadPool();

	if (attr == nullptr) {
		attr = g_pthread_context->GetDefaultAttr();
	}
	if (GetPthreadAttrValue(attr, "PthreadCreate") == nullptr) {
		attr = g_pthread_context->GetDefaultAttr();
	}

	PRINT_NAME_ENABLE(false);

	auto* created_thread = pthread_pool->Create();
	*thread              = created_thread;

	if (created_thread->attr != nullptr) {
		PthreadAttrDestroy(&created_thread->attr);
	}

	PthreadAttrInit(&created_thread->attr);

	int result = PthreadAttrCopy(&created_thread->attr, attr);

	if (result == 0) {
		EXIT_IF(created_thread->free);

		if (created_thread->attr->stack_addr == nullptr) {
			created_thread->attr->stack_size += PTHREAD_STACK_EXTRA;
		}

		result = CreateGuestStack(created_thread->attr);
	}

	if (result == 0) {
		created_thread->name            = (name != nullptr ? name : "");
		created_thread->entry           = entry;
		created_thread->arg             = arg;
		created_thread->guest.thread_id = ++g_pthread_thread_id;
		created_thread->almost_done     = false;
		created_thread->detached        = created_thread->attr->detached;
		created_thread->unique_id       = -1;

		// Phase 55: arm Mixed/Compute entry trampoline before first run.
		if (reinterpret_cast<uint64_t>(entry) == Libs::VideoOut::kPhase55MixedEntry) {
			Libs::VideoOut::Phase55TryArmMixedThunk();
		}

		result =
		    pthread_create(&created_thread->p, &created_thread->attr->p, RunThread, created_thread);

		if (result == 0) {
			created_thread->p_valid = true;
		} else {
			FreeGuestStack(created_thread->attr);
		}
	}

	if (result != 0) {
		created_thread->p_valid = false;
		created_thread->free    = true;
	}

	LOGF("\tthread create: %s, id = %d, %d\n", created_thread->name.c_str(),
	     created_thread->unique_id, result);

	PthreadAttrDbgPrint(&created_thread->attr);

	PRINT_NAME_ENABLE(true);

	if (result < 0) {
		return result;
	}

	switch (result) {
		case 0: return OK;
		case ENOMEM: return KERNEL_ERROR_ENOMEM;
		case EAGAIN: return KERNEL_ERROR_EAGAIN;
		case EDEADLK: return KERNEL_ERROR_EDEADLK;
		case EPERM: return KERNEL_ERROR_EPERM;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadDetach(Pthread thread) {
	PRINT_NAME();

	if (thread == nullptr || thread->free || thread->joining) {
		return KERNEL_ERROR_EINVAL;
	}

	LOGF("\tthread detach: %s, %d\n", thread->name.c_str(), 0);

	thread->detached = true;

	return OK;
}

int KYTY_SYSV_ABI PthreadJoin(Pthread thread, void** value) {
	PRINT_NAME();

	if (thread == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	pthread_t native {};
	{
		std::lock_guard lock(thread->life_mutex);
		if (thread->free || thread->joining.exchange(true)) {
			return KERNEL_ERROR_ESRCH;
		}
		native = thread->p;
	}

	int result = pthread_join(native, value);

	if (PRINT_NAME_ENABLED) {
		LOGF("\tthread join: %s, %d\n", thread->name.c_str(), result);
	}

	{
		std::lock_guard lock(thread->life_mutex);
		if (result == 0) {
			FreeGuestStack(thread->attr);

			thread->almost_done = false;
			thread->p_valid     = false;
			// Mark free before clearing joining so Setprio/Getprio never observe a joined
			// pthread_t with free==false (TOCTOU → winpthread/ntdll heap corruption).
			thread->free = true;
		}
		thread->joining = false;
	}

	switch (result) {
		case 0: return OK;
		case ESRCH: return KERNEL_ERROR_ESRCH;
		case EDEADLK: return KERNEL_ERROR_EDEADLK;
		case EOPNOTSUPP: return KERNEL_ERROR_EOPNOTSUPP;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadCancel(Pthread thread) {
	PRINT_NAME();

	if (thread == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	std::lock_guard lock(thread->life_mutex);
	if (thread->free || thread->joining || !thread->p_valid) {
		return KERNEL_ERROR_ESRCH;
	}

	int result = pthread_cancel(thread->p);

	LOGF("\tthread cancel: %s, %d\n", thread->name.c_str(), result);

	switch (result) {
		case 0: return OK;
		case ESRCH: return KERNEL_ERROR_ESRCH;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int KYTY_SYSV_ABI PthreadSetaffinity(Pthread thread, KernelCpumask mask) {
	PRINT_NAME();

	if (thread == nullptr || thread->free || thread->joining) {
		return KERNEL_ERROR_ESRCH;
	}

	auto result = PthreadAttrSetaffinity(&thread->attr, mask);

	return result;
}

int KYTY_SYSV_ABI PthreadGetaffinity(Pthread thread, KernelCpumask* mask) {
	PRINT_NAME();

	if (thread == nullptr || thread->free || thread->joining) {
		return KERNEL_ERROR_ESRCH;
	}
	if (mask == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	return PthreadAttrGetaffinity(&thread->attr, mask);
}

int KYTY_SYSV_ABI PthreadSetcancelstate(int state, int* old_state) {
	PRINT_NAME();

	int pstate = PTHREAD_CANCEL_DISABLE;

	switch (state) {
		case 0: pstate = PTHREAD_CANCEL_ENABLE; break;
		case 1: pstate = PTHREAD_CANCEL_DISABLE; break;
		default: EXIT("unknown state: %d", state);
	}

	int result = pthread_setcancelstate(pstate, old_state);

	LOGF("\tthread setcancelstate: %d\n", result);

	if (old_state != nullptr) {
		switch (*old_state) {
			case PTHREAD_CANCEL_ENABLE: *old_state = 0; break;
			case PTHREAD_CANCEL_DISABLE: *old_state = 1; break;
			default: EXIT("unknown old_state: %d", *old_state);
		}
	}

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadSetcanceltype(int type, int* old_type) {
	PRINT_NAME();

	int ptype = PTHREAD_CANCEL_DEFERRED;

	switch (type) {
		case 0: ptype = PTHREAD_CANCEL_DEFERRED; break;
		case 2: ptype = PTHREAD_CANCEL_ASYNCHRONOUS; break;
		default: EXIT("unknown type: %d", type);
	}

	int result = pthread_setcanceltype(ptype, old_type);

	LOGF("\tthread setcanceltype: %d\n", result);

	if (old_type != nullptr) {
		switch (*old_type) {
			case PTHREAD_CANCEL_DEFERRED: *old_type = 0; break;
			case PTHREAD_CANCEL_ASYNCHRONOUS: *old_type = 2; break;
			default: EXIT("unknown type: %d", *old_type);
		}
	}

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI PthreadGetprio(Pthread thread, int* prio) {
	PRINT_NAME();

	if (thread == nullptr) {
		return KERNEL_ERROR_ESRCH;
	}

	EXIT_NOT_IMPLEMENTED(prio == nullptr);

	std::lock_guard lock(thread->life_mutex);
	if (thread->free || thread->joining || !thread->p_valid) {
		return KERNEL_ERROR_ESRCH;
	}

	*prio = 700;
	LOGF("\t PthreadGetprio: %d, %d\n", thread->unique_id, *prio);
	return OK;
}

int KYTY_SYSV_ABI PthreadSetprio(Pthread thread, int prio) {
	if (thread == nullptr) {
		return KERNEL_ERROR_ESRCH;
	}

	std::lock_guard lock(thread->life_mutex);
	if (thread->free || thread->joining || !thread->p_valid) {
		return KERNEL_ERROR_ESRCH;
	}

	// Avoid winpthread pthread_setschedparam during mass Setprio storms (TLOU boot).
	// Skip PRINT_NAME/LOGF on the guest-stack HLE hot path.
	(void)prio;
	return OK;
}

void KYTY_SYSV_ABI PthreadTestcancel() {
	PRINT_NAME();

	pthread_testcancel();
}

void KYTY_SYSV_ABI PthreadExit(void* value) {
	PRINT_NAME();

#if defined(__x86_64__) || defined(_M_X64)
	if (g_pthread_self != nullptr && g_pthread_self->host_return_valid) {
		g_pthread_self->host_return_value = reinterpret_cast<uint64_t>(value);
		g_pthread_self->host_return_valid = false;
		HostRestoreContext(&g_pthread_self->host_return_ctx, 1);
	}
#endif

	pthread_exit(value);
}

int KYTY_SYSV_ABI PthreadEqual(Pthread thread1, Pthread thread2) {
	static std::atomic<uint32_t> log_count {0};

	const int result = (thread1 == thread2 ? 1 : 0);

	const auto count = log_count.fetch_add(1);
	if (count < 32 || (result != 0 && count < 128)) {
		LOGF("\tPthreadEqual: t1 = 0x%016" PRIx64 ", t2 = 0x%016" PRIx64 ", result = %d\n",
		     reinterpret_cast<uint64_t>(thread1), reinterpret_cast<uint64_t>(thread2), result);
	}

	return result;
}

int KYTY_SYSV_ABI PthreadGetname(Pthread thread, char* name) {
	PRINT_NAME();

	if (thread == nullptr || thread->free || thread->joining) {
		return KERNEL_ERROR_ESRCH;
	}

	if (name == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	strncpy(name, thread->name.c_str(), 32);
	name[31] = '\0';

	return OK;
}

int KYTY_SYSV_ABI PthreadRename(Pthread thread, const char* name) {
	PRINT_NAME();

	if (thread == nullptr || thread->free || thread->joining) {
		return KERNEL_ERROR_EINVAL;
	}

	if (name == nullptr) {
		return OK;
	}

	thread->name = std::string(name);

	return OK;
}

void KYTY_SYSV_ABI PthreadYield() {
	SchedulerBackoffOnce();
}

int KYTY_SYSV_ABI PthreadGetthreadid() {
	// PRINT_NAME();

	return g_pthread_self != nullptr ? g_pthread_self->guest.thread_id : 0;
}

int KYTY_SYSV_ABI KernelClockGetres(KernelClockid clock_id, KernelTimespec* tp) {
	PRINT_NAME();

	if (tp == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (clock_id == KERNEL_CLOCK_PROCTIME || clock_id == KERNEL_CLOCK_THREAD_CPUTIME_ID ||
	    clock_id == KERNEL_CLOCK_VIRTUAL || clock_id == KERNEL_CLOCK_PROF) {
		const auto frequency = KernelGetTscFrequencyNative();
		tp->tv_sec           = 0;
		tp->tv_nsec          = static_cast<int64_t>(
		    frequency != 0 ? std::max<uint64_t>((1000000000 + (frequency >> 1)) / frequency, 1)
		                   : 1);
		return OK;
	}

	clockid_t pclock_id = {};
	if (!GetPosixClockId(clock_id, &pclock_id)) {
		LOGF("\tunknown clock_id: %d\n", clock_id);
		return KERNEL_ERROR_EINVAL;
	}

	timespec t {};

	int result = clock_getres(pclock_id, &t);

	tp->tv_sec  = t.tv_sec;
	tp->tv_nsec = t.tv_nsec;

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI KernelClockGettime(KernelClockid clock_id, KernelTimespec* tp) {
	// Called constantly by Python frame/timer code.

	if (tp == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	int special_error = OK;
	if (KernelClockGettimeSpecial(clock_id, tp, &special_error)) {
		return special_error;
	}

	clockid_t pclock_id = {};
	if (!GetPosixClockId(clock_id, &pclock_id)) {
		LOGF("\tunknown clock_id: %d\n", clock_id);
		return KERNEL_ERROR_EINVAL;
	}

	timespec t {};

	int result = clock_gettime(pclock_id, &t);

	tp->tv_sec  = t.tv_sec;
	tp->tv_nsec = t.tv_nsec;

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI KernelGettimeofday(KernelTimeval* tp) {
	// PRINT_NAME();

	if (tp == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	int result = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	FILETIME ft {};
	GetSystemTimePreciseAsFileTime(&ft);
	uint64_t ticks = ft.dwHighDateTime;
	ticks <<= 32;
	ticks |= ft.dwLowDateTime;
	ticks /= 10;
	ticks -= 11644473600000000ULL;
	tp->tv_sec  = static_cast<int64_t>(ticks / 1000000);
	tp->tv_usec = static_cast<int64_t>(ticks % 1000000);
#else
	auto dt = Common::DateTime::FromSystemUTC();
	sec_to_timeval(tp, dt.ToUnix());
#endif

	if (result == 0) {
		return OK;
	}
	return KERNEL_ERROR_EINVAL;
}

int KYTY_SYSV_ABI KernelGettimezone(KernelTimezone* tz) {
	// Hot path in Unity's time/date code.

	if (tz == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TIME_ZONE_INFORMATION tzi {};
	const DWORD           result = GetTimeZoneInformation(&tzi);

	tz->tz_minuteswest = tzi.Bias;
	tz->tz_dsttime     = (result == TIME_ZONE_ID_UNKNOWN ? DST_NONE : DST_MET);
#else
	const std::time_t now = std::time(nullptr);
	std::tm           local_tm {};
	std::tm           utc_tm {};
	localtime_r(&now, &local_tm);
	gmtime_r(&now, &utc_tm);
	const auto local = std::mktime(&local_tm);
	const auto utc   = std::mktime(&utc_tm);

	tz->tz_minuteswest = static_cast<int32_t>((utc - local) / 60);
	tz->tz_dsttime     = (local_tm.tm_isdst > 0 ? DST_MET : DST_NONE);
#endif

	return OK;
}

int KYTY_SYSV_ABI KernelConvertLocaltimeToUtc(int64_t  local_time, int64_t /*reserved*/,
                                              int64_t* utc_time, KernelTimezone* timezone,
                                              int32_t* dst_seconds) {
	// Hot path in Unity's time/date code.

	if (timezone == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	KernelTimezone local_timezone {};
	auto           ret = KernelGettimezone(&local_timezone);
	if (ret != OK) {
		return ret;
	}

	const auto dst_sec = GetDstSeconds();
	*timezone          = local_timezone;

	if (utc_time != nullptr) {
		*utc_time = local_time + static_cast<int64_t>(local_timezone.tz_minuteswest) * 60 - dst_sec;
	}
	if (dst_seconds != nullptr) {
		*dst_seconds = dst_sec;
	}

	return OK;
}

int KYTY_SYSV_ABI KernelConvertUtcToLocaltime(int64_t utc_time, int64_t* local_time,
                                              KernelTimesec* st, uint64_t* dst_sec) {
	// Hot path in Unity's time/date code.

	KernelTimezone tz {};
	auto           ret = KernelGettimezone(&tz);
	if (ret != OK) {
		return ret;
	}

	const auto dst  = static_cast<int64_t>(GetDstSeconds());
	const auto west = static_cast<int64_t>(tz.tz_minuteswest) * 60;
	if (local_time != nullptr) {
		*local_time = utc_time - west + dst;
	}
	if (st != nullptr) {
		st->t        = utc_time;
		st->west_sec = static_cast<uint32_t>(-west);
		st->dst_sec  = static_cast<uint32_t>(dst);
	}
	if (dst_sec != nullptr) {
		*dst_sec = static_cast<uint64_t>(dst);
	}

	return OK;
}

uint64_t KYTY_SYSV_ABI KernelGetTscFrequency() {
	return KernelGetTscFrequencyNative();
}

uint64_t KYTY_SYSV_ABI KernelReadTsc() {
	return KernelReadTscNative();
}

uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
	const auto frequency = KernelGetTscFrequencyNative();
	if (frequency == 0) {
		return static_cast<uint64_t>(Loader::Timer::GetTimeMs() * 1000.0);
	}

	const auto elapsed = KernelGetElapsedTsc();
	return static_cast<uint64_t>((static_cast<long double>(elapsed) * 1000000.0L) /
	                             static_cast<long double>(frequency));
}

uint64_t KYTY_SYSV_ABI KernelGetProcessTimeCounter() {
	return KernelGetElapsedTsc();
}

uint64_t KYTY_SYSV_ABI KernelGetProcessTimeCounterFrequency() {
	return KernelGetTscFrequencyNative();
}

void KYTY_SYSV_ABI KernelSetThreadDtors(thread_dtors_func_t dtors) {
	PRINT_NAME();

	// EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	EXIT_NOT_IMPLEMENTED(g_pthread_context->GetThreadDtors() != nullptr);

	g_pthread_context->SetThreadDtors(dtors);
	// g_thread_dtors = dtors;
}

int KYTY_SYSV_ABI KernelUsleep(KernelUseconds microseconds) {
	Common::Timer t;
	t.Start();
	SleepMicroWithSignalPoll(microseconds);
	// double ts = t.GetTimeS();
	// LOGF("\tactual: %g microseconds\n", ts * 1000000.0);
	return OK;
}

unsigned int KYTY_SYSV_ABI KernelSleep(unsigned int seconds) {
	PRINT_NAME();
	LOGF("\tsleep: %u\n", seconds);
	Common::Timer t;
	t.Start();
	SleepMicroWithSignalPoll(static_cast<uint64_t>(seconds) * 1000000ull);
	double ts = t.GetTimeS();
	LOGF("\tactual: %g seconds\n", ts);
	return OK;
}

int KYTY_SYSV_ABI KernelNanosleep(const KernelTimespec* rqtp, KernelTimespec* rmtp) {
	PRINT_NAME();

	if (rqtp == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (rqtp->tv_sec < 0 || rqtp->tv_nsec < 0 || rqtp->tv_nsec >= 1000000000) {
		return KERNEL_ERROR_EINVAL;
	}

	if (static_cast<uint64_t>(rqtp->tv_sec) >
	    (UINT64_MAX - static_cast<uint64_t>(rqtp->tv_nsec)) / 1000000000ull) {
		return KERNEL_ERROR_EINVAL;
	}

	uint64_t nanos =
	    static_cast<uint64_t>(rqtp->tv_sec) * 1000000000ull + static_cast<uint64_t>(rqtp->tv_nsec);

	LOGF("\tnanosleep: %" PRIu64 "\n", nanos);

	Common::Timer t;
	t.Start();
	SleepNanoWithSignalPoll(nanos);
	double ts = t.GetTimeS();
	LOGF("\tactual: %g nanoseconds\n", ts * 1000000000.0);

	if (rmtp != nullptr) {
		rmtp->tv_sec  = 0;
		rmtp->tv_nsec = 0;
	}

	return OK;
}

int KYTY_SYSV_ABI PthreadKeyCreate(PthreadKey* key, pthread_key_destructor_func_t destructor) {
	PRINT_NAME();

	if (key == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	if (!g_pthread_context->GetPthreadKeys()->Create(key, destructor)) {
		return KERNEL_ERROR_EAGAIN;
	}

	LOGF("\t destructor = %016" PRIx64 "\n"
	     "\t key        = %d\n",
	     reinterpret_cast<uint64_t>(destructor), *key);

	return OK;
}

int KYTY_SYSV_ABI PthreadKeyDelete(PthreadKey key) {
	PRINT_NAME();

	LOGF("\t key = %d\n", key);

	if (!g_pthread_context->GetPthreadKeys()->Delete(key)) {
		return KERNEL_ERROR_EINVAL;
	}

	return OK;
}

int KYTY_SYSV_ABI PthreadSetspecific(PthreadKey key, void* value) {
	// PRINT_NAME();

	int thread_id = Common::Thread::GetThreadIdUnique();

	if (PRINT_NAME_ENABLED) {
		LOGF("\t key       = %d\n"
		     "\t thread_id = %d\n"
		     "\t value     = %016" PRIx64 "\n",
		     key, thread_id, reinterpret_cast<uint64_t>(value));
	}

	if (!g_pthread_context->GetPthreadKeys()->Set(key, thread_id, value)) {
		return KERNEL_ERROR_EINVAL;
	}

	return OK;
}

void* KYTY_SYSV_ABI PthreadGetspecific(PthreadKey key) {
	int thread_id = Common::Thread::GetThreadIdUnique();

	void* value = nullptr;

	if (!g_pthread_context->GetPthreadKeys()->Get(key, thread_id, &value)) {
		return nullptr;
	}

	return value;
}

} // namespace LibKernel

namespace Posix {

LIB_NAME("Posix", "libkernel");

struct PthreadOnce {
	int                     state;
	LibKernel::PthreadMutex mutex;
};

static_assert(sizeof(PthreadOnce) == 0x10);

constexpr int PTHREAD_ONCE_NEEDS_INIT  = 0;
constexpr int PTHREAD_ONCE_DONE_INIT   = 1;
constexpr int PTHREAD_ONCE_IN_PROGRESS = 2;
constexpr int PTHREAD_ONCE_WAIT        = 3;

static std::mutex              g_pthread_once_mutex;
static std::condition_variable g_pthread_once_cv;

int KYTY_SYSV_ABI pthread_once(void* once_control, void(KYTY_SYSV_ABI* init_routine)()) {
	PRINT_NAME();

	if (once_control == nullptr || init_routine == nullptr) {
		return POSIX_EINVAL;
	}

	auto* once = static_cast<PthreadOnce*>(once_control);

	{
		std::unique_lock lock(g_pthread_once_mutex);

		for (;;) {
			switch (once->state) {
				case PTHREAD_ONCE_DONE_INIT: return OK;
				case PTHREAD_ONCE_NEEDS_INIT: once->state = PTHREAD_ONCE_IN_PROGRESS; goto run_init;
				case PTHREAD_ONCE_IN_PROGRESS:
				case PTHREAD_ONCE_WAIT:
					once->state = PTHREAD_ONCE_WAIT;
					while (once->state == PTHREAD_ONCE_WAIT) {
						g_pthread_once_cv.wait_for(
						    lock, std::chrono::microseconds(LibKernel::SIGNAL_APC_POLL_MICROS));
						if (once->state == PTHREAD_ONCE_WAIT) {
							lock.unlock();
							LibKernel::KernelDispatchPendingSignalForCurrentThread();
							lock.lock();
						}
					}
					break;
				default: return POSIX_EINVAL;
			}
		}
	}

run_init:
	init_routine();

	{
		std::lock_guard lock(g_pthread_once_mutex);
		once->state = PTHREAD_ONCE_DONE_INIT;
	}
	g_pthread_once_cv.notify_all();

	return OK;
}

int KYTY_SYSV_ABI pthread_create(LibKernel::Pthread* thread, const LibKernel::PthreadAttr* attr,
                                 LibKernel::pthread_entry_func_t entry, void* arg) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadCreate(thread, attr, entry, arg, ""));
}

int KYTY_SYSV_ABI pthread_create_name_np(LibKernel::Pthread*             thread,
                                         const LibKernel::PthreadAttr*   attr,
                                         LibKernel::pthread_entry_func_t entry, void* arg,
                                         const char* name) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(
	    LibKernel::PthreadCreate(thread, attr, entry, arg, (name != nullptr ? name : "")));
}

int KYTY_SYSV_ABI pthread_detach(LibKernel::Pthread thread) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadDetach(thread));
}

void KYTY_SYSV_ABI pthread_exit(void* value) {
	PRINT_NAME();

	LibKernel::PthreadExit(value);
}

LibKernel::Pthread KYTY_SYSV_ABI pthread_self() {
	PRINT_NAME();

	return LibKernel::PthreadSelf();
}

int KYTY_SYSV_ABI pthread_rename_np(LibKernel::Pthread thread, const char* name) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadRename(thread, name));
}

int KYTY_SYSV_ABI pthread_setcancelstate(int state, int* old_state) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadSetcancelstate(state, old_state));
}

int KYTY_SYSV_ABI pthread_setprio(LibKernel::Pthread thread, int prio) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadSetprio(thread, prio));
}

int KYTY_SYSV_ABI pthread_getschedparam(LibKernel::Pthread thread, int* policy,
                                        LibKernel::KernelSchedParam* param) {
	PRINT_NAME();

	if (thread == nullptr || thread->free || thread->joining || policy == nullptr ||
	    param == nullptr) {
		return thread != nullptr && (thread->free || thread->joining) ? POSIX_ESRCH : POSIX_EINVAL;
	}

	int result = LibKernel::PthreadAttrGetschedpolicy(&thread->attr, policy);
	result = (result == OK ? LibKernel::PthreadAttrGetschedparam(&thread->attr, param) : result);

	return (result == OK ? 0 : LibKernel::KernelToPosix(result));
}

int KYTY_SYSV_ABI pthread_setschedparam(LibKernel::Pthread thread, int policy,
                                        const LibKernel::KernelSchedParam* param) {
	PRINT_NAME();

	if (thread == nullptr || thread->free || thread->joining || param == nullptr) {
		return thread != nullptr && (thread->free || thread->joining) ? POSIX_ESRCH : POSIX_EINVAL;
	}

	int result = LibKernel::PthreadAttrSetschedpolicy(&thread->attr, policy);
	result = (result == OK ? LibKernel::PthreadAttrSetschedparam(&thread->attr, param) : result);

	return (result == OK ? 0 : LibKernel::KernelToPosix(result));
}

void KYTY_SYSV_ABI pthread_yield() {
	PRINT_NAME();

	LibKernel::PthreadYield();
}

int KYTY_SYSV_ABI sched_get_priority_max(int policy) {
	PRINT_NAME();

	LOGF("\t policy = %d\n", policy);

	return 256;
}

int KYTY_SYSV_ABI sched_get_priority_min(int policy) {
	PRINT_NAME();

	LOGF("\t policy = %d\n", policy);

	return 767;
}

int KYTY_SYSV_ABI pthread_join(LibKernel::Pthread thread, void** value) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadJoin(thread, value));
}

int KYTY_SYSV_ABI pthread_attr_init(LibKernel::PthreadAttr* attr) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrInit(attr));
}

int KYTY_SYSV_ABI pthread_attr_destroy(LibKernel::PthreadAttr* attr) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrDestroy(attr));
}

int KYTY_SYSV_ABI pthread_attr_get_np(LibKernel::Pthread thread, LibKernel::PthreadAttr* attr) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrGet(thread, attr));
}

int KYTY_SYSV_ABI pthread_attr_getdetachstate(const LibKernel::PthreadAttr* attr, int* state) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrGetdetachstate(attr, state));
}

int KYTY_SYSV_ABI pthread_attr_getguardsize(const LibKernel::PthreadAttr* attr,
                                            size_t*                       guard_size) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrGetguardsize(attr, guard_size));
}

int KYTY_SYSV_ABI pthread_attr_getinheritsched(const LibKernel::PthreadAttr* attr,
                                               int*                          inherit_sched) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrGetinheritsched(attr, inherit_sched));
}

int KYTY_SYSV_ABI pthread_attr_getschedparam(const LibKernel::PthreadAttr* attr,
                                             LibKernel::KernelSchedParam*  param) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrGetschedparam(attr, param));
}

int KYTY_SYSV_ABI pthread_attr_getschedpolicy(const LibKernel::PthreadAttr* attr, int* policy) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrGetschedpolicy(attr, policy));
}

int KYTY_SYSV_ABI pthread_attr_getstack(const LibKernel::PthreadAttr* __restrict attr,
                                        void** __restrict stack_addr,
                                        size_t* __restrict stack_size) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrGetstack(attr, stack_addr, stack_size));
}

int KYTY_SYSV_ABI pthread_attr_getstacksize(const LibKernel::PthreadAttr* attr,
                                            size_t*                       stack_size) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrGetstacksize(attr, stack_size));
}

int KYTY_SYSV_ABI pthread_attr_setdetachstate(LibKernel::PthreadAttr* attr, int state) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrSetdetachstate(attr, state));
}

int KYTY_SYSV_ABI pthread_attr_setguardsize(LibKernel::PthreadAttr* attr, size_t guard_size) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrSetguardsize(attr, guard_size));
}

int KYTY_SYSV_ABI pthread_attr_setinheritsched(LibKernel::PthreadAttr* attr, int inherit_sched) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrSetinheritsched(attr, inherit_sched));
}

int KYTY_SYSV_ABI pthread_attr_setschedparam(LibKernel::PthreadAttr*            attr,
                                             const LibKernel::KernelSchedParam* param) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrSetschedparam(attr, param));
}

int KYTY_SYSV_ABI pthread_attr_setschedpolicy(LibKernel::PthreadAttr* attr, int policy) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrSetschedpolicy(attr, policy));
}

int KYTY_SYSV_ABI pthread_attr_setstacksize(LibKernel::PthreadAttr* attr, size_t stack_size) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadAttrSetstacksize(attr, stack_size));
}

int KYTY_SYSV_ABI pthread_cond_broadcast(LibKernel::PthreadCond* cond) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadCondBroadcast(cond));
}

int KYTY_SYSV_ABI pthread_cond_signal(LibKernel::PthreadCond* cond) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadCondSignal(cond));
}

int KYTY_SYSV_ABI pthread_condattr_setclock(LibKernel::PthreadCondattr* attr,
                                            LibKernel::KernelClockid    clock_id) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadCondattrSetclock(attr, clock_id));
}

int KYTY_SYSV_ABI pthread_condattr_init(LibKernel::PthreadCondattr* attr) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadCondattrInit(attr));
}

int KYTY_SYSV_ABI pthread_condattr_destroy(LibKernel::PthreadCondattr* attr) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadCondattrDestroy(attr));
}

int KYTY_SYSV_ABI pthread_cond_wait(LibKernel::PthreadCond* cond, LibKernel::PthreadMutex* mutex) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadCondWait(cond, mutex));
}

int KYTY_SYSV_ABI pthread_cond_timedwait(LibKernel::PthreadCond*          cond,
                                         LibKernel::PthreadMutex*         mutex,
                                         const LibKernel::KernelTimespec* abstime) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadCondTimedwaitAbs(cond, mutex, abstime));
}

int KYTY_SYSV_ABI pthread_mutex_lock(LibKernel::PthreadMutex* mutex) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadMutexLock(mutex));
}

int KYTY_SYSV_ABI pthread_mutex_trylock(LibKernel::PthreadMutex* mutex) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadMutexTrylock(mutex));
}

int KYTY_SYSV_ABI pthread_mutex_timedlock(LibKernel::PthreadMutex*         mutex,
                                          const LibKernel::KernelTimespec* abstime) {
	PRINT_NAME();

	if (abstime == nullptr || abstime->tv_sec < 0 || abstime->tv_nsec < 0 ||
	    abstime->tv_nsec >= 1000000000) {
		return POSIX_PTHREAD_CALL(LibKernel::KERNEL_ERROR_EINVAL);
	}

	LibKernel::KernelTimespec now {};
	if (LibKernel::KernelClockGettime(0, &now) != OK) {
		return POSIX_PTHREAD_CALL(LibKernel::KERNEL_ERROR_EINVAL);
	}

	int64_t sec_delta  = abstime->tv_sec - now.tv_sec;
	int64_t nsec_delta = abstime->tv_nsec - now.tv_nsec;
	if (nsec_delta < 0) {
		sec_delta--;
		nsec_delta += 1000000000;
	}

	uint64_t usec = 0;
	if (sec_delta > 0 || (sec_delta == 0 && nsec_delta > 0)) {
		usec = (static_cast<uint64_t>(sec_delta) > UINT32_MAX / 1000000ull
		            ? UINT32_MAX
		            : static_cast<uint64_t>(sec_delta) * 1000000ull);
		usec = std::min<uint64_t>(UINT32_MAX, usec + static_cast<uint64_t>(nsec_delta / 1000));
	}

	return POSIX_PTHREAD_CALL(
	    LibKernel::PthreadMutexTimedlock(mutex, static_cast<LibKernel::KernelUseconds>(usec)));
}

int KYTY_SYSV_ABI pthread_mutex_unlock(LibKernel::PthreadMutex* mutex) {
	// PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadMutexUnlock(mutex));
}

int KYTY_SYSV_ABI pthread_rwlock_rdlock(LibKernel::PthreadRwlock* rwlock) {
	// Hot path for some PS5 titles; per-call name logging can dominate runtime.

	return POSIX_PTHREAD_CALL(LibKernel::PthreadRwlockRdlock(rwlock));
}

int KYTY_SYSV_ABI pthread_rwlock_unlock(LibKernel::PthreadRwlock* rwlock) {
	// Hot path for some PS5 titles; per-call name logging can dominate runtime.

	return POSIX_PTHREAD_CALL(LibKernel::PthreadRwlockUnlock(rwlock));
}

int KYTY_SYSV_ABI pthread_rwlock_wrlock(LibKernel::PthreadRwlock* rwlock) {
	// Hot path for some PS5 titles; per-call name logging can dominate runtime.

	return POSIX_PTHREAD_CALL(LibKernel::PthreadRwlockWrlock(rwlock));
}

int KYTY_SYSV_ABI pthread_rwlock_destroy(LibKernel::PthreadRwlock* rwlock) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadRwlockDestroy(rwlock));
}

int KYTY_SYSV_ABI pthread_key_create(LibKernel::PthreadKey*                   key,
                                     LibKernel::pthread_key_destructor_func_t destructor) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadKeyCreate(key, destructor));
}

int KYTY_SYSV_ABI pthread_key_delete(LibKernel::PthreadKey key) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadKeyDelete(key));
}

int KYTY_SYSV_ABI pthread_setspecific(LibKernel::PthreadKey key, void* value) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadSetspecific(key, value));
}

void* KYTY_SYSV_ABI pthread_getspecific(LibKernel::PthreadKey key) {
	PRINT_NAME();

	return (LibKernel::PthreadGetspecific(key));
}

int KYTY_SYSV_ABI pthread_mutex_destroy(LibKernel::PthreadMutex* mutex) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadMutexDestroy(mutex));
}

int KYTY_SYSV_ABI pthread_mutex_init(LibKernel::PthreadMutex*           mutex,
                                     const LibKernel::PthreadMutexattr* attr) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadMutexInit(mutex, attr));
}

int KYTY_SYSV_ABI pthread_mutexattr_init(LibKernel::PthreadMutexattr* attr) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadMutexattrInit(attr));
}

int KYTY_SYSV_ABI pthread_mutexattr_settype(LibKernel::PthreadMutexattr* attr, int type) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadMutexattrSettype(attr, type));
}

int KYTY_SYSV_ABI pthread_mutexattr_setprotocol(LibKernel::PthreadMutexattr* attr, int protocol) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadMutexattrSetprotocol(attr, protocol));
}

int KYTY_SYSV_ABI pthread_mutexattr_destroy(LibKernel::PthreadMutexattr* attr) {
	PRINT_NAME();

	return POSIX_PTHREAD_CALL(LibKernel::PthreadMutexattrDestroy(attr));
}

int KYTY_SYSV_ABI pthread_getstack(const LibKernel::PthreadAttr* __restrict attr,
                                   void** __restrict stack_addr, size_t* __restrict stack_size) {
	PRINT_NAME();

	return pthread_attr_getstack(attr, stack_addr, stack_size);
}

} // namespace Posix

} // namespace Libs

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#pragma GCC diagnostic pop
#endif
