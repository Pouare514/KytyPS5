#include "graphics/presentation/videoOut.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/renderer/textureCache.h"
#include "graphics/presentation/displayBuffer.h"
#include "graphics/presentation/window.h"
#include "common/crashDiagnostics.h"
#include "common/fatalLog.h"
#include "common/singleton.h"
#include "kernel/pthread.h"
#include "kernel/eventQueue.h"
#include "libs/errno.h"
#include "libs/agc.h"
#include "libs/libs.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <intrin.h>
#include <windows.h>
#endif

namespace Libs::Graphics {
struct GraphicContext;
} // namespace Libs::Graphics

namespace Libs::VideoOut {

LIB_NAME("VideoOut", "VideoOut");

namespace EventQueue = LibKernel::EventQueue;

// Phase 30/32 bisect after hold gate (KYTY_PHASE30_BISECT):
// 1 = currentBuffer=0, pending sticky; no Flip EQ; no wake
// 2 = pending=0 immediately after hold (currentBuffer blank); no EQ
// 3 = Flip EQ + wake, pending sticky (no clear / no inject)
// 4 = Flip EQ, then clear pending (crash repro)
// 5 / unset = Phase 32 default: Flip EQ + wake + pending sticky (baseline survivable)
static int Phase30BisectMode() {
	static const int mode = [] {
		const char* env = std::getenv("KYTY_PHASE30_BISECT");
		if (env == nullptr || env[0] == '\0') {
			return 5;
		}
		const int v = std::atoi(env);
		return (v >= 1 && v <= 5) ? v : 5;
	}();
	return mode;
}

// KYTY_PHASE31_INJECT_FLIP0=1: one-shot SubmitFlip(buffer0) + sticky×32 (Phase 30 present repro).
static bool Phase31StickyInjectEnabled() {
	const char* env = std::getenv("KYTY_PHASE31_INJECT_FLIP0");
	return env != nullptr && env[0] == '1';
}

// KYTY_PHASE31_SOFT_ONLY=1: clear pending without inject (known failfast; diagnostic).
static bool Phase31SoftOnlyEnabled() {
	const char* env = std::getenv("KYTY_PHASE31_SOFT_ONLY");
	return env != nullptr && env[0] == '1';
}

// KYTY_PHASE32_PENDING0=1: after EQ+wake, clear pending once (experimental unlock).
static bool Phase32Pending0Enabled() {
	const char* env = std::getenv("KYTY_PHASE32_PENDING0");
	return env != nullptr && env[0] == '1';
}

// After PHASE32_PENDING0 arms clear, keep returning pending=1 for N polls so FlipHandler
// can consume the Flip EQ before MainThread leaves the spin.
static std::atomic<int> g_pending0_grace_polls {0};
static constexpr int    kPending0GracePolls = 64;

// Flip EQ consumption (FlipEventResetFunc) — MainThread must not see pending=0 until
// FlipHandler has drained at least one Flip after we arm PENDING0.
static std::atomic<uint64_t> g_flip_eq_consumed {0};
static std::atomic<uint64_t> g_pending0_need_flip_consumed {0};

static std::atomic<bool> g_pending0_seen {false};
static std::atomic<int>  g_pending0_watchdog_ticks {0};

// Phase 34 menu bootstrap state (workers defined after VideoOutConfig).
struct Phase34MenuSnapshot {
	int                                  handle      = 0;
	int                                  set_id      = 0;
	int                                  start_index = 0;
	int                                  buffer_num  = 0;
	std::vector<const void*>             addresses;
	std::vector<Graphics::VideoOutInfo>  infos;
};
static std::mutex          g_phase34_mu;
static Phase34MenuSnapshot g_phase34_snap;
static std::atomic<bool>   g_phase34_bootstrap_started {false};
// Declared early — Phase41/42 handoff checks these before the Phase37 block.
static std::atomic<bool>   g_phase37_post_unreg {false};
static std::atomic<bool>   g_phase37_guest_flip_seen {false};
static std::atomic<bool>   g_phase37_guest_reg_seen {false};

static bool Phase34MenuBootstrapDisabled() {
	// Phase 35: menu host worker is opt-in (guest path is the default goal).
	// KYTY_PHASE34_MENU=1 restores Phase 34 MenuRegister/Submit/Flip worker.
	// KYTY_PHASE34_NO_MENU=1 also disables (compat).
	const char* no_menu = std::getenv("KYTY_PHASE34_NO_MENU");
	if (no_menu != nullptr && no_menu[0] == '1') {
		return true;
	}
	const char* menu = std::getenv("KYTY_PHASE34_MENU");
	return menu == nullptr || menu[0] != '1';
}

static void Phase34SaveMenuSnapshot(int handle, int set_id, int start_index, int buffer_num,
                                    const void* const* addresses,
                                    const std::vector<Graphics::VideoOutInfo>& infos);
static void Phase34ArmMenuBootstrap();

static void ArmPending0Grace(const char* why, bool require_flip_eq = true) {
	// Phase 33 no-bridge: no host SubmitFlip → Flip EQ may never fire. Clear via polls only.
	const uint64_t need =
	    require_flip_eq ? (g_flip_eq_consumed.load(std::memory_order_acquire) + 1) : 0;
	g_pending0_need_flip_consumed.store(need, std::memory_order_release);
	g_pending0_grace_polls.store(kPending0GracePolls, std::memory_order_release);
	LOGF("FlipTrace: phase32 arm pending0 grace=%d need_flip_consumed=%" PRIu64 " (%s)\n",
	     kPending0GracePolls, need, why != nullptr ? why : "?");
	fprintf(stderr, "FlipTrace: phase32 arm pending0 grace=%d need_flip=%" PRIu64 " (%s)\n",
	        kPending0GracePolls, need, why != nullptr ? why : "?");
}

static void DumpGuestCodeAround(uint64_t addr) {
	auto* linker  = Common::Singleton<Loader::RuntimeLinker>::Instance();
	auto* program = linker != nullptr ? linker->FindProgramByAddr(addr) : nullptr;
	char  header[192];
	if (program != nullptr) {
		std::snprintf(header, sizeof(header),
		              "FlipTrace: guest code dump ra=0x%016" PRIx64 " module_base=0x%016" PRIx64,
		              addr, program->base_vaddr);
	} else {
		std::snprintf(header, sizeof(header),
		              "FlipTrace: guest code dump ra=0x%016" PRIx64 " module=<unknown>", addr);
	}
	LOGF("%s\n", header);
	fprintf(stderr, "%s\n", header);
	Common::LogFatalToFile(header);

	const uint64_t dump_start = addr >= 64 ? addr - 64 : 0;
	const uint64_t dump_end   = addr + 128;
	for (uint64_t row = dump_start; row < dump_end; row += 16) {
		char line[160];
		int  n = std::snprintf(line, sizeof(line), "  0x%016" PRIx64 ":", row);
		for (uint32_t i = 0; i < 16 && n > 0 && static_cast<size_t>(n) + 3 < sizeof(line); i++) {
			const uint64_t va = row + i;
			uint8_t        byte = 0xCC;
			// Prefer live guest mapping (SELF is decrypted in RAM).
			const auto* live = reinterpret_cast<const uint8_t*>(va);
			if (live != nullptr) {
				byte = live[0];
			}
			n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n), "%c%02" PRIx8,
			                   va == addr ? '>' : ' ', byte);
		}
		LOGF("%s\n", line);
		fprintf(stderr, "%s\n", line);
		Common::LogFatalToFile(line);
	}
}

static void DumpNextGuestCalls(uint64_t addr, uint32_t span, uint32_t max_calls, const char* tag);

// Soft-stub a guest function prologue to `xor eax,eax; ret` so BootCards teardown
// helpers return success without FiberSwitch/failfast.
static bool Phase33StubGuestProc(uint64_t target, const char* tag) {
	if (target < 0x1000) {
		return false;
	}
	auto* p = reinterpret_cast<uint8_t*>(target);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	DWORD old_prot = 0;
	if (VirtualProtect(p, 16, PAGE_EXECUTE_READWRITE, &old_prot) == 0) {
		LOGF("FlipTrace: stub VirtualProtect failed tgt=0x%016" PRIx64 " err=%lu\n", target,
		     GetLastError());
		return false;
	}
#endif
	// mov eax,1 ; ret  — non-zero so BootCards `test ecx; jle err` takes success path.
	p[0] = 0xb8;
	p[1] = 0x01;
	p[2] = 0x00;
	p[3] = 0x00;
	p[4] = 0x00;
	p[5] = 0xc3;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	DWORD unused = 0;
	VirtualProtect(p, 16, old_prot, &unused);
#endif
	LOGF("FlipTrace: phase33 stub %s tgt=0x%016" PRIx64 " -> mov eax,1; ret\n", tag, target);
	fprintf(stderr, "FlipTrace: phase33 stub %s tgt=0x%016" PRIx64 "\n", tag, target);
	return true;
}

// Phase 38/40: HLE / live-probe for toxic post-Unregister [3]/[4].
static void Phase38NudgeBootWorkersOnce();
static void Phase39SignalVblankAndLabels();

static std::atomic<uint64_t> g_phase39_hle_hits[5] {};
static std::atomic<bool>     g_phase39_post_unreg {false};

// Phase 39: dump prologue + call-site for handoff targets (keep/HLE).
static void Phase39TraceHandoffTarget(uint32_t index, uint64_t call_site, uint64_t target,
                                      const char* kind) {
	char bytes[40] {};
	int  bn = 0;
	const auto* code = reinterpret_cast<const uint8_t*>(target);
	for (int i = 0; i < 16 && bn + 3 < static_cast<int>(sizeof(bytes)); ++i) {
		bn += std::snprintf(bytes + bn, sizeof(bytes) - static_cast<size_t>(bn), "%02x", code[i]);
	}
	LOGF("FlipTrace: phase39 handoff %s[%u] site=0x%016" PRIx64 " tgt=0x%016" PRIx64
	     " prologue=%s\n",
	     kind, index, call_site, target, bytes);
	fprintf(stderr, "FlipTrace: phase39 handoff %s[%u] tgt=0x%016" PRIx64 "\n", kind, index,
	        target);
	DumpGuestCodeAround(target);
}

// ---------------------------------------------------------------------------
// Phase 41 — keep[1] (0x901A1FBF0) trace + NdJob/menu handoff + rich HLE
// ---------------------------------------------------------------------------
constexpr uint64_t kPhase41NdJobFiberEnt = 0x0000000901A212C0ULL;
constexpr uint64_t kPhase41NdJobWorker   = 0x0000000901A22CD0ULL;
constexpr uint32_t kPhase41Keep1Steal    = 12; // ends after xor ecx (no rip-rel)
constexpr uint32_t kPhase41SnapBytes     = 128;
constexpr uint32_t kPhase41Keep1FieldOff = 0x20858; // [r14+0x20858] in keep[1] body

static std::atomic<uint64_t> g_phase41_keep1_obj {0};
static std::atomic<uint64_t> g_phase41_keep1_rsi {0};
static std::atomic<uint64_t> g_phase41_status_rdi {0};
static std::atomic<uint64_t> g_phase41_keep1_hits {0};
static std::atomic<uint64_t> g_phase41_handoff_n {0};
static std::atomic<int>      g_phase41_keep1_pending_log {0}; // 1=enter, 2=exit
static std::atomic<uint32_t> g_phase41_keep1_last_eax {0};
static std::atomic<bool>     g_phase42_reregister_ok {false};
static std::atomic<bool>     g_phase42_flip_attempted {false};
static std::atomic<int>      g_phase42_flip_handle {0};
static std::atomic<int>      g_phase42_flip_num {0};
// Phase 43: sustained menu frames (presented delta / submit_dcb) before soft-idle park.
static std::atomic<bool>     g_phase43_menu_frames_ok {false};
static std::atomic<bool>     g_phase43_baseline_set {false};
static std::atomic<uint64_t> g_phase43_presented_baseline {0};
static std::atomic<uint64_t> g_phase43_submit_dcb_baseline {0};
static std::atomic<uint64_t> g_phase43_sustain_n {0};
static std::atomic<bool>     g_phase43_dcb_seeded {false};
// Phase 44: true ABI RegisterBuffers2 + NdJob DCB (not snapshot / not P43 seed).
static std::atomic<bool>     g_phase44_guest_reg2_ok {false};
static std::atomic<bool>     g_phase44_ndjob_dcb_ok {false};
static std::atomic<bool>     g_phase44_dcb_baseline_set {false};
static std::atomic<uint64_t> g_phase44_dcb_baseline {0};
static std::atomic<bool>     g_phase44_snapshot_cleared {false};
static std::atomic<bool>     g_phase44_snapshot_done {false};
static uint8_t*              g_phase41_keep1_thunk    = nullptr;
static uint8_t*              g_phase41_keep1_live     = nullptr; // relocated prologue → body+12
static uint8_t*              g_phase41_call_thunk     = nullptr; // host→guest SysV call
static uint8_t               g_phase41_keep1_before[kPhase41SnapBytes] {};
static uint8_t               g_phase41_keep1_after[kPhase41SnapBytes] {};
static uint8_t               g_phase41_keep1_field_before[64] {};
static uint8_t               g_phase41_keep1_field_after[64] {};
static bool                  g_phase41_keep1_have_after = false;

static void Phase41DumpHex(const char* tag, const uint8_t* buf, size_t n) {
	char line[400];
	int  pos = std::snprintf(line, sizeof(line), "FlipTrace: phase41 %s", tag);
	for (size_t i = 0; i < n && pos > 0 && static_cast<size_t>(pos) + 3 < sizeof(line); i++) {
		pos += std::snprintf(line + pos, sizeof(line) - static_cast<size_t>(pos), " %02x", buf[i]);
	}
	LOGF("%s\n", line);
	fprintf(stderr, "%s\n", line);
}

static void Phase41DumpDiff(const char* tag, const uint8_t* before, const uint8_t* after,
                            size_t n) {
	char line[400];
	int  pos = std::snprintf(line, sizeof(line), "FlipTrace: phase41 %s writes:", tag);
	bool any = false;
	for (size_t off = 0; off + 8 <= n; off += 8) {
		uint64_t b = 0;
		uint64_t a = 0;
		std::memcpy(&b, before + off, 8);
		std::memcpy(&a, after + off, 8);
		if (b == a) {
			continue;
		}
		any = true;
		pos += std::snprintf(line + pos, sizeof(line) - static_cast<size_t>(pos),
		                     " +0x%zx:%016" PRIx64 "->%016" PRIx64, off, b, a);
		if (pos > 320) {
			break;
		}
	}
	if (!any) {
		pos += std::snprintf(line + pos, sizeof(line) - static_cast<size_t>(pos), " (none)");
	}
	LOGF("%s\n", line);
	fprintf(stderr, "%s\n", line);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static void Phase41SafeRead(void* dst, const void* src, size_t n) {
	__try {
		std::memcpy(dst, src, n);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		std::memset(dst, 0, n);
	}
}
static void Phase41SafeWrite(volatile void* dst, const void* src, size_t n) {
	__try {
		std::memcpy(const_cast<void*>(static_cast<const volatile void*>(dst)), src, n);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
	}
}
#else
static void Phase41SafeRead(void* dst, const void* src, size_t n) {
	std::memcpy(dst, src, n);
}
static void Phase41SafeWrite(volatile void* dst, const void* src, size_t n) {
	std::memcpy(const_cast<void*>(dst), src, n);
}
#endif

extern "C" void Phase41Keep1OnEntry(uint64_t guest_rdi, uint64_t guest_rsi, uint64_t guest_rdx) {
	// Guest-stack trampoline context: no LOGF/fmt (host AV inside fmt::write).
	const uint64_t n = g_phase41_keep1_hits.fetch_add(1, std::memory_order_relaxed);
	g_phase41_keep1_obj.store(guest_rdi, std::memory_order_release);
	g_phase41_keep1_rsi.store(guest_rsi, std::memory_order_release);
	(void)guest_rdx;
	std::memset(g_phase41_keep1_before, 0, sizeof(g_phase41_keep1_before));
	std::memset(g_phase41_keep1_field_before, 0, sizeof(g_phase41_keep1_field_before));
	if (guest_rdi >= 0x10000ULL && guest_rdi < 0x0000800000000000ULL) {
		Phase41SafeRead(g_phase41_keep1_before, reinterpret_cast<const void*>(guest_rdi),
		                kPhase41SnapBytes);
		Phase41SafeRead(g_phase41_keep1_field_before,
		                reinterpret_cast<const void*>(guest_rdi + kPhase41Keep1FieldOff), 64);
	}
	if (n < 8) {
		g_phase41_keep1_pending_log.store(1, std::memory_order_release);
	}
}

extern "C" void Phase41Keep1OnExit(uint64_t guest_rdi, uint32_t eax) {
	std::memset(g_phase41_keep1_after, 0, sizeof(g_phase41_keep1_after));
	std::memset(g_phase41_keep1_field_after, 0, sizeof(g_phase41_keep1_field_after));
	if (guest_rdi >= 0x10000ULL && guest_rdi < 0x0000800000000000ULL) {
		Phase41SafeRead(g_phase41_keep1_after, reinterpret_cast<const void*>(guest_rdi),
		                kPhase41SnapBytes);
		Phase41SafeRead(g_phase41_keep1_field_after,
		                reinterpret_cast<const void*>(guest_rdi + kPhase41Keep1FieldOff), 64);
		g_phase41_keep1_have_after = true;
	}
	g_phase41_keep1_last_eax.store(eax, std::memory_order_release);
	g_phase41_keep1_pending_log.store(2, std::memory_order_release);
}

extern "C" int Phase41Keep1ShouldHle(uint64_t guest_rdi) {
	if (guest_rdi < 0x10000ULL || guest_rdi >= 0x0000800000000000ULL) {
		return 1;
	}
	// Default: HLE empty-head keep[1]. Live body was observed clearing +0x20858
	// job slots and returning -1 (no NdJob enqueue) — destroys menu producer state.
	// Opt out with KYTY_PHASE41_LIVE_KEEP1=1.
	const char* live = std::getenv("KYTY_PHASE41_LIVE_KEEP1");
	if (live != nullptr && live[0] == '1') {
		return 0;
	}
	uint64_t head = 0;
	Phase41SafeRead(&head, reinterpret_cast<const void*>(guest_rdi), sizeof(head));
	if (head != 0) {
		return 0;
	}
	const uint64_t neg1 = ~uint64_t {0};
	Phase41SafeWrite(reinterpret_cast<volatile void*>(guest_rdi), &neg1, sizeof(neg1));
	return 1;
}

static void Phase41FlushKeep1Log() {
	const int pending = g_phase41_keep1_pending_log.exchange(0, std::memory_order_acq_rel);
	if (pending == 0) {
		return;
	}
	const uint64_t n   = g_phase41_keep1_hits.load(std::memory_order_relaxed);
	const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_relaxed);
	const uint64_t rsi = g_phase41_keep1_rsi.load(std::memory_order_relaxed);
	const uint32_t eax = g_phase41_keep1_last_eax.load(std::memory_order_relaxed);
	if (pending >= 1 && n <= 8) {
		LOGF("FlipTrace: phase41 keep[1] ENTER n=%" PRIu64 " rdi=0x%016" PRIx64
		     " rsi=0x%016" PRIx64 " [*rdi]=0x%016" PRIx64 "\n",
		     n, obj, rsi, *reinterpret_cast<uint64_t*>(g_phase41_keep1_before));
		fprintf(stderr, "FlipTrace: phase41 keep[1] ENTER n=%" PRIu64 " rdi=0x%016" PRIx64 "\n",
		        n, obj);
		Phase41DumpHex("keep1_obj_before", g_phase41_keep1_before, 64);
		Phase41DumpHex("keep1_field+0x20858_before", g_phase41_keep1_field_before, 32);
	}
	if (pending >= 2 && n <= 8) {
		LOGF("FlipTrace: phase41 keep[1] EXIT n=%" PRIu64 " eax=%u rdi=0x%016" PRIx64 "\n", n, eax,
		     obj);
		fprintf(stderr, "FlipTrace: phase41 keep[1] EXIT n=%" PRIu64 " eax=%u\n", n, eax);
		Phase41DumpHex("keep1_obj_after", g_phase41_keep1_after, 64);
		Phase41DumpDiff("keep1_obj", g_phase41_keep1_before, g_phase41_keep1_after,
		                kPhase41SnapBytes);
		Phase41DumpDiff("keep1_field+0x20858", g_phase41_keep1_field_before,
		                g_phase41_keep1_field_after, 64);
	}
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static bool Phase41InstallKeep1Trace(uint64_t target) {
	if (target < 0x1000) {
		return false;
	}
	if (g_phase41_keep1_thunk != nullptr) {
		return true;
	}
	auto* guest = reinterpret_cast<uint8_t*>(target);
	uint8_t orig[kPhase41Keep1Steal] {};
	Phase41SafeRead(orig, guest, kPhase41Keep1Steal);

	g_phase41_keep1_live = static_cast<uint8_t*>(
	    VirtualAlloc(nullptr, 64, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	g_phase41_keep1_thunk = static_cast<uint8_t*>(
	    VirtualAlloc(nullptr, 256, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	if (g_phase41_keep1_live == nullptr || g_phase41_keep1_thunk == nullptr) {
		return false;
	}
	// live stub: stolen prologue + jmp target+12
	std::memcpy(g_phase41_keep1_live, orig, kPhase41Keep1Steal);
	g_phase41_keep1_live[kPhase41Keep1Steal]     = 0x48;
	g_phase41_keep1_live[kPhase41Keep1Steal + 1] = 0xb8;
	const uint64_t cont = target + kPhase41Keep1Steal;
	std::memcpy(g_phase41_keep1_live + kPhase41Keep1Steal + 2, &cont, 8);
	g_phase41_keep1_live[kPhase41Keep1Steal + 10] = 0xff;
	g_phase41_keep1_live[kPhase41Keep1Steal + 11] = 0xe0;

	uint8_t* p = g_phase41_keep1_thunk;
	auto emit = [&](std::initializer_list<uint8_t> bytes) {
		for (uint8_t b: bytes) {
			*p++ = b;
		}
	};
	auto emit_u64 = [&](uint64_t v) {
		std::memcpy(p, &v, 8);
		p += 8;
	};
	// SysV → MS: keep guest args in MS non-volatiles across OnEntry/OnExit.
	// Prior bug: rdi saved in r10 (MS-volatile) → OnEntry clobbered it → live
	// prologue `mov r14,rdi` got 0 → `cmp [r14],0` AV at keep[1]+0x23.
	emit({0x53});                   // push rbx
	emit({0x55});                   // push rbp
	emit({0x41, 0x54});             // push r12
	emit({0x41, 0x55});             // push r13
	emit({0x48, 0x89, 0xfb});       // mov rbx, rdi
	emit({0x48, 0x89, 0xf5});       // mov rbp, rsi
	emit({0x49, 0x89, 0xd4});       // mov r12, rdx
	emit({0x48, 0x89, 0xd9});       // mov rcx, rbx
	emit({0x48, 0x89, 0xea});       // mov rdx, rbp
	emit({0x4d, 0x89, 0xe0});       // mov r8, r12
	emit({0x48, 0x83, 0xec, 0x20});
	emit({0x48, 0xb8});
	emit_u64(reinterpret_cast<uint64_t>(&Phase41Keep1OnEntry));
	emit({0xff, 0xd0});
	emit({0x48, 0x83, 0xc4, 0x20});
	// if (ShouldHle(rbx)) { eax=1; goto on_exit; }
	emit({0x48, 0x89, 0xd9}); // mov rcx, rbx
	emit({0x48, 0x83, 0xec, 0x20});
	emit({0x48, 0xb8});
	emit_u64(reinterpret_cast<uint64_t>(&Phase41Keep1ShouldHle));
	emit({0xff, 0xd0});
	emit({0x48, 0x83, 0xc4, 0x20});
	emit({0x85, 0xc0});             // test eax, eax
	emit({0x74, 0x07});             // jz do_live (+7)
	emit({0xb8, 0x01, 0x00, 0x00, 0x00}); // mov eax, 1
	emit({0xeb, 0x1d});             // jmp after_live (+29 = live block size)
	// do_live:
	emit({0x48, 0x89, 0xdf}); // mov rdi, rbx
	emit({0x48, 0x89, 0xee}); // mov rsi, rbp
	emit({0x4c, 0x89, 0xe2}); // mov rdx, r12
	emit({0x48, 0x83, 0xec, 0x20});
	emit({0x48, 0xb8});
	emit_u64(reinterpret_cast<uint64_t>(g_phase41_keep1_live));
	emit({0xff, 0xd0});
	emit({0x48, 0x83, 0xc4, 0x20});
	// after_live:
	emit({0x41, 0x89, 0xc5}); // mov r13d, eax
	emit({0x48, 0x89, 0xd9}); // mov rcx, rbx
	emit({0x44, 0x89, 0xea}); // mov edx, r13d
	emit({0x48, 0x83, 0xec, 0x20});
	emit({0x48, 0xb8});
	emit_u64(reinterpret_cast<uint64_t>(&Phase41Keep1OnExit));
	emit({0xff, 0xd0});
	emit({0x48, 0x83, 0xc4, 0x20});
	emit({0x44, 0x89, 0xe8}); // mov eax, r13d
	emit({0x41, 0x5d});       // pop r13
	emit({0x41, 0x5c});       // pop r12
	emit({0x5d});             // pop rbp
	emit({0x5b});             // pop rbx
	emit({0xc3});             // ret

	DWORD old_prot = 0;
	if (VirtualProtect(guest, 16, PAGE_EXECUTE_READWRITE, &old_prot) == 0) {
		return false;
	}
	guest[0] = 0x48;
	guest[1] = 0xb8;
	const uint64_t th = reinterpret_cast<uint64_t>(g_phase41_keep1_thunk);
	std::memcpy(guest + 2, &th, 8);
	guest[10] = 0xff;
	guest[11] = 0xe0;
	DWORD unused = 0;
	VirtualProtect(guest, 16, old_prot, &unused);

	// Host→guest call thunk (MS ABI: rcx=obj) → SysV keep1 live stub.
	g_phase41_call_thunk = static_cast<uint8_t*>(
	    VirtualAlloc(nullptr, 64, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	if (g_phase41_call_thunk != nullptr) {
		uint8_t*     c         = g_phase41_call_thunk;
		const uint64_t live_addr = reinterpret_cast<uint64_t>(g_phase41_keep1_live);
		c[0]  = 0x48;
		c[1]  = 0x89;
		c[2]  = 0xcf; // mov rdi, rcx
		c[3]  = 0x48;
		c[4]  = 0x31;
		c[5]  = 0xf6; // xor rsi, rsi
		c[6]  = 0x48;
		c[7]  = 0x83;
		c[8]  = 0xec;
		c[9]  = 0x28; // sub rsp, 0x28
		c[10] = 0x48;
		c[11] = 0xb8; // movabs rax, live
		std::memcpy(c + 12, &live_addr, 8);
		c[20] = 0xff;
		c[21] = 0xd0; // call rax
		c[22] = 0x48;
		c[23] = 0x83;
		c[24] = 0xc4;
		c[25] = 0x28; // add rsp, 0x28
		c[26] = 0xc3; // ret
	}

	LOGF("FlipTrace: phase41 keep[1] trace trampoline installed tgt=0x%016" PRIx64
	     " thunk=%p live=%p\n",
	     target, static_cast<void*>(g_phase41_keep1_thunk),
	     static_cast<void*>(g_phase41_keep1_live));
	fprintf(stderr, "FlipTrace: phase41 keep[1] trace trampoline installed\n");
	return true;
}
#endif

static void Phase41TraceKeep1AtPatch(uint64_t target) {
	DumpNextGuestCalls(target, 384, 20, "keep1");
	DumpGuestCodeAround(target);
	DumpNextGuestCalls(kPhase41NdJobFiberEnt, 128, 12, "ndjobFiber");
	DumpNextGuestCalls(kPhase41NdJobWorker, 128, 12, "ndjobWorker");
	LOGF("FlipTrace: phase41 keep[1] correlate fiber=0x%016" PRIx64 " worker=0x%016" PRIx64
	     " field_off=0x%x\n",
	     kPhase41NdJobFiberEnt, kPhase41NdJobWorker, kPhase41Keep1FieldOff);
	fprintf(stderr, "FlipTrace: phase41 keep[1] trace armed tgt=0x%016" PRIx64 "\n", target);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	(void)Phase41InstallKeep1Trace(target);
#else
	(void)target;
#endif
}

static void Phase41ApplyKeep1SideEffects() {
	const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL) {
		return;
	}
	uint64_t field_ptr = 0;
	Phase41SafeRead(&field_ptr, reinterpret_cast<const void*>(obj + kPhase41Keep1FieldOff),
	                sizeof(field_ptr));
	static std::atomic<int> side_log {0};
	if (side_log.fetch_add(1, std::memory_order_relaxed) < 8) {
		uint64_t head = 0;
		uint32_t gate = 0;
		Phase41SafeRead(&head, reinterpret_cast<const void*>(obj), sizeof(head));
		Phase41SafeRead(&gate, reinterpret_cast<const void*>(obj + 8), sizeof(gate));
		LOGF("FlipTrace: phase41 keep1 side-effect snap obj=0x%016" PRIx64 " *obj=0x%016" PRIx64
		     " gate+8=%u field=0x%016" PRIx64 "\n",
		     obj, head, gate, field_ptr);
		fprintf(stderr, "FlipTrace: phase41 keep1 side-effect snap obj=0x%016" PRIx64 "\n", obj);
	}
	// Default poke gate+8 when zero — wakes keep[1]/status consumers without
	// touching the +0x20858 bitmap (live keep[1] cleared that and returned -1).
	uint32_t gate = 0;
	Phase41SafeRead(&gate, reinterpret_cast<const void*>(obj + 8), sizeof(gate));
	if (gate == 0) {
		gate = 1;
		Phase41SafeWrite(reinterpret_cast<volatile void*>(obj + 8), &gate, sizeof(gate));
	}
	const char* poke = std::getenv("KYTY_PHASE41_POKE");
	if (poke == nullptr || poke[0] != '1') {
		return;
	}
	if (field_ptr >= 0x10000ULL && field_ptr < 0x0000800000000000ULL) {
		uint32_t ready = 0;
		Phase41SafeRead(&ready, reinterpret_cast<const void*>(field_ptr), sizeof(ready));
		if (ready == 0) {
			ready = 1;
			Phase41SafeWrite(reinterpret_cast<volatile void*>(field_ptr), &ready, sizeof(ready));
			LOGF("FlipTrace: phase41 HLE side-effect field_ptr=0x%016" PRIx64 " ready=1\n",
			     field_ptr);
		}
	}
}

// Phase 42/44: restore keep[1] +0x20858 bitmap/slots from ENTER snap (full 64 B with
// slot ptrs — not bits alone) + wake status consumers. Never replay live *obj=-1.
static void Phase42RearmNdJobEnqueue() {
	static std::atomic<int> rearm_n {0};
	const int               n   = rearm_n.fetch_add(1, std::memory_order_relaxed);
	const uint64_t          obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL) {
		return;
	}
	const uint64_t before_head =
	    *reinterpret_cast<const uint64_t*>(g_phase41_keep1_field_before);
	// Phase 44: always re-assert full ENTER field snap (bits + slot ptrs) when present.
	if (before_head != 0) {
		Phase41SafeWrite(reinterpret_cast<volatile void*>(obj + kPhase41Keep1FieldOff),
		                 g_phase41_keep1_field_before, 64);
		if (n < 8 || (n % 50) == 0) {
			LOGF("FlipTrace: phase44 NdJob rearm restored field+0x20858 from ENTER snap "
			     "obj=0x%016" PRIx64 " before=0x%016" PRIx64 "\n",
			     obj, before_head);
			fprintf(stderr, "FlipTrace: phase44 NdJob rearm restored field\n");
		}
	} else {
		uint64_t field_head = 0;
		Phase41SafeRead(&field_head, reinterpret_cast<const void*>(obj + kPhase41Keep1FieldOff),
		                sizeof(field_head));
		if (field_head == 0) {
			const uint64_t bits = 0x7fffffULL;
			Phase41SafeWrite(reinterpret_cast<volatile void*>(obj + kPhase41Keep1FieldOff), &bits,
			                 sizeof(bits));
			if (n < 4) {
				LOGF("FlipTrace: phase44 NdJob rearm seeded field bits=0x7fffff obj=0x%016" PRIx64
				     "\n",
				     obj);
				fprintf(stderr, "FlipTrace: phase44 NdJob rearm seeded field bits\n");
			}
		}
	}
	const uint64_t status = g_phase41_status_rdi.load(std::memory_order_acquire);
	if (status >= 0x10000ULL && status < 0x0000800000000000ULL) {
		uint32_t fill[8] = {1u, 1u, 0u, 0u, 1u, 0u, 0u, 0u};
		Phase41SafeWrite(reinterpret_cast<volatile void*>(status), fill, sizeof(fill));
	}
	// Keep gate+8=1; do not write *obj (live path sets -1 and kills enqueue).
	uint32_t gate = 1;
	Phase41SafeWrite(reinterpret_cast<volatile void*>(obj + 8), &gate, sizeof(gate));
	uint32_t pending = 1;
	Phase41SafeWrite(reinterpret_cast<volatile void*>(obj + 0x10), &pending, sizeof(pending));
	Phase41SafeWrite(reinterpret_cast<volatile void*>(obj + 0x18), &pending, sizeof(pending));
	// Extra EQ / status wakes so consumers leave WaitEqueue toward RegisterBuffers2.
	(void)LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
	(void)LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
}

// Defined after VideoOutConfig / g_video_out_context / phase37 atomics.
static void Phase41AttemptSnapshotReregister();
static void Phase43SustainMenuFlips();
static void Phase43UpdateMenuFramesOk();
static void Phase43SeedNdJobDcb();
static void Phase44ClearSnapshotForGuestRetry();
static void Phase44CheckNdJobDcb();
static void Phase44CaptureDcbBaselineIfNeeded();
static void Phase44AttemptAbiReregister();

void Phase41MenuHandoffAttempt() {
	Phase41FlushKeep1Log();
	const uint64_t n = g_phase41_handoff_n.fetch_add(1, std::memory_order_relaxed);
	const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	const uint64_t status = g_phase41_status_rdi.load(std::memory_order_acquire);

	// Re-assert rich status pattern (eax path already returned; buffer may be cleared).
	if (status >= 0x10000ULL && status < 0x0000800000000000ULL) {
		uint32_t fill[8] = {1u, 1u, 0u, 0u, 1u, 0u, 0u, 0u};
		Phase41SafeWrite(reinterpret_cast<volatile void*>(status), fill, sizeof(fill));
	}
	Phase41ApplyKeep1SideEffects();
	Phase42RearmNdJobEnqueue();
	Phase44CheckNdJobDcb();

	// Host snapshot fallback while waiting for sustained frames (adopt keep-alive set).
	// After clear-once for ABI Reg2, do not re-snapshot (avoids thrash).
	if (!g_phase44_guest_reg2_ok.load(std::memory_order_acquire) &&
	    !g_phase44_snapshot_done.load(std::memory_order_acquire) &&
	    !g_phase44_snapshot_cleared.load(std::memory_order_acquire) &&
	    (n == 40 || n == 80 || (n > 80 && (n % 100) == 0))) {
		Phase41AttemptSnapshotReregister();
	}
	// After sustained frames: clear host set once, then ABI re-Register from MainThread.
	if (g_phase43_menu_frames_ok.load(std::memory_order_acquire) &&
	    !g_phase44_guest_reg2_ok.load(std::memory_order_acquire)) {
		if (g_phase44_snapshot_done.load(std::memory_order_acquire) &&
		    !g_phase44_snapshot_cleared.load(std::memory_order_acquire)) {
			Phase44ClearSnapshotForGuestRetry();
		}
		Phase44AttemptAbiReregister();
	}
	// Sustain flips while buffers are linked — continue after menu_frames_ok until Reg2.
	if (g_phase42_reregister_ok.load(std::memory_order_acquire) && n >= 6) {
		if (!g_phase43_menu_frames_ok.load(std::memory_order_acquire)) {
			Phase43SeedNdJobDcb();
			Phase43SustainMenuFlips();
			Phase43UpdateMenuFramesOk();
		} else if (!g_phase44_guest_reg2_ok.load(std::memory_order_acquire)) {
			Phase43SustainMenuFlips();
		}
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Re-enter keep[1] live stub — opt-in KYTY_PHASE41_RECALL_KEEP1=1 (can AV).
	const char* recall = std::getenv("KYTY_PHASE41_RECALL_KEEP1");
	if (recall != nullptr && recall[0] == '1' && obj >= 0x10000ULL &&
	    g_phase41_call_thunk != nullptr && (n % 5) == 0) {
		using Keep1Call = uint32_t (*)(uint64_t);
		uint32_t eax    = 0;
		__try {
			eax = reinterpret_cast<Keep1Call>(g_phase41_call_thunk)(obj);
			if (n < 16 || (n % 50) == 0) {
				LOGF("FlipTrace: phase41 handoff keep[1] re-call n=%" PRIu64
				     " obj=0x%016" PRIx64 " eax=%u\n",
				     n, obj, eax);
				fprintf(stderr,
				        "FlipTrace: phase41 handoff keep[1] re-call n=%" PRIu64 " eax=%u\n", n,
				        eax);
			}
		} __except (EXCEPTION_EXECUTE_HANDLER) {
			if (n < 8) {
				LOGF("FlipTrace: phase41 handoff keep[1] re-call FAULT n=%" PRIu64
				     " code=0x%08lx\n",
				     n, GetExceptionCode());
				fprintf(stderr, "FlipTrace: phase41 handoff keep[1] re-call FAULT\n");
			}
		}
	}
#endif

	const size_t woken = LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
	const int    ue =
	    LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
	if (n < 8 || (n % 50) == 0) {
		LOGF("FlipTrace: phase41 handoff n=%" PRIu64 " obj=0x%016" PRIx64
		     " status=0x%016" PRIx64 " woken=%zu ue=%d keep1_hits=%" PRIu64
		     " rereg=%d flip_seen=%d\n",
		     n, obj, status, woken, ue, g_phase41_keep1_hits.load(std::memory_order_relaxed),
		     g_phase42_reregister_ok.load(std::memory_order_relaxed) ? 1 : 0,
		     g_phase37_guest_flip_seen.load(std::memory_order_relaxed) ? 1 : 0);
		fprintf(stderr, "FlipTrace: phase41 handoff n=%" PRIu64 " woken=%zu\n", n, woken);
	}
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

constexpr uint32_t kPhase40PrologueCopy = 64;
constexpr uint32_t kPhase40SnapBytes    = 128;

struct Phase40Slot {
	uint64_t           target        = 0;
	uint32_t           index         = 0;
	uint8_t            orig[kPhase40PrologueCopy] {};
	uint8_t*           live_entry    = nullptr;
	uint8_t*           entry_thunk   = nullptr;
	uint64_t           hle_continue  = 0; // RIP to resume into HLE path inside entry_thunk
	std::atomic<int>   veh_hits {0};
	std::atomic<int>   live_ok {0};
	uint64_t           av_rip        = 0;
	uint64_t           av_addr       = 0;
	uint8_t            rdi_before[kPhase40SnapBytes] {};
	uint8_t            rdi_after[kPhase40SnapBytes] {};
	uint8_t            rsi_before[kPhase40SnapBytes] {};
	uint8_t            rsi_after[kPhase40SnapBytes] {};
	uint8_t            captured_rdi[kPhase40SnapBytes] {};
	bool               have_captured = false;
	uint64_t           last_rdi      = 0;
	uint64_t           last_rsi      = 0;
};

static Phase40Slot              g_phase40_slots[5] {};
static std::atomic<uint32_t>    g_phase40_active_idx {UINT32_MAX};
static PVOID                    g_phase40_veh = nullptr;
static thread_local uint32_t    g_phase40_tls_idx = UINT32_MAX;
static thread_local uint64_t    g_phase40_saved_rsp = 0;

static bool Phase40LiveEnabled() {
	const char* off = std::getenv("KYTY_PHASE40_LIVE");
	return off == nullptr || off[0] != '0';
}

static bool Phase40LiveForIndex(uint32_t index) {
	if (!Phase40LiveEnabled()) {
		return false;
	}
	// Live guest bodies for [3]/[4] end in non-continuable failfast (0xC0000409);
	// VEH cannot resume. Opt-in only: KYTY_PHASE40_LIVE_3=1 / LIVE_4=1.
	if (index == 3) {
		const char* live3 = std::getenv("KYTY_PHASE40_LIVE_3");
		return live3 != nullptr && live3[0] == '1';
	}
	if (index == 4) {
		const char* live4 = std::getenv("KYTY_PHASE40_LIVE_4");
		return live4 != nullptr && live4[0] == '1';
	}
	return false;
}

static void Phase40SafeMemcpy(void* dst, const void* src, size_t n) {
#if defined(_MSC_VER)
	__try {
		std::memcpy(dst, src, n);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		std::memset(dst, 0, n);
	}
#else
	std::memcpy(dst, src, n);
#endif
}

static void Phase40SafeMemsetWrite(volatile void* dst, const void* src, size_t n) {
#if defined(_MSC_VER)
	__try {
		std::memcpy(const_cast<void*>(static_cast<const volatile void*>(dst)), src, n);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
	}
#else
	std::memcpy(const_cast<void*>(dst), src, n);
#endif
}

static void Phase40DumpHex(const char* tag, uint32_t index, const uint8_t* buf, size_t n) {
	char line[320];
	int  pos = std::snprintf(line, sizeof(line), "FlipTrace: phase40 %s[%u]", tag, index);
	for (size_t i = 0; i < n && pos > 0 && static_cast<size_t>(pos) + 3 < sizeof(line); i++) {
		pos += std::snprintf(line + pos, sizeof(line) - static_cast<size_t>(pos), " %02x", buf[i]);
	}
	LOGF("%s\n", line);
	fprintf(stderr, "%s\n", line);
}

extern "C" void Phase40ReapplyHook(uint32_t index);
extern "C" void Phase40RestoreLiveBody(uint32_t index);

static void Phase40ApplyHleStatus(uint64_t guest_rdi, Phase40Slot* slot) {
	if (guest_rdi < 0x10000ULL || guest_rdi >= 0x0000800000000000ULL) {
		return;
	}
	g_phase41_status_rdi.store(guest_rdi, std::memory_order_release);
	if (slot != nullptr && slot->have_captured) {
		Phase40SafeMemsetWrite(reinterpret_cast<volatile void*>(guest_rdi), slot->captured_rdi,
		                       kPhase40SnapBytes);
		Phase41ApplyKeep1SideEffects();
		return;
	}
	// Phase 41 rich HLE: status dword0=1 (epilogue test ecx;jle), dword1=1 ready,
	// dword4=1 secondary gate — never LIVE failfast bodies.
	uint32_t fill[8] = {1u, 1u, 0u, 0u, 1u, 0u, 0u, 0u};
	Phase40SafeMemsetWrite(reinterpret_cast<volatile void*>(guest_rdi), fill, sizeof(fill));
	Phase41ApplyKeep1SideEffects();
	static std::atomic<int> rich_log {0};
	if (rich_log.fetch_add(1, std::memory_order_relaxed) < 8) {
		LOGF("FlipTrace: phase41 rich HLE status rdi=0x%016" PRIx64
		     " pattern=1,1,0,0,1 keep1_obj=0x%016" PRIx64 "\n",
		     guest_rdi, g_phase41_keep1_obj.load(std::memory_order_relaxed));
		fprintf(stderr, "FlipTrace: phase41 rich HLE status rdi=0x%016" PRIx64 "\n", guest_rdi);
	}
}

// Host-side HLE body (Microsoft x64). Also used as VEH fallback.
extern "C" uint32_t Phase39HleDispatch(uint32_t index, uint64_t guest_rdi, uint64_t guest_rsi,
                                       uint64_t guest_ra) {
	const uint64_t hit =
	    g_phase39_hle_hits[index < 5 ? index : 0].fetch_add(1, std::memory_order_relaxed);
	Phase40Slot* slot = (index < 5) ? &g_phase40_slots[index] : nullptr;
	uint64_t     qword0 = 0;
	uint64_t     qword1 = 0;
	if (guest_rdi >= 0x10000ULL && guest_rdi < 0x0000800000000000ULL) {
		Phase40SafeMemcpy(&qword0, reinterpret_cast<const void*>(guest_rdi), sizeof(qword0));
		Phase40SafeMemcpy(&qword1, reinterpret_cast<const void*>(guest_rdi + 8), sizeof(qword1));
	}
	if (hit < 16) {
		LOGF("FlipTrace: phase39 HLE hit[%u] n=%" PRIu64 " guest_ra=0x%016" PRIx64
		     " rdi=0x%016" PRIx64 " rsi=0x%016" PRIx64 " [*rdi]=0x%016" PRIx64
		     " [*rdi+8]=0x%016" PRIx64 " veh=%d captured=%d\n",
		     index, hit + 1, guest_ra, guest_rdi, guest_rsi, qword0, qword1,
		     slot != nullptr ? slot->veh_hits.load() : 0, slot != nullptr && slot->have_captured ? 1 : 0);
		fprintf(stderr,
		        "FlipTrace: phase39 HLE hit[%u] n=%" PRIu64 " ra=0x%016" PRIx64
		        " [*rdi]=0x%016" PRIx64 "\n",
		        index, hit + 1, guest_ra, qword0);
	}
	Phase40ApplyHleStatus(guest_rdi, slot);
	Phase38NudgeBootWorkersOnce();
	Phase39SignalVblankAndLabels();
	Phase41MenuHandoffAttempt();
	return 1u;
}

static LONG CALLBACK Phase40VectoredHandler(PEXCEPTION_POINTERS info) {
	if (info == nullptr || info->ExceptionRecord == nullptr || info->ContextRecord == nullptr) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	const uint32_t idx = g_phase40_tls_idx;
	if (idx >= 5 || g_phase40_active_idx.load(std::memory_order_acquire) != idx) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	Phase40Slot& slot = g_phase40_slots[idx];
	if (slot.hle_continue == 0) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	// Catch ANY exception during live probe (AV, failfast-ish, illegal, etc.).
	const uint32_t code = info->ExceptionRecord->ExceptionCode;
	const uint64_t rip  = static_cast<uint64_t>(info->ContextRecord->Rip);
	uint64_t       fault = 0;
	if (code == EXCEPTION_ACCESS_VIOLATION && info->ExceptionRecord->NumberParameters >= 2) {
		fault = static_cast<uint64_t>(info->ExceptionRecord->ExceptionInformation[1]);
	}
	slot.av_rip   = rip;
	slot.av_addr  = fault;
	slot.last_rdi = info->ContextRecord->Rdi;
	slot.last_rsi = info->ContextRecord->Rsi;
	if (slot.last_rdi >= 0x10000ULL && slot.last_rdi < 0x0000800000000000ULL) {
		Phase40SafeMemcpy(slot.rdi_after, reinterpret_cast<const void*>(slot.last_rdi),
		                  kPhase40SnapBytes);
		std::memcpy(slot.captured_rdi, slot.rdi_after, kPhase40SnapBytes);
		slot.have_captured = true;
	}
	if (slot.last_rsi >= 0x10000ULL && slot.last_rsi < 0x0000800000000000ULL) {
		Phase40SafeMemcpy(slot.rsi_after, reinterpret_cast<const void*>(slot.last_rsi),
		                  kPhase40SnapBytes);
	}
	const int n = slot.veh_hits.fetch_add(1, std::memory_order_relaxed);
	if (n < 8) {
		LOGF("FlipTrace: phase40 live EXC idx=%u n=%d code=0x%08" PRIx32 " rip=0x%016" PRIx64
		     " fault=0x%016" PRIx64 " rdi=0x%016" PRIx64 " rsi=0x%016" PRIx64 " -> HLE\n",
		     idx, n + 1, code, rip, fault, slot.last_rdi, slot.last_rsi);
		fprintf(stderr,
		        "FlipTrace: phase40 live EXC idx=%u code=0x%08" PRIx32 " rip=0x%016" PRIx64 "\n",
		        idx, code, rip);
		Phase40DumpHex("rdi_before", idx, slot.rdi_before, 32);
		Phase40DumpHex("rdi_after", idx, slot.rdi_after, 32);
		Phase40DumpHex("rsi_before", idx, slot.rsi_before, 32);
		Phase40DumpHex("rsi_after", idx, slot.rsi_after, 32);
	}
	Phase40ReapplyHook(idx);
	if (g_phase40_saved_rsp != 0) {
		info->ContextRecord->Rsp = g_phase40_saved_rsp;
	}
	info->ContextRecord->Rip = slot.hle_continue;
	info->ContextRecord->Rax = 1;
	g_phase40_tls_idx        = UINT32_MAX;
	g_phase40_active_idx.store(UINT32_MAX, std::memory_order_release);
	g_phase40_saved_rsp = 0;
	return EXCEPTION_CONTINUE_EXECUTION;
}

static void Phase40EnsureVeh() {
	static std::atomic<bool> once {false};
	if (once.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	g_phase40_veh = AddVectoredExceptionHandler(1, Phase40VectoredHandler);
	LOGF("FlipTrace: phase40 VEH installed %s\n", g_phase40_veh != nullptr ? "ok" : "FAIL");
	fprintf(stderr, "FlipTrace: phase40 VEH installed %s\n",
	        g_phase40_veh != nullptr ? "ok" : "FAIL");
}

// Called from entry thunk (Microsoft ABI): prepare live call, return live_entry pointer.
extern "C" uint64_t Phase40PrepareLive(uint32_t index, uint64_t guest_rdi, uint64_t guest_rsi) {
	if (index >= 5) {
		return 0;
	}
	Phase40Slot& slot = g_phase40_slots[index];
	slot.last_rdi     = guest_rdi;
	slot.last_rsi     = guest_rsi;
	std::memset(slot.rdi_before, 0, sizeof(slot.rdi_before));
	std::memset(slot.rsi_before, 0, sizeof(slot.rsi_before));
	std::memset(slot.rdi_after, 0, sizeof(slot.rdi_after));
	std::memset(slot.rsi_after, 0, sizeof(slot.rsi_after));
	if (guest_rdi >= 0x10000ULL && guest_rdi < 0x0000800000000000ULL) {
		Phase40SafeMemcpy(slot.rdi_before, reinterpret_cast<const void*>(guest_rdi),
		                  kPhase40SnapBytes);
	}
	if (guest_rsi >= 0x10000ULL && guest_rsi < 0x0000800000000000ULL) {
		Phase40SafeMemcpy(slot.rsi_before, reinterpret_cast<const void*>(guest_rsi),
		                  kPhase40SnapBytes);
	}
	g_phase40_tls_idx = index;
	g_phase40_active_idx.store(index, std::memory_order_release);
	Phase40RestoreLiveBody(index);
	return slot.target; // call original guest body in-place
}

extern "C" void Phase40ArmProbe(uint32_t index) {
	if (index >= 5) {
		return;
	}
	g_phase40_tls_idx = index;
	g_phase40_active_idx.store(index, std::memory_order_release);
	LOGF("FlipTrace: phase40 arm probe idx=%u\n", index);
	fprintf(stderr, "FlipTrace: phase40 arm probe idx=%u\n", index);
}

extern "C" void Phase40ArmRsp(uint64_t rsp) {
	g_phase40_saved_rsp = rsp;
}

extern "C" void Phase40RestoreLiveBody(uint32_t index) {
	if (index >= 5) {
		return;
	}
	Phase40Slot& slot = g_phase40_slots[index];
	if (slot.target < 0x1000) {
		return;
	}
	auto* guest = reinterpret_cast<uint8_t*>(slot.target);
	DWORD old_prot = 0;
	if (VirtualProtect(guest, kPhase40PrologueCopy, PAGE_EXECUTE_READWRITE, &old_prot) == 0) {
		return;
	}
	std::memcpy(guest, slot.orig, kPhase40PrologueCopy);
	DWORD unused = 0;
	VirtualProtect(guest, kPhase40PrologueCopy, old_prot, &unused);
}

extern "C" void Phase40ReapplyHook(uint32_t index) {
	if (index >= 5) {
		return;
	}
	Phase40Slot& slot = g_phase40_slots[index];
	if (slot.target < 0x1000 || slot.entry_thunk == nullptr) {
		return;
	}
	auto* guest = reinterpret_cast<uint8_t*>(slot.target);
	DWORD old_prot = 0;
	if (VirtualProtect(guest, 16, PAGE_EXECUTE_READWRITE, &old_prot) == 0) {
		return;
	}
	guest[0] = 0x48;
	guest[1] = 0xb8;
	const uint64_t thunk_addr = reinterpret_cast<uint64_t>(slot.entry_thunk);
	std::memcpy(guest + 2, &thunk_addr, 8);
	guest[10] = 0xff;
	guest[11] = 0xe0;
	DWORD unused = 0;
	VirtualProtect(guest, 16, old_prot, &unused);
}

extern "C" void Phase40AfterLiveOk(uint32_t index, uint64_t guest_rdi, uint32_t eax) {
	if (index >= 5) {
		return;
	}
	Phase40Slot& slot = g_phase40_slots[index];
	Phase40ReapplyHook(index);
	slot.live_ok.fetch_add(1, std::memory_order_relaxed);
	if (guest_rdi >= 0x10000ULL && guest_rdi < 0x0000800000000000ULL) {
		Phase40SafeMemcpy(slot.rdi_after, reinterpret_cast<const void*>(guest_rdi),
		                  kPhase40SnapBytes);
		std::memcpy(slot.captured_rdi, slot.rdi_after, kPhase40SnapBytes);
		slot.have_captured = true;
	}
	g_phase40_tls_idx = UINT32_MAX;
	g_phase40_active_idx.store(UINT32_MAX, std::memory_order_release);
	g_phase40_saved_rsp = 0;
	LOGF("FlipTrace: phase40 live OK idx=%u eax=%u rdi=0x%016" PRIx64 "\n", index, eax, guest_rdi);
	fprintf(stderr, "FlipTrace: phase40 live OK idx=%u eax=%u\n", index, eax);
	Phase40DumpHex("rdi_after_ok", index, slot.rdi_after, 32);
	Phase38NudgeBootWorkersOnce();
	Phase39SignalVblankAndLabels();
}

static bool Phase40InstallLiveProbeTrampoline(uint64_t target, uint32_t index) {
	if (target < 0x1000 || index >= 5) {
		return false;
	}
	Phase40EnsureVeh();
	Phase40Slot& slot = g_phase40_slots[index];
	slot.target       = target;
	slot.index        = index;
	Phase40SafeMemcpy(slot.orig, reinterpret_cast<const void*>(target), kPhase40PrologueCopy);
	slot.live_entry = nullptr; // unused — call restored target in-place

	// entry_thunk: restore body, call target, re-hook / HLE on AV
	slot.entry_thunk = static_cast<uint8_t*>(
	    VirtualAlloc(nullptr, 256, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	if (slot.entry_thunk == nullptr) {
		return false;
	}
	uint8_t* p    = slot.entry_thunk;
	auto     emit = [&](std::initializer_list<uint8_t> bytes) {
        for (uint8_t b: bytes) {
			*p++ = b;
		}
	};
	auto emit_u32 = [&](uint32_t v) {
		std::memcpy(p, &v, 4);
		p += 4;
	};
	auto emit_u64 = [&](uint64_t v) {
		std::memcpy(p, &v, 8);
		p += 8;
	};

	// Save guest RA into r12, args into r10/r11
	emit({0x4c, 0x8b, 0x24, 0x24});             // mov r12, [rsp]
	emit({0x49, 0x89, 0xfa});                   // mov r10, rdi
	emit({0x49, 0x89, 0xf3});                   // mov r11, rsi

	if (Phase40LiveForIndex(index)) {
		// Arm probe TLS before any live work so VEH catches all faults.
		emit({0xb9});
		emit_u32(index);
		emit({0x48, 0x83, 0xec, 0x28});
		emit({0x48, 0xb8});
		emit_u64(reinterpret_cast<uint64_t>(&Phase40ArmProbe));
		emit({0xff, 0xd0});
		emit({0x48, 0x83, 0xc4, 0x28});
		// Phase40PrepareLive(index, rdi, rsi) -> rax = target
		emit({0xb9});
		emit_u32(index);                        // mov ecx, index
		emit({0x4c, 0x89, 0xd2});               // mov rdx, r10
		emit({0x4d, 0x89, 0xd8});               // mov r8, r11
		emit({0x48, 0x83, 0xec, 0x28});         // sub rsp, 0x28
		emit({0x48, 0xb8});
		emit_u64(reinterpret_cast<uint64_t>(&Phase40PrepareLive));
		emit({0xff, 0xd0});                     // call rax
		emit({0x48, 0x83, 0xc4, 0x28});         // add rsp, 0x28
		emit({0x49, 0x89, 0xc5});               // mov r13, rax (target)
		// ArmRsp(rsp) so VEH can restore stack
		emit({0x48, 0x89, 0xe1});               // mov rcx, rsp
		emit({0x48, 0x83, 0xec, 0x28});
		emit({0x48, 0xb8});
		emit_u64(reinterpret_cast<uint64_t>(&Phase40ArmRsp));
		emit({0xff, 0xd0});
		emit({0x48, 0x83, 0xc4, 0x28});
		// call live with SysV args
		emit({0x4c, 0x89, 0xd7});               // mov rdi, r10
		emit({0x4c, 0x89, 0xde});               // mov rsi, r11
		emit({0x48, 0x83, 0xec, 0x28});         // sub rsp, 0x28
		emit({0x41, 0xff, 0xd5});               // call r13
		emit({0x48, 0x83, 0xc4, 0x28});         // add rsp, 0x28
		// live succeeded
		emit({0x89, 0xc6});                     // mov esi, eax
		emit({0xb9});
		emit_u32(index);
		emit({0x4c, 0x89, 0xd2});               // mov rdx, r10
		emit({0x41, 0x89, 0xf0});               // mov r8d, esi
		emit({0x48, 0x83, 0xec, 0x28});
		emit({0x48, 0xb8});
		emit_u64(reinterpret_cast<uint64_t>(&Phase40AfterLiveOk));
		emit({0xff, 0xd0});
		emit({0x48, 0x83, 0xc4, 0x28});
		emit({0x89, 0xf0});                     // mov eax, esi
		emit({0xc3});                           // ret to guest
	}

	// hle_continue: VEH lands here after stack restore
	slot.hle_continue = reinterpret_cast<uint64_t>(p);
	emit({0xb9});
	emit_u32(index);                            // mov ecx, index
	emit({0x4c, 0x89, 0xd2});                   // mov rdx, r10
	emit({0x4d, 0x89, 0xd8});                   // mov r8, r11
	emit({0x4d, 0x89, 0xe1});                   // mov r9, r12 (guest_ra)
	emit({0x48, 0x83, 0xec, 0x28});
	emit({0x48, 0xb8});
	emit_u64(reinterpret_cast<uint64_t>(&Phase39HleDispatch));
	emit({0xff, 0xd0});
	emit({0x48, 0x83, 0xc4, 0x28});
	emit({0xc3});

	// Patch guest target: mov rax, entry_thunk; jmp rax
	auto* guest = reinterpret_cast<uint8_t*>(target);
	DWORD old_prot = 0;
	if (VirtualProtect(guest, 16, PAGE_EXECUTE_READWRITE, &old_prot) == 0) {
		VirtualFree(slot.entry_thunk, 0, MEM_RELEASE);
		return false;
	}
	guest[0] = 0x48;
	guest[1] = 0xb8;
	const uint64_t thunk_addr = reinterpret_cast<uint64_t>(slot.entry_thunk);
	std::memcpy(guest + 2, &thunk_addr, 8);
	guest[10] = 0xff;
	guest[11] = 0xe0;
	DWORD unused = 0;
	VirtualProtect(guest, 16, old_prot, &unused);

	LOGF("FlipTrace: phase40 live probe[%u] tgt=0x%016" PRIx64 " thunk=%p hle=0x%016" PRIx64
	     " live_body=%d\n",
	     index, target, static_cast<void*>(slot.entry_thunk), slot.hle_continue,
	     Phase40LiveForIndex(index) ? 1 : 0);
	fprintf(stderr, "FlipTrace: phase40 live probe[%u] tgt=0x%016" PRIx64 " live_body=%d\n",
	        index, target, Phase40LiveForIndex(index) ? 1 : 0);
	return true;
}

static bool Phase39InstallHleTrampoline(uint64_t target, uint32_t index) {
	// Phase 40: live probe trampoline (falls back to HLE on AV).
	return Phase40InstallLiveProbeTrampoline(target, index);
}
#endif

static bool Phase38HleStubGuestProc(uint64_t target, uint32_t index, uint64_t call_site) {
	Phase39TraceHandoffTarget(index, call_site, target, "hle");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (Phase39InstallHleTrampoline(target, index)) {
		Phase38NudgeBootWorkersOnce();
		Phase39SignalVblankAndLabels();
		return true;
	}
#endif
	// Fallback: static ret (no call-time side effects).
	char tag[48];
	std::snprintf(tag, sizeof(tag), "phase39HleFallback[%u]", index);
	if (!Phase33StubGuestProc(target, tag)) {
		return false;
	}
	LOGF("FlipTrace: phase39 HLE fallback postUnreg[%u] tgt=0x%016" PRIx64 "\n", index, target);
	Phase38NudgeBootWorkersOnce();
	return true;
}

// Phase 33 default: skip r15 callback only (je→jmp) + status eax=0; stub remaining
// teardown targets so body can run without failfast/FiberSwitch storm.
// KYTY_PHASE32_SKIP_R15_ONLY=0 — skip-body+eax0 (CRT → catchReturnFromMain divert).
// Fallbacks:
//   KYTY_PHASE32_EPILOGUE_SKIP=1 — mov eax,1; jmp status (needs PARK_EXIT)
//   KYTY_PHASE32_SUCCESS_EPILOGUE=1 / RET_PATCH=1 — diag
//   KYTY_PHASE32_EMPTY_SUBMIT=1 — inject SubmitCommandBuffer(0,null,0)
static bool Phase32PatchPostUnregisterReturn(void* ra) {
	if (ra == nullptr || !Phase32Pending0Enabled()) {
		return false;
	}
	auto* p = reinterpret_cast<uint8_t*>(ra);
	const uint64_t ra64 = reinterpret_cast<uint64_t>(ra);
	DumpGuestCodeAround(ra64);
	DumpNextGuestCalls(ra64, 96, 8, "postUnreg");

	// Soft-stub post-Unregister call targets (Phase 36 selective mask).
	// Map: [0] failfast r15 — must stub; [1..3] teardown; [4] status producer.
	// KYTY_PHASE33_STUB_MASK=0xNN — bit i stubs postUnregStub[i] (hex/dec). Default 0x01
	// (failfast only) so guest body can run; raise mask if AV/C0000409.
	// Legacy: STUB_N / STUB_ALL / NO_STUB still work when STUB_MASK unset.
	const char* no_stub   = std::getenv("KYTY_PHASE33_NO_STUB");
	const char* stub_all  = std::getenv("KYTY_PHASE33_STUB_ALL");
	const char* stub_n    = std::getenv("KYTY_PHASE33_STUB_N");
	const char* stub_mask = std::getenv("KYTY_PHASE33_STUB_MASK");
	uint32_t    mask      = 0x19u; // Phase 37 nominal: stub [0,3,4]; keep [1,2] (bisect-proven)
	bool        use_mask  = true;
	uint32_t    max_scan  = 8;
	if (no_stub == nullptr || no_stub[0] != '1') {
		if (stub_mask != nullptr && stub_mask[0] != '\0') {
			char* end = nullptr;
			mask      = static_cast<uint32_t>(std::strtoul(stub_mask, &end, 0));
		} else if (stub_all != nullptr && stub_all[0] == '1') {
			use_mask = false;
			max_scan = 8;
			mask     = 0xffu;
		} else if (stub_n != nullptr && stub_n[0] != '\0') {
			use_mask = false;
			max_scan = static_cast<uint32_t>(std::atoi(stub_n));
			mask     = (max_scan >= 32) ? 0xffffffffu : ((1u << max_scan) - 1u);
		} else {
			// Phase 37 nominal (bisect): stub failfast[0]+teardown[3]+status[4]; keep [1]+[2].
			// Unstub [3]→crash; unstub [4]→AV(-1). 0x19 survives with BootCards keep-alive.
			mask = 0x19u;
		}
		const char* live3 = std::getenv("KYTY_PHASE38_LIVE_3");
		const char* live4 = std::getenv("KYTY_PHASE38_LIVE_4");
		const auto* code  = reinterpret_cast<const uint8_t*>(ra64);
		uint32_t    found = 0;
		const uint32_t limit = use_mask ? 8u : max_scan;
		for (uint32_t off = 0; off + 5 <= 96 && found < limit; ++off) {
			if (code[off] != 0xe8) {
				continue;
			}
			const int32_t rel = static_cast<int32_t>(code[off + 1] | (code[off + 2] << 8) |
			                                         (code[off + 3] << 16) | (code[off + 4] << 24));
			const uint64_t target = ra64 + off + 5 + static_cast<int64_t>(rel);
			const bool     stub_it =
			    use_mask ? ((mask & (1u << found)) != 0) : (found < max_scan);
			char tag[40];
			// [3]/[4] live crash/AV — HLE unless KYTY_PHASE38_LIVE_3/4=1 for diag.
			const bool hle3 = (found == 3) && (live3 == nullptr || live3[0] != '1');
			const bool hle4 = (found == 4) && (live4 == nullptr || live4[0] != '1');
			const uint64_t call_site = ra64 + off;
			if (hle3 || hle4) {
				(void)Phase38HleStubGuestProc(target, found, call_site);
			} else if (stub_it) {
				std::snprintf(tag, sizeof(tag), "postUnregStub[%u]", found);
				(void)Phase33StubGuestProc(target, tag);
			} else {
				std::snprintf(tag, sizeof(tag), "postUnregKeep[%u]", found);
				Phase39TraceHandoffTarget(found, call_site, target, "keep");
				if (found == 1) {
					Phase41TraceKeep1AtPatch(target);
				}
				LOGF("FlipTrace: phase36 keep %s tgt=0x%016" PRIx64 " (mask=0x%x)\n", tag,
				     target, mask);
				fprintf(stderr, "FlipTrace: phase36 keep %s tgt=0x%016" PRIx64 "\n", tag,
				        target);
			}
			++found;
			off += 4;
		}
		LOGF("FlipTrace: phase38 stub policy mask=0x%x use_mask=%d scanned=%u "
		     "(HLE [3]/[4] unless LIVE_3/4)\n",
		     mask, use_mask ? 1 : 0, found);
		fprintf(stderr, "FlipTrace: phase38 stub policy mask=0x%x scanned=%u\n", mask, found);
	}

	int epilogue_rel   = -1;
	int status_mov_rel = -1;
	int add_rsp_rel    = -1;
	for (int i = 8; i < 160; ++i) {
		if (p[i] == 0x48 && p[i + 1] == 0x89 && p[i + 2] == 0xc1 && p[i + 3] == 0xb8 &&
		    p[i + 4] == 0x01 && p[i + 5] == 0x00 && p[i + 6] == 0x00 && p[i + 7] == 0x00 &&
		    p[i + 8] == 0x85 && p[i + 9] == 0xc9) {
			epilogue_rel   = i;
			status_mov_rel = i + 3;
			break;
		}
	}
	for (int i = 8; i < 160; ++i) {
		if (p[i] == 0x48 && p[i + 1] == 0x81 && p[i + 2] == 0xc4 && p[i + 7] == 0x5b) {
			add_rsp_rel = i;
			break;
		}
	}
	if (epilogue_rel < 0) {
		epilogue_rel = (add_rsp_rel > 0) ? add_rsp_rel : 70;
	}
	if (add_rsp_rel < 0) {
		add_rsp_rel = epilogue_rel;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	DWORD old_prot = 0;
	if (VirtualProtect(p, 96, PAGE_EXECUTE_READWRITE, &old_prot) == 0) {
		LOGF("FlipTrace: post-Unregister VirtualProtect failed ra=%p err=%lu\n", ra,
		     GetLastError());
		return false;
	}
#endif
	const char* mode         = "skip-r15+stubs+eax0";
	const char* success_env  = std::getenv("KYTY_PHASE32_SUCCESS_EPILOGUE");
	const char* ret_env      = std::getenv("KYTY_PHASE32_RET_PATCH");
	const char* epilogue_env = std::getenv("KYTY_PHASE32_EPILOGUE_SKIP");
	const char* r15_only     = std::getenv("KYTY_PHASE32_SKIP_R15_ONLY");
	auto        write_jmp_eax = [&](uint8_t eax_imm, int target_rel) {
		p[0] = 0xB8;
		p[1] = eax_imm;
		p[2] = 0x00;
		p[3] = 0x00;
		p[4] = 0x00;
		const int disp = target_rel - 7;
		if (disp >= -128 && disp <= 127) {
			p[5] = 0xEB;
			p[6] = static_cast<uint8_t>(disp);
		} else {
			p[5] = 0xE9;
			const int32_t d = target_rel - 10;
			std::memcpy(p + 6, &d, sizeof(d));
		}
	};

	if (ret_env != nullptr && ret_env[0] == '1') {
		mode = "ret";
		p[0] = 0xC3;
	} else if (epilogue_env != nullptr && epilogue_env[0] == '1') {
		mode = "epilogue-skip";
		write_jmp_eax(0x01, epilogue_rel);
	} else if (r15_only != nullptr && r15_only[0] == '0') {
		mode = "skip-body+eax0";
		write_jmp_eax(0x00, add_rsp_rel);
		epilogue_rel = add_rsp_rel;
	} else if (success_env != nullptr && success_env[0] == '1') {
		mode = "success-epilogue";
		write_jmp_eax(0x00, add_rsp_rel);
		epilogue_rel = add_rsp_rel;
	} else {
		// Default: skip r15 + stubs already applied; force status success.
		bool patched = false;
		for (int i = 0; i < 24; ++i) {
			if (p[i] == 0x4d && p[i + 1] == 0x85 && p[i + 2] == 0xff && p[i + 3] == 0x74) {
				p[i + 3] = 0xeb;
				patched  = true;
				break;
			}
		}
		if (status_mov_rel > 0) {
			// Force status=0 only when status producer [4] is stubbed; otherwise let body
			// / postUnregKeep[4] write the real status (Phase 36 guest continuation).
			if ((mask & 0x10u) != 0) {
				p[status_mov_rel + 1] = 0x00;
			} else {
				LOGF("FlipTrace: phase36 leave status_mov intact (mask=0x%x bit4 clear)\n",
				     mask);
			}
		}
		if (!patched) {
			mode = "skip-body+eax0-fallback";
			write_jmp_eax(0x00, add_rsp_rel);
			epilogue_rel = add_rsp_rel;
		}
	}

	const char* empty_submit = std::getenv("KYTY_PHASE32_EMPTY_SUBMIT");
	if (empty_submit != nullptr && empty_submit[0] == '1') {
		static std::atomic<bool> once {false};
		if (!once.exchange(true, std::memory_order_acq_rel)) {
			(void)Libs::Graphics::Gen5Driver::GraphicsDriverSubmitCommandBuffer(0, nullptr, 0);
			LOGF("SubmitTrace: phase32 EMPTY_SUBMIT Unregister->SubmitCommandBuffer(0,null,0)\n");
			fprintf(stderr, "SubmitTrace: phase32 EMPTY_SUBMIT\n");
		}
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	DWORD unused = 0;
	VirtualProtect(p, 96, old_prot, &unused);
#endif
	LOGF("FlipTrace: patched post-Unregister %s rel=%d status_mov=%d ra=0x%016" PRIx64 "\n", mode,
	     epilogue_rel, status_mov_rel, ra64);
	fprintf(stderr, "FlipTrace: patched post-Unregister %s rel=%d\n", mode, epilogue_rel);
	Common::LogFatalToFile("patched post-Unregister phase33");
	return true;
}

static void AfterPending0HostTick() {
	const int left = g_pending0_watchdog_ticks.load(std::memory_order_relaxed);
	if (left <= 0) {
		return;
	}
	if (g_pending0_watchdog_ticks.fetch_sub(1, std::memory_order_acq_rel) <= 0) {
		return;
	}
	char reason[64];
	std::snprintf(reason, sizeof(reason), "post_pending0_watchdog_%d", left);
	Common::FlushHleRingToFatal(reason);
	// Keep Mixed/Compute unblocked while MainThread runs post-Unregister boot work.
	if ((left % 5) == 0) {
		const auto woken = LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
		const int  ue =
		    LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
		LOGF("SubmitTrace: pending0 watchdog wake woken=%zu ue=%d left=%d\n", woken, ue, left);
		fprintf(stderr, "SubmitTrace: pending0 watchdog wake woken=%zu ue=%d left=%d\n", woken, ue,
		        left);
	}
}

[[maybe_unused]] static const char* ResolveJmprelaSymbol(Loader::Program* prog, uint32_t reloc_idx) {
	if (prog == nullptr || prog->dynamic_info == nullptr ||
	    prog->dynamic_info->jmprela_table == nullptr || prog->dynamic_info->symbol_table == nullptr ||
	    prog->dynamic_info->str_table == nullptr) {
		return nullptr;
	}
	const auto table_n =
	    prog->dynamic_info->jmprela_table_size / sizeof(Loader::Elf64_Rela);
	if (reloc_idx >= table_n) {
		return nullptr;
	}
	auto* rela = prog->dynamic_info->jmprela_table + reloc_idx;
	auto* syms = prog->dynamic_info->symbol_table;
	auto* strs = prog->dynamic_info->str_table;
	const auto sym = syms[rela->GetSymbol()];
	return strs + static_cast<uint32_t>(sym.st_name);
}

// Resolve classic PLT entry (ff 25 disp32 / push reloc_idx) at `target`.
// Also dump first bytes for non-PLT internal targets (BootCards helpers).
static void ResolveAndLogPltTarget(uint64_t call_site, uint64_t target, const char* tag) {
	const auto* plt = reinterpret_cast<const uint8_t*>(target);
	uint32_t    reloc_idx = 0;
	uint64_t    got_addr  = 0;
	uint64_t    got_val   = 0;
	const char* nm        = "?";
	char        bytes[48] {};
	int         bn = 0;
	for (int i = 0; i < 12 && bn + 3 < static_cast<int>(sizeof(bytes)); ++i) {
		bn += std::snprintf(bytes + bn, sizeof(bytes) - static_cast<size_t>(bn), "%02x", plt[i]);
	}
	if (plt[0] == 0xff && plt[1] == 0x25) {
		const int32_t got_rel =
		    static_cast<int32_t>(plt[2] | (plt[3] << 8) | (plt[4] << 16) | (plt[5] << 24));
		got_addr = target + 6 + static_cast<int64_t>(got_rel);
		std::memcpy(&got_val, reinterpret_cast<const void*>(got_addr), sizeof(got_val));
		if (plt[6] == 0x68) {
			reloc_idx =
			    static_cast<uint32_t>(plt[7] | (plt[8] << 8) | (plt[9] << 16) | (plt[10] << 24));
		}
		auto* prog =
		    Common::Singleton<Loader::RuntimeLinker>::Instance()->FindProgramByAddr(call_site);
		if (const char* resolved = ResolveJmprelaSymbol(prog, reloc_idx); resolved != nullptr) {
			nm = resolved;
		}
	} else if (plt[0] == 0xe9 || plt[0] == 0xeb) {
		nm = "jmp-stub";
	} else if (plt[0] == 0x55 || plt[0] == 0x48) {
		nm = "internal";
	}
	char msg[448];
	std::snprintf(msg, sizeof(msg),
	              "%s site=0x%016" PRIx64 " tgt=0x%016" PRIx64 " reloc=%u sym=%s got=0x%016" PRIx64
	              " bytes=%s",
	              tag, call_site, target, reloc_idx, nm, got_val, bytes);
	LOGF("FlipTrace: %s\n", msg);
	fprintf(stderr, "FlipTrace: %s\n", msg);
	Common::LogFatalToFile(msg);
}

// Scan up to `max_calls` relative CALLs in [addr, addr+span) and resolve PLT symbols.
// Phase 33/36 map @ Unregister RA (all guest-internal, not HLE/PLT):
//   [0] 0x901A547F0 — r15 callback (failfast) — stub bit0
//   [1] 0x901A1FBF0 — teardown helper
//   [2] 0x9000018A0 — teardown helper (may FiberSwitch)
//   [3] 0x900A9AB80 — teardown helper
//   [4] 0x900A87AB0 — status producer
// Phase 37/39 nominal STUB_MASK=0x19 — stub/HLE [0,3,4]; keep [1]+[2].
// [3]/[4] HLE trampoline (crash/AV if live). LIVE_3/4=1 for diag. Soft-idle only after guest progress.
static void DumpNextGuestCalls(uint64_t addr, uint32_t span, uint32_t max_calls, const char* tag) {
	const auto* code = reinterpret_cast<const uint8_t*>(addr);
	uint32_t    found = 0;
	for (uint32_t off = 0; off + 5 <= span && found < max_calls; ++off) {
		if (code[off] != 0xe8) {
			continue;
		}
		const int32_t rel = static_cast<int32_t>(code[off + 1] | (code[off + 2] << 8) |
		                                         (code[off + 3] << 16) | (code[off + 4] << 24));
		const uint64_t site   = addr + off;
		const uint64_t target = site + 5 + static_cast<int64_t>(rel);
		char           call_tag[48];
		std::snprintf(call_tag, sizeof(call_tag), "%s[%u]", tag, found);
		ResolveAndLogPltTarget(site, target, call_tag);
		++found;
		off += 4;
	}
	LOGF("FlipTrace: DumpNextGuestCalls %s found=%u in span=%u\n", tag, found, span);
	fprintf(stderr, "FlipTrace: DumpNextGuestCalls %s found=%u\n", tag, found);
}

constexpr int      VIDEO_OUT_EVENT_FLIP                                 = 0;
constexpr int      VIDEO_OUT_EVENT_VBLANK                               = 1;
constexpr int      VIDEO_OUT_EVENT_PRE_VBLANK_START                     = 2;
constexpr int      VIDEO_OUT_EVENT_SET_MODE                             = 8;
constexpr int      VIDEO_OUT_TRUE                                       = 1;
constexpr int      VIDEO_OUT_FALSE                                      = 0;
constexpr int      VIDEO_OUT_FLIP_MODE_VSYNC                            = 1;
constexpr int      VIDEO_OUT_FLIP_MODE_VSYNC_MULTI                      = 4;
constexpr int      VIDEO_OUT_BUFFER_INDEX_BLACK                         = -2;
constexpr int      VIDEO_OUT_BUFFER_INDEX_BLANK                         = -1;
constexpr int      VIDEO_OUT_BUFFER_NUM_MAX                             = 16;
constexpr size_t   VIDEO_OUT_FLIP_QUEUE_CAPACITY                        = 16;
constexpr int      VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX                   = 4;
constexpr uint64_t VIDEO_OUT_OUTPUT_MODE_DEFAULT                        = 0x0000000000000001ULL;
constexpr uint64_t VIDEO_OUT_OUTPUT_MODE_119_88HZ                       = 0x000000000000000FULL;
constexpr uint64_t VIDEO_OUT_REFRESH_RATE_59_94HZ                       = 3;
constexpr uint64_t VIDEO_OUT_REFRESH_RATE_119_88HZ                      = 13;
constexpr int      VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_UNCOMPRESSED     = 0;
constexpr int      VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_COMPRESSED       = 1;
constexpr uint64_t VIDEO_OUT_BUFFER_ATTRIBUTE_OPTION_STRICT_COLORIMETRY = 8;

enum class VideoOutEventKind : uintptr_t {
	Flip           = VIDEO_OUT_EVENT_FLIP,
	Vblank         = VIDEO_OUT_EVENT_VBLANK,
	PreVblankStart = VIDEO_OUT_EVENT_PRE_VBLANK_START,
	OutputMode     = VIDEO_OUT_EVENT_SET_MODE,
};

enum class FlipRequestSource { Cpu, GpuEop };

struct VideoOutBufferAttribute2 {
	uint32_t reserved0;
	uint32_t tiling_mode;
	uint32_t aspect_ratio;
	uint32_t width;
	uint32_t height;
	uint32_t pitch_in_pixel;
	uint64_t option;
	uint64_t pixel_format;
	uint64_t dcc_cb_register_clear_color;
	uint32_t dcc_control;
	uint32_t pad0;
	uint64_t reserved1[3];
};

// Phase 44: BootCards RegisterBuffers2 attribute (saved for ABI re-Register after Unreg).
static VideoOutBufferAttribute2 g_phase34_attr {};
static std::atomic<bool>        g_phase34_attr_valid {false};
static std::atomic<int>         g_phase34_attr_category {0};

// PS4/PS5 VideoOut FlipStatus (64 bytes) — matches shadPS4 / Orbis ABI.
// Oversized copies previously overflowed guest buffers and corrupted BootCards state.
struct VideoOutFlipStatus {
	uint64_t count         = 0;
	uint64_t processTime   = 0;
	uint64_t tsc           = 0;
	int64_t  flipArg       = -1;
	uint64_t submitTsc     = 0;
	uint64_t reserved0     = 0;
	int32_t  gcQueueNum    = 0;
	int32_t  flipPendingNum = 0;
	int32_t  currentBuffer = -1;
	uint32_t reserved1     = 0;
};
static_assert(sizeof(VideoOutFlipStatus) == 64, "FlipStatus must be 64 bytes (Orbis ABI)");

// PS5 layout
struct VideoOutVblankStatus {
	uint64_t count              = 0;
	uint64_t processTime        = 0;
	uint64_t reserved           = 0;
	uint64_t processTimeCounter = 0;
	uint8_t  flags              = 0;
	uint8_t  phase              = 0;
	uint8_t  pad1[6]            = {};
};

struct VideoOutOutputStatus {
	uint32_t resolution   = 0;
	uint32_t dynamicRange = 0;
	uint64_t refreshRate  = 0;
	uint64_t flags        = 0;
	uint64_t reserved[3]  = {};
};

struct VideoOutOutputOptions {
	uint32_t internalData[16] = {};
};

struct VideoOutColorSettings {
	float    gamma       = 1.0f;
	uint32_t reserved[3] = {};
};

// Conservative PS5-sized blob for VideoOutVrrStatus (exact layout unknown; zero-fill is safe).
struct VideoOutVrrStatus {
	uint64_t count              = 0;
	uint64_t processTime        = 0;
	uint64_t processTimeCounter = 0;
	uint32_t flags              = 0;
	uint32_t pad0               = 0;
	uint64_t reserved[5]        = {};
};

struct VideoOutBuffers {
	const void* data;
	const void* metadata;
	const void* reserved[2];
};

struct VideoOutBufferSet {
	int start_index = 0;
	int num         = 0;
	int set_id      = 0;
};

struct VideoOutBufferInfo {
	const void*                    buffer        = nullptr;
	Graphics::VideoOutVulkanImage* buffer_vulkan = nullptr;
	uint64_t                       buffer_size   = 0;
	uint64_t                       buffer_pitch  = 0;
	int                            set_id        = 0;
};

struct VideoOutConfig {
	Common::Mutex                         mutex;
	uint32_t                              width             = 0;
	uint32_t                              height            = 0;
	bool                                  opened            = false;
	bool                                  closing           = false;
	bool                                  unregistering[16] = {};
	int                                   flip_rate         = 0;
	int                                   prev_flip_index   = -1;
	std::vector<EventQueue::KernelEqueue> flip_eqs;
	std::vector<EventQueue::KernelEqueue> pre_vblank_eqs;
	std::vector<EventQueue::KernelEqueue> vblank_eqs;
	std::vector<EventQueue::KernelEqueue> output_mode_eqs;
	uint64_t                              output_mode = VIDEO_OUT_OUTPUT_MODE_DEFAULT;
	float                                 gamma       = 1.0f;
	VideoOutFlipStatus                    flip_status;
	VideoOutVblankStatus                  pre_vblank_status;
	VideoOutVblankStatus                  vblank_status;
	VideoOutBufferInfo                    buffers[16];
	// Guest/GPU flip labels (sceVideoOutGetBufferLabelAddress). Cleared to 0 when flipped.
	uint64_t                              buffer_labels[VIDEO_OUT_BUFFER_NUM_MAX] = {};
	std::vector<VideoOutBufferSet>        buffers_sets;
};

namespace FlipStats {
std::atomic<uint64_t> reserve_cpu {0};
std::atomic<uint64_t> reserve_gpu {0};
std::atomic<uint64_t> complete {0};
std::atomic<uint64_t> presented {0};
std::atomic<uint64_t> queue_empty {0};
std::atomic<uint64_t> not_ready {0};
std::atomic<uint64_t> not_due {0};
std::atomic<uint64_t> submit_cpu {0};
std::atomic<uint64_t> submit_gpu {0};
std::atomic<uint64_t> register_buffers {0};
std::atomic<uint64_t> open {0};

static void Log(const char* reason) {
	LOGF("FlipTrace: stats reason=%s reserve_cpu=%" PRIu64 " reserve_gpu=%" PRIu64
	     " complete=%" PRIu64 " presented=%" PRIu64 " queue_empty=%" PRIu64 " not_ready=%" PRIu64
	     " not_due=%" PRIu64 " submit_cpu=%" PRIu64 " submit_gpu=%" PRIu64
	     " register_buffers=%" PRIu64 " open=%" PRIu64 "\n",
	     reason, reserve_cpu.load(std::memory_order_relaxed),
	     reserve_gpu.load(std::memory_order_relaxed), complete.load(std::memory_order_relaxed),
	     presented.load(std::memory_order_relaxed), queue_empty.load(std::memory_order_relaxed),
	     not_ready.load(std::memory_order_relaxed), not_due.load(std::memory_order_relaxed),
	     submit_cpu.load(std::memory_order_relaxed), submit_gpu.load(std::memory_order_relaxed),
	     register_buffers.load(std::memory_order_relaxed), open.load(std::memory_order_relaxed));
}

static void LogRateLimited(const char* reason, std::atomic<uint64_t>& counter, uint64_t every) {
	const auto n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 8 || (every != 0 && (n % every) == 0)) {
		Log(reason);
	}
}

static void LogPeriodic() {
	using clock = std::chrono::steady_clock;
	static std::atomic<int64_t> last_ms {0};
	const auto                  now =
	    std::chrono::duration_cast<std::chrono::milliseconds>(clock::now().time_since_epoch())
	        .count();
	auto prev = last_ms.load(std::memory_order_relaxed);
	if (prev != 0 && (now - prev) < 30000) {
		return;
	}
	if (!last_ms.compare_exchange_strong(prev, now, std::memory_order_relaxed)) {
		return;
	}
	Log("periodic_30s");
	fprintf(stderr,
	        "FlipTrace: periodic_30s reserve_cpu=%" PRIu64 " reserve_gpu=%" PRIu64
	        " complete=%" PRIu64 " presented=%" PRIu64 " submit_cpu=%" PRIu64
	        " submit_gpu=%" PRIu64 "\n",
	        reserve_cpu.load(std::memory_order_relaxed),
	        reserve_gpu.load(std::memory_order_relaxed), complete.load(std::memory_order_relaxed),
	        presented.load(std::memory_order_relaxed), submit_cpu.load(std::memory_order_relaxed),
	        submit_gpu.load(std::memory_order_relaxed));
}
} // namespace FlipStats

class FlipQueue {
public:
	FlipQueue() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	~FlipQueue() { KYTY_NOT_IMPLEMENTED; }
	KYTY_CLASS_NO_COPY(FlipQueue);

	bool     Reserve(VideoOutConfig* cfg, int index, int64_t flip_arg, FlipRequestSource source,
	                 uint64_t* request_id);
	void     Prepare(uint64_t request_id, Graphics::CommandBuffer* buffer);
	uint64_t PrepareNextCpu(Graphics::CommandBuffer* buffer);
	void     Complete(uint64_t request_id);
	void     WaitForSubmitSlot();
	bool     Flip(uint32_t micros);
	bool     HasPending(VideoOutConfig* cfg, int start_index, int count);
	void     GetFlipStatus(VideoOutConfig* cfg, VideoOutFlipStatus* out);
	void     Wait(VideoOutConfig* cfg, int index);

private:
	enum class RequestState { Reserved, Recording, Ready, Presenting };

	struct Request {
		uint64_t                 id;
		VideoOutConfig*          cfg;
		int                      index;
		int64_t                  flip_arg;
		uint64_t                 submit_ptc;
		FlipRequestSource        source;
		RequestState             state;
		Graphics::PreparedFrame* frame;
	};

	Common::Mutex      m_mutex;
	Common::CondVar    m_submit_cond_var;
	Common::CondVar    m_submit_slot_cond_var;
	Common::CondVar    m_done_cond_var;
	std::list<Request> m_requests;
	std::list<Request> m_cpu_requests;
	bool               m_processing      = false;
	uint64_t           m_next_request_id = 1;
};

class VideoOutContext {
public:
	static constexpr int VIDEO_OUT_NUM_MAX = 2;

	VideoOutContext() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	~VideoOutContext() { KYTY_NOT_IMPLEMENTED; }

	KYTY_CLASS_NO_COPY(VideoOutContext);

	int             Open();
	void            Close(int handle);
	VideoOutConfig* Get(int handle);
	bool            IsOpened(int handle);

	Presentation::DisplayBufferImage FindImage(const void* buffer, bool render_target);

	void Init(uint32_t width, uint32_t height);

	Graphics::GraphicContext* GetGraphicCtx() {
		Common::LockGuard lock(m_mutex);

		if (m_graphic_ctx == nullptr) {
			m_graphic_ctx = Graphics::WindowGetGraphicContext();
		}

		return m_graphic_ctx;
	}

	FlipQueue& GetFlipQueue() { return m_flip_queue; }

	void VblankBegin();
	void VblankEnd();

private:
	Common::Mutex             m_mutex;
	VideoOutConfig            m_video_out_ctx[VIDEO_OUT_NUM_MAX];
	Graphics::GraphicContext* m_graphic_ctx = nullptr;
	FlipQueue                 m_flip_queue;
};

static VideoOutContext* g_video_out_context = nullptr;

using VideoOutEventQueues = std::vector<EventQueue::KernelEqueue>;

static uintptr_t VideoOutEventId(VideoOutEventKind kind) {
	return static_cast<uintptr_t>(kind);
}

static VideoOutEventKind GetVideoOutEventKind(uintptr_t event_id) {
	switch (event_id) {
		case VIDEO_OUT_EVENT_FLIP: return VideoOutEventKind::Flip;
		case VIDEO_OUT_EVENT_VBLANK: return VideoOutEventKind::Vblank;
		case VIDEO_OUT_EVENT_PRE_VBLANK_START: return VideoOutEventKind::PreVblankStart;
		case VIDEO_OUT_EVENT_SET_MODE: return VideoOutEventKind::OutputMode;
		default: EXIT("unsupported video-out event id=%" PRIuPTR "\n", event_id);
	}
	return VideoOutEventKind::Flip;
}

static VideoOutEventQueues& VideoOutEventQueuesFor(VideoOutConfig&   video_out,
                                                   VideoOutEventKind kind) {
	switch (kind) {
		case VideoOutEventKind::Flip: return video_out.flip_eqs;
		case VideoOutEventKind::Vblank: return video_out.vblank_eqs;
		case VideoOutEventKind::PreVblankStart: return video_out.pre_vblank_eqs;
		case VideoOutEventKind::OutputMode: return video_out.output_mode_eqs;
	}
	EXIT("unsupported video-out event kind\n");
	return video_out.flip_eqs;
}

static intptr_t MakeVideoOutEventData(intptr_t current_data, void* trigger_data) {
	const uint64_t old_data = static_cast<uint64_t>(current_data);
	uint64_t       counter  = (old_data >> 12u) & 0xfu;
	if (counter != 0xfu) {
		counter++;
	}

	const uint64_t time    = LibKernel::KernelReadTsc() & 0xfffu;
	const uint64_t payload = static_cast<uint64_t>(reinterpret_cast<intptr_t>(trigger_data));

	return static_cast<intptr_t>(time | (counter << 12u) |
	                             ((payload & 0x0000ffffffffffffULL) << 16u));
}

static void ResetVideoOutEvent(EventQueue::KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	event->triggered    = false;
	event->event.fflags = 0;
	event->event.data   = 0;
}

static void TriggerVideoOutEvent(EventQueue::KernelEqueueEvent* event, void* trigger_data) {
	EXIT_IF(event == nullptr);

	auto triggered_event = event->event;
	triggered_event.fflags =
	    triggered_event.fflags < 0xfu ? triggered_event.fflags + 1u : triggered_event.fflags;
	triggered_event.data = MakeVideoOutEventData(triggered_event.data, trigger_data);
	if (event->triggered) {
		event->pending_events.push_back(triggered_event);
		return;
	}
	event->event     = triggered_event;
	event->triggered = true;
}

static void RemoveVideoOutEventQueue(EventQueue::KernelEqueue       eq,
                                     EventQueue::KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	EXIT_IF(event->filter.data == nullptr);
	EXIT_NOT_IMPLEMENTED(event->event.filter != EventQueue::KERNEL_EVFILT_VIDEO_OUT);

	auto* video_out = static_cast<VideoOutConfig*>(event->filter.data);
	auto& queues    = VideoOutEventQueuesFor(*video_out, GetVideoOutEventKind(event->event.ident));
	Common::LockGuard lock(video_out->mutex);
	EXIT_IF(queues.empty());
	const auto entry = std::find(queues.begin(), queues.end(), eq);
	EXIT_NOT_IMPLEMENTED(entry == queues.end());
	*entry = nullptr;
}

static void TriggerVideoOutEventsLocked(const VideoOutEventQueues& queues, VideoOutEventKind kind,
                                        void* trigger_data) {
	for (auto eq: queues) {
		if (eq == nullptr) {
			continue;
		}
		const auto result = EventQueue::KernelTriggerEvent(
		    eq, VideoOutEventId(kind), EventQueue::KERNEL_EVFILT_VIDEO_OUT, trigger_data);
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

static void DeleteVideoOutEventsLocked(VideoOutEventQueues& queues, VideoOutEventKind kind) {
	for (auto eq: queues) {
		if (eq == nullptr) {
			continue;
		}
		const auto result = EventQueue::KernelDeleteEvent(eq, VideoOutEventId(kind),
		                                                  EventQueue::KERNEL_EVFILT_VIDEO_OUT);
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
	queues.clear();
}

static int RegisterVideoOutEvent(int handle, EventQueue::KernelEqueue eq, VideoOutEventKind kind,
                                 void* udata) {
	EXIT_IF(g_video_out_context == nullptr);

	auto* video_out = g_video_out_context->Get(handle);
	if (video_out == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	Common::LockGuard lock(video_out->mutex);
	if (kind == VideoOutEventKind::OutputMode) {
		LOGF("\t eq     = 0x%016" PRIx64 "\n"
		     "\t handle = %d\n"
		     "\t udata  = 0x%016" PRIx64 "\n",
		     reinterpret_cast<uint64_t>(eq), handle, reinterpret_cast<uint64_t>(udata));
	}
	if (eq == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
	}
	auto&       queues              = VideoOutEventQueuesFor(*video_out, kind);
	const bool  add_queue           = std::find(queues.begin(), queues.end(), eq) == queues.end();
	const bool  initially_triggered = kind == VideoOutEventKind::OutputMode;
	void* const initial_trigger_data =
	    initially_triggered ? reinterpret_cast<void*>(video_out->output_mode) : nullptr;

	EventQueue::KernelEqueueEvent event {};
	event.triggered    = initially_triggered;
	event.event.ident  = VideoOutEventId(kind);
	event.event.filter = EventQueue::KERNEL_EVFILT_VIDEO_OUT;
	event.event.udata  = udata;
	event.event.fflags = initially_triggered ? 1u : 0u;
	event.event.data   = initially_triggered ? MakeVideoOutEventData(0, initial_trigger_data) : 0;
	event.filter.delete_event_func = RemoveVideoOutEventQueue;
	event.filter.reset_func        = ResetVideoOutEvent;
	event.filter.trigger_func      = TriggerVideoOutEvent;
	event.filter.data              = video_out;

	const int result = EventQueue::KernelAddEvent(eq, event);
	if (result == OK && add_queue) {
		queues.push_back(eq);
	}
	return result;
}

static int DeleteVideoOutEvent(int handle, EventQueue::KernelEqueue eq, VideoOutEventKind kind) {
	EXIT_IF(g_video_out_context == nullptr);

	if (g_video_out_context->Get(handle) == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (eq == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
	}
	return EventQueue::KernelDeleteEvent(eq, VideoOutEventId(kind),
	                                     EventQueue::KERNEL_EVFILT_VIDEO_OUT);
}

static void WaitForNextVblank() {
	static uint64_t next_vblank_ticks = 0;

	const uint64_t frequency = Common::Timer::QueryPerformanceFrequency();
	const uint64_t period    = frequency / Config::GetVblankFrequency();
	uint64_t       now       = Common::Timer::QueryPerformanceCounter();

	if (next_vblank_ticks == 0) {
		next_vblank_ticks = now;
	}

	if (now < next_vblank_ticks) {
		const uint64_t wait_ticks  = next_vblank_ticks - now;
		const uint64_t wait_micros = (wait_ticks * 1000000u) / frequency;

		if (wait_micros > 0) {
			Common::Thread::SleepMicro(
			    static_cast<uint32_t>(std::min<uint64_t>(wait_micros, UINT32_MAX)));
		}

		now = Common::Timer::QueryPerformanceCounter();
	}

	do {
		next_vblank_ticks += period;
	} while (next_vblank_ticks <= now);
}

static bool IsFlipDue(VideoOutConfig* cfg) {
	EXIT_IF(cfg == nullptr);

	Common::LockGuard lock(cfg->mutex);

	const int interval = cfg->flip_rate + 1;

	return interval <= 1 || (cfg->vblank_status.count % static_cast<uint64_t>(interval)) == 0;
}

static bool IsValidBufferIndex(int index) {
	return index >= VIDEO_OUT_BUFFER_INDEX_BLACK && index < VIDEO_OUT_BUFFER_NUM_MAX;
}

static bool IsSpecialBufferIndex(int index) {
	return index == VIDEO_OUT_BUFFER_INDEX_BLANK || index == VIDEO_OUT_BUFFER_INDEX_BLACK;
}

static bool IsValidFlipMode(int mode) {
	return mode >= VIDEO_OUT_FLIP_MODE_VSYNC && mode <= VIDEO_OUT_FLIP_MODE_VSYNC_MULTI;
}

static int ReserveFlipRequest(int handle, int index, int flip_mode, int64_t flip_arg,
                              FlipRequestSource source, uint64_t* request_id) {
	EXIT_IF(g_video_out_context == nullptr || request_id == nullptr);

	auto* video_out = g_video_out_context->Get(handle);
	if (video_out == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (!IsValidFlipMode(flip_mode)) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}
	if (!IsValidBufferIndex(index)) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}

	Common::LockGuard lock(video_out->mutex);
	if (video_out->closing ||
	    (!IsSpecialBufferIndex(index) &&
	     (video_out->unregistering[index] || video_out->buffers[index].buffer_vulkan == nullptr))) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}
	if (!g_video_out_context->GetFlipQueue().Reserve(video_out, index, flip_arg, source,
	                                                 request_id)) {
		return VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
	}
	return OK;
}

static Graphics::VideoOutInfo MakeVideoOutInfo(const VideoOutBufferAttribute2& attribute,
                                               uint64_t address, uint64_t metadata_address,
                                               Graphics::VideoOutCompression compression) {
	if (attribute.reserved0 != 0 || attribute.aspect_ratio != 0 || attribute.width == 0 ||
	    attribute.height == 0 || attribute.width > 16384 || attribute.height > 16384 ||
	    attribute.pitch_in_pixel != 0 ||
	    (attribute.option != 0 &&
	     attribute.option != VIDEO_OUT_BUFFER_ATTRIBUTE_OPTION_STRICT_COLORIMETRY) ||
	    attribute.tiling_mode != 0 || attribute.pad0 != 0 || attribute.reserved1[0] != 0 ||
	    attribute.reserved1[1] != 0 || attribute.reserved1[2] != 0 || address == 0 ||
	    compression == Graphics::VideoOutCompression::Unsupported) {
		EXIT("unsupported or invalid video-out surface attributes\n");
	}
	Graphics::VideoOutPixelFormatInfo pixel_format {};
	if (!Graphics::DecodeVideoOutPixelFormat(attribute.pixel_format, &pixel_format)) {
		EXIT("unsupported video-out pixel format: 0x%016" PRIx64 "\n", attribute.pixel_format);
	}
	const auto tile_mode =
	    Graphics::Prospero::GpuEnumValue(Graphics::Prospero::TileMode::kRenderTarget);
	const auto pitch =
	    Graphics::TileGetTexturePitch(pixel_format.guest_format, attribute.width, 1, tile_mode);
	Graphics::TileSizeAlign total {};
	Graphics::TileGetTextureTotalSize(pixel_format.guest_format, attribute.width, attribute.height,
	                                  1, pitch, 1, tile_mode, false, &total);
	if (total.size == 0 || total.align != 65536 || (address & (total.align - 1u)) != 0) {
		EXIT("invalid video-out surface footprint or alignment\n");
	}
	Graphics::VideoOutInfo info {};
	info.address           = address;
	info.size              = total.size;
	info.metadata_address  = metadata_address;
	info.format            = pixel_format.format;
	info.guest_format      = pixel_format.guest_format;
	info.width             = attribute.width;
	info.height            = attribute.height;
	info.pitch             = pitch;
	info.bytes_per_element = pixel_format.bytes_per_element;
	info.tile_mode         = tile_mode;
	info.dcc_control       = attribute.dcc_control;
	info.compression       = compression;
	info.bgra16            = pixel_format.bgra16;
	return info;
}

void VideoOutInit(uint32_t width, uint32_t height) {
	EXIT_IF(g_video_out_context != nullptr);

	g_video_out_context = new VideoOutContext;

	g_video_out_context->Init(width, height);
}

void VideoOutContext::Init(uint32_t width, uint32_t height) {
	for (auto& ctx: m_video_out_ctx) {
		ctx.width  = width;
		ctx.height = height;
	}
}

int VideoOutContext::Open() {
	Common::LockGuard lock(m_mutex);

	int handle = -1;

	for (int i = 1; i < VIDEO_OUT_NUM_MAX; i++) {
		if (!m_video_out_ctx[i].opened) {
			handle = i;
			break;
		}
	}

	if (handle < 0) {
		return -1;
	}
	auto&             config = m_video_out_ctx[handle];
	Common::LockGuard config_lock(config.mutex);

	EXIT_IF(!config.flip_eqs.empty());
	EXIT_IF(!config.pre_vblank_eqs.empty());
	EXIT_IF(!config.vblank_eqs.empty());
	EXIT_IF(!config.output_mode_eqs.empty());
	EXIT_IF(config.flip_rate != 0);
	EXIT_IF(!config.buffers_sets.empty());
	for (const auto& buffer: config.buffers) {
		EXIT_IF(buffer.buffer != nullptr || buffer.buffer_vulkan != nullptr ||
		        buffer.buffer_size != 0 || buffer.buffer_pitch != 0);
	}
	for (bool unregistering: config.unregistering) {
		EXIT_IF(unregistering);
	}

	config.closing                   = false;
	config.opened                    = true;
	config.output_mode               = VIDEO_OUT_OUTPUT_MODE_DEFAULT;
	config.prev_flip_index           = -1;
	config.flip_status               = VideoOutFlipStatus();
	config.flip_status.flipArg       = -1;
	config.flip_status.currentBuffer = -1;
	config.flip_status.count         = 0;
	config.pre_vblank_status         = VideoOutVblankStatus();
	config.vblank_status             = VideoOutVblankStatus();
	std::memset(config.buffer_labels, 0, sizeof(config.buffer_labels));

	return handle;
}

void VideoOutContext::Close(int handle) {
	Common::LockGuard lock(m_mutex);

	EXIT_NOT_IMPLEMENTED(handle >= VIDEO_OUT_NUM_MAX);
	EXIT_NOT_IMPLEMENTED(!m_video_out_ctx[handle].opened);

	auto& config  = m_video_out_ctx[handle];
	config.opened = false;

	config.mutex.Lock();
	if (config.closing) {
		EXIT("video-out handle is already closing\n");
	}
	config.closing = true;
	if (m_flip_queue.HasPending(&config, VIDEO_OUT_BUFFER_INDEX_BLACK,
	                            VIDEO_OUT_BUFFER_NUM_MAX - VIDEO_OUT_BUFFER_INDEX_BLACK)) {
		EXIT("cannot close video-out handle with pending flips\n");
	}
	DeleteVideoOutEventsLocked(config.flip_eqs, VideoOutEventKind::Flip);
	DeleteVideoOutEventsLocked(config.pre_vblank_eqs, VideoOutEventKind::PreVblankStart);
	DeleteVideoOutEventsLocked(config.vblank_eqs, VideoOutEventKind::Vblank);
	DeleteVideoOutEventsLocked(config.output_mode_eqs, VideoOutEventKind::OutputMode);

	config.flip_rate = 0;

	std::vector<Graphics::VideoOutVulkanImage*> images;
	for (const auto& buffer: config.buffers) {
		if ((buffer.buffer == nullptr) != (buffer.buffer_vulkan == nullptr) ||
		    (buffer.buffer_vulkan != nullptr &&
		     (buffer.buffer_size == 0 || buffer.buffer_pitch == 0))) {
			EXIT("inconsistent registered video-out buffer state\n");
		}
		if (buffer.buffer_vulkan != nullptr) {
			images.push_back(buffer.buffer_vulkan);
		}
	}
	if (!images.empty()) {
		if (Graphics::g_render_ctx == nullptr) {
			EXIT("cannot unregister video-out surfaces without a render context\n");
		}
		Graphics::g_render_ctx->GetTextureCache()->UnregisterVideoOutSurfaces(images);
	}
	for (auto& buffer: config.buffers) {
		buffer = VideoOutBufferInfo {};
	}

	config.buffers_sets.clear();
	for (bool unregistering: config.unregistering) {
		if (unregistering) {
			EXIT("video-out close raced with buffer unregistration\n");
		}
	}
	config.mutex.Unlock();
}

VideoOutConfig* VideoOutContext::Get(int handle) {
	Common::LockGuard lock(m_mutex);
	if (handle <= 0 || handle >= VIDEO_OUT_NUM_MAX || !m_video_out_ctx[handle].opened) {
		return nullptr;
	}

	return m_video_out_ctx + handle;
}

bool VideoOutContext::IsOpened(int handle) {
	Common::LockGuard lock(m_mutex);

	return handle > 0 && handle < VIDEO_OUT_NUM_MAX && m_video_out_ctx[handle].opened;
}

void VideoOutContext::VblankBegin() {
	Common::LockGuard lock(m_mutex);

	for (int i = 1; i < VIDEO_OUT_NUM_MAX; i++) {
		auto& ctx = m_video_out_ctx[i];
		if (ctx.opened) {
			ctx.mutex.Lock();
			ctx.pre_vblank_status.count++;
			ctx.pre_vblank_status.processTime        = LibKernel::KernelGetProcessTime();
			ctx.pre_vblank_status.reserved           = LibKernel::KernelReadTsc();
			ctx.pre_vblank_status.processTimeCounter = LibKernel::KernelGetProcessTimeCounter();

			TriggerVideoOutEventsLocked(ctx.pre_vblank_eqs, VideoOutEventKind::PreVblankStart,
			                            reinterpret_cast<void*>(ctx.pre_vblank_status.count));
			ctx.mutex.Unlock();
		}
	}
}

void VideoOutContext::VblankEnd() {
	Common::LockGuard lock(m_mutex);

	for (int i = 1; i < VIDEO_OUT_NUM_MAX; i++) {
		auto& ctx = m_video_out_ctx[i];
		if (ctx.opened) {
			ctx.mutex.Lock();
			ctx.vblank_status.count++;
			ctx.vblank_status.processTime        = LibKernel::KernelGetProcessTime();
			ctx.vblank_status.reserved           = LibKernel::KernelReadTsc();
			ctx.vblank_status.processTimeCounter = LibKernel::KernelGetProcessTimeCounter();

			TriggerVideoOutEventsLocked(ctx.vblank_eqs, VideoOutEventKind::Vblank,
			                            reinterpret_cast<void*>(ctx.vblank_status.count));
			ctx.mutex.Unlock();
		}
	}
}

Presentation::DisplayBufferImage VideoOutContext::FindImage(const void* buffer,
                                                            bool        render_target) {
	Presentation::DisplayBufferImage ret;
	Common::LockGuard                lock(m_mutex);
	for (auto& ctx: m_video_out_ctx) {
		if (!ctx.opened) {
			continue;
		}
		Common::LockGuard config_lock(ctx.mutex);
		if (ctx.closing) {
			EXIT("display-buffer lookup raced with video-out close\n");
		}
		for (const auto& set: ctx.buffers_sets) {
			for (int j = set.start_index; j < set.start_index + set.num; j++) {
				if (ctx.buffers[j].buffer == buffer) {
					if (ctx.unregistering[j] || ctx.buffers[j].buffer_vulkan == nullptr) {
						EXIT("display-buffer lookup found an unavailable video-out surface\n");
					}
					ret.image = ctx.buffers[j].buffer_vulkan;
					ret.size  = ctx.buffers[j].buffer_size;
					ret.pitch = ctx.buffers[j].buffer_pitch;
					ret.index = j - set.start_index;
					Graphics::g_render_ctx->GetTextureCache()->RefreshVideoOut(ret.image,
					                                                           render_target);
					return ret;
				}
			}
		}
	}
	return ret;
}

bool FlipQueue::Reserve(VideoOutConfig* cfg, int index, int64_t flip_arg, FlipRequestSource source,
                        uint64_t* request_id) {
	EXIT_IF(cfg == nullptr || request_id == nullptr);
	Common::LockGuard lock(m_mutex);

	if (m_requests.size() + m_cpu_requests.size() >= VIDEO_OUT_FLIP_QUEUE_CAPACITY) {
		return false;
	}
	auto& pending = source == FlipRequestSource::GpuEop ? m_requests : m_cpu_requests;

	Request r {};
	r.id         = m_next_request_id++;
	r.cfg        = cfg;
	r.index      = index;
	r.flip_arg   = flip_arg;
	r.submit_ptc = LibKernel::KernelGetProcessTimeCounter();
	r.source     = source;
	r.state      = RequestState::Reserved;

	pending.push_back(r);
	*request_id = r.id;

	cfg->flip_status.flipPendingNum = static_cast<int>(m_requests.size() + m_cpu_requests.size());
	cfg->flip_status.submitProcessTimeCounter = r.submit_ptc;
	if (source == FlipRequestSource::GpuEop) {
		cfg->flip_status.gcQueueNum++;
		FlipStats::reserve_gpu.fetch_add(1, std::memory_order_relaxed);
	} else {
		FlipStats::reserve_cpu.fetch_add(1, std::memory_order_relaxed);
	}
	LOGF("FlipTrace: Reserve id=%" PRIu64 " index=%d gpu_eop=%d pending=%d cpu_q=%zu gpu_q=%zu\n",
	     r.id, index, gpu_eop ? 1 : 0, cfg->flip_status.flipPendingNum, m_cpu_requests.size(),
	     m_requests.size());
	FlipStats::Log(gpu_eop ? "reserve_gpu" : "reserve_cpu");

	return true;
}

void FlipQueue::Prepare(uint64_t request_id, Graphics::CommandBuffer* buffer) {
	EXIT_IF(buffer == nullptr);

	VideoOutConfig* cfg   = nullptr;
	int             index = 0;
	{
		Common::LockGuard lock(m_mutex);
		auto request = std::find_if(m_requests.begin(), m_requests.end(),
		                            [request_id](const auto& r) { return r.id == request_id; });
		if (request == m_requests.end()) {
			auto pending = std::find_if(m_cpu_requests.begin(), m_cpu_requests.end(),
			                            [request_id](const auto& r) { return r.id == request_id; });
			if (pending == m_cpu_requests.end()) {
				EXIT("cannot prepare video-out request id=%" PRIu64 "\n", request_id);
			}
			request = m_requests.insert(m_requests.end(), *pending);
			m_cpu_requests.erase(pending);
		}
		if (request->state != RequestState::Reserved) {
			EXIT("cannot prepare video-out request id=%" PRIu64 "\n", request_id);
		}
		request->state = RequestState::Recording;
		cfg            = request->cfg;
		index          = request->index;
	}

	const bool                     special = IsSpecialBufferIndex(index);
	Graphics::VideoOutVulkanImage* source  = nullptr;
	uint32_t                       width   = 0;
	uint32_t                       height  = 0;
	{
		Common::LockGuard lock(cfg->mutex);
		if (cfg->closing) {
			EXIT("cannot prepare flip for a closing video-out, id=%" PRIu64 "\n", request_id);
		}
		if (special) {
			width  = cfg->width;
			height = cfg->height;
		} else {
			if (cfg->unregistering[index]) {
				EXIT("cannot prepare flip from an unavailable surface, id=%" PRIu64 " index=%d\n",
				     request_id, index);
			}
			source = cfg->buffers[index].buffer_vulkan;
			if (source == nullptr) {
				EXIT("cannot prepare flip without a native surface, id=%" PRIu64 " index=%d\n",
				     request_id, index);
			}
		}
	}
	// Blank and black must be opaque so the SDL surface shows a solid black frame.
	auto* frame = special ? Graphics::WindowPrepareBlankFrame(buffer, width, height, true)
	                      : Graphics::WindowPrepareFrame(buffer, source);

	Common::LockGuard lock(m_mutex);
	auto request = std::find_if(m_requests.begin(), m_requests.end(),
	                            [request_id](const auto& r) { return r.id == request_id; });
	if (request == m_requests.end() || request->state != RequestState::Recording ||
	    request->frame != nullptr) {
		EXIT("video-out request changed while recording, id=%" PRIu64 "\n", request_id);
	}
	request->frame = frame;
	LOGF("FlipTrace: Prepare id=%" PRIu64 " index=%d state=Recording frame=%p\n", request_id, index,
	     static_cast<void*>(frame));
}

uint64_t FlipQueue::PrepareNextCpu(Graphics::CommandBuffer* buffer) {
	uint64_t request_id = 0;
	{
		Common::LockGuard lock(m_mutex);
		if (m_cpu_requests.empty()) {
			EXIT("CPU flip preparation has no accepted request\n");
		}
		request_id = m_cpu_requests.front().id;
		LOGF("FlipTrace: PrepareNextCpu id=%" PRIu64 " cpu_q=%zu gpu_q=%zu\n", request_id,
		     m_cpu_requests.size(), m_requests.size());
	}
	Prepare(request_id, buffer);
	return request_id;
}

void FlipQueue::Complete(uint64_t request_id) {
	Common::LockGuard lock(m_mutex);
	auto request = std::find_if(m_requests.begin(), m_requests.end(),
	                            [request_id](const auto& r) { return r.id == request_id; });
	if (request == m_requests.end() || request->state != RequestState::Recording ||
	    request->frame == nullptr) {
		EXIT("completed GPU flip has no prepared recording, id=%" PRIu64 "\n", request_id);
	}
	request->state = RequestState::Ready;
	m_submit_cond_var.Signal();
	FlipStats::complete.fetch_add(1, std::memory_order_relaxed);
	LOGF("FlipTrace: Complete id=%" PRIu64 " index=%d state=Ready\n", request_id, request->index);
	FlipStats::Log("complete");
}

void FlipQueue::WaitForSubmitSlot() {
	Common::LockGuard lock(m_mutex);
	while (m_requests.size() + m_cpu_requests.size() >= VIDEO_OUT_FLIP_QUEUE_CAPACITY) {
		if (m_requests.empty()) {
			EXIT("video-out queue is saturated by CPU flips queued behind the current EOP\n");
		}
		m_submit_slot_cond_var.Wait(&m_mutex);
	}
}

void FlipQueue::Wait(VideoOutConfig* cfg, int index) {
	Common::LockGuard lock(m_mutex);

	auto has_request = [this, cfg, index] {
		auto matches = [cfg, index](const auto& r) { return r.cfg == cfg && r.index == index; };
		return std::any_of(m_requests.begin(), m_requests.end(), matches) ||
		       std::any_of(m_cpu_requests.begin(), m_cpu_requests.end(), matches);
	};
	while (has_request()) {
		m_done_cond_var.Wait(&m_mutex);
	}
}

bool FlipQueue::HasPending(VideoOutConfig* cfg, int start_index, int count) {
	if (cfg == nullptr || count <= 0 || start_index > INT_MAX - count) {
		EXIT("invalid video-out pending-flip query range\n");
	}
	Common::LockGuard lock(m_mutex);
	auto              matches = [&](const auto& request) {
		return request.cfg == cfg && request.index >= start_index &&
		       request.index < start_index + count;
	};
	return std::any_of(m_requests.begin(), m_requests.end(), matches) ||
	       std::any_of(m_cpu_requests.begin(), m_cpu_requests.end(), matches);
}

bool FlipQueue::Flip(uint32_t micros) {
	KYTY_PROFILER_BLOCK("FlipQueue::Flip");

	m_mutex.Lock();
	if (m_requests.empty()) {
		m_submit_cond_var.WaitFor(&m_mutex, micros);

		if (m_requests.empty()) {
			const auto cpu_q   = m_cpu_requests.size();
			const int  pending =
			    m_cpu_requests.empty() && m_requests.empty()
			        ? 0
			        : static_cast<int>(m_cpu_requests.size() + m_requests.size());
			int front_state = -1;
			uint64_t front_id = 0;
			int front_index = -1;
			if (!m_cpu_requests.empty()) {
				front_state = static_cast<int>(m_cpu_requests.front().state);
				front_id    = m_cpu_requests.front().id;
				front_index = m_cpu_requests.front().index;
			}
			m_mutex.Unlock();
			FlipStats::LogRateLimited("queue_empty", FlipStats::queue_empty, 256);
			FlipStats::LogPeriodic();
			AfterPending0HostTick();
			if (cpu_q != 0) {
				static std::atomic<uint32_t> stuck_logs {0};
				if (stuck_logs.fetch_add(1, std::memory_order_relaxed) < 32) {
					LOGF("FlipTrace: Flip skipped reason=queue_empty stuck_cpu_q=%zu "
					     "pending_est=%d front_id=%" PRIu64 " front_index=%d front_state=%d\n",
					     cpu_q, pending, front_id, front_index, front_state);
				}
			}
			return false;
		}
	}
	if (m_processing) {
		EXIT("video-out flip queue processing is already active\n");
	}
	if (m_requests.front().state != RequestState::Ready) {
		const auto state = m_requests.front().state;
		const auto id      = m_requests.front().id;
		const auto index   = m_requests.front().index;
		m_mutex.Unlock();
		FlipStats::not_ready.fetch_add(1, std::memory_order_relaxed);
		LOGF("FlipTrace: Flip skipped id=%" PRIu64 " index=%d reason=not_ready state=%d\n", id,
		     index, static_cast<int>(state));
		FlipStats::Log("not_ready");
		return false;
	}
	m_processing = true;
	auto r       = m_requests.front();
	m_mutex.Unlock();
	if (!IsFlipDue(r.cfg)) {
		Common::LockGuard lock(m_mutex);
		m_processing = false;
		FlipStats::not_due.fetch_add(1, std::memory_order_relaxed);
		if (Graphics::boot_trace_log()) {
			LOGF("FlipTrace: Flip skipped id=%" PRIu64 " index=%d reason=not_due flip_rate=%d\n",
			     r.id, r.index, r.cfg->flip_rate);
			FlipStats::Log("not_due");
		}
		return false;
	}

	m_mutex.Lock();
	if (m_requests.empty() || m_requests.front().id != r.id ||
	    m_requests.front().state != RequestState::Ready || !m_processing) {
		EXIT("video-out request changed before presentation, id=%" PRIu64 "\n", r.id);
	}
	m_requests.front().state = RequestState::Presenting;
	m_mutex.Unlock();

	Graphics::WindowPresentFrame(r.frame);

	m_mutex.Lock();
	if (m_requests.empty() || m_requests.front().id != r.id ||
	    m_requests.front().state != RequestState::Presenting) {
		EXIT("video-out flip queue changed while processing its front request\n");
	}
	m_requests.pop_front();

	r.cfg->flip_status.count++;
	r.cfg->flip_status.processTime              = LibKernel::KernelGetProcessTime();
	r.cfg->flip_status.processTimeCounter       = LibKernel::KernelGetProcessTimeCounter();
	r.cfg->flip_status.submitProcessTimeCounter = r.submit_ptc;
	r.cfg->flip_status.flipArg                  = r.flip_arg;
	r.cfg->flip_status.currentBuffer            = r.index;
	r.cfg->flip_status.flipPendingNum = static_cast<int>(m_requests.size() + m_cpu_requests.size());
	if (r.source == FlipRequestSource::GpuEop && r.cfg->flip_status.gcQueueNum > 0) {
		r.cfg->flip_status.gcQueueNum--;
	}

	m_processing = false;
	if (!first_blank_hold) {
		m_done_cond_var.SignalAll();
	}
	m_submit_slot_cond_var.Signal();
	m_mutex.Unlock();

	r.cfg->mutex.Lock();
	TriggerVideoOutEventsLocked(r.cfg->flip_eqs, VideoOutEventKind::Flip,
	                            reinterpret_cast<void*>(r.flip_arg));
	r.cfg->mutex.Unlock();

		// Phase A (safe): currentBuffer=0 + sticky pending=1 — bisect proved this does not
		// failfast; MainThread stays in IsFlipPending.
		{
			Common::LockGuard lock(m_mutex);
			if (bisect == 1) {
				r.cfg->flip_status.currentBuffer  = 0;
				r.cfg->flip_status.flipPendingNum = 1;
			} else if (bisect == 2) {
				r.cfg->flip_status.currentBuffer  = VIDEO_OUT_BUFFER_INDEX_BLANK;
				r.cfg->flip_status.flipPendingNum = 0;
			} else {
				r.cfg->flip_status.currentBuffer  = VIDEO_OUT_BUFFER_INDEX_BLANK;
				r.cfg->flip_status.flipPendingNum = 1;
			}
			LOGF("FlipTrace: hold gate phaseA currentBuffer=%d pending=%d bisect=%d\n",
			     r.cfg->flip_status.currentBuffer, r.cfg->flip_status.flipPendingNum, bisect);
			fprintf(stderr, "FlipTrace: hold gate phaseA currentBuffer=%d pending=%d bisect=%d\n",
			        r.cfg->flip_status.currentBuffer, r.cfg->flip_status.flipPendingNum, bisect);
		}
		Common::NoteHleCall("VideoOut", "VideoOut", "HoldGatePhaseA");
		Common::FlushHleRingToFatal("post_hold_gate_phaseA");

		// Modes 1/2 are sticky/partial diagnostics.
		// Mode 3: phaseA + Flip EQ + wake, keep pending sticky (no phaseC) — isolates EQ vs clear.
		// Mode 4: phaseA + EQ, no wake, then phaseC clear.
		// Mode 5/default: full sequence phaseA → EQ → wake → phaseC.
		if (bisect == 1 || bisect == 2) {
			LOGF("FlipTrace: bisect sticky/partial mode=%d (no phaseB/C)\n", bisect);
			fprintf(stderr, "FlipTrace: bisect sticky/partial mode=%d\n", bisect);
		} else {
			// TLOU creates "FLIP QUEUE" then only AddOutputModeEvent — never AddFlipEvent.
			// Steal those equeues into flip_eqs so blank Flip notifications can wake FlipHandler.
			{
				Common::LockGuard lock(r.cfg->mutex);
				if (r.cfg->flip_eqs.empty()) {
					for (auto* eq: r.cfg->output_mode_eqs) {
						if (eq == nullptr) {
							continue;
						}
						if (std::find(r.cfg->flip_eqs.begin(), r.cfg->flip_eqs.end(), eq) ==
						    r.cfg->flip_eqs.end()) {
							EventQueue::KernelEqueueEvent event;
							event.triggered                = false;
							event.event.ident              = VIDEO_OUT_EVENT_FLIP;
							event.event.filter             = EventQueue::KERNEL_EVFILT_VIDEO_OUT;
							event.event.udata              = nullptr;
							event.event.fflags             = 0;
							event.event.data               = 0;
							event.filter.delete_event_func = FlipEventDeleteFunc;
							event.filter.reset_func        = FlipEventResetFunc;
							event.filter.trigger_func      = FlipEventTriggerFunc;
							event.filter.data              = r.cfg;
							const int add_result = EventQueue::KernelAddEvent(eq, event);
							if (add_result == OK) {
								r.cfg->flip_eqs.push_back(eq);
							}
							LOGF("FlipTrace: auto AddFlipEvent on output_mode eq=%p result=%d\n",
							     static_cast<void*>(eq), add_result);
							fprintf(stderr,
							        "FlipTrace: auto AddFlipEvent on output_mode eq result=%d\n",
							        add_result);
						}
					}
				}
			}

			const int64_t flip_eq_arg = (r.flip_arg < 0) ? 0 : r.flip_arg;
			r.cfg->mutex.Lock();
			size_t flip_eq_count = 0;
			for (auto& flip_eq: r.cfg->flip_eqs) {
				if (flip_eq != nullptr) {
					auto result = EventQueue::KernelTriggerEvent(
					    flip_eq, VIDEO_OUT_EVENT_FLIP, EventQueue::KERNEL_EVFILT_VIDEO_OUT,
					    reinterpret_cast<void*>(static_cast<intptr_t>(flip_eq_arg)));
					EXIT_NOT_IMPLEMENTED(result != OK);
					flip_eq_count++;
				}
			}
			r.cfg->mutex.Unlock();
			LOGF("FlipTrace: Flip EQ triggered after hold (phaseB) eqs=%zu arg=%" PRId64 "\n",
			     flip_eq_count, flip_eq_arg);
			fprintf(stderr, "FlipTrace: Flip EQ triggered after hold (phaseB) eqs=%zu\n",
			        flip_eq_count);
			Common::NoteHleCall("VideoOut", "VideoOut", "FlipEqPhaseB");

			if (bisect >= 5 || bisect == 3) {
				const auto woken = LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
				LOGF("SubmitTrace: WakeSubmissionCond after flip woken=%zu\n", woken);
				fprintf(stderr, "SubmitTrace: WakeSubmissionCond after flip woken=%zu\n", woken);
			}

			if (bisect == 3) {
				LOGF("FlipTrace: bisect mode=3 keep pending sticky after EQ (no phaseC)\n");
				fprintf(stderr, "FlipTrace: bisect mode=3 keep pending sticky after EQ\n");
				m_mutex.Lock();
				m_done_cond_var.SignalAll();
				m_mutex.Unlock();
			} else if (bisect == 4) {
				Graphics::WindowHoldVisible(1);
				Common::LockGuard lock(m_mutex);
				r.cfg->flip_status.currentBuffer = VIDEO_OUT_BUFFER_INDEX_BLANK;
				if (r.cfg->flip_status.flipArg < 0) {
					r.cfg->flip_status.flipArg = 0;
				}
				r.cfg->flip_status.flipPendingNum =
				    static_cast<int>(m_requests.size() + m_cpu_requests.size());
				m_done_cond_var.SignalAll();
				LOGF("FlipTrace: hold gate phaseC pending cleared currentBuffer=%d pending=%d "
				     "flipArg=%" PRId64 "\n",
				     r.cfg->flip_status.currentBuffer, r.cfg->flip_status.flipPendingNum,
				     r.cfg->flip_status.flipArg);
				fprintf(stderr,
				        "FlipTrace: hold gate phaseC pending cleared currentBuffer=%d "
				        "pending=%d flipArg=%" PRId64 "\n",
				        r.cfg->flip_status.currentBuffer, r.cfg->flip_status.flipPendingNum,
				        r.cfg->flip_status.flipArg);
				Common::NoteHleCall("VideoOut", "VideoOut", "HoldGatePhaseC");
				Common::FlushHleRingToFatal("post_hold_gate_phaseC");
			} else {
				// Phase 32 default (bisect 5): Flip EQ + wake done — keep pending sticky
				// (Phase 30 baseline). Experimental unlock via KYTY_PHASE32_PENDING0=1.
				// KYTY_PHASE31_INJECT_FLIP0=1 = SubmitFlip(0)+sticky×32 present repro.
				// KYTY_PHASE31_SOFT_ONLY=1 = clear pending without inject (failfast diag).
				Graphics::WindowHoldVisible(1);
				m_mutex.Lock();
				m_done_cond_var.SignalAll();
				m_mutex.Unlock();

				if (Phase31StickyInjectEnabled()) {
					sticky_after_blank.store(32, std::memory_order_release);
					{
						Common::LockGuard lock(m_mutex);
						r.cfg->flip_status.flipPendingNum = 1;
					}
					int inject_handle = 0;
					for (int h = 1; h < 8; ++h) {
						if (g_video_out_context->IsOpened(h) &&
						    g_video_out_context->Get(h) == r.cfg) {
							inject_handle = h;
							break;
						}
					}
					int inject_result = VIDEO_OUT_ERROR_INVALID_HANDLE;
					if (inject_handle > 0) {
						inject_result =
						    VideoOutSubmitFlip(inject_handle, 0, VIDEO_OUT_FLIP_MODE_VSYNC, 0);
					}
					LOGF("FlipTrace: KYTY_PHASE31_INJECT_FLIP0 handle=%d result=%d sticky=%d\n",
					     inject_handle, inject_result,
					     sticky_after_blank.load(std::memory_order_relaxed));
					fprintf(stderr,
					        "FlipTrace: KYTY_PHASE31_INJECT_FLIP0 handle=%d result=%d sticky=%d\n",
					        inject_handle, inject_result,
					        sticky_after_blank.load(std::memory_order_relaxed));
					Common::NoteHleCall("VideoOut", "VideoOut", "HoldGateStickyInject");
					Common::FlushHleRingToFatal("post_hold_gate_sticky_inject");
				} else if (Phase31SoftOnlyEnabled() || Phase32Pending0Enabled()) {
					const bool pending0 = Phase32Pending0Enabled() && !Phase31SoftOnlyEnabled();
					// Phase 33: no host bridge by default. KYTY_PHASE33_BRIDGE=1 restores
					// Phase 32 SubmitFlip0 bridge for present-without-guest-submit diag.
					const char* bridge_env = std::getenv("KYTY_PHASE33_BRIDGE");
					const bool  want_bridge =
					    pending0 && bridge_env != nullptr && bridge_env[0] == '1';
					if (want_bridge) {
						int inject_handle = 0;
						for (int h = 1; h < 8; ++h) {
							if (g_video_out_context->IsOpened(h) &&
							    g_video_out_context->Get(h) == r.cfg) {
								inject_handle = h;
								break;
							}
						}
						int inject_result = VIDEO_OUT_ERROR_INVALID_HANDLE;
						if (inject_handle > 0) {
							inject_result = VideoOutSubmitFlip(inject_handle, 0,
							                                   VIDEO_OUT_FLIP_MODE_VSYNC, 0);
						}
						LOGF("FlipTrace: phase33 BRIDGE SubmitFlip0 handle=%d "
						     "result=%d (grace until Flip EQ consumed)\n",
						     inject_handle, inject_result);
						fprintf(stderr,
						        "FlipTrace: phase33 BRIDGE SubmitFlip0 handle=%d "
						        "result=%d\n",
						        inject_handle, inject_result);
						Common::NoteHleCall("VideoOut", "VideoOut", "HoldGatePending0Bridge");
						Common::FlushHleRingToFatal("post_hold_gate_pending0_bridge");
						{
							Common::LockGuard lock(m_mutex);
							r.cfg->flip_status.flipPendingNum = 1;
							if (inject_result != OK) {
								r.cfg->flip_status.currentBuffer = VIDEO_OUT_BUFFER_INDEX_BLANK;
							}
						}
						ArmPending0Grace(inject_result == OK ? "bridge_ok" : "bridge_fail");
					} else {
						// Phase 36: present BootCards buffer 0 once (guest UploadVideoOut
						// pixels) so the window is not stuck on blank. Not a MenuFlip pump.
						// KYTY_PHASE36_NO_REVEAL=1 disables.
						const char* no_reveal = std::getenv("KYTY_PHASE36_NO_REVEAL");
						const bool  do_reveal =
						    pending0 && (no_reveal == nullptr || no_reveal[0] != '1');
						int inject_handle = 0;
						int inject_result = VIDEO_OUT_ERROR_INVALID_HANDLE;
						if (do_reveal) {
							for (int h = 1; h < 8; ++h) {
								if (g_video_out_context->IsOpened(h) &&
								    g_video_out_context->Get(h) == r.cfg) {
									inject_handle = h;
									break;
								}
							}
							if (inject_handle > 0 &&
							    r.cfg->buffers[0].buffer_vulkan != nullptr) {
								inject_result = VideoOutSubmitFlip(
								    inject_handle, 0, VIDEO_OUT_FLIP_MODE_VSYNC, 0);
							}
							LOGF("FlipTrace: phase36 BootCardsReveal SubmitFlip0 handle=%d "
							     "result=%d\n",
							     inject_handle, inject_result);
							fprintf(stderr,
							        "FlipTrace: phase36 BootCardsReveal SubmitFlip0 "
							        "handle=%d result=%d\n",
							        inject_handle, inject_result);
						}
						Common::LockGuard lock(m_mutex);
						if (inject_result == OK) {
							r.cfg->flip_status.currentBuffer  = 0;
							r.cfg->flip_status.flipPendingNum = 1;
						} else if (r.cfg->flip_status.currentBuffer < 0) {
							r.cfg->flip_status.currentBuffer = 0;
						}
						if (r.cfg->flip_status.flipArg < 0) {
							r.cfg->flip_status.flipArg = 0;
						}
						if (inject_result != OK) {
							r.cfg->flip_status.flipPendingNum = 1;
						}
						ArmPending0Grace(inject_result == OK ? "phase36_reveal"
						                                     : (pending0 ? "phase33_no_bridge"
						                                                  : "soft_only"),
						                 /*require_flip_eq=*/inject_result == OK);
						LOGF("FlipTrace: hold gate phase36 pending0 currentBuffer=%d "
						     "reveal=%d\n",
						     r.cfg->flip_status.currentBuffer, inject_result == OK ? 1 : 0);
						fprintf(stderr,
						        "FlipTrace: hold gate phase36 pending0 reveal=%d\n",
						        inject_result == OK ? 1 : 0);
						Common::NoteHleCall("VideoOut", "VideoOut", "HoldGatePhase36Reveal");
						Common::FlushHleRingToFatal("post_hold_gate_phase36_reveal");
					}
				} else {
					{
						Common::LockGuard lock(m_mutex);
						r.cfg->flip_status.currentBuffer =
						    VIDEO_OUT_BUFFER_INDEX_BLANK;
						r.cfg->flip_status.flipPendingNum = 1;
					}
					LOGF("FlipTrace: hold gate phase32 default pending sticky "
					     "currentBuffer=-1 (set KYTY_PHASE32_PENDING0=1 to clear)\n");
					fprintf(stderr,
					        "FlipTrace: hold gate phase32 default pending sticky\n");
					Common::NoteHleCall("VideoOut", "VideoOut", "HoldGateStickyDefault");
					Common::FlushHleRingToFatal("post_hold_gate_sticky_default");
				}
			}
		}
	} else {
		const int64_t flip_eq_arg = (r.flip_arg < 0) ? 0 : r.flip_arg;
		r.cfg->mutex.Lock();
		for (auto& flip_eq: r.cfg->flip_eqs) {
			if (flip_eq != nullptr) {
				auto result = EventQueue::KernelTriggerEvent(
				    flip_eq, VIDEO_OUT_EVENT_FLIP, EventQueue::KERNEL_EVFILT_VIDEO_OUT,
				    reinterpret_cast<void*>(static_cast<intptr_t>(flip_eq_arg)));
				EXIT_NOT_IMPLEMENTED(result != OK);
			}
		}
		r.cfg->mutex.Unlock();

		if (Config::GraphicsDebugDumpEnabled() &&
		    Config::GetPrintfDirection() != Config::OutputDirection::Silent) {
			LOGF("Flip done: %d\n", r.index);
		}
		FlipStats::presented.fetch_add(1, std::memory_order_relaxed);
		LOGF("FlipTrace: presented index=%d id=%" PRIu64 " flip_count=%" PRIu64
		     " currentBuffer=%d pending=%d\n",
		     r.index, r.id, r.cfg->flip_status.count, r.cfg->flip_status.currentBuffer,
		     r.cfg->flip_status.flipPendingNum);
		FlipStats::Log("presented");
	}

	return true;
}

void FlipQueue::GetFlipStatus(VideoOutConfig* cfg, VideoOutFlipStatus* out) {
	EXIT_IF(cfg == nullptr);
	EXIT_IF(out == nullptr);

	Common::LockGuard lock(m_mutex);

	*out = cfg->flip_status;
}

void FlipQueue::ClearPendingPhase32(VideoOutConfig* cfg) {
	EXIT_IF(cfg == nullptr);
	Common::LockGuard lock(m_mutex);
	if (cfg->flip_status.flipArg < 0) {
		cfg->flip_status.flipArg = 0;
	}
	// Only clear pending — preserve currentBuffer from the last real present (buffer0 bridge).
	cfg->flip_status.flipPendingNum = 0;
}

bool VideoOutFlipWindow(uint32_t micros) {
	EXIT_IF(g_video_out_context == nullptr);

	WaitForNextVblank();

	const bool presented = g_video_out_context->GetFlipQueue().Flip(micros);
	if (presented || Graphics::boot_trace_log()) {
		LOGF("FlipTrace: VideoOutFlipWindow presented=%s\n", presented ? "true" : "false");
	}
	return presented;
}

void VideoOutBeginVblank() {
	EXIT_IF(g_video_out_context == nullptr);

	g_video_out_context->VblankBegin();
}

void VideoOutEndVblank() {
	EXIT_IF(g_video_out_context == nullptr);

	g_video_out_context->VblankEnd();
}

KYTY_SYSV_ABI int VideoOutOpen(int user_id, int bus_type, int index, const void* param) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	EXIT_NOT_IMPLEMENTED(user_id != 255 && user_id != 0);
	EXIT_NOT_IMPLEMENTED(bus_type != 0);
	EXIT_NOT_IMPLEMENTED(index != 0);

	LOGF("\t param = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(param));

	int handle = g_video_out_context->Open();

	if (handle < 0) {
		return VIDEO_OUT_ERROR_RESOURCE_BUSY;
	}

	FlipStats::open.fetch_add(1, std::memory_order_relaxed);
	LOGF("FlipTrace: VideoOutOpen handle=%d\n", handle);
	FlipStats::Log("open");
	Graphics::WindowEnsureVisible();

	return handle;
}

KYTY_SYSV_ABI int VideoOutClose(int handle) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	if (!g_video_out_context->IsOpened(handle)) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	g_video_out_context->Close(handle);

	return OK;
}

KYTY_SYSV_ABI void VideoOutSetBufferAttribute2(VideoOutBufferAttribute2* attribute,
                                               uint64_t pixel_format, uint32_t tiling_mode,
                                               uint32_t width, uint32_t height, uint64_t option,
                                               uint32_t dcc_control,
                                               uint64_t dcc_cb_register_clear_color) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(attribute == nullptr);

	LOGF("\t pixel_format                = %016" PRIx64 "\n"
	     "\t tiling_mode                 = %" PRIu32 "\n"
	     "\t width                       = %" PRIu32 "\n"
	     "\t height                      = %" PRIu32 "\n"
	     "\t option                      = %016" PRIx64 "\n"
	     "\t dcc_control                 = %08" PRIx32 "\n"
	     "\t dcc_cb_register_clear_color = %016" PRIx64 "\n",
	     pixel_format, tiling_mode, width, height, option, dcc_control,
	     dcc_cb_register_clear_color);

	memset(attribute, 0, sizeof(VideoOutBufferAttribute2));

	attribute->tiling_mode                 = tiling_mode;
	attribute->aspect_ratio                = 0;
	attribute->width                       = width;
	attribute->height                      = height;
	attribute->pitch_in_pixel              = 0;
	attribute->option                      = option;
	attribute->pixel_format                = pixel_format;
	attribute->dcc_cb_register_clear_color = dcc_cb_register_clear_color;
	attribute->dcc_control                 = dcc_control;
}

KYTY_SYSV_ABI int VideoOutSetFlipRate(int handle, int rate) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	LOGF("\trate = %d\n", rate);

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (rate < 0 || rate > 2) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	ctx->flip_rate = rate;

	return OK;
}

KYTY_SYSV_ABI int VideoOutDeleteFlipEvent(EventQueue::KernelEqueue eq, int handle) {
	PRINT_NAME();
	return DeleteVideoOutEvent(handle, eq, VideoOutEventKind::Flip);
}

KYTY_SYSV_ABI int VideoOutAddFlipEvent(EventQueue::KernelEqueue eq, int handle, void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::Flip, udata);
}

KYTY_SYSV_ABI int VideoOutDeleteVblankEvent(EventQueue::KernelEqueue eq, int handle) {
	PRINT_NAME();
	return DeleteVideoOutEvent(handle, eq, VideoOutEventKind::Vblank);
}

KYTY_SYSV_ABI int VideoOutDeletePreVblankStartEvent(EventQueue::KernelEqueue eq, int handle) {
	PRINT_NAME();
	return DeleteVideoOutEvent(handle, eq, VideoOutEventKind::PreVblankStart);
}

KYTY_SYSV_ABI int VideoOutAddVblankEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                         void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::Vblank, udata);
}

KYTY_SYSV_ABI int VideoOutAddPreVblankStartEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                                 void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::PreVblankStart, udata);
}

KYTY_SYSV_ABI int VideoOutAddOutputModeEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                             void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::OutputMode, udata);
}

static int RegisterBuffersInternal(VideoOutConfig* ctx, int set_id, int start_index,
                                   const void* const* addresses, int buffer_num,
                                   const std::vector<Graphics::VideoOutInfo>& infos);

static void Phase34SaveMenuSnapshot(int handle, int set_id, int start_index, int buffer_num,
                                    const void* const* addresses,
                                    const std::vector<Graphics::VideoOutInfo>& infos) {
	// Always snapshot under PENDING0 — Phase 35 Mixed-thread path needs it even when
	// the Phase 34 host menu worker is disabled.
	if (!Phase32Pending0Enabled()) {
		return;
	}
	std::lock_guard<std::mutex> lock(g_phase34_mu);
	g_phase34_snap.handle      = handle;
	g_phase34_snap.set_id      = set_id;
	g_phase34_snap.start_index = start_index;
	g_phase34_snap.buffer_num  = buffer_num;
	g_phase34_snap.addresses.assign(addresses, addresses + buffer_num);
	g_phase34_snap.infos = infos;
	LOGF("FlipTrace: phase34 saved menu snapshot handle=%d set=%d start=%d num=%d\n", handle,
	     set_id, start_index, buffer_num);
}

static void Phase34MenuBootstrapThread() {
	std::this_thread::sleep_for(std::chrono::milliseconds(400));
	Phase34MenuSnapshot snap;
	{
		std::lock_guard<std::mutex> lock(g_phase34_mu);
		snap = g_phase34_snap;
	}
	if (snap.buffer_num <= 0 || snap.addresses.empty() ||
	    snap.infos.size() != static_cast<size_t>(snap.buffer_num) ||
	    g_video_out_context == nullptr) {
		LOGF("FlipTrace: phase34 menu bootstrap skipped — no snapshot\n");
		fprintf(stderr, "FlipTrace: phase34 menu bootstrap skipped — no snapshot\n");
		return;
	}
	auto* ctx = g_video_out_context->Get(snap.handle);
	if (ctx == nullptr) {
		LOGF("FlipTrace: phase34 menu bootstrap skipped — invalid handle\n");
		return;
	}

	// Re-bind guest registration onto deferred-vk images (no RegisterVideoOutSurfaces —
	// same guest pages still alias the texture-cache entries from BootCards).
	int reg = VIDEO_OUT_ERROR_INVALID_INDEX;
	{
		Common::LockGuard lock(ctx->mutex);
		if (ctx->closing) {
			reg = VIDEO_OUT_ERROR_INVALID_HANDLE;
		} else if (std::any_of(ctx->buffers_sets.begin(), ctx->buffers_sets.end(),
		                       [&](const auto& set) { return set.set_id == snap.set_id; })) {
			reg = VIDEO_OUT_ERROR_INVALID_INDEX;
		} else {
			bool ok = true;
			for (int i = 0; i < snap.buffer_num; i++) {
				const int slot = snap.start_index + i;
				if (ctx->buffers[slot].buffer != nullptr) {
					ok  = false;
					reg = VIDEO_OUT_ERROR_SLOT_OCCUPIED;
					break;
				}
				if (ctx->buffers[slot].buffer_vulkan == nullptr) {
					ok = false;
					break;
				}
			}
			if (ok) {
				ctx->buffers_sets.push_back(
				    {snap.start_index, snap.buffer_num, snap.set_id});
				for (int i = 0; i < snap.buffer_num; i++) {
					const int slot      = snap.start_index + i;
					auto&     dst       = ctx->buffers[slot];
					dst.set_id          = snap.set_id;
					dst.buffer          = snap.addresses[static_cast<size_t>(i)];
					dst.buffer_size     = snap.infos[static_cast<size_t>(i)].size;
					dst.buffer_pitch    = snap.infos[static_cast<size_t>(i)].pitch;
					ctx->buffer_labels[slot] = 0;
					LOGF("\tbuffers[%d] = %016" PRIx64 " (phase34 rebind deferred-vk)\n", slot,
					     reinterpret_cast<uint64_t>(dst.buffer));
				}
				FlipStats::register_buffers.fetch_add(1, std::memory_order_relaxed);
				LOGF("FlipTrace: RegisterBuffers set_id=%d start=%d num=%d vulkan_ok=%d "
				     "(phase34 rebind)\n",
				     snap.set_id, snap.start_index, snap.buffer_num, 1);
				FlipStats::Log("register_buffers");
				reg = OK;
			}
		}
	}
	LOGF("FlipTrace: phase34 MenuRegister result=%d set_id=%d\n", reg, snap.set_id);
	fprintf(stderr, "FlipTrace: phase34 MenuRegister result=%d set_id=%d\n", reg, snap.set_id);
	if (reg != OK) {
		return;
	}

	static uint32_t dcb_storage[8] = {0x80000000u, 0x80000000u, 0, 0, 0, 0, 0, 0};
	const uint32_t  dcb_size       = 2;
	LOGF("SubmitTrace: phase34 MenuSubmit dcb=%p size=0x%x (worker, not Kick)\n",
	     static_cast<void*>(dcb_storage), dcb_size);
	fprintf(stderr, "SubmitTrace: phase34 MenuSubmit size=0x%x\n", dcb_size);
	(void)Libs::Graphics::Gen5Driver::GraphicsDriverSubmitCommandBuffer(0, dcb_storage, dcb_size);

	const int flip0 = VideoOutSubmitFlip(snap.handle, 0, /*VIDEO_OUT_FLIP_MODE_VSYNC*/ 1, 0);
	LOGF("FlipTrace: phase34 MenuFlip index=0 result=%d (no KickFlip/bridge)\n", flip0);
	fprintf(stderr, "FlipTrace: phase34 MenuFlip index=0 result=%d\n", flip0);
	(void)LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
	(void)LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);

	for (int i = 0; i < 900; ++i) {
		std::this_thread::sleep_for(std::chrono::milliseconds(33));
		const int idx = (i & 1);
		(void)VideoOutSubmitFlip(snap.handle, idx, 1, 0);
		if ((i % 30) == 0) {
			(void)LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
			(void)LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
			Graphics::WindowArmIgnoreQuit(60);
		}
	}
}

static void Phase34ArmMenuBootstrap() {
	if (!Phase32Pending0Enabled() || Phase34MenuBootstrapDisabled()) {
		return;
	}
	if (g_phase34_bootstrap_started.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	LOGF("FlipTrace: phase34 arming menu bootstrap worker\n");
	fprintf(stderr, "FlipTrace: phase34 arming menu bootstrap worker\n");
	std::thread(Phase34MenuBootstrapThread).detach();
}

// Phase 35/36: host Mixed rebind+flip pump — BLACK frames (NOP DCB). Opt-in only.
// KYTY_PHASE35_HOST_FLIP=1 enables; KYTY_PHASE35_NO_GUEST_MENU=1 also disables.
static std::atomic<bool> g_phase35_guest_menu_armed {false};
static std::atomic<bool> g_phase35_guest_menu_done {false};

// Phase 37: stop BootCards keep-alive once a real guest flip arrives post-Unregister.
static std::atomic<bool>     g_phase37_keepalive_stop {false};
static std::atomic<uint64_t> g_phase37_keepalive_host_tid {0};

// Phase 43: soft-idle / boot progress requires sustained menu frames post-Unreg —
// first SubmitFlip alone must not park MainThread.
bool Phase38GuestBootProgressSeen() {
	return g_phase43_menu_frames_ok.load(std::memory_order_acquire);
}

bool Phase44GuestRegisterBuffers2Seen() {
	return g_phase44_guest_reg2_ok.load(std::memory_order_acquire);
}

static void Phase44CaptureDcbBaselineIfNeeded() {
	if (g_phase44_dcb_baseline_set.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint64_t dcb =
	    Libs::Graphics::Sync::SubmitTrace::submit_dcb.load(std::memory_order_relaxed);
	g_phase44_dcb_baseline.store(dcb, std::memory_order_release);
	LOGF("FlipTrace: phase44 dcb baseline submit_dcb=%" PRIu64 "\n", dcb);
	fprintf(stderr, "FlipTrace: phase44 dcb baseline submit_dcb=%" PRIu64 "\n", dcb);
}

static void Phase44CheckNdJobDcb() {
	if (g_phase44_ndjob_dcb_ok.load(std::memory_order_acquire)) {
		return;
	}
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	Phase44CaptureDcbBaselineIfNeeded();
	const uint64_t dcb =
	    Libs::Graphics::Sync::SubmitTrace::submit_dcb.load(std::memory_order_relaxed);
	const uint64_t base = g_phase44_dcb_baseline.load(std::memory_order_acquire);
	if (dcb > base) {
		if (!g_phase44_ndjob_dcb_ok.exchange(true, std::memory_order_acq_rel)) {
			LOGF("FlipTrace: phase44 ndjob submit_dcb=%" PRIu64 " baseline=%" PRIu64
			     " tid=%d\n",
			     dcb, base, Common::Thread::GetThreadIdUnique());
			fprintf(stderr, "FlipTrace: phase44 ndjob submit_dcb=%" PRIu64 "\n", dcb);
		}
	}
}

// Free snapshot-held slots (deferred-vk style) so guest VideoOutRegisterBuffers2 can succeed.
static void Phase44ClearSnapshotForGuestRetry() {
	if (g_phase44_snapshot_cleared.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	if (g_phase44_guest_reg2_ok.load(std::memory_order_acquire) ||
	    g_video_out_context == nullptr) {
		return;
	}
	Phase34MenuSnapshot snap;
	{
		std::lock_guard<std::mutex> lock(g_phase34_mu);
		snap = g_phase34_snap;
	}
	if (snap.buffer_num <= 0) {
		return;
	}
	auto* ctx = g_video_out_context->Get(snap.handle);
	if (ctx == nullptr) {
		g_phase44_snapshot_cleared.store(false, std::memory_order_release);
		return;
	}
	{
		Common::LockGuard lock(ctx->mutex);
		auto set_it = std::find_if(ctx->buffers_sets.begin(), ctx->buffers_sets.end(),
		                           [&](const auto& set) { return set.set_id == snap.set_id; });
		if (set_it == ctx->buffers_sets.end()) {
			return;
		}
		const int start = set_it->start_index;
		const int num   = set_it->num;
		for (int i = start; i < start + num; i++) {
			ctx->unregistering[i] = false;
			auto* kept_vk         = ctx->buffers[i].buffer_vulkan;
			ctx->buffers[i]       = VideoOutBufferInfo {};
			ctx->buffers[i].buffer_vulkan = kept_vk;
			ctx->buffer_labels[i] = 0;
		}
		ctx->buffers_sets.erase(set_it);
		if (ctx->prev_flip_index >= start && ctx->prev_flip_index < start + num) {
			ctx->prev_flip_index = -1;
		}
	}
	g_phase42_reregister_ok.store(false, std::memory_order_release);
	g_phase44_snapshot_done.store(false, std::memory_order_release);
	g_phase37_keepalive_stop.store(true, std::memory_order_release);
	// Do NOT reset g_phase44_snapshot_cleared — clear-once forever (no thrash).
	LOGF("FlipTrace: phase44 cleared snapshot set=%d for guest RegisterBuffers2 retry "
	     "(keep-alive stop)\n",
	     snap.set_id);
	fprintf(stderr, "FlipTrace: phase44 cleared snapshot for guest Reg2 retry\n");
	(void)LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
	(void)LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
}

// Phase 44: re-Register via VideoOutRegisterBuffers2 ABI from MainThread (guest divert
// never returns to the title's own Reg2 call-site). Counts as ABI Reg2, not snapshot poke.
static void Phase44AttemptAbiReregister() {
	if (g_phase44_guest_reg2_ok.load(std::memory_order_acquire)) {
		return;
	}
	if (!g_phase44_snapshot_cleared.load(std::memory_order_acquire)) {
		return;
	}
	if (!g_phase34_attr_valid.load(std::memory_order_acquire) ||
	    g_video_out_context == nullptr) {
		return;
	}
	static std::atomic<bool> in_flight {false};
	if (in_flight.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	Phase34MenuSnapshot snap;
	VideoOutBufferAttribute2 attr {};
	int category = 0;
	{
		std::lock_guard<std::mutex> lock(g_phase34_mu);
		snap     = g_phase34_snap;
		attr     = g_phase34_attr;
		category = g_phase34_attr_category.load(std::memory_order_relaxed);
	}
	if (snap.buffer_num <= 0 || snap.addresses.empty() ||
	    snap.addresses.size() != static_cast<size_t>(snap.buffer_num)) {
		in_flight.store(false, std::memory_order_release);
		return;
	}
	std::vector<VideoOutBuffers> bufs(static_cast<size_t>(snap.buffer_num));
	for (int i = 0; i < snap.buffer_num; i++) {
		bufs[static_cast<size_t>(i)].data =
		    snap.addresses[static_cast<size_t>(i)];
		bufs[static_cast<size_t>(i)].metadata =
		    snap.infos.size() == static_cast<size_t>(snap.buffer_num)
		        ? reinterpret_cast<const void*>(snap.infos[static_cast<size_t>(i)].metadata_address)
		        : nullptr;
		bufs[static_cast<size_t>(i)].reserved[0] = nullptr;
		bufs[static_cast<size_t>(i)].reserved[1] = nullptr;
	}
	LOGF("FlipTrace: phase44 ABI RegisterBuffers2 attempt handle=%d set=%d num=%d tid=%d\n",
	     snap.handle, snap.set_id, snap.buffer_num, Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "FlipTrace: phase44 ABI RegisterBuffers2 attempt set=%d\n", snap.set_id);
	const int reg = VideoOutRegisterBuffers2(snap.handle, snap.set_id, snap.start_index,
	                                         bufs.data(), snap.buffer_num, &attr, category,
	                                         nullptr);
	LOGF("FlipTrace: phase44 ABI RegisterBuffers2 result=%d set=%d\n", reg, snap.set_id);
	fprintf(stderr, "FlipTrace: phase44 ABI RegisterBuffers2 result=%d\n", reg);
	in_flight.store(false, std::memory_order_release);
}

static void Phase43CaptureBaselineIfNeeded() {
	if (g_phase43_baseline_set.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint64_t presented =
	    FlipStats::presented.load(std::memory_order_relaxed);
	const uint64_t dcb =
	    Libs::Graphics::Sync::SubmitTrace::submit_dcb.load(std::memory_order_relaxed);
	g_phase43_presented_baseline.store(presented, std::memory_order_release);
	g_phase43_submit_dcb_baseline.store(dcb, std::memory_order_release);
	LOGF("FlipTrace: phase43 baseline presented=%" PRIu64 " submit_dcb=%" PRIu64 "\n",
	     presented, dcb);
	fprintf(stderr, "FlipTrace: phase43 baseline presented=%" PRIu64 " submit_dcb=%" PRIu64 "\n",
	        presented, dcb);
}

static void Phase43UpdateMenuFramesOk() {
	if (g_phase43_menu_frames_ok.load(std::memory_order_acquire)) {
		return;
	}
	if (!g_phase43_baseline_set.load(std::memory_order_acquire)) {
		return;
	}
	const uint64_t presented =
	    FlipStats::presented.load(std::memory_order_relaxed);
	const uint64_t baseline = g_phase43_presented_baseline.load(std::memory_order_acquire);
	const uint64_t delta =
	    presented >= baseline ? (presented - baseline) : 0;
	const uint64_t dcb =
	    Libs::Graphics::Sync::SubmitTrace::submit_dcb.load(std::memory_order_relaxed);
	const uint64_t dcb_base = g_phase43_submit_dcb_baseline.load(std::memory_order_acquire);
	const bool     dcb_ok   = dcb > dcb_base;
	// Park only on sustained presented frames — submit_dcb is secondary success, not park.
	if (delta >= 30) {
		if (!g_phase43_menu_frames_ok.exchange(true, std::memory_order_acq_rel)) {
			LOGF("FlipTrace: phase43 menu_frames_ok presented=%" PRIu64 " baseline=%" PRIu64
			     " delta=%" PRIu64 " submit_dcb=%" PRIu64 " dcb_base=%" PRIu64
			     " dcb_ok=%d sustain_n=%" PRIu64 "\n",
			     presented, baseline, delta, dcb, dcb_base, dcb_ok ? 1 : 0,
			     g_phase43_sustain_n.load(std::memory_order_relaxed));
			fprintf(stderr,
			        "FlipTrace: phase43 menu_frames_ok delta=%" PRIu64 " submit_dcb=%" PRIu64
			        "\n",
			        delta, dcb);
		}
	} else if (dcb_ok && (g_phase43_sustain_n.load(std::memory_order_relaxed) % 30) == 0) {
		LOGF("FlipTrace: phase43 submit_dcb progress dcb=%" PRIu64 " base=%" PRIu64
		     " presented_delta=%" PRIu64 " (not parking yet)\n",
		     dcb, dcb_base, delta);
	}
}

// Minimal DCB seed — opt-in KYTY_PHASE43_DCB_SEED=1 (P44 default: off; not a P44 success).
static void Phase43SeedNdJobDcb() {
	const char* seed = std::getenv("KYTY_PHASE43_DCB_SEED");
	if (seed == nullptr || seed[0] != '1') {
		Phase44CaptureDcbBaselineIfNeeded();
		return;
	}
	if (g_phase43_dcb_seeded.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		g_phase43_dcb_seeded.store(false, std::memory_order_release);
		return;
	}
	Phase42RearmNdJobEnqueue();
	static uint32_t dcb_storage[8] = {0x80000000u, 0x80000000u, 0, 0, 0, 0, 0, 0};
	const uint32_t  dcb_size       = 2;
	LOGF("SubmitTrace: phase43 NdJob DCB seed dcb=%p size=0x%x\n",
	     static_cast<void*>(dcb_storage), dcb_size);
	fprintf(stderr, "SubmitTrace: phase43 NdJob DCB seed size=0x%x\n", dcb_size);
	(void)Libs::Graphics::Gen5Driver::GraphicsDriverSubmitCommandBuffer(0, dcb_storage, dcb_size);
	const uint64_t dcb =
	    Libs::Graphics::Sync::SubmitTrace::submit_dcb.load(std::memory_order_relaxed);
	LOGF("FlipTrace: phase43 NdJob DCB seed done submit_dcb=%" PRIu64 "\n", dcb);
	fprintf(stderr, "FlipTrace: phase43 NdJob DCB seed done submit_dcb=%" PRIu64 "\n", dcb);
	// P44 baseline is after seed so only further submits count as ndjob success.
	g_phase44_dcb_baseline_set.store(false, std::memory_order_release);
	Phase44CaptureDcbBaselineIfNeeded();
}

static void Phase43SustainMenuFlips() {
	if (!g_phase42_reregister_ok.load(std::memory_order_acquire) ||
	    g_video_out_context == nullptr) {
		return;
	}
	// Stop host sustain once true guest Reg2 owns the buffers.
	if (g_phase44_guest_reg2_ok.load(std::memory_order_acquire) &&
	    g_phase43_menu_frames_ok.load(std::memory_order_acquire)) {
		return;
	}
	Phase43CaptureBaselineIfNeeded();
	const int handle = g_phase42_flip_handle.load(std::memory_order_acquire);
	const int num    = g_phase42_flip_num.load(std::memory_order_acquire);
	if (handle <= 0 || num <= 0) {
		return;
	}
	const uint64_t sn = g_phase43_sustain_n.fetch_add(1, std::memory_order_relaxed);
	const int      index = static_cast<int>(sn & 1ULL) % (num > 0 ? num : 1);
	const int result =
	    VideoOutSubmitFlip(handle, index, /*VIDEO_OUT_FLIP_MODE_VSYNC*/ 1, 0);
	(void)LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
	(void)LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
	const uint64_t presented =
	    FlipStats::presented.load(std::memory_order_relaxed);
	if (sn < 16 || (sn % 30) == 0) {
		LOGF("FlipTrace: phase43 sustain flip n=%" PRIu64 " handle=%d index=%d result=%d "
		     "presented=%" PRIu64 "\n",
		     sn, handle, index, result, presented);
		fprintf(stderr,
		        "FlipTrace: phase43 sustain flip n=%" PRIu64 " presented=%" PRIu64 "\n", sn,
		        presented);
	}
	(void)g_phase42_flip_attempted.exchange(true, std::memory_order_acq_rel);
}

static void Phase41AttemptSnapshotReregister() {
	if (g_phase44_guest_reg2_ok.load(std::memory_order_acquire) ||
	    g_phase44_snapshot_done.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<bool> in_flight {false};
	if (in_flight.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	if (!g_phase37_post_unreg.load(std::memory_order_acquire) || g_video_out_context == nullptr) {
		in_flight.store(false, std::memory_order_release);
		return;
	}
	Phase34MenuSnapshot snap;
	{
		std::lock_guard<std::mutex> lock(g_phase34_mu);
		snap = g_phase34_snap;
	}
	if (snap.buffer_num <= 0 || snap.addresses.empty() ||
	    snap.infos.size() != static_cast<size_t>(snap.buffer_num)) {
		LOGF("FlipTrace: phase41 snapshot re-Register skipped — no snapshot\n");
		in_flight.store(false, std::memory_order_release);
		return;
	}
	auto* ctx = g_video_out_context->Get(snap.handle);
	if (ctx == nullptr) {
		in_flight.store(false, std::memory_order_release);
		return;
	}
	int reg = VIDEO_OUT_ERROR_INVALID_INDEX;
	{
		Common::LockGuard lock(ctx->mutex);
		if (ctx->closing) {
			reg = VIDEO_OUT_ERROR_INVALID_HANDLE;
		} else if (std::any_of(ctx->buffers_sets.begin(), ctx->buffers_sets.end(),
		                       [&](const auto& set) { return set.set_id == snap.set_id; })) {
			// Keep-alive may already hold the set — adopt it as host snapshot fallback.
			for (int i = 0; i < snap.buffer_num; i++) {
				const int slot = snap.start_index + i;
				auto&     dst  = ctx->buffers[slot];
				dst.set_id       = snap.set_id;
				dst.buffer       = snap.addresses[static_cast<size_t>(i)];
				dst.buffer_size  = snap.infos[static_cast<size_t>(i)].size;
				dst.buffer_pitch = snap.infos[static_cast<size_t>(i)].pitch;
				ctx->buffer_labels[slot] = 0;
			}
			reg = OK;
			LOGF("FlipTrace: phase41 snapshot adopt existing set_id=%d (keep-alive rebind)\n",
			     snap.set_id);
		} else {
			bool ok = true;
			for (int i = 0; i < snap.buffer_num; i++) {
				const int slot = snap.start_index + i;
				if (ctx->buffers[slot].buffer != nullptr) {
					ok  = false;
					reg = VIDEO_OUT_ERROR_SLOT_OCCUPIED;
					break;
				}
				if (ctx->buffers[slot].buffer_vulkan == nullptr) {
					ok = false;
					break;
				}
			}
			if (ok) {
				ctx->buffers_sets.push_back(
				    {snap.start_index, snap.buffer_num, snap.set_id});
				for (int i = 0; i < snap.buffer_num; i++) {
					const int slot   = snap.start_index + i;
					auto&     dst    = ctx->buffers[slot];
					dst.set_id       = snap.set_id;
					dst.buffer       = snap.addresses[static_cast<size_t>(i)];
					dst.buffer_size  = snap.infos[static_cast<size_t>(i)].size;
					dst.buffer_pitch = snap.infos[static_cast<size_t>(i)].pitch;
					ctx->buffer_labels[slot] = 0;
				}
				FlipStats::register_buffers.fetch_add(1, std::memory_order_relaxed);
				reg = OK;
			}
		}
	}
	LOGF("FlipTrace: phase41 snapshot re-Register handle=%d set=%d start=%d num=%d "
	     "result=%d\n",
	     snap.handle, snap.set_id, snap.start_index, snap.buffer_num, reg);
	fprintf(stderr, "FlipTrace: phase41 snapshot re-Register result=%d set=%d\n", reg,
	        snap.set_id);
	if (reg == OK) {
		g_phase37_guest_reg_seen.store(true, std::memory_order_release);
		g_phase42_reregister_ok.store(true, std::memory_order_release);
		g_phase42_flip_handle.store(snap.handle, std::memory_order_release);
		g_phase42_flip_num.store(snap.buffer_num, std::memory_order_release);
		g_phase44_snapshot_done.store(true, std::memory_order_release);
		Phase43CaptureBaselineIfNeeded();
		LOGF("FlipTrace: phase41 snapshot re-Register OK set_id=%d num=%d tid=%d "
		     "(host fallback, not ABI Reg2)\n",
		     snap.set_id, snap.buffer_num, Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "FlipTrace: phase41 snapshot re-Register OK set=%d (not ABI Reg2)\n",
		        snap.set_id);
	}
	in_flight.store(false, std::memory_order_release);
}

void Phase38NudgeBootWorkers() {
	Phase38NudgeBootWorkersOnce();
}

static uint64_t Phase37HostThreadId() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return static_cast<uint64_t>(::GetCurrentThreadId());
#else
	return std::hash<std::thread::id> {}(std::this_thread::get_id());
#endif
}

static void Phase37NoteGuestSubmitFlip(int index) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire) || index < 0) {
		return;
	}
	const uint64_t ka = g_phase37_keepalive_host_tid.load(std::memory_order_acquire);
	const uint64_t self = Phase37HostThreadId();
	if (ka != 0 && self == ka) {
		return; // keep-alive host pump
	}
	if (!g_phase37_guest_flip_seen.exchange(true, std::memory_order_acq_rel)) {
		g_phase37_keepalive_stop.store(true, std::memory_order_release);
		LOGF("FlipTrace: phase42 guest SubmitFlip post-Unreg index=%d tid=%d — keep-alive stop\n",
		     index, Common::Thread::GetThreadIdUnique());
		fprintf(stderr,
		        "FlipTrace: phase42 guest SubmitFlip post-Unreg index=%d — keep-alive stop\n",
		        index);
		LOGF("FlipTrace: phase37 guest SubmitFlip seen index=%d tid=%d — keep-alive stop\n",
		     index, Common::Thread::GetThreadIdUnique());
	}
}

static bool Phase35HostFlipEnabled() {
	const char* no = std::getenv("KYTY_PHASE35_NO_GUEST_MENU");
	if (no != nullptr && no[0] == '1') {
		return false;
	}
	const char* host = std::getenv("KYTY_PHASE35_HOST_FLIP");
	return host != nullptr && host[0] == '1';
}

static int Phase35RebindAndSubmitFlip(const char* tag) {
	Phase34MenuSnapshot snap;
	{
		std::lock_guard<std::mutex> lock(g_phase34_mu);
		snap = g_phase34_snap;
	}
	if (snap.buffer_num <= 0 || snap.addresses.empty() ||
	    snap.infos.size() != static_cast<size_t>(snap.buffer_num) ||
	    g_video_out_context == nullptr) {
		LOGF("FlipTrace: %s skipped — no snapshot\n", tag);
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}
	auto* ctx = g_video_out_context->Get(snap.handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	int reg = VIDEO_OUT_ERROR_INVALID_INDEX;
	{
		Common::LockGuard lock(ctx->mutex);
		if (ctx->closing) {
			return VIDEO_OUT_ERROR_INVALID_HANDLE;
		}
		if (std::any_of(ctx->buffers_sets.begin(), ctx->buffers_sets.end(),
		                [&](const auto& set) { return set.set_id == snap.set_id; })) {
			reg = OK; // already registered
		} else {
			bool ok = true;
			for (int i = 0; i < snap.buffer_num; i++) {
				const int slot = snap.start_index + i;
				if (ctx->buffers[slot].buffer != nullptr ||
				    ctx->buffers[slot].buffer_vulkan == nullptr) {
					ok = false;
					break;
				}
			}
			if (ok) {
				ctx->buffers_sets.push_back(
				    {snap.start_index, snap.buffer_num, snap.set_id});
				for (int i = 0; i < snap.buffer_num; i++) {
					const int slot   = snap.start_index + i;
					auto&     dst    = ctx->buffers[slot];
					dst.set_id       = snap.set_id;
					dst.buffer       = snap.addresses[static_cast<size_t>(i)];
					dst.buffer_size  = snap.infos[static_cast<size_t>(i)].size;
					dst.buffer_pitch = snap.infos[static_cast<size_t>(i)].pitch;
					ctx->buffer_labels[slot] = 0;
				}
				FlipStats::register_buffers.fetch_add(1, std::memory_order_relaxed);
				LOGF("FlipTrace: RegisterBuffers set_id=%d start=%d num=%d vulkan_ok=%d "
				     "(%s rebind)\n",
				     snap.set_id, snap.start_index, snap.buffer_num, 1, tag);
				FlipStats::Log("register_buffers");
				reg = OK;
			}
		}
	}
	LOGF("FlipTrace: %s Register result=%d set_id=%d tid=%d\n", tag, reg, snap.set_id,
	     Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "FlipTrace: %s Register result=%d tid=%d\n", tag, reg,
	        Common::Thread::GetThreadIdUnique());
	if (reg != OK) {
		return reg;
	}

	static uint32_t dcb_storage[8] = {0x80000000u, 0x80000000u, 0, 0, 0, 0, 0, 0};
	const uint32_t  dcb_size       = 2;
	LOGF("SubmitTrace: %s SubmitCommandBuffer dcb=%p size=0x%x tid=%d\n", tag,
	     static_cast<void*>(dcb_storage), dcb_size, Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "SubmitTrace: %s SubmitCommandBuffer size=0x%x tid=%d\n", tag, dcb_size,
	        Common::Thread::GetThreadIdUnique());
	(void)Libs::Graphics::Gen5Driver::GraphicsDriverSubmitCommandBuffer(0, dcb_storage, dcb_size);

	const int flip0 = VideoOutSubmitFlip(snap.handle, 0, 1, 0);
	LOGF("FlipTrace: %s SubmitFlip index=0 result=%d tid=%d\n", tag, flip0,
	     Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "FlipTrace: %s SubmitFlip index=0 result=%d tid=%d\n", tag, flip0,
	        Common::Thread::GetThreadIdUnique());
	(void)LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
	(void)LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);

	const int handle = snap.handle;
	std::thread([handle]() {
		for (int i = 0; i < 900; ++i) {
			std::this_thread::sleep_for(std::chrono::milliseconds(33));
			(void)VideoOutSubmitFlip(handle, i & 1, 1, 0);
			if ((i % 30) == 0) {
				(void)LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
				(void)LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
				Graphics::WindowArmIgnoreQuit(60);
			}
		}
	}).detach();
	return OK;
}

void Phase35ArmGuestMenuAfterUnregister() {
	g_phase37_post_unreg.store(true, std::memory_order_release);
	g_phase39_post_unreg.store(true, std::memory_order_release);
	g_phase37_guest_flip_seen.store(false, std::memory_order_release);
	g_phase37_guest_reg_seen.store(false, std::memory_order_release);
	g_phase37_keepalive_stop.store(false, std::memory_order_release);
	g_phase42_reregister_ok.store(false, std::memory_order_release);
	g_phase42_flip_attempted.store(false, std::memory_order_release);
	g_phase42_flip_handle.store(0, std::memory_order_release);
	g_phase42_flip_num.store(0, std::memory_order_release);

	// Phase 39: periodic wake until guest Register/Flip (no hard 20s stop).
	{
		static std::atomic<bool> nudge_started {false};
		if (!nudge_started.exchange(true, std::memory_order_acq_rel)) {
			std::thread([]() {
				LOGF("FlipTrace: phase39 boot nudge thread start (until guest progress)\n");
				fprintf(stderr, "FlipTrace: phase39 boot nudge thread start\n");
				for (int i = 0;; ++i) {
					if (Phase38GuestBootProgressSeen()) {
						LOGF("FlipTrace: phase39 boot nudge stop — guest progress i=%d\n", i);
						fprintf(stderr, "FlipTrace: phase39 boot nudge stop — guest progress\n");
						break;
					}
					if (i > 0 && (i % 300) == 0) {
						LOGF("FlipTrace: phase39 nudge still waiting (%ds) hle3=%" PRIu64
						     " hle4=%" PRIu64 "\n",
						     i / 10, g_phase39_hle_hits[3].load(std::memory_order_relaxed),
						     g_phase39_hle_hits[4].load(std::memory_order_relaxed));
						fprintf(stderr, "FlipTrace: phase39 nudge waiting %ds\n", i / 10);
					}
					Phase41MenuHandoffAttempt();
					Phase38NudgeBootWorkersOnce();
					Graphics::WindowArmIgnoreQuit(60);
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
				}
			}).detach();
		}
	}

	if (!Phase32Pending0Enabled() || !Phase35HostFlipEnabled()) {
		LOGF("FlipTrace: phase36 host Mixed flip pump OFF (set KYTY_PHASE35_HOST_FLIP=1)\n");
		fprintf(stderr, "FlipTrace: phase36 host Mixed flip pump OFF\n");
		// Keep BootCards guest pixels on screen (rebind + flip only, no NOP DCB).
		// Phase 37: stops when guest SubmitFlip arrives (g_phase37_keepalive_stop).
		const char* no_reveal = std::getenv("KYTY_PHASE36_NO_REVEAL");
		if (no_reveal == nullptr || no_reveal[0] != '1') {
			static std::atomic<bool> keep_started {false};
			if (!keep_started.exchange(true, std::memory_order_acq_rel)) {
				std::thread([]() {
					g_phase37_keepalive_host_tid.store(Phase37HostThreadId(),
					                                   std::memory_order_release);
					std::this_thread::sleep_for(std::chrono::milliseconds(200));
					Phase34MenuSnapshot snap;
					{
						std::lock_guard<std::mutex> lock(g_phase34_mu);
						snap = g_phase34_snap;
					}
					if (snap.buffer_num <= 0 || g_video_out_context == nullptr) {
						g_phase37_keepalive_host_tid.store(0, std::memory_order_release);
						return;
					}
					auto* ctx = g_video_out_context->Get(snap.handle);
					if (ctx == nullptr) {
						g_phase37_keepalive_host_tid.store(0, std::memory_order_release);
						return;
					}
					{
						Common::LockGuard lock(ctx->mutex);
						if (ctx->closing) {
							g_phase37_keepalive_host_tid.store(0, std::memory_order_release);
							return;
						}
						const bool have = std::any_of(
						    ctx->buffers_sets.begin(), ctx->buffers_sets.end(),
						    [&](const auto& set) { return set.set_id == snap.set_id; });
						if (!have &&
						    snap.addresses.size() == static_cast<size_t>(snap.buffer_num)) {
							bool ok = true;
							for (int i = 0; i < snap.buffer_num; i++) {
								const int slot = snap.start_index + i;
								if (ctx->buffers[slot].buffer_vulkan == nullptr) {
									ok = false;
									break;
								}
							}
							if (ok) {
								ctx->buffers_sets.push_back(
								    {snap.start_index, snap.buffer_num, snap.set_id});
								for (int i = 0; i < snap.buffer_num; i++) {
									const int slot = snap.start_index + i;
									auto&     dst  = ctx->buffers[slot];
									dst.set_id       = snap.set_id;
									dst.buffer       = snap.addresses[static_cast<size_t>(i)];
									dst.buffer_size  = snap.infos[static_cast<size_t>(i)].size;
									dst.buffer_pitch = snap.infos[static_cast<size_t>(i)].pitch;
								}
								LOGF("FlipTrace: phase36 keep-alive rebind set=%d num=%d "
								     "(BootCards pixels, no DCB)\n",
								     snap.set_id, snap.buffer_num);
								fprintf(stderr, "FlipTrace: phase36 keep-alive rebind\n");
							}
						}
					}
					for (int i = 0; i < 1800; ++i) {
						if (g_phase37_keepalive_stop.load(std::memory_order_acquire)) {
							LOGF("FlipTrace: phase37 keep-alive stopped at i=%d "
							     "(guest flip took over)\n",
							     i);
							fprintf(stderr, "FlipTrace: phase37 keep-alive stopped at i=%d\n",
							        i);
							break;
						}
						(void)VideoOutSubmitFlip(snap.handle, i & 1, 1, 0);
						if ((i % 30) == 0) {
							(void)LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
							Graphics::WindowArmIgnoreQuit(60);
						}
						std::this_thread::sleep_for(std::chrono::milliseconds(33));
					}
					g_phase37_keepalive_host_tid.store(0, std::memory_order_release);
				}).detach();
			}
		}
		return;
	}
	g_phase35_guest_menu_armed.store(true, std::memory_order_release);
	LOGF("FlipTrace: phase35 arm host Mixed flip (KYTY_PHASE35_HOST_FLIP=1)\n");
	fprintf(stderr, "FlipTrace: phase35 arm host Mixed flip\n");
}

void Phase35TryGuestMenuFromSubmissionThread(const char* thread_name) {
	if (!g_phase35_guest_menu_armed.load(std::memory_order_acquire) ||
	    g_phase35_guest_menu_done.load(std::memory_order_acquire)) {
		return;
	}
	if (thread_name == nullptr || std::strstr(thread_name, "Mixed") == nullptr) {
		return;
	}
	if (g_phase35_guest_menu_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	LOGF("FlipTrace: phase35 Mixed guest menu begin name=%s tid=%d\n", thread_name,
	     Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "FlipTrace: phase35 Mixed guest menu begin tid=%d\n",
	        Common::Thread::GetThreadIdUnique());
	(void)Phase35RebindAndSubmitFlip("phase35 Mixed");
}

static int RegisterBuffersInternal(VideoOutConfig* ctx, int set_id, int start_index,
                                   const void* const* addresses, int buffer_num,
                                   const std::vector<Graphics::VideoOutInfo>& infos) {
	if (ctx == nullptr || addresses == nullptr || buffer_num <= 0 ||
	    infos.size() != static_cast<size_t>(buffer_num)) {
		EXIT("invalid internal video-out buffer registration arguments\n");
	}
	if (set_id < 0 || set_id >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX) {
		EXIT("internal video-out buffer set identifier is out of range\n");
	}
	Graphics::WindowWaitForGraphicInitialized();
	Graphics::GraphicsRenderCreateContext();
	auto*             graphic_ctx = g_video_out_context->GetGraphicCtx();
	Common::LockGuard lock(ctx->mutex);
	if (ctx->closing) {
		EXIT("cannot register buffers on a closing video-out handle\n");
	}
	if (std::any_of(ctx->buffers_sets.begin(), ctx->buffers_sets.end(),
	                [set_id](const auto& set) { return set.set_id == set_id; })) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}
	for (int i = 0; i < buffer_num; i++) {
		if (ctx->unregistering[start_index + i]) {
			EXIT("video-out buffer registration raced with unregistration\n");
		}
		if (ctx->buffers[start_index + i].buffer != nullptr) {
			return VIDEO_OUT_ERROR_SLOT_OCCUPIED;
		}
	}
	auto images =
	    Graphics::g_render_ctx->GetTextureCache()->RegisterVideoOutSurfaces(graphic_ctx, infos);
	if (images.size() != infos.size()) {
		EXIT("video-out texture cache returned an incomplete surface set\n");
	}
	ctx->buffers_sets.push_back({start_index, buffer_num, set_id});
	for (int i = 0; i < buffer_num; i++) {
		auto& dst         = ctx->buffers[i + start_index];
		dst.set_id        = set_id;
		dst.buffer        = addresses[i];
		dst.buffer_size   = infos[i].size;
		dst.buffer_pitch  = infos[i].pitch;
		dst.buffer_vulkan = images[i];
		ctx->buffer_labels[i + start_index] = 0;
		LOGF("\tbuffers[%d] = %016" PRIx64 " metadata = %016" PRIx64 " dcc = %08" PRIx32 "\n",
		     i + start_index, reinterpret_cast<uint64_t>(addresses[i]), infos[i].metadata_address,
		     infos[i].dcc_control);
	}

	FlipStats::register_buffers.fetch_add(1, std::memory_order_relaxed);
	LOGF("FlipTrace: RegisterBuffers set_id=%d start=%d num=%d vulkan_ok=%d\n", set_id, start_index,
	     buffer_num, 1);
	FlipStats::Log("register_buffers");
	if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
		g_phase37_guest_reg_seen.store(true, std::memory_order_release);
		g_phase42_reregister_ok.store(true, std::memory_order_release);
		g_phase42_flip_num.store(buffer_num, std::memory_order_release);
		if (!g_phase44_guest_reg2_ok.exchange(true, std::memory_order_acq_rel)) {
			LOGF("FlipTrace: phase44 guest RegisterBuffers2 set_id=%d start=%d num=%d tid=%d "
			     "(ABI post-Unregister)\n",
			     set_id, start_index, buffer_num, Common::Thread::GetThreadIdUnique());
			fprintf(stderr,
			        "FlipTrace: phase44 guest RegisterBuffers2 set_id=%d num=%d tid=%d\n",
			        set_id, buffer_num, Common::Thread::GetThreadIdUnique());
		}
	}
	// Phase 34: arm IgnoreQuit early under PENDING0 so BootCards survives accidental SDL_QUIT
	// before hold-gate / Unregister (those also re-arm).
	if (Phase32Pending0Enabled()) {
		Graphics::WindowArmIgnoreQuit(180);
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutRegisterBuffers2(int handle, int set_index, int buffer_index_start,
                                           const VideoOutBuffers* buffers, int buffer_num,
                                           const VideoOutBufferAttribute2* attribute, int category,
                                           void* option) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (buffers == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (attribute == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}

	if (set_index < 0 || set_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX ||
	    buffer_index_start < 0 || buffer_index_start >= VIDEO_OUT_BUFFER_NUM_MAX ||
	    buffer_num < 1 || buffer_num > VIDEO_OUT_BUFFER_NUM_MAX ||
	    buffer_index_start + buffer_num > VIDEO_OUT_BUFFER_NUM_MAX) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	LOGF("\t start_index    = %d\n"
	     "\t buffer_num     = %d\n"
	     "\t set_index      = %d\n"
	     "\t pixel_format   = 0x%016" PRIx64 "\n"
	     "\t tiling_mode    = %" PRIu32 "\n"
	     "\t aspect_ratio   = %" PRIu32 "\n"
	     "\t width          = %" PRIu32 "\n"
	     "\t height         = %" PRIu32 "\n"
	     "\t pitch_in_pixel = %" PRIu32 "\n"
	     "\t option         = %" PRIu64 "\n"
	     "\t category       = %d\n",
	     buffer_index_start, buffer_num, set_index, attribute->pixel_format, attribute->tiling_mode,
	     attribute->aspect_ratio, attribute->width, attribute->height, attribute->pitch_in_pixel,
	     attribute->option, category);

	if (option != nullptr) {
		EXIT("video-out buffer registration options are unsupported\n");
	}
	if (category != VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_UNCOMPRESSED &&
	    category != VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_COMPRESSED) {
		return VIDEO_OUT_ERROR_INVALID_CATEGORY;
	}
	const bool compressed = category == VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_COMPRESSED;
	std::vector<const void*>            addresses(static_cast<size_t>(buffer_num));
	std::vector<Graphics::VideoOutInfo> infos;
	infos.reserve(static_cast<size_t>(buffer_num));

	for (int i = 0; i < buffer_num; i++) {
		LOGF("\t buffers[%d]: data=%p metadata=%p\n", i, buffers[i].data, buffers[i].metadata);
		if (buffers[i].reserved[0] != nullptr || buffers[i].reserved[1] != nullptr) {
			LOGF("\t buffers[%d]: ignoring reserved fields {%p, %p}\n", i, buffers[i].reserved[0],
			     buffers[i].reserved[1]);
		}
		const auto data_address     = reinterpret_cast<uint64_t>(buffers[i].data);
		const auto metadata_address = reinterpret_cast<uint64_t>(buffers[i].metadata);
		const auto reserved0        = reinterpret_cast<uint64_t>(buffers[i].reserved[0]);
		const auto reserved1        = reinterpret_cast<uint64_t>(buffers[i].reserved[1]);
		LOGF("\t buffers[%d] data=0x%016" PRIx64 " metadata=0x%016" PRIx64
		     " reserved0=0x%016" PRIx64 " reserved1=0x%016" PRIx64 "\n",
		     i, data_address, metadata_address, reserved0, reserved1);
		if (reserved0 != 0 || reserved1 != 0) {
			LOGF("warning: ignoring non-null video-out buffer reserved fields\n");
		}
		const auto compression      = Graphics::ClassifyVideoOutCompression(
		    compressed, metadata_address, attribute->dcc_control,
		    attribute->dcc_cb_register_clear_color);
		if (compression == Graphics::VideoOutCompression::Unsupported) {
			EXIT("unsupported video-out compression, category=%d data=0x%016" PRIx64
			     " metadata=0x%016" PRIx64 " dcc_control=0x%08" PRIx32 " clear_color=0x%016" PRIx64
			     "\n",
			     category, data_address, metadata_address, attribute->dcc_control,
			     attribute->dcc_cb_register_clear_color);
		}
		addresses[static_cast<size_t>(i)] = buffers[i].data;
		infos.push_back(MakeVideoOutInfo(*attribute, data_address, metadata_address, compression));
	}

	// Phase 44: if keep-alive/snapshot already holds this set post-Unreg, yield slots to
	// the guest ABI so RegisterBuffers2 can succeed.
	if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
		Common::LockGuard lock(ctx->mutex);
		auto set_it = std::find_if(ctx->buffers_sets.begin(), ctx->buffers_sets.end(),
		                           [set_index](const auto& set) { return set.set_id == set_index; });
		if (set_it != ctx->buffers_sets.end()) {
			const int start = set_it->start_index;
			const int num   = set_it->num;
			for (int i = start; i < start + num; i++) {
				ctx->buffer_labels[i] = 0;
				ctx->unregistering[i] = false;
				auto* kept_vk         = ctx->buffers[i].buffer_vulkan;
				ctx->buffers[i]       = VideoOutBufferInfo {};
				ctx->buffers[i].buffer_vulkan = kept_vk;
			}
			if (ctx->prev_flip_index >= start && ctx->prev_flip_index < start + num) {
				ctx->prev_flip_index = -1;
			}
			ctx->buffers_sets.erase(set_it);
			g_phase37_keepalive_stop.store(true, std::memory_order_release);
			g_phase44_snapshot_done.store(false, std::memory_order_release);
			LOGF("FlipTrace: phase44 yield host set=%d for guest RegisterBuffers2 ABI\n",
			     set_index);
			fprintf(stderr, "FlipTrace: phase44 yield host set for guest Reg2 ABI\n");
		}
	}

	const int reg_result =
	    RegisterBuffersInternal(ctx, set_index, buffer_index_start, addresses.data(), buffer_num,
	                            infos);
	if (reg_result == OK) {
		Phase34SaveMenuSnapshot(handle, set_index, buffer_index_start, buffer_num,
		                        addresses.data(), infos);
		{
			std::lock_guard<std::mutex> lock(g_phase34_mu);
			g_phase34_attr = *attribute;
			g_phase34_attr_valid.store(true, std::memory_order_release);
			g_phase34_attr_category.store(category, std::memory_order_release);
		}
		if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
			g_phase42_flip_handle.store(handle, std::memory_order_release);
			g_phase42_flip_num.store(buffer_num, std::memory_order_release);
		}
	}
	return reg_result;
}

KYTY_SYSV_ABI int VideoOutSubmitChangeBufferAttribute2(int handle, int set_index,
                                                       const VideoOutBufferAttribute2* attribute,
                                                       void*                           option) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (attribute == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}
	if (set_index < 0 || set_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}

	if (option != nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}
	(void)ctx;
	(void)set_index;
	EXIT("video-out buffer attribute query is unsupported\n");
}

KYTY_SYSV_ABI int VideoOutUnregisterBuffers(int handle, int set_index) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (set_index < 0 || set_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}

	{
		Common::LockGuard lock(ctx->mutex);
		if (ctx->closing) {
			LOGF("FlipTrace: UnregisterBuffers handle=%d set=%d while closing — ignore\n", handle,
			     set_index);
			return VIDEO_OUT_ERROR_INVALID_HANDLE;
		}
		auto set_it = std::find_if(ctx->buffers_sets.begin(), ctx->buffers_sets.end(),
		                           [set_index](const auto& set) { return set.set_id == set_index; });
		if (set_it == ctx->buffers_sets.end()) {
			LOGF("FlipTrace: UnregisterBuffers handle=%d set=%d — not registered\n", handle,
			     set_index);
			return VIDEO_OUT_ERROR_INVALID_INDEX;
		}

		// Phase 33 diag: soft-NOOP keeps sets (blocks re-RegisterBuffers).
		const char* noop = std::getenv("KYTY_PHASE33_UNREG_NOOP");
		if (noop != nullptr && noop[0] == '1') {
			LOGF("FlipTrace: UnregisterBuffers handle=%d set=%d soft-NOOP count=%" PRIu64
			     " currentBuffer=%d\n",
			     handle, set_index, ctx->flip_status.count, ctx->flip_status.currentBuffer);
			fprintf(stderr, "FlipTrace: UnregisterBuffers handle=%d set=%d soft-NOOP\n", handle,
			        set_index);
			std::fflush(stderr);
			Common::NoteHleCall("VideoOut", "VideoOut", "UnregisterBuffersSoftNoop");
		} else {
			// Phase 34: clear guest registration so re-RegisterBuffers works. Defer Vulkan
			// destroy — keep buffer_vulkan for any in-flight present of currentBuffer.
			const int start = set_it->start_index;
			const int num   = set_it->num;
			for (int i = start; i < start + num; i++) {
				ctx->buffer_labels[i] = 0;
				ctx->unregistering[i] = false;
				auto* kept_vk         = ctx->buffers[i].buffer_vulkan;
				ctx->buffers[i]       = VideoOutBufferInfo {};
				ctx->buffers[i].buffer_vulkan = kept_vk;
			}
			if (ctx->prev_flip_index >= start && ctx->prev_flip_index < start + num) {
				ctx->prev_flip_index = -1;
			}
			ctx->buffers_sets.erase(set_it);
			LOGF("FlipTrace: UnregisterBuffers handle=%d set=%d deferred-vk count=%" PRIu64
			     " currentBuffer=%d (re-register allowed)\n",
			     handle, set_index, ctx->flip_status.count, ctx->flip_status.currentBuffer);
			fprintf(stderr,
			        "FlipTrace: UnregisterBuffers handle=%d set=%d deferred-vk "
			        "(re-register allowed)\n",
			        handle, set_index);
			std::fflush(stderr);
			Common::NoteHleCall("VideoOut", "VideoOut", "UnregisterBuffersDeferredVk");
		}
	}

	if (Phase32Pending0Enabled()) {
		g_pending0_watchdog_ticks.store(120, std::memory_order_release);
		Graphics::WindowArmIgnoreQuit(120);
		Graphics::WindowArmPhase35FiberSoftAck();
		(void)Phase32PatchPostUnregisterReturn(__builtin_return_address(0));
		Phase34ArmMenuBootstrap();
		Phase35ArmGuestMenuAfterUnregister();
	}

	const auto woken = LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
	LOGF("SubmitTrace: WakeSubmissionCond after UnregisterBuffers woken=%zu\n", woken);
	fprintf(stderr, "SubmitTrace: WakeSubmissionCond after UnregisterBuffers woken=%zu\n", woken);
	const int ue =
	    LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
	LOGF("SubmitTrace: UserEventTriggerForAll after UnregisterBuffers result=%d\n", ue);

	return OK;
}

} // namespace Libs::VideoOut

namespace Libs::Presentation {

DisplayBufferImage DisplayBufferFind(uint64_t addr, bool render_target) {
	EXIT_IF(VideoOut::g_video_out_context == nullptr);

	return VideoOut::g_video_out_context->FindImage(reinterpret_cast<void*>(addr), render_target);
}

} // namespace Libs::Presentation

namespace Libs::VideoOut {

KYTY_SYSV_ABI int VideoOutSubmitFlip(int handle, int index, int flip_mode, int64_t flip_arg) {
	PRINT_NAME();

	uint64_t  request_id = 0;
	const int result =
	    ReserveFlipRequest(handle, index, flip_mode, flip_arg, FlipRequestSource::Cpu, &request_id);
	if (result == VIDEO_OUT_ERROR_INVALID_VALUE) {
		LOGF("\t unsupported flip_mode = %d\n", flip_mode);
	}
	if (result != OK) {
		return result;
	}
	FlipStats::submit_cpu.fetch_add(1, std::memory_order_relaxed);
	LOGF("FlipTrace: SubmitFlipCPU handle=%d index=%d mode=%d arg=%" PRId64 " id=%" PRIu64 "\n",
	     handle, index, flip_mode, flip_arg, request_id);
	FlipStats::Log("submit_cpu");
	Phase37NoteGuestSubmitFlip(index);
	Graphics::GraphicsRunSubmitFlipPreparation();

	return OK;
}

} // namespace Libs::VideoOut

namespace Libs::Presentation {

int DisplayBufferSubmitFlipFromGpu(Graphics::CommandBuffer* buffer, int handle, int index,
                                   int flip_mode, int64_t flip_arg, uint64_t* request_id) {
	EXIT_IF(VideoOut::g_video_out_context == nullptr || Graphics::g_render_ctx == nullptr ||
	        buffer == nullptr || request_id == nullptr);

	const int result = VideoOut::ReserveFlipRequest(
	    handle, index, flip_mode, flip_arg, VideoOut::FlipRequestSource::GpuEop, request_id);
	if (result != OK) {
		return result;
	}
	VideoOut::FlipStats::submit_gpu.fetch_add(1, std::memory_order_relaxed);
	LOGF("FlipTrace: SubmitFlipGPU handle=%d index=%d mode=%d arg=%" PRId64 " id=%" PRIu64 "\n",
	     handle, index, flip_mode, flip_arg, *request_id);
	VideoOut::FlipStats::Log("submit_gpu");
	VideoOut::Phase37NoteGuestSubmitFlip(index);
	VideoOut::g_video_out_context->GetFlipQueue().Prepare(*request_id, buffer);

	return OK;
}

uint64_t DisplayBufferPrepareNextFlipOnGpu(Graphics::CommandBuffer* buffer) {
	EXIT_IF(VideoOut::g_video_out_context == nullptr || Graphics::g_render_ctx == nullptr);
	return VideoOut::g_video_out_context->GetFlipQueue().PrepareNextCpu(buffer);
}

void DisplayBufferCompleteFlipFromGpu(uint64_t request_id) {
	EXIT_IF(VideoOut::g_video_out_context == nullptr || Graphics::g_render_ctx == nullptr);
	VideoOut::g_video_out_context->GetFlipQueue().Complete(request_id);
}

void DisplayBufferWaitForFlipQueueSlot() {
	EXIT_IF(VideoOut::g_video_out_context == nullptr || Graphics::g_render_ctx == nullptr);
	VideoOut::g_video_out_context->GetFlipQueue().WaitForSubmitSlot();
}

} // namespace Libs::Presentation

namespace Libs::VideoOut {

void VideoOutWaitFlipDone(int handle, int index) {
	EXIT_IF(g_video_out_context == nullptr);

	auto* ctx = g_video_out_context->Get(handle);
	EXIT_IF(ctx == nullptr);

	EXIT_NOT_IMPLEMENTED(!IsValidBufferIndex(index));
	g_video_out_context->GetFlipQueue().Wait(ctx, index);
}

KYTY_SYSV_ABI int VideoOutGetFlipStatus(int handle, VideoOutFlipStatus* status) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	if (status == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	g_video_out_context->GetFlipQueue().GetFlipStatus(ctx, status);

	const auto presented = FlipStats::presented.load(std::memory_order_relaxed);
	LOGF("\t count = %" PRIu64 "\n"
	     "\t processTime = %" PRIu64 "\n"
	     "\t tsc = %" PRIu64 "\n"
	     "\t submitTsc = %" PRIu64 "\n"
	     "\t flipArg = %" PRId64 "\n"
	     "\t gcQueueNum = %d\n"
	     "\t flipPendingNum = %d\n"
	     "\t currentBuffer = %d\n",
	     status->count, status->processTime, status->tsc, status->submitTsc, status->flipArg,
	     status->gcQueueNum, status->flipPendingNum, status->currentBuffer);
	if (presented > 0) {
		static std::atomic<uint32_t> post_present_logs {0};
		if (post_present_logs.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("FlipTrace: GetFlipStatus post_present pending=%d currentBuffer=%d flipArg=%" PRId64
			     "\n",
			     status->flipPendingNum, status->currentBuffer, status->flipArg);
			fprintf(stderr,
			        "FlipTrace: GetFlipStatus pending=%d currentBuffer=%d\n",
			        status->flipPendingNum, status->currentBuffer);
		}
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutIsFlipPending(int handle) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	// Phase 32 grace: hold pending=1 until FlipHandler consumed Flip EQ AND N polls elapsed.
	const uint64_t need_flip = g_pending0_need_flip_consumed.load(std::memory_order_acquire);
	if (need_flip != 0) {
		const uint64_t got = g_flip_eq_consumed.load(std::memory_order_acquire);
		if (got < need_flip) {
			LOGF("\t flipPendingNum = 1 (phase32 wait Flip EQ consumed got=%" PRIu64
			     " need=%" PRIu64 ")\n",
			     got, need_flip);
			return 1;
		}
	}
	const int grace = g_pending0_grace_polls.load(std::memory_order_relaxed);
	if (grace > 0) {
		const int left = g_pending0_grace_polls.fetch_sub(1, std::memory_order_acq_rel);
		if (left > 1) {
			LOGF("\t flipPendingNum = 1 (phase32 grace left=%d)\n", left - 1);
			return 1;
		}
		// Last grace poll → clear pending for real (keep currentBuffer from last present).
		g_pending0_need_flip_consumed.store(0, std::memory_order_release);
		g_video_out_context->GetFlipQueue().ClearPendingPhase32(ctx);
		LOGF("FlipTrace: phase32 grace exhausted → pending cleared (Flip EQ was consumed)\n");
		fprintf(stderr, "FlipTrace: phase32 grace exhausted → pending cleared\n");
		Common::NoteHleCall("VideoOut", "VideoOut", "HoldGatePhaseC");
		Common::FlushHleRingToFatal("post_hold_gate_phaseC");
		const auto woken = LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
		LOGF("SubmitTrace: WakeSubmissionCond after pending0 grace woken=%zu\n", woken);
		fprintf(stderr, "SubmitTrace: WakeSubmissionCond after pending0 grace woken=%zu\n", woken);
	}

	VideoOutFlipStatus status {};
	g_video_out_context->GetFlipQueue().GetFlipStatus(ctx, &status);

	LOGF("\t flipPendingNum = %d\n", status.flipPendingNum);
	const auto presented = FlipStats::presented.load(std::memory_order_relaxed);
	if (presented > 0) {
		static std::atomic<uint32_t> pending_logs {0};
		if (pending_logs.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("FlipTrace: IsFlipPending post_present pending=%d currentBuffer=%d\n",
			     status.flipPendingNum, status.currentBuffer);
			fprintf(stderr, "FlipTrace: IsFlipPending pending=%d currentBuffer=%d\n",
			        status.flipPendingNum, status.currentBuffer);
		}
		if (status.flipPendingNum == 0) {
			static std::atomic<bool> first_clear {false};
			if (!first_clear.exchange(true, std::memory_order_acq_rel)) {
				void* ra0 = __builtin_return_address(0);
#if defined(_MSC_VER) || defined(__clang__)
				void*  ret_slot = _AddressOfReturnAddress();
				auto*  stack    = reinterpret_cast<uint64_t*>(ret_slot);
#else
				auto* stack = reinterpret_cast<uint64_t*>(__builtin_frame_address(0));
#endif
				LOGF("FlipTrace: IsFlipPending first return 0 after present (MainThread may proceed) "
				     "ra0=%p\n",
				     ra0);
				fprintf(stderr, "FlipTrace: IsFlipPending first return 0 after present ra0=%p\n",
				        ra0);
				char status_msg[256];
				std::snprintf(status_msg, sizeof(status_msg),
				              "FlipStatus@firstZero count=%" PRIu64 " pending=%d currentBuffer=%d "
				              "flipArg=%" PRId64,
				              status.count, status.flipPendingNum, status.currentBuffer,
				              status.flipArg);
				LOGF("FlipTrace: %s\n", status_msg);
				fprintf(stderr, "FlipTrace: %s\n", status_msg);
				Common::LogFatalToFile(status_msg);
				Common::NoteHleCall("VideoOut", "VideoOut", "IsFlipPendingFirstZero");
				g_pending0_seen.store(true, std::memory_order_release);
				g_pending0_watchdog_ticks.store(120, std::memory_order_release);
				Common::EnableGuestAccessViolationLogging(true);
				Graphics::WindowArmIgnoreQuit(90);
				Common::FlushHleRingToFatal("post_pending0_first");
				(void)stack;
				(void)ra0;
			}
		}
	}
	if (status.flipPendingNum != 0 && Graphics::boot_trace_log(64)) {
		LOGF("FlipTrace: IsFlipPending handle=%d pending=%d (see FlipTrace stats for reserve)\n",
		     handle, status.flipPendingNum);
		FlipStats::Log("is_flip_pending");
	}

	return status.flipPendingNum;
}

KYTY_SYSV_ABI int VideoOutGetVblankStatus(int handle, VideoOutVblankStatus* status) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	if (status == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	ctx->mutex.Lock();
	*status = ctx->vblank_status;
	ctx->mutex.Unlock();

	LOGF("\t count = %" PRIu64 "\n"
	     "\t processTime = %" PRIu64 "\n"
	     "\t processTimeCounter = %" PRIu64 "\n",
	     status->count, status->processTime, status->processTimeCounter);

	return OK;
}

KYTY_SYSV_ABI int VideoOutGetEventId(const EventQueue::KernelEvent* ev) {
	PRINT_NAME();

	if (ev == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (ev->filter != EventQueue::KERNEL_EVFILT_VIDEO_OUT) {
		return VIDEO_OUT_ERROR_INVALID_EVENT;
	}

	switch (ev->ident) {
		case VIDEO_OUT_EVENT_FLIP:
		case VIDEO_OUT_EVENT_VBLANK:
		case VIDEO_OUT_EVENT_PRE_VBLANK_START:
		case VIDEO_OUT_EVENT_SET_MODE: return static_cast<int>(ev->ident);
		default: return VIDEO_OUT_ERROR_INVALID_EVENT;
	}
}

KYTY_SYSV_ABI int VideoOutGetEventData(const EventQueue::KernelEvent* ev, int64_t* data) {
	PRINT_NAME();

	if (ev == nullptr || data == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (ev->filter != EventQueue::KERNEL_EVFILT_VIDEO_OUT) {
		return VIDEO_OUT_ERROR_INVALID_EVENT;
	}

	uint64_t event_data = static_cast<uint64_t>(ev->data) >> 16u;
	if (ev->ident == VIDEO_OUT_EVENT_FLIP &&
	    (static_cast<uint64_t>(ev->data) & 0x8000000000000000ULL) != 0) {
		event_data |= 0xffff000000000000ULL;
	}

	*data = static_cast<int64_t>(event_data);

	return OK;
}

KYTY_SYSV_ABI int VideoOutGetEventCount(const EventQueue::KernelEvent* ev) {
	PRINT_NAME();

	if (ev == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (ev->filter != EventQueue::KERNEL_EVFILT_VIDEO_OUT) {
		return VIDEO_OUT_ERROR_INVALID_EVENT;
	}

	return static_cast<int>((static_cast<uint64_t>(ev->data) >> 12u) & 0xfu);
}

KYTY_SYSV_ABI int VideoOutWaitVblank(int handle) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	[[maybe_unused]] auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	WaitForNextVblank();
	g_video_out_context->VblankEnd();

	return OK;
}

KYTY_SYSV_ABI int VideoOutGetOutputStatus(int handle, VideoOutOutputStatus* status) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	if (status == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	ctx->mutex.Lock();
	status->resolution   = (ctx->width >= 3840 || ctx->height >= 2160 ? 2u : 1u);
	status->dynamicRange = 1;
	status->refreshRate =
	    (ctx->output_mode == VIDEO_OUT_OUTPUT_MODE_119_88HZ || Config::GetVblankFrequency() >= 119
	         ? VIDEO_OUT_REFRESH_RATE_119_88HZ
	         : VIDEO_OUT_REFRESH_RATE_59_94HZ);
	status->flags       = 0;
	status->reserved[0] = 0;
	status->reserved[1] = 0;
	status->reserved[2] = 0;
	ctx->mutex.Unlock();

	return OK;
}

static int ValidateOutputConfig(int handle, uint64_t mode, const VideoOutOutputOptions* options,
                                void* reserved_ptr, uint64_t reserved) {
	EXIT_IF(g_video_out_context == nullptr);

	if (!g_video_out_context->IsOpened(handle)) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (reserved_ptr != nullptr || reserved != 0) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	if (options != nullptr) {
		for (auto v: options->internalData) {
			if (v != 0) {
				return VIDEO_OUT_ERROR_INVALID_OPTION;
			}
		}
	}

	if (mode != VIDEO_OUT_OUTPUT_MODE_DEFAULT && mode != VIDEO_OUT_OUTPUT_MODE_119_88HZ) {
		return VIDEO_OUT_ERROR_UNSUPPORTED_OUTPUT_MODE;
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutInitializeOutputOptions(VideoOutOutputOptions* options) {
	PRINT_NAME();

	if (options == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	memset(options, 0, sizeof(VideoOutOutputOptions));

	return OK;
}

KYTY_SYSV_ABI int VideoOutIsOutputSupported(int handle, uint64_t mode,
                                            const VideoOutOutputOptions* options,
                                            void* reserved_ptr, uint64_t reserved) {
	PRINT_NAME();

	LOGF("\t mode = 0x%016" PRIx64 "\n", mode);

	int result = ValidateOutputConfig(handle, mode, options, reserved_ptr, reserved);
	if (result != OK) {
		return result;
	}

	if (mode == VIDEO_OUT_OUTPUT_MODE_119_88HZ) {
		return (Config::GetVblankFrequency() >= 119 ? VIDEO_OUT_TRUE : VIDEO_OUT_FALSE);
	}

	return VIDEO_OUT_TRUE;
}

KYTY_SYSV_ABI int VideoOutConfigureOutput(int handle, uint64_t mode,
                                          const VideoOutOutputOptions* options, void* reserved_ptr,
                                          uint64_t reserved) {
	PRINT_NAME();

	LOGF("\t mode = 0x%016" PRIx64 "\n", mode);

	int result = VideoOutIsOutputSupported(handle, mode, options, reserved_ptr, reserved);
	if (result < 0) {
		return result;
	}
	if (result == VIDEO_OUT_FALSE) {
		return VIDEO_OUT_ERROR_UNAVAILABLE_OUTPUT_MODE;
	}

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	ctx->mutex.Lock();
	ctx->output_mode = mode;
	TriggerVideoOutEventsLocked(ctx->output_mode_eqs, VideoOutEventKind::OutputMode,
	                            reinterpret_cast<void*>(ctx->output_mode));
	ctx->mutex.Unlock();

	return OK;
}

KYTY_SYSV_ABI int VideoOutSetWindowModeMargins(int handle, int top, int bottom) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	[[maybe_unused]] auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	LOGF("\t top    = %d\n"
	     "\t bottom = %d\n",
	     top, bottom);

	return OK;
}

KYTY_SYSV_ABI int VideoOutLatencyControlWaitBeforeInput(int handle) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	[[maybe_unused]] auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutLatencyMeasureSetStartPoint(int handle, uint32_t point) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	[[maybe_unused]] auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	LOGF("\t point = %" PRIu32 "\n", point);

	return OK;
}

KYTY_SYSV_ABI int VideoOutColorSettingsSetGamma(VideoOutColorSettings* settings, float gamma) {
	PRINT_NAME();

	if (settings == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (gamma < 0.1f || gamma > 2.0f) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	settings->gamma = gamma;
	return OK;
}

KYTY_SYSV_ABI int VideoOutAdjustColor(int handle, const VideoOutColorSettings* settings) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	if (settings == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (!g_video_out_context->IsOpened(handle)) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	ctx->mutex.Lock();
	ctx->gamma = settings->gamma;
	ctx->mutex.Unlock();

	return OK;
}

KYTY_SYSV_ABI int VideoOutAddVrrStatusFlagsPrivilege(int handle, uint32_t flags, uint64_t arg2,
                                                      uint64_t arg3) {
	PRINT_NAME();

	// Always OK — matches Phase 18 soft-stub. Args logged only; ABI still uncertain.
	LOGF("\t VRR FlagsPrivilege stub a0=%d a1=0x%08" PRIx32 " a2=0x%016" PRIx64
	     " a3=0x%016" PRIx64 "\n",
	     handle, flags, arg2, arg3);
	return OK;
}

KYTY_SYSV_ABI int VideoOutGetVrrStatus(int handle, void* status) {
	PRINT_NAME();

	const auto status_addr = reinterpret_cast<uintptr_t>(status);
	LOGF("\t VRR GetVrrStatus stub a0=%d a1=0x%016" PRIx64 "\n", handle,
	     static_cast<uint64_t>(status_addr));
	// Soft-stub behavior: never fault. Only zero-fill if a1 looks like a real pointer
	// (Phase 19 AV: Write @0x31 after memset on status=0x1 from wrong ABI).
	constexpr uintptr_t kMinGuestPtr = 0x10000;
	if (status_addr >= kMinGuestPtr) {
		std::memset(status, 0, sizeof(VideoOutVrrStatus));
	}
	return OK;
}

KYTY_SYSV_ABI int VideoOutGetBufferLabelAddress(int handle, uintptr_t* label_addr) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	if (label_addr == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	*label_addr = reinterpret_cast<uintptr_t>(ctx->buffer_labels);
	LOGF("FlipTrace: GetBufferLabelAddress handle=%d labels=%p count=%d\n", handle,
	     static_cast<void*>(ctx->buffer_labels), VIDEO_OUT_BUFFER_NUM_MAX);
	fprintf(stderr, "FlipTrace: GetBufferLabelAddress handle=%d\n", handle);
	// Sony returns the number of label slots (16).
	return VIDEO_OUT_BUFFER_NUM_MAX;
}

KYTY_SYSV_ABI int VideoOutGetDeviceCapabilityInfo(int handle, void* info) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	if (info == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}
	if (!g_video_out_context->IsOpened(handle)) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	// SceVideoOutDeviceCapabilityInfo — capability flags dword at offset 0.
	std::memset(info, 0, 16);
	LOGF("FlipTrace: GetDeviceCapabilityInfo handle=%d (stub caps=0)\n", handle);
	return OK;
}

KYTY_SYSV_ABI int VideoOutGetResolutionStatus(int handle, void* status) {
	PRINT_NAME();

	EXIT_IF(g_video_out_context == nullptr);

	if (status == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = g_video_out_context->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	// Minimal PS4/PS5 resolution status: full/pane width/height as uint32s.
	struct ResolutionStatus {
		uint32_t full_width  = 0;
		uint32_t full_height = 0;
		uint32_t pane_width  = 0;
		uint32_t pane_height = 0;
		uint32_t pad[4]      = {};
	};
	auto* out        = static_cast<ResolutionStatus*>(status);
	*out             = ResolutionStatus {};
	out->full_width  = ctx->width;
	out->full_height = ctx->height;
	out->pane_width  = ctx->width;
	out->pane_height = ctx->height;
	LOGF("FlipTrace: GetResolutionStatus handle=%d %ux%u\n", handle, ctx->width, ctx->height);
	return OK;
}

// Unresolved TLOU VideoOut NIDs (T4ucGB8CsnM / 5tRaBjtdTzY) — soft stubs that log and return OK.
KYTY_SYSV_ABI int VideoOutUnknownNidStub0(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
	PRINT_NAME();
	LOGF("FlipTrace: VideoOutUnknownNidStub0 a0=0x%016" PRIx64 " a1=0x%016" PRIx64
	     " a2=0x%016" PRIx64 " a3=0x%016" PRIx64 "\n",
	     a0, a1, a2, a3);
	fprintf(stderr, "FlipTrace: VideoOutUnknownNidStub0 called\n");
	Common::NoteHleCall("VideoOut", "VideoOut", "UnknownNidStub0");
	return OK;
}

KYTY_SYSV_ABI int VideoOutUnknownNidStub1(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
	PRINT_NAME();
	LOGF("FlipTrace: VideoOutUnknownNidStub1 a0=0x%016" PRIx64 " a1=0x%016" PRIx64
	     " a2=0x%016" PRIx64 " a3=0x%016" PRIx64 "\n",
	     a0, a1, a2, a3);
	fprintf(stderr, "FlipTrace: VideoOutUnknownNidStub1 called\n");
	Common::NoteHleCall("VideoOut", "VideoOut", "UnknownNidStub1");
	return OK;
}

} // namespace Libs::VideoOut
