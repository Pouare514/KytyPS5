#include "graphics/presentation/videoOut.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/fatalLog.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/renderer/textureCache.h"
#include "graphics/host_gpu/guestImageWriteTracker.h"
#include "graphics/presentation/displayBuffer.h"
#include "graphics/presentation/window.h"
#include "common/crashDiagnostics.h"
#include "common/fatalLog.h"
#include "common/singleton.h"
#include "kernel/pthread.h"
#include "kernel/eventQueue.h"
#include "kernel/semaphore.h"
#include "libs/errno.h"
#include "libs/agc.h"
#include "libs/libs.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"
#include "loader/x64InstructionEmulator.h"

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
// Phase 45: NdJob-path submit_dcb after Unregister (timeout waiter / post-baseline).
static std::atomic<bool>     g_phase45_ndjob_dcb_ok {false};
static std::atomic<bool>     g_phase45_timeout_submit_done {false};
static std::atomic<uint32_t> g_phase45_timeout_submit_attempts {0};
// Phase 46: real non-NOP DCB + native PM4 EOP (not host_force / not P45 NOP).
static std::atomic<bool>     g_phase46_real_dcb_ok {false};
static std::atomic<bool>     g_phase46_native_eop_ok {false};
static std::atomic<uint32_t> g_phase46_dcb_dump_n {0};
static const uint32_t*       g_phase46_eop_seed_dcb {nullptr};
// Phase 47: guest DRAW/DISPATCH DCB post-Unregister (not P46 EOP-only seed).
static std::atomic<bool>     g_phase47_guest_draw_ok {false};
static std::atomic<bool>     g_phase47_guest_dispatch_ok {false};
// Phase 48: present content classification (L0 blank vs L2 non-zero).
static std::atomic<bool>     g_phase48_content_nonzero {false};
static std::atomic<bool>     g_phase48_content_zero {false};
static std::atomic<bool>     g_phase48_float_flip {false};
// Phase 49: VideoOut pipeline gates for keep[1] live (must precede NdJob).
static std::atomic<bool>     g_phase49_vo_vk_ok {false};
static std::atomic<bool>     g_phase49_submit_accepted {false};
static std::atomic<uint32_t> g_phase49_submit_ok_n {0};
static std::atomic<uint32_t> g_phase49_submit_full_n {0};
// Phase 50: *obj mutation / poll tracking (logging outside keep[1] trampoline).
static std::atomic<uint64_t> g_phase50_last_head {0};
static std::atomic<bool>     g_phase50_obj_nonzero {false};
static std::atomic<bool>     g_phase50_poll_thread_started {false};
static std::atomic<uint32_t> g_phase47_post_seed_wake_n {0};
// Phase 51: 256 B NdJob struct snapshot + bypass / failfast one-shots.
constexpr size_t             kPhase51DumpBytes = 256;
static uint8_t               g_phase51_prev_dump[kPhase51DumpBytes] {};
static bool                  g_phase51_have_prev_dump = false;
static std::atomic<uint32_t> g_phase51_dump_n {0};
static std::atomic<bool>     g_phase51_bypass_done {false};
static std::atomic<bool>     g_phase51_failfast_fired {false};
// Phase 52: falsify/remap keep[1] head — audit, follow, seed, time-series.
constexpr uint64_t           kPhase52GuestLo         = 0x0000000900000000ULL;
constexpr uint64_t           kPhase52GuestHi         = 0x0000000910000000ULL;
constexpr uint64_t           kPhase52UserLo          = 0x0000001000000000ULL;
constexpr uint64_t           kPhase52UserHi          = 0x0000001800000000ULL;
constexpr uint32_t           kPhase52MaxSeedAttempts = 3;
constexpr size_t             kPhase52FollowBytes     = 128;
static std::atomic<uint32_t> g_phase52_write_audit_n {0};
static std::atomic<uint32_t> g_phase52_follow_n {0};
static std::atomic<uint32_t> g_phase52_head_attempts {0};
static std::atomic<uint64_t> g_phase52_best_candidate {0};
static std::atomic<bool>     g_phase52_candidate_strong {false};
static std::atomic<bool>     g_phase52_give_up {false};
static std::atomic<bool>     g_phase52_rearm_clean_logged {false};
static std::atomic<bool>     g_phase52_ts_started {false};
static std::atomic<uint64_t> g_phase52_submit_gpu_mirror {0};
// Phase 53: retarget — worker/fiber trampolines + deep scan (no *obj seed).
constexpr uint32_t           kPhase53StealBytes = 12;
constexpr uint32_t           kPhase53DeepEnd    = 0x21000;
static std::atomic<uint64_t> g_phase53_real_queue {0};
static std::atomic<uint32_t> g_phase53_deep_n {0};
static std::atomic<uint32_t> g_phase53_deep_hit_n {0};
static std::atomic<bool>     g_phase53_user_dumped {false};
static std::atomic<bool>     g_phase53_status_dumped {false};
static std::atomic<bool>     g_phase53_traces_armed {false};
static uint8_t*              g_phase53_worker_thunk = nullptr;
static uint8_t*              g_phase53_worker_live  = nullptr;
static uint8_t*              g_phase53_fiber_thunk  = nullptr;
static uint8_t*              g_phase53_fiber_live   = nullptr;
static std::atomic<uint64_t> g_phase53_worker_rdi {0};
static std::atomic<uint64_t> g_phase53_worker_rsi {0};
static std::atomic<uint64_t> g_phase53_worker_rdx {0};
static std::atomic<uint64_t> g_phase53_worker_hits {0};
static std::atomic<int>      g_phase53_worker_pending {0};
static std::atomic<uint32_t> g_phase53_worker_eax {0};
static uint8_t               g_phase53_worker_snap[64] {};
static std::atomic<uint64_t> g_phase53_fiber_rdi {0};
static std::atomic<uint64_t> g_phase53_fiber_rsi {0};
static std::atomic<uint64_t> g_phase53_fiber_rdx {0};
static std::atomic<uint64_t> g_phase53_fiber_hits {0};
static std::atomic<int>      g_phase53_fiber_pending {0};
static std::atomic<uint32_t> g_phase53_fiber_eax {0};
static uint8_t               g_phase53_fiber_snap[64] {};
// Phase 54: producer Mixed enqueue timeline / wake budget / fake job.
static std::atomic<uint64_t> g_phase54_cycle_id {0};
static std::atomic<uint64_t> g_phase54_budget_cycle {0};
static std::atomic<uint32_t> g_phase54_host_wake_n {0};
static std::atomic<bool>     g_phase54_cycle_guest_real {false};
static std::atomic<bool>     g_phase54_cycle_mixed_leave {false};
static std::atomic<uint32_t> g_phase54_mixed_wake_n {0};
static std::atomic<uint32_t> g_phase54_rewait_n {0};
static std::atomic<bool>     g_phase54_fake_job_done {false};
// Phase 55: Mixed entry / queue layout / watch / fake inject.
static std::atomic<bool>     g_phase55_traces_armed {false};
static uint8_t*              g_phase55_mixed_thunk = nullptr;
static uint8_t*              g_phase55_mixed_live  = nullptr;
static std::atomic<uint64_t> g_phase55_entry_hits {0};
static std::atomic<uint64_t> g_phase55_mixed_rdi {0};
static std::atomic<uint64_t> g_phase55_mixed_rsi {0};
static std::atomic<uint64_t> g_phase55_mixed_rdx {0};
static std::atomic<uint32_t> g_phase55_mixed_eax {0};
static std::atomic<int>      g_phase55_mixed_pending {0}; // 1=enter 2=exit
static std::atomic<bool>     g_phase55_first_hit_after_unreg {false};
static std::atomic<bool>     g_phase55_role_logged {false};
static uint8_t               g_phase55_mixed_snap[64] {};
static std::atomic<uint64_t> g_phase55_queue_base {0};
static std::atomic<uint32_t> g_phase55_dump_runs {0};
static std::atomic<uint32_t> g_phase55_main_wake_alt_n {0};
static std::atomic<uint32_t> g_phase55_enqueue_write_n {0};
static std::atomic<uint32_t> g_phase55_submit_guest_real_n {0};
static std::atomic<int>      g_phase55_count_off {-1};
static std::atomic<int>      g_phase55_head_off {-1};
static std::atomic<int>      g_phase55_tail_off {-1};
static std::atomic<uint64_t> g_phase55_watch_count {0};
static std::atomic<uint64_t> g_phase55_watch_head {0};
static std::atomic<uint64_t> g_phase55_watch_tail {0};
static std::atomic<bool>     g_phase55_watch_armed {false};
static std::atomic<bool>     g_phase55_watch_started {false};
static std::atomic<bool>     g_phase55_fake_queue_done {false};
static std::atomic<bool>     g_phase55_heatmap_done {false};
static char                  g_phase55_last_verdict[32] = "unknown";
static uint8_t               g_phase55_prev_dump[256] {};
static bool                  g_phase55_have_prev_dump = false;
// Phase 56: LIST_CANDIDATE retarget / writers / FAKE_COUNT.
constexpr uint64_t           kPhase56BannedBase = 0x0000000905f25cd0ULL;
static std::atomic<uint64_t> g_phase56_sync_id {0};
static std::atomic<uint64_t> g_phase56_queue_base {0};
static std::atomic<uint64_t> g_phase56_job_cond {0};
static std::atomic<uint64_t> g_phase56_job_mutex {0};
static std::atomic<uint32_t> g_phase56_dump_runs {0};
static std::atomic<uint32_t> g_phase56_enqueue_write_n {0};
static std::atomic<uint32_t> g_phase56_enqueue_from_main {0};
static std::atomic<uint32_t> g_phase56_enqueue_from_other {0};
static std::atomic<uint32_t> g_phase56_list_cand_n {0};
static std::atomic<int>      g_phase56_count_off {-1};
static std::atomic<uint64_t> g_phase56_watch_count {0};
static std::atomic<uint64_t> g_phase56_watch_q0 {0};
static std::atomic<uint64_t> g_phase56_watch_q8 {0};
static std::atomic<bool>     g_phase56_watch_armed {false};
static std::atomic<bool>     g_phase56_lea_scanned {false};
static std::atomic<bool>     g_phase56_fake_count_done {false};
static std::atomic<bool>     g_phase56_heatmap_done {false};
static std::atomic<bool>     g_phase56_base_elected {false};
static std::atomic<int>      g_phase56_best_score {-1};
static std::atomic<bool>     g_phase56_watch_seeded {false};
static std::atomic<uint32_t> g_phase56_main_signal_job {0};
static std::atomic<uint32_t> g_phase56_main_signal_alt {0};
static char                  g_phase56_base_type[24] = "none";
static char                  g_phase56_cause[40]     = "unknown";
static uint64_t              g_phase56_lea_cand[2]   = {0, 0};
static int                   g_phase56_lea_cand_n    = 0;
// Phase 57: global queue / Main producer (A+C).
static constexpr size_t      kPhase57HeatSlots       = 32;
static constexpr uint64_t    kPhase57StabHitsNeed    = 3;
struct Phase57HeatSlot {
	std::atomic<uint64_t> va {0};
	std::atomic<uint32_t> read_n {0};
	std::atomic<uint32_t> write_n {0};
	std::atomic<uint32_t> main_n {0};
	std::atomic<uint32_t> mixed_n {0};
	std::atomic<uint32_t> other_n {0};
	std::atomic<uint32_t> main_touch {0};
	std::atomic<int>      score {0};
	char                  type[24] {};
	uint8_t               prev[64] {};
	bool                  have_prev = false;
};
static Phase57HeatSlot       g_phase57_heat[kPhase57HeatSlots] {};
static std::atomic<uint32_t> g_phase57_heat_used {0};
static std::atomic<uint64_t> g_phase57_queue_base {0};
static std::atomic<uint64_t> g_phase57_stable_ctx {0};
static std::atomic<uint64_t> g_phase57_rdi_hist {0};
static std::atomic<uint64_t> g_phase57_rsi_hist {0};
static std::atomic<uint64_t> g_phase57_rdx_hist {0};
static std::atomic<uint32_t> g_phase57_rdi_hits {0};
static std::atomic<uint32_t> g_phase57_rsi_hits {0};
static std::atomic<uint32_t> g_phase57_rdx_hits {0};
static std::atomic<uint32_t> g_phase57_scan_runs {0};
static std::atomic<uint32_t> g_phase57_cand_n {0};
static std::atomic<uint32_t> g_phase57_list_n {0};
static std::atomic<uint32_t> g_phase57_main_agc_n {0};
static std::atomic<uint32_t> g_phase57_main_write_n {0};
static std::atomic<uint32_t> g_phase57_main_touch_cand {0};
static std::atomic<bool>     g_phase57_body_scanned {false};
static std::atomic<bool>     g_phase57_heatmap_done {false};
static std::atomic<bool>     g_phase57_elected {false};
static std::atomic<int>      g_phase57_best_score {-1};
static char                  g_phase57_cause[40]     = "unknown";
static char                  g_phase57_base_type[24] = "none";
// Phase 58: NdJob ancre + subblock classify + discriminant heatmap.
static constexpr size_t      kPhase58HeatSlots       = 32;
static constexpr uint64_t    kPhase58StatusDefault   = 0x000000090386ed80ULL;
static constexpr uint64_t    kPhase58CtxLo           = 0x0000000903400000ULL;
static constexpr uint64_t    kPhase58CtxHi           = 0x0000000903900000ULL;
static constexpr uint32_t    kPhase58StabHitsNeed    = 3;
struct Phase58HeatSlot {
	std::atomic<uint64_t> va {0};
	std::atomic<uint32_t> read_n {0};
	std::atomic<uint32_t> write_n {0};
	std::atomic<uint32_t> main_n {0};
	std::atomic<uint32_t> other_n {0};
	std::atomic<uint32_t> main_touch {0};
	char                  type[24] {};
	uint8_t               prev[64] {};
	bool                  have_prev = false;
};
static Phase58HeatSlot       g_phase58_heat[kPhase58HeatSlots] {};
static std::atomic<uint32_t> g_phase58_heat_used {0};
static std::atomic<uint64_t> g_phase58_queue_base {0};
static std::atomic<uint64_t> g_phase58_ndjob_obj {0};
static std::atomic<uint64_t> g_phase58_status {0};
static std::atomic<uint64_t> g_phase58_stable_ctx {0};
static std::atomic<uint32_t> g_phase58_cand_n {0};
static std::atomic<uint32_t> g_phase58_list_n {0};
static std::atomic<uint32_t> g_phase58_sub_list_n {0};
static std::atomic<uint32_t> g_phase58_buffer_n {0};
static std::atomic<uint32_t> g_phase58_hop_reject_n {0};
static std::atomic<uint32_t> g_phase58_main_touch_n {0};
static std::atomic<uint32_t> g_phase58_ancre_runs {0};
static std::atomic<bool>     g_phase58_elected {false};
static std::atomic<bool>     g_phase58_heatmap_done {false};
static std::atomic<bool>     g_phase58_saw_mutation {false};
static std::atomic<bool>     g_phase58_subblocks_done {false};
static std::atomic<int>      g_phase58_best_score {-1};
static char                  g_phase58_cause[48]     = "unknown";
static char                  g_phase58_base_type[24] = "none";
static std::atomic<uint64_t> g_phase58_w_rdi_hist {0};
static std::atomic<uint64_t> g_phase58_w_rsi_hist {0};
static std::atomic<uint32_t> g_phase58_w_rdi_hits {0};
static std::atomic<uint32_t> g_phase58_w_rsi_hits {0};
static std::atomic<uint64_t> g_phase58_f_rdi_hist {0};
static std::atomic<uint32_t> g_phase58_f_rdi_hits {0};
// Phase 59: AGC context/queue/stream guest↔host map.
static constexpr size_t   kPhase59MapSlots          = 48;
static constexpr size_t   kPhase59StreamSlots       = 16;
static constexpr uint64_t kPhase59UserRingDefault   = 0x0000001000000000ULL;
static constexpr uint64_t kPhase59UserRingAlt       = 0x0000001002000000ULL;
static constexpr uint64_t kPhase59StatusDefault     = 0x000000090386ed80ULL;
static constexpr uint64_t kPhase59CtxBandLo         = 0x0000000903400000ULL;
static constexpr uint64_t kPhase59CtxBandHi         = 0x0000000903900000ULL;
struct Phase59MapSlot {
	std::atomic<uint64_t> guest_va {0};
	std::atomic<uint64_t> host_id {0};
	std::atomic<uint32_t> submit_n {0};
	std::atomic<uint32_t> main_n {0};
	std::atomic<uint64_t> last_tsc {0};
	char                  kind[24] {};
	char                  owner_role[12] {};
	char                  first_seen_epoch[12] {};
	char                  source_tag[48] {};
};
struct Phase59StreamSlot {
	std::atomic<uint32_t> stream_id {0};
	std::atomic<uint64_t> stream_va {0};
	std::atomic<uint64_t> ctx_va {0};
	std::atomic<uint64_t> queue_va {0};
	std::atomic<uint32_t> active_n {0};
};
static Phase59MapSlot        g_phase59_map[kPhase59MapSlots] {};
static std::atomic<uint32_t> g_phase59_map_used {0};
static Phase59StreamSlot     g_phase59_streams[kPhase59StreamSlots] {};
static std::atomic<uint32_t> g_phase59_stream_used {0};
static std::atomic<uint32_t> g_phase59_submit_n {0};
static std::atomic<uint32_t> g_phase59_submit_mapped_n {0};
static std::atomic<uint32_t> g_phase59_orphan_n {0};
static std::atomic<uint32_t> g_phase59_guest_real_n {0};
static std::atomic<uint32_t> g_phase59_eq_n {0};
static std::atomic<uint32_t> g_phase59_ring_dump_n {0};
static std::atomic<bool>     g_phase59_heatmap_done {false};
static std::atomic<bool>     g_phase59_seeded {false};
static char                  g_phase59_cause[40] = "unknown";
// Phase 61: ring probe unlock gate.
static constexpr size_t   kPhase61RingBytes = 1024;
static constexpr uint32_t kPhase61Pm4M      = 2;  // min pm4 hits per rich probe
static constexpr uint32_t kPhase61RichY     = 3;  // rich probes needed
static constexpr uint32_t kPhase61Pm4X      = 6;  // cumulative pm4 hits
static constexpr uint32_t kPhase61MutNeed   = 2;  // mutating probes needed
static std::atomic<uint32_t> g_phase61_probe_n {0};
static std::atomic<uint32_t> g_phase61_pm4_total {0};
static std::atomic<uint32_t> g_phase61_ptr_total {0};
static std::atomic<uint32_t> g_phase61_rich_n {0};
static std::atomic<uint32_t> g_phase61_mut_n {0};
static std::atomic<uint32_t> g_phase61_pre_agc_n {0};
static std::atomic<uint32_t> g_phase61_post_guest_real_n {0};
static std::atomic<bool>     g_phase61_heatmap_done {false};
static std::atomic<bool>     g_phase61_have_prev {false};
static uint8_t               g_phase61_prev[kPhase61RingBytes] {};
static char                  g_phase61_cause[40] = "unknown";
// Phase 62: Unreg silence / RegisterResource map (KPRI ring abandoned as submit gate).
static constexpr size_t   kPhase62ResSlots     = 64;
static constexpr size_t   kPhase62ProbeBytes   = 256;
static constexpr uint32_t kPhase62PreAgcNeed   = 8;
static constexpr uint32_t kPhase62AltPm4Need   = 2;
static constexpr uint32_t kPhase62AltMutNeed   = 1;
struct Phase62ResSlot {
	std::atomic<uint64_t> va {0};
	std::atomic<uint64_t> size {0};
	std::atomic<uint32_t> handle {0};
	std::atomic<int32_t>  res_type {0};
	std::atomic<bool>     is_kpri {false};
	std::atomic<bool>     have_prev {false};
	std::atomic<uint32_t> mut_n {0};
	std::atomic<uint32_t> pm4_hits {0};
	std::atomic<uint32_t> ptr_hits {0};
	char                  name[48] {};
	char                  magic[8] {};
	char                  epoch[12] {};
	uint8_t               prev[kPhase62ProbeBytes] {};
};
static Phase62ResSlot        g_phase62_res[kPhase62ResSlots] {};
static std::atomic<uint32_t> g_phase62_res_used {0};
static std::atomic<uint32_t> g_phase62_res_kpri_n {0};
static std::atomic<uint32_t> g_phase62_res_other_n {0};
static std::atomic<uint32_t> g_phase62_post_agc_n {0};
static std::atomic<uint32_t> g_phase62_post_seed_n {0};
static std::atomic<uint32_t> g_phase62_alt_anchor_n {0};
static std::atomic<uint32_t> g_phase62_non_kpri_mut_n {0};
static std::atomic<uint64_t> g_phase62_unreg_tsc {0};
static std::atomic<bool>     g_phase62_heatmap_done {false};
static char                  g_phase62_cause[48] = "unknown";
// Phase 63: Unreg / Submit forensics.
static constexpr uint32_t kPhase63UnregResNeed = 16;
static std::atomic<uint64_t> g_phase63_cycle_id {0};
static std::atomic<uint32_t> g_phase63_unreg_owners {0};
static std::atomic<uint32_t> g_phase63_unreg_res_total {0};
static std::atomic<uint32_t> g_phase63_submit_attempt_post {0};
static std::atomic<uint32_t> g_phase63_submit_guest_real_post {0};
static std::atomic<uint32_t> g_phase63_submit_seed_post {0};
static std::atomic<uint32_t> g_phase63_attempt_guest_dcb_post {0};
static std::atomic<uint32_t> g_phase63_attempt_host_post {0};
static std::atomic<uint32_t> g_phase63_post_agc {0};
static std::atomic<uint64_t> g_phase63_unreg_tsc {0};
static std::atomic<bool>     g_phase63_heatmap_done {false};
static char                  g_phase63_cause[48] = "unknown";
// Phase 64: waiters post-Unreg (minimal).
static constexpr uint32_t kPhase64CondWaitNeed = 8;
static constexpr uint32_t kPhase64StuckY       = 10;
static constexpr uint64_t kPhase64NdJobCtrlDef = 0x0000000903420b00ULL;
static std::atomic<uint32_t> g_phase64_main_cond_wait_post {0};
static std::atomic<uint32_t> g_phase64_main_cond_signal_post {0};
static std::atomic<uint32_t> g_phase64_main_cond_signal_match {0};
static std::atomic<uint64_t> g_phase64_last_main_wait_cond {0};
static std::atomic<uint32_t> g_phase64_flip_stuck_streak {0};
static std::atomic<bool>     g_phase64_flip_pending_stuck {false};
static std::atomic<uint32_t> g_phase64_ndjob_static_streak {0};
static std::atomic<bool>     g_phase64_ndjob_static {false};
static std::atomic<uint32_t> g_phase64_fiber_post {0};
static std::atomic<bool>     g_phase64_have_flip_prev {false};
static std::atomic<bool>     g_phase64_have_ndjob_prev {false};
static int32_t               g_phase64_prev_pending {0};
static int32_t               g_phase64_prev_curbuf {-1};
static uint64_t              g_phase64_prev_flip_count {0};
static uint64_t              g_phase64_prev_ndjob[5] {};
static std::atomic<bool>     g_phase64_heatmap_done {false};
static char                  g_phase64_cause[48] = "unknown";
// Phase 65: who waits / menu entry post-Unreg (read-only).
static constexpr uint32_t kPhase65AliasNeed   = 8;
static constexpr uint32_t kPhase65NonMainNeed = 32;
static std::atomic<uint32_t> g_phase65_wait_main {0};
static std::atomic<uint32_t> g_phase65_wait_mixed {0};
static std::atomic<uint32_t> g_phase65_wait_compute {0};
static std::atomic<uint32_t> g_phase65_wait_other {0};
static std::atomic<uint32_t> g_phase65_wait_alias_main {0};
static std::atomic<uint32_t> g_phase65_guest_regbuf2 {0};
static std::atomic<uint32_t> g_phase65_guest_flip {0};
static std::atomic<int>      g_phase65_main_alive {1};
static std::atomic<int>      g_phase65_host_flip_active {0};
static std::atomic<bool>     g_phase65_have_flip_prev {false};
static uint64_t              g_phase65_prev_flip_count {0};
static std::atomic<bool>     g_phase65_heatmap_done {false};
static char                  g_phase65_cause[48] = "unknown";
// Phase 66: recycle Flip L0 after menu detection (opt-in KYTY_PHASE66_MENU_RECYCLE=1).
static constexpr uint32_t kPhase66IdleNeed          = 1;
static constexpr uint64_t kPhase66MinFlipIntervalMs = 2000;
static std::atomic<uint32_t> g_phase66_pending_streak {0};
static std::atomic<uint32_t> g_phase66_recycle_n {0};
static std::atomic<bool>     g_phase66_heatmap_done {false};
static std::atomic<bool>     g_phase66_have_last_flip {false};
static std::chrono::steady_clock::time_point g_phase66_last_flip {};
static char                  g_phase66_cause[48] = "unknown";
// Phase 69: NdJob ctrl/status dump + opt-in soft ready HLE (KYTY_PHASE69_NDJOB_READY=1).
static constexpr uint32_t kPhase69StaticNeed = 8;
static constexpr uint64_t kPhase69CtrlDef    = 0x0000000903420b00ULL;
static constexpr uint64_t kPhase69StatusDef  = 0x000000090386ed80ULL;
static std::atomic<uint32_t> g_phase69_dump_n {0};
static std::atomic<uint32_t> g_phase69_static_streak {0};
static std::atomic<bool>     g_phase69_field_static {false};
static std::atomic<uint32_t> g_phase69_hle_n {0};
static std::atomic<bool>     g_phase69_hle_done {false};
static std::atomic<bool>     g_phase69_have_prev {false};
static std::atomic<bool>     g_phase69_heatmap_done {false};
static uint64_t              g_phase69_prev[8] {};
static char                  g_phase69_cause[48] = "unknown";
// Phase 70: Mixed predicate byte at *(*0x904bb6de8)+0x3eea (offline eboot @ 0x901DE418d).
static constexpr uint32_t kPhase70StaticNeed   = 8;
static constexpr uint64_t kPhase70BaseSlot     = 0x0000000904bb6de8ULL;
static constexpr uint32_t kPhase70FieldOff     = 0x3eea;
static constexpr uint64_t kPhase70NdJobCtrlDef = 0x0000000903420b00ULL;
static std::atomic<uint32_t> g_phase70_dump_n {0};
static std::atomic<uint32_t> g_phase70_static_streak {0};
static std::atomic<bool>     g_phase70_field_static {false};
static std::atomic<uint32_t> g_phase70_hle_n {0};
static std::atomic<bool>     g_phase70_hle_done {false};
static std::atomic<bool>     g_phase70_have_prev {false};
static std::atomic<bool>     g_phase70_heatmap_done {false};
static std::atomic<uint64_t> g_phase70_last_guest_rip {0};
static std::atomic<uint32_t> g_phase70_guest_rip_mixed {0};
static std::atomic<uint64_t> g_phase70_resolved_base {0};
static uint64_t              g_phase70_prev_base {0};
static uint8_t               g_phase70_prev_byte {0};
static uint64_t              g_phase70_prev_obj {0};
static char                  g_phase70_cause[48] = "unknown";
// Phase 71: CTX_FIELD Mixed *(u32*)(0x905f254c0+0x24) + soft-HLE (KYTY_PHASE71_CTX_FIELD=1).
static constexpr uint32_t kPhase71StaticNeed = 8;
static constexpr uint64_t kPhase71CtxBase    = 0x0000000905f254c0ULL;
static constexpr uint32_t kPhase71FieldOff   = 0x24;
static constexpr uint32_t kPhase71CorrOff    = 0x8;
static std::atomic<uint32_t> g_phase71_dump_n {0};
static std::atomic<uint32_t> g_phase71_static_streak {0};
static std::atomic<bool>     g_phase71_field_static {false};
static std::atomic<uint32_t> g_phase71_hle_n {0};
static std::atomic<bool>     g_phase71_hle_done {false};
static std::atomic<bool>     g_phase71_have_prev {false};
static std::atomic<bool>     g_phase71_heatmap_done {false};
static std::atomic<bool>     g_phase71_base_ok {false};
static uint32_t              g_phase71_prev_f24 {0};
static uint32_t              g_phase71_prev_f8 {0};
static char                  g_phase71_cause[48] = "unknown";
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

// Phase 52: audited writes to NdJob +0x10/+0x18 (detect post-Unreg corruption paths).
// Rearm no longer pokes these offs; keep the API so any future write is logged.
[[maybe_unused]] static void Phase52SafeWriteNdJobField(uint64_t obj, uint32_t off, const void* src,
                                                        size_t n) {
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL || src == nullptr || n == 0) {
		return;
	}
	if ((off == 0x10u || off == 0x18u) &&
	    g_phase37_post_unreg.load(std::memory_order_acquire)) {
		uint64_t before = 0;
		uint64_t after  = 0;
		Phase41SafeRead(&before, reinterpret_cast<const void*>(obj + off),
		                sizeof(before) < n ? sizeof(before) : n);
		std::memcpy(&after, src, sizeof(after) < n ? sizeof(after) : n);
		const uint32_t an = g_phase52_write_audit_n.fetch_add(1, std::memory_order_relaxed);
		if (an < 32) {
			LOGF("FlipTrace: phase52 write_audit off=0x%x before=0x%016" PRIx64
			     " after=0x%016" PRIx64 " n=%u tid=%d\n",
			     off, before, after, an, Common::Thread::GetThreadIdUnique());
			fprintf(stderr,
			        "FlipTrace: phase52 write_audit off=0x%x before=0x%016" PRIx64
			        " after=0x%016" PRIx64 "\n",
			        off, before, after);
		}
	}
	Phase41SafeWrite(reinterpret_cast<volatile void*>(obj + off), src, n);
}

// Phase 49: allow live keep[1] only when VideoOut slots are armed + a SubmitFlip was accepted
// and NdJob field/+slots look populated (avoids AV Read[ffffffffffffffff] on empty *obj).
static bool Phase49FieldSlotsLookValid(uint64_t obj) {
	uint8_t field[64] {};
	Phase41SafeRead(field, reinterpret_cast<const void*>(obj + kPhase41Keep1FieldOff), 64);
	const uint64_t bits = *reinterpret_cast<const uint64_t*>(field);
	if (bits != 0 && bits != ~uint64_t {0}) {
		return true;
	}
	for (size_t off = 8; off + 8 <= 64; off += 8) {
		uint64_t slot = 0;
		std::memcpy(&slot, field + off, 8);
		if (slot >= 0x10000ULL && slot < 0x0000800000000000ULL && slot != ~uint64_t {0}) {
			return true;
		}
	}
	return false;
}

static void Phase49LogKeep1Diag(uint64_t obj, const char* why) {
	uint64_t head = 0;
	uint32_t gate = 0;
	uint8_t  field[64] {};
	Phase41SafeRead(&head, reinterpret_cast<const void*>(obj), sizeof(head));
	Phase41SafeRead(&gate, reinterpret_cast<const void*>(obj + 8), sizeof(gate));
	Phase41SafeRead(field, reinterpret_cast<const void*>(obj + kPhase41Keep1FieldOff), 64);
	uint64_t slots[4] {};
	for (int i = 0; i < 4; ++i) {
		std::memcpy(&slots[i], field + 8 + static_cast<size_t>(i) * 8, 8);
	}
	LOGF("FlipTrace: phase49 keep1 diag why=%s obj=0x%016" PRIx64 " *obj=0x%016" PRIx64
	     " gate+8=%u field0=0x%016" PRIx64 " slots=%016" PRIx64 ",%016" PRIx64 ",%016" PRIx64
	     ",%016" PRIx64 " vo_vk=%d submit_ok=%d\n",
	     why, obj, head, gate, *reinterpret_cast<uint64_t*>(field), slots[0], slots[1], slots[2],
	     slots[3], g_phase49_vo_vk_ok.load(std::memory_order_relaxed) ? 1 : 0,
	     g_phase49_submit_accepted.load(std::memory_order_relaxed) ? 1 : 0);
	fprintf(stderr, "FlipTrace: phase49 keep1 diag why=%s vo_vk=%d submit_ok=%d\n", why,
	        g_phase49_vo_vk_ok.load(std::memory_order_relaxed) ? 1 : 0,
	        g_phase49_submit_accepted.load(std::memory_order_relaxed) ? 1 : 0);
}

static bool Phase49CanLiveKeep1(uint64_t guest_rdi) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return false;
	}
	// Phase 52: after saturated seed attempts without GPU progress, stay HLE.
	if (g_phase52_give_up.load(std::memory_order_acquire)) {
		return false;
	}
	if (!g_phase49_vo_vk_ok.load(std::memory_order_acquire)) {
		return false;
	}
	if (!g_phase49_submit_accepted.load(std::memory_order_acquire)) {
		return false;
	}
	uint64_t head = 0;
	Phase41SafeRead(&head, reinterpret_cast<const void*>(guest_rdi), sizeof(head));
	// Empty / poisoned head → live AV Read[ffffffffffffffff]. Bitmap bits alone (P44
	// rearm 0x7fffff) are NOT enough; require a real queue head pointer.
	if (head == 0 || head == ~uint64_t {0}) {
		return false;
	}
	if (head < 0x10000ULL || head >= 0x0000800000000000ULL) {
		return false;
	}
	const uint32_t attempts = g_phase52_head_attempts.load(std::memory_order_acquire);
	if (attempts > kPhase52MaxSeedAttempts) {
		return false;
	}
	return Phase49FieldSlotsLookValid(guest_rdi);
}

static uint64_t Phase50ReadKeep1Head(uint64_t obj) {
	uint64_t head = 0;
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL) {
		return 0;
	}
	Phase41SafeRead(&head, reinterpret_cast<const void*>(obj), sizeof(head));
	return head;
}

static void Phase50NoteObjMutation(const char* why, uint64_t before, uint64_t after) {
	if (before == after) {
		return;
	}
	const uint64_t tsc = LibKernel::KernelGetProcessTimeCounter();
	LOGF("FlipTrace: phase50 obj_mut why=%s before=0x%016" PRIx64 " after=0x%016" PRIx64
	     " tsc=%" PRIu64 " tid=%d\n",
	     why != nullptr ? why : "?", before, after, tsc, Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "FlipTrace: phase50 obj_mut why=%s before=0x%016" PRIx64 " after=0x%016" PRIx64 "\n",
	        why != nullptr ? why : "?", before, after);
	if (after != 0 && after != ~uint64_t {0}) {
		g_phase50_obj_nonzero.store(true, std::memory_order_release);
	}
	g_phase50_last_head.store(after, std::memory_order_release);
}

void Phase50PollKeep1Obj(const char* why) {
	const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL) {
		return;
	}
	uint64_t head = 0;
	uint32_t gate = 0;
	uint8_t  field[64] {};
	Phase41SafeRead(&head, reinterpret_cast<const void*>(obj), sizeof(head));
	Phase41SafeRead(&gate, reinterpret_cast<const void*>(obj + 8), sizeof(gate));
	Phase41SafeRead(field, reinterpret_cast<const void*>(obj + kPhase41Keep1FieldOff), 64);
	uint64_t slots[4] {};
	for (int i = 0; i < 4; ++i) {
		std::memcpy(&slots[i], field + 8 + static_cast<size_t>(i) * 8, 8);
	}
	const uint64_t prev = g_phase50_last_head.exchange(head, std::memory_order_acq_rel);
	if (prev != head) {
		Phase50NoteObjMutation(why != nullptr ? why : "poll", prev, head);
	}
	if (head != 0 && head != ~uint64_t {0}) {
		g_phase50_obj_nonzero.store(true, std::memory_order_release);
	}
	static std::atomic<uint32_t> poll_n {0};
	const uint32_t               pn = poll_n.fetch_add(1, std::memory_order_relaxed);
	if (pn < 48 || (head != 0 && head != ~uint64_t {0} && pn < 96)) {
		const uint64_t tsc = LibKernel::KernelGetProcessTimeCounter();
		LOGF("FlipTrace: phase50 obj_poll why=%s *obj=0x%016" PRIx64 " gate+8=%u field0=0x%016" PRIx64
		     " slots=%016" PRIx64 ",%016" PRIx64 ",%016" PRIx64 ",%016" PRIx64
		     " vo_vk=%d submit_ok=%d tsc=%" PRIu64 " tid=%d\n",
		     why != nullptr ? why : "?", head, gate, *reinterpret_cast<uint64_t*>(field), slots[0],
		     slots[1], slots[2], slots[3],
		     g_phase49_vo_vk_ok.load(std::memory_order_relaxed) ? 1 : 0,
		     g_phase49_submit_accepted.load(std::memory_order_relaxed) ? 1 : 0, tsc,
		     Common::Thread::GetThreadIdUnique());
		if (pn < 16 || (head != 0 && head != ~uint64_t {0})) {
			fprintf(stderr,
			        "FlipTrace: phase50 obj_poll why=%s *obj=0x%016" PRIx64 " gate=%u\n",
			        why != nullptr ? why : "?", head, gate);
		}
	}
	if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
		Phase58NoteNdJobAncre(why != nullptr ? why : "poll");
	}
}

void Phase50NoteWake(const char* name, size_t woken) {
	if (woken == 0 && (name == nullptr || std::strstr(name, "Mixed") == nullptr) &&
	    (name == nullptr || std::strstr(name, "Compute") == nullptr)) {
		return;
	}
	static std::atomic<uint32_t> wake_n {0};
	const uint32_t               wn = wake_n.fetch_add(1, std::memory_order_relaxed);
	if (wn >= 64) {
		return;
	}
	const uint64_t head = Phase50ReadKeep1Head(g_phase41_keep1_obj.load(std::memory_order_acquire));
	const uint64_t tsc  = LibKernel::KernelGetProcessTimeCounter();
	LOGF("SubmitTrace: phase50 wake name=%s woken=%zu *obj=0x%016" PRIx64 " tsc=%" PRIu64
	     " tid=%d\n",
	     name != nullptr ? name : "?", woken, head, tsc, Common::Thread::GetThreadIdUnique());
	if (wn < 24) {
		fprintf(stderr, "SubmitTrace: phase50 wake name=%s woken=%zu *obj=0x%016" PRIx64 "\n",
		        name != nullptr ? name : "?", woken, head);
	}
	if (g_phase37_post_unreg.load(std::memory_order_acquire) && name != nullptr &&
	    (std::strstr(name, "Mixed") != nullptr || std::strstr(name, "Compute") != nullptr ||
	     std::strstr(name, "NdJob") != nullptr || std::strstr(name, "rearm") != nullptr)) {
		Phase51DumpNdJobStruct(name);
	}
}

void Phase50NoteNdJobBatch(int index, int total, uint64_t ident, int filter, uint64_t data) {
	static std::atomic<uint32_t> batch_n {0};
	const uint32_t               bn = batch_n.fetch_add(1, std::memory_order_relaxed);
	if (bn >= 48) {
		return;
	}
	const uint64_t head = Phase50ReadKeep1Head(g_phase41_keep1_obj.load(std::memory_order_acquire));
	const uint64_t tsc  = LibKernel::KernelGetProcessTimeCounter();
	LOGF("SubmitTrace: phase50 ndjob_batch [%d/%d] ident=0x%016" PRIx64 " filter=%d data=0x%016"
	     PRIx64 " *obj=0x%016" PRIx64 " tsc=%" PRIu64 " tid=%d\n",
	     index, total, ident, filter, data, head, tsc, Common::Thread::GetThreadIdUnique());
	if (bn < 24) {
		fprintf(stderr,
		        "SubmitTrace: phase50 ndjob_batch [%d/%d] *obj=0x%016" PRIx64 " filter=%d\n",
		        index, total, head, filter);
	}
	if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
		Phase51DumpNdJobStruct("ndjob_batch");
		Phase51CheckWorkerFailfast("ndjob_batch");
	}
}

void Phase50NoteSubmitGpu(int handle, int index, uint64_t request_id, uint64_t submit_gpu_total) {
	g_phase52_submit_gpu_mirror.store(submit_gpu_total, std::memory_order_release);
	const uint64_t tsc = LibKernel::KernelGetProcessTimeCounter();
	LOGF("FlipTrace: phase50 submit_gpu handle=%d index=%d id=%" PRIu64 " total=%" PRIu64
	     " tsc=%" PRIu64 " tid=%d\n",
	     handle, index, request_id, submit_gpu_total, tsc, Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "FlipTrace: phase50 submit_gpu index=%d total=%" PRIu64 "\n", index,
	        submit_gpu_total);
	char beat[128];
	std::snprintf(beat, sizeof(beat), "heartbeat submit_gpu total=%" PRIu64 " index=%d tid=%d",
	              submit_gpu_total, index, Common::Thread::GetThreadIdUnique());
	Common::HeartbeatLog(beat);
}

bool Phase50ObjNonZeroSeen() {
	return g_phase50_obj_nonzero.load(std::memory_order_acquire);
}

void Phase52NoteAfterDump(const char* why);
void Phase53ProbeRetarget(const char* why);
void Phase53ScanObjDeep(const char* why);
void Phase53DumpStatusRdi(const char* why);
void Phase53DumpUserAnchors(const char* why);
void Phase55OnMixedRewait(const char* role, uintptr_t cond_ptr);
void Phase55PollWatch();
void Phase55TryFakeQueueInject(const char* why);
void Phase55EmitHeatmap(const char* why);
void Phase55FlushDeferredLogs();
void Phase55TryArmMixedThunk();
void Phase55NoteGuestCond(uint64_t guest_cond_va, uint64_t guest_arg, const char* role);
static void Phase55QueueDump(const char* why, uint64_t base);
void Phase56PollWatch();
void Phase56EmitHeatmap(const char* why);
void Phase56TryFakeCount(const char* why);
void Phase56NoteMainSignal(uint64_t guest_cond_va, const char* role);
void Phase57PollHeatmap();
void Phase57EmitHeatmap(const char* why);
void Phase57TryScanMixedBody();
void Phase57NoteMixedRegs(uint64_t rdi, uint64_t rsi, uint64_t rdx);
void Phase58NoteNdJobAncre(const char* why);
void Phase58PollWatch();
void Phase58EmitHeatmap(const char* why);
void Phase59EmitHeatmap(const char* why);
void Phase59Poll();
void Phase61RingProbe();
void Phase61EmitHeatmap(const char* why);
void Phase61Poll();
void Phase62NoteRegisterResource(uint64_t addr, uint64_t size, int res_type, const char* name,
                                 uint32_t handle);
void Phase62NoteUnreg();
void Phase62EmitHeatmap(const char* why);
void Phase62Poll();
void Phase63NoteUnregister(uint32_t owner, int n_resources, const char* why, bool owner_event);
void Phase63NoteSubmitEntry(const char* api, uint32_t queue, const uint32_t* dcb,
                            uint32_t size_in_dwords);
void Phase63NotePostAgc(const char* why);
void Phase63EmitHeatmap(const char* why);
void Phase63Poll();
void Phase64NoteMainCondWait(uint64_t cond_va);
void Phase64NoteMainCondSignal(uint64_t cond_va, bool match_wait);
void Phase64EmitHeatmap(const char* why);
void Phase64Poll();
void Phase64SampleFlipAndNdJob(); // defined after VideoOutContext
void Phase65NoteCondWait(const char* role, const char* name, int tid, uint64_t ra);
void Phase65NoteGuestRegbuf2();
void Phase65NoteGuestFlip();
void Phase65EmitHeatmap(const char* why);
void Phase65Poll();
void Phase65SampleHostFlip(); // defined after VideoOutContext
void Phase66TryMenuRecycle(); // defined after Phase37 helpers
void Phase66EmitHeatmap(const char* why);
void Phase66Poll();
void Phase69EmitHeatmap(const char* why);
void Phase69Poll();
void Phase69SampleNdJob();
void Phase70EmitHeatmap(const char* why);
void Phase70Poll();
void Phase70SampleField();
void Phase70NoteGuestRip(uint64_t guest_rip, const char* role);
void Phase71EmitHeatmap(const char* why);
void Phase71Poll();
void Phase71SampleCtx();
void Phase58NoteWorkerFiberFromAtomics();
void Phase58NoteMainAgcCross(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3);

bool Phase53RealQueueSeen() {
	return g_phase53_real_queue.load(std::memory_order_acquire) != 0;
}

bool Phase37PostUnregisterSeen() {
	return g_phase37_post_unreg.load(std::memory_order_acquire);
}

uint64_t Phase54CurrentCycleId() {
	return g_phase54_cycle_id.load(std::memory_order_acquire);
}

uint64_t Phase54BumpCycle(const char* why) {
	const uint64_t id = g_phase54_cycle_id.fetch_add(1, std::memory_order_acq_rel) + 1;
	g_phase54_budget_cycle.store(id, std::memory_order_release);
	g_phase54_host_wake_n.store(0, std::memory_order_release);
	g_phase54_cycle_guest_real.store(false, std::memory_order_release);
	g_phase54_cycle_mixed_leave.store(false, std::memory_order_release);
	static std::atomic<uint32_t> logs {0};
	if (logs.fetch_add(1, std::memory_order_relaxed) < 64) {
		LOGF("SubmitTrace: phase54 cycle_bump id=%" PRIu64 " why=%s tid=%d\n", id,
		     why != nullptr ? why : "?", Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase54 cycle_bump id=%" PRIu64 " why=%s\n", id,
		        why != nullptr ? why : "?");
	}
	return id;
}

void Phase54NoteHostWake(const char* why) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const uint64_t cycle = g_phase54_cycle_id.load(std::memory_order_acquire);
	const uint32_t n     = g_phase54_host_wake_n.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 8 || (n % 32) == 0) {
		LOGF("SubmitTrace: phase54 hostWake cycle=%" PRIu64 " n=%u why=%s guest_real=%d "
		     "leave=%d tid=%d\n",
		     cycle, n, why != nullptr ? why : "?",
		     g_phase54_cycle_guest_real.load(std::memory_order_relaxed) ? 1 : 0,
		     g_phase54_cycle_mixed_leave.load(std::memory_order_relaxed) ? 1 : 0,
		     Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase54 hostWake cycle=%" PRIu64 " n=%u why=%s\n", cycle, n,
		        why != nullptr ? why : "?");
	}
}

bool Phase54AllowHostWake() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return true;
	}
	if (g_phase54_cycle_guest_real.load(std::memory_order_acquire) ||
	    g_phase54_cycle_mixed_leave.load(std::memory_order_acquire) ||
	    g_phase47_guest_draw_ok.load(std::memory_order_acquire)) {
		return true;
	}
	const uint32_t n = g_phase54_host_wake_n.load(std::memory_order_acquire);
	// Budget: max 5 host wakes per cycle without Mixed submit/leave.
	return n < 5;
}

void Phase54NoteMixedLeave(const char* role) {
	g_phase54_cycle_mixed_leave.store(true, std::memory_order_release);
	const uint64_t cycle = g_phase54_cycle_id.load(std::memory_order_acquire);
	static std::atomic<uint32_t> logs {0};
	if (logs.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("SubmitTrace: phase54 mixed_leave cycle=%" PRIu64 " role=%s tid=%d\n", cycle,
		     role != nullptr ? role : "?", Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase54 mixed_leave cycle=%" PRIu64 " role=%s\n", cycle,
		        role != nullptr ? role : "?");
	}
}

void Phase54NoteMixedWake(const char* role, uintptr_t cond_ptr, const char* outcome) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const uint64_t cycle = g_phase54_cycle_id.load(std::memory_order_acquire);
	const uint32_t n     = g_phase54_mixed_wake_n.fetch_add(1, std::memory_order_relaxed) + 1;
	const bool     rewait =
	    outcome != nullptr && (std::strcmp(outcome, "rewait") == 0 || std::strcmp(outcome, "woken") == 0);
	if (rewait && outcome != nullptr && std::strcmp(outcome, "rewait") == 0) {
		(void)g_phase54_rewait_n.fetch_add(1, std::memory_order_relaxed);
	}
	if (n <= 48 || (n % 64) == 0) {
		LOGF("SubmitTrace: phase54 mixed_wake cycle=%" PRIu64 " n=%u role=%s cond=0x%016" PRIx64
		     " outcome=%s tid=%d\n",
		     cycle, n, role != nullptr ? role : "?", static_cast<uint64_t>(cond_ptr),
		     outcome != nullptr ? outcome : "?", Common::Thread::GetThreadIdUnique());
		fprintf(stderr,
		        "SubmitTrace: phase54 mixed_wake cycle=%" PRIu64 " role=%s outcome=%s\n", cycle,
		        role != nullptr ? role : "?", outcome != nullptr ? outcome : "?");
	}
	if (outcome != nullptr && std::strcmp(outcome, "leave") == 0) {
		Phase54NoteMixedLeave(role);
	}
	if (outcome != nullptr && std::strcmp(outcome, "rewait") == 0) {
		Phase55OnMixedRewait(role, cond_ptr);
	}
	Phase54TryFakeJobAfterWake(role);
	Phase55PollWatch();
}

bool Phase54FakeJobEnabled() {
	static int cached = -1;
	if (cached < 0) {
		const char* e = std::getenv("KYTY_PHASE54_FAKE_JOB");
		cached        = (e != nullptr && e[0] == '1') ? 1 : 0;
	}
	return cached == 1;
}

void Phase54TryFakeJobAfterWake(const char* role) {
	if (!Phase54FakeJobEnabled() || !g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	if (g_phase54_fake_job_done.load(std::memory_order_acquire)) {
		return;
	}
	if (g_phase47_guest_draw_ok.load(std::memory_order_acquire) ||
	    g_phase54_cycle_guest_real.load(std::memory_order_acquire)) {
		return;
	}
	// Inject after first Mixed wake post-Unreg, or after 8 rewaits.
	const uint32_t wakes   = g_phase54_mixed_wake_n.load(std::memory_order_acquire);
	const uint32_t rewaits = g_phase54_rewait_n.load(std::memory_order_acquire);
	if (wakes < 1 && rewaits < 8) {
		return;
	}
	if (g_phase54_fake_job_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	static uint32_t fake_dcb[8] = {0x80000000u, 0x80000000u, 0, 0, 0, 0, 0, 0};
	const uint64_t  cycle       = g_phase54_cycle_id.load(std::memory_order_acquire);
	LOGF("SubmitTrace: phase54 fake_job cycle=%" PRIu64 " role=%s wakes=%u rewaits=%u "
	     "dcb=NOP size=2 tid=%d\n",
	     cycle, role != nullptr ? role : "?", wakes, rewaits, Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "SubmitTrace: phase54 fake_job cycle=%" PRIu64 " role=%s\n", cycle,
	        role != nullptr ? role : "?");
	(void)Libs::Graphics::Gen5Driver::GraphicsDriverSubmitCommandBuffer(0, fake_dcb, 2);
}

void Phase54NoteSubmit(uint32_t queue, const uint32_t* dcb, uint32_t size_in_dwords);

bool Phase51GraphicsIdent0IsNotCompletion(uint64_t ident, int filter) {
	// SharpEmu-aligned: KERNEL_EVFILT_GRAPHICS (-14) with ident==0 is IRQ0 /
	// QUEUED_GRAPHICS_INTERRUPT bootstrap — not a useful GPU job completion.
	return filter == LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS && ident == 0;
}

void Phase51DumpNdJobStruct(const char* why) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const uint32_t dn = g_phase51_dump_n.fetch_add(1, std::memory_order_relaxed);
	if (dn >= 64) {
		return;
	}
	const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL) {
		if (dn < 8) {
			LOGF("FlipTrace: phase51 dump why=%s obj=invalid/0\n", why != nullptr ? why : "?");
			fprintf(stderr, "FlipTrace: phase51 dump why=%s obj=invalid\n",
			        why != nullptr ? why : "?");
		}
		return;
	}
	uint8_t buf[kPhase51DumpBytes] {};
	Phase41SafeRead(buf, reinterpret_cast<const void*>(obj), kPhase51DumpBytes);
	uint64_t field0 = 0;
	Phase41SafeRead(&field0, reinterpret_cast<const void*>(obj + kPhase41Keep1FieldOff),
	                sizeof(field0));
	uint64_t q0 = 0;
	uint64_t q8 = 0;
	uint64_t q10 = 0;
	uint64_t q18 = 0;
	std::memcpy(&q0, buf + 0x00, 8);
	std::memcpy(&q8, buf + 0x08, 8);
	std::memcpy(&q10, buf + 0x10, 8);
	std::memcpy(&q18, buf + 0x18, 8);
	const uint64_t tsc = LibKernel::KernelGetProcessTimeCounter();
	LOGF("FlipTrace: phase51 dump why=%s base=0x%016" PRIx64 " +0=*obj=0x%016" PRIx64
	     " +8=gate=0x%016" PRIx64 " +0x10=0x%016" PRIx64 " +0x18=0x%016" PRIx64
	     " +0x20858=0x%016" PRIx64 " tsc=%" PRIu64 " tid=%d n=%u\n",
	     why != nullptr ? why : "?", obj, q0, q8, q10, q18, field0, tsc,
	     Common::Thread::GetThreadIdUnique(), dn);
	if (dn < 24) {
		fprintf(stderr,
		        "FlipTrace: phase51 dump why=%s *obj=0x%016" PRIx64 " +8=0x%016" PRIx64
		        " +0x20858=0x%016" PRIx64 "\n",
		        why != nullptr ? why : "?", q0, q8, field0);
	}
	char nz[512];
	int  nzpos = std::snprintf(nz, sizeof(nz), "FlipTrace: phase51 dump nonzero:");
	bool any_nz = false;
	for (size_t off = 0; off + 8 <= kPhase51DumpBytes; off += 8) {
		uint64_t v = 0;
		std::memcpy(&v, buf + off, 8);
		if (v == 0) {
			continue;
		}
		any_nz = true;
		nzpos += std::snprintf(nz + nzpos, sizeof(nz) - static_cast<size_t>(nzpos),
		                       " +0x%zx=%016" PRIx64, off, v);
		if (nzpos > 420) {
			break;
		}
	}
	if (!any_nz) {
		nzpos += std::snprintf(nz + nzpos, sizeof(nz) - static_cast<size_t>(nzpos), " (none)");
	}
	LOGF("%s\n", nz);
	if (dn < 16) {
		fprintf(stderr, "%s\n", nz);
	}
	if (g_phase51_have_prev_dump) {
		char diff[512];
		int  dpos = std::snprintf(diff, sizeof(diff), "FlipTrace: phase51 dump diff:");
		bool any  = false;
		for (size_t off = 0; off + 8 <= kPhase51DumpBytes; off += 8) {
			uint64_t b = 0;
			uint64_t a = 0;
			std::memcpy(&b, g_phase51_prev_dump + off, 8);
			std::memcpy(&a, buf + off, 8);
			if (b == a) {
				continue;
			}
			any = true;
			dpos += std::snprintf(diff + dpos, sizeof(diff) - static_cast<size_t>(dpos),
			                      " +0x%zx:%016" PRIx64 "->%016" PRIx64, off, b, a);
			if (dpos > 420) {
				break;
			}
		}
		if (!any) {
			dpos += std::snprintf(diff + dpos, sizeof(diff) - static_cast<size_t>(dpos),
			                      " (none)");
		}
		LOGF("%s\n", diff);
		if (dn < 24 || any) {
			fprintf(stderr, "%s\n", diff);
		}
	}
	std::memcpy(g_phase51_prev_dump, buf, kPhase51DumpBytes);
	g_phase51_have_prev_dump = true;
	Phase52NoteAfterDump(why);
}

void Phase51NoteEqDelivery(const char* eq_name, uint64_t ident, int filter, uint64_t data,
                           void* udata) {
	static std::atomic<uint32_t> eq_n {0};
	const uint32_t               n = eq_n.fetch_add(1, std::memory_order_relaxed);
	if (n >= 64) {
		return;
	}
	const bool not_completion = Phase51GraphicsIdent0IsNotCompletion(ident, filter);
	const uint64_t tsc = LibKernel::KernelGetProcessTimeCounter();
	LOGF("SubmitTrace: phase51 eq name=%s ident=0x%016" PRIx64 " filter=%d data=0x%016" PRIx64
	     " udata=0x%016" PRIx64 " not_completion=%d tsc=%" PRIu64 " tid=%d\n",
	     eq_name != nullptr ? eq_name : "?", ident, filter, data,
	     reinterpret_cast<uint64_t>(udata), not_completion ? 1 : 0, tsc,
	     Common::Thread::GetThreadIdUnique());
	if (n < 32) {
		fprintf(stderr,
		        "SubmitTrace: phase51 eq filter=%d ident=0x%016" PRIx64 " not_completion=%d\n",
		        filter, ident, not_completion ? 1 : 0);
	}
	// Dump on delivery; GRAPHICS ident=0 is logged but not treated as writer progress.
	if (!not_completion) {
		Phase51DumpNdJobStruct("eq_delivery");
	} else if (n < 8) {
		Phase51DumpNdJobStruct("eq_graphics0");
	}
}

void Phase51NoteFiber(const char* op, const char* name, const char* path) {
	if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
		(void)g_phase64_fiber_post.fetch_add(1, std::memory_order_relaxed);
	}
	static std::atomic<uint32_t> fiber_n {0};
	const uint32_t               n = fiber_n.fetch_add(1, std::memory_order_relaxed);
	if (n >= 64) {
		return;
	}
	LOGF("SubmitTrace: phase51 fiber op=%s name=%s path=%s tid=%d\n", op != nullptr ? op : "?",
	     name != nullptr ? name : "?", path != nullptr ? path : "?",
	     Common::Thread::GetThreadIdUnique());
	if (n < 32) {
		fprintf(stderr, "SubmitTrace: phase51 fiber op=%s name=%s path=%s\n",
		        op != nullptr ? op : "?", name != nullptr ? name : "?",
		        path != nullptr ? path : "?");
	}
}

bool Phase51ShouldSkipFiberSoftAck(const char* name) {
	// Minimal fix: soft-ACK must not short-circuit NdJob fibers (BootCards unchanged).
	if (name == nullptr) {
		return false;
	}
	return std::strstr(name, "NdJob") != nullptr;
}

void Phase51CheckWorkerFailfast(const char* why) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const uint64_t rsi = g_phase41_keep1_rsi.load(std::memory_order_acquire);
	// Worker early path: esi == -1 (low 32 bits all ones).
	if (static_cast<uint32_t>(rsi) != 0xffffffffu) {
		return;
	}
	const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL) {
		return;
	}
	const uint64_t head = Phase50ReadKeep1Head(obj);
	if (head != 0 && head != ~uint64_t {0}) {
		return;
	}
	uint8_t field[64] {};
	Phase41SafeRead(field, reinterpret_cast<const void*>(obj + kPhase41Keep1FieldOff), 64);
	bool slots_nz = false;
	for (size_t off = 0; off + 8 <= 64; off += 8) {
		uint64_t slot = 0;
		std::memcpy(&slot, field + off, 8);
		if (slot != 0 && slot != ~uint64_t {0}) {
			slots_nz = true;
			break;
		}
	}
	if (!slots_nz) {
		return;
	}
	static std::atomic<uint32_t> probe_n {0};
	const uint32_t               pn = probe_n.fetch_add(1, std::memory_order_relaxed);
	if (pn < 16) {
		LOGF("FlipTrace: phase51 worker_probe why=%s esi=-1 *obj=0x%016" PRIx64
		     " slots_nz=1 rsi=0x%016" PRIx64 " tid=%d\n",
		     why != nullptr ? why : "?", head, rsi, Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "FlipTrace: phase51 worker_probe why=%s esi=-1 *obj=0\n",
		        why != nullptr ? why : "?");
	}
	const char* ff = std::getenv("KYTY_PHASE51_FAILFAST");
	if (ff == nullptr || ff[0] != '1') {
		return;
	}
	if (g_phase51_failfast_fired.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	Phase51DumpNdJobStruct("failfast");
	EXIT("phase51 failfast: worker esi=-1 with *obj==0 and nonempty slots (why=%s)\n",
	     why != nullptr ? why : "?");
}

void Phase51TryBypassFlipL0(const char* why) {
	const char* env = std::getenv("KYTY_PHASE51_BYPASS_FLIP");
	if (env == nullptr || env[0] != '1') {
		return;
	}
	if (g_phase51_bypass_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const int handle = g_phase42_flip_handle.load(std::memory_order_acquire);
	if (handle <= 0) {
		LOGF("FlipTrace: phase51 bypass_flip why=%s SKIP no handle\n",
		     why != nullptr ? why : "?");
		fprintf(stderr, "FlipTrace: phase51 bypass_flip SKIP no handle\n");
		return;
	}
	int         result = VIDEO_OUT_ERROR_INVALID_HANDLE;
	const char* path   = "none";
	if (Graphics::HasRenderContext()) {
		// One-shot proof: heap CB intentionally retained until process exit (Prepare
		// records into it; Complete may run later on GPU timeline).
		auto*    host_cb    = new Graphics::CommandBuffer(0);
		uint64_t request_id = 0;
		result = Presentation::DisplayBufferSubmitFlipFromGpu(*host_cb, handle, 0,
		                                                     /*VIDEO_OUT_FLIP_MODE_VSYNC*/ 1, 0,
		                                                     request_id);
		path = "gpu_submit";
		if (result != OK) {
			result = VideoOutSubmitFlip(handle, 0, /*VIDEO_OUT_FLIP_MODE_VSYNC*/ 1, 0);
			path   = "cpu_submit_fallback";
		}
	} else {
		result = VideoOutSubmitFlip(handle, 0, /*VIDEO_OUT_FLIP_MODE_VSYNC*/ 1, 0);
		path   = "cpu_submit";
	}
	LOGF("FlipTrace: phase51 bypass_flip why=%s handle=%d index=0 result=%d path=%s tid=%d\n",
	     why != nullptr ? why : "?", handle, result, path, Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "FlipTrace: phase51 bypass_flip handle=%d index=0 result=%d path=%s\n",
	        handle, result, path);
	(void)LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
}

static int Phase52SeedMode() {
	const char* e = std::getenv("KYTY_PHASE52_SEED_HEAD");
	if (e == nullptr || e[0] == '\0' || e[0] == '0') {
		return 0;
	}
	if (e[0] == '2') {
		return 2;
	}
	if (e[0] == '1') {
		return 1;
	}
	return 0;
}

static bool Phase52IsGuestPtr(uint64_t v) {
	return v >= kPhase52GuestLo && v < kPhase52GuestHi;
}

static bool Phase52IsUserPtr(uint64_t v) {
	return v >= kPhase52UserLo && v < kPhase52UserHi;
}

static bool Phase52IsPlausiblePtr(uint64_t v) {
	if (v < 0x10000ULL || v == ~uint64_t {0}) {
		return false;
	}
	// Ignore pure 32-bit-ish counters extended to 64 (high half zero, small low).
	if ((v >> 32) == 0 && v < 0x100000ULL) {
		return false;
	}
	return Phase52IsGuestPtr(v) || Phase52IsUserPtr(v);
}

static int Phase52LookupVoAnchorIndex(uint64_t ptr) {
	if (!Phase52IsUserPtr(ptr)) {
		return -1;
	}
	const auto img = Presentation::DisplayBufferFind(ptr, false);
	if (img.image == nullptr) {
		return -1;
	}
	return static_cast<int>(img.index);
}

static uint64_t Phase52ReadNextHop(uint64_t node) {
	uint64_t q0 = 0;
	uint64_t q8 = 0;
	Phase41SafeRead(&q0, reinterpret_cast<const void*>(node), sizeof(q0));
	Phase41SafeRead(&q8, reinterpret_cast<const void*>(node + 8), sizeof(q8));
	if (Phase52IsGuestPtr(q0) || q0 == 0) {
		return q0;
	}
	if (Phase52IsGuestPtr(q8) || q8 == 0) {
		return q8;
	}
	return ~uint64_t {0}; // invalid
}

enum class Phase52FollowKind : int { Ignore = 0, VoAnchor = 1, WeakList = 2, StrongList = 3 };

static Phase52FollowKind Phase52ClassifyNode(uint64_t ptr, uint8_t* dump128) {
	if (!Phase52IsGuestPtr(ptr)) {
		return Phase52FollowKind::Ignore;
	}
	Phase41SafeRead(dump128, reinterpret_cast<const void*>(ptr), kPhase52FollowBytes);
	const uint64_t next1 = Phase52ReadNextHop(ptr);
	if (next1 == ~uint64_t {0}) {
		return Phase52FollowKind::WeakList;
	}
	if (next1 == 0) {
		// Single-node empty-next: weak (don't seed).
		return Phase52FollowKind::WeakList;
	}
	if (!Phase52IsGuestPtr(next1)) {
		return Phase52FollowKind::WeakList;
	}
	const uint64_t next2 = Phase52ReadNextHop(next1);
	if (next2 == ~uint64_t {0}) {
		return Phase52FollowKind::WeakList;
	}
	if (next2 != 0 && !Phase52IsGuestPtr(next2)) {
		return Phase52FollowKind::WeakList;
	}
	// Optional length/type dword in first 32 B.
	uint32_t maybe_len = 0;
	std::memcpy(&maybe_len, dump128 + 16, 4);
	if (maybe_len != 0 && maybe_len > 0x10000u) {
		return Phase52FollowKind::WeakList;
	}
	return Phase52FollowKind::StrongList;
}

void Phase52FollowNdJobPtrs(const char* why) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const uint32_t fn = g_phase52_follow_n.fetch_add(1, std::memory_order_relaxed);
	if (fn >= 48) {
		return;
	}
	const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL) {
		return;
	}
	uint64_t q10 = 0;
	uint64_t q40 = 0;
	uint8_t  field[64] {};
	Phase41SafeRead(&q10, reinterpret_cast<const void*>(obj + 0x10), sizeof(q10));
	Phase41SafeRead(&q40, reinterpret_cast<const void*>(obj + 0x40), sizeof(q40));
	Phase41SafeRead(field, reinterpret_cast<const void*>(obj + kPhase41Keep1FieldOff), 64);
	// Detect residual rearm corruption pattern without our poke.
	if ((q10 & 0xffffffffULL) == 1ULL && (q10 >> 32) == 0x10000000ULL && fn < 8) {
		LOGF("FlipTrace: phase52 write_audit_suspect +0x10=0x%016" PRIx64
		     " (rearm-like …0001) why=%s\n",
		     q10, why != nullptr ? why : "?");
		fprintf(stderr, "FlipTrace: phase52 write_audit_suspect +0x10=0x%016" PRIx64 "\n", q10);
	}
	struct Cand {
		uint32_t    off;
		uint64_t    ptr;
	};
	Cand cands[8] {};
	int  nc = 0;
	auto push = [&](uint32_t off, uint64_t ptr) {
		if (nc >= 8 || !Phase52IsPlausiblePtr(ptr)) {
			return;
		}
		cands[nc++] = {off, ptr};
	};
	push(0x10, q10);
	push(0x40, q40);
	for (int i = 0; i < 4; ++i) {
		uint64_t slot = 0;
		std::memcpy(&slot, field + 8 + static_cast<size_t>(i) * 8, 8);
		push(kPhase41Keep1FieldOff + 8u + static_cast<uint32_t>(i) * 8u, slot);
	}
	for (int i = 0; i < nc; ++i) {
		const uint64_t ptr = cands[i].ptr;
		const uint32_t off = cands[i].off;
		if (Phase52IsUserPtr(ptr)) {
			const int vo_idx = Phase52LookupVoAnchorIndex(ptr);
			if (vo_idx >= 0) {
				LOGF("FlipTrace: phase52 follow why=%s off=0x%x ptr=0x%016" PRIx64
				     " kind=vo_anchor index=%d tid=%d\n",
				     why != nullptr ? why : "?", off, ptr, vo_idx,
				     Common::Thread::GetThreadIdUnique());
				if (fn < 24) {
					fprintf(stderr,
					        "FlipTrace: phase52 follow off=0x%x kind=vo_anchor index=%d\n", off,
					        vo_idx);
				}
				continue;
			}
			LOGF("FlipTrace: phase52 follow why=%s off=0x%x ptr=0x%016" PRIx64
			     " kind=user_ptr (not VO) tid=%d\n",
			     why != nullptr ? why : "?", off, ptr, Common::Thread::GetThreadIdUnique());
			continue;
		}
		uint8_t            dump[kPhase52FollowBytes] {};
		Phase52FollowKind  kind = Phase52ClassifyNode(ptr, dump);
		const char*        kn =
		    kind == Phase52FollowKind::StrongList
		        ? "strong_list"
		        : (kind == Phase52FollowKind::WeakList ? "weak_list" : "ignore");
		char hex[200];
		int  hpos = 0;
		for (size_t b = 0; b < 32 && hpos >= 0 && static_cast<size_t>(hpos) + 3 < sizeof(hex);
		     ++b) {
			hpos += std::snprintf(hex + hpos, sizeof(hex) - static_cast<size_t>(hpos), "%02x",
			                      dump[b]);
		}
		LOGF("FlipTrace: phase52 follow why=%s off=0x%x ptr=0x%016" PRIx64 " kind=%s head32=%s "
		     "tid=%d\n",
		     why != nullptr ? why : "?", off, ptr, kn, hex, Common::Thread::GetThreadIdUnique());
		if (fn < 24) {
			fprintf(stderr, "FlipTrace: phase52 follow off=0x%x ptr=0x%016" PRIx64 " kind=%s\n",
			        off, ptr, kn);
		}
		if (kind == Phase52FollowKind::StrongList) {
			g_phase52_best_candidate.store(ptr, std::memory_order_release);
			g_phase52_candidate_strong.store(true, std::memory_order_release);
		} else if (kind == Phase52FollowKind::WeakList &&
		           g_phase52_best_candidate.load(std::memory_order_relaxed) == 0) {
			g_phase52_best_candidate.store(ptr, std::memory_order_release);
		}
	}
}

void Phase52TrySeedKeep1Head(const char* why) {
	const int mode = Phase52SeedMode();
	if (mode == 0 || g_phase52_give_up.load(std::memory_order_acquire)) {
		return;
	}
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const uint64_t cand = g_phase52_best_candidate.load(std::memory_order_acquire);
	const bool     strong = g_phase52_candidate_strong.load(std::memory_order_acquire);
	if (cand == 0 || !Phase52IsGuestPtr(cand)) {
		return;
	}
	// Seed write requires strong_list; dry-run may log weak too.
	if (mode == 2 && !strong) {
		return;
	}
	const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL) {
		return;
	}
	uint64_t head = 0;
	Phase41SafeRead(&head, reinterpret_cast<const void*>(obj), sizeof(head));
	if (head != 0 && head != ~uint64_t {0}) {
		return;
	}
	uint8_t dump[kPhase52FollowBytes] {};
	Phase41SafeRead(dump, reinterpret_cast<const void*>(cand), kPhase52FollowBytes);
	if (mode == 1) {
		static std::atomic<uint32_t> dry_n {0};
		const uint32_t               dn = dry_n.fetch_add(1, std::memory_order_relaxed);
		if (dn >= 16) {
			return;
		}
		char hex[260];
		int  hpos = 0;
		for (size_t b = 0; b < 64 && hpos >= 0 && static_cast<size_t>(hpos) + 3 < sizeof(hex);
		     ++b) {
			hpos += std::snprintf(hex + hpos, sizeof(hex) - static_cast<size_t>(hpos), "%02x",
			                      dump[b]);
		}
		LOGF("FlipTrace: phase52 seed_candidate why=%s ptr=0x%016" PRIx64 " strong=%d dry=1 "
		     "dump64=%s tid=%d\n",
		     why != nullptr ? why : "?", cand, strong ? 1 : 0, hex,
		     Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "FlipTrace: phase52 seed_candidate ptr=0x%016" PRIx64 " strong=%d\n",
		        cand, strong ? 1 : 0);
		return;
	}
	// mode == 2
	uint32_t attempts = g_phase52_head_attempts.load(std::memory_order_acquire);
	if (attempts >= kPhase52MaxSeedAttempts) {
		if (!g_phase47_guest_draw_ok.load(std::memory_order_acquire) &&
		    g_phase52_submit_gpu_mirror.load(std::memory_order_relaxed) == 0) {
			g_phase52_give_up.store(true, std::memory_order_release);
			LOGF("FlipTrace: phase52 seed give_up attempts=%u (no guest_draw/submit_gpu)\n",
			     attempts);
			fprintf(stderr, "FlipTrace: phase52 seed give_up attempts=%u\n", attempts);
		}
		return;
	}
	attempts = g_phase52_head_attempts.fetch_add(1, std::memory_order_acq_rel) + 1;
	Phase41SafeWrite(reinterpret_cast<volatile void*>(obj), &cand, sizeof(cand));
	LOGF("FlipTrace: phase52 seed_head why=%s ptr=0x%016" PRIx64 " n=%u tid=%d\n",
	     why != nullptr ? why : "?", cand, attempts, Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "FlipTrace: phase52 seed_head ptr=0x%016" PRIx64 " n=%u\n", cand, attempts);
	Phase50PollKeep1Obj("seed");
	if (attempts >= kPhase52MaxSeedAttempts &&
	    !g_phase47_guest_draw_ok.load(std::memory_order_acquire) &&
	    g_phase52_submit_gpu_mirror.load(std::memory_order_relaxed) == 0) {
		// Defer give_up to next TrySeed when attempts already at max.
	}
}

void Phase52NoteAfterDump(const char* why) {
	Phase52FollowNdJobPtrs(why);
	Phase52TrySeedKeep1Head(why);
	Phase53ProbeRetarget(why);
}

static void Phase53NoteRealQueue(uint64_t ptr, const char* why) {
	if (!Phase52IsGuestPtr(ptr)) {
		return;
	}
	const uint64_t prev = g_phase53_real_queue.exchange(ptr, std::memory_order_acq_rel);
	if (prev == ptr) {
		return;
	}
	LOGF("FlipTrace: phase53 real_queue why=%s ptr=0x%016" PRIx64 " prev=0x%016" PRIx64
	     " tid=%d\n",
	     why != nullptr ? why : "?", ptr, prev, Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "FlipTrace: phase53 real_queue ptr=0x%016" PRIx64 "\n", ptr);
}

static void Phase53ScanArgsForQueue(uint64_t a, uint64_t b, uint64_t c, const char* why) {
	const uint64_t args[3] = {a, b, c};
	for (uint64_t v: args) {
		if (Phase52IsGuestPtr(v)) {
			Phase53NoteRealQueue(v, why);
			uint64_t head = 0;
			Phase41SafeRead(&head, reinterpret_cast<const void*>(v), sizeof(head));
			if (Phase52IsGuestPtr(head)) {
				Phase53NoteRealQueue(head, why);
			}
		}
	}
}

static void Phase53FlushWorkerFiberLogs() {
	const int wp = g_phase53_worker_pending.exchange(0, std::memory_order_acq_rel);
	if (wp != 0) {
		const uint64_t n   = g_phase53_worker_hits.load(std::memory_order_relaxed);
		const uint64_t rdi = g_phase53_worker_rdi.load(std::memory_order_relaxed);
		const uint64_t rsi = g_phase53_worker_rsi.load(std::memory_order_relaxed);
		const uint64_t rdx = g_phase53_worker_rdx.load(std::memory_order_relaxed);
		const uint32_t eax = g_phase53_worker_eax.load(std::memory_order_relaxed);
		const bool     empty_esi = static_cast<uint32_t>(rsi) == 0xffffffffu;
		if (n <= 32 || empty_esi) {
			LOGF("FlipTrace: phase53 worker_%s n=%" PRIu64 " rdi=0x%016" PRIx64
			     " rsi=0x%016" PRIx64 " rdx=0x%016" PRIx64 " eax=%u esi_m1=%d "
			     "[*rdi]=0x%016" PRIx64 " tid=%d\n",
			     wp >= 2 ? "exit" : "enter", n, rdi, rsi, rdx, eax, empty_esi ? 1 : 0,
			     *reinterpret_cast<uint64_t*>(g_phase53_worker_snap),
			     Common::Thread::GetThreadIdUnique());
			fprintf(stderr,
			        "FlipTrace: phase53 worker_%s n=%" PRIu64 " rdi=0x%016" PRIx64
			        " esi_m1=%d\n",
			        wp >= 2 ? "exit" : "enter", n, rdi, empty_esi ? 1 : 0);
			if (empty_esi) {
				LOGF("FlipTrace: phase53 worker_empty n=%" PRIu64 " rdi=0x%016" PRIx64 "\n", n,
				     rdi);
				fprintf(stderr, "FlipTrace: phase53 worker_empty\n");
			}
			Phase53ScanArgsForQueue(rdi, rsi, rdx, "worker");
		}
	}
	const int fp = g_phase53_fiber_pending.exchange(0, std::memory_order_acq_rel);
	if (fp != 0) {
		const uint64_t n   = g_phase53_fiber_hits.load(std::memory_order_relaxed);
		const uint64_t rdi = g_phase53_fiber_rdi.load(std::memory_order_relaxed);
		const uint64_t rsi = g_phase53_fiber_rsi.load(std::memory_order_relaxed);
		const uint64_t rdx = g_phase53_fiber_rdx.load(std::memory_order_relaxed);
		const uint32_t eax = g_phase53_fiber_eax.load(std::memory_order_relaxed);
		if (n <= 32) {
			LOGF("FlipTrace: phase53 fiber_%s n=%" PRIu64 " rdi=0x%016" PRIx64
			     " rsi=0x%016" PRIx64 " rdx=0x%016" PRIx64 " eax=%u [*rdi]=0x%016" PRIx64
			     " tid=%d\n",
			     fp >= 2 ? "exit" : "enter", n, rdi, rsi, rdx, eax,
			     *reinterpret_cast<uint64_t*>(g_phase53_fiber_snap),
			     Common::Thread::GetThreadIdUnique());
			fprintf(stderr, "FlipTrace: phase53 fiber_%s n=%" PRIu64 " rdi=0x%016" PRIx64 "\n",
			        fp >= 2 ? "exit" : "enter", n, rdi);
			Phase53ScanArgsForQueue(rdi, rsi, rdx, "fiber");
		}
	}
}

void Phase53ScanObjDeep(const char* why) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const uint32_t sn = g_phase53_deep_n.fetch_add(1, std::memory_order_relaxed);
	if (sn >= 6) {
		return;
	}
	const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL) {
		return;
	}
	LOGF("FlipTrace: phase53 deep_scan why=%s obj=0x%016" PRIx64 " range=+0x100..+0x%x n=%u\n",
	     why != nullptr ? why : "?", obj, kPhase53DeepEnd, sn);
	fprintf(stderr, "FlipTrace: phase53 deep_scan n=%u\n", sn);
	for (uint32_t off = 0x100; off + 8 <= kPhase53DeepEnd; off += 8) {
		uint64_t v = 0;
		Phase41SafeRead(&v, reinterpret_cast<const void*>(obj + off), sizeof(v));
		if (!Phase52IsGuestPtr(v)) {
			continue;
		}
		uint8_t           dump[kPhase52FollowBytes] {};
		Phase52FollowKind kind = Phase52ClassifyNode(v, dump);
		const char*       kn   = kind == Phase52FollowKind::StrongList
		                             ? "strong_list"
		                             : (kind == Phase52FollowKind::WeakList ? "weak_list"
		                                                                   : "guest_ptr");
		const uint32_t hn = g_phase53_deep_hit_n.fetch_add(1, std::memory_order_relaxed);
		if (hn < 48) {
			LOGF("FlipTrace: phase53 deep_hit why=%s off=0x%x ptr=0x%016" PRIx64 " kind=%s "
			     "tid=%d\n",
			     why != nullptr ? why : "?", off, v, kn, Common::Thread::GetThreadIdUnique());
			if (hn < 24) {
				fprintf(stderr,
				        "FlipTrace: phase53 deep_hit off=0x%x ptr=0x%016" PRIx64 " kind=%s\n",
				        off, v, kn);
			}
		}
		if (kind == Phase52FollowKind::StrongList || kind == Phase52FollowKind::WeakList) {
			Phase53NoteRealQueue(v, "deep");
		}
	}
}

void Phase53DumpStatusRdi(const char* why) {
	if (g_phase53_status_dumped.load(std::memory_order_acquire)) {
		return;
	}
	const uint64_t status = g_phase41_status_rdi.load(std::memory_order_acquire);
	if (status < 0x10000ULL || status >= 0x0000800000000000ULL) {
		static std::atomic<uint32_t> inv_n {0};
		if (inv_n.fetch_add(1, std::memory_order_relaxed) < 4) {
			LOGF("FlipTrace: phase53 status_dump why=%s status=invalid\n",
			     why != nullptr ? why : "?");
		}
		return;
	}
	if (g_phase53_status_dumped.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	uint8_t buf[256] {};
	Phase41SafeRead(buf, reinterpret_cast<const void*>(status), sizeof(buf));
	char hex[520];
	int  hpos = 0;
	for (size_t i = 0; i < 64 && hpos >= 0 && static_cast<size_t>(hpos) + 3 < sizeof(hex); ++i) {
		hpos += std::snprintf(hex + hpos, sizeof(hex) - static_cast<size_t>(hpos), "%02x",
		                      buf[i]);
	}
	LOGF("FlipTrace: phase53 status_dump why=%s base=0x%016" PRIx64 " head64=%s tid=%d\n",
	     why != nullptr ? why : "?", status, hex, Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "FlipTrace: phase53 status_dump base=0x%016" PRIx64 "\n", status);
	for (size_t off = 0; off + 8 <= 256; off += 8) {
		uint64_t v = 0;
		std::memcpy(&v, buf + off, 8);
		if (Phase52IsGuestPtr(v)) {
			Phase53NoteRealQueue(v, "status");
			LOGF("FlipTrace: phase53 status_dump guest_ptr +0x%zx=0x%016" PRIx64 "\n", off, v);
		}
	}
}

void Phase53DumpUserAnchors(const char* why) {
	if (g_phase53_user_dumped.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint64_t addrs[2] = {0x0000001020000000ULL, 0x0000001020200000ULL};
	for (uint64_t addr: addrs) {
		uint8_t buf[128] {};
		Phase41SafeRead(buf, reinterpret_cast<const void*>(addr), sizeof(buf));
		char hex[400];
		int  hpos = 0;
		for (size_t i = 0; i < 64 && hpos >= 0 && static_cast<size_t>(hpos) + 3 < sizeof(hex);
		     ++i) {
			hpos += std::snprintf(hex + hpos, sizeof(hex) - static_cast<size_t>(hpos), "%02x",
			                      buf[i]);
		}
		LOGF("FlipTrace: phase53 user_dump why=%s addr=0x%016" PRIx64 " head64=%s tid=%d\n",
		     why != nullptr ? why : "?", addr, hex, Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "FlipTrace: phase53 user_dump addr=0x%016" PRIx64 "\n", addr);
	}
}

void Phase53ProbeRetarget(const char* why) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	Phase53FlushWorkerFiberLogs();
	Phase53ScanObjDeep(why);
	Phase53DumpStatusRdi(why);
	Phase53DumpUserAnchors(why);
}

extern "C" void Phase53WorkerOnEntry(uint64_t guest_rdi, uint64_t guest_rsi, uint64_t guest_rdx) {
	const uint64_t n = g_phase53_worker_hits.fetch_add(1, std::memory_order_relaxed);
	g_phase53_worker_rdi.store(guest_rdi, std::memory_order_release);
	g_phase53_worker_rsi.store(guest_rsi, std::memory_order_release);
	g_phase53_worker_rdx.store(guest_rdx, std::memory_order_release);
	std::memset(g_phase53_worker_snap, 0, sizeof(g_phase53_worker_snap));
	if (guest_rdi >= 0x10000ULL && guest_rdi < 0x0000800000000000ULL) {
		Phase41SafeRead(g_phase53_worker_snap, reinterpret_cast<const void*>(guest_rdi), 64);
	}
	if (n < 32 || static_cast<uint32_t>(guest_rsi) == 0xffffffffu) {
		g_phase53_worker_pending.store(1, std::memory_order_release);
	}
}

extern "C" void Phase53WorkerOnExit(uint64_t guest_rdi, uint32_t eax) {
	(void)guest_rdi;
	g_phase53_worker_eax.store(eax, std::memory_order_release);
	g_phase53_worker_pending.store(2, std::memory_order_release);
}

extern "C" void Phase53FiberOnEntry(uint64_t guest_rdi, uint64_t guest_rsi, uint64_t guest_rdx) {
	const uint64_t n = g_phase53_fiber_hits.fetch_add(1, std::memory_order_relaxed);
	g_phase53_fiber_rdi.store(guest_rdi, std::memory_order_release);
	g_phase53_fiber_rsi.store(guest_rsi, std::memory_order_release);
	g_phase53_fiber_rdx.store(guest_rdx, std::memory_order_release);
	std::memset(g_phase53_fiber_snap, 0, sizeof(g_phase53_fiber_snap));
	if (guest_rdi >= 0x10000ULL && guest_rdi < 0x0000800000000000ULL) {
		Phase41SafeRead(g_phase53_fiber_snap, reinterpret_cast<const void*>(guest_rdi), 64);
	}
	if (n < 32) {
		g_phase53_fiber_pending.store(1, std::memory_order_release);
	}
}

extern "C" void Phase53FiberOnExit(uint64_t guest_rdi, uint32_t eax) {
	(void)guest_rdi;
	g_phase53_fiber_eax.store(eax, std::memory_order_release);
	g_phase53_fiber_pending.store(2, std::memory_order_release);
}

static void Phase52StartTimeSeriesThread() {
	if (g_phase52_ts_started.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	std::thread([]() {
		LOGF("FlipTrace: phase52 ts thread start (100ms, ~180s)\n");
		fprintf(stderr, "FlipTrace: phase52 ts thread start\n");
		for (int i = 0; i < 1800; ++i) {
			const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
			uint64_t       head = 0;
			uint64_t       q10  = 0;
			uint64_t       q40  = 0;
			uint64_t       slot0 = 0;
			if (obj >= 0x10000ULL && obj < 0x0000800000000000ULL) {
				Phase41SafeRead(&head, reinterpret_cast<const void*>(obj), sizeof(head));
				Phase41SafeRead(&q10, reinterpret_cast<const void*>(obj + 0x10), sizeof(q10));
				Phase41SafeRead(&q40, reinterpret_cast<const void*>(obj + 0x40), sizeof(q40));
				Phase41SafeRead(&slot0,
				                reinterpret_cast<const void*>(obj + kPhase41Keep1FieldOff + 8),
				                sizeof(slot0));
			}
			const uint64_t submit_gpu =
			    g_phase52_submit_gpu_mirror.load(std::memory_order_relaxed);
			const int guest_draw =
			    g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0;
			// Log every 500ms (every 5th sample) + first 8.
			if (i < 8 || (i % 5) == 0) {
				LOGF("FlipTrace: phase52 ts i=%d *obj=0x%016" PRIx64 " +0x10=0x%016" PRIx64
				     " +0x40=0x%016" PRIx64 " slot0=0x%016" PRIx64 " submit_gpu=%" PRIu64
				     " guest_draw=%d\n",
				     i, head, q10, q40, slot0, submit_gpu, guest_draw);
				if (i < 8 || (i % 50) == 0) {
					fprintf(stderr,
					        "FlipTrace: phase52 ts i=%d *obj=0x%016" PRIx64 " guest_draw=%d\n",
					        i, head, guest_draw);
				}
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		LOGF("FlipTrace: phase52 ts thread done give_up=%d attempts=%u strong=%d cand=0x%016"
		     PRIx64 "\n",
		     g_phase52_give_up.load(std::memory_order_relaxed) ? 1 : 0,
		     g_phase52_head_attempts.load(std::memory_order_relaxed),
		     g_phase52_candidate_strong.load(std::memory_order_relaxed) ? 1 : 0,
		     g_phase52_best_candidate.load(std::memory_order_relaxed));
		LOGF("FlipTrace: phase53 summary worker_hits=%" PRIu64 " fiber_hits=%" PRIu64
		     " real_queue=0x%016" PRIx64 " deep_hits=%u guest_draw=%d\n",
		     g_phase53_worker_hits.load(std::memory_order_relaxed),
		     g_phase53_fiber_hits.load(std::memory_order_relaxed),
		     g_phase53_real_queue.load(std::memory_order_relaxed),
		     g_phase53_deep_hit_n.load(std::memory_order_relaxed),
		     g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0);
		fprintf(stderr, "FlipTrace: phase52 ts thread done\n");
		fprintf(stderr, "FlipTrace: phase53 summary worker_hits=%" PRIu64 " fiber_hits=%" PRIu64
		                " real_queue=0x%016" PRIx64 "\n",
		        g_phase53_worker_hits.load(std::memory_order_relaxed),
		        g_phase53_fiber_hits.load(std::memory_order_relaxed),
		        g_phase53_real_queue.load(std::memory_order_relaxed));
		Phase55EmitHeatmap("ts_end");
		g_phase59_heatmap_done.store(false, std::memory_order_release);
		Phase59EmitHeatmap("ts_end");
	}).detach();
}

static void Phase50StartObjPollThread() {
	if (g_phase50_poll_thread_started.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	Phase52StartTimeSeriesThread();
	std::thread([]() {
		LOGF("FlipTrace: phase50 obj_poll thread start (2s post-Unreg)\n");
		fprintf(stderr, "FlipTrace: phase50 obj_poll thread start\n");
		for (int i = 0; i < 40; ++i) {
			Phase50PollKeep1Obj(i < 4 ? "post_unreg_early" : "post_unreg_2s");
			if ((i % 4) == 0) {
				Phase52NoteAfterDump(i < 4 ? "poll_early" : "poll_2s");
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
		Phase50PollKeep1Obj("post_unreg_done");
		Phase52NoteAfterDump("poll_done");
		LOGF("FlipTrace: phase50 obj_poll thread done nonzero=%d\n",
		     g_phase50_obj_nonzero.load(std::memory_order_relaxed) ? 1 : 0);
		fprintf(stderr, "FlipTrace: phase50 obj_poll thread done nonzero=%d\n",
		        g_phase50_obj_nonzero.load(std::memory_order_relaxed) ? 1 : 0);
	}).detach();
}

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
	uint64_t head = 0;
	Phase41SafeRead(&head, reinterpret_cast<const void*>(guest_rdi), sizeof(head));
	// Empty / poisoned: always HLE (live AV). No logging here — guest trampoline thread.
	if (head == 0 || head == ~uint64_t {0}) {
		if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
			if (head == ~uint64_t {0}) {
				const uint64_t zero = 0;
				Phase41SafeWrite(reinterpret_cast<volatile void*>(guest_rdi), &zero, sizeof(zero));
			}
			return 1;
		}
		const uint64_t neg1 = ~uint64_t {0};
		Phase41SafeWrite(reinterpret_cast<volatile void*>(guest_rdi), &neg1, sizeof(neg1));
		return 1;
	}
	// Non-empty: KYTY_PHASE41_LIVE_KEEP1=1 forces live; else gate on VO after Unreg.
	const char* live = std::getenv("KYTY_PHASE41_LIVE_KEEP1");
	if (live != nullptr && live[0] == '1') {
		return 0;
	}
	if (!g_phase37_post_unreg.load(std::memory_order_acquire) ||
	    Phase49CanLiveKeep1(guest_rdi)) {
		return 0;
	}
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
		if (obj >= 0x10000ULL && obj < 0x0000800000000000ULL) {
			Phase49LogKeep1Diag(obj, "enter_flush");
			if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
				Phase58NoteNdJobAncre("keep1_enter");
			}
		}
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
	Phase51CheckWorkerFailfast(pending >= 2 ? "keep1_exit" : "keep1_enter");
	Phase53FlushWorkerFiberLogs();
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
#endif // KYTY_PLATFORM_WINDOWS — Phase41InstallKeep1Trace

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
using Phase53EntryFn = void (*)(uint64_t, uint64_t, uint64_t);
using Phase53ExitFn  = void (*)(uint64_t, uint32_t);

static bool Phase53InstallPassThroughTrace(uint64_t target, uint8_t** thunk_slot, uint8_t** live_slot,
                                           Phase53EntryFn on_entry, Phase53ExitFn on_exit,
                                           const char* tag) {
	if (target < 0x1000 || thunk_slot == nullptr || live_slot == nullptr) {
		return false;
	}
	if (*thunk_slot != nullptr) {
		return true;
	}
	auto*   guest = reinterpret_cast<uint8_t*>(target);
	uint8_t orig[kPhase53StealBytes] {};
	Phase41SafeRead(orig, guest, kPhase53StealBytes);

	uint8_t* live = static_cast<uint8_t*>(
	    VirtualAlloc(nullptr, 64, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	uint8_t* thunk = static_cast<uint8_t*>(
	    VirtualAlloc(nullptr, 256, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	if (live == nullptr || thunk == nullptr) {
		return false;
	}
	std::memcpy(live, orig, kPhase53StealBytes);
	live[kPhase53StealBytes]     = 0x48;
	live[kPhase53StealBytes + 1] = 0xb8;
	const uint64_t cont          = target + kPhase53StealBytes;
	std::memcpy(live + kPhase53StealBytes + 2, &cont, 8);
	live[kPhase53StealBytes + 10] = 0xff;
	live[kPhase53StealBytes + 11] = 0xe0;

	uint8_t* p = thunk;
	auto emit = [&](std::initializer_list<uint8_t> bytes) {
		for (uint8_t b: bytes) {
			*p++ = b;
		}
	};
	auto emit_u64 = [&](uint64_t v) {
		std::memcpy(p, &v, 8);
		p += 8;
	};
	emit({0x53});             // push rbx
	emit({0x55});             // push rbp
	emit({0x41, 0x54});       // push r12
	emit({0x41, 0x55});       // push r13
	emit({0x48, 0x89, 0xfb}); // mov rbx, rdi
	emit({0x48, 0x89, 0xf5}); // mov rbp, rsi
	emit({0x49, 0x89, 0xd4}); // mov r12, rdx
	emit({0x48, 0x89, 0xd9}); // mov rcx, rbx
	emit({0x48, 0x89, 0xea}); // mov rdx, rbp
	emit({0x4d, 0x89, 0xe0}); // mov r8, r12
	emit({0x48, 0x83, 0xec, 0x20});
	emit({0x48, 0xb8});
	emit_u64(reinterpret_cast<uint64_t>(on_entry));
	emit({0xff, 0xd0});
	emit({0x48, 0x83, 0xc4, 0x20});
	// always call original (pass-through)
	emit({0x48, 0x89, 0xdf}); // mov rdi, rbx
	emit({0x48, 0x89, 0xee}); // mov rsi, rbp
	emit({0x4c, 0x89, 0xe2}); // mov rdx, r12
	emit({0x48, 0x83, 0xec, 0x20});
	emit({0x48, 0xb8});
	emit_u64(reinterpret_cast<uint64_t>(live));
	emit({0xff, 0xd0});
	emit({0x48, 0x83, 0xc4, 0x20});
	emit({0x41, 0x89, 0xc5}); // mov r13d, eax
	emit({0x48, 0x89, 0xd9}); // mov rcx, rbx
	emit({0x44, 0x89, 0xea}); // mov edx, r13d
	emit({0x48, 0x83, 0xec, 0x20});
	emit({0x48, 0xb8});
	emit_u64(reinterpret_cast<uint64_t>(on_exit));
	emit({0xff, 0xd0});
	emit({0x48, 0x83, 0xc4, 0x20});
	emit({0x44, 0x89, 0xe8}); // mov eax, r13d
	emit({0x41, 0x5d});
	emit({0x41, 0x5c});
	emit({0x5d});
	emit({0x5b});
	emit({0xc3});

	DWORD old_prot = 0;
	if (VirtualProtect(guest, 16, PAGE_EXECUTE_READWRITE, &old_prot) == 0) {
		return false;
	}
	guest[0] = 0x48;
	guest[1] = 0xb8;
	const uint64_t th = reinterpret_cast<uint64_t>(thunk);
	std::memcpy(guest + 2, &th, 8);
	guest[10] = 0xff;
	guest[11] = 0xe0;
	DWORD unused = 0;
	VirtualProtect(guest, 16, old_prot, &unused);

	*thunk_slot = thunk;
	*live_slot  = live;
	LOGF("FlipTrace: phase53 %s trampoline installed tgt=0x%016" PRIx64 " thunk=%p\n", tag,
	     target, static_cast<void*>(thunk));
	fprintf(stderr, "FlipTrace: phase53 %s trampoline installed\n", tag);
	return true;
}

extern "C" void Phase55MixedOnEntry(uint64_t guest_rdi, uint64_t guest_rsi, uint64_t guest_rdx);
extern "C" void Phase55MixedOnExit(uint64_t guest_rdi, uint32_t eax);

static void Phase53ArmWorkerFiberTraces() {
	if (g_phase53_traces_armed.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	(void)Phase53InstallPassThroughTrace(kPhase41NdJobWorker, &g_phase53_worker_thunk,
	                                     &g_phase53_worker_live, &Phase53WorkerOnEntry,
	                                     &Phase53WorkerOnExit, "worker");
	(void)Phase53InstallPassThroughTrace(kPhase41NdJobFiberEnt, &g_phase53_fiber_thunk,
	                                     &g_phase53_fiber_live, &Phase53FiberOnEntry,
	                                     &Phase53FiberOnExit, "fiber");
	if (!g_phase55_traces_armed.exchange(true, std::memory_order_acq_rel)) {
		(void)Phase53InstallPassThroughTrace(kPhase55MixedEntry, &g_phase55_mixed_thunk,
		                                     &g_phase55_mixed_live, &Phase55MixedOnEntry,
		                                     &Phase55MixedOnExit, "mixed_entry");
	}
}
#else
static void Phase53ArmWorkerFiberTraces() {
	if (g_phase53_traces_armed.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	LOGF("FlipTrace: phase53 worker/fiber trampolines skipped (non-Windows)\n");
}
#endif

void Phase55TryArmMixedThunk() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Re-enter through Phase53Arm so InstallPassThrough is in translation unit scope.
	Phase53ArmWorkerFiberTraces();
#else
	if (g_phase55_traces_armed.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	LOGF("SubmitTrace: phase55 mixed_entry trampoline skipped (non-Windows)\n");
#endif
}

extern "C" void Phase55MixedOnEntry(uint64_t guest_rdi, uint64_t guest_rsi, uint64_t guest_rdx) {
	const uint64_t n = g_phase55_entry_hits.fetch_add(1, std::memory_order_relaxed);
	g_phase55_mixed_rdi.store(guest_rdi, std::memory_order_release);
	g_phase55_mixed_rsi.store(guest_rsi, std::memory_order_release);
	g_phase55_mixed_rdx.store(guest_rdx, std::memory_order_release);
	// At trampoline 0x901DE4140: rdi = *slot + 0x1920 → recover module data base.
	if (guest_rdi >= 0x10000ULL + 0x1920ULL && guest_rdi < 0x0000800000000000ULL) {
		const uint64_t cand = guest_rdi - 0x1920ULL;
		if (cand >= 0x10000ULL) {
			g_phase70_resolved_base.store(cand, std::memory_order_release);
		}
	}
	std::memset(g_phase55_mixed_snap, 0, sizeof(g_phase55_mixed_snap));
	if (guest_rdi >= 0x10000ULL && guest_rdi < 0x0000800000000000ULL) {
		Phase41SafeRead(g_phase55_mixed_snap, reinterpret_cast<const void*>(guest_rdi), 64);
		// Phase 56 elects LIST_CANDIDATE; do not blindly watch Mixed rdi.
	}
	if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
		Phase57NoteMixedRegs(guest_rdi, guest_rsi, guest_rdx);
		Phase57TryScanMixedBody();
	}
	if (g_phase37_post_unreg.load(std::memory_order_acquire) &&
	    !g_phase55_first_hit_after_unreg.exchange(true, std::memory_order_acq_rel)) {
		g_phase55_mixed_pending.store(3, std::memory_order_release);
	} else if (n < 48) {
		g_phase55_mixed_pending.store(1, std::memory_order_release);
	}
}

extern "C" void Phase55MixedOnExit(uint64_t guest_rdi, uint32_t eax) {
	(void)guest_rdi;
	g_phase55_mixed_eax.store(eax, std::memory_order_release);
	g_phase55_mixed_pending.store(2, std::memory_order_release);
}

bool Phase55FakeQueueEnabled() {
	static int cached = -1;
	if (cached < 0) {
		const char* e = std::getenv("KYTY_PHASE55_FAKE_QUEUE");
		cached        = (e != nullptr && e[0] == '1') ? 1 : 0;
	}
	return cached == 1;
}

static const char* Phase55KindOfQword(uint64_t v) {
	if (Phase52IsGuestPtr(v)) {
		return "guest_ptr";
	}
	if (Phase52IsUserPtr(v)) {
		return "user_ptr";
	}
	return "scalar";
}

static void Phase55SetVerdict(const char* v) {
	if (v == nullptr) {
		return;
	}
	std::snprintf(g_phase55_last_verdict, sizeof(g_phase55_last_verdict), "%s", v);
}

static void Phase55QueueDump(const char* why, uint64_t base) {
	if (base < 0x10000ULL || base >= 0x0000800000000000ULL) {
		return;
	}
	uint8_t buf[256] {};
	Phase41SafeRead(buf, reinterpret_cast<const void*>(base), sizeof(buf));
	const uint32_t run = g_phase55_dump_runs.fetch_add(1, std::memory_order_relaxed) + 1;

	uint32_t guest_n = 0;
	uint32_t user_n  = 0;
	uint32_t scalar_n = 0;
	int      count_off = -1;
	int      head_off  = -1;
	int      tail_off  = -1;
	uint64_t count_val = 0;
	uint64_t head_val  = 0;
	uint64_t tail_val  = 0;

	char kinds[512];
	int  kpos = 0;
	for (size_t off = 0; off + 8 <= sizeof(buf); off += 8) {
		uint64_t v = 0;
		std::memcpy(&v, buf + off, 8);
		const char* kind = Phase55KindOfQword(v);
		if (std::strcmp(kind, "guest_ptr") == 0) {
			++guest_n;
			if (head_off < 0) {
				head_off = static_cast<int>(off);
				head_val = v;
			} else if (tail_off < 0 && static_cast<int>(off) != head_off) {
				tail_off = static_cast<int>(off);
				tail_val = v;
			}
		} else if (std::strcmp(kind, "user_ptr") == 0) {
			++user_n;
		} else {
			++scalar_n;
			if (count_off < 0 && v < 0x10000ULL) {
				count_off = static_cast<int>(off);
				count_val = v;
			}
		}
		if (off < 64 && kpos >= 0 && static_cast<size_t>(kpos) + 24 < sizeof(kinds)) {
			kpos += std::snprintf(kinds + kpos, sizeof(kinds) - static_cast<size_t>(kpos),
			                      " +0x%zx=%s", off, kind);
		}
	}

	// Prefer stable count at +0x10/+0x18/+0x20 if small.
	for (size_t cand: {0x10ull, 0x18ull, 0x20ull, 0x28ull, 0x08ull, 0x00ull}) {
		uint64_t v = 0;
		std::memcpy(&v, buf + cand, 8);
		if (v < 0x10000ULL) {
			count_off = static_cast<int>(cand);
			count_val = v;
			break;
		}
	}

	const char* verdict = "unknown";
	if (user_n > guest_n + 4 && guest_n == 0) {
		verdict = "not_job_queue";
	} else if (count_off >= 0 && count_val == 0) {
		verdict = "empty_queue";
	} else if (count_off >= 0 && count_val > 0) {
		verdict = "live_blocked";
	} else if (guest_n >= 2) {
		verdict = (head_val == 0 && tail_val == 0) ? "empty_queue" : "unknown";
	}
	Phase55SetVerdict(verdict);
	if (count_off >= 0) {
		g_phase55_count_off.store(count_off, std::memory_order_release);
	}
	if (head_off >= 0) {
		g_phase55_head_off.store(head_off, std::memory_order_release);
	}
	if (tail_off >= 0) {
		g_phase55_tail_off.store(tail_off, std::memory_order_release);
	}
	g_phase55_watch_count.store(count_val, std::memory_order_release);
	g_phase55_watch_head.store(head_val, std::memory_order_release);
	g_phase55_watch_tail.store(tail_val, std::memory_order_release);
	g_phase55_watch_armed.store(true, std::memory_order_release);

	if (g_phase55_have_prev_dump) {
		for (size_t off = 0; off + 8 <= sizeof(buf); off += 8) {
			uint64_t b = 0;
			uint64_t a = 0;
			std::memcpy(&b, g_phase55_prev_dump + off, 8);
			std::memcpy(&a, buf + off, 8);
			if (b != a) {
				const uint32_t wn =
				    g_phase55_enqueue_write_n.fetch_add(1, std::memory_order_relaxed) + 1;
				if (wn <= 32 || (wn % 16) == 0) {
					LOGF("SubmitTrace: phase55 enqueue_write off=0x%zx old=0x%016" PRIx64
					     " new=0x%016" PRIx64 " n=%u why=%s\n",
					     off, b, a, wn, why != nullptr ? why : "?");
					fprintf(stderr,
					        "SubmitTrace: phase55 enqueue_write off=0x%zx n=%u\n", off, wn);
				}
			}
		}
	}
	std::memcpy(g_phase55_prev_dump, buf, sizeof(buf));
	g_phase55_have_prev_dump = true;

	if (run <= 24 || (run % 16) == 0) {
		char hex[200];
		int  hpos = 0;
		for (size_t i = 0; i < 32 && hpos >= 0 && static_cast<size_t>(hpos) + 3 < sizeof(hex);
		     ++i) {
			hpos += std::snprintf(hex + hpos, sizeof(hex) - static_cast<size_t>(hpos), "%02x",
			                      buf[i]);
		}
		LOGF("SubmitTrace: phase55 queue_dump why=%s base=0x%016" PRIx64 " run=%u "
		     "verdict=%s count_off=%d count=%" PRIu64 " head_off=%d head=0x%016" PRIx64
		     " tail_off=%d guest_n=%u user_n=%u scalar_n=%u head32=%s kinds=%s tid=%d\n",
		     why != nullptr ? why : "?", base, run, verdict, count_off, count_val, head_off,
		     head_val, tail_off, guest_n, user_n, scalar_n, hex, kinds,
		     Common::Thread::GetThreadIdUnique());
		fprintf(stderr,
		        "SubmitTrace: phase55 queue_layout verdict=%s base=0x%016" PRIx64 " count=%" PRIu64
		        " run=%u\n",
		        verdict, base, count_val, run);
	}

	if (Phase55FakeQueueEnabled() && std::strcmp(verdict, "empty_queue") == 0) {
		Phase55TryFakeQueueInject(why);
	}
}

void Phase55OnMixedRewait(const char* role, uintptr_t cond_ptr) {
	(void)cond_ptr;
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	if (!g_phase55_role_logged.exchange(true, std::memory_order_acq_rel)) {
		LOGF("SubmitTrace: phase55 role=%s phase=post_unreg tid=%d\n",
		     role != nullptr ? role : "?", Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase55 role=%s phase=post_unreg\n",
		        role != nullptr ? role : "?");
	}
	Phase55FlushDeferredLogs();
	uint64_t base = g_phase55_queue_base.load(std::memory_order_acquire);
	if (base < 0x10000ULL) {
		base = g_phase55_mixed_rdi.load(std::memory_order_acquire);
	}
	if (base >= 0x10000ULL) {
		Phase55QueueDump(role != nullptr ? role : "rewait", base);
	}
}

void Phase55PollWatch() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire) ||
	    !g_phase55_watch_armed.load(std::memory_order_acquire)) {
		return;
	}
	const uint64_t base = g_phase55_queue_base.load(std::memory_order_acquire);
	const int      coff = g_phase55_count_off.load(std::memory_order_acquire);
	const int      hoff = g_phase55_head_off.load(std::memory_order_acquire);
	const int      toff = g_phase55_tail_off.load(std::memory_order_acquire);
	if (base < 0x10000ULL) {
		return;
	}
	auto check = [&](int off, std::atomic<uint64_t>& slot, const char* tag) {
		if (off < 0) {
			return;
		}
		uint64_t cur = 0;
		Phase41SafeRead(&cur, reinterpret_cast<const void*>(base + static_cast<uint64_t>(off)),
		                sizeof(cur));
		const uint64_t prev = slot.exchange(cur, std::memory_order_acq_rel);
		if (prev != cur && prev != 0) {
			const uint32_t wn =
			    g_phase55_enqueue_write_n.fetch_add(1, std::memory_order_relaxed) + 1;
			if (wn <= 48) {
				LOGF("SubmitTrace: phase55 enqueue_write tag=%s off=0x%x old=0x%016" PRIx64
				     " new=0x%016" PRIx64 " n=%u\n",
				     tag, off, prev, cur, wn);
				fprintf(stderr, "SubmitTrace: phase55 enqueue_write tag=%s n=%u\n", tag, wn);
			}
		} else if (prev != cur) {
			slot.store(cur, std::memory_order_release);
			if (prev == 0 && cur != 0) {
				const uint32_t wn =
				    g_phase55_enqueue_write_n.fetch_add(1, std::memory_order_relaxed) + 1;
				LOGF("SubmitTrace: phase55 enqueue_write tag=%s off=0x%x old=0 new=0x%016" PRIx64
				     " n=%u\n",
				     tag, off, cur, wn);
				fprintf(stderr, "SubmitTrace: phase55 enqueue_write tag=%s n=%u\n", tag, wn);
			}
		}
	};
	check(coff, g_phase55_watch_count, "count");
	check(hoff, g_phase55_watch_head, "head");
	check(toff, g_phase55_watch_tail, "tail");
}

void Phase55TryFakeQueueInject(const char* why) {
	if (!Phase55FakeQueueEnabled() || !g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	if (g_phase55_entry_hits.load(std::memory_order_acquire) == 0) {
		return;
	}
	if (std::strcmp(g_phase55_last_verdict, "empty_queue") != 0) {
		return;
	}
	if (g_phase55_fake_queue_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint64_t base = g_phase55_queue_base.load(std::memory_order_acquire);
	const int      coff = g_phase55_count_off.load(std::memory_order_acquire);
	const int      hoff = g_phase55_head_off.load(std::memory_order_acquire);
	const int      toff = g_phase55_tail_off.load(std::memory_order_acquire);
	if (base < 0x10000ULL || coff < 0) {
		Phase55SetVerdict("wrong_queue_or_predicate");
		LOGF("SubmitTrace: phase55 fake_queue skip no_base/count why=%s\n",
		     why != nullptr ? why : "?");
		return;
	}
	// Stub node lives inside the same guest buffer at +0x100 (away from head fields).
	const uint64_t node = base + 0x100;
	uint8_t        stub[64] {};
	std::memset(stub, 0, sizeof(stub));
	Phase41SafeWrite(reinterpret_cast<volatile void*>(node), stub, sizeof(stub));
	const uint64_t one = 1;
	Phase41SafeWrite(reinterpret_cast<volatile void*>(base + static_cast<uint64_t>(coff)), &one,
	                 sizeof(one));
	if (hoff >= 0) {
		Phase41SafeWrite(reinterpret_cast<volatile void*>(base + static_cast<uint64_t>(hoff)),
		                 &node, sizeof(node));
	}
	if (toff >= 0) {
		Phase41SafeWrite(reinterpret_cast<volatile void*>(base + static_cast<uint64_t>(toff)),
		                 &node, sizeof(node));
	}
	LOGF("SubmitTrace: phase55 fake_queue inject base=0x%016" PRIx64 " count_off=0x%x "
	     "head_off=%d node=0x%016" PRIx64 " why=%s tid=%d\n",
	     base, coff, hoff, node, why != nullptr ? why : "?",
	     Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "SubmitTrace: phase55 fake_queue inject base=0x%016" PRIx64 "\n", base);
	Phase55SetVerdict("fake_injected");
	(void)LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
}

void Phase55NoteMainWakeAlt(const char* kind, uint64_t a0, uint64_t a1) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	if (!LibKernel::PthreadCurrentIsMainRelated()) {
		return;
	}
	const uint32_t n = g_phase55_main_wake_alt_n.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 48 || (n % 32) == 0) {
		LOGF("SubmitTrace: phase55 main_wake_alt kind=%s a0=0x%016" PRIx64 " a1=0x%016" PRIx64
		     " n=%u cycle=%" PRIu64 " tid=%d\n",
		     kind != nullptr ? kind : "?", a0, a1, n, Phase54CurrentCycleId(),
		     Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase55 main_wake_alt kind=%s n=%u\n",
		        kind != nullptr ? kind : "?", n);
	}
	Phase55PollWatch();
}

void Phase55FlushDeferredLogs() {
	const int p = g_phase55_mixed_pending.exchange(0, std::memory_order_acq_rel);
	if (p == 0) {
		return;
	}
	const uint64_t n     = g_phase55_entry_hits.load(std::memory_order_relaxed);
	const uint64_t rdi   = g_phase55_mixed_rdi.load(std::memory_order_relaxed);
	const uint64_t rsi   = g_phase55_mixed_rsi.load(std::memory_order_relaxed);
	const uint64_t rdx   = g_phase55_mixed_rdx.load(std::memory_order_relaxed);
	const uint32_t eax   = g_phase55_mixed_eax.load(std::memory_order_relaxed);
	const uint64_t cycle = Phase54CurrentCycleId();
	const int first = (p == 3) ? 1 : 0;
	const char* role = (static_cast<uint32_t>(rdi) == 0)   ? "Mixed"
	                   : (static_cast<uint32_t>(rdi) == 1) ? "Compute"
	                                                       : "?";
	if (!g_phase55_role_logged.exchange(true, std::memory_order_acq_rel)) {
		LOGF("SubmitTrace: phase55 role=%s tid=%d phase=post_unreg arg=%u\n", role,
		     Common::Thread::GetThreadIdUnique(), static_cast<uint32_t>(rdi));
		fprintf(stderr, "SubmitTrace: phase55 role=%s phase=post_unreg\n", role);
	}
	LOGF("SubmitTrace: phase55 mixed_entry n=%" PRIu64 " cycle=%" PRIu64
	     " first_hit_after_unreg=%d role=%s rdi=0x%016" PRIx64 " rsi=0x%016" PRIx64
	     " rdx=0x%016" PRIx64 " rax=0x%x [*rdi]=0x%016" PRIx64 " pending=%d tid=%d\n",
	     n, cycle, first, role, rdi, rsi, rdx, eax,
	     *reinterpret_cast<uint64_t*>(g_phase55_mixed_snap), p,
	     Common::Thread::GetThreadIdUnique());
	fprintf(stderr,
	        "SubmitTrace: phase55 mixed_entry n=%" PRIu64 " first=%d role=%s\n", n, first, role);
	// pthread entry arg is 0/1 — not a queue VA; prefer guest_cond / rdx if set.
	uint64_t base = g_phase55_queue_base.load(std::memory_order_acquire);
	if (base < 0x10000ULL && rdx >= 0x10000ULL) {
		base = rdx;
	}
	if (base >= 0x10000ULL && Phase52IsGuestPtr(base)) {
		Phase55QueueDump("mixed_entry", base);
	}
}

void Phase55NoteGuestCond(uint64_t guest_cond_va, uint64_t guest_arg, const char* role) {
	if (guest_cond_va < 0x10000ULL) {
		return;
	}
	static std::atomic<uint32_t> logs {0};
	const uint32_t               n = logs.fetch_add(1, std::memory_order_relaxed);
	if (n < 24) {
		LOGF("SubmitTrace: phase55 guest_cond va=0x%016" PRIx64 " arg=%" PRIu64 " role=%s tid=%d\n",
		     guest_cond_va, guest_arg, role != nullptr ? role : "?",
		     Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase55 guest_cond va=0x%016" PRIx64 " role=%s\n",
		        guest_cond_va, role != nullptr ? role : "?");
	}
	// Phase 56: do NOT elect cond-0x40 (banned). Dump raw cond only; sync path does multi-base.
	if (guest_cond_va != kPhase56BannedBase) {
		Phase55QueueDump("guest_cond_raw", guest_cond_va);
	}
	(void)guest_arg;
	(void)role;
}

void Phase55EmitHeatmap(const char* why) {
	if (g_phase55_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	LOGF("SubmitTrace: phase55 heatmap why=%s entry_hits=%" PRIu64 " dump_runs=%u "
	     "main_wake_alt=%u enqueue_write=%u submit_guest_real=%u verdict=%s guest_draw=%d\n",
	     why != nullptr ? why : "?", g_phase55_entry_hits.load(std::memory_order_relaxed),
	     g_phase55_dump_runs.load(std::memory_order_relaxed),
	     g_phase55_main_wake_alt_n.load(std::memory_order_relaxed),
	     g_phase55_enqueue_write_n.load(std::memory_order_relaxed),
	     g_phase55_submit_guest_real_n.load(std::memory_order_relaxed), g_phase55_last_verdict,
	     g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0);
	fprintf(stderr,
	        "SubmitTrace: phase55 heatmap entry_hits=%" PRIu64 " verdict=%s dump_runs=%u\n",
	        g_phase55_entry_hits.load(std::memory_order_relaxed), g_phase55_last_verdict,
	        g_phase55_dump_runs.load(std::memory_order_relaxed));
}

static void Phase55StartWatchThread() {
	if (g_phase55_watch_started.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	std::thread([]() {
		LOGF("SubmitTrace: phase55 watch thread start (100ms, ~180s)\n");
		fprintf(stderr, "SubmitTrace: phase55 watch thread start\n");
		for (int i = 0; i < 1800; ++i) {
			Phase55FlushDeferredLogs();
			Phase55PollWatch();
			Phase56PollWatch();
			Phase57PollHeatmap();
			Phase58PollWatch();
			Phase59Poll();
			Phase61Poll();
			Phase62Poll();
			Phase63Poll();
			Phase64Poll();
			Phase65Poll();
			Phase66Poll();
			Phase69Poll();
			Phase70Poll();
			Phase71Poll();
			if ((i % 300) == 299) {
				g_phase55_heatmap_done.store(false, std::memory_order_release);
				Phase55EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase56_heatmap_done.store(false, std::memory_order_release);
				Phase56EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase57_heatmap_done.store(false, std::memory_order_release);
				Phase57EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase58_heatmap_done.store(false, std::memory_order_release);
				Phase58EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase59_heatmap_done.store(false, std::memory_order_release);
				Phase59EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase61_heatmap_done.store(false, std::memory_order_release);
				Phase61EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase62_heatmap_done.store(false, std::memory_order_release);
				Phase62EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase63_heatmap_done.store(false, std::memory_order_release);
				Phase63EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase64_heatmap_done.store(false, std::memory_order_release);
				Phase64EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase65_heatmap_done.store(false, std::memory_order_release);
				Phase65EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase66_heatmap_done.store(false, std::memory_order_release);
				Phase66EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase69_heatmap_done.store(false, std::memory_order_release);
				Phase69EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase70_heatmap_done.store(false, std::memory_order_release);
				Phase70EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
				g_phase71_heatmap_done.store(false, std::memory_order_release);
				Phase71EmitHeatmap(i < 600 ? "watch_30s" : "watch_mid");
			}
			if (i == 1799) {
				g_phase55_heatmap_done.store(false, std::memory_order_release);
				Phase55EmitHeatmap("watch_end");
				g_phase56_heatmap_done.store(false, std::memory_order_release);
				Phase56EmitHeatmap("watch_end");
				g_phase57_heatmap_done.store(false, std::memory_order_release);
				Phase57EmitHeatmap("watch_end");
				g_phase58_heatmap_done.store(false, std::memory_order_release);
				Phase58EmitHeatmap("watch_end");
				g_phase59_heatmap_done.store(false, std::memory_order_release);
				Phase59EmitHeatmap("watch_end");
				g_phase61_heatmap_done.store(false, std::memory_order_release);
				Phase61EmitHeatmap("watch_end");
				g_phase62_heatmap_done.store(false, std::memory_order_release);
				Phase62EmitHeatmap("watch_end");
				g_phase63_heatmap_done.store(false, std::memory_order_release);
				Phase63EmitHeatmap("watch_end");
				g_phase64_heatmap_done.store(false, std::memory_order_release);
				Phase64EmitHeatmap("watch_end");
				g_phase65_heatmap_done.store(false, std::memory_order_release);
				Phase65EmitHeatmap("watch_end");
				g_phase66_heatmap_done.store(false, std::memory_order_release);
				Phase66EmitHeatmap("watch_end");
				g_phase69_heatmap_done.store(false, std::memory_order_release);
				Phase69EmitHeatmap("watch_end");
				g_phase70_heatmap_done.store(false, std::memory_order_release);
				Phase70EmitHeatmap("watch_end");
				g_phase71_heatmap_done.store(false, std::memory_order_release);
				Phase71EmitHeatmap("watch_end");
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}).detach();
}

// ---------------------------------------------------------------------------
// Phase 56 — LIST_CANDIDATE retarget + writers + FAKE_COUNT
// ---------------------------------------------------------------------------

bool Phase56FakeCountEnabled() {
	static int cached = -1;
	if (cached < 0) {
		const char* e = std::getenv("KYTY_PHASE56_FAKE_COUNT");
		cached        = (e != nullptr && e[0] == '1') ? 1 : 0;
	}
	return cached == 1;
}

uint64_t Phase56CurrentSyncId() {
	return g_phase56_sync_id.load(std::memory_order_acquire);
}

uint64_t Phase56QueueBase() {
	return g_phase56_queue_base.load(std::memory_order_acquire);
}

static bool Phase56IsBannedBase(uint64_t base) {
	return base == kPhase56BannedBase;
}

static const char* Phase56WriterRole() {
	if (LibKernel::PthreadCurrentIsMainRelated()) {
		return "Main"; // includes BootCards / unique_id==8
	}
	if (LibKernel::PthreadCurrentIsSubmissionRelated()) {
		return "Mixed";
	}
	return "other";
}

static void Phase56SetCause(const char* c) {
	if (c != nullptr) {
		std::snprintf(g_phase56_cause, sizeof(g_phase56_cause), "%s", c);
	}
}

struct Phase56BaseScore {
	uint64_t    base      = 0;
	uint32_t    guest_n   = 0;
	uint32_t    user_n    = 0;
	uint32_t    scalar_n  = 0;
	int         count_off = -1;
	uint64_t    count_val = 0;
	const char* type      = "STATE_BLOCK";
	int         score     = 0;
};

static Phase56BaseScore Phase56ClassifyBase(uint64_t base) {
	Phase56BaseScore s {};
	s.base = base;
	if (base < 0x10000ULL || base >= 0x0000800000000000ULL || Phase56IsBannedBase(base)) {
		s.type = "BANNED";
		return s;
	}
	uint8_t buf[256] {};
	Phase41SafeRead(buf, reinterpret_cast<const void*>(base), sizeof(buf));
	for (size_t off = 0; off + 8 <= sizeof(buf); off += 8) {
		uint64_t v = 0;
		std::memcpy(&v, buf + off, 8);
		if (Phase52IsGuestPtr(v)) {
			++s.guest_n;
		} else if (Phase52IsUserPtr(v)) {
			++s.user_n;
		} else {
			++s.scalar_n;
		}
	}
	for (size_t cand: {0x10ull, 0x18ull, 0x20ull, 0x28ull, 0x08ull, 0x00ull}) {
		uint64_t v = 0;
		std::memcpy(&v, buf + cand, 8);
		if (v < 0x10000ULL) {
			s.count_off = static_cast<int>(cand);
			s.count_val = v;
			break;
		}
	}
	if (s.guest_n >= 2 && s.scalar_n >= 1) {
		s.type  = "LIST_CANDIDATE";
		s.score = static_cast<int>(s.guest_n) * 10 + static_cast<int>(s.scalar_n) +
		          (s.count_off >= 0 ? 5 : 0);
	} else if (s.user_n > s.guest_n + s.scalar_n) {
		s.type  = "BUFFER_ANCHOR";
		s.score = static_cast<int>(s.user_n);
	} else {
		s.type  = "STATE_BLOCK";
		s.score = 0;
	}
	return s;
}

static void Phase56ElectQueueBase(const Phase56BaseScore& cand, uint64_t sync_id) {
	if (std::strcmp(cand.type, "LIST_CANDIDATE") != 0) {
		return;
	}
	(void)g_phase56_list_cand_n.fetch_add(1, std::memory_order_relaxed);
	const int best = g_phase56_best_score.load(std::memory_order_acquire);
	if (cand.score <= best && g_phase56_base_elected.load(std::memory_order_acquire)) {
		return; // keep unique best-scoring LIST_CANDIDATE
	}
	g_phase56_best_score.store(cand.score, std::memory_order_release);
	g_phase56_queue_base.store(cand.base, std::memory_order_release);
	g_phase56_count_off.store(cand.count_off, std::memory_order_release);
	g_phase56_watch_count.store(cand.count_val, std::memory_order_release);
	g_phase56_watch_armed.store(true, std::memory_order_release);
	g_phase56_base_elected.store(true, std::memory_order_release);
	g_phase56_watch_seeded.store(false, std::memory_order_release);
	std::snprintf(g_phase56_base_type, sizeof(g_phase56_base_type), "%s", cand.type);
	// Also point Phase55 watch away from banned base.
	if (!Phase56IsBannedBase(cand.base)) {
		g_phase55_queue_base.store(cand.base, std::memory_order_release);
		g_phase55_count_off.store(cand.count_off, std::memory_order_release);
	}
	LOGF("SubmitTrace: phase56 elect queue_base=0x%016" PRIx64 " type=%s score=%d guest_n=%u "
	     "scalar_n=%u count_off=%d count=%" PRIu64 " sync_id=%" PRIu64 "\n",
	     cand.base, cand.type, cand.score, cand.guest_n, cand.scalar_n, cand.count_off,
	     cand.count_val, sync_id);
	fprintf(stderr, "SubmitTrace: phase56 elect queue_base=0x%016" PRIx64 " type=%s score=%d\n",
	        cand.base, cand.type, cand.score);
}

static void Phase56DumpAndClassify(const char* tag, uint64_t base, uint64_t sync_id) {
	if (base < 0x10000ULL || Phase56IsBannedBase(base)) {
		if (Phase56IsBannedBase(base)) {
			static std::atomic<uint32_t> ban_logs {0};
			if (ban_logs.fetch_add(1, std::memory_order_relaxed) < 4) {
				LOGF("SubmitTrace: phase56 skip_banned base=0x%016" PRIx64 " tag=%s\n", base,
				     tag != nullptr ? tag : "?");
			}
		}
		return;
	}
	const Phase56BaseScore s = Phase56ClassifyBase(base);
	const uint32_t         run =
	    g_phase56_dump_runs.fetch_add(1, std::memory_order_relaxed) + 1;
	if (run <= 32 || (run % 32) == 0 || std::strcmp(s.type, "LIST_CANDIDATE") == 0) {
		LOGF("SubmitTrace: phase56 base_class tag=%s base=0x%016" PRIx64 " type=%s guest_n=%u "
		     "user_n=%u scalar_n=%u score=%d count_off=%d count=%" PRIu64 " sync_id=%" PRIu64
		     " run=%u\n",
		     tag != nullptr ? tag : "?", base, s.type, s.guest_n, s.user_n, s.scalar_n, s.score,
		     s.count_off, s.count_val, sync_id, run);
		fprintf(stderr, "SubmitTrace: phase56 base_class type=%s base=0x%016" PRIx64 "\n", s.type,
		        base);
	}
	Phase56ElectQueueBase(s, sync_id);
}

static void Phase56ScanRipRelPtrs(uint64_t entry) {
	if (g_phase56_lea_scanned.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	if (entry < 0x1000ULL) {
		return;
	}
	DumpGuestCodeAround(entry);
	uint8_t code[256] {};
	Phase41SafeRead(code, reinterpret_cast<const void*>(entry), sizeof(code));
	g_phase56_lea_cand_n = 0;
	const uint64_t cond  = g_phase56_job_cond.load(std::memory_order_acquire);
	const uint64_t mtx   = g_phase56_job_mutex.load(std::memory_order_acquire);
	auto consider = [&](uint64_t va, const char* how) {
		if (!Phase52IsGuestPtr(va) || Phase56IsBannedBase(va)) {
			return;
		}
		if (g_phase56_lea_cand_n >= 2) {
			return; // max 1–2 lea candidates
		}
		const bool near_sync =
		    (cond != 0 && va + 0x200 >= cond && va <= cond + 0x200) ||
		    (mtx != 0 && va + 0x200 >= mtx && va <= mtx + 0x200);
		const char* role = near_sync ? "QUEUE_BASE_CAND" : "CTX_BASE";
		g_phase56_lea_cand[g_phase56_lea_cand_n++] = va;
		LOGF("SubmitTrace: phase56 lea_cand va=0x%016" PRIx64 " role=%s how=%s near_sync=%d\n",
		     va, role, how, near_sync ? 1 : 0);
		fprintf(stderr, "SubmitTrace: phase56 lea_cand va=0x%016" PRIx64 " role=%s\n", va, role);
		Phase56DumpAndClassify(role, va, Phase56CurrentSyncId());
		if (near_sync) {
			Phase56DumpAndClassify("lea_near", va, Phase56CurrentSyncId());
		}
	};
	for (size_t i = 0; i + 7 < sizeof(code); ++i) {
		// lea r64, [rip+disp32] : 48 8d xx xx xx xx xx  (modrm RIP-relative)
		if (code[i] == 0x48 && code[i + 1] == 0x8d && i + 7 < sizeof(code)) {
			const uint8_t modrm = code[i + 2];
			if ((modrm & 0xc7u) == 0x05u) { // mod=00 rm=101 → rip+disp32
				int32_t disp = 0;
				std::memcpy(&disp, code + i + 3, 4);
				const uint64_t rip = entry + i + 7;
				consider(rip + static_cast<int64_t>(disp), "lea_rip");
			}
		}
		// movabs r64, imm64 : 48 b8..bf
		if (code[i] == 0x48 && (code[i + 1] >= 0xb8 && code[i + 1] <= 0xbf) &&
		    i + 10 <= sizeof(code)) {
			uint64_t imm = 0;
			std::memcpy(&imm, code + i + 2, 8);
			consider(imm, "movabs");
		}
	}
	if (g_phase56_lea_cand_n == 0 &&
	    !g_phase56_base_elected.load(std::memory_order_acquire)) {
		Phase56SetCause("wrong_structure");
	}
}

static void Phase56ScanMultiBase(uint64_t cond, uint64_t mutex, uint64_t sync_id) {
	const uint64_t bases[] = {
	    mutex,
	    mutex >= 0x40 ? mutex - 0x40 : 0,
	    cond,
	    cond >= 0x80 ? cond - 0x80 : 0,
	    cond + 0x20,
	    mutex >= 0x100 ? mutex - 0x100 : 0,
	    cond + 0x100,
	};
	const char* tags[] = {"mutex", "mutex-0x40", "cond", "cond-0x80", "cond+0x20", "mutex-0x100",
	                      "cond+0x100"};
	for (size_t i = 0; i < sizeof(bases) / sizeof(bases[0]); ++i) {
		if (bases[i] == 0 || Phase56IsBannedBase(bases[i])) {
			continue;
		}
		Phase56DumpAndClassify(tags[i], bases[i], sync_id);
	}
	if (!g_phase56_base_elected.load(std::memory_order_acquire)) {
		Phase56SetCause("not_near_cond");
	}
}

void Phase56NoteGuestSync(uint64_t guest_cond_va, uint64_t guest_mutex_va, uint64_t guest_arg,
                          const char* role) {
	if (guest_cond_va < 0x10000ULL && guest_mutex_va < 0x10000ULL) {
		return;
	}
	const uint64_t sync_id =
	    g_phase56_sync_id.fetch_add(1, std::memory_order_acq_rel) + 1;
	const char* epoch =
	    g_phase37_post_unreg.load(std::memory_order_acquire) ? "post_unreg" : "pre_unreg";
	const int64_t delta =
	    static_cast<int64_t>(guest_cond_va) - static_cast<int64_t>(guest_mutex_va);
	if (sync_id <= 48 || (sync_id % 32) == 0) {
		LOGF("SubmitTrace: phase56 sync sync_id=%" PRIu64 " epoch=%s cond=0x%016" PRIx64
		     " mutex=0x%016" PRIx64 " delta=%" PRId64 " role=%s arg=%" PRIu64 " tid=%d\n",
		     sync_id, epoch, guest_cond_va, guest_mutex_va, delta, role != nullptr ? role : "?",
		     guest_arg, Common::Thread::GetThreadIdUnique());
		fprintf(stderr,
		        "SubmitTrace: phase56 sync sync_id=%" PRIu64 " epoch=%s role=%s\n", sync_id, epoch,
		        role != nullptr ? role : "?");
	}
	if (role != nullptr && (std::strcmp(role, "Mixed") == 0 || std::strcmp(role, "Compute") == 0)) {
		if (guest_cond_va >= 0x10000ULL) {
			g_phase56_job_cond.store(guest_cond_va, std::memory_order_release);
		}
		if (guest_mutex_va >= 0x10000ULL) {
			g_phase56_job_mutex.store(guest_mutex_va, std::memory_order_release);
		}
		if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
			Phase56ScanMultiBase(guest_cond_va, guest_mutex_va, sync_id);
			Phase56ScanRipRelPtrs(kPhase55MixedEntry);
			Phase57TryScanMixedBody();
			Phase56TryFakeCount("sync");
		}
	}
}

void Phase56PollWatch() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire) ||
	    !g_phase56_watch_armed.load(std::memory_order_acquire)) {
		return;
	}
	const uint64_t base = g_phase56_queue_base.load(std::memory_order_acquire);
	if (base < 0x10000ULL || Phase56IsBannedBase(base)) {
		return;
	}
	uint64_t q0  = 0;
	uint64_t q8  = 0;
	uint64_t cnt = 0;
	Phase41SafeRead(&q0, reinterpret_cast<const void*>(base), sizeof(q0));
	Phase41SafeRead(&q8, reinterpret_cast<const void*>(base + 8), sizeof(q8));
	const int coff = g_phase56_count_off.load(std::memory_order_acquire);
	if (coff >= 0) {
		Phase41SafeRead(&cnt, reinterpret_cast<const void*>(base + static_cast<uint64_t>(coff)),
		                sizeof(cnt));
	}
	if (!g_phase56_watch_seeded.exchange(true, std::memory_order_acq_rel)) {
		g_phase56_watch_q0.store(q0, std::memory_order_release);
		g_phase56_watch_q8.store(q8, std::memory_order_release);
		g_phase56_watch_count.store(cnt, std::memory_order_release);
		Phase56TryFakeCount("watch_seed");
		return;
	}
	auto note = [&](const char* tag, uint64_t prev, uint64_t cur, int off, const char* wr) {
		if (prev == cur) {
			return;
		}
		const uint32_t wn = g_phase56_enqueue_write_n.fetch_add(1, std::memory_order_relaxed) + 1;
		// Periodic sampler is not a guest writer — only count activity here.
		// Main/BootCards attribution comes from Phase56NoteMainSignal.
		if (std::strcmp(wr, "Main") == 0 || std::strcmp(wr, "BootCards") == 0) {
			(void)g_phase56_enqueue_from_main.fetch_add(1, std::memory_order_relaxed);
		} else if (std::strcmp(wr, "other") == 0 || std::strcmp(wr, "Mixed") == 0) {
			(void)g_phase56_enqueue_from_other.fetch_add(1, std::memory_order_relaxed);
		}
		const uint64_t ra = reinterpret_cast<uint64_t>(__builtin_return_address(0));
		if (wn <= 64 || (wn % 16) == 0) {
			LOGF("SubmitTrace: phase56 enqueue_write sync_id=%" PRIu64 " tag=%s off=0x%x "
			     "old=0x%016" PRIx64 " new=0x%016" PRIx64 " writer_tid=%d writer_role=%s "
			     "ra=0x%016" PRIx64 " n=%u\n",
			     Phase56CurrentSyncId(), tag, off, prev, cur, Common::Thread::GetThreadIdUnique(),
			     wr, ra, wn);
			fprintf(stderr, "SubmitTrace: phase56 enqueue_write tag=%s role=%s n=%u\n", tag, wr,
			        wn);
		}
	};
	const char* wr = "sampler";
	if (LibKernel::PthreadCurrentIsMainRelated() ||
	    LibKernel::PthreadCurrentIsSubmissionRelated()) {
		wr = Phase56WriterRole();
	}
	const uint64_t prev0 = g_phase56_watch_q0.exchange(q0, std::memory_order_acq_rel);
	const uint64_t prev8 = g_phase56_watch_q8.exchange(q8, std::memory_order_acq_rel);
	const uint64_t prevc = g_phase56_watch_count.exchange(cnt, std::memory_order_acq_rel);
	note("q0", prev0, q0, 0, wr);
	note("q8", prev8, q8, 8, wr);
	if (coff >= 0) {
		note("count", prevc, cnt, coff, wr);
	}
	Phase56TryFakeCount("watch");
}

void Phase56NoteMainSignal(uint64_t guest_cond_va, const char* role) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	if (role == nullptr ||
	    (std::strcmp(role, "Main") != 0 && std::strcmp(role, "BootCards") != 0)) {
		return;
	}
	const uint64_t job = g_phase56_job_cond.load(std::memory_order_acquire);
	if (job != 0 && guest_cond_va == job) {
		(void)g_phase56_main_signal_job.fetch_add(1, std::memory_order_relaxed);
	} else {
		(void)g_phase56_main_signal_alt.fetch_add(1, std::memory_order_relaxed);
	}
	// Attribute queue diffs seen under Main/BootCards signal path.
	if (!g_phase56_watch_armed.load(std::memory_order_acquire)) {
		return;
	}
	const uint64_t base = g_phase56_queue_base.load(std::memory_order_acquire);
	if (base < 0x10000ULL || Phase56IsBannedBase(base)) {
		return;
	}
	uint64_t q0  = 0;
	uint64_t q8  = 0;
	uint64_t cnt = 0;
	Phase41SafeRead(&q0, reinterpret_cast<const void*>(base), sizeof(q0));
	Phase41SafeRead(&q8, reinterpret_cast<const void*>(base + 8), sizeof(q8));
	const int coff = g_phase56_count_off.load(std::memory_order_acquire);
	if (coff >= 0) {
		Phase41SafeRead(&cnt, reinterpret_cast<const void*>(base + static_cast<uint64_t>(coff)),
		                sizeof(cnt));
	}
	if (!g_phase56_watch_seeded.load(std::memory_order_acquire)) {
		g_phase56_watch_q0.store(q0, std::memory_order_release);
		g_phase56_watch_q8.store(q8, std::memory_order_release);
		g_phase56_watch_count.store(cnt, std::memory_order_release);
		g_phase56_watch_seeded.store(true, std::memory_order_release);
		return;
	}
	auto note_main = [&](const char* tag, uint64_t prev, uint64_t cur, int off) {
		if (prev == cur) {
			return;
		}
		const uint32_t wn = g_phase56_enqueue_write_n.fetch_add(1, std::memory_order_relaxed) + 1;
		(void)g_phase56_enqueue_from_main.fetch_add(1, std::memory_order_relaxed);
		if (wn <= 64 || (wn % 16) == 0) {
			LOGF("SubmitTrace: phase56 enqueue_write sync_id=%" PRIu64 " tag=%s off=0x%x "
			     "old=0x%016" PRIx64 " new=0x%016" PRIx64 " writer_tid=%d writer_role=%s "
			     "ra=0 signal_cond=0x%016" PRIx64 " n=%u\n",
			     Phase56CurrentSyncId(), tag, off, prev, cur,
			     Common::Thread::GetThreadIdUnique(), role, guest_cond_va, wn);
			fprintf(stderr, "SubmitTrace: phase56 enqueue_write tag=%s role=%s n=%u\n", tag, role,
			        wn);
		}
	};
	const uint64_t prev0 = g_phase56_watch_q0.exchange(q0, std::memory_order_acq_rel);
	const uint64_t prev8 = g_phase56_watch_q8.exchange(q8, std::memory_order_acq_rel);
	const uint64_t prevc = g_phase56_watch_count.exchange(cnt, std::memory_order_acq_rel);
	note_main("q0", prev0, q0, 0);
	note_main("q8", prev8, q8, 8);
	if (coff >= 0) {
		note_main("count", prevc, cnt, coff);
	}
}

void Phase56TryFakeCount(const char* why) {
	if (!Phase56FakeCountEnabled() || !g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	if (!g_phase56_base_elected.load(std::memory_order_acquire)) {
		return;
	}
	if (g_phase56_enqueue_write_n.load(std::memory_order_acquire) == 0) {
		return; // never inject without real activity
	}
	const uint64_t base = g_phase56_queue_base.load(std::memory_order_acquire);
	const int      coff = g_phase56_count_off.load(std::memory_order_acquire);
	if (base < 0x10000ULL || Phase56IsBannedBase(base) || coff < 0) {
		return;
	}
	uint64_t cnt = 0;
	Phase41SafeRead(&cnt, reinterpret_cast<const void*>(base + static_cast<uint64_t>(coff)),
	                sizeof(cnt));
	if (cnt != 0) {
		return;
	}
	if (g_phase56_fake_count_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint64_t one = 1;
	Phase41SafeWrite(reinterpret_cast<volatile void*>(base + static_cast<uint64_t>(coff)), &one,
	                 sizeof(one));
	LOGF("SubmitTrace: phase56 FAKE_COUNT base=0x%016" PRIx64 " count_off=0x%x why=%s tid=%d\n",
	     base, coff, why != nullptr ? why : "?", Common::Thread::GetThreadIdUnique());
	fprintf(stderr, "SubmitTrace: phase56 FAKE_COUNT base=0x%016" PRIx64 "\n", base);
	(void)LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
}

void Phase56EmitHeatmap(const char* why) {
	if (g_phase56_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t writes = g_phase56_enqueue_write_n.load(std::memory_order_relaxed);
	const uint32_t from_m = g_phase56_enqueue_from_main.load(std::memory_order_relaxed);
	const uint32_t from_o = g_phase56_enqueue_from_other.load(std::memory_order_relaxed);
	const uint64_t base   = g_phase56_queue_base.load(std::memory_order_relaxed);
	const int      elected = g_phase56_base_elected.load(std::memory_order_relaxed) ? 1 : 0;
	const int      guest_real =
	    g_phase55_submit_guest_real_n.load(std::memory_order_relaxed) > 0 ? 1 : 0;
	if (guest_real && from_m > 0 && elected) {
		Phase56SetCause("pipeline_ok");
	} else if (elected && writes == 0) {
		Phase56SetCause("producer_dead");
	} else if (elected && from_m > 0 && !guest_real) {
		Phase56SetCause("wrong_cond_or_predicate");
	} else if (elected && from_o > 0 && from_m == 0) {
		Phase56SetCause("wrong_queue_generic");
	} else if (!elected) {
		Phase56SetCause("wrong_structure");
	}
	const uint32_t sig_job = g_phase56_main_signal_job.load(std::memory_order_relaxed);
	const uint32_t sig_alt = g_phase56_main_signal_alt.load(std::memory_order_relaxed);
	if (elected && writes > 0 && from_m > 0 && sig_job == 0 && sig_alt > 0 && !guest_real) {
		Phase56SetCause("wrong_cond_handoff");
	} else if (elected && writes > 0 && from_m > 0 && sig_job == 0 && sig_alt == 0 &&
	           !guest_real) {
		Phase56SetCause("non_cond_wake");
	}
	LOGF("SubmitTrace: phase56 heatmap why=%s queue_base=0x%016" PRIx64 " type=%s elected=%d "
	     "list_cands=%u dumps=%u enqueue_write=%u from_main=%u from_other=%u "
	     "main_sig_job=%u main_sig_alt=%u submit_guest_real=%d sync_id=%" PRIu64
	     " cause=%s guest_draw=%d\n",
	     why != nullptr ? why : "?", base, g_phase56_base_type, elected,
	     g_phase56_list_cand_n.load(std::memory_order_relaxed),
	     g_phase56_dump_runs.load(std::memory_order_relaxed), writes, from_m, from_o, sig_job,
	     sig_alt, guest_real, Phase56CurrentSyncId(), g_phase56_cause,
	     g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0);
	fprintf(stderr, "SubmitTrace: phase56 heatmap cause=%s base=0x%016" PRIx64 " writes=%u\n",
	        g_phase56_cause, base, writes);
}

// ---------------------------------------------------------------------------
// Phase 57 — global queue (A) + Main producer (C)
// ---------------------------------------------------------------------------

static void Phase57SetCause(const char* c) {
	if (c != nullptr) {
		std::snprintf(g_phase57_cause, sizeof(g_phase57_cause), "%s", c);
	}
}

static bool Phase57IsNearSync(uint64_t va) {
	const uint64_t cond = g_phase56_job_cond.load(std::memory_order_acquire);
	const uint64_t mtx  = g_phase56_job_mutex.load(std::memory_order_acquire);
	auto within = [](uint64_t a, uint64_t b) {
		return b != 0 && a + 0x200 >= b && a <= b + 0x200;
	};
	return within(va, cond) || within(va, mtx);
}

static Phase57HeatSlot* Phase57FindOrAddSlot(uint64_t va) {
	if (!Phase52IsGuestPtr(va) || Phase56IsBannedBase(va)) {
		return nullptr;
	}
	for (size_t i = 0; i < kPhase57HeatSlots; ++i) {
		if (g_phase57_heat[i].va.load(std::memory_order_acquire) == va) {
			return &g_phase57_heat[i];
		}
	}
	const uint32_t idx = g_phase57_heat_used.fetch_add(1, std::memory_order_relaxed);
	if (idx >= kPhase57HeatSlots) {
		(void)g_phase57_heat_used.fetch_sub(1, std::memory_order_relaxed);
		return nullptr;
	}
	uint64_t expected = 0;
	if (!g_phase57_heat[idx].va.compare_exchange_strong(expected, va, std::memory_order_acq_rel)) {
		// Slot raced — find again.
		for (size_t i = 0; i < kPhase57HeatSlots; ++i) {
			if (g_phase57_heat[i].va.load(std::memory_order_acquire) == va) {
				return &g_phase57_heat[i];
			}
		}
		return nullptr;
	}
	(void)g_phase57_cand_n.fetch_add(1, std::memory_order_relaxed);
	return &g_phase57_heat[idx];
}

static void Phase57ElectIfList(uint64_t va, const Phase56BaseScore& s, const char* tag) {
	if (std::strcmp(s.type, "LIST_CANDIDATE") != 0 || Phase56IsBannedBase(va)) {
		return;
	}
	(void)g_phase57_list_n.fetch_add(1, std::memory_order_relaxed);
	int score = s.score;
	auto* slot = Phase57FindOrAddSlot(va);
	if (slot != nullptr) {
		score += static_cast<int>(slot->main_n.load(std::memory_order_relaxed)) * 3 +
		         static_cast<int>(slot->main_touch.load(std::memory_order_relaxed)) * 5 +
		         static_cast<int>(slot->write_n.load(std::memory_order_relaxed));
		std::snprintf(slot->type, sizeof(slot->type), "%s", s.type);
		slot->score.store(score, std::memory_order_release);
	}
	const int best = g_phase57_best_score.load(std::memory_order_acquire);
	if (score <= best && g_phase57_elected.load(std::memory_order_acquire)) {
		return;
	}
	g_phase57_best_score.store(score, std::memory_order_release);
	g_phase57_queue_base.store(va, std::memory_order_release);
	g_phase57_elected.store(true, std::memory_order_release);
	std::snprintf(g_phase57_base_type, sizeof(g_phase57_base_type), "%s", s.type);
	// Feed Phase56 watch on global LIST (never banned).
	Phase56ElectQueueBase(s, Phase56CurrentSyncId());
	LOGF("SubmitTrace: phase57 elect queue_base=0x%016" PRIx64 " type=%s score=%d tag=%s "
	     "guest_n=%u write_boost=%d\n",
	     va, s.type, score, tag != nullptr ? tag : "?", s.guest_n, score - s.score);
	fprintf(stderr, "SubmitTrace: phase57 elect queue_base=0x%016" PRIx64 " type=%s score=%d\n",
	        va, s.type, score);
}

static void Phase57NoteCand(uint64_t va, const char* tag, const char* role_hint) {
	if (!Phase52IsGuestPtr(va) || Phase56IsBannedBase(va)) {
		return;
	}
	// Data anchors must be 8-byte aligned; reject decoy lea/imm noise.
	if ((va & 7ULL) != 0) {
		return;
	}
	if (Phase57IsNearSync(va)) {
		static std::atomic<uint32_t> skip_logs {0};
		if (skip_logs.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("SubmitTrace: phase57 near_sync_skip va=0x%016" PRIx64 " tag=%s\n", va,
			     tag != nullptr ? tag : "?");
		}
		return; // P56 already covered cond neighborhood — do not elect
	}
	auto* slot = Phase57FindOrAddSlot(va);
	if (slot == nullptr) {
		return;
	}
	if (role_hint != nullptr) {
		if (std::strcmp(role_hint, "Main") == 0) {
			(void)slot->main_n.fetch_add(1, std::memory_order_relaxed);
		} else if (std::strcmp(role_hint, "Mixed") == 0) {
			(void)slot->mixed_n.fetch_add(1, std::memory_order_relaxed);
		} else {
			(void)slot->other_n.fetch_add(1, std::memory_order_relaxed);
		}
	}
	(void)slot->read_n.fetch_add(1, std::memory_order_relaxed);
	const Phase56BaseScore s = Phase56ClassifyBase(va);
	std::snprintf(slot->type, sizeof(slot->type), "%s", s.type);
	const uint32_t runs = g_phase57_scan_runs.fetch_add(1, std::memory_order_relaxed) + 1;
	if (runs <= 48 || (runs % 32) == 0 || std::strcmp(s.type, "LIST_CANDIDATE") == 0) {
		LOGF("SubmitTrace: phase57 cand va=0x%016" PRIx64 " tag=%s type=%s guest_n=%u user_n=%u "
		     "scalar_n=%u score=%d role=%s\n",
		     va, tag != nullptr ? tag : "?", s.type, s.guest_n, s.user_n, s.scalar_n, s.score,
		     role_hint != nullptr ? role_hint : "?");
		fprintf(stderr, "SubmitTrace: phase57 cand type=%s va=0x%016" PRIx64 " tag=%s\n", s.type,
		        va, tag != nullptr ? tag : "?");
	}
	Phase57ElectIfList(va, s, tag);
	// Also dump ±0x200 anchors as sibling candidates (ctx fields).
	if (std::strcmp(tag, "stable_ctx") == 0 || std::strcmp(tag, "CTX_FIELD") == 0) {
		for (int64_t delta: {-0x100LL, -0x80LL, 0x80LL, 0x100LL, 0x180LL, 0x200LL}) {
			const uint64_t sib = static_cast<uint64_t>(static_cast<int64_t>(va) + delta);
			if (!Phase52IsGuestPtr(sib) || Phase56IsBannedBase(sib) || Phase57IsNearSync(sib)) {
				continue;
			}
			const Phase56BaseScore ss = Phase56ClassifyBase(sib);
			if (std::strcmp(ss.type, "LIST_CANDIDATE") == 0) {
				Phase57NoteCand(sib, "ctx_sib", role_hint);
			}
		}
	}
}

static void Phase57ScanCodeWindow(uint64_t entry, uint32_t span, const char* tag, int depth) {
	if (entry < 0x1000ULL || span < 16 || depth < 0) {
		return;
	}
	uint8_t code[512] {};
	const uint32_t nread = span > sizeof(code) ? static_cast<uint32_t>(sizeof(code)) : span;
	Phase41SafeRead(code, reinterpret_cast<const void*>(entry), nread);
	uint64_t callees[4] {};
	int      ncall = 0;
	for (size_t i = 0; i + 7 < nread; ++i) {
		if (code[i] == 0x48 && code[i + 1] == 0x8d && i + 7 < nread) {
			const uint8_t modrm = code[i + 2];
			if ((modrm & 0xc7u) == 0x05u) {
				int32_t disp = 0;
				std::memcpy(&disp, code + i + 3, 4);
				const uint64_t rip = entry + i + 7;
				Phase57NoteCand(rip + static_cast<int64_t>(disp), tag, "Mixed");
			}
		}
		if (code[i] == 0x48 && (code[i + 1] >= 0xb8 && code[i + 1] <= 0xbf) && i + 10 <= nread) {
			uint64_t imm = 0;
			std::memcpy(&imm, code + i + 2, 8);
			Phase57NoteCand(imm, "movabs", "Mixed");
		}
		// mov r64, [r/m64+disp8/32] with small disp (CTX_FIELD hint)
		if (code[i] == 0x48 && code[i + 1] == 0x8b && i + 3 < nread) {
			const uint8_t modrm = code[i + 2];
			const uint8_t mod   = static_cast<uint8_t>((modrm >> 6) & 3u);
			int32_t       disp  = 0;
			size_t        skip  = 3;
			if (mod == 1 && i + 4 <= nread) {
				disp = static_cast<int8_t>(code[i + 3]);
				skip = 4;
			} else if (mod == 2 && i + 7 <= nread) {
				std::memcpy(&disp, code + i + 3, 4);
				skip = 7;
			}
			if ((mod == 1 || mod == 2) && disp >= 0 && disp <= 0x40) {
				static std::atomic<uint32_t> field_logs {0};
				if (field_logs.fetch_add(1, std::memory_order_relaxed) < 24) {
					LOGF("SubmitTrace: phase57 CTX_FIELD entry=0x%016" PRIx64 " ip=+0x%zx "
					     "disp=0x%x tag=%s\n",
					     entry, i, disp, tag != nullptr ? tag : "?");
				}
			}
			(void)skip;
		}
		// E8 call rel32
		if (code[i] == 0xe8 && i + 5 <= nread && ncall < 4) {
			int32_t rel = 0;
			std::memcpy(&rel, code + i + 1, 4);
			callees[ncall++] = entry + i + 5 + static_cast<int64_t>(rel);
		}
	}
	if (depth > 0) {
		for (int c = 0; c < ncall && c < 2; ++c) {
			// Only follow nearby callees in the same module text neighborhood.
			if (Phase52IsGuestPtr(callees[c]) && (callees[c] & 0xfULL) == 0 &&
			    callees[c] >= 0x0000000901000000ULL && callees[c] < 0x0000000902000000ULL) {
				Phase57ScanCodeWindow(callees[c], 192, "callee", depth - 1);
			}
		}
	}
}

void Phase57TryScanMixedBody() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	if (g_phase57_body_scanned.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	LOGF("SubmitTrace: phase57 mixed_body_scan start entry=0x%016" PRIx64 "\n",
	     kPhase55MixedEntry);
	fprintf(stderr, "SubmitTrace: phase57 mixed_body_scan start\n");
	// SafeRead-only scan (no DumpGuestCodeAround / DumpNextGuestCalls — those can AV).
	Phase57ScanCodeWindow(kPhase55MixedEntry, 384, "mixed_body", 1);
	LOGF("SubmitTrace: phase57 mixed_body_scan done entry=0x%016" PRIx64 " cands=%u lists=%u\n",
	     kPhase55MixedEntry, g_phase57_cand_n.load(std::memory_order_relaxed),
	     g_phase57_list_n.load(std::memory_order_relaxed));
	fprintf(stderr, "SubmitTrace: phase57 mixed_body_scan done cands=%u\n",
	        g_phase57_cand_n.load(std::memory_order_relaxed));
}

static void Phase57StabilizeReg(std::atomic<uint64_t>& hist, std::atomic<uint32_t>& hits,
                                uint64_t val, const char* which) {
	if (!Phase52IsGuestPtr(val) || Phase56IsBannedBase(val) || Phase57IsNearSync(val)) {
		return;
	}
	const uint64_t prev = hist.load(std::memory_order_acquire);
	if (prev == val) {
		const uint32_t h = hits.fetch_add(1, std::memory_order_relaxed) + 1;
		if (h == kPhase57StabHitsNeed) {
			g_phase57_stable_ctx.store(val, std::memory_order_release);
			LOGF("SubmitTrace: phase57 stable_ctx which=%s va=0x%016" PRIx64 " hits=%u\n", which,
			     val, h);
			fprintf(stderr, "SubmitTrace: phase57 stable_ctx which=%s va=0x%016" PRIx64 "\n", which,
			        val);
			Phase57NoteCand(val, "stable_ctx", "Mixed");
			// Dump ±0x200 classify
			for (int64_t d = -0x200; d <= 0x200; d += 0x40) {
				const uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(val) + d);
				if (Phase52IsGuestPtr(a) && !Phase56IsBannedBase(a) && !Phase57IsNearSync(a)) {
					Phase57NoteCand(a, "ctx_window", "Mixed");
				}
			}
		}
	} else {
		hist.store(val, std::memory_order_release);
		hits.store(1, std::memory_order_release);
	}
}

void Phase57NoteMixedRegs(uint64_t rdi, uint64_t rsi, uint64_t rdx) {
	Phase57StabilizeReg(g_phase57_rdi_hist, g_phase57_rdi_hits, rdi, "rdi");
	Phase57StabilizeReg(g_phase57_rsi_hist, g_phase57_rsi_hits, rsi, "rsi");
	Phase57StabilizeReg(g_phase57_rdx_hist, g_phase57_rdx_hits, rdx, "rdx");
	if (Phase52IsGuestPtr(rdi) && !Phase57IsNearSync(rdi)) {
		Phase57NoteCand(rdi, "mixed_rdi", "Mixed");
	}
	if (Phase52IsGuestPtr(rsi) && !Phase57IsNearSync(rsi)) {
		Phase57NoteCand(rsi, "mixed_rsi", "Mixed");
	}
	if (Phase52IsGuestPtr(rdx) && !Phase57IsNearSync(rdx)) {
		Phase57NoteCand(rdx, "mixed_rdx", "Mixed");
	}
}

void Phase57NoteMainAgcTouch(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire) ||
	    !LibKernel::PthreadCurrentIsMainRelated()) {
		return;
	}
	const uint32_t n = g_phase57_main_agc_n.fetch_add(1, std::memory_order_relaxed) + 1;
	const uint64_t args[4] = {a0, a1, a2, a3};
	int            touch   = 0;
	for (uint64_t a: args) {
		if (!Phase52IsGuestPtr(a) || Phase56IsBannedBase(a)) {
			continue;
		}
		Phase57NoteCand(a, "main_agc_arg", "Main");
		auto* slot = Phase57FindOrAddSlot(a);
		if (slot != nullptr) {
			(void)slot->main_touch.fetch_add(1, std::memory_order_relaxed);
			++touch;
		}
	}
	Phase58NoteMainAgcCross(nid, a0, a1, a2, a3);
	if (touch > 0) {
		(void)g_phase57_main_touch_cand.fetch_add(1, std::memory_order_relaxed);
	}
	if (n <= 64 || (n % 16) == 0) {
		LOGF("SubmitTrace: phase57 main_agc nid=%s n=%u touch=%d a0=0x%016" PRIx64
		     " a1=0x%016" PRIx64 " a2=0x%016" PRIx64 " a3=0x%016" PRIx64 "\n",
		     nid != nullptr ? nid : "?", n, touch, a0, a1, a2, a3);
		fprintf(stderr, "SubmitTrace: phase57 main_agc nid=%s touch=%d n=%u\n",
		        nid != nullptr ? nid : "?", touch, n);
	}
}

void Phase57NoteMainObjectWrite(uint64_t guest_va, const char* why) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire) ||
	    !LibKernel::PthreadCurrentIsMainRelated()) {
		return;
	}
	if (!Phase52IsGuestPtr(guest_va) || Phase56IsBannedBase(guest_va) ||
	    Phase57IsNearSync(guest_va)) {
		return;
	}
	const uint32_t n = g_phase57_main_write_n.fetch_add(1, std::memory_order_relaxed) + 1;
	Phase57NoteCand(guest_va, why != nullptr ? why : "main_write", "Main");
	auto* slot = Phase57FindOrAddSlot(guest_va);
	if (slot != nullptr) {
		(void)slot->main_touch.fetch_add(1, std::memory_order_relaxed);
		(void)slot->write_n.fetch_add(1, std::memory_order_relaxed);
		(void)g_phase57_main_touch_cand.fetch_add(1, std::memory_order_relaxed);
	}
	if (n <= 48 || (n % 16) == 0) {
		LOGF("SubmitTrace: phase57 main_write va=0x%016" PRIx64 " why=%s n=%u\n", guest_va,
		     why != nullptr ? why : "?", n);
		fprintf(stderr, "SubmitTrace: phase57 main_write va=0x%016" PRIx64 " n=%u\n", guest_va, n);
	}
}

void Phase57PollHeatmap() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	Phase57TryScanMixedBody();
	const uint32_t used = g_phase57_heat_used.load(std::memory_order_acquire);
	const uint32_t lim  = used < kPhase57HeatSlots ? used : static_cast<uint32_t>(kPhase57HeatSlots);
	for (uint32_t i = 0; i < lim; ++i) {
		auto*          slot = &g_phase57_heat[i];
		const uint64_t va   = slot->va.load(std::memory_order_acquire);
		if (va < 0x10000ULL || Phase56IsBannedBase(va)) {
			continue;
		}
		uint8_t cur[64] {};
		Phase41SafeRead(cur, reinterpret_cast<const void*>(va), sizeof(cur));
		(void)slot->read_n.fetch_add(1, std::memory_order_relaxed);
		if (!slot->have_prev) {
			std::memcpy(slot->prev, cur, sizeof(cur));
			slot->have_prev = true;
			continue;
		}
		bool changed = false;
		for (size_t off = 0; off < sizeof(cur); ++off) {
			if (slot->prev[off] != cur[off]) {
				changed = true;
				break;
			}
		}
		if (changed) {
			const uint32_t wn = slot->write_n.fetch_add(1, std::memory_order_relaxed) + 1;
			const char*    wr = "sampler";
			if (LibKernel::PthreadCurrentIsMainRelated()) {
				wr = "Main";
				(void)slot->main_n.fetch_add(1, std::memory_order_relaxed);
			} else if (LibKernel::PthreadCurrentIsSubmissionRelated()) {
				wr = "Mixed";
				(void)slot->mixed_n.fetch_add(1, std::memory_order_relaxed);
			} else {
				(void)slot->other_n.fetch_add(1, std::memory_order_relaxed);
			}
			if (wn <= 32 || (wn % 16) == 0) {
				LOGF("SubmitTrace: phase57 heat_write va=0x%016" PRIx64 " role=%s n=%u type=%s\n",
				     va, wr, wn, slot->type[0] != '\0' ? slot->type : "?");
				fprintf(stderr, "SubmitTrace: phase57 heat_write va=0x%016" PRIx64 " role=%s\n", va,
				        wr);
			}
			std::memcpy(slot->prev, cur, sizeof(cur));
			const Phase56BaseScore s = Phase56ClassifyBase(va);
			Phase57ElectIfList(va, s, "heat_write");
		}
	}
}

void Phase57EmitHeatmap(const char* why) {
	if (g_phase57_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const int      elected = g_phase57_elected.load(std::memory_order_relaxed) ? 1 : 0;
	const uint64_t base    = g_phase57_queue_base.load(std::memory_order_relaxed);
	const uint32_t list_n  = g_phase57_list_n.load(std::memory_order_relaxed);
	const uint32_t from_m =
	    g_phase56_enqueue_from_main.load(std::memory_order_relaxed) +
	    g_phase57_main_touch_cand.load(std::memory_order_relaxed);
	uint32_t heat_writes = 0;
	uint32_t heat_main   = 0;
	const uint32_t used  = g_phase57_heat_used.load(std::memory_order_relaxed);
	const uint32_t lim = used < kPhase57HeatSlots ? used : static_cast<uint32_t>(kPhase57HeatSlots);
	for (uint32_t i = 0; i < lim; ++i) {
		heat_writes += g_phase57_heat[i].write_n.load(std::memory_order_relaxed);
		heat_main += g_phase57_heat[i].main_n.load(std::memory_order_relaxed) +
		             g_phase57_heat[i].main_touch.load(std::memory_order_relaxed);
	}
	if (elected && (from_m > 0 || heat_main > 0) &&
	    g_phase55_submit_guest_real_n.load(std::memory_order_relaxed) > 0) {
		Phase57SetCause("pipeline_ok");
	} else if (elected && (from_m > 0 || heat_main > 0)) {
		Phase57SetCause("anchor_ok"); // LIST + Main activity → P58 predicate eligible
	} else if (elected && heat_writes == 0 &&
	           g_phase56_enqueue_write_n.load(std::memory_order_relaxed) == 0) {
		Phase57SetCause("producer_dead");
	} else if (!elected) {
		Phase57SetCause("still_wrong_structure");
	}
	LOGF("SubmitTrace: phase57 heatmap why=%s queue_base=0x%016" PRIx64 " type=%s elected=%d "
	     "cands=%u lists=%u heat_writes=%u heat_main=%u main_agc=%u main_write=%u "
	     "main_touch_cand=%u stable_ctx=0x%016" PRIx64 " cause=%s guest_draw=%d\n",
	     why != nullptr ? why : "?", base, g_phase57_base_type, elected,
	     g_phase57_cand_n.load(std::memory_order_relaxed), list_n, heat_writes, heat_main,
	     g_phase57_main_agc_n.load(std::memory_order_relaxed),
	     g_phase57_main_write_n.load(std::memory_order_relaxed),
	     g_phase57_main_touch_cand.load(std::memory_order_relaxed),
	     g_phase57_stable_ctx.load(std::memory_order_relaxed), g_phase57_cause,
	     g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0);
	fprintf(stderr, "SubmitTrace: phase57 heatmap cause=%s base=0x%016" PRIx64 " lists=%u\n",
	        g_phase57_cause, base, list_n);
}

uint64_t Phase57QueueBase() {
	return g_phase57_queue_base.load(std::memory_order_acquire);
}

bool Phase57Elected() {
	return g_phase57_elected.load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// Phase 58 — NdJob ancre + subblock classify + discriminant heatmap
// ---------------------------------------------------------------------------

static void Phase58SetCause(const char* c) {
	if (c != nullptr) {
		std::snprintf(g_phase58_cause, sizeof(g_phase58_cause), "%s", c);
	}
}

static bool Phase58InNdJobCtxBand(uint64_t va) {
	return va >= kPhase58CtxLo && va < kPhase58CtxHi;
}

static bool Phase58IsDeepScanInterior(uint64_t obj, uint64_t va) {
	if (obj < 0x10000ULL || va < obj + 0x100ULL) {
		return false;
	}
	return va < obj + 0x21000ULL;
}

static Phase58HeatSlot* Phase58FindOrAddSlot(uint64_t va) {
	if (va < 0x10000ULL || Phase56IsBannedBase(va)) {
		return nullptr;
	}
	if ((va & 7ULL) != 0 && !Phase52IsUserPtr(va)) {
		return nullptr;
	}
	for (size_t i = 0; i < kPhase58HeatSlots; ++i) {
		if (g_phase58_heat[i].va.load(std::memory_order_acquire) == va) {
			return &g_phase58_heat[i];
		}
	}
	const uint32_t idx = g_phase58_heat_used.fetch_add(1, std::memory_order_relaxed);
	if (idx >= kPhase58HeatSlots) {
		(void)g_phase58_heat_used.fetch_sub(1, std::memory_order_relaxed);
		return nullptr;
	}
	uint64_t expected = 0;
	if (!g_phase58_heat[idx].va.compare_exchange_strong(expected, va, std::memory_order_acq_rel)) {
		for (size_t i = 0; i < kPhase58HeatSlots; ++i) {
			if (g_phase58_heat[i].va.load(std::memory_order_acquire) == va) {
				return &g_phase58_heat[i];
			}
		}
		return nullptr;
	}
	(void)g_phase58_cand_n.fetch_add(1, std::memory_order_relaxed);
	return &g_phase58_heat[idx];
}

struct Phase58WinScore {
	uint64_t    base     = 0;
	uint32_t    guest_n  = 0;
	uint32_t    user_n   = 0;
	uint32_t    scalar_n = 0;
	const char* type     = "STATE_BLOCK";
	int         score    = 0;
};

static Phase58WinScore Phase58ClassifyWindow(uint64_t base, size_t nbytes) {
	Phase58WinScore s {};
	s.base = base;
	if (base < 0x10000ULL || Phase56IsBannedBase(base) || nbytes < 8 || nbytes > 256) {
		s.type = "BANNED";
		return s;
	}
	uint8_t buf[256] {};
	Phase41SafeRead(buf, reinterpret_cast<const void*>(base), nbytes);
	for (size_t off = 0; off + 8 <= nbytes; off += 8) {
		uint64_t v = 0;
		std::memcpy(&v, buf + off, 8);
		if (Phase52IsGuestPtr(v)) {
			++s.guest_n;
		} else if (Phase52IsUserPtr(v)) {
			++s.user_n;
		} else {
			++s.scalar_n;
		}
	}
	if (s.guest_n >= 2 && s.scalar_n >= 1) {
		s.type  = "LIST_CANDIDATE";
		s.score = static_cast<int>(s.guest_n) * 10 + static_cast<int>(s.scalar_n);
	} else if (s.user_n > s.guest_n + s.scalar_n) {
		s.type  = "BUFFER_ANCHOR";
		s.score = static_cast<int>(s.user_n);
		(void)g_phase58_buffer_n.fetch_add(1, std::memory_order_relaxed);
	} else {
		s.type  = "STATE_BLOCK";
		s.score = 0;
	}
	return s;
}

static bool Phase58TryHops(uint64_t list_base) {
	uint64_t cur = list_base;
	for (int hop = 0; hop < 2; ++hop) {
		const uint64_t next = Phase52ReadNextHop(cur);
		if (next == 0 || next == ~uint64_t {0}) {
			return true; // empty/end OK
		}
		if (Phase52IsUserPtr(next)) {
			(void)g_phase58_hop_reject_n.fetch_add(1, std::memory_order_relaxed);
			LOGF("SubmitTrace: phase58 reject_hop_user base=0x%016" PRIx64 " hop=%d next=0x%016" PRIx64
			     "\n",
			     list_base, hop + 1, next);
			fprintf(stderr, "SubmitTrace: phase58 reject_hop_user hop=%d\n", hop + 1);
			return false;
		}
		if (!Phase52IsGuestPtr(next) || Phase56IsBannedBase(next)) {
			(void)g_phase58_hop_reject_n.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		cur = next;
	}
	return true;
}

static void Phase58ElectIfList(uint64_t va, const Phase58WinScore& s, const char* tag) {
	if (std::strcmp(s.type, "LIST_CANDIDATE") != 0 || Phase56IsBannedBase(va)) {
		return;
	}
	if (!Phase58TryHops(va)) {
		return;
	}
	(void)g_phase58_list_n.fetch_add(1, std::memory_order_relaxed);
	auto* slot = Phase58FindOrAddSlot(va);
	int   score = s.score;
	if (slot != nullptr) {
		std::snprintf(slot->type, sizeof(slot->type), "%s", s.type);
		score += static_cast<int>(slot->main_n.load(std::memory_order_relaxed)) * 3 +
		         static_cast<int>(slot->main_touch.load(std::memory_order_relaxed)) * 5 +
		         static_cast<int>(slot->write_n.load(std::memory_order_relaxed));
	}
	const int best = g_phase58_best_score.load(std::memory_order_acquire);
	if (score <= best && g_phase58_elected.load(std::memory_order_acquire)) {
		return;
	}
	g_phase58_best_score.store(score, std::memory_order_release);
	g_phase58_queue_base.store(va, std::memory_order_release);
	g_phase58_elected.store(true, std::memory_order_release);
	std::snprintf(g_phase58_base_type, sizeof(g_phase58_base_type), "%s", s.type);
	Phase56BaseScore p56 {};
	p56.base     = va;
	p56.guest_n  = s.guest_n;
	p56.user_n   = s.user_n;
	p56.scalar_n = s.scalar_n;
	p56.type     = "LIST_CANDIDATE";
	p56.score    = score;
	Phase56ElectQueueBase(p56, Phase56CurrentSyncId());
	LOGF("SubmitTrace: phase58 elect queue_base=0x%016" PRIx64 " type=%s score=%d tag=%s guest_n=%u\n",
	     va, s.type, score, tag != nullptr ? tag : "?", s.guest_n);
	fprintf(stderr, "SubmitTrace: phase58 elect queue_base=0x%016" PRIx64 " tag=%s\n", va,
	        tag != nullptr ? tag : "?");
}

static void Phase58ClassifySubblocks(uint64_t U, const char* tag) {
	if (U < 0x40ULL) {
		return;
	}
	struct Win {
		const char* name;
		uint64_t    base;
	};
	const Win wins[] = {
	    {"lo", U - 0x40ULL},
	    {"mid", U},
	    {"hi", U + 0x40ULL},
	};
	bool any_list = false;
	for (const Win& w: wins) {
		const Phase58WinScore s = Phase58ClassifyWindow(w.base, 64);
		if (std::strcmp(s.type, "LIST_CANDIDATE") == 0) {
			any_list = true;
			(void)g_phase58_sub_list_n.fetch_add(1, std::memory_order_relaxed);
		}
		static std::atomic<uint32_t> sub_logs {0};
		if (sub_logs.fetch_add(1, std::memory_order_relaxed) < 64 ||
		    std::strcmp(s.type, "LIST_CANDIDATE") == 0) {
			LOGF("SubmitTrace: phase58 subblock base=0x%016" PRIx64 " win=%s type=%s guest_n=%u "
			     "user_n=%u scalar_n=%u tag=%s U=0x%016" PRIx64 "\n",
			     w.base, w.name, s.type, s.guest_n, s.user_n, s.scalar_n,
			     tag != nullptr ? tag : "?", U);
			fprintf(stderr, "SubmitTrace: phase58 subblock win=%s type=%s U=0x%016" PRIx64 "\n",
			        w.name, s.type, U);
		}
		Phase58ElectIfList(w.base, s, tag);
	}
	(void)any_list;
	g_phase58_subblocks_done.store(true, std::memory_order_release);
}

static void Phase58SeedPtr(uint64_t va, const char* tag) {
	if (va < 0x10000ULL || Phase56IsBannedBase(va)) {
		return;
	}
	const uint64_t obj = g_phase58_ndjob_obj.load(std::memory_order_acquire);
	if (obj != 0 && Phase58IsDeepScanInterior(obj, va) && !Phase52IsUserPtr(va)) {
		return; // skip interior already falsified by P53
	}
	auto* slot = Phase58FindOrAddSlot(va);
	if (slot == nullptr && !Phase52IsUserPtr(va)) {
		return;
	}
	if (Phase52IsUserPtr(va) || Phase52IsGuestPtr(va)) {
		Phase58ClassifySubblocks(va, tag);
	}
	if (Phase52IsGuestPtr(va) && (va & 7ULL) == 0) {
		const Phase58WinScore s = Phase58ClassifyWindow(va, 256);
		if (slot != nullptr) {
			std::snprintf(slot->type, sizeof(slot->type), "%s", s.type);
		}
		Phase58ElectIfList(va, s, tag);
	}
}

void Phase58NoteNdJobAncre(const char* why) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const uint64_t obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	if (obj < 0x10000ULL || Phase56IsBannedBase(obj) || Phase57IsNearSync(obj)) {
		return;
	}
	g_phase58_ndjob_obj.store(obj, std::memory_order_release);
	uint64_t head = 0;
	uint32_t gate = 0;
	uint64_t u10  = 0;
	uint64_t u40  = 0;
	uint64_t field0 = 0;
	uint64_t u20860 = 0;
	uint64_t u20878 = 0;
	Phase41SafeRead(&head, reinterpret_cast<const void*>(obj), sizeof(head));
	Phase41SafeRead(&gate, reinterpret_cast<const void*>(obj + 8), sizeof(gate));
	Phase41SafeRead(&u10, reinterpret_cast<const void*>(obj + 0x10), sizeof(u10));
	Phase41SafeRead(&u40, reinterpret_cast<const void*>(obj + 0x40), sizeof(u40));
	Phase41SafeRead(&field0, reinterpret_cast<const void*>(obj + kPhase41Keep1FieldOff),
	                sizeof(field0));
	Phase41SafeRead(&u20860, reinterpret_cast<const void*>(obj + 0x20860), sizeof(u20860));
	Phase41SafeRead(&u20878, reinterpret_cast<const void*>(obj + 0x20878), sizeof(u20878));
	uint64_t status = g_phase41_status_rdi.load(std::memory_order_acquire);
	if (status < 0x10000ULL) {
		status = kPhase58StatusDefault;
	}
	g_phase58_status.store(status, std::memory_order_release);
	const uint32_t run = g_phase58_ancre_runs.fetch_add(1, std::memory_order_relaxed) + 1;
	if (run <= 24 || (run % 32) == 0) {
		LOGF("SubmitTrace: phase58 ndjob_ancre why=%s obj=0x%016" PRIx64 " *obj=0x%016" PRIx64
		     " gate=%u u10=0x%016" PRIx64 " u40=0x%016" PRIx64 " field=0x%016" PRIx64
		     " status=0x%016" PRIx64 " run=%u\n",
		     why != nullptr ? why : "?", obj, head, gate, u10, u40, field0, status, run);
		fprintf(stderr, "SubmitTrace: phase58 ndjob_ancre obj=0x%016" PRIx64 " run=%u\n", obj, run);
	}
	// Phase 59 seed every ancre (cheap); P58 classify remains rate-limited below.
	Phase59SeedNdJobAnchors(obj, status, u10, u40);
	// Full classify / subblocks on first hits and periodically (avoid poll spam).
	if (run > 8 && (run % 25) != 0) {
		return;
	}
	Phase58SeedPtr(obj, "ndjob_obj");
	Phase58SeedPtr(obj + kPhase41Keep1FieldOff, "field_page");
	Phase58SeedPtr(status, "status");
	for (int64_t d = -0x100; d <= 0x100; d += 0x40) {
		Phase58SeedPtr(static_cast<uint64_t>(static_cast<int64_t>(status) + d), "status_win");
	}
	uint8_t head64[64] {};
	Phase41SafeRead(head64, reinterpret_cast<const void*>(obj), sizeof(head64));
	for (size_t off = 0; off + 8 <= sizeof(head64); off += 8) {
		uint64_t v = 0;
		std::memcpy(&v, head64 + off, 8);
		if (Phase52IsUserPtr(v) || Phase52IsGuestPtr(v)) {
			Phase58SeedPtr(v, "head64");
		}
	}
	Phase58SeedPtr(u10, "u10");
	Phase58SeedPtr(u40, "u40");
	Phase58SeedPtr(u20860, "u20860");
	Phase58SeedPtr(u20878, "u20878");
}

static void Phase58StabilizeCtx(std::atomic<uint64_t>& hist, std::atomic<uint32_t>& hits,
                                uint64_t val, const char* which) {
	if (!Phase52IsGuestPtr(val) || Phase56IsBannedBase(val) || !Phase58InNdJobCtxBand(val)) {
		return;
	}
	const uint64_t obj = g_phase58_ndjob_obj.load(std::memory_order_acquire);
	if (obj != 0 && val == obj) {
		return;
	}
	const uint64_t prev = hist.load(std::memory_order_acquire);
	if (prev == val) {
		const uint32_t h = hits.fetch_add(1, std::memory_order_relaxed) + 1;
		if (h == kPhase58StabHitsNeed) {
			g_phase58_stable_ctx.store(val, std::memory_order_release);
			LOGF("SubmitTrace: phase58 ctx_ndjob which=%s va=0x%016" PRIx64 " hits=%u\n", which,
			     val, h);
			fprintf(stderr, "SubmitTrace: phase58 ctx_ndjob which=%s va=0x%016" PRIx64 "\n", which,
			        val);
			Phase58SeedPtr(val, "ctx_ndjob");
			Phase58ClassifySubblocks(val, "ctx_ndjob");
		}
	} else {
		hist.store(val, std::memory_order_release);
		hits.store(1, std::memory_order_release);
	}
}

void Phase58NoteWorkerFiberFromAtomics() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	Phase58StabilizeCtx(g_phase58_w_rdi_hist, g_phase58_w_rdi_hits,
	                    g_phase53_worker_rdi.load(std::memory_order_acquire), "worker_rdi");
	Phase58StabilizeCtx(g_phase58_w_rsi_hist, g_phase58_w_rsi_hits,
	                    g_phase53_worker_rsi.load(std::memory_order_acquire), "worker_rsi");
	Phase58StabilizeCtx(g_phase58_f_rdi_hist, g_phase58_f_rdi_hits,
	                    g_phase53_fiber_rdi.load(std::memory_order_acquire), "fiber_rdi");
}

void Phase58NoteMainAgcCross(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire) ||
	    !LibKernel::PthreadCurrentIsMainRelated()) {
		return;
	}
	const uint64_t args[4] = {a0, a1, a2, a3};
	const uint64_t obj     = g_phase58_ndjob_obj.load(std::memory_order_acquire);
	const uint64_t status  = g_phase58_status.load(std::memory_order_acquire);
	const uint64_t ctx     = g_phase58_stable_ctx.load(std::memory_order_acquire);
	int            touch   = 0;
	for (uint64_t a: args) {
		if (a < 0x10000ULL || Phase56IsBannedBase(a)) {
			continue;
		}
		bool overlap = false;
		if (obj != 0 && a + 0x200 >= obj && a <= obj + 0x21000) {
			overlap = true;
		}
		if (status != 0 && a + 0x100 >= status && a <= status + 0x100) {
			overlap = true;
		}
		if (ctx != 0 && a + 0x100 >= ctx && a <= ctx + 0x100) {
			overlap = true;
		}
		if (Phase58InNdJobCtxBand(a)) {
			overlap = true;
			Phase58SeedPtr(a, "main_agc_ctx");
		}
		if (!overlap) {
			continue;
		}
		++touch;
		auto* slot = Phase58FindOrAddSlot(a);
		if (slot != nullptr) {
			(void)slot->main_touch.fetch_add(1, std::memory_order_relaxed);
			(void)slot->main_n.fetch_add(1, std::memory_order_relaxed);
		}
		(void)g_phase58_main_touch_n.fetch_add(1, std::memory_order_relaxed);
	}
	if (touch > 0) {
		static std::atomic<uint32_t> logs {0};
		const uint32_t               n = logs.fetch_add(1, std::memory_order_relaxed);
		if (n < 48) {
			LOGF("SubmitTrace: phase58 main_agc_cross nid=%s touch=%d a0=0x%016" PRIx64
			     " a1=0x%016" PRIx64 "\n",
			     nid != nullptr ? nid : "?", touch, a0, a1);
			fprintf(stderr, "SubmitTrace: phase58 main_agc_cross nid=%s touch=%d\n",
			        nid != nullptr ? nid : "?", touch);
		}
	}
}

void Phase58PollWatch() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	Phase58NoteNdJobAncre("poll");
	Phase58NoteWorkerFiberFromAtomics();
	const uint64_t base = g_phase58_queue_base.load(std::memory_order_acquire);
	if (!g_phase58_elected.load(std::memory_order_acquire) || base < 0x10000ULL ||
	    Phase56IsBannedBase(base)) {
		return;
	}
	auto* slot = Phase58FindOrAddSlot(base);
	if (slot == nullptr) {
		return;
	}
	uint8_t cur[64] {};
	Phase41SafeRead(cur, reinterpret_cast<const void*>(base), sizeof(cur));
	(void)slot->read_n.fetch_add(1, std::memory_order_relaxed);
	if (!slot->have_prev) {
		std::memcpy(slot->prev, cur, sizeof(cur));
		slot->have_prev = true;
		return;
	}
	bool changed = false;
	for (size_t i = 0; i < sizeof(cur); ++i) {
		if (slot->prev[i] != cur[i]) {
			changed = true;
			break;
		}
	}
	if (!changed) {
		return;
	}
	g_phase58_saw_mutation.store(true, std::memory_order_release);
	const uint32_t wn = slot->write_n.fetch_add(1, std::memory_order_relaxed) + 1;
	const char*    wr = "sampler";
	if (LibKernel::PthreadCurrentIsMainRelated()) {
		wr = "Main";
		(void)slot->main_n.fetch_add(1, std::memory_order_relaxed);
		(void)g_phase56_enqueue_from_main.fetch_add(1, std::memory_order_relaxed);
	} else {
		wr = "other";
		(void)slot->other_n.fetch_add(1, std::memory_order_relaxed);
		(void)g_phase56_enqueue_from_other.fetch_add(1, std::memory_order_relaxed);
	}
	(void)g_phase56_enqueue_write_n.fetch_add(1, std::memory_order_relaxed);
	if (wn <= 48 || (wn % 16) == 0) {
		LOGF("SubmitTrace: phase58 enqueue_write base=0x%016" PRIx64 " role=%s n=%u\n", base, wr,
		     wn);
		fprintf(stderr, "SubmitTrace: phase58 enqueue_write role=%s n=%u\n", wr, wn);
	}
	std::memcpy(slot->prev, cur, sizeof(cur));
}

void Phase58EmitHeatmap(const char* why) {
	if (g_phase58_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const int      elected = g_phase58_elected.load(std::memory_order_relaxed) ? 1 : 0;
	const uint64_t base    = g_phase58_queue_base.load(std::memory_order_relaxed);
	const uint32_t list_n  = g_phase58_list_n.load(std::memory_order_relaxed);
	const uint32_t sub_l   = g_phase58_sub_list_n.load(std::memory_order_relaxed);
	const uint32_t buf_n   = g_phase58_buffer_n.load(std::memory_order_relaxed);
	const bool     mut     = g_phase58_saw_mutation.load(std::memory_order_relaxed);
	const bool     sub_done = g_phase58_subblocks_done.load(std::memory_order_relaxed);
	uint32_t       writes  = 0;
	uint32_t       from_m  = 0;
	uint32_t       from_o  = 0;
	const uint32_t used    = g_phase58_heat_used.load(std::memory_order_relaxed);
	const uint32_t lim = used < kPhase58HeatSlots ? used : static_cast<uint32_t>(kPhase58HeatSlots);
	for (uint32_t i = 0; i < lim; ++i) {
		writes += g_phase58_heat[i].write_n.load(std::memory_order_relaxed);
		from_m += g_phase58_heat[i].main_n.load(std::memory_order_relaxed) +
		          g_phase58_heat[i].main_touch.load(std::memory_order_relaxed);
		from_o += g_phase58_heat[i].other_n.load(std::memory_order_relaxed);
	}
	from_m += g_phase58_main_touch_n.load(std::memory_order_relaxed);
	from_m += g_phase56_enqueue_from_main.load(std::memory_order_relaxed);
	from_o += g_phase56_enqueue_from_other.load(std::memory_order_relaxed);
	writes += g_phase56_enqueue_write_n.load(std::memory_order_relaxed);
	if (elected && from_m > 0) {
		Phase58SetCause("anchor_ok");
	} else if (elected && writes == 0 && !mut) {
		Phase58SetCause("LIST_STATIC_ONLY");
	} else if (elected && writes == 0) {
		Phase58SetCause("producer_dead");
	} else if (elected && from_o > 0 && from_m == 0) {
		Phase58SetCause("LIST_NON_MAIN");
	} else if (!elected && sub_done && buf_n > 0 && list_n == 0 && sub_l == 0) {
		Phase58SetCause("user_buffer_only");
	} else if (!elected) {
		Phase58SetCause("still_wrong_structure_ndjob");
	}
	LOGF("SubmitTrace: phase58 heatmap why=%s queue_base=0x%016" PRIx64 " type=%s elected=%d "
	     "cands=%u lists=%u sub_lists=%u buffers=%u writes=%u from_main=%u from_other=%u "
	     "main_touch=%u hop_reject=%u mut=%d ndjob=0x%016" PRIx64 " status=0x%016" PRIx64
	     " stable_ctx=0x%016" PRIx64 " cause=%s guest_draw=%d\n",
	     why != nullptr ? why : "?", base, g_phase58_base_type, elected,
	     g_phase58_cand_n.load(std::memory_order_relaxed), list_n, sub_l, buf_n, writes, from_m,
	     from_o, g_phase58_main_touch_n.load(std::memory_order_relaxed),
	     g_phase58_hop_reject_n.load(std::memory_order_relaxed), mut ? 1 : 0,
	     g_phase58_ndjob_obj.load(std::memory_order_relaxed),
	     g_phase58_status.load(std::memory_order_relaxed),
	     g_phase58_stable_ctx.load(std::memory_order_relaxed), g_phase58_cause,
	     g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0);
	fprintf(stderr, "SubmitTrace: phase58 heatmap cause=%s base=0x%016" PRIx64 " lists=%u\n",
	        g_phase58_cause, base, list_n);
}

uint64_t Phase58QueueBase() {
	return g_phase58_queue_base.load(std::memory_order_acquire);
}

bool Phase58Elected() {
	return g_phase58_elected.load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// Phase 59 — AGC guest VA ↔ host queue/stream/ctx mapping
// ---------------------------------------------------------------------------

static const char* Phase59OwnerRole() {
	char name[40] {};
	const auto self = LibKernel::PthreadSelfOrNull();
	if (self != nullptr) {
		(void)LibKernel::PthreadGetname(self, name);
	}
	if (std::strstr(name, "Mixed") != nullptr) {
		return "Mixed";
	}
	if (std::strstr(name, "Compute") != nullptr) {
		return "Compute";
	}
	if (self != nullptr && LibKernel::PthreadCurrentIsMainRelated()) {
		return "Main";
	}
	if (std::strstr(name, "GPU") != nullptr || std::strstr(name, "Gpu") != nullptr ||
	    std::strstr(name, "Submit") != nullptr) {
		return "GPU";
	}
	// Host watch / non-guest threads.
	if (self == nullptr) {
		return "other";
	}
	if (LibKernel::PthreadCurrentIsMainRelated()) {
		return "Main";
	}
	return "other";
}

static bool Phase59TagContains(const char* tags, const char* tag) {
	if (tags == nullptr || tag == nullptr || tag[0] == '\0') {
		return false;
	}
	const char* p = tags;
	const size_t tlen = std::strlen(tag);
	while (*p != '\0') {
		if (std::strncmp(p, tag, tlen) == 0 && (p[tlen] == '\0' || p[tlen] == '|')) {
			return true;
		}
		const char* bar = std::strchr(p, '|');
		if (bar == nullptr) {
			break;
		}
		p = bar + 1;
	}
	return false;
}

static void Phase59AppendTag(char* tags, size_t cap, const char* tag) {
	if (tags == nullptr || tag == nullptr || tag[0] == '\0' || cap == 0) {
		return;
	}
	if (tags[0] == '\0') {
		std::snprintf(tags, cap, "%s", tag);
		return;
	}
	if (Phase59TagContains(tags, tag)) {
		return;
	}
	const size_t cur = std::strlen(tags);
	if (cur + 1 + std::strlen(tag) >= cap) {
		return;
	}
	std::snprintf(tags + cur, cap - cur, "|%s", tag);
}

static void Phase59MergeKind(char* kind_buf, size_t cap, const char* kind) {
	if (kind_buf == nullptr || kind == nullptr || kind[0] == '\0') {
		return;
	}
	if (kind_buf[0] == '\0') {
		std::snprintf(kind_buf, cap, "%s", kind);
		return;
	}
	if (Phase59TagContains(kind_buf, kind)) {
		return;
	}
	// Prefer compound for ring+queue overlap.
	Phase59AppendTag(kind_buf, cap, kind);
}

static Phase59MapSlot* Phase59FindSlot(uint64_t va) {
	if (va < 0x10000ULL) {
		return nullptr;
	}
	const uint32_t used = g_phase59_map_used.load(std::memory_order_acquire);
	const uint32_t lim  = used < kPhase59MapSlots ? used : static_cast<uint32_t>(kPhase59MapSlots);
	for (uint32_t i = 0; i < lim; ++i) {
		if (g_phase59_map[i].guest_va.load(std::memory_order_acquire) == va) {
			return &g_phase59_map[i];
		}
	}
	return nullptr;
}

static Phase59MapSlot* Phase59FindOrAddSlot(uint64_t va) {
	Phase59MapSlot* existing = Phase59FindSlot(va);
	if (existing != nullptr) {
		return existing;
	}
	uint32_t idx = g_phase59_map_used.load(std::memory_order_relaxed);
	while (idx < kPhase59MapSlots) {
		if (g_phase59_map_used.compare_exchange_weak(idx, idx + 1, std::memory_order_acq_rel,
		                                             std::memory_order_relaxed)) {
			g_phase59_map[idx].guest_va.store(va, std::memory_order_release);
			return &g_phase59_map[idx];
		}
	}
	return nullptr;
}

static Phase59StreamSlot* Phase59FindStream(uint32_t stream_id) {
	const uint32_t used = g_phase59_stream_used.load(std::memory_order_acquire);
	const uint32_t lim =
	    used < kPhase59StreamSlots ? used : static_cast<uint32_t>(kPhase59StreamSlots);
	for (uint32_t i = 0; i < lim; ++i) {
		if (g_phase59_streams[i].stream_id.load(std::memory_order_acquire) == stream_id) {
			return &g_phase59_streams[i];
		}
	}
	return nullptr;
}

static Phase59StreamSlot* Phase59FindOrAddStream(uint32_t stream_id) {
	Phase59StreamSlot* existing = Phase59FindStream(stream_id);
	if (existing != nullptr) {
		return existing;
	}
	uint32_t idx = g_phase59_stream_used.load(std::memory_order_relaxed);
	while (idx < kPhase59StreamSlots) {
		if (g_phase59_stream_used.compare_exchange_weak(idx, idx + 1, std::memory_order_acq_rel,
		                                               std::memory_order_relaxed)) {
			g_phase59_streams[idx].stream_id.store(stream_id, std::memory_order_release);
			return &g_phase59_streams[idx];
		}
	}
	return nullptr;
}

static bool Phase59IsUserRingVa(uint64_t va) {
	if (!Phase52IsUserPtr(va)) {
		return false;
	}
	const uint64_t page = va & ~0xfffffull;
	return page == kPhase59UserRingDefault || page == kPhase59UserRingAlt ||
	       (va >= kPhase59UserRingDefault && va < kPhase59UserRingDefault + 0x400000ULL);
}

static bool Phase59IsCtxBand(uint64_t va) {
	return va >= kPhase59CtxBandLo && va < kPhase59CtxBandHi;
}

static void Phase59DumpRingHeaderOnce(uint64_t ring_va) {
	if (ring_va < 0x10000ULL) {
		return;
	}
	if (g_phase59_ring_dump_n.fetch_add(1, std::memory_order_relaxed) != 0) {
		return;
	}
	uint8_t hdr[128] {};
	Phase41SafeRead(hdr, reinterpret_cast<const void*>(ring_va), sizeof(hdr));
	LOGF("SubmitTrace: phase59 user_ring_hdr va=0x%016" PRIx64 "\n", ring_va);
	Phase41DumpHex("phase59_user_ring", hdr, 64);
	fprintf(stderr, "SubmitTrace: phase59 user_ring_hdr va=0x%016" PRIx64 "\n", ring_va);
}

void Phase59NoteGuestVa(const char* kind, uint64_t va, uint64_t host_id, const char* source_tag) {
	if (va < 0x10000ULL || Phase56IsBannedBase(va)) {
		return;
	}
	auto* slot = Phase59FindOrAddSlot(va);
	if (slot == nullptr) {
		return;
	}
	const bool first = (slot->kind[0] == '\0');
	const char* prev_tag = slot->source_tag;
	Phase59MergeKind(slot->kind, sizeof(slot->kind), kind != nullptr ? kind : "?");
	if (host_id != 0) {
		const uint64_t prev = slot->host_id.load(std::memory_order_relaxed);
		if (prev == 0) {
			slot->host_id.store(host_id, std::memory_order_release);
		}
	}
	// Fill owner/epoch if still empty (handles races on first insert).
	if (slot->owner_role[0] == '\0') {
		std::snprintf(slot->owner_role, sizeof(slot->owner_role), "%s", Phase59OwnerRole());
	}
	if (slot->first_seen_epoch[0] == '\0') {
		std::snprintf(slot->first_seen_epoch, sizeof(slot->first_seen_epoch), "%s",
		              g_phase37_post_unreg.load(std::memory_order_acquire) ? "post_unreg"
		                                                                  : "pre_unreg");
		if (std::strcmp(slot->first_seen_epoch, "pre_unreg") == 0) {
			(void)g_phase61_pre_agc_n.fetch_add(1, std::memory_order_relaxed);
		}
	}
	Phase59AppendTag(slot->source_tag, sizeof(slot->source_tag),
	                 source_tag != nullptr ? source_tag : "?");
	slot->last_tsc.store(LibKernel::KernelGetProcessTimeCounter(), std::memory_order_relaxed);
	if (LibKernel::PthreadCurrentIsMainRelated()) {
		(void)slot->main_n.fetch_add(1, std::memory_order_relaxed);
	}
	// Cross-link: stream_reg then submit on same VA.
	if (!first && source_tag != nullptr && std::strcmp(source_tag, "submit") == 0 &&
	    Phase59TagContains(prev_tag, "stream_reg")) {
		static std::atomic<uint32_t> link_logs {0};
		if (link_logs.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("SubmitTrace: phase59 link stream→submit va=0x%016" PRIx64 " kind=%s host=0x%" PRIx64
			     "\n",
			     va, slot->kind, slot->host_id.load(std::memory_order_relaxed));
			fprintf(stderr, "SubmitTrace: phase59 link stream→submit va=0x%016" PRIx64 "\n", va);
		}
	}
	static std::atomic<uint32_t> note_logs {0};
	const uint32_t               n = note_logs.fetch_add(1, std::memory_order_relaxed);
	if (n < 96 || first) {
		LOGF("SubmitTrace: phase59 map va=0x%016" PRIx64 " kind=%s host_id=0x%" PRIx64
		     " owner=%s epoch=%s tag=%s submit_n=%u main_n=%u first=%d\n",
		     va, slot->kind, slot->host_id.load(std::memory_order_relaxed), slot->owner_role,
		     slot->first_seen_epoch, slot->source_tag,
		     slot->submit_n.load(std::memory_order_relaxed),
		     slot->main_n.load(std::memory_order_relaxed), first ? 1 : 0);
		if (n < 48 || first) {
			fprintf(stderr, "SubmitTrace: phase59 map va=0x%016" PRIx64 " kind=%s tag=%s\n", va,
			        slot->kind, slot->source_tag);
		}
	}
}

void Phase59NoteSubmit(uint32_t queue, const uint32_t* dcb, uint32_t size_in_dwords,
                       const char* submit_kind) {
	(void)g_phase59_submit_n.fetch_add(1, std::memory_order_relaxed);
	if (submit_kind != nullptr && std::strcmp(submit_kind, "guest_real") == 0) {
		(void)g_phase59_guest_real_n.fetch_add(1, std::memory_order_relaxed);
		if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
			(void)g_phase61_post_guest_real_n.fetch_add(1, std::memory_order_relaxed);
			(void)g_phase63_submit_guest_real_post.fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (submit_kind != nullptr && std::strcmp(submit_kind, "seed_host") == 0 &&
	    g_phase37_post_unreg.load(std::memory_order_acquire)) {
		(void)g_phase62_post_seed_n.fetch_add(1, std::memory_order_relaxed);
		(void)g_phase63_submit_seed_post.fetch_add(1, std::memory_order_relaxed);
	}
	const uint64_t dcb_va = reinterpret_cast<uint64_t>(dcb);
	const uint64_t host_q = static_cast<uint64_t>(queue);
	int            mapped = 0;
	if (dcb != nullptr && size_in_dwords != 0 &&
	    (Phase52IsGuestPtr(dcb_va) || Phase52IsUserPtr(dcb_va))) {
		const char* cb_kind = Phase59IsUserRingVa(dcb_va) ? "user_ring|cb" : "cb";
		Phase59NoteGuestVa(cb_kind, dcb_va, host_q, "submit");
		auto* slot = Phase59FindSlot(dcb_va);
		if (slot != nullptr) {
			(void)slot->submit_n.fetch_add(1, std::memory_order_relaxed);
		}
		++mapped;
		if (Phase59IsUserRingVa(dcb_va)) {
			const uint64_t ring = dcb_va & ~0xfffffull;
			Phase59NoteGuestVa("user_ring|queue", ring, host_q, "submit");
			auto* rslot = Phase59FindSlot(ring);
			if (rslot != nullptr) {
				(void)rslot->submit_n.fetch_add(1, std::memory_order_relaxed);
			}
		}
	}
	// Host queue id without a guest queue VA → orphan counter.
	if (queue != 0 && mapped == 0) {
		(void)g_phase59_orphan_n.fetch_add(1, std::memory_order_relaxed);
	} else if (mapped > 0) {
		(void)g_phase59_submit_mapped_n.fetch_add(1, std::memory_order_relaxed);
	}
	static std::atomic<uint32_t> logs {0};
	const uint32_t               n = logs.fetch_add(1, std::memory_order_relaxed);
	if (n < 64 || (submit_kind != nullptr && std::strcmp(submit_kind, "guest_real") == 0)) {
		LOGF("SubmitTrace: phase59 submit kind=%s queue=0x%" PRIx32 " dcb=0x%016" PRIx64
		     " size=0x%" PRIx32 " main=%d mapped=%d\n",
		     submit_kind != nullptr ? submit_kind : "?", queue, dcb_va, size_in_dwords,
		     LibKernel::PthreadCurrentIsMainRelated() ? 1 : 0, mapped);
		fprintf(stderr, "SubmitTrace: phase59 submit kind=%s queue=0x%" PRIx32 " mapped=%d\n",
		        submit_kind != nullptr ? submit_kind : "?", queue, mapped);
	}
}

void Phase59NoteWorkloadStream(uint32_t stream_id, const void* stream) {
	const uint64_t stream_va = reinterpret_cast<uint64_t>(stream);
	if (stream == nullptr) {
		return;
	}
	auto* sslot = Phase59FindOrAddStream(stream_id);
	if (sslot != nullptr) {
		sslot->stream_va.store(stream_va, std::memory_order_release);
	}
	Phase59NoteGuestVa("stream", stream_va, static_cast<uint64_t>(stream_id), "stream_reg");
	uint8_t rec[32] {};
	Phase41SafeRead(rec, stream, sizeof(rec));
	uint64_t ctx_va   = 0;
	uint64_t queue_va = 0;
	for (size_t off = 0; off + 8 <= sizeof(rec); off += 8) {
		uint64_t v = 0;
		std::memcpy(&v, rec + off, 8);
		if (Phase59IsCtxBand(v) || Phase52IsGuestPtr(v)) {
			if (ctx_va == 0 && Phase59IsCtxBand(v)) {
				ctx_va = v;
			}
			Phase59NoteGuestVa(Phase59IsCtxBand(v) ? "ctx" : "queue", v,
			                   static_cast<uint64_t>(stream_id), "stream_reg");
		} else if (Phase59IsUserRingVa(v) || Phase52IsUserPtr(v)) {
			if (queue_va == 0) {
				queue_va = v;
			}
			Phase59NoteGuestVa(Phase59IsUserRingVa(v) ? "user_ring|queue" : "queue", v,
			                   static_cast<uint64_t>(stream_id), "stream_reg");
		}
	}
	if (sslot != nullptr) {
		if (ctx_va != 0) {
			sslot->ctx_va.store(ctx_va, std::memory_order_release);
		}
		if (queue_va != 0) {
			sslot->queue_va.store(queue_va, std::memory_order_release);
		}
	}
	static std::atomic<uint32_t> logs {0};
	if (logs.fetch_add(1, std::memory_order_relaxed) < 48) {
		LOGF("SubmitTrace: phase59 stream_reg id=%u stream=0x%016" PRIx64 " ctx=0x%016" PRIx64
		     " queue=0x%016" PRIx64 "\n",
		     stream_id, stream_va, ctx_va, queue_va);
		Phase41DumpHex("phase59_stream_rec", rec, 32);
		fprintf(stderr, "SubmitTrace: phase59 stream_reg id=%u ctx=0x%016" PRIx64 "\n", stream_id,
		        ctx_va);
	}
}

void Phase59NoteWorkloadActive(uint32_t stream_id, const char* why) {
	auto* sslot = Phase59FindStream(stream_id);
	uint64_t ctx_va   = 0;
	uint64_t queue_va = 0;
	uint64_t stream_va = 0;
	if (sslot != nullptr) {
		(void)sslot->active_n.fetch_add(1, std::memory_order_relaxed);
		ctx_va    = sslot->ctx_va.load(std::memory_order_acquire);
		queue_va  = sslot->queue_va.load(std::memory_order_acquire);
		stream_va = sslot->stream_va.load(std::memory_order_acquire);
	}
	static std::atomic<uint32_t> logs {0};
	if (logs.fetch_add(1, std::memory_order_relaxed) < 64) {
		LOGF("SubmitTrace: phase59 stream_active why=%s stream_id=%u stream=0x%016" PRIx64
		     " ctx=0x%016" PRIx64 " queue=0x%016" PRIx64 "\n",
		     why != nullptr ? why : "?", stream_id, stream_va, ctx_va, queue_va);
		fprintf(stderr, "SubmitTrace: phase59 stream_active id=%u ctx=0x%016" PRIx64 "\n",
		        stream_id, ctx_va);
	}
}

void Phase59NoteStubArgs(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
	const uint64_t args[4] = {a0, a1, a2, a3};
	const char*    classes[4] {};
	const char*    hint = nullptr;
	if (nid != nullptr) {
		if (std::strstr(nid, "CreateContext") != nullptr ||
		    std::strstr(nid, "Context") != nullptr) {
			hint = "ctx";
		} else if (std::strstr(nid, "Queue") != nullptr) {
			hint = "queue";
		} else if (std::strstr(nid, "Stream") != nullptr) {
			hint = "stream";
		}
	}
	for (int i = 0; i < 4; ++i) {
		const uint64_t a = args[i];
		if (Phase52IsUserPtr(a)) {
			classes[i] = "user_ptr";
			Phase59NoteGuestVa(Phase59IsUserRingVa(a) ? "user_ring" : "queue", a, 0, "stub_arg");
		} else if (Phase52IsGuestPtr(a)) {
			classes[i] = "guest_ptr";
			const char* k = hint;
			if (k == nullptr) {
				if (Phase59IsCtxBand(a)) {
					k = "ctx";
				} else {
					continue; // skip untyped guest ptr outside ctx band
				}
			}
			Phase59NoteGuestVa(k, a, 0, "stub_arg");
		} else {
			classes[i] = "scalar";
		}
	}
	static std::atomic<uint32_t> logs {0};
	const uint32_t               n = logs.fetch_add(1, std::memory_order_relaxed);
	if (n < 96 || LibKernel::PthreadCurrentIsMainRelated()) {
		LOGF("SubmitTrace: phase59 stub_args nid=%s tid=%d main=%d "
		     "a0=%s:0x%016" PRIx64 " a1=%s:0x%016" PRIx64 " a2=%s:0x%016" PRIx64
		     " a3=%s:0x%016" PRIx64 "\n",
		     nid != nullptr ? nid : "?", Common::Thread::GetThreadIdUnique(),
		     LibKernel::PthreadCurrentIsMainRelated() ? 1 : 0, classes[0], a0, classes[1], a1,
		     classes[2], a2, classes[3], a3);
		if (n < 48) {
			fprintf(stderr, "SubmitTrace: phase59 stub_args nid=%s\n", nid != nullptr ? nid : "?");
		}
	}
}

void Phase59NoteEq(int id, uint64_t eq_va, uint32_t context_id, const char* why) {
	(void)g_phase59_eq_n.fetch_add(1, std::memory_order_relaxed);
	if (context_id != 0) {
		// Bind context_id onto any already-mapped ctx/queue slot sharing this host_id,
		// and record a synthetic eq entry if eq_va looks like a guest ptr.
		const uint32_t used = g_phase59_map_used.load(std::memory_order_acquire);
		const uint32_t lim =
		    used < kPhase59MapSlots ? used : static_cast<uint32_t>(kPhase59MapSlots);
		for (uint32_t i = 0; i < lim; ++i) {
			auto& slot = g_phase59_map[i];
			const uint64_t va = slot.guest_va.load(std::memory_order_acquire);
			if (va == 0) {
				continue;
			}
			if (slot.host_id.load(std::memory_order_relaxed) == context_id) {
				Phase59AppendTag(slot.source_tag, sizeof(slot.source_tag), "eq");
			}
		}
		if (Phase52IsGuestPtr(eq_va) || Phase52IsUserPtr(eq_va)) {
			Phase59NoteGuestVa("ctx", eq_va, context_id, "eq");
		}
	}
	static std::atomic<uint32_t> logs {0};
	if (logs.fetch_add(1, std::memory_order_relaxed) < 48) {
		LOGF("SubmitTrace: phase59 eq why=%s id=%d eq=0x%016" PRIx64 " context_id=%u "
		     "ndjob=0x%016" PRIx64 "\n",
		     why != nullptr ? why : "?", id, eq_va, context_id,
		     g_phase58_ndjob_obj.load(std::memory_order_relaxed));
		fprintf(stderr, "SubmitTrace: phase59 eq id=%d context_id=%u\n", id, context_id);
	}
}

void Phase59SeedNdJobAnchors(uint64_t ndjob, uint64_t status, uint64_t user10, uint64_t user40) {
	if (ndjob >= 0x10000ULL) {
		Phase59NoteGuestVa("ndjob_ctrl", ndjob, 0, "ndjob_slot");
	}
	const uint64_t st = status >= 0x10000ULL ? status : kPhase59StatusDefault;
	Phase59NoteGuestVa("ndjob_status", st, 0, "ndjob_slot");
	uint64_t ring = 0;
	if (Phase59IsUserRingVa(user10)) {
		ring = user10 & ~0xfffffull;
	} else if (Phase59IsUserRingVa(user40)) {
		ring = user40 & ~0xfffffull;
	} else {
		ring = kPhase59UserRingDefault;
	}
	Phase59NoteGuestVa("user_ring", ring, 0, "ndjob_slot");
	if (Phase59IsUserRingVa(user10)) {
		Phase59NoteGuestVa("user_ring", user10, 0, "ndjob_slot");
	}
	if (Phase59IsUserRingVa(user40)) {
		Phase59NoteGuestVa("user_ring", user40, 0, "ndjob_slot");
	}
	Phase59DumpRingHeaderOnce(ring);
	if (!g_phase59_seeded.exchange(true, std::memory_order_acq_rel)) {
		LOGF("SubmitTrace: phase59 seed ndjob_ctrl=0x%016" PRIx64 " ndjob_status=0x%016" PRIx64
		     " user_ring=0x%016" PRIx64 "\n",
		     ndjob, st, ring);
		fprintf(stderr, "SubmitTrace: phase59 seed ctrl=0x%016" PRIx64 " ring=0x%016" PRIx64 "\n",
		        ndjob, ring);
		g_phase59_heatmap_done.store(false, std::memory_order_release);
		Phase59EmitHeatmap("seed");
	}
}

void Phase59Poll() {
	// Lightweight: ensure anchors stay present once NdJob is known.
	const uint64_t obj = g_phase58_ndjob_obj.load(std::memory_order_acquire);
	if (obj >= 0x10000ULL && !g_phase59_seeded.load(std::memory_order_acquire)) {
		Phase59SeedNdJobAnchors(obj, g_phase58_status.load(std::memory_order_acquire),
		                        kPhase59UserRingDefault, kPhase59UserRingAlt);
	}
	// Emit heatmap every ~10s of watch so short post-Unreg windows still verdict.
	static std::atomic<uint32_t> polls {0};
	const uint32_t n = polls.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n == 1 || (n % 100) == 0) {
		g_phase59_heatmap_done.store(false, std::memory_order_release);
		Phase59EmitHeatmap(n == 1 ? "poll_first" : "poll_10s");
	}
}

static void Phase59SetCause(const char* c) {
	if (c == nullptr) {
		return;
	}
	std::snprintf(g_phase59_cause, sizeof(g_phase59_cause), "%s", c);
}

void Phase59EmitHeatmap(const char* why) {
	if (g_phase59_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t used = g_phase59_map_used.load(std::memory_order_relaxed);
	const uint32_t lim  = used < kPhase59MapSlots ? used : static_cast<uint32_t>(kPhase59MapSlots);
	uint32_t       ctx_n = 0;
	uint32_t       queue_n = 0;
	uint32_t       stream_n = 0;
	uint32_t       ring_n = 0;
	uint32_t       ctrl_n = 0;
	uint32_t       mapped_submit = 0;
	uint32_t       post_n = 0;
	for (uint32_t i = 0; i < lim; ++i) {
		auto&          slot = g_phase59_map[i];
		const uint64_t va   = slot.guest_va.load(std::memory_order_relaxed);
		if (va == 0) {
			continue;
		}
		const uint32_t sn = slot.submit_n.load(std::memory_order_relaxed);
		const uint64_t hid = slot.host_id.load(std::memory_order_relaxed);
		if (std::strcmp(slot.first_seen_epoch, "post_unreg") == 0) {
			++post_n;
		}
		const bool is_ctx    = Phase59TagContains(slot.kind, "ctx");
		const bool is_queue  = Phase59TagContains(slot.kind, "queue");
		const bool is_stream = Phase59TagContains(slot.kind, "stream");
		const bool is_ring   = Phase59TagContains(slot.kind, "user_ring");
		const bool is_ctrl   = Phase59TagContains(slot.kind, "ndjob_ctrl") ||
		                     Phase59TagContains(slot.kind, "ndjob_status");
		if (is_ctx) {
			++ctx_n;
		}
		if (is_queue) {
			++queue_n;
		}
		if (is_stream) {
			++stream_n;
		}
		if (is_ring) {
			++ring_n;
		}
		if (is_ctrl) {
			++ctrl_n;
		}
		if ((is_ctx || is_queue || is_stream) && hid != 0 && sn > 0) {
			++mapped_submit;
		}
		LOGF("SubmitTrace: phase59 map_dump[%u] va=0x%016" PRIx64 " kind=%s host_id=0x%" PRIx64
		     " owner=%s epoch=%s tag=%s submit_n=%u main_n=%u\n",
		     i, va, slot.kind, hid, slot.owner_role, slot.first_seen_epoch, slot.source_tag, sn,
		     slot.main_n.load(std::memory_order_relaxed));
	}
	const uint32_t submit_n  = g_phase59_submit_n.load(std::memory_order_relaxed);
	const uint32_t orphan_n  = g_phase59_orphan_n.load(std::memory_order_relaxed);
	const uint32_t guest_real = g_phase59_guest_real_n.load(std::memory_order_relaxed);
	const uint32_t mapped_n  = g_phase59_submit_mapped_n.load(std::memory_order_relaxed);
	if (mapped_submit > 0) {
		Phase59SetCause("ctx_mapped");
	} else if (submit_n > 0 && orphan_n > 0 && mapped_n == 0 && ctx_n == 0 && stream_n == 0) {
		Phase59SetCause("queue_orphan");
	} else if (guest_real > 0 && mapped_submit == 0 && ctx_n == 0 && stream_n == 0) {
		Phase59SetCause("submit_without_ctx");
	} else if ((ctrl_n > 0 || ring_n > 0) && ctx_n == 0 && stream_n == 0 && mapped_submit == 0) {
		Phase59SetCause("user_ring_only");
	} else if (used == 0 || post_n == 0) {
		Phase59SetCause("still_opaque");
	} else {
		Phase59SetCause("still_opaque");
	}
	LOGF("SubmitTrace: phase59 heatmap why=%s cause=%s used=%u post=%u ctx=%u queue=%u stream=%u "
	     "ring=%u ctrl=%u mapped_submit=%u submit=%u orphan=%u guest_real=%u eq=%u "
	     "streams=%u guest_draw=%d\n",
	     why != nullptr ? why : "?", g_phase59_cause, used, post_n, ctx_n, queue_n, stream_n, ring_n,
	     ctrl_n, mapped_submit, submit_n, orphan_n, guest_real,
	     g_phase59_eq_n.load(std::memory_order_relaxed),
	     g_phase59_stream_used.load(std::memory_order_relaxed),
	     g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0);
	fprintf(stderr, "SubmitTrace: phase59 heatmap cause=%s used=%u mapped_submit=%u\n",
	        g_phase59_cause, used, mapped_submit);
}

// ---------------------------------------------------------------------------
// Phase 61 — unlock gate: read-only user_ring PM4 / ptr probe
// ---------------------------------------------------------------------------

static bool Phase61IsKnownPm4Op(uint32_t op) {
	using namespace Libs::Graphics::Pm4;
	switch (op) {
	case IT_NOP:
	case IT_SET_BASE:
	case IT_CLEAR_STATE:
	case IT_INDEX_BUFFER_SIZE:
	case IT_DISPATCH_DIRECT:
	case IT_DISPATCH_INDIRECT:
	case IT_SET_PREDICATION:
	case IT_COND_EXEC:
	case IT_DRAW_INDIRECT:
	case IT_DRAW_INDEX_INDIRECT:
	case IT_INDEX_BASE:
	case IT_DRAW_INDEX_2:
	case IT_CONTEXT_CONTROL:
	case IT_INDEX_TYPE:
	case IT_DRAW_INDIRECT_MULTI:
	case IT_DRAW_INDEX_AUTO:
	case IT_NUM_INSTANCES:
	case IT_INDIRECT_BUFFER_CNST:
	case IT_DRAW_INDEX_OFFSET_2:
	case IT_WRITE_DATA:
	case IT_DRAW_INDEX_INDIRECT_MULTI:
	case IT_INDIRECT_BUFFER:
	case IT_COPY_DATA:
	case IT_CP_DMA:
	case IT_EVENT_WRITE:
	case IT_EVENT_WRITE_EOP:
	case IT_EVENT_WRITE_EOS:
	case IT_RELEASE_MEM:
	case IT_DMA_DATA:
	case IT_ACQUIRE_MEM:
	case IT_REWIND:
	case IT_SET_CONFIG_REG:
	case IT_SET_CONTEXT_REG:
	case IT_SET_SH_REG:
	case IT_SET_UCONFIG_REG:
	case IT_WAIT_ON_CE_COUNTER:
	case IT_WAIT_ON_DE_COUNTER_DIFF:
	case 0x3Cu: // SharpEmu WAIT_REG_MEM
		return true;
	default:
		return false;
	}
}

static void Phase61ClassifyRing(const uint8_t* buf, size_t nbytes, uint32_t* pm4_out,
                                uint32_t* ptr_g_out, uint32_t* ptr_u_out) {
	uint32_t pm4 = 0;
	uint32_t pg  = 0;
	uint32_t pu  = 0;
	for (size_t off = 0; off + 4 <= nbytes; off += 4) {
		uint32_t dw = 0;
		std::memcpy(&dw, buf + off, 4);
		if ((dw >> 30) == 3u) {
			const uint32_t op = (dw >> 8) & 0xffu;
			if (Phase61IsKnownPm4Op(op)) {
				++pm4;
			}
		}
	}
	for (size_t off = 0; off + 8 <= nbytes; off += 8) {
		uint64_t v = 0;
		std::memcpy(&v, buf + off, 8);
		if (Phase52IsGuestPtr(v)) {
			++pg;
		} else if (Phase52IsUserPtr(v)) {
			++pu;
		}
	}
	*pm4_out   = pm4;
	*ptr_g_out = pg;
	*ptr_u_out = pu;
}

void Phase61RingProbe() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	uint64_t ring = kPhase59UserRingDefault;
	// Prefer seeded map user_ring VA if present.
	const uint32_t used = g_phase59_map_used.load(std::memory_order_acquire);
	const uint32_t lim  = used < kPhase59MapSlots ? used : static_cast<uint32_t>(kPhase59MapSlots);
	for (uint32_t i = 0; i < lim; ++i) {
		if (Phase59TagContains(g_phase59_map[i].kind, "user_ring")) {
			const uint64_t va = g_phase59_map[i].guest_va.load(std::memory_order_relaxed);
			if (Phase59IsUserRingVa(va)) {
				ring = va & ~0xfffffull;
				break;
			}
		}
	}

	uint8_t cur[kPhase61RingBytes] {};
	Phase41SafeRead(cur, reinterpret_cast<const void*>(ring), sizeof(cur));
	uint32_t pm4 = 0;
	uint32_t pg  = 0;
	uint32_t pu  = 0;
	Phase61ClassifyRing(cur, sizeof(cur), &pm4, &pg, &pu);
	const uint32_t ptr_n = pg + pu;

	bool mutated = false;
	if (g_phase61_have_prev.load(std::memory_order_acquire)) {
		if (std::memcmp(g_phase61_prev, cur, sizeof(cur)) != 0) {
			mutated = true;
			(void)g_phase61_mut_n.fetch_add(1, std::memory_order_relaxed);
		}
	} else {
		g_phase61_have_prev.store(true, std::memory_order_release);
	}
	std::memcpy(g_phase61_prev, cur, sizeof(cur));

	(void)g_phase61_probe_n.fetch_add(1, std::memory_order_relaxed);
	(void)g_phase61_pm4_total.fetch_add(pm4, std::memory_order_relaxed);
	(void)g_phase61_ptr_total.fetch_add(ptr_n, std::memory_order_relaxed);
	const int rich = (pm4 >= kPhase61Pm4M) ? 1 : 0;
	if (rich) {
		(void)g_phase61_rich_n.fetch_add(1, std::memory_order_relaxed);
	}

	const uint64_t tsc = LibKernel::KernelGetProcessTimeCounter();
	const uint32_t n   = g_phase61_probe_n.load(std::memory_order_relaxed);
	char           magic[8] {};
	std::memcpy(magic, cur, 4);
	magic[4] = '\0';
	if (n <= 24 || (n % 8) == 0 || rich || mutated) {
		LOGF("SubmitTrace: phase61 ring_probe va=0x%016" PRIx64 " tsc=%" PRIu64
		     " epoch=post_unreg pm4=%u ptr_g=%u ptr_u=%u mut=%d rich=%d probe_n=%u magic=%s\n",
		     ring, tsc, pm4, pg, pu, mutated ? 1 : 0, rich, n, magic);
		fprintf(stderr,
		        "SubmitTrace: phase61 ring_probe pm4=%u ptr=%u mut=%d rich=%d n=%u magic=%s\n", pm4,
		        ptr_n, mutated ? 1 : 0, rich, n, magic);
	}

	// Also probe alt base once in a while if distinct.
	if ((n % 16) == 1) {
		uint8_t alt[256] {};
		Phase41SafeRead(alt, reinterpret_cast<const void*>(kPhase59UserRingAlt), sizeof(alt));
		uint32_t ap = 0;
		uint32_t ag = 0;
		uint32_t au = 0;
		Phase61ClassifyRing(alt, sizeof(alt), &ap, &ag, &au);
		if (ap > 0 || ag + au > 0) {
			LOGF("SubmitTrace: phase61 ring_probe_alt va=0x%016" PRIx64 " pm4=%u ptr_g=%u ptr_u=%u\n",
			     kPhase59UserRingAlt, ap, ag, au);
		}
	}
}

static void Phase61SetCause(const char* c) {
	if (c == nullptr) {
		return;
	}
	std::snprintf(g_phase61_cause, sizeof(g_phase61_cause), "%s", c);
}

void Phase61EmitHeatmap(const char* why) {
	if (g_phase61_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t pm4_total = g_phase61_pm4_total.load(std::memory_order_relaxed);
	const uint32_t ptr_total = g_phase61_ptr_total.load(std::memory_order_relaxed);
	const uint32_t rich      = g_phase61_rich_n.load(std::memory_order_relaxed);
	const uint32_t mut       = g_phase61_mut_n.load(std::memory_order_relaxed);
	const uint32_t probe_n   = g_phase61_probe_n.load(std::memory_order_relaxed);
	const uint32_t pre_agc   = g_phase61_pre_agc_n.load(std::memory_order_relaxed);
	const uint32_t post_gr   = g_phase61_post_guest_real_n.load(std::memory_order_relaxed);

	if (pm4_total >= kPhase61Pm4X && rich >= kPhase61RichY && mut >= kPhase61MutNeed) {
		Phase61SetCause("ring_is_submit_path");
	} else if (pm4_total == 0 && post_gr == 0 && pre_agc > 0) {
		Phase61SetCause("producer_pre_unreg_only");
	} else {
		Phase61SetCause("still_opaque_ring");
	}

	LOGF("SubmitTrace: phase61 heatmap why=%s cause=%s pm4_total=%u ptr_total=%u rich=%u mut=%u "
	     "probe_n=%u pre_agc=%u post_guest_real=%u guest_draw=%d\n",
	     why != nullptr ? why : "?", g_phase61_cause, pm4_total, ptr_total, rich, mut, probe_n,
	     pre_agc, post_gr, g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0);
	fprintf(stderr,
	        "SubmitTrace: phase61 heatmap cause=%s pm4_total=%u ptr_total=%u rich=%u mut=%u "
	        "pre_agc=%u post_guest_real=%u\n",
	        g_phase61_cause, pm4_total, ptr_total, rich, mut, pre_agc, post_gr);
}

void Phase61Poll() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<uint32_t> ticks {0};
	const uint32_t               t = ticks.fetch_add(1, std::memory_order_relaxed) + 1;
	// ~2 Hz from 100ms watch loop (every 5 ticks).
	if ((t % 5) == 0) {
		Phase61RingProbe();
	}
	if (t == 1 || (t % 100) == 0) {
		g_phase61_heatmap_done.store(false, std::memory_order_release);
		Phase61EmitHeatmap(t == 1 ? "poll_first" : "poll_10s");
	}
}

// ---------------------------------------------------------------------------
// Phase 62 — producer silence post-Unreg (RegisterResource map; not KPRI gate)
// ---------------------------------------------------------------------------

static bool Phase62IsKpriMagic(const uint8_t* buf, size_t n) {
	return n >= 4 && buf[0] == 'K' && buf[1] == 'P' && buf[2] == 'R' && buf[3] == 'I';
}

void Phase62NoteUnreg() {
	const uint64_t tsc = LibKernel::KernelGetProcessTimeCounter();
	g_phase62_unreg_tsc.store(tsc, std::memory_order_release);
	LOGF("SubmitTrace: phase62 unreg_gate tsc=%" PRIu64 " epoch=post_unreg pre_agc=%u\n", tsc,
	     g_phase61_pre_agc_n.load(std::memory_order_relaxed));
	fprintf(stderr, "SubmitTrace: phase62 unreg_gate tsc=%" PRIu64 " pre_agc=%u\n", tsc,
	        g_phase61_pre_agc_n.load(std::memory_order_relaxed));
}

void Phase62NoteRegisterResource(uint64_t addr, uint64_t size, int res_type, const char* name,
                                 uint32_t handle) {
	if (addr < 0x10000ULL || Phase56IsBannedBase(addr)) {
		return;
	}
	const bool post = g_phase37_post_unreg.load(std::memory_order_acquire);
	if (post) {
		(void)g_phase62_post_agc_n.fetch_add(1, std::memory_order_relaxed);
	}

	uint8_t hdr[16] {};
	Phase41SafeRead(hdr, reinterpret_cast<const void*>(addr), sizeof(hdr));
	const bool kpri = Phase62IsKpriMagic(hdr, sizeof(hdr));

	// Find existing slot or allocate.
	Phase62ResSlot* slot = nullptr;
	const uint32_t  used = g_phase62_res_used.load(std::memory_order_acquire);
	const uint32_t  lim  = used < kPhase62ResSlots ? used : static_cast<uint32_t>(kPhase62ResSlots);
	for (uint32_t i = 0; i < lim; ++i) {
		if (g_phase62_res[i].va.load(std::memory_order_relaxed) == addr) {
			slot = &g_phase62_res[i];
			break;
		}
	}
	if (slot == nullptr) {
		const uint32_t idx = g_phase62_res_used.fetch_add(1, std::memory_order_acq_rel);
		if (idx >= kPhase62ResSlots) {
			return;
		}
		slot = &g_phase62_res[idx];
		slot->va.store(addr, std::memory_order_release);
		if (kpri) {
			(void)g_phase62_res_kpri_n.fetch_add(1, std::memory_order_relaxed);
		} else {
			(void)g_phase62_res_other_n.fetch_add(1, std::memory_order_relaxed);
		}
	}

	slot->size.store(size, std::memory_order_relaxed);
	slot->handle.store(handle, std::memory_order_relaxed);
	slot->res_type.store(res_type, std::memory_order_relaxed);
	slot->is_kpri.store(kpri, std::memory_order_relaxed);
	if (slot->epoch[0] == '\0') {
		std::snprintf(slot->epoch, sizeof(slot->epoch), "%s", post ? "post_unreg" : "pre_unreg");
	}
	if (name != nullptr && name[0] != '\0' && slot->name[0] == '\0') {
		std::snprintf(slot->name, sizeof(slot->name), "%s", name);
	}
	if (slot->magic[0] == '\0') {
		for (int i = 0; i < 4; ++i) {
			const char c = static_cast<char>(hdr[i]);
			slot->magic[i] = (c >= 32 && c < 127) ? c : '.';
		}
		slot->magic[4] = '\0';
	}

	static std::atomic<uint32_t> logs {0};
	const uint32_t               n = logs.fetch_add(1, std::memory_order_relaxed);
	if (n < 48 || kpri) {
		LOGF("SubmitTrace: phase62 resource va=0x%016" PRIx64 " size=0x%" PRIx64 " type=%d "
		     "handle=0x%" PRIx32 " name=%s magic=%s epoch=%s kpri=%d\n",
		     addr, size, res_type, handle, slot->name[0] ? slot->name : "?", slot->magic,
		     slot->epoch, kpri ? 1 : 0);
		if (n < 24 || kpri) {
			fprintf(stderr,
			        "SubmitTrace: phase62 resource va=0x%016" PRIx64 " name=%s magic=%s kpri=%d\n",
			        addr, slot->name[0] ? slot->name : "?", slot->magic, kpri ? 1 : 0);
		}
	}
}

static void Phase62ProbeResources() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const uint32_t used = g_phase62_res_used.load(std::memory_order_acquire);
	const uint32_t lim  = used < kPhase62ResSlots ? used : static_cast<uint32_t>(kPhase62ResSlots);
	for (uint32_t i = 0; i < lim; ++i) {
		auto&          slot = g_phase62_res[i];
		const uint64_t va   = slot.va.load(std::memory_order_relaxed);
		if (va < 0x10000ULL || Phase56IsBannedBase(va)) {
			continue;
		}
		// Skip KPRI blobs — proven not a submit ring (P61).
		if (slot.is_kpri.load(std::memory_order_relaxed)) {
			continue;
		}
		uint8_t cur[kPhase62ProbeBytes] {};
		Phase41SafeRead(cur, reinterpret_cast<const void*>(va), sizeof(cur));
		uint32_t pm4 = 0;
		uint32_t pg  = 0;
		uint32_t pu  = 0;
		Phase61ClassifyRing(cur, sizeof(cur), &pm4, &pg, &pu);
		const uint32_t ptr_n = pg + pu;
		slot.pm4_hits.store(pm4, std::memory_order_relaxed);
		slot.ptr_hits.store(ptr_n, std::memory_order_relaxed);

		bool mutated = false;
		if (slot.have_prev.load(std::memory_order_acquire)) {
			if (std::memcmp(slot.prev, cur, sizeof(cur)) != 0) {
				mutated = true;
				(void)slot.mut_n.fetch_add(1, std::memory_order_relaxed);
				(void)g_phase62_non_kpri_mut_n.fetch_add(1, std::memory_order_relaxed);
			}
		} else {
			slot.have_prev.store(true, std::memory_order_release);
		}
		std::memcpy(slot.prev, cur, sizeof(cur));

		if (pm4 >= kPhase62AltPm4Need &&
		    (mutated || slot.mut_n.load(std::memory_order_relaxed) >= kPhase62AltMutNeed) &&
		    ptr_n > 0) {
			(void)g_phase62_alt_anchor_n.fetch_add(1, std::memory_order_relaxed);
			static std::atomic<uint32_t> alt_logs {0};
			if (alt_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("SubmitTrace: phase62 alt_anchor va=0x%016" PRIx64 " pm4=%u ptr=%u mut=%u "
				     "name=%s\n",
				     va, pm4, ptr_n, slot.mut_n.load(std::memory_order_relaxed),
				     slot.name[0] ? slot.name : "?");
			}
		}
	}
}

static void Phase62SetCause(const char* c) {
	if (c == nullptr) {
		return;
	}
	std::snprintf(g_phase62_cause, sizeof(g_phase62_cause), "%s", c);
}

void Phase62EmitHeatmap(const char* why) {
	if (g_phase62_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t pre_agc     = g_phase61_pre_agc_n.load(std::memory_order_relaxed);
	const uint32_t post_gr     = g_phase61_post_guest_real_n.load(std::memory_order_relaxed);
	const uint32_t post_seed   = g_phase62_post_seed_n.load(std::memory_order_relaxed);
	const uint32_t post_agc    = g_phase62_post_agc_n.load(std::memory_order_relaxed);
	const uint32_t kpri_n      = g_phase62_res_kpri_n.load(std::memory_order_relaxed);
	const uint32_t other_n     = g_phase62_res_other_n.load(std::memory_order_relaxed);
	const uint32_t non_kpri_mut = g_phase62_non_kpri_mut_n.load(std::memory_order_relaxed);
	const uint32_t alt_n       = g_phase62_alt_anchor_n.load(std::memory_order_relaxed);
	const uint64_t unreg_tsc   = g_phase62_unreg_tsc.load(std::memory_order_relaxed);
	const uint32_t res_used    = g_phase62_res_used.load(std::memory_order_relaxed);
	const uint32_t res_cap =
	    res_used < static_cast<uint32_t>(kPhase62ResSlots) ? res_used
	                                                       : static_cast<uint32_t>(kPhase62ResSlots);

	if (post_gr > 0 || alt_n > 0) {
		Phase62SetCause("alternate_submit_anchor");
	} else if (post_gr == 0 && pre_agc >= kPhase62PreAgcNeed && non_kpri_mut == 0) {
		Phase62SetCause("producer_pre_unreg_only");
	} else {
		Phase62SetCause("still_opaque_producer");
	}

	LOGF("SubmitTrace: phase62 heatmap why=%s cause=%s pre_agc=%u post_guest_real=%u "
	     "post_seed=%u post_agc=%u res_used=%u res_cap=%u kpri=%u other=%u non_kpri_mut=%u alt=%u "
	     "unreg_tsc=%" PRIu64 " guest_draw=%d\n",
	     why != nullptr ? why : "?", g_phase62_cause, pre_agc, post_gr, post_seed, post_agc,
	     res_used, res_cap, kpri_n, other_n, non_kpri_mut, alt_n, unreg_tsc,
	     g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0);
	fprintf(stderr,
	        "SubmitTrace: phase62 heatmap cause=%s pre_agc=%u post_guest_real=%u post_seed=%u "
	        "res_used=%u res_cap=%u kpri=%u other=%u non_kpri_mut=%u alt=%u unreg_tsc=%" PRIu64 "\n",
	        g_phase62_cause, pre_agc, post_gr, post_seed, res_used, res_cap, kpri_n, other_n,
	        non_kpri_mut, alt_n, unreg_tsc);
}

void Phase62Poll() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<uint32_t> ticks {0};
	const uint32_t               t = ticks.fetch_add(1, std::memory_order_relaxed) + 1;
	// ~1 Hz resource probe (every 10 ticks of 100ms watch).
	if ((t % 10) == 0) {
		Phase62ProbeResources();
	}
	if (t == 1 || (t % 100) == 0) {
		g_phase62_heatmap_done.store(false, std::memory_order_release);
		Phase62EmitHeatmap(t == 1 ? "poll_first" : "poll_10s");
	}
}

// ---------------------------------------------------------------------------
// Phase 63 — Unreg forensics (Owner cycle_id + SubmitEntry; no P59 reclass)
// Priority: submit_attempted_blocked > unreg_cleared_owners > producer_never_armed
// ---------------------------------------------------------------------------

void Phase63NoteUnregister(uint32_t owner, int n_resources, const char* why, bool owner_event) {
	uint64_t cycle = g_phase63_cycle_id.load(std::memory_order_relaxed);
	if (owner_event) {
		cycle = g_phase63_cycle_id.fetch_add(1, std::memory_order_acq_rel) + 1;
		(void)g_phase63_unreg_owners.fetch_add(1, std::memory_order_relaxed);
		if (n_resources > 0) {
			(void)g_phase63_unreg_res_total.fetch_add(static_cast<uint32_t>(n_resources),
			                                          std::memory_order_relaxed);
		}
	}
	const bool     post = g_phase37_post_unreg.load(std::memory_order_acquire);
	const uint64_t tsc  = LibKernel::KernelGetProcessTimeCounter();
	if (post && owner_event && g_phase63_unreg_tsc.load(std::memory_order_relaxed) == 0) {
		g_phase63_unreg_tsc.store(tsc, std::memory_order_release);
	}
	static std::atomic<uint32_t> res_logs {0};
	if (owner_event || res_logs.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("SubmitTrace: phase63 unregister cycle_id=%" PRIu64 " owner=0x%08" PRIx32
		     " n_resources=%d epoch=%s why=%s owner_event=%d tsc=%" PRIu64 "\n",
		     cycle, owner, n_resources, post ? "post_unreg" : "pre_unreg",
		     why != nullptr ? why : "?", owner_event ? 1 : 0, tsc);
		if (owner_event) {
			fprintf(stderr,
			        "SubmitTrace: phase63 unregister cycle_id=%" PRIu64 " owner=0x%08" PRIx32
			        " n_resources=%d why=%s\n",
			        cycle, owner, n_resources, why != nullptr ? why : "?");
		}
	}
}

void Phase63NoteSubmitEntry(const char* api, uint32_t queue, const uint32_t* dcb,
                            uint32_t size_in_dwords) {
	const bool     post   = g_phase37_post_unreg.load(std::memory_order_acquire);
	const uint64_t dcb_va = reinterpret_cast<uint64_t>(dcb);
	const bool     guest_va =
	    dcb != nullptr && size_in_dwords != 0 &&
	    (Phase52IsGuestPtr(dcb_va) || Phase52IsUserPtr(dcb_va));
	const char* kind = "?";
	if (dcb == nullptr || size_in_dwords == 0) {
		kind = "empty";
	} else if (guest_va) {
		kind = "guest_va";
	} else {
		kind = "host_va";
	}
	if (post) {
		(void)g_phase63_submit_attempt_post.fetch_add(1, std::memory_order_relaxed);
		if (guest_va) {
			(void)g_phase63_attempt_guest_dcb_post.fetch_add(1, std::memory_order_relaxed);
		} else {
			(void)g_phase63_attempt_host_post.fetch_add(1, std::memory_order_relaxed);
		}
	}
	const int tid = Common::Thread::GetThreadIdUnique();
	static std::atomic<uint32_t> logs {0};
	const uint32_t               n = logs.fetch_add(1, std::memory_order_relaxed);
	if (n < 64 || (post && guest_va)) {
		LOGF("SubmitTrace: phase63 submit_entry api=%s queue=0x%" PRIx32 " dcb=0x%016" PRIx64
		     " size=0x%" PRIx32 " tid=%d epoch=%s kind=%s\n",
		     api != nullptr ? api : "?", queue, dcb_va, size_in_dwords, tid,
		     post ? "post_unreg" : "pre_unreg", kind);
		fprintf(stderr,
		        "SubmitTrace: phase63 submit_entry api=%s kind=%s tid=%d epoch=%s\n",
		        api != nullptr ? api : "?", kind, tid, post ? "post" : "pre");
	}
}

void Phase63NotePostAgc(const char* why) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	(void)g_phase63_post_agc.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> logs {0};
	if (logs.fetch_add(1, std::memory_order_relaxed) < 24) {
		LOGF("SubmitTrace: phase63 post_agc why=%s n=%u\n", why != nullptr ? why : "?",
		     g_phase63_post_agc.load(std::memory_order_relaxed));
	}
}

static void Phase63SetCause(const char* c) {
	if (c == nullptr) {
		return;
	}
	std::snprintf(g_phase63_cause, sizeof(g_phase63_cause), "%s", c);
}

void Phase63EmitHeatmap(const char* why) {
	if (g_phase63_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t unreg_owners = g_phase63_unreg_owners.load(std::memory_order_relaxed);
	const uint32_t unreg_res    = g_phase63_unreg_res_total.load(std::memory_order_relaxed);
	const uint32_t attempt_post = g_phase63_submit_attempt_post.load(std::memory_order_relaxed);
	const uint32_t guest_real =
	    g_phase63_submit_guest_real_post.load(std::memory_order_relaxed);
	const uint32_t seed_post  = g_phase63_submit_seed_post.load(std::memory_order_relaxed);
	const uint32_t guest_dcb  = g_phase63_attempt_guest_dcb_post.load(std::memory_order_relaxed);
	const uint32_t host_post  = g_phase63_attempt_host_post.load(std::memory_order_relaxed);
	const uint32_t post_agc   = g_phase63_post_agc.load(std::memory_order_relaxed);
	const uint64_t unreg_tsc  = g_phase63_unreg_tsc.load(std::memory_order_relaxed);
	const uint64_t cycle_id   = g_phase63_cycle_id.load(std::memory_order_relaxed);

	// Priority: blocked > cleared_owners > never_armed
	if (attempt_post >= 1 && guest_dcb >= 1 && guest_real == 0) {
		Phase63SetCause("submit_attempted_blocked");
	} else if (unreg_res >= kPhase63UnregResNeed && post_agc == 0 && guest_real == 0) {
		Phase63SetCause("unreg_cleared_owners");
	} else if (guest_real == 0 && guest_dcb == 0 && post_agc <= 2) {
		// attempts post are host/seed only (or empty); AGC silence
		Phase63SetCause("producer_never_armed");
	} else {
		Phase63SetCause("still_opaque_unreg");
	}

	LOGF("SubmitTrace: phase63 heatmap why=%s cause=%s unreg_owners=%u unreg_res_total=%u "
	     "submit_attempt_post=%u submit_guest_real=%u submit_seed_post=%u "
	     "attempt_guest_dcb=%u attempt_host=%u post_agc=%u cycle_id=%" PRIu64
	     " unreg_tsc=%" PRIu64 " guest_draw=%d\n",
	     why != nullptr ? why : "?", g_phase63_cause, unreg_owners, unreg_res, attempt_post,
	     guest_real, seed_post, guest_dcb, host_post, post_agc, cycle_id, unreg_tsc,
	     g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0);
	fprintf(stderr,
	        "SubmitTrace: phase63 heatmap cause=%s unreg_owners=%u unreg_res_total=%u "
	        "submit_attempt_post=%u submit_guest_real=%u post_agc=%u cycle_id=%" PRIu64 "\n",
	        g_phase63_cause, unreg_owners, unreg_res, attempt_post, guest_real, post_agc,
	        cycle_id);
}

void Phase63Poll() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<uint32_t> ticks {0};
	const uint32_t               t = ticks.fetch_add(1, std::memory_order_relaxed) + 1;
	if (t == 1 || (t % 100) == 0) {
		g_phase63_heatmap_done.store(false, std::memory_order_release);
		Phase63EmitHeatmap(t == 1 ? "poll_first" : "poll_10s");
	}
}

// ---------------------------------------------------------------------------
// Phase 64 — waiters post-Unreg (minimal)
// Priority: stuck_on_cond_main > stuck_on_eq_or_flip > stuck_on_ndjob_label
// ---------------------------------------------------------------------------

void Phase64NoteMainCondWait(uint64_t cond_va) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	(void)g_phase64_main_cond_wait_post.fetch_add(1, std::memory_order_relaxed);
	g_phase64_last_main_wait_cond.store(cond_va, std::memory_order_release);
}

void Phase64NoteMainCondSignal(uint64_t cond_va, bool /*unused_match*/) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	(void)g_phase64_main_cond_signal_post.fetch_add(1, std::memory_order_relaxed);
	const uint64_t last = g_phase64_last_main_wait_cond.load(std::memory_order_acquire);
	if (last != 0 && last == cond_va) {
		(void)g_phase64_main_cond_signal_match.fetch_add(1, std::memory_order_relaxed);
	}
}

static void Phase64SetCause(const char* c) {
	if (c == nullptr) {
		return;
	}
	std::snprintf(g_phase64_cause, sizeof(g_phase64_cause), "%s", c);
}

void Phase64EmitHeatmap(const char* why) {
	if (g_phase64_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t wait_n   = g_phase64_main_cond_wait_post.load(std::memory_order_relaxed);
	const uint32_t sig_n    = g_phase64_main_cond_signal_post.load(std::memory_order_relaxed);
	const uint32_t sig_match = g_phase64_main_cond_signal_match.load(std::memory_order_relaxed);
	const int      flip_stuck =
	    g_phase64_flip_pending_stuck.load(std::memory_order_relaxed) ? 1 : 0;
	const int ndjob_static = g_phase64_ndjob_static.load(std::memory_order_relaxed) ? 1 : 0;
	const uint32_t guest_real =
	    g_phase63_submit_guest_real_post.load(std::memory_order_relaxed);
	const uint32_t fiber_n = g_phase64_fiber_post.load(std::memory_order_relaxed);
	const bool     cond_dom =
	    wait_n >= kPhase64CondWaitNeed && sig_n <= 1 && guest_real == 0;

	// Priority: cond_main > eq_or_flip > ndjob_label
	if (cond_dom) {
		Phase64SetCause("stuck_on_cond_main");
	} else if (flip_stuck != 0 && guest_real == 0 &&
	           (wait_n < kPhase64CondWaitNeed || sig_n > 1)) {
		Phase64SetCause("stuck_on_eq_or_flip");
	} else if (ndjob_static != 0 && flip_stuck == 0 && guest_real == 0 &&
	           (wait_n < kPhase64CondWaitNeed || sig_n > 1)) {
		Phase64SetCause("stuck_on_ndjob_label");
	} else {
		Phase64SetCause("still_opaque_wait");
	}

	LOGF("SubmitTrace: phase64 heatmap why=%s cause=%s main_cond_wait_post=%u "
	     "main_cond_signal_post=%u signal_match=%u flip_pending_stuck=%d ndjob_static=%d "
	     "guest_real_post=%u fiber_post=%u\n",
	     why != nullptr ? why : "?", g_phase64_cause, wait_n, sig_n, sig_match, flip_stuck,
	     ndjob_static, guest_real, fiber_n);
	fprintf(stderr,
	        "SubmitTrace: phase64 heatmap cause=%s main_cond_wait_post=%u "
	        "main_cond_signal_post=%u flip_pending_stuck=%d ndjob_static=%d guest_real_post=%u\n",
	        g_phase64_cause, wait_n, sig_n, flip_stuck, ndjob_static, guest_real);
}

void Phase64Poll() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<uint32_t> ticks {0};
	const uint32_t               t = ticks.fetch_add(1, std::memory_order_relaxed) + 1;
	// ~1 Hz
	if ((t % 10) == 0) {
		Phase64SampleFlipAndNdJob();
	}
	if (t == 1 || (t % 100) == 0) {
		g_phase64_heatmap_done.store(false, std::memory_order_release);
		Phase64EmitHeatmap(t == 1 ? "poll_first" : "poll_10s");
	}
}

// ---------------------------------------------------------------------------
// Phase 65 — who waits / menu entry post-Unreg (read-only)
// Priority: main_misclassified > waits_non_main_dominant > menu_path_never_entered
// ---------------------------------------------------------------------------

void Phase65NoteCondWait(const char* role, const char* name, int tid, uint64_t ra) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const char* r = role != nullptr ? role : "other";
	if (std::strcmp(r, "Main") == 0) {
		(void)g_phase65_wait_main.fetch_add(1, std::memory_order_relaxed);
	} else if (std::strcmp(r, "Mixed") == 0) {
		(void)g_phase65_wait_mixed.fetch_add(1, std::memory_order_relaxed);
	} else if (std::strcmp(r, "Compute") == 0) {
		(void)g_phase65_wait_compute.fetch_add(1, std::memory_order_relaxed);
	} else {
		(void)g_phase65_wait_other.fetch_add(1, std::memory_order_relaxed);
	}
	const bool looks_main =
	    tid == 8 ||
	    (name != nullptr &&
	     (std::strstr(name, "Main") != nullptr || std::strstr(name, "BootCards") != nullptr));
	if (looks_main && std::strcmp(r, "Main") != 0) {
		(void)g_phase65_wait_alias_main.fetch_add(1, std::memory_order_relaxed);
	}
	Phase70NoteGuestRip(ra, r);
	static std::atomic<uint32_t> logs {0};
	if (logs.fetch_add(1, std::memory_order_relaxed) < 96) {
		// fprintf only — LOGF/fmt on guest-stack CondWait → silent 0xC0000409.
		fprintf(stderr, "SubmitTrace: phase65 cond_wait role=%s name=%s tid=%d ra=0x%016" PRIx64 "\n",
		        r, name != nullptr ? name : "?", tid, ra);
		std::fflush(stderr);
	}
}

void Phase65NoteGuestRegbuf2() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	(void)g_phase65_guest_regbuf2.fetch_add(1, std::memory_order_relaxed);
}

void Phase65NoteGuestFlip() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	(void)g_phase65_guest_flip.fetch_add(1, std::memory_order_relaxed);
}

static void Phase65SetCause(const char* c) {
	if (c == nullptr) {
		return;
	}
	std::snprintf(g_phase65_cause, sizeof(g_phase65_cause), "%s", c);
}

void Phase65EmitHeatmap(const char* why) {
	if (g_phase65_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t wait_main    = g_phase65_wait_main.load(std::memory_order_relaxed);
	const uint32_t wait_mixed   = g_phase65_wait_mixed.load(std::memory_order_relaxed);
	const uint32_t wait_compute = g_phase65_wait_compute.load(std::memory_order_relaxed);
	const uint32_t wait_other   = g_phase65_wait_other.load(std::memory_order_relaxed);
	const uint32_t wait_alias   = g_phase65_wait_alias_main.load(std::memory_order_relaxed);
	const int      main_alive   = g_phase65_main_alive.load(std::memory_order_relaxed);
	const uint32_t guest_reg2   = g_phase65_guest_regbuf2.load(std::memory_order_relaxed);
	const uint32_t guest_flip   = g_phase65_guest_flip.load(std::memory_order_relaxed);
	const int      host_flip    = g_phase65_host_flip_active.load(std::memory_order_relaxed);
	const uint32_t guest_real =
	    g_phase63_submit_guest_real_post.load(std::memory_order_relaxed);
	const uint32_t non_main = wait_mixed + wait_compute;

	if (wait_alias >= kPhase65AliasNeed && wait_main == 0 && guest_real == 0) {
		Phase65SetCause("main_misclassified");
	} else if (non_main >= kPhase65NonMainNeed && wait_main == 0 && guest_real == 0) {
		Phase65SetCause("waits_non_main_dominant");
	} else if (guest_reg2 == 0 && guest_flip == 0 && host_flip != 0 && guest_real == 0) {
		Phase65SetCause("menu_path_never_entered");
	} else {
		Phase65SetCause("still_opaque_progress");
	}

	LOGF("SubmitTrace: phase65 heatmap why=%s cause=%s wait_main=%u wait_mixed=%u "
	     "wait_compute=%u wait_other=%u wait_alias_main=%u main_alive=%d guest_regbuf2=%u "
	     "guest_flip_seen=%u guest_real_post=%u host_flip_active=%d\n",
	     why != nullptr ? why : "?", g_phase65_cause, wait_main, wait_mixed, wait_compute,
	     wait_other, wait_alias, main_alive, guest_reg2, guest_flip, guest_real, host_flip);
	fprintf(stderr,
	        "SubmitTrace: phase65 heatmap cause=%s wait_main=%u wait_mixed=%u wait_compute=%u "
	        "wait_other=%u wait_alias_main=%u main_alive=%d guest_regbuf2=%u guest_flip_seen=%u "
	        "guest_real_post=%u host_flip_active=%d\n",
	        g_phase65_cause, wait_main, wait_mixed, wait_compute, wait_other, wait_alias,
	        main_alive, guest_reg2, guest_flip, guest_real, host_flip);
}

void Phase65Poll() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<uint32_t> ticks {0};
	const uint32_t               t = ticks.fetch_add(1, std::memory_order_relaxed) + 1;
	// ~1 Hz: main_alive + host flip progress + menu L0 recycle (opt-in)
	if ((t % 10) == 0) {
		g_phase65_main_alive.store(LibKernel::PthreadMainThreadAlive() ? 1 : 0,
		                           std::memory_order_release);
		Phase65SampleHostFlip();
		Phase66TryMenuRecycle();
	}
	if (t == 1 || (t % 100) == 0) {
		g_phase65_heatmap_done.store(false, std::memory_order_release);
		Phase65EmitHeatmap(t == 1 ? "poll_first" : "poll_10s");
	}
}

static void Phase66SetCause(const char* c) {
	if (c == nullptr) {
		return;
	}
	std::snprintf(g_phase66_cause, sizeof(g_phase66_cause), "%s", c);
}

static bool Phase66MenuRecycleEnabled() {
	static int cached = -1;
	if (cached < 0) {
		const char* e = std::getenv("KYTY_PHASE66_MENU_RECYCLE");
		cached        = (e != nullptr && e[0] == '1') ? 1 : 0;
	}
	return cached == 1;
}

void Phase66EmitHeatmap(const char* why) {
	if (g_phase66_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t guest_reg2 = g_phase65_guest_regbuf2.load(std::memory_order_relaxed);
	const uint32_t guest_flip = g_phase65_guest_flip.load(std::memory_order_relaxed);
	const uint32_t recycle_n  = g_phase66_recycle_n.load(std::memory_order_relaxed);
	const uint32_t streak     = g_phase66_pending_streak.load(std::memory_order_relaxed);
	const int      host_flip  = g_phase65_host_flip_active.load(std::memory_order_relaxed);
	const bool     menu       = guest_reg2 > 0 && (guest_flip > 0 ||
	                                       g_phase37_guest_flip_seen.load(std::memory_order_relaxed));

	if (!Phase66MenuRecycleEnabled()) {
		Phase66SetCause("menu_recycle_off");
	} else if (!menu) {
		Phase66SetCause("menu_not_detected");
	} else if (recycle_n >= 1) {
		Phase66SetCause("menu_recycle_active");
	} else if (streak >= kPhase66IdleNeed) {
		Phase66SetCause("menu_l0_idle");
	} else {
		Phase66SetCause("menu_l0_idle");
	}

	LOGF("SubmitTrace: phase66 heatmap why=%s cause=%s recycle_n=%u pending_streak=%u "
	     "guest_regbuf2=%u guest_flip_seen=%u host_flip_active=%d enabled=%d\n",
	     why != nullptr ? why : "?", g_phase66_cause, recycle_n, streak, guest_reg2, guest_flip,
	     host_flip, Phase66MenuRecycleEnabled() ? 1 : 0);
	fprintf(stderr,
	        "SubmitTrace: phase66 heatmap cause=%s recycle_n=%u pending_streak=%u "
	        "guest_regbuf2=%u guest_flip_seen=%u host_flip_active=%d\n",
	        g_phase66_cause, recycle_n, streak, guest_reg2, guest_flip, host_flip);
}

void Phase66Poll() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<uint32_t> ticks {0};
	const uint32_t               t = ticks.fetch_add(1, std::memory_order_relaxed) + 1;
	if (t == 1 || (t % 100) == 0) {
		g_phase66_heatmap_done.store(false, std::memory_order_release);
		Phase66EmitHeatmap(t == 1 ? "poll_first" : "poll_10s");
	}
}

// ---------------------------------------------------------------------------
// Phase 69 — NdJob predicate dump + soft ready HLE (env-gated)
// Priority: guest_submit_armed > ndjob_hle_no_submit > ndjob_field_static > still_opaque_predicate
// ---------------------------------------------------------------------------

static bool Phase69NdJobReadyEnabled() {
	static int cached = -1;
	if (cached < 0) {
		const char* e = std::getenv("KYTY_PHASE69_NDJOB_READY");
		cached        = (e != nullptr && e[0] == '1') ? 1 : 0;
	}
	return cached == 1;
}

static void Phase69SetCause(const char* c) {
	if (c == nullptr) {
		return;
	}
	std::snprintf(g_phase69_cause, sizeof(g_phase69_cause), "%s", c);
}

static void Phase69TryHleReady(uint64_t ctrl, uint64_t status) {
	if (!Phase69NdJobReadyEnabled()) {
		return;
	}
	if (g_phase69_hle_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	// Soft ready only: gate+8=1 + status pattern seen pre/post Unreg (P41/P53).
	// Do NOT invent *obj job pointers (FAKE_JOB ban) or poke LIST near 0x905f25cd0.
	uint32_t gate = 0;
	Phase41SafeRead(&gate, reinterpret_cast<const void*>(ctrl + 8), sizeof(gate));
	if (gate == 0) {
		gate = 1;
		Phase41SafeWrite(reinterpret_cast<volatile void*>(ctrl + 8), &gate, sizeof(gate));
	}
	if (status >= 0x10000ULL && status < 0x0000800000000000ULL) {
		uint32_t fill[8] = {1u, 1u, 0u, 0u, 1u, 0u, 0u, 0u};
		Phase41SafeWrite(reinterpret_cast<volatile void*>(status), fill, sizeof(fill));
	}
	const uint32_t n = g_phase69_hle_n.fetch_add(1, std::memory_order_relaxed) + 1;
	LOGF("SubmitTrace: phase69 ndjob_hle ready=1 ctrl=0x%016" PRIx64 " status=0x%016" PRIx64
	     " gate=%u n=%u\n",
	     ctrl, status, gate, n);
	fprintf(stderr, "SubmitTrace: phase69 ndjob_hle ready ctrl=0x%016" PRIx64 " gate=%u\n", ctrl,
	        gate);
	Libs::Graphics::Sync::TriggerAgcUserInterrupt();
}

void Phase69SampleNdJob() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	uint64_t ctrl = Phase58QueueBase();
	if (ctrl < 0x10000ULL) {
		ctrl = kPhase69CtrlDef;
	}
	const uint64_t status = kPhase69StatusDef;
	uint64_t       cur[8] {};
	Phase41SafeRead(&cur[0], reinterpret_cast<const void*>(ctrl), sizeof(uint64_t));
	Phase41SafeRead(&cur[1], reinterpret_cast<const void*>(ctrl + 8), sizeof(uint64_t));
	Phase41SafeRead(&cur[2], reinterpret_cast<const void*>(ctrl + 0x10), sizeof(uint64_t));
	Phase41SafeRead(&cur[3], reinterpret_cast<const void*>(ctrl + 0x18), sizeof(uint64_t));
	Phase41SafeRead(&cur[4], reinterpret_cast<const void*>(ctrl + kPhase41Keep1FieldOff),
	                sizeof(uint64_t));
	Phase41SafeRead(&cur[5], reinterpret_cast<const void*>(status), sizeof(uint64_t));
	Phase41SafeRead(&cur[6], reinterpret_cast<const void*>(status + 8), sizeof(uint64_t));
	Phase41SafeRead(&cur[7], reinterpret_cast<const void*>(status + 16), sizeof(uint64_t));

	if (g_phase69_have_prev.load(std::memory_order_acquire)) {
		const bool same = std::memcmp(cur, g_phase69_prev, sizeof(cur)) == 0;
		if (same) {
			const uint32_t st =
			    g_phase69_static_streak.fetch_add(1, std::memory_order_relaxed) + 1;
			if (st >= kPhase69StaticNeed) {
				g_phase69_field_static.store(true, std::memory_order_release);
			}
		} else {
			g_phase69_static_streak.store(0, std::memory_order_relaxed);
		}
	} else {
		g_phase69_have_prev.store(true, std::memory_order_release);
	}
	std::memcpy(g_phase69_prev, cur, sizeof(cur));

	const uint32_t dn = g_phase69_dump_n.fetch_add(1, std::memory_order_relaxed);
	if (dn < 48 || (dn % 10) == 0) {
		LOGF("SubmitTrace: phase69 ndjob_dump n=%u ctrl=0x%016" PRIx64 " *obj=0x%016" PRIx64
		     " gate=0x%016" PRIx64 " u10=0x%016" PRIx64 " u18=0x%016" PRIx64
		     " field=0x%016" PRIx64 " st0=0x%016" PRIx64 " st8=0x%016" PRIx64
		     " st10=0x%016" PRIx64 " static=%d\n",
		     dn, ctrl, cur[0], cur[1], cur[2], cur[3], cur[4], cur[5], cur[6], cur[7],
		     g_phase69_field_static.load(std::memory_order_relaxed) ? 1 : 0);
		if (dn < 24 || (dn % 10) == 0) {
			fprintf(stderr,
			        "SubmitTrace: phase69 ndjob_dump *obj=0x%016" PRIx64 " gate=0x%016" PRIx64
			        " st0=0x%016" PRIx64 " static=%d\n",
			        cur[0], cur[1], cur[5],
			        g_phase69_field_static.load(std::memory_order_relaxed) ? 1 : 0);
		}
	}

	Phase69TryHleReady(ctrl, status);
}

void Phase69EmitHeatmap(const char* why) {
	if (g_phase69_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t guest_real =
	    g_phase63_submit_guest_real_post.load(std::memory_order_relaxed);
	const uint32_t dump_n   = g_phase69_dump_n.load(std::memory_order_relaxed);
	const uint32_t streak   = g_phase69_static_streak.load(std::memory_order_relaxed);
	const int      field_st = g_phase69_field_static.load(std::memory_order_relaxed) ? 1 : 0;
	const uint32_t hle_n    = g_phase69_hle_n.load(std::memory_order_relaxed);
	const int      hle_on   = Phase69NdJobReadyEnabled() ? 1 : 0;

	if (guest_real > 0) {
		Phase69SetCause("guest_submit_armed");
	} else if (hle_n > 0 && guest_real == 0) {
		Phase69SetCause("ndjob_hle_no_submit");
	} else if (field_st != 0 && guest_real == 0) {
		Phase69SetCause("ndjob_field_static");
	} else {
		Phase69SetCause("still_opaque_predicate");
	}

	LOGF("SubmitTrace: phase69 heatmap why=%s cause=%s dump_n=%u static_streak=%u "
	     "field_static=%d hle_n=%u hle_on=%d guest_real_post=%u "
	     "ctrl=*obj=0x%016" PRIx64 " gate=0x%016" PRIx64 " st0=0x%016" PRIx64 "\n",
	     why != nullptr ? why : "?", g_phase69_cause, dump_n, streak, field_st, hle_n, hle_on,
	     guest_real, g_phase69_prev[0], g_phase69_prev[1], g_phase69_prev[5]);
	fprintf(stderr,
	        "SubmitTrace: phase69 heatmap cause=%s dump_n=%u field_static=%d hle_n=%u "
	        "guest_real_post=%u\n",
	        g_phase69_cause, dump_n, field_st, hle_n, guest_real);
}

void Phase69Poll() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<uint32_t> ticks {0};
	const uint32_t               t = ticks.fetch_add(1, std::memory_order_relaxed) + 1;
	// ~1 Hz sample first so heatmap never races dump_n=0 / hle_n=0.
	if ((t % 10) == 0) {
		Phase69SampleNdJob();
	}
	if ((t % 100) == 0) {
		g_phase69_heatmap_done.store(false, std::memory_order_release);
		Phase69EmitHeatmap(t == 100 ? "poll_first" : "poll_10s");
	}
}

// ---------------------------------------------------------------------------
// Phase 70 — Mixed predicate byte (*0x904bb6de8)+0x3eea + soft-HLE
// Priority: guest_submit_armed > field_hle_no_submit > field_still_static
// ---------------------------------------------------------------------------

static bool Phase70NdJobFieldEnabled() {
	static int cached = -1;
	if (cached < 0) {
		const char* e = std::getenv("KYTY_PHASE70_NDJOB_FIELD");
		cached        = (e != nullptr && e[0] == '1') ? 1 : 0;
	}
	return cached == 1;
}

static void Phase70SetCause(const char* c) {
	if (c == nullptr) {
		return;
	}
	std::snprintf(g_phase70_cause, sizeof(g_phase70_cause), "%s", c);
}

void Phase70NoteGuestRip(uint64_t guest_rip, const char* role) {
	if (guest_rip == 0) {
		return;
	}
	g_phase70_last_guest_rip.store(guest_rip, std::memory_order_release);
	const bool near_mixed =
	    guest_rip >= 0x0000000901DE4000ULL && guest_rip < 0x0000000901DE5000ULL;
	if (near_mixed || (role != nullptr && (std::strcmp(role, "Mixed") == 0 ||
	                                       std::strcmp(role, "Compute") == 0))) {
		if (near_mixed) {
			(void)g_phase70_guest_rip_mixed.fetch_add(1, std::memory_order_relaxed);
		}
	}
}

static void Phase70TryHleField(uint64_t base, uint8_t cur) {
	if (!Phase70NdJobFieldEnabled()) {
		return;
	}
	if (g_phase70_hle_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	// Offline: cmp byte [base+0x3eea],0 / sete sil — force non-zero if stuck at 0.
	if (cur != 0) {
		LOGF("SubmitTrace: phase70 ndjob_hle skip byte=0x%02x (already non-zero) base=0x%016"
		     PRIx64 "\n",
		     cur, base);
		return;
	}
	const uint8_t ready = 1;
	Phase41SafeWrite(reinterpret_cast<volatile void*>(base + kPhase70FieldOff), &ready,
	                 sizeof(ready));
	const uint32_t n = g_phase70_hle_n.fetch_add(1, std::memory_order_relaxed) + 1;
	LOGF("SubmitTrace: phase70 ndjob_hle field=1 base=0x%016" PRIx64 " off=0x%x n=%u\n", base,
	     kPhase70FieldOff, n);
	fprintf(stderr, "SubmitTrace: phase70 ndjob_hle field=1 base=0x%016" PRIx64 "\n", base);
	Libs::Graphics::Sync::TriggerAgcUserInterrupt();
}

void Phase70SampleField() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	uint64_t base = 0;
	Phase41SafeRead(&base, reinterpret_cast<const void*>(kPhase70BaseSlot), sizeof(base));
	if (base < 0x10000ULL || base >= 0x0000800000000000ULL) {
		base = g_phase70_resolved_base.load(std::memory_order_acquire);
	}
	uint8_t byte = 0;
	if (base >= 0x10000ULL && base < 0x0000800000000000ULL) {
		Phase41SafeRead(&byte, reinterpret_cast<const void*>(base + kPhase70FieldOff),
		                sizeof(byte));
	}
	uint64_t obj_head = 0;
	Phase41SafeRead(&obj_head, reinterpret_cast<const void*>(kPhase70NdJobCtrlDef),
	                sizeof(obj_head));

	if (g_phase70_have_prev.load(std::memory_order_acquire)) {
		const bool same =
		    (base == g_phase70_prev_base && byte == g_phase70_prev_byte &&
		     obj_head == g_phase70_prev_obj);
		if (same) {
			const uint32_t st =
			    g_phase70_static_streak.fetch_add(1, std::memory_order_relaxed) + 1;
			if (st >= kPhase70StaticNeed) {
				g_phase70_field_static.store(true, std::memory_order_release);
			}
		} else {
			g_phase70_static_streak.store(0, std::memory_order_relaxed);
		}
	} else {
		g_phase70_have_prev.store(true, std::memory_order_release);
	}
	g_phase70_prev_base = base;
	g_phase70_prev_byte = byte;
	g_phase70_prev_obj  = obj_head;

	const uint32_t dn = g_phase70_dump_n.fetch_add(1, std::memory_order_relaxed);
	const uint64_t rip = g_phase70_last_guest_rip.load(std::memory_order_relaxed);
	if (dn < 48 || (dn % 10) == 0) {
		LOGF("SubmitTrace: phase70 field_dump n=%u slot=0x%016" PRIx64 " base=0x%016" PRIx64
		     " byte=0x%02x *obj=0x%016" PRIx64 " static=%d guest_rip=0x%016" PRIx64
		     " rip_mixed=%u\n",
		     dn, kPhase70BaseSlot, base, byte, obj_head,
		     g_phase70_field_static.load(std::memory_order_relaxed) ? 1 : 0, rip,
		     g_phase70_guest_rip_mixed.load(std::memory_order_relaxed));
		if (dn < 24 || (dn % 10) == 0) {
			fprintf(stderr,
			        "SubmitTrace: phase70 field_dump byte=0x%02x base=0x%016" PRIx64
			        " *obj=0x%016" PRIx64 " static=%d\n",
			        byte, base, obj_head,
			        g_phase70_field_static.load(std::memory_order_relaxed) ? 1 : 0);
		}
	}

	if (base >= 0x10000ULL && base < 0x0000800000000000ULL) {
		Phase70TryHleField(base, byte);
	}
}

void Phase70EmitHeatmap(const char* why) {
	if (g_phase70_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t guest_real =
	    g_phase63_submit_guest_real_post.load(std::memory_order_relaxed);
	const uint32_t dump_n   = g_phase70_dump_n.load(std::memory_order_relaxed);
	const uint32_t streak   = g_phase70_static_streak.load(std::memory_order_relaxed);
	const int      field_st = g_phase70_field_static.load(std::memory_order_relaxed) ? 1 : 0;
	const uint32_t hle_n    = g_phase70_hle_n.load(std::memory_order_relaxed);
	const int      hle_on   = Phase70NdJobFieldEnabled() ? 1 : 0;
	const uint32_t rip_m    = g_phase70_guest_rip_mixed.load(std::memory_order_relaxed);
	const uint64_t rip      = g_phase70_last_guest_rip.load(std::memory_order_relaxed);

	if (guest_real > 0) {
		Phase70SetCause("guest_submit_armed");
	} else if (hle_n > 0 && guest_real == 0) {
		Phase70SetCause("field_hle_no_submit");
	} else if (field_st != 0 && guest_real == 0) {
		Phase70SetCause("field_still_static");
	} else {
		Phase70SetCause("still_opaque_predicate");
	}

	LOGF("SubmitTrace: phase70 heatmap why=%s cause=%s dump_n=%u static_streak=%u "
	     "field_static=%d hle_n=%u hle_on=%d guest_real_post=%u byte=0x%02x "
	     "base=0x%016" PRIx64 " *obj=0x%016" PRIx64 " guest_rip=0x%016" PRIx64
	     " rip_mixed=%u\n",
	     why != nullptr ? why : "?", g_phase70_cause, dump_n, streak, field_st, hle_n, hle_on,
	     guest_real, g_phase70_prev_byte, g_phase70_prev_base, g_phase70_prev_obj, rip, rip_m);
	fprintf(stderr,
	        "SubmitTrace: phase70 heatmap cause=%s dump_n=%u field_static=%d hle_n=%u "
	        "byte=0x%02x guest_real_post=%u\n",
	        g_phase70_cause, dump_n, field_st, hle_n, g_phase70_prev_byte, guest_real);
}

void Phase70Poll() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<uint32_t> ticks {0};
	const uint32_t               t = ticks.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((t % 10) == 0) {
		Phase70SampleField();
	}
	if ((t % 100) == 0) {
		g_phase70_heatmap_done.store(false, std::memory_order_release);
		Phase70EmitHeatmap(t == 100 ? "poll_first" : "poll_10s");
	}
}

// ---------------------------------------------------------------------------
// Phase 71 — CTX_FIELD Mixed *(u32*)(0x905f254c0+0x24) + soft-HLE
// Priority: guest_submit_armed > field_hle_no_submit > field_still_static
//           > ctx_base_unresolved > still_opaque_predicate
// ---------------------------------------------------------------------------

static bool Phase71CtxFieldEnabled() {
	static int cached = -1;
	if (cached < 0) {
		const char* e = std::getenv("KYTY_PHASE71_CTX_FIELD");
		cached        = (e != nullptr && e[0] == '1') ? 1 : 0;
	}
	return cached == 1;
}

static void Phase71SetCause(const char* c) {
	if (c == nullptr) {
		return;
	}
	std::snprintf(g_phase71_cause, sizeof(g_phase71_cause), "%s", c);
}

static void Phase71TryHleField(uint32_t cur) {
	if (!Phase71CtxFieldEnabled()) {
		return;
	}
	if (g_phase71_hle_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	if (cur != 0) {
		LOGF("SubmitTrace: phase71 ctx_hle skip f24=0x%x (already non-zero) base=0x%016" PRIx64
		     "\n",
		     cur, kPhase71CtxBase);
		return;
	}
	const uint32_t ready = 1;
	Phase41SafeWrite(reinterpret_cast<volatile void*>(kPhase71CtxBase + kPhase71FieldOff),
	                 &ready, sizeof(ready));
	const uint32_t n = g_phase71_hle_n.fetch_add(1, std::memory_order_relaxed) + 1;
	LOGF("SubmitTrace: phase71 ctx_hle f24=1 base=0x%016" PRIx64 " off=0x%x n=%u\n",
	     kPhase71CtxBase, kPhase71FieldOff, n);
	fprintf(stderr, "SubmitTrace: phase71 ctx_hle f24=1 base=0x%016" PRIx64 "\n",
	        kPhase71CtxBase);
	Libs::Graphics::Sync::TriggerAgcUserInterrupt();
}

void Phase71SampleCtx() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	// Ban check: base must stay distinct from LIST 0x905f25cd0 (Δ=0x110).
	if (kPhase71CtxBase == kPhase56BannedBase || Phase56IsBannedBase(kPhase71CtxBase)) {
		return;
	}

	uint32_t f24   = 0;
	uint32_t f8    = 0;
	uint32_t probe = 0;
	Phase41SafeRead(&probe, reinterpret_cast<const void*>(kPhase71CtxBase), sizeof(probe));
	Phase41SafeRead(&f24, reinterpret_cast<const void*>(kPhase71CtxBase + kPhase71FieldOff),
	                sizeof(f24));
	Phase41SafeRead(&f8, reinterpret_cast<const void*>(kPhase71CtxBase + kPhase71CorrOff),
	                sizeof(f8));
	if (probe != 0 || f24 != 0 || f8 != 0) {
		g_phase71_base_ok.store(true, std::memory_order_release);
	}

	if (g_phase71_have_prev.load(std::memory_order_acquire)) {
		const bool same = (f24 == g_phase71_prev_f24 && f8 == g_phase71_prev_f8);
		if (same) {
			const uint32_t st =
			    g_phase71_static_streak.fetch_add(1, std::memory_order_relaxed) + 1;
			if (st >= kPhase71StaticNeed) {
				g_phase71_field_static.store(true, std::memory_order_release);
			}
		} else {
			g_phase71_static_streak.store(0, std::memory_order_relaxed);
		}
	} else {
		g_phase71_have_prev.store(true, std::memory_order_release);
	}
	g_phase71_prev_f24 = f24;
	g_phase71_prev_f8  = f8;

	const uint32_t dn  = g_phase71_dump_n.fetch_add(1, std::memory_order_relaxed);
	const uint64_t rip = g_phase70_last_guest_rip.load(std::memory_order_relaxed);
	if (dn < 48 || (dn % 10) == 0) {
		LOGF("SubmitTrace: phase71 ctx_dump n=%u base=0x%016" PRIx64 " f24=0x%x f8=0x%x "
		     "probe=0x%x static=%d base_ok=%d guest_rip=0x%016" PRIx64 "\n",
		     dn, kPhase71CtxBase, f24, f8, probe,
		     g_phase71_field_static.load(std::memory_order_relaxed) ? 1 : 0,
		     g_phase71_base_ok.load(std::memory_order_relaxed) ? 1 : 0, rip);
		if (dn < 24 || (dn % 10) == 0) {
			fprintf(stderr,
			        "SubmitTrace: phase71 ctx_dump f24=0x%x f8=0x%x static=%d base_ok=%d\n",
			        f24, f8, g_phase71_field_static.load(std::memory_order_relaxed) ? 1 : 0,
			        g_phase71_base_ok.load(std::memory_order_relaxed) ? 1 : 0);
		}
	}

	if (g_phase71_base_ok.load(std::memory_order_acquire) || dn >= 2) {
		Phase71TryHleField(f24);
		if (g_phase71_hle_n.load(std::memory_order_relaxed) > 0) {
			g_phase71_base_ok.store(true, std::memory_order_release);
		}
	}
}

void Phase71EmitHeatmap(const char* why) {
	if (g_phase71_heatmap_done.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	const uint32_t guest_real =
	    g_phase63_submit_guest_real_post.load(std::memory_order_relaxed);
	const uint32_t dump_n   = g_phase71_dump_n.load(std::memory_order_relaxed);
	const uint32_t streak   = g_phase71_static_streak.load(std::memory_order_relaxed);
	const int      field_st = g_phase71_field_static.load(std::memory_order_relaxed) ? 1 : 0;
	const uint32_t hle_n    = g_phase71_hle_n.load(std::memory_order_relaxed);
	const int      hle_on   = Phase71CtxFieldEnabled() ? 1 : 0;
	const int      base_ok  = g_phase71_base_ok.load(std::memory_order_relaxed) ? 1 : 0;
	const uint64_t rip      = g_phase70_last_guest_rip.load(std::memory_order_relaxed);

	if (guest_real > 0) {
		Phase71SetCause("guest_submit_armed");
	} else if (hle_n > 0 && guest_real == 0) {
		Phase71SetCause("field_hle_no_submit");
	} else if (field_st != 0 && guest_real == 0) {
		Phase71SetCause("field_still_static");
	} else if (!base_ok && dump_n > 0 && guest_real == 0) {
		Phase71SetCause("ctx_base_unresolved");
	} else {
		Phase71SetCause("still_opaque_predicate");
	}

	LOGF("SubmitTrace: phase71 heatmap why=%s cause=%s dump_n=%u static_streak=%u "
	     "field_static=%d hle_n=%u hle_on=%d base_ok=%d guest_real_post=%u f24=0x%x f8=0x%x "
	     "base=0x%016" PRIx64 " guest_rip=0x%016" PRIx64 "\n",
	     why != nullptr ? why : "?", g_phase71_cause, dump_n, streak, field_st, hle_n, hle_on,
	     base_ok, guest_real, g_phase71_prev_f24, g_phase71_prev_f8, kPhase71CtxBase, rip);
	fprintf(stderr,
	        "SubmitTrace: phase71 heatmap cause=%s dump_n=%u field_static=%d hle_n=%u "
	        "base_ok=%d f24=0x%x guest_real_post=%u\n",
	        g_phase71_cause, dump_n, field_st, hle_n, base_ok, g_phase71_prev_f24, guest_real);
}

void Phase71Poll() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<uint32_t> ticks {0};
	const uint32_t               t = ticks.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((t % 10) == 0) {
		Phase71SampleCtx();
	}
	if ((t % 100) == 0) {
		g_phase71_heatmap_done.store(false, std::memory_order_release);
		Phase71EmitHeatmap(t == 100 ? "poll_first" : "poll_10s");
	}
}

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
	Phase53ArmWorkerFiberTraces();
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

// Phase 42/44/50: restore keep[1] +0x20858 bitmap/slots from ENTER snap (full 64 B with
// slot ptrs — not bits alone) + wake status consumers. Never replay live *obj=-1.
// Phase 50: soft rearm only (no job stub/sentinel) + obj_mut / poll hooks.
static void Phase42RearmNdJobEnqueue() {
	static std::atomic<int> rearm_n {0};
	const int               n   = rearm_n.fetch_add(1, std::memory_order_relaxed);
	const uint64_t          obj = g_phase41_keep1_obj.load(std::memory_order_acquire);
	if (obj < 0x10000ULL || obj >= 0x0000800000000000ULL) {
		return;
	}
	const uint64_t head_before = Phase50ReadKeep1Head(obj);
	const uint64_t before_head =
	    *reinterpret_cast<const uint64_t*>(g_phase41_keep1_field_before);
	// Phase 44/50: always re-assert full ENTER field snap (bits + slots) when present.
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
	// Keep gate+8=1; clear poisoned *obj=-1 so consumers can see an empty-but-live head.
	uint64_t head = 0;
	Phase41SafeRead(&head, reinterpret_cast<const void*>(obj), sizeof(head));
	if (head == ~uint64_t {0}) {
		const uint64_t zero = 0;
		Phase41SafeWrite(reinterpret_cast<volatile void*>(obj), &zero, sizeof(zero));
		if (n < 8) {
			LOGF("FlipTrace: phase45 NdJob rearm cleared *obj=-1 → 0 obj=0x%016" PRIx64 "\n",
			     obj);
			fprintf(stderr, "FlipTrace: phase45 NdJob rearm cleared *obj=-1\n");
		}
	}
	uint32_t gate = 1;
	Phase41SafeWrite(reinterpret_cast<volatile void*>(obj + 8), &gate, sizeof(gate));
	// Phase 52: do NOT poke +0x10/+0x18 (was corrupting GPU/USER VAs → …0001).
	if (!g_phase52_rearm_clean_logged.exchange(true, std::memory_order_acq_rel)) {
		uint64_t q10 = 0;
		uint64_t q18 = 0;
		Phase41SafeRead(&q10, reinterpret_cast<const void*>(obj + 0x10), sizeof(q10));
		Phase41SafeRead(&q18, reinterpret_cast<const void*>(obj + 0x18), sizeof(q18));
		LOGF("FlipTrace: phase52 rearm_clean +0x10=0x%016" PRIx64 " +0x18=0x%016" PRIx64
		     " (no pending poke)\n",
		     q10, q18);
		fprintf(stderr, "FlipTrace: phase52 rearm_clean +0x10=0x%016" PRIx64 "\n", q10);
	}
	// Extra EQ / status wakes so consumers leave WaitEqueue toward RegisterBuffers2.
	const size_t woken = LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
	(void)LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
	const uint64_t head_after = Phase50ReadKeep1Head(obj);
	Phase50NoteObjMutation("rearm", head_before, head_after);
	if (n < 16 || (n % 40) == 0) {
		const uint64_t tsc = LibKernel::KernelGetProcessTimeCounter();
		LOGF("FlipTrace: phase50 rearm_soft n=%d *obj=0x%016" PRIx64 " gate=1 woken=%zu "
		     "field_snap=0x%016" PRIx64 " tsc=%" PRIu64 " tid=%d\n",
		     n, head_after, woken, before_head, tsc, Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "FlipTrace: phase50 rearm_soft n=%d *obj=0x%016" PRIx64 " woken=%zu\n", n,
		        head_after, woken);
	}
	Phase50NoteWake("rearm_soft", woken);
	if ((n % 5) == 0) {
		Phase50PollKeep1Obj("rearm");
	}
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
	if ((n % 8) == 0) {
		Phase50PollKeep1Obj("handoff");
	}
	if (g_phase45_timeout_submit_done.load(std::memory_order_acquire) &&
	    !Phase47GuestDrawSeen()) {
		Phase47PostSeedWake("menu-handoff");
	}
	Phase44CheckNdJobDcb();
	{
		const uint64_t dcb =
		    Libs::Graphics::Sync::SubmitTrace::submit_dcb.load(std::memory_order_relaxed);
		Phase45NoteSubmitDcb(dcb);
	}

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
	// Sustain flips while buffers are linked — keep going after Reg2 (Phase 47 soft-idle).
	if (g_phase42_reregister_ok.load(std::memory_order_acquire) && n >= 6) {
		if (!g_phase43_menu_frames_ok.load(std::memory_order_acquire)) {
			Phase43SeedNdJobDcb();
			Phase43SustainMenuFlips();
			Phase43UpdateMenuFramesOk();
		} else {
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

void PhaseDescribeFaultRip(uint64_t rip) {
	char line[320];
	auto check = [&](const char* name, const uint8_t* base, uint64_t size) {
		if (base == nullptr || size == 0) {
			return false;
		}
		const uint64_t b = reinterpret_cast<uint64_t>(base);
		if (rip < b || rip >= b + size) {
			return false;
		}
		std::snprintf(line, sizeof(line), "poison-rip in %s base=0x%016" PRIx64 " off=0x%llx",
		              name, b, static_cast<unsigned long long>(rip - b));
		Common::LogFatalToFile(line);
		return true;
	};
	bool hit = false;
	hit |= check("phase41_keep1_thunk", g_phase41_keep1_thunk, 256);
	hit |= check("phase41_keep1_live", g_phase41_keep1_live, 64);
	hit |= check("phase41_call_thunk", g_phase41_call_thunk, 64);
	hit |= check("phase53_worker_thunk", g_phase53_worker_thunk, 256);
	hit |= check("phase53_worker_live", g_phase53_worker_live, 64);
	hit |= check("phase53_fiber_thunk", g_phase53_fiber_thunk, 256);
	hit |= check("phase53_fiber_live", g_phase53_fiber_live, 64);
	hit |= check("phase55_mixed_thunk", g_phase55_mixed_thunk, 256);
	hit |= check("phase55_mixed_live", g_phase55_mixed_live, 64);
	for (uint32_t i = 0; i < 5; i++) {
		char name[40];
		std::snprintf(name, sizeof(name), "phase40_entry_thunk[%u]", i);
		hit |= check(name, g_phase40_slots[i].entry_thunk, 256);
		std::snprintf(name, sizeof(name), "phase40_live_entry[%u]", i);
		hit |= check(name, g_phase40_slots[i].live_entry, 128);
	}
	std::snprintf(line, sizeof(line),
	              "poison-ctx keep1_obj=0x%016" PRIx64 " keep1_rsi=0x%016" PRIx64
	              " keep1_hits=%" PRIu64 " status_rdi=0x%016" PRIx64,
	              g_phase41_keep1_obj.load(std::memory_order_relaxed),
	              g_phase41_keep1_rsi.load(std::memory_order_relaxed),
	              g_phase41_keep1_hits.load(std::memory_order_relaxed),
	              g_phase41_status_rdi.load(std::memory_order_relaxed));
	Common::LogFatalToFile(line);
	if (!hit) {
		Common::LogFatalToFile("poison-rip not in known Phase trampolines");
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	MEMORY_BASIC_INFORMATION mbi {};
	if (VirtualQuery(reinterpret_cast<const void*>(rip), &mbi, sizeof(mbi)) != 0 &&
	    mbi.State == MEM_COMMIT && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0) {
		const auto* bytes = reinterpret_cast<const uint8_t*>(rip >= 16 ? rip - 16 : rip);
		const uint32_t skip = rip >= 16 ? 0u : static_cast<uint32_t>(16 - rip);
		int         pos   = std::snprintf(line, sizeof(line), "poison-bytes[-16]:");
		for (uint32_t i = skip; i < 32 && pos > 0 && static_cast<size_t>(pos) + 3 < sizeof(line);
		     i++) {
			pos += std::snprintf(line + pos, sizeof(line) - static_cast<size_t>(pos), " %02x",
			                     bytes[i]);
		}
		Common::LogFatalToFile(line);
		std::snprintf(line, sizeof(line),
		              "poison-mbi base=0x%016" PRIx64 " size=0x%llx prot=0x%08" PRIx32
		              " type=0x%08" PRIx32,
		              reinterpret_cast<uint64_t>(mbi.BaseAddress),
		              static_cast<unsigned long long>(mbi.RegionSize),
		              static_cast<uint32_t>(mbi.Protect), static_cast<uint32_t>(mbi.Type));
		Common::LogFatalToFile(line);
		{
			char abort_detail[160] {};
			if (Loader::X64InstructionEmulator::DescribeGuestAbortTrap(rip, abort_detail,
			                                                           sizeof(abort_detail))) {
				Common::LogFatalToFile(abort_detail);
			}
		}
		HMODULE owner = nullptr;
		if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                       reinterpret_cast<LPCSTR>(rip), &owner) != 0 &&
		    owner != nullptr) {
			char module_name[MAX_PATH] = {};
			if (GetModuleFileNameA(owner, module_name, MAX_PATH) != 0) {
				std::snprintf(line, sizeof(line),
				              "poison-module %s base=0x%016" PRIx64 " rva=0x%08" PRIx64, module_name,
				              reinterpret_cast<uint64_t>(owner),
				              rip - reinterpret_cast<uint64_t>(owner));
				Common::LogFatalToFile(line);
			}
		} else {
			Common::LogFatalToFile("poison-module (none / not a PE module handle)");
		}
	}
#endif
}

// Overload used from runtimeLinker with full register context.
void PhaseDescribeFaultRip(uint64_t rip, uint64_t rsp) {
	char line[160];
	std::snprintf(line, sizeof(line), "poison-rsp align rsp=0x%016" PRIx64 " rsp&0xf=%u%s", rsp,
	              static_cast<unsigned>(rsp & 0xfu),
	              (rsp & 0xfu) == 8u
	                  ? " (MISALIGNED for movaps — likely SSE #GP reported as AV/-1)"
	                  : "");
	Common::LogFatalToFile(line);
	PhaseDescribeFaultRip(rip);
}

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

static void AfterGpuQueueEmptyHostTick(); // defined after FlipStats

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

// Upstream Kyty FlipStatus layout (post Vulkan-Hpp / graphics refactor).
// Sentinels flipArg/currentBuffer stay -1 for Phase 30–36 hold-gate / pending0 logic.
struct VideoOutFlipStatus {
	uint64_t count                    = 0;
	uint64_t processTime              = 0;
	uint64_t reserved0                = 0;
	int64_t  flipArg                  = -1;
	uint64_t reserved1                = 0;
	uint64_t processTimeCounter       = 0;
	int32_t  gcQueueNum               = 0;
	int32_t  flipPendingNum           = 0;
	int32_t  currentBuffer            = -1;
	uint32_t reserved2                = 0;
	uint64_t submitProcessTimeCounter = 0;
	uint64_t reserved3[7]             = {};
};

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
	uint32_t                       width         = 0;
	uint32_t                       height        = 0;
	uint32_t                       bytes_per_element = 0;
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

uint64_t FlipStatsSubmitCpu() {
	return FlipStats::submit_cpu.load(std::memory_order_relaxed);
}
uint64_t FlipStatsSubmitGpu() {
	return FlipStats::submit_gpu.load(std::memory_order_relaxed);
}
uint64_t FlipStatsPresented() {
	return FlipStats::presented.load(std::memory_order_relaxed);
}
uint64_t FlipStatsRegisterBuffers() {
	return FlipStats::register_buffers.load(std::memory_order_relaxed);
}

static void AfterGpuQueueEmptyHostTick() {
	AfterPending0HostTick();
	static std::atomic<uint64_t> last_submit_gpu {0};
	static std::atomic<uint32_t> empty_streak {0};
	static std::atomic<uint32_t> nosubmit_streak {0};
	const uint64_t               sg = FlipStats::submit_gpu.load(std::memory_order_relaxed);
	const uint64_t               sc = FlipStats::submit_cpu.load(std::memory_order_relaxed);
	const uint64_t               prev = last_submit_gpu.load(std::memory_order_relaxed);
	if (sg != prev) {
		last_submit_gpu.store(sg, std::memory_order_relaxed);
		empty_streak.store(0, std::memory_order_relaxed);
		nosubmit_streak.store(0, std::memory_order_relaxed);
		return;
	}

	// PPSA21564: after Reg2, Main blocks on hkSemaphore forever with submit_*=0. The legacy
	// stall wake only runs once submit_gpu has been seen — extend it for the never-submit case.
	const uint64_t reg_n = FlipStats::register_buffers.load(std::memory_order_relaxed);
	if (reg_n > 0 && sg == 0 && sc == 0) {
		const uint32_t n = nosubmit_streak.fetch_add(1, std::memory_order_relaxed) + 1;
		// Cause (PPSA21564): Main's hkSemaphore IS signaled (Havok + host hit_main), but Draw*
		// threads starve on job-queue conds that WakeSubmission previously ignored.
		if (n >= 8 && (n % 8u) == 0u && n <= 1024u) {
			// Cond wakes only — never ForceSignal RenderContextDrawSema / hk max=1
			// (ceiling poison → guest ASSERT Semaphore.cpp:86 on DrawThread).
			const auto woken = LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
			const int  ue =
			    LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
			const size_t hk_main = LibKernel::Semaphore::SignalMainHkSemaphore(1);
			if (n <= 64 || (n % 32u) == 0u) {
				char breadcrumb[320];
				std::snprintf(breadcrumb, sizeof(breadcrumb),
				              "SubmitTrace: reg2_nosubmit stall wake n=%u woken=%zu ue=%d "
				              "hk_main=%zu cause=draw_sema_ceiling_poison_fixed",
				              n, woken, ue, hk_main);
				Common::EmergencyLogRaw(breadcrumb);
				Common::LogFatalToFile(breadcrumb);
				fprintf(stderr, "%s\n", breadcrumb);
				LOGF("%s\n", breadcrumb);
			}

			if (n == 64 || n == 192 || n == 384) {
				LibKernel::PthreadDumpAllGuestThreads("reg2_nosubmit_diag");
				char cause[160];
				std::snprintf(cause, sizeof(cause),
				              "SubmitTrace: cause_probe n=%u submit_cpu=0 submit_gpu=0 "
				              "(no ForceSignal DrawSema)",
				              n);
				Common::EmergencyLogRaw(cause);
				Common::LogFatalToFile(cause);
				fprintf(stderr, "%s\n", cause);
			}
		}
		return;
	}

	if (sg == 0) {
		return;
	}
	// PPSA21564: after first organic GPU flip, nudge DrawWait soft-HLE once more for
	// presented>=2 (CondWait may not re-enter promptly).
	{
		const uint64_t presented = FlipStats::presented.load(std::memory_order_relaxed);
		if (presented == 1) {
			static std::atomic<uint32_t> post_n {0};
			const uint32_t               pn = post_n.fetch_add(1, std::memory_order_relaxed) + 1;
			if ((pn % 16u) == 0u && pn <= 256u) {
				(void)LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
				Libs::Graphics::Gen5::TrySoftHleOrganicGuestFlipAfterDrawWait();
			}
		}
	}
	const uint32_t n = empty_streak.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((n % 64u) != 0u || n > 4096u) {
		return;
	}
	const auto woken = LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
	const int  ue =
	    LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
	LOGF("SubmitTrace: queue_empty stall wake n=%u submit_gpu=%" PRIu64 " woken=%zu ue=%d\n", n, sg,
	     woken, ue);
	fprintf(stderr,
	        "SubmitTrace: queue_empty stall wake n=%u submit_gpu=%" PRIu64 " woken=%zu ue=%d\n", n,
	        sg, woken, ue);
	// Once after CFG soft-continue: dump who is waiting (producer freeze diagnosis).
	if (Common::CfgSoftContinueSeen()) {
		static std::atomic<bool> dumped {false};
		bool                     expect = false;
		if (dumped.compare_exchange_strong(expect, true, std::memory_order_relaxed)) {
			fprintf(stderr,
			        "SubmitTrace: cfg_stall_snapshot submit_gpu=%" PRIu64 " empty_streak=%u\n", sg,
			        n);
			LibKernel::PthreadDumpAllGuestThreads("cfg_stall_after_softcontinue");
			const int      uid = Common::CfgSoftContinueFaultUniqueId();
			const uint32_t htid = Common::CfgSoftContinueFaultHostTid();
			// Prefer host_tid of the CFG fault thread (Main). Do NOT default uid→8
			// (TaskGraph) when unique_id was unavailable in the VEH.
			LibKernel::PthreadSnapshotGuestThread(uid, htid, "cfg_stall_after_softcontinue");
			if (htid != 0) {
				LibKernel::PthreadSnapshotGuestThread(-1, htid, "cfg_stall_fault_host");
			}
		}
	}
}

class FlipQueue {
public:
	FlipQueue() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	~FlipQueue() { KYTY_NOT_IMPLEMENTED; }
	KYTY_CLASS_NO_COPY(FlipQueue);

	bool     Reserve(VideoOutConfig& cfg, int index, int64_t flip_arg, FlipRequestSource source,
	                 uint64_t& request_id);
	void     Prepare(uint64_t request_id, Graphics::CommandBuffer& buffer);
	uint64_t PrepareNextCpu(Graphics::CommandBuffer& buffer);
	void     Complete(uint64_t request_id);
	void     WaitForSubmitSlot();
	bool     Flip(uint32_t micros);
	bool     HasPending(VideoOutConfig& cfg, int start_index, int count);
	void     GetFlipStatus(VideoOutConfig& cfg, VideoOutFlipStatus& out);
	void     ClearPendingPhase32(VideoOutConfig* cfg);
	void     Wait(VideoOutConfig& cfg, int index);

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

	FlipQueue& GetFlipQueue() { return m_flip_queue; }

	void VblankBegin();
	void VblankEnd();

private:
	Common::Mutex  m_mutex;
	VideoOutConfig m_video_out_ctx[VIDEO_OUT_NUM_MAX];
	FlipQueue      m_flip_queue;
};

static VideoOutContext* g_video_out_context = nullptr;

void Phase64SampleFlipAndNdJob() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	int32_t  pending  = -1;
	int32_t  curbuf   = -2;
	uint64_t fcount   = 0;
	bool     got_flip = false;
	if (g_video_out_context != nullptr) {
		for (int h = 1; h < 8; ++h) {
			if (!g_video_out_context->IsOpened(h)) {
				continue;
			}
			auto* ctx = g_video_out_context->Get(h);
			if (ctx == nullptr) {
				continue;
			}
			Common::LockGuard lock(ctx->mutex);
			pending  = ctx->flip_status.flipPendingNum;
			curbuf   = ctx->flip_status.currentBuffer;
			fcount   = ctx->flip_status.count;
			got_flip = true;
			break;
		}
	}
	if (got_flip) {
		if (g_phase64_have_flip_prev.load(std::memory_order_acquire)) {
			const bool same =
			    (pending == g_phase64_prev_pending && curbuf == g_phase64_prev_curbuf &&
			     fcount == g_phase64_prev_flip_count);
			if (same && pending > 0) {
				const uint32_t st =
				    g_phase64_flip_stuck_streak.fetch_add(1, std::memory_order_relaxed) + 1;
				if (st >= kPhase64StuckY) {
					g_phase64_flip_pending_stuck.store(true, std::memory_order_release);
				}
			} else {
				g_phase64_flip_stuck_streak.store(0, std::memory_order_relaxed);
			}
		} else {
			g_phase64_have_flip_prev.store(true, std::memory_order_release);
		}
		g_phase64_prev_pending    = pending;
		g_phase64_prev_curbuf     = curbuf;
		g_phase64_prev_flip_count = fcount;
	}

	uint64_t ctrl = Phase58QueueBase();
	if (ctrl < 0x10000ULL) {
		ctrl = kPhase64NdJobCtrlDef;
	}
	const uint64_t status = kPhase59StatusDefault;
	uint64_t       cur[5] {};
	Phase41SafeRead(&cur[0], reinterpret_cast<const void*>(ctrl), sizeof(uint64_t));
	Phase41SafeRead(&cur[1], reinterpret_cast<const void*>(ctrl + 8), sizeof(uint64_t));
	Phase41SafeRead(&cur[2], reinterpret_cast<const void*>(ctrl + kPhase41Keep1FieldOff),
	                sizeof(uint64_t));
	Phase41SafeRead(&cur[3], reinterpret_cast<const void*>(status), sizeof(uint64_t));
	Phase41SafeRead(&cur[4], reinterpret_cast<const void*>(status + 8), sizeof(uint64_t));

	if (g_phase64_have_ndjob_prev.load(std::memory_order_acquire)) {
		const bool same = std::memcmp(cur, g_phase64_prev_ndjob, sizeof(cur)) == 0;
		if (same) {
			const uint32_t st =
			    g_phase64_ndjob_static_streak.fetch_add(1, std::memory_order_relaxed) + 1;
			const uint32_t fiber = g_phase64_fiber_post.load(std::memory_order_relaxed);
			if (st >= kPhase64StuckY && fiber > 0) {
				g_phase64_ndjob_static.store(true, std::memory_order_release);
			}
		} else {
			g_phase64_ndjob_static_streak.store(0, std::memory_order_relaxed);
		}
	} else {
		g_phase64_have_ndjob_prev.store(true, std::memory_order_release);
	}
	std::memcpy(g_phase64_prev_ndjob, cur, sizeof(cur));

	static std::atomic<uint32_t> probe_logs {0};
	const uint32_t               pn = probe_logs.fetch_add(1, std::memory_order_relaxed);
	if (pn < 24 || (pn % 16) == 0) {
		LOGF("SubmitTrace: phase64 waiter_probe pending=%d curbuf=%d flip_count=%" PRIu64
		     " flip_stuck=%d ndjob_static=%d fiber_post=%u ctrl=0x%016" PRIx64 "\n",
		     pending, curbuf, fcount,
		     g_phase64_flip_pending_stuck.load(std::memory_order_relaxed) ? 1 : 0,
		     g_phase64_ndjob_static.load(std::memory_order_relaxed) ? 1 : 0,
		     g_phase64_fiber_post.load(std::memory_order_relaxed), ctrl);
	}
}

void Phase65SampleHostFlip() {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire) || g_video_out_context == nullptr) {
		return;
	}
	uint64_t fcount   = 0;
	bool     got_flip = false;
	for (int h = 1; h < 8; ++h) {
		if (!g_video_out_context->IsOpened(h)) {
			continue;
		}
		auto* ctx = g_video_out_context->Get(h);
		if (ctx == nullptr) {
			continue;
		}
		Common::LockGuard lock(ctx->mutex);
		fcount   = ctx->flip_status.count;
		got_flip = true;
		break;
	}
	if (!got_flip) {
		return;
	}
	if (g_phase65_have_flip_prev.load(std::memory_order_acquire)) {
		if (fcount > g_phase65_prev_flip_count) {
			g_phase65_host_flip_active.store(1, std::memory_order_release);
		}
	} else {
		g_phase65_have_flip_prev.store(true, std::memory_order_release);
	}
	g_phase65_prev_flip_count = fcount;
}

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
	if (event->event.ident == VIDEO_OUT_EVENT_FLIP) {
		const auto n = g_flip_eq_consumed.fetch_add(1, std::memory_order_acq_rel) + 1;
		if (g_pending0_need_flip_consumed.load(std::memory_order_relaxed) != 0) {
			LOGF("FlipTrace: Flip EQ consumed by waiter count=%" PRIu64 " need=%" PRIu64 "\n", n,
			     g_pending0_need_flip_consumed.load(std::memory_order_relaxed));
			fprintf(stderr, "FlipTrace: Flip EQ consumed count=%" PRIu64 "\n", n);
		}
	}
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

static void Phase39SignalVblankAndLabels() {
	if (g_video_out_context == nullptr) {
		return;
	}
	static std::atomic<int> once {0};
	const int               n = once.fetch_add(1, std::memory_order_relaxed);
	size_t                  vblank_eqs = 0;
	size_t                  labels_cleared = 0;
	for (int h = 1; h < 8; ++h) {
		if (!g_video_out_context->IsOpened(h)) {
			continue;
		}
		auto* ctx = g_video_out_context->Get(h);
		if (ctx == nullptr) {
			continue;
		}
		Common::LockGuard lock(ctx->mutex);
		// shad-like: VO labels released so WaitRegMem / guest polls can proceed after Unregister.
		for (int i = 0; i < VIDEO_OUT_BUFFER_NUM_MAX; ++i) {
			if (ctx->buffer_labels[i] != 0) {
				ctx->buffer_labels[i] = 0;
				++labels_cleared;
			}
		}
		for (auto* vb_eq: ctx->vblank_eqs) {
			if (vb_eq == nullptr) {
				continue;
			}
			(void)EventQueue::KernelTriggerEvent(
			    vb_eq, VIDEO_OUT_EVENT_VBLANK, EventQueue::KERNEL_EVFILT_VIDEO_OUT,
			    reinterpret_cast<void*>(ctx->vblank_status.count));
			++vblank_eqs;
		}
		for (auto* vb_eq: ctx->pre_vblank_eqs) {
			if (vb_eq == nullptr) {
				continue;
			}
			(void)EventQueue::KernelTriggerEvent(
			    vb_eq, VIDEO_OUT_EVENT_PRE_VBLANK_START, EventQueue::KERNEL_EVFILT_VIDEO_OUT,
			    reinterpret_cast<void*>(ctx->pre_vblank_status.count));
			++vblank_eqs;
		}
	}
	if (n < 8 || (n % 50) == 0) {
		LOGF("FlipTrace: phase39 signal vblank_eqs=%zu labels_cleared=%zu n=%d "
		     "(capacity=%zu)\n",
		     vblank_eqs, labels_cleared, n, VIDEO_OUT_FLIP_QUEUE_CAPACITY);
		fprintf(stderr, "FlipTrace: phase39 signal vblank_eqs=%zu labels=%zu\n", vblank_eqs,
		        labels_cleared);
	}
}

static void Phase38NudgeBootWorkersOnce() {
	// Phase 41: unlimited wakes after Unregister — budgeted AfterFlip exhausted by n≈4.
	const size_t woken =
	    g_phase39_post_unreg.load(std::memory_order_acquire)
	        ? LibKernel::PthreadWakeSubmissionCondWaitersUnlimited()
	        : LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
	const int ue = LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
	size_t    flip_eqs = 0;
	if (g_video_out_context != nullptr) {
		for (int h = 1; h < 8; ++h) {
			if (!g_video_out_context->IsOpened(h)) {
				continue;
			}
			auto* ctx = g_video_out_context->Get(h);
			if (ctx == nullptr) {
				continue;
			}
			Common::LockGuard lock(ctx->mutex);
			for (auto* flip_eq: ctx->flip_eqs) {
				if (flip_eq == nullptr) {
					continue;
				}
				(void)EventQueue::KernelTriggerEvent(
				    flip_eq, VIDEO_OUT_EVENT_FLIP, EventQueue::KERNEL_EVFILT_VIDEO_OUT,
				    reinterpret_cast<void*>(static_cast<intptr_t>(0)));
				++flip_eqs;
			}
		}
	}
	Phase39SignalVblankAndLabels();
	if (g_phase39_post_unreg.load(std::memory_order_acquire)) {
		Phase41MenuHandoffAttempt();
	}
	static std::atomic<int> nudge_log {0};
	const int               n = nudge_log.fetch_add(1, std::memory_order_relaxed);
	if (n < 8 || (n % 50) == 0) {
		LOGF("SubmitTrace: phase38 boot nudge n=%d woken=%zu ue=%d flip_eqs=%zu\n", n, woken, ue,
		     flip_eqs);
		fprintf(stderr, "SubmitTrace: phase38 boot nudge n=%d woken=%zu flip_eqs=%zu\n", n, woken,
		        flip_eqs);
	}
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

static bool IsFlipDue(VideoOutConfig& cfg) {
	Common::LockGuard lock(cfg.mutex);

	const int interval = cfg.flip_rate + 1;

	return interval <= 1 || (cfg.vblank_status.count % static_cast<uint64_t>(interval)) == 0;
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
                              FlipRequestSource source, uint64_t& request_id) {
	EXIT_IF(g_video_out_context == nullptr);

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
	const bool        special = IsSpecialBufferIndex(index);
	const void*       guest_buf =
	    special ? nullptr : video_out->buffers[index].buffer;
	auto* vk = special ? nullptr : video_out->buffers[index].buffer_vulkan;
	if (video_out->closing ||
	    (!special && (video_out->unregistering[index] || vk == nullptr))) {
		static std::atomic<uint32_t> bad_n {0};
		if (bad_n.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("FlipTrace: phase49 submit reject INVALID_INDEX handle=%d index=%d closing=%d "
			     "unreg=%d buffer=%p vk=%p src=%s\n",
			     handle, index, video_out->closing ? 1 : 0,
			     (!special && video_out->unregistering[index]) ? 1 : 0, guest_buf,
			     static_cast<void*>(vk),
			     source == FlipRequestSource::GpuEop ? "gpu" : "cpu");
			fprintf(stderr, "FlipTrace: phase49 submit reject INVALID_INDEX index=%d vk=%p\n",
			        index, static_cast<void*>(vk));
			char breadcrumb[192];
			std::snprintf(breadcrumb, sizeof(breadcrumb),
			              "FlipTrace: ReserveFlip INVALID_INDEX index=%d vk=%p closing=%d src=%s",
			              index, static_cast<void*>(vk), video_out->closing ? 1 : 0,
			              source == FlipRequestSource::GpuEop ? "gpu" : "cpu");
			Common::EmergencyLogRaw(breadcrumb);
			Common::LogFatalToFile(breadcrumb);
		}
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}
	if (!g_video_out_context->GetFlipQueue().Reserve(*video_out, index, flip_arg, source,
	                                                 request_id)) {
		const uint32_t full_n =
		    g_phase49_submit_full_n.fetch_add(1, std::memory_order_relaxed);
		if (full_n < 24) {
			LOGF("FlipTrace: phase49 submit QUEUE_FULL handle=%d index=%d mode=%d arg=%" PRId64
			     " buffer=%p vk=%p flip_rate=%d src=%s full_n=%u\n",
			     handle, index, flip_mode, flip_arg, guest_buf, static_cast<void*>(vk),
			     video_out->flip_rate, source == FlipRequestSource::GpuEop ? "gpu" : "cpu",
			     full_n + 1);
			fprintf(stderr, "FlipTrace: phase49 submit QUEUE_FULL index=%d src=%s\n", index,
			        source == FlipRequestSource::GpuEop ? "gpu" : "cpu");
		}
		return VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
	}
	g_phase49_submit_accepted.store(true, std::memory_order_release);
	const uint32_t ok_n = g_phase49_submit_ok_n.fetch_add(1, std::memory_order_relaxed);
	if (ok_n < 32 || (source == FlipRequestSource::GpuEop && ok_n < 64)) {
		LOGF("FlipTrace: phase49 submit OK handle=%d index=%d mode=%d arg=%" PRId64
		     " id=%" PRIu64 " buffer=%p vk=%p flip_rate=%d pending=%d src=%s\n",
		     handle, index, flip_mode, flip_arg, request_id, guest_buf, static_cast<void*>(vk),
		     video_out->flip_rate, video_out->flip_status.flipPendingNum,
		     source == FlipRequestSource::GpuEop ? "gpu" : "cpu");
		if (ok_n < 8 || source == FlipRequestSource::GpuEop) {
			fprintf(stderr, "FlipTrace: phase49 submit OK index=%d src=%s id=%" PRIu64 "\n",
			        index, source == FlipRequestSource::GpuEop ? "gpu" : "cpu", request_id);
		}
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
	if (!Graphics::DecodeVideoOutPixelFormat(attribute.pixel_format, pixel_format)) {
		EXIT("unsupported video-out pixel format: 0x%016" PRIx64 "\n", attribute.pixel_format);
	}
	const auto tile_mode =
	    Graphics::Prospero::GpuEnumValue(Graphics::Prospero::TileMode::kRenderTarget);
	const auto pitch =
	    Graphics::TileGetTexturePitch(pixel_format.guest_format, attribute.width, 1, tile_mode);
	Graphics::TileSizeAlign total {};
	Graphics::TileGetTextureTotalSize(pixel_format.guest_format, attribute.width, attribute.height,
	                                  1, pitch, 1, tile_mode, false, total);
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
	if (m_flip_queue.HasPending(config, VIDEO_OUT_BUFFER_INDEX_BLACK,
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
		Graphics::GetRenderContext().GetTextureCache().UnregisterVideoOutSurfaces(images);
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
					Graphics::GetRenderContext().GetTextureCache().RefreshVideoOut(*ret.image,
					                                                               render_target);
					return ret;
				}
			}
		}
	}
	return ret;
}

bool FlipQueue::Reserve(VideoOutConfig& cfg, int index, int64_t flip_arg, FlipRequestSource source,
                        uint64_t& request_id) {
	Common::LockGuard lock(m_mutex);

	if (m_requests.size() + m_cpu_requests.size() >= VIDEO_OUT_FLIP_QUEUE_CAPACITY) {
		return false;
	}
	auto& pending = source == FlipRequestSource::GpuEop ? m_requests : m_cpu_requests;

	Request r {};
	r.id         = m_next_request_id++;
	r.cfg        = &cfg;
	r.index      = index;
	r.flip_arg   = flip_arg;
	r.submit_ptc = LibKernel::KernelGetProcessTimeCounter();
	r.source     = source;
	r.state      = RequestState::Reserved;

	pending.push_back(r);
	request_id = r.id;

	cfg.flip_status.flipPendingNum = static_cast<int>(m_requests.size() + m_cpu_requests.size());
	cfg.flip_status.submitProcessTimeCounter = r.submit_ptc;
	const bool gpu_eop = (source == FlipRequestSource::GpuEop);
	if (gpu_eop) {
		cfg.flip_status.gcQueueNum++;
		FlipStats::reserve_gpu.fetch_add(1, std::memory_order_relaxed);
	} else {
		FlipStats::reserve_cpu.fetch_add(1, std::memory_order_relaxed);
	}
	LOGF("FlipTrace: Reserve id=%" PRIu64 " index=%d gpu_eop=%d pending=%d cpu_q=%zu gpu_q=%zu\n",
	     r.id, index, gpu_eop ? 1 : 0, cfg.flip_status.flipPendingNum, m_cpu_requests.size(),
	     m_requests.size());
	FlipStats::Log(gpu_eop ? "reserve_gpu" : "reserve_cpu");

	return true;
}

static void Phase48ClassifyPresentContent(VideoOutConfig* cfg, int index,
                                          Graphics::VideoOutVulkanImage* source) {
	if (cfg == nullptr || source == nullptr || index < 0 || index >= 16) {
		return;
	}
	const auto& info   = cfg->buffers[index];
	const auto  format = source->format;
	const bool  is_float =
	    format == vk::Format::eR16G16B16A16Sfloat || format == vk::Format::eR32G32B32A32Sfloat;
	if (is_float) {
		g_phase48_float_flip.store(true, std::memory_order_release);
	}
	const void* guest = info.buffer;
	if (guest == nullptr || info.bytes_per_element == 0 || info.width == 0 || info.height == 0) {
		static std::atomic<int> miss {0};
		if (miss.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("FlipTrace: phase48 content probe skip index=%d format=%d (no guest attrs)\n",
			     index, static_cast<int>(format));
		}
		return;
	}
	const uint32_t bpe   = info.bytes_per_element;
	const uint32_t pitch = info.buffer_pitch != 0 ? static_cast<uint32_t>(info.buffer_pitch) : info.width;
	const uint32_t xs[4] = {0u, info.width / 2u, info.width > 1 ? info.width - 1u : 0u, info.width / 4u};
	const uint32_t ys[4] = {0u, info.height / 2u, info.height > 1 ? info.height - 1u : 0u,
	                        info.height / 4u};
	uint64_t nonzero_bits = 0;
	uint32_t sample_ok    = 0;
	for (int s = 0; s < 4; ++s) {
		const uint64_t off =
		    (static_cast<uint64_t>(ys[s]) * pitch + xs[s]) * static_cast<uint64_t>(bpe);
		if (off + bpe > info.buffer_size && info.buffer_size != 0) {
			continue;
		}
		uint8_t pixel[16] {};
		const size_t n = bpe < sizeof(pixel) ? bpe : sizeof(pixel);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		__try {
			std::memcpy(pixel, static_cast<const uint8_t*>(guest) + off, n);
		} __except (EXCEPTION_EXECUTE_HANDLER) {
			continue;
		}
#else
		std::memcpy(pixel, static_cast<const uint8_t*>(guest) + off, n);
#endif
		++sample_ok;
		for (size_t i = 0; i < n; ++i) {
			nonzero_bits |= pixel[i];
		}
	}
	const bool nonzero = nonzero_bits != 0;
	if (nonzero) {
		g_phase48_content_nonzero.store(true, std::memory_order_release);
	} else if (sample_ok > 0) {
		g_phase48_content_zero.store(true, std::memory_order_release);
	}
	static std::atomic<uint32_t> log_n {0};
	const uint32_t               n = log_n.fetch_add(1, std::memory_order_relaxed);
	if (n < 24 || (nonzero && n < 64)) {
		const char* layer = !nonzero && sample_ok > 0 ? "L0_blank"
		                    : (nonzero && is_float)   ? "L2_float_nonzero"
		                    : nonzero                 ? "content_nonzero"
		                                              : "probe_fail";
		LOGF("FlipTrace: phase48 content index=%d format=%d float=%d nonzero=%d samples=%u "
		     "layer=%s addr=0x%016" PRIx64 " %ux%u bpe=%u submit_gpu=%" PRIu64 "\n",
		     index, static_cast<int>(format), is_float ? 1 : 0, nonzero ? 1 : 0, sample_ok, layer,
		     reinterpret_cast<uint64_t>(guest), info.width, info.height, bpe,
		     FlipStats::submit_gpu.load(std::memory_order_relaxed));
		fprintf(stderr,
		        "FlipTrace: phase48 content layer=%s float=%d nonzero=%d format=%d\n", layer,
		        is_float ? 1 : 0, nonzero ? 1 : 0, static_cast<int>(format));
	}
}

void FlipQueue::Prepare(uint64_t request_id, Graphics::CommandBuffer& buffer) {
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
	// Blank frames stay opaque so the SDL surface shows a solid black frame.
	auto& frame = special ? Graphics::WindowPrepareBlankFrame(buffer, width, height, true)
	                      : Graphics::WindowPrepareFrame(buffer, *source);
	if (!special && source != nullptr) {
		Phase48ClassifyPresentContent(cfg, index, source);
		static std::atomic<uint32_t> prep_n {0};
		const uint32_t               pn = prep_n.fetch_add(1, std::memory_order_relaxed);
		if (pn < 24) {
			const auto& buf = cfg->buffers[index];
			LOGF("FlipTrace: phase49 flip prepare id=%" PRIu64 " index=%d buffer=%p vk=%p "
			     "count=%" PRIu64 " currentBuffer=%d pending=%d nonzero=%d\n",
			     request_id, index, buf.buffer, static_cast<void*>(source),
			     cfg->flip_status.count, cfg->flip_status.currentBuffer,
			     cfg->flip_status.flipPendingNum,
			     g_phase48_content_nonzero.load(std::memory_order_relaxed) ? 1 : 0);
			fprintf(stderr,
			        "FlipTrace: phase49 flip prepare index=%d vk=%p pending=%d\n", index,
			        static_cast<void*>(source), cfg->flip_status.flipPendingNum);
		}
	}

	Common::LockGuard lock(m_mutex);
	auto request = std::find_if(m_requests.begin(), m_requests.end(),
	                            [request_id](const auto& r) { return r.id == request_id; });
	if (request == m_requests.end() || request->state != RequestState::Recording ||
	    request->frame != nullptr) {
		EXIT("video-out request changed while recording, id=%" PRIu64 "\n", request_id);
	}
	request->frame = &frame;
	LOGF("FlipTrace: Prepare id=%" PRIu64 " index=%d state=Recording frame=%p\n", request_id, index,
	     static_cast<void*>(&frame));
}

uint64_t FlipQueue::PrepareNextCpu(Graphics::CommandBuffer& buffer) {
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

void FlipQueue::Wait(VideoOutConfig& cfg, int index) {
	Common::LockGuard lock(m_mutex);

	auto has_request = [this, &cfg, index] {
		auto matches = [&cfg, index](const auto& r) { return r.cfg == &cfg && r.index == index; };
		return std::any_of(m_requests.begin(), m_requests.end(), matches) ||
		       std::any_of(m_cpu_requests.begin(), m_cpu_requests.end(), matches);
	};
	while (has_request()) {
		m_done_cond_var.Wait(&m_mutex);
	}
}

bool FlipQueue::HasPending(VideoOutConfig& cfg, int start_index, int count) {
	if (count <= 0 || start_index > INT_MAX - count) {
		EXIT("invalid video-out pending-flip query range\n");
	}
	Common::LockGuard lock(m_mutex);
	auto              matches = [&](const auto& request) {
		return request.cfg == &cfg && request.index >= start_index &&
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
			AfterGpuQueueEmptyHostTick();
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
	if (!IsFlipDue(*r.cfg)) {
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

	Graphics::WindowPresentFrame(*r.frame);

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
	// Phase 29/31: during first blank hold keep currentBuffer blank + sticky pending.
	static std::atomic<bool> hold_started {false};
	static std::atomic<int>  sticky_after_blank {0};
	const bool first_blank_hold =
	    IsSpecialBufferIndex(r.index) &&
	    FlipStats::presented.load(std::memory_order_relaxed) == 0 &&
	    !hold_started.exchange(true, std::memory_order_acq_rel);
	if (first_blank_hold) {
		r.cfg->flip_status.currentBuffer  = VIDEO_OUT_BUFFER_INDEX_BLANK;
		r.cfg->flip_status.flipPendingNum = 1;
	} else {
		r.cfg->flip_status.currentBuffer =
		    IsSpecialBufferIndex(r.index) ? VIDEO_OUT_BUFFER_INDEX_BLANK : r.index;
		const int sticky = sticky_after_blank.load(std::memory_order_relaxed);
		if (sticky > 0 && sticky_after_blank.fetch_sub(1, std::memory_order_acq_rel) > 0) {
			r.cfg->flip_status.flipPendingNum = 1;
			LOGF("FlipTrace: sticky pending after present index=%d remaining=%d\n", r.index,
			     sticky - 1);
			fprintf(stderr, "FlipTrace: sticky pending after present index=%d\n", r.index);
		} else {
			r.cfg->flip_status.flipPendingNum =
			    static_cast<int>(m_requests.size() + m_cpu_requests.size());
		}
	}
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

	{
		static std::atomic<uint32_t> flip_n {0};
		const uint32_t               fn = flip_n.fetch_add(1, std::memory_order_relaxed);
		if (fn < 32 || (r.source == FlipRequestSource::GpuEop && fn < 64)) {
			auto* vk = (!IsSpecialBufferIndex(r.index) && r.cfg != nullptr)
			               ? r.cfg->buffers[r.index].buffer_vulkan
			               : nullptr;
			LOGF("FlipTrace: phase49 flip done id=%" PRIu64 " index=%d src=%s vk=%p count=%" PRIu64
			     " currentBuffer=%d pending=%d submit_cpu=%" PRIu64 " submit_gpu=%" PRIu64
			     " nonzero=%d\n",
			     r.id, r.index, r.source == FlipRequestSource::GpuEop ? "gpu" : "cpu",
			     static_cast<void*>(vk), r.cfg->flip_status.count, r.cfg->flip_status.currentBuffer,
			     r.cfg->flip_status.flipPendingNum,
			     FlipStats::submit_cpu.load(std::memory_order_relaxed),
			     FlipStats::submit_gpu.load(std::memory_order_relaxed),
			     g_phase48_content_nonzero.load(std::memory_order_relaxed) ? 1 : 0);
			if (fn < 12 || r.source == FlipRequestSource::GpuEop) {
				fprintf(stderr,
				        "FlipTrace: phase49 flip done index=%d src=%s pending=%d nonzero=%d\n",
				        r.index, r.source == FlipRequestSource::GpuEop ? "gpu" : "cpu",
				        r.cfg->flip_status.flipPendingNum,
				        g_phase48_content_nonzero.load(std::memory_order_relaxed) ? 1 : 0);
			}
		}
	}

	if (first_blank_hold) {
		const int bisect = Phase30BisectMode();
		FlipStats::presented.fetch_add(1, std::memory_order_relaxed);
		LOGF("FlipTrace: presented index=%d id=%" PRIu64 " flip_count=%" PRIu64
		     " currentBuffer=%d pending=%d (hold gate) bisect=%d\n",
		     r.index, r.id, r.cfg->flip_status.count, r.cfg->flip_status.currentBuffer,
		     r.cfg->flip_status.flipPendingNum, bisect);
		fprintf(stderr, "FlipTrace: hold gate bisect=%d\n", bisect);
		FlipStats::Log("presented");
		Graphics::WindowHoldVisible(10);

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
							event.event.filter = EventQueue::KERNEL_EVFILT_VIDEO_OUT;
							event.event.udata              = nullptr;
							event.event.fflags             = 0;
							event.event.data               = 0;
							event.filter.delete_event_func = RemoveVideoOutEventQueue;
							event.filter.reset_func        = ResetVideoOutEvent;
							event.filter.trigger_func      = TriggerVideoOutEvent;
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

void FlipQueue::GetFlipStatus(VideoOutConfig& cfg, VideoOutFlipStatus& out) {
	Common::LockGuard lock(m_mutex);

	out = cfg.flip_status;
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

static int RegisterBuffersInternal(VideoOutConfig& ctx, int set_id, int start_index,
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
// first SubmitFlip alone must not park MainThread. Phase 46 park also needs real DCB
// or native PM4 EOP (see Phase38DeferredSoftIdle / Phase46ReadyForSoftIdle).
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

bool Phase45NdJobDcbSeen() {
	return g_phase45_ndjob_dcb_ok.load(std::memory_order_acquire);
}

bool Phase46RealDcbSeen() {
	return g_phase46_real_dcb_ok.load(std::memory_order_acquire);
}

bool Phase46NativeEopSeen() {
	return g_phase46_native_eop_ok.load(std::memory_order_acquire);
}

bool Phase46ReadyForSoftIdle() {
	return Phase46NativeEopSeen() || Phase46RealDcbSeen();
}

bool Phase47GuestDrawSeen() {
	return g_phase47_guest_draw_ok.load(std::memory_order_acquire) ||
	       g_phase47_guest_dispatch_ok.load(std::memory_order_acquire);
}

bool Phase48ContentNonZeroSeen() {
	return g_phase48_content_nonzero.load(std::memory_order_acquire);
}

bool Phase48FloatFlipSeen() {
	return g_phase48_float_flip.load(std::memory_order_acquire);
}

bool Phase47ReadyForSoftIdle() {
	// Phase 48: host DRAW seed must not park soft-idle (L0 black).
	if (Phase47GuestDrawSeen()) {
		return true;
	}
	const char* allow = std::getenv("KYTY_PHASE47_ALLOW_EOP_ONLY");
	if (allow != nullptr && allow[0] == '1') {
		return Phase46ReadyForSoftIdle();
	}
	return false;
}

static bool Phase46IsPaddingOnlyDcb(const uint32_t* dcb, uint32_t size_in_dwords) {
	if (dcb == nullptr || size_in_dwords == 0) {
		return true;
	}
	for (uint32_t i = 0; i < size_in_dwords; i++) {
		const uint32_t dw = dcb[i];
		if (dw != 0u && dw != 0x80000000u) {
			return false;
		}
	}
	return true;
}

static bool Phase46HasRealPm4(const uint32_t* dcb, uint32_t size_in_dwords) {
	if (dcb == nullptr || size_in_dwords < 2) {
		return false;
	}
	const uint32_t n = size_in_dwords < 64u ? size_in_dwords : 64u;
	for (uint32_t i = 0; i < n; i++) {
		const uint32_t dw = dcb[i];
		if ((dw >> 30) == 3u) {
			return true;
		}
	}
	return !Phase46IsPaddingOnlyDcb(dcb, size_in_dwords);
}

static bool Phase47IsDrawOpcode(uint32_t op) {
	namespace Pm4 = Libs::Graphics::Pm4;
	switch (op) {
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DRAW:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE: return true;
		default: return false;
	}
}

static bool Phase47IsDispatchOpcode(uint32_t op) {
	namespace Pm4 = Libs::Graphics::Pm4;
	return op == Pm4::IT_DISPATCH_DIRECT || op == Pm4::IT_DISPATCH_INDIRECT;
}

enum class Phase47DcbKind { Padding, EopOnly, Draw, Dispatch, Other };

static Phase47DcbKind Phase47ClassifyDcb(const uint32_t* dcb, uint32_t size_in_dwords) {
	namespace Pm4 = Libs::Graphics::Pm4;
	if (dcb == nullptr || size_in_dwords == 0 || Phase46IsPaddingOnlyDcb(dcb, size_in_dwords)) {
		return Phase47DcbKind::Padding;
	}
	bool saw_type3    = false;
	bool saw_draw     = false;
	bool saw_dispatch = false;
	bool saw_eop      = false;
	bool saw_other    = false;
	const uint32_t n  = size_in_dwords < 64u ? size_in_dwords : 64u;
	for (uint32_t i = 0; i < n;) {
		const uint32_t dw   = dcb[i];
		const uint32_t type = dw >> 30;
		if (type == 2u) {
			++i;
			continue;
		}
		if (type != 3u) {
			++i;
			continue;
		}
		saw_type3             = true;
		const uint32_t op     = (dw >> 8) & 0xffu;
		const uint32_t pkt_dw = ((((dw) >> 16u) & 0x3fffu) + 2u);
		if (pkt_dw == 0 || i + pkt_dw > size_in_dwords) {
			saw_other = true;
			break;
		}
		const uint32_t r_field = ((dw) >> 2u) & (Pm4::R_NUM - 1u);
		if (Phase47IsDrawOpcode(op)) {
			saw_draw = true;
		} else if (Phase47IsDispatchOpcode(op)) {
			saw_dispatch = true;
		} else if (op == Pm4::IT_EVENT_WRITE_EOP || op == Pm4::IT_EVENT_WRITE_EOS ||
		           op == Pm4::IT_RELEASE_MEM ||
		           (op == Pm4::IT_NOP && r_field == Pm4::R_RELEASE_MEM)) {
			saw_eop = true;
		} else if (op != Pm4::IT_NOP) {
			saw_other = true;
		}
		i += pkt_dw;
	}
	if (saw_draw) {
		return Phase47DcbKind::Draw;
	}
	if (saw_dispatch) {
		return Phase47DcbKind::Dispatch;
	}
	if (saw_type3 && saw_eop && !saw_other) {
		return Phase47DcbKind::EopOnly;
	}
	if (saw_type3) {
		return Phase47DcbKind::Other;
	}
	return Phase47DcbKind::Padding;
}

static const char* Phase47DcbKindName(Phase47DcbKind kind) {
	switch (kind) {
		case Phase47DcbKind::Draw: return "draw";
		case Phase47DcbKind::Dispatch: return "dispatch";
		case Phase47DcbKind::EopOnly: return "eop_only";
		case Phase47DcbKind::Other: return "other";
		case Phase47DcbKind::Padding:
		default: return "padding";
	}
}

void Phase47PostSeedWake(const char* why) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	if (Phase47GuestDrawSeen()) {
		return;
	}
	if (!Phase54AllowHostWake()) {
		static std::atomic<uint32_t> skips {0};
		const uint32_t               sn = skips.fetch_add(1, std::memory_order_relaxed);
		if (sn < 8 || (sn % 64) == 0) {
			LOGF("SubmitTrace: phase54 wake_budget_skip post-seed cycle=%" PRIu64 " n=%u why=%s\n",
			     Phase54CurrentCycleId(), sn, why != nullptr ? why : "?");
			fprintf(stderr, "SubmitTrace: phase54 wake_budget_skip post-seed n=%u\n", sn);
		}
		return;
	}
	Phase54NoteHostWake(why != nullptr ? why : "post-seed");
	Phase41ApplyKeep1SideEffects();
	Phase42RearmNdJobEnqueue();
	(void)LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
	(void)LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);
	const uint32_t n = g_phase47_post_seed_wake_n.fetch_add(1, std::memory_order_relaxed);
	if (n < 8 || (n % 32) == 0) {
		LOGF("SubmitTrace: phase47 post-seed wake n=%u why=%s tid=%d\n", n,
		     why != nullptr ? why : "?", Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase47 post-seed wake n=%u why=%s\n", n,
		        why != nullptr ? why : "?");
	}
}

void Phase47NoteSubmitAcb(uint64_t submit_count) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	static std::atomic<bool> logged {false};
	if (!logged.exchange(true, std::memory_order_acq_rel)) {
		LOGF("SubmitTrace: phase47 submit_acb=%" PRIu64 " tid=%d\n", submit_count,
		     Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase47 submit_acb=%" PRIu64 "\n", submit_count);
	}
}

void Phase46InspectSubmitDcb(uint32_t queue, const uint32_t* dcb, uint32_t size_in_dwords) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	if (dcb == nullptr || size_in_dwords == 0) {
		return;
	}
	const uint32_t       dump_n  = g_phase46_dcb_dump_n.fetch_add(1, std::memory_order_relaxed);
	const bool           is_nop  = Phase46IsPaddingOnlyDcb(dcb, size_in_dwords);
	const bool           is_real = Phase46HasRealPm4(dcb, size_in_dwords);
	const Phase47DcbKind kind    = Phase47ClassifyDcb(dcb, size_in_dwords);
	const bool is_seed = (g_phase46_eop_seed_dcb != nullptr && dcb == g_phase46_eop_seed_dcb);
	if (dump_n < 16 || (is_real && !g_phase46_real_dcb_ok.load(std::memory_order_acquire)) ||
	    ((kind == Phase47DcbKind::Draw || kind == Phase47DcbKind::Dispatch) &&
	     !Phase47GuestDrawSeen())) {
		char line[512];
		int  pos = std::snprintf(
		    line, sizeof(line),
		    "SubmitTrace: phase46 dcb dump n=%u queue=0x%" PRIx32 " size=0x%" PRIx32
		    " nop=%d real=%d kind=%s seed=%d tid=%d dw:",
		    dump_n, queue, size_in_dwords, is_nop ? 1 : 0, is_real ? 1 : 0,
		    Phase47DcbKindName(kind), is_seed ? 1 : 0, Common::Thread::GetThreadIdUnique());
		const uint32_t show = size_in_dwords < 16u ? size_in_dwords : 16u;
		for (uint32_t i = 0; i < show && pos > 0 && static_cast<size_t>(pos) + 12 < sizeof(line);
		     i++) {
			pos += std::snprintf(line + pos, sizeof(line) - static_cast<size_t>(pos), " %08x",
			                     dcb[i]);
		}
		LOGF("%s\n", line);
		fprintf(stderr, "%s\n", line);
	}
	if (is_real && !g_phase46_real_dcb_ok.exchange(true, std::memory_order_acq_rel)) {
		LOGF("SubmitTrace: phase46 real_dcb queue=0x%" PRIx32 " size=0x%" PRIx32 " tid=%d\n",
		     queue, size_in_dwords, Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase46 real_dcb size=0x%" PRIx32 " tid=%d\n",
		        size_in_dwords, Common::Thread::GetThreadIdUnique());
		g_phase45_ndjob_dcb_ok.store(true, std::memory_order_release);
		g_phase44_ndjob_dcb_ok.store(true, std::memory_order_release);
	}
	if (kind == Phase47DcbKind::Draw && !is_seed &&
	    !g_phase47_guest_draw_ok.exchange(true, std::memory_order_acq_rel)) {
		LOGF("SubmitTrace: phase47 guest_draw_dcb queue=0x%" PRIx32 " size=0x%" PRIx32
		     " seed=0 tid=%d\n",
		     queue, size_in_dwords, Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase47 guest_draw_dcb size=0x%" PRIx32 " seed=0 tid=%d\n",
		        size_in_dwords, Common::Thread::GetThreadIdUnique());
		const uint64_t tsc = LibKernel::KernelGetProcessTimeCounter();
		LOGF("SubmitTrace: phase50 dcb guest_draw seed=0 queue=0x%" PRIx32 " size=0x%" PRIx32
		     " tsc=%" PRIu64 " tid=%d\n",
		     queue, size_in_dwords, tsc, Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase50 dcb guest_draw seed=0 size=0x%" PRIx32 "\n",
		        size_in_dwords);
		Phase51TryBypassFlipL0("guest_draw");
	} else if (kind == Phase47DcbKind::Draw && is_seed) {
		static std::atomic<bool> seed_logged {false};
		if (!seed_logged.exchange(true, std::memory_order_acq_rel)) {
			LOGF("SubmitTrace: phase48 host_draw_seed ignored for soft-idle size=0x%" PRIx32
			     " tid=%d\n",
			     size_in_dwords, Common::Thread::GetThreadIdUnique());
			fprintf(stderr, "SubmitTrace: phase48 host_draw_seed ignored for soft-idle\n");
		}
	}
	if (kind == Phase47DcbKind::Dispatch && !is_seed &&
	    !g_phase47_guest_dispatch_ok.exchange(true, std::memory_order_acq_rel)) {
		LOGF("SubmitTrace: phase47 guest_dispatch_dcb queue=0x%" PRIx32 " size=0x%" PRIx32
		     " seed=0 tid=%d\n",
		     queue, size_in_dwords, Common::Thread::GetThreadIdUnique());
		fprintf(stderr,
		        "SubmitTrace: phase47 guest_dispatch_dcb size=0x%" PRIx32 " seed=0 tid=%d\n",
		        size_in_dwords, Common::Thread::GetThreadIdUnique());
	}
}

void Phase54NoteSubmit(uint32_t queue, const uint32_t* dcb, uint32_t size_in_dwords) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const char* kind = "error";
	if (dcb != nullptr && size_in_dwords != 0) {
		const bool           is_seed = (g_phase46_eop_seed_dcb != nullptr && dcb == g_phase46_eop_seed_dcb);
		const Phase47DcbKind dcb_kind = Phase47ClassifyDcb(dcb, size_in_dwords);
		if (is_seed) {
			kind = "seed_host";
		} else if (dcb_kind == Phase47DcbKind::Draw || dcb_kind == Phase47DcbKind::Dispatch) {
			kind = "guest_real";
			g_phase54_cycle_guest_real.store(true, std::memory_order_release);
			(void)g_phase55_submit_guest_real_n.fetch_add(1, std::memory_order_relaxed);
		} else {
			kind = "error";
		}
	}
	const uint64_t cycle = g_phase54_cycle_id.load(std::memory_order_acquire);
	const int      tid   = Common::Thread::GetThreadIdUnique();
	const int      sub   = LibKernel::PthreadCurrentIsSubmissionRelated() ? 1 : 0;
	const int      mainr = LibKernel::PthreadCurrentIsMainRelated() ? 1 : 0;
	static std::atomic<uint32_t> logs {0};
	const uint32_t               n = logs.fetch_add(1, std::memory_order_relaxed);
	if (n < 64 || std::strcmp(kind, "guest_real") == 0) {
		LOGF("SubmitTrace: phase54 submit cycle=%" PRIu64 " tid=%d sub=%d main=%d queue=0x%" PRIx32
		     " size=0x%" PRIx32 " kind=%s\n",
		     cycle, tid, sub, mainr, queue, size_in_dwords, kind);
		fprintf(stderr,
		        "SubmitTrace: phase54 submit cycle=%" PRIu64 " tid=%d kind=%s size=0x%" PRIx32 "\n",
		        cycle, tid, kind, size_in_dwords);
	}
	if (mainr != 0 && dcb != nullptr) {
		Phase57NoteMainAgcTouch("SubmitCommandBuffer", reinterpret_cast<uint64_t>(dcb),
		                        static_cast<uint64_t>(size_in_dwords),
		                        static_cast<uint64_t>(queue), 0);
	}
	Phase59NoteSubmit(queue, dcb, size_in_dwords, kind);
}

void Phase46NoteNativeEop(const char* source) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	if (!g_phase46_native_eop_ok.exchange(true, std::memory_order_acq_rel)) {
		LOGF("SubmitTrace: phase46 native_eop source=%s tid=%d\n",
		     source != nullptr ? source : "?", Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase46 native_eop source=%s\n",
		        source != nullptr ? source : "?");
		const uint64_t tsc = LibKernel::KernelGetProcessTimeCounter();
		LOGF("SubmitTrace: phase50 dcb native_eop source=%s submit_gpu=%" PRIu64
		     " guest_draw=%d tsc=%" PRIu64 " tid=%d\n",
		     source != nullptr ? source : "?",
		     FlipStats::submit_gpu.load(std::memory_order_relaxed),
		     g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0, tsc,
		     Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase50 dcb native_eop guest_draw=%d submit_gpu=%" PRIu64 "\n",
		        g_phase47_guest_draw_ok.load(std::memory_order_relaxed) ? 1 : 0,
		        FlipStats::submit_gpu.load(std::memory_order_relaxed));
		Phase50PollKeep1Obj("native_eop");
	}
}

void Phase45NoteSubmitDcb(uint64_t submit_count) {
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	// MainThread P43 seed must not count as Phase 45 success.
	if (g_phase43_dcb_seeded.load(std::memory_order_acquire) &&
	    !g_phase45_timeout_submit_done.load(std::memory_order_acquire)) {
		Phase44CaptureDcbBaselineIfNeeded();
		return;
	}
	Phase44CaptureDcbBaselineIfNeeded();
	const uint64_t base = g_phase44_dcb_baseline.load(std::memory_order_acquire);
	if (submit_count <= base) {
		return;
	}
	if (!g_phase45_ndjob_dcb_ok.exchange(true, std::memory_order_acq_rel)) {
		g_phase44_ndjob_dcb_ok.store(true, std::memory_order_release);
		LOGF("FlipTrace: phase45 ndjob submit_dcb=%" PRIu64 " baseline=%" PRIu64 " tid=%d\n",
		     submit_count, base, Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "FlipTrace: phase45 ndjob submit_dcb=%" PRIu64 "\n", submit_count);
	}
}

void Phase45OnNdJobSyncTimeout(LibKernel::EventQueue::KernelEqueue eq) {
	(void)eq;
	if (!g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	// Phase 47: once guest draw/dispatch is seen, stop timeout bridging.
	if (Phase47GuestDrawSeen()) {
		return;
	}
	Phase44CaptureDcbBaselineIfNeeded();
	const uint64_t dcb =
	    Libs::Graphics::Sync::SubmitTrace::submit_dcb.load(std::memory_order_relaxed);
	const uint64_t base = g_phase44_dcb_baseline.load(std::memory_order_acquire);
	if (dcb > base) {
		Phase45NoteSubmitDcb(dcb);
	}

	// Seed already done: keep rearming so the fiber can produce a real draw DCB.
	if (g_phase45_timeout_submit_done.load(std::memory_order_acquire)) {
		Phase47PostSeedWake("ndjob-timeout-post-seed");
		return;
	}

	Phase42RearmNdJobEnqueue();
	(void)LibKernel::PthreadWakeSubmissionCondWaitersUnlimited();
	(void)LibKernel::EventQueue::KernelTriggerUserEventForAll(0x1800, nullptr);

	// Opt-in legacy NOP (does not count as phase46 real_dcb / native_eop).
	const char* nop_env = std::getenv("KYTY_PHASE45_NOP_SUBMIT");
	if (nop_env != nullptr && nop_env[0] == '1') {
		const uint32_t attempt =
		    g_phase45_timeout_submit_attempts.fetch_add(1, std::memory_order_relaxed);
		if (attempt < 4) {
			return;
		}
		if (g_phase45_timeout_submit_done.exchange(true, std::memory_order_acq_rel)) {
			return;
		}
		static uint32_t nop_storage[8] = {0x80000000u, 0x80000000u, 0, 0, 0, 0, 0, 0};
		LOGF("SubmitTrace: phase45 NdJob timeout submit (opt-in NOP) dcb=%p size=0x2 tid=%d\n",
		     static_cast<void*>(nop_storage), Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase45 NdJob timeout submit (opt-in NOP) tid=%d\n",
		        Common::Thread::GetThreadIdUnique());
		(void)Libs::Graphics::Gen5Driver::GraphicsDriverSubmitCommandBuffer(0, nop_storage, 2);
		Phase47PostSeedWake("after-nop-seed");
		return;
	}

	// Phase 46 default bridge: one-shot real PM4 IT_EVENT_WRITE_EOP (interrupt_sel=1).
	const char* seed_env = std::getenv("KYTY_PHASE46_PM4_EOP_SEED");
	if (seed_env != nullptr && seed_env[0] == '0') {
		return;
	}
	const uint32_t attempt =
	    g_phase45_timeout_submit_attempts.fetch_add(1, std::memory_order_relaxed);
	if (attempt < 4) {
		return;
	}
	if (g_phase45_timeout_submit_done.exchange(true, std::memory_order_acq_rel)) {
		Phase47PostSeedWake("ndjob-timeout-race");
		return;
	}
	// Phase 47 bridge: IT_DRAW_INDEX_AUTO (triangle) + IT_EVENT_WRITE_EOP.
	// keep[1] stays HLE with empty *obj so the fiber never builds a guest DRAW DCB;
	// this NdJob-path seed is the analogue of the P46 EOP-only seed.
	// Disable with KYTY_PHASE46_PM4_EOP_SEED=0 (same gate) or KYTY_PHASE47_DRAW_SEED=0.
	const char* draw_env = std::getenv("KYTY_PHASE47_DRAW_SEED");
	static uint32_t draw_eop_dcb[12] = {
	    // KYTY_PM4(3, IT_DRAW_INDEX_AUTO, 0) — index_count=3, flags=0x2
	    0xC0012D00u,
	    0x00000003u,
	    0x00000002u,
	    // IT_EVENT_WRITE_EOP, interrupt_sel=1
	    0xC0044700u,
	    0x00000000u,
	    0x00000000u,
	    0x01000000u,
	    0x00000000u,
	    0x00000000u,
	    0,
	    0,
	    0,
	};
	if (draw_env != nullptr && draw_env[0] == '0') {
		static uint32_t eop_only[8] = {
		    0xC0044700u, 0x00000000u, 0x00000000u, 0x01000000u, 0x00000000u, 0x00000000u, 0, 0,
		};
		g_phase46_eop_seed_dcb = eop_only;
		LOGF("SubmitTrace: phase46 NdJob timeout PM4 EVENT_WRITE_EOP dcb=%p size=0x6 tid=%d\n",
		     static_cast<void*>(eop_only), Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase46 NdJob timeout PM4 EVENT_WRITE_EOP tid=%d\n",
		        Common::Thread::GetThreadIdUnique());
		(void)Libs::Graphics::Gen5Driver::GraphicsDriverSubmitCommandBuffer(0, eop_only, 6);
	} else {
		g_phase46_eop_seed_dcb = draw_eop_dcb;
		LOGF("SubmitTrace: phase47 NdJob timeout PM4 DRAW_INDEX_AUTO+EOP dcb=%p size=0x9 tid=%d\n",
		     static_cast<void*>(draw_eop_dcb), Common::Thread::GetThreadIdUnique());
		fprintf(stderr, "SubmitTrace: phase47 NdJob timeout PM4 DRAW+EOP tid=%d\n",
		        Common::Thread::GetThreadIdUnique());
		(void)Libs::Graphics::Gen5Driver::GraphicsDriverSubmitCommandBuffer(0, draw_eop_dcb, 9);
	}
	Phase47PostSeedWake("after-draw-eop-seed");
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
	// Phase 47: keep host sustain after Reg2+menu_frames — stopping here froze flips at
	// presented≈60 once soft-idle parked (MainThread stopped handoff).
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

void Phase66TryMenuRecycle() {
	if (!Phase66MenuRecycleEnabled() ||
	    !g_phase37_post_unreg.load(std::memory_order_acquire)) {
		return;
	}
	const uint32_t guest_reg2 = g_phase65_guest_regbuf2.load(std::memory_order_relaxed);
	const bool     guest_flip = g_phase37_guest_flip_seen.load(std::memory_order_acquire) ||
	                        g_phase65_guest_flip.load(std::memory_order_relaxed) > 0;
	if (guest_reg2 == 0 || !guest_flip) {
		g_phase66_pending_streak.store(0, std::memory_order_relaxed);
		return;
	}
	if (g_video_out_context == nullptr) {
		return;
	}

	int      handle  = g_phase42_flip_handle.load(std::memory_order_acquire);
	int32_t  pending = -1;
	bool     got     = false;
	for (int h = 1; h < 8; ++h) {
		if (!g_video_out_context->IsOpened(h)) {
			continue;
		}
		auto* ctx = g_video_out_context->Get(h);
		if (ctx == nullptr) {
			continue;
		}
		Common::LockGuard lock(ctx->mutex);
		pending = ctx->flip_status.flipPendingNum;
		got     = true;
		if (handle <= 0) {
			handle = h;
		}
		break;
	}
	if (!got || handle <= 0) {
		return;
	}

	if (pending == 0) {
		(void)g_phase66_pending_streak.fetch_add(1, std::memory_order_relaxed);
	} else {
		g_phase66_pending_streak.store(0, std::memory_order_relaxed);
		return;
	}
	const uint32_t streak = g_phase66_pending_streak.load(std::memory_order_relaxed);
	if (streak < kPhase66IdleNeed) {
		return;
	}

	const auto now = std::chrono::steady_clock::now();
	if (g_phase66_have_last_flip.load(std::memory_order_acquire)) {
		const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		                    now - g_phase66_last_flip)
		                    .count();
		if (ms < static_cast<int64_t>(kPhase66MinFlipIntervalMs)) {
			return;
		}
	}

	// Mark this host tid so Phase37NoteGuestSubmitFlip ignores synthetic recycle flips.
	const uint64_t prev_ka = g_phase37_keepalive_host_tid.load(std::memory_order_acquire);
	g_phase37_keepalive_host_tid.store(Phase37HostThreadId(), std::memory_order_release);
	const int result = VideoOutSubmitFlip(handle, 0, /*VIDEO_OUT_FLIP_MODE_VSYNC*/ 1, 0);
	g_phase37_keepalive_host_tid.store(prev_ka, std::memory_order_release);

	g_phase66_last_flip = now;
	g_phase66_have_last_flip.store(true, std::memory_order_release);
	const uint32_t n = g_phase66_recycle_n.fetch_add(1, std::memory_order_relaxed) + 1;
	g_phase66_pending_streak.store(0, std::memory_order_relaxed);

	LOGF("FlipTrace: phase66 menu_recycle flip index=0 handle=%d pending_streak=%u "
	     "result=%d recycle_n=%u\n",
	     handle, streak, result, n);
	fprintf(stderr,
	        "FlipTrace: phase66 menu_recycle flip index=0 pending_streak=%u result=%d "
	        "recycle_n=%u\n",
	        streak, result, n);
	(void)LibKernel::PthreadWakeSubmissionCondWaitersAfterFlip();
	if (result == OK) {
		g_phase66_heatmap_done.store(false, std::memory_order_release);
		Phase66EmitHeatmap("after_recycle");
	}
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
		Phase65NoteGuestFlip();
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
	Phase62NoteUnreg();
	g_phase37_guest_flip_seen.store(false, std::memory_order_release);
	g_phase37_guest_reg_seen.store(false, std::memory_order_release);
	g_phase37_keepalive_stop.store(false, std::memory_order_release);
	g_phase42_reregister_ok.store(false, std::memory_order_release);
	g_phase42_flip_attempted.store(false, std::memory_order_release);
	g_phase42_flip_handle.store(0, std::memory_order_release);
	g_phase42_flip_num.store(0, std::memory_order_release);
	Phase53ArmWorkerFiberTraces();
	Phase50StartObjPollThread();
	Phase53ProbeRetarget("post_unreg");
	Phase55StartWatchThread();
	Phase55TryArmMixedThunk();

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

static int RegisterBuffersInternal(VideoOutConfig& ctx, int set_id, int start_index,
                                   const void* const* addresses, int buffer_num,
                                   const std::vector<Graphics::VideoOutInfo>& infos) {
	if (addresses == nullptr || buffer_num <= 0 ||
	    infos.size() != static_cast<size_t>(buffer_num)) {
		EXIT("invalid internal video-out buffer registration arguments\n");
	}
	if (set_id < 0 || set_id >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX) {
		EXIT("internal video-out buffer set identifier is out of range\n");
	}
	Common::LockGuard lock(ctx.mutex);
	if (ctx.closing) {
		EXIT("cannot register buffers on a closing video-out handle\n");
	}
	if (std::any_of(ctx.buffers_sets.begin(), ctx.buffers_sets.end(),
	                [set_id](const auto& set) { return set.set_id == set_id; })) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}
	for (int i = 0; i < buffer_num; i++) {
		if (ctx.unregistering[start_index + i]) {
			EXIT("video-out buffer registration raced with unregistration\n");
		}
		if (ctx.buffers[start_index + i].buffer != nullptr) {
			return VIDEO_OUT_ERROR_SLOT_OCCUPIED;
		}
	}
	// After Unregister deferred-vk, buffer_vulkan remains — reuse to avoid TextureCache
	// "aliases cached image pages" EXIT on post-Unreg ABI re-Register (P44).
	std::vector<Graphics::VideoOutVulkanImage*> images;
	images.reserve(static_cast<size_t>(buffer_num));
	bool reuse_vk = g_phase37_post_unreg.load(std::memory_order_acquire);
	if (reuse_vk) {
		for (int i = 0; i < buffer_num; i++) {
			auto* vk = ctx.buffers[start_index + i].buffer_vulkan;
			if (vk == nullptr) {
				reuse_vk = false;
				break;
			}
			images.push_back(vk);
		}
	}
	if (reuse_vk) {
		LOGF("FlipTrace: phase44 reuse deferred-vk VideoOut surfaces set_id=%d num=%d\n", set_id,
		     buffer_num);
		fprintf(stderr, "FlipTrace: phase44 reuse deferred-vk surfaces num=%d\n", buffer_num);
	} else {
		{
			char breadcrumb[128];
			std::snprintf(breadcrumb, sizeof(breadcrumb),
			              "FlipTrace: RegisterVideoOutSurfaces begin num=%d", buffer_num);
			Common::EmergencyLogRaw(breadcrumb);
			Common::LogFatalToFile(breadcrumb);
		}
		images = Graphics::GetRenderContext().GetTextureCache().RegisterVideoOutSurfaces(infos);
		{
			char breadcrumb[128];
			std::snprintf(breadcrumb, sizeof(breadcrumb),
			              "FlipTrace: RegisterVideoOutSurfaces done num=%zu", images.size());
			Common::EmergencyLogRaw(breadcrumb);
			Common::LogFatalToFile(breadcrumb);
		}
	}
	if (images.size() != infos.size()) {
		EXIT("video-out texture cache returned an incomplete surface set\n");
	}
	ctx.buffers_sets.push_back({start_index, buffer_num, set_id});
	for (int i = 0; i < buffer_num; i++) {
		auto& dst         = ctx.buffers[i + start_index];
		dst.set_id        = set_id;
		dst.buffer        = addresses[i];
		dst.buffer_size   = infos[i].size;
		dst.buffer_pitch  = infos[i].pitch;
		dst.width         = infos[i].width;
		dst.height        = infos[i].height;
		dst.bytes_per_element = infos[i].bytes_per_element;
		dst.buffer_vulkan = images[static_cast<size_t>(i)];
		ctx.buffer_labels[i + start_index] = 0;
		Libs::Graphics::GuestImageWriteTracker::Track(infos[i].address, infos[i].size);
		LOGF("\tbuffers[%d] = %016" PRIx64 " metadata = %016" PRIx64 " dcc = %08" PRIx32 "\n",
		     i + start_index, reinterpret_cast<uint64_t>(addresses[i]), infos[i].metadata_address,
		     infos[i].dcc_control);
		LOGF("FlipTrace: phase49 register slot=%d set_id=%d buffer=%p vk=%p size=%" PRIu64
		     " pitch=%u %ux%u bpe=%u tile=%u guest_fmt=0x%08" PRIx32 " vk_fmt=%d reuse_vk=%d\n",
		     i + start_index, set_id, addresses[i], static_cast<void*>(dst.buffer_vulkan),
		     infos[i].size, infos[i].pitch, infos[i].width, infos[i].height,
		     infos[i].bytes_per_element, infos[i].tile_mode, infos[i].guest_format,
		     static_cast<int>(infos[i].format), reuse_vk ? 1 : 0);
		fprintf(stderr,
		        "FlipTrace: phase49 register slot=%d buffer=%p vk=%p %ux%u\n", i + start_index,
		        addresses[i], static_cast<void*>(dst.buffer_vulkan), infos[i].width,
		        infos[i].height);
	}

	bool all_vk = true;
	for (int i = 0; i < buffer_num; i++) {
		if (ctx.buffers[start_index + i].buffer == nullptr ||
		    ctx.buffers[start_index + i].buffer_vulkan == nullptr ||
		    ctx.buffers[start_index + i].buffer_size == 0) {
			all_vk = false;
			break;
		}
	}
	if (all_vk) {
		g_phase49_vo_vk_ok.store(true, std::memory_order_release);
	}
	LOGF("FlipTrace: phase49 register set_id=%d start=%d num=%d menu_slots_ok=%d\n", set_id,
	     start_index, buffer_num, all_vk ? 1 : 0);
	fprintf(stderr, "FlipTrace: phase49 register set_id=%d num=%d menu_slots_ok=%d\n", set_id,
	        buffer_num, all_vk ? 1 : 0);

	FlipStats::register_buffers.fetch_add(1, std::memory_order_relaxed);
	LOGF("FlipTrace: RegisterBuffers set_id=%d start=%d num=%d vulkan_ok=%d\n", set_id, start_index,
	     buffer_num, 1);
	FlipStats::Log("register_buffers");
	if (g_phase37_post_unreg.load(std::memory_order_acquire)) {
		g_phase37_guest_reg_seen.store(true, std::memory_order_release);
		g_phase42_reregister_ok.store(true, std::memory_order_release);
		g_phase42_flip_num.store(buffer_num, std::memory_order_release);
		if (!g_phase44_guest_reg2_ok.exchange(true, std::memory_order_acq_rel)) {
			Phase65NoteGuestRegbuf2();
			LOGF("FlipTrace: phase44 guest RegisterBuffers2 set_id=%d start=%d num=%d tid=%d "
			     "(ABI post-Unregister)\n",
			     set_id, start_index, buffer_num, Common::Thread::GetThreadIdUnique());
			fprintf(stderr,
			        "FlipTrace: phase44 guest RegisterBuffers2 set_id=%d num=%d tid=%d\n",
			        set_id, buffer_num, Common::Thread::GetThreadIdUnique());
			if (!infos.empty()) {
				LOGF("FlipTrace: phase46 videoout compression=%d dcc_control=0x%08" PRIx32
				     " metadata=0x%016" PRIx64 "\n",
				     static_cast<int>(infos[0].compression), infos[0].dcc_control,
				     infos[0].metadata_address);
				fprintf(stderr,
				        "FlipTrace: phase46 videoout compression=%d dcc=0x%08" PRIx32 "\n",
				        static_cast<int>(infos[0].compression), infos[0].dcc_control);
			}
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

	{
		char breadcrumb[384];
		std::snprintf(
		    breadcrumb, sizeof(breadcrumb),
		    "FlipTrace: RegisterBuffers2 enter handle=%d set=%d start=%d num=%d attr=%p bufs=%p "
		    "cat=%d opt=%p",
		    handle, set_index, buffer_index_start, buffer_num, static_cast<const void*>(attribute),
		    static_cast<const void*>(buffers), category, option);
		Common::EmergencyLogRaw(breadcrumb);
		Common::LogFatalToFile(breadcrumb);
	}

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

	{
		char breadcrumb[384];
		std::snprintf(breadcrumb, sizeof(breadcrumb),
		              "FlipTrace: RegisterBuffers2 attrs fmt=0x%016" PRIx64
		              " %ux%u pitch=%u tiling=%u dcc=0x%08" PRIx32 " cat=%d",
		              attribute->pixel_format, attribute->width, attribute->height,
		              attribute->pitch_in_pixel, attribute->tiling_mode, attribute->dcc_control,
		              category);
		Common::EmergencyLogRaw(breadcrumb);
		Common::LogFatalToFile(breadcrumb);
	}

	if (set_index < 0 || set_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX ||
	    buffer_index_start < 0 || buffer_index_start >= VIDEO_OUT_BUFFER_NUM_MAX ||
	    buffer_num < 1 || buffer_num > VIDEO_OUT_BUFFER_NUM_MAX ||
	    buffer_index_start + buffer_num > VIDEO_OUT_BUFFER_NUM_MAX) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

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
	    RegisterBuffersInternal(*ctx, set_index, buffer_index_start, addresses.data(), buffer_num,
	                            infos);
	{
		char breadcrumb[192];
		std::snprintf(breadcrumb, sizeof(breadcrumb),
		              "FlipTrace: RegisterBuffers2 leave result=%d set=%d num=%d", reg_result,
		              set_index, buffer_num);
		Common::EmergencyLogRaw(breadcrumb);
		Common::LogFatalToFile(breadcrumb);
	}
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

	{
		char breadcrumb[192];
		std::snprintf(breadcrumb, sizeof(breadcrumb),
		              "FlipTrace: SubmitFlipCPU enter handle=%d index=%d mode=%d arg=%" PRId64,
		              handle, index, flip_mode, flip_arg);
		Common::EmergencyLogRaw(breadcrumb);
		Common::LogFatalToFile(breadcrumb);
	}

	uint64_t  request_id = 0;
	const int result =
	    ReserveFlipRequest(handle, index, flip_mode, flip_arg, FlipRequestSource::Cpu, request_id);
	if (result == VIDEO_OUT_ERROR_INVALID_VALUE) {
		LOGF("\t unsupported flip_mode = %d\n", flip_mode);
	}
	if (result != OK) {
		char breadcrumb[160];
		std::snprintf(breadcrumb, sizeof(breadcrumb),
		              "FlipTrace: SubmitFlipCPU reject handle=%d index=%d result=0x%08x", handle,
		              index, static_cast<unsigned>(result));
		Common::EmergencyLogRaw(breadcrumb);
		Common::LogFatalToFile(breadcrumb);
		return result;
	}
	FlipStats::submit_cpu.fetch_add(1, std::memory_order_relaxed);
	LOGF("FlipTrace: SubmitFlipCPU handle=%d index=%d mode=%d arg=%" PRId64 " id=%" PRIu64 "\n",
	     handle, index, flip_mode, flip_arg, request_id);
	{
		char breadcrumb[192];
		std::snprintf(breadcrumb, sizeof(breadcrumb),
		              "FlipTrace: SubmitFlipCPU ok handle=%d index=%d id=%" PRIu64, handle, index,
		              request_id);
		Common::EmergencyLogRaw(breadcrumb);
		Common::LogFatalToFile(breadcrumb);
	}
	FlipStats::Log("submit_cpu");
	Phase37NoteGuestSubmitFlip(index);
	Graphics::GraphicsRunSubmitFlipPreparation();

	return OK;
}

} // namespace Libs::VideoOut

namespace Libs::Presentation {

int DisplayBufferSubmitFlipFromGpu(Graphics::CommandBuffer& buffer, int handle, int index,
                                   int flip_mode, int64_t flip_arg, uint64_t& request_id) {
	EXIT_IF(VideoOut::g_video_out_context == nullptr || buffer.IsInvalid());

	{
		char breadcrumb[192];
		std::snprintf(breadcrumb, sizeof(breadcrumb),
		              "FlipTrace: SubmitFlipGPU enter handle=%d index=%d mode=%d arg=%" PRId64,
		              handle, index, flip_mode, flip_arg);
		Common::EmergencyLogRaw(breadcrumb);
		Common::LogFatalToFile(breadcrumb);
	}

	const int result = VideoOut::ReserveFlipRequest(
	    handle, index, flip_mode, flip_arg, VideoOut::FlipRequestSource::GpuEop, request_id);
	if (result != OK) {
		char breadcrumb[160];
		std::snprintf(breadcrumb, sizeof(breadcrumb),
		              "FlipTrace: SubmitFlipGPU reject handle=%d index=%d result=0x%08x", handle,
		              index, static_cast<unsigned>(result));
		Common::EmergencyLogRaw(breadcrumb);
		Common::LogFatalToFile(breadcrumb);
		return result;
	}
	VideoOut::FlipStats::submit_gpu.fetch_add(1, std::memory_order_relaxed);
	LOGF("FlipTrace: SubmitFlipGPU handle=%d index=%d mode=%d arg=%" PRId64 " id=%" PRIu64 "\n",
	     handle, index, flip_mode, flip_arg, request_id);
	{
		char breadcrumb[160];
		std::snprintf(breadcrumb, sizeof(breadcrumb),
		              "FlipTrace: SubmitFlipGPU ok handle=%d index=%d id=%" PRIu64, handle, index,
		              request_id);
		Common::EmergencyLogRaw(breadcrumb);
		Common::LogFatalToFile(breadcrumb);
	}
	VideoOut::FlipStats::Log("submit_gpu");
	VideoOut::Phase50NoteSubmitGpu(
	    handle, index, request_id,
	    VideoOut::FlipStats::submit_gpu.load(std::memory_order_relaxed));
	VideoOut::Phase37NoteGuestSubmitFlip(index);
	VideoOut::g_video_out_context->GetFlipQueue().Prepare(request_id, buffer);

	return OK;
}

uint64_t DisplayBufferPrepareNextFlipOnGpu(Graphics::CommandBuffer& buffer) {
	EXIT_IF(VideoOut::g_video_out_context == nullptr);
	return VideoOut::g_video_out_context->GetFlipQueue().PrepareNextCpu(buffer);
}

void DisplayBufferCompleteFlipFromGpu(uint64_t request_id) {
	EXIT_IF(VideoOut::g_video_out_context == nullptr);
	VideoOut::g_video_out_context->GetFlipQueue().Complete(request_id);
}

void DisplayBufferWaitForFlipQueueSlot() {
	EXIT_IF(VideoOut::g_video_out_context == nullptr);
	VideoOut::g_video_out_context->GetFlipQueue().WaitForSubmitSlot();
}

} // namespace Libs::Presentation

namespace Libs::VideoOut {

void VideoOutWaitFlipDone(int handle, int index) {
	EXIT_IF(g_video_out_context == nullptr);

	auto* ctx = g_video_out_context->Get(handle);
	EXIT_IF(ctx == nullptr);

	EXIT_NOT_IMPLEMENTED(!IsValidBufferIndex(index));
	g_video_out_context->GetFlipQueue().Wait(*ctx, index);
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

	g_video_out_context->GetFlipQueue().GetFlipStatus(*ctx, *status);

	const auto presented = FlipStats::presented.load(std::memory_order_relaxed);
	LOGF("\t count = %" PRIu64 "\n"
	     "\t processTime = %" PRIu64 "\n"
	     "\t processTimeCounter = %" PRIu64 "\n"
	     "\t submitProcessTimeCounter = %" PRIu64 "\n"
	     "\t flipArg = %" PRId64 "\n"
	     "\t gcQueueNum = %d\n"
	     "\t flipPendingNum = %d\n"
	     "\t currentBuffer = %d\n",
	     status->count, status->processTime, status->processTimeCounter,
	     status->submitProcessTimeCounter, status->flipArg, status->gcQueueNum,
	     status->flipPendingNum, status->currentBuffer);
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
		static std::atomic<uint32_t> p49_status_n {0};
		const uint32_t               sn = p49_status_n.fetch_add(1, std::memory_order_relaxed);
		if (sn < 32 || (sn % 120) == 0) {
			LOGF("FlipTrace: phase49 status count=%" PRIu64 " currentBuffer=%d pending=%d "
			     "gcQueue=%d presented=%" PRIu64 " submit_cpu=%" PRIu64 " submit_gpu=%" PRIu64
			     " nonzero=%d vo_vk=%d\n",
			     status->count, status->currentBuffer, status->flipPendingNum, status->gcQueueNum,
			     presented, FlipStats::submit_cpu.load(std::memory_order_relaxed),
			     FlipStats::submit_gpu.load(std::memory_order_relaxed),
			     g_phase48_content_nonzero.load(std::memory_order_relaxed) ? 1 : 0,
			     g_phase49_vo_vk_ok.load(std::memory_order_relaxed) ? 1 : 0);
			fprintf(stderr,
			        "FlipTrace: phase49 status count=%" PRIu64 " cur=%d pending=%d nonzero=%d\n",
			        status->count, status->currentBuffer, status->flipPendingNum,
			        g_phase48_content_nonzero.load(std::memory_order_relaxed) ? 1 : 0);
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
	g_video_out_context->GetFlipQueue().GetFlipStatus(*ctx, status);

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
