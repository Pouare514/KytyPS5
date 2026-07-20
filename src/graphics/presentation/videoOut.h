#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_VIDEOOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_VIDEOOUT_H_

#include "common/common.h"
// #include "common/subsystems.h"

#include "common/abi.h"
#include "kernel/eventQueue.h"

namespace Libs::Graphics {
// struct VulkanSwapchain;
struct VideoOutVulkanImage;
} // namespace Libs::Graphics

namespace Libs::VideoOut {

struct VideoOutBufferAttribute2;
struct VideoOutFlipStatus;
struct VideoOutVblankStatus;
struct VideoOutVrrStatus;
struct VideoOutOutputStatus;
struct VideoOutOutputOptions;
struct VideoOutBuffers;
struct VideoOutColorSettings;

void VideoOutInit(uint32_t width, uint32_t height);
void VideoOutWaitFlipDone(int handle, int index);
// Phase 35/36: Mixed host flip pump (black NOP DCB) — opt-in KYTY_PHASE35_HOST_FLIP=1.
void Phase35ArmGuestMenuAfterUnregister();
void Phase35TryGuestMenuFromSubmissionThread(const char* thread_name);
// Phase 38/42: soft-idle waits on guest SubmitFlip post-Unregister (Register alone is not enough).
// Phase 39: MainThread does not park until progress is seen (no timeout fallback).
// Phase 43: park CRT divert on sustained presented frames (menu_frames_ok).
bool Phase38GuestBootProgressSeen();
void Phase38NudgeBootWorkers();
// Phase 41/42: MainThread anti-CRT handoff (re-Register + SubmitFlip + NdJob rearm).
void Phase41MenuHandoffAttempt();
// Phase 44: true VideoOutRegisterBuffers2 ABI post-Unregister (not snapshot).
bool Phase44GuestRegisterBuffers2Seen();
// Phase 45: NdJob-path submit_dcb post-Unregister (not MainThread P43 seed).
bool Phase45NdJobDcbSeen();
void Phase45OnNdJobSyncTimeout(LibKernel::EventQueue::KernelEqueue eq);
void Phase45NoteSubmitDcb(uint64_t submit_count);
// Phase 46: real (non-NOP) DCB + native PM4 EOP post-Unregister.
bool Phase46RealDcbSeen();
bool Phase46NativeEopSeen();
bool Phase46ReadyForSoftIdle();
void Phase46InspectSubmitDcb(uint32_t queue, const uint32_t* dcb, uint32_t size_in_dwords);
void Phase46NoteNativeEop(const char* source);
// Phase 47: guest DRAW/DISPATCH DCB (excludes host seed).
bool Phase47GuestDrawSeen();
bool Phase47ReadyForSoftIdle();
void Phase47PostSeedWake(const char* why);
void Phase47NoteSubmitAcb(uint64_t submit_count);
// Phase 48: present content classification.
bool Phase48ContentNonZeroSeen();
bool Phase48FloatFlipSeen();
// Phase 50: NdJob *obj / wake / batch / GPU flip correlation (no trampoline logging).
void Phase50PollKeep1Obj(const char* why);
void Phase50NoteWake(const char* name, size_t woken);
void Phase50NoteNdJobBatch(int index, int total, uint64_t ident, int filter, uint64_t data);
void Phase50NoteSubmitGpu(int handle, int index, uint64_t request_id, uint64_t submit_gpu_total);
bool Phase50ObjNonZeroSeen();
// Phase 51: NdJob producer — struct dump / EQ / fiber / failfast / bypass flip.
void Phase51DumpNdJobStruct(const char* why);
void Phase51NoteEqDelivery(const char* eq_name, uint64_t ident, int filter, uint64_t data,
                           void* udata);
bool Phase51GraphicsIdent0IsNotCompletion(uint64_t ident, int filter);
void Phase51NoteFiber(const char* op, const char* name, const char* path);
bool Phase51ShouldSkipFiberSoftAck(const char* name);
void Phase51CheckWorkerFailfast(const char* why);
void Phase51TryBypassFlipL0(const char* why);
void Phase52NoteAfterDump(const char* why);
void Phase53ProbeRetarget(const char* why);
void Phase53ScanObjDeep(const char* why);
void Phase53DumpStatusRdi(const char* why);
void Phase53DumpUserAnchors(const char* why);
bool Phase53RealQueueSeen();
// Phase 54: producer Mixed enqueue (cond map / cycle / submit kind / wake budget / fake job).
bool     Phase37PostUnregisterSeen();
uint64_t Phase54CurrentCycleId();
uint64_t Phase54BumpCycle(const char* why);
void     Phase54NoteHostWake(const char* why);
[[nodiscard]] bool Phase54AllowHostWake();
void     Phase54NoteMixedWake(const char* role, uintptr_t cond_ptr, const char* outcome);
void     Phase54NoteMixedLeave(const char* role);
void     Phase54NoteSubmit(uint32_t queue, const uint32_t* dcb, uint32_t size_in_dwords);
void     Phase54TryFakeJobAfterWake(const char* role);
[[nodiscard]] bool Phase54FakeJobEnabled();
// Phase 55: Mixed entry thunk + queue layout + watch + fake queue inject.
constexpr uint64_t kPhase55MixedEntry = 0x0000000901DE4140ULL;
void Phase55TryArmMixedThunk();
void Phase55OnMixedRewait(const char* role, uintptr_t cond_ptr);
void Phase55NoteMainWakeAlt(const char* kind, uint64_t a0, uint64_t a1);
void Phase55FlushDeferredLogs();
void Phase55PollWatch();
void Phase55EmitHeatmap(const char* why);
void Phase55NoteGuestCond(uint64_t guest_cond_va, uint64_t guest_arg, const char* role);
[[nodiscard]] bool Phase55FakeQueueEnabled();
// Phase 56: LIST_CANDIDATE retarget + writers + FAKE_COUNT.
void Phase56NoteGuestSync(uint64_t guest_cond_va, uint64_t guest_mutex_va, uint64_t guest_arg,
                          const char* role);
void Phase56NoteMainSignal(uint64_t guest_cond_va, const char* role);
void Phase56PollWatch();
void Phase56EmitHeatmap(const char* why);
void Phase56TryFakeCount(const char* why);
[[nodiscard]] bool Phase56FakeCountEnabled();
uint64_t Phase56CurrentSyncId();
uint64_t Phase56QueueBase();
// Phase 57: global queue (A) + Main producer (C).
void Phase57TryScanMixedBody();
void Phase57NoteMixedRegs(uint64_t rdi, uint64_t rsi, uint64_t rdx);
void Phase57NoteMainAgcTouch(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3);
void Phase57NoteMainObjectWrite(uint64_t guest_va, const char* why);
void Phase57PollHeatmap();
void Phase57EmitHeatmap(const char* why);
uint64_t Phase57QueueBase();
[[nodiscard]] bool Phase57Elected();
// Phase 58: NdJob ancre + subblock classify + discriminant heatmap.
void Phase58NoteNdJobAncre(const char* why);
void Phase58NoteMainAgcCross(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3);
void Phase58PollWatch();
void Phase58EmitHeatmap(const char* why);
uint64_t Phase58QueueBase();
[[nodiscard]] bool Phase58Elected();
// Phase 59: AGC guest VA ↔ host queue/stream/ctx mapping (instrumentation only).
void Phase59NoteGuestVa(const char* kind, uint64_t va, uint64_t host_id, const char* source_tag);
void Phase59NoteSubmit(uint32_t queue, const uint32_t* dcb, uint32_t size_in_dwords,
                       const char* submit_kind);
void Phase59NoteWorkloadStream(uint32_t stream_id, const void* stream);
void Phase59NoteWorkloadActive(uint32_t stream_id, const char* why);
void Phase59NoteStubArgs(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3);
void Phase59NoteEq(int id, uint64_t eq_va, uint32_t context_id, const char* why);
void Phase59SeedNdJobAnchors(uint64_t ndjob, uint64_t status, uint64_t user10, uint64_t user40);
void Phase59EmitHeatmap(const char* why);
void Phase59Poll();
// Phase 61: unlock gate — read-only user_ring PM4/ptr probe.
void Phase61RingProbe();
void Phase61EmitHeatmap(const char* why);
void Phase61Poll();
// Phase 62: producer silence post-Unreg (RegisterResource map; not KPRI ring gate).
void Phase62NoteRegisterResource(uint64_t addr, uint64_t size, int res_type, const char* name,
                                 uint32_t handle);
void Phase62NoteUnreg();
void Phase62EmitHeatmap(const char* why);
void Phase62Poll();
// Phase 63: Unreg forensics — Owner cycle_id + SubmitEntry (read-only).
void Phase63NoteUnregister(uint32_t owner, int n_resources, const char* why, bool owner_event);
void Phase63NoteSubmitEntry(const char* api, uint32_t queue, const uint32_t* dcb,
                            uint32_t size_in_dwords);
void Phase63NotePostAgc(const char* why);
void Phase63EmitHeatmap(const char* why);
void Phase63Poll();
// Phase 64: post-Unreg waiters (Main cond / flip stuck / NdJob static) — read-only.
void Phase64NoteMainCondWait(uint64_t cond_va);
void Phase64NoteMainCondSignal(uint64_t cond_va, bool match_wait);
void Phase64EmitHeatmap(const char* why);
void Phase64Poll();
// Phase 65: who waits / menu entry markers post-Unreg — read-only.
void Phase65NoteCondWait(const char* role, const char* name, int tid, uint64_t ra);
void Phase65NoteGuestRegbuf2();
void Phase65NoteGuestFlip();
void Phase65EmitHeatmap(const char* why);
void Phase65Poll();
// Phase 66: recycle Flip L0 after menu detection (opt-in KYTY_PHASE66_MENU_RECYCLE).
void Phase66EmitHeatmap(const char* why);
void Phase66Poll();

KYTY_SYSV_ABI int  VideoOutOpen(int user_id, int bus_type, int index, const void* param);
KYTY_SYSV_ABI int  VideoOutClose(int handle);
KYTY_SYSV_ABI void VideoOutSetBufferAttribute2(VideoOutBufferAttribute2* attribute,
                                               uint64_t pixel_format, uint32_t tiling_mode,
                                               uint32_t width, uint32_t height, uint64_t option,
                                               uint32_t dcc_control,
                                               uint64_t dcc_cb_register_clear_color);
KYTY_SYSV_ABI int  VideoOutSetFlipRate(int handle, int rate);
KYTY_SYSV_ABI int  VideoOutAddFlipEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                        void* udata);
KYTY_SYSV_ABI int  VideoOutAddVblankEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                          void* udata);
KYTY_SYSV_ABI int VideoOutAddPreVblankStartEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                                 void* udata);
KYTY_SYSV_ABI int VideoOutAddOutputModeEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                             void* udata);
KYTY_SYSV_ABI int VideoOutDeleteFlipEvent(LibKernel::EventQueue::KernelEqueue eq, int handle);
KYTY_SYSV_ABI int VideoOutDeleteVblankEvent(LibKernel::EventQueue::KernelEqueue eq, int handle);
KYTY_SYSV_ABI int VideoOutDeletePreVblankStartEvent(LibKernel::EventQueue::KernelEqueue eq,
                                                    int                                 handle);
KYTY_SYSV_ABI int VideoOutRegisterBuffers2(int handle, int set_index, int buffer_index_start,
                                           const VideoOutBuffers* buffers, int buffer_num,
                                           const VideoOutBufferAttribute2* attribute, int category,
                                           void* option);
KYTY_SYSV_ABI int VideoOutSubmitChangeBufferAttribute2(int handle, int set_index,
                                                       const VideoOutBufferAttribute2* attribute,
                                                       void*                           option);
KYTY_SYSV_ABI int VideoOutUnregisterBuffers(int handle, int set_index);
KYTY_SYSV_ABI int VideoOutSubmitFlip(int handle, int index, int flip_mode, int64_t flip_arg);
KYTY_SYSV_ABI int VideoOutGetFlipStatus(int handle, VideoOutFlipStatus* status);
KYTY_SYSV_ABI int VideoOutIsFlipPending(int handle);
KYTY_SYSV_ABI int VideoOutGetVblankStatus(int handle, VideoOutVblankStatus* status);
KYTY_SYSV_ABI int VideoOutGetEventId(const LibKernel::EventQueue::KernelEvent* ev);
KYTY_SYSV_ABI int VideoOutGetEventData(const LibKernel::EventQueue::KernelEvent* ev, int64_t* data);
KYTY_SYSV_ABI int VideoOutGetEventCount(const LibKernel::EventQueue::KernelEvent* ev);
KYTY_SYSV_ABI int VideoOutWaitVblank(int handle);
KYTY_SYSV_ABI int VideoOutGetOutputStatus(int handle, VideoOutOutputStatus* status);
KYTY_SYSV_ABI int VideoOutInitializeOutputOptions(VideoOutOutputOptions* options);
KYTY_SYSV_ABI int VideoOutIsOutputSupported(int handle, uint64_t mode,
                                            const VideoOutOutputOptions* options,
                                            void* reserved_ptr, uint64_t reserved);
KYTY_SYSV_ABI int VideoOutConfigureOutput(int handle, uint64_t mode,
                                          const VideoOutOutputOptions* options, void* reserved_ptr,
                                          uint64_t reserved);
KYTY_SYSV_ABI int VideoOutSetWindowModeMargins(int handle, int top, int bottom);
KYTY_SYSV_ABI int VideoOutLatencyControlWaitBeforeInput(int handle);
KYTY_SYSV_ABI int VideoOutLatencyMeasureSetStartPoint(int handle, uint32_t point);
KYTY_SYSV_ABI int VideoOutColorSettingsSetGamma(VideoOutColorSettings* settings, float gamma);
KYTY_SYSV_ABI int VideoOutAdjustColor(int handle, const VideoOutColorSettings* settings);
// sceVideoOutAddVrrStatusFlagsPrivilege — always OK (Phase 20); wrong ABI caused 0x8029000B assert.
KYTY_SYSV_ABI int VideoOutAddVrrStatusFlagsPrivilege(int handle, uint32_t flags, uint64_t arg2,
                                                      uint64_t arg3);
KYTY_SYSV_ABI int VideoOutGetVrrStatus(int handle, void* status);
KYTY_SYSV_ABI int VideoOutGetBufferLabelAddress(int handle, uintptr_t* label_addr);
KYTY_SYSV_ABI int VideoOutGetDeviceCapabilityInfo(int handle, void* info);
KYTY_SYSV_ABI int VideoOutGetResolutionStatus(int handle, void* status);
KYTY_SYSV_ABI int VideoOutUnknownNidStub0(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3);
KYTY_SYSV_ABI int VideoOutUnknownNidStub1(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3);

void VideoOutBeginVblank();
void VideoOutEndVblank();
bool VideoOutFlipWindow(uint32_t micros);

} // namespace Libs::VideoOut

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_VIDEOOUT_H_ */
