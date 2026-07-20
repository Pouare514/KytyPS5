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
