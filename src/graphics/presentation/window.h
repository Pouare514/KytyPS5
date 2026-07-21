#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

namespace Libs::Graphics {

class CommandBuffer;
struct VideoOutVulkanImage;
struct PreparedFrame;

void           WindowInit(uint32_t width, uint32_t height);
void           WindowRun();
PreparedFrame& WindowPrepareFrame(CommandBuffer& buffer, VideoOutVulkanImage& image);
PreparedFrame& WindowPrepareBlankFrame(CommandBuffer& buffer, uint32_t width, uint32_t height,
                                       bool opaque);
void           WindowPresentFrame(PreparedFrame& frame);
// Phase 29: show SDL window early so a black surface is visible before the first flip.
void           WindowEnsureVisible();
// Pump SDL events while holding the window visible (avoids "Not Responding").
void           WindowHoldVisible(uint32_t seconds);
// Phase 32/33: after pending→0, ignore SDL_QUIT for N seconds so a stray close
// does not _Exit(0) before MainThread can RegisterBuffers / SubmitDcb.
void           WindowArmIgnoreQuit(uint32_t seconds);
void           WindowDisarmIgnoreQuit();
bool           WindowShouldIgnoreQuit();
// Phase 35: arm FiberSwitch soft-ACK only after Unregister (not during BootCards).
void           WindowArmPhase35FiberSoftAck();
bool           WindowPhase35FiberSoftAckArmed();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */
