#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"

namespace Libs::Graphics {

class CommandProcessor;

class GraphicsRunSubmissionLock final {
public:
	GraphicsRunSubmissionLock();
	~GraphicsRunSubmissionLock();
	KYTY_CLASS_NO_COPY(GraphicsRunSubmissionLock);
};

void GraphicsRunInit();

void GraphicsRunSubmit(uint32_t* cmd_draw_buffer, uint32_t num_draw_dw, uint32_t* cmd_const_buffer,
                       uint32_t num_const_dw, bool trigger_agc_interrupt_on_done = false);
void GraphicsRunSubmitCompute(uint32_t queue, uint32_t* cmd_buffer, uint32_t num_dw,
                              bool trigger_agc_interrupt_on_done = false);
void GraphicsRunSubmitFlipPreparation();
void GraphicsRunWait();
void GraphicsRunDone();
int  GraphicsRunGetFrameNum();
[[nodiscard]] bool              GraphicsRunGpuIsReady() noexcept;
[[nodiscard]] bool              GraphicsRunIsCommandProcessorThread() noexcept;
[[nodiscard]] CommandProcessor* GraphicsRunCurrentCommandProcessor() noexcept;
void                            GraphicsRunFinishCommandProcessors();
[[nodiscard]] bool              GraphicsRunSubmissionLockHeld() noexcept;
[[nodiscard]] bool              GraphicsRunGpuLockHeld() noexcept;

// Phase 68: SharpEmu-like submit-state + WAIT_REG_MEM ack (KYTY_PHASE68_SHARPEMU_WAIT=1).
namespace Phase68 {
[[nodiscard]] bool Enabled();
void               NoteSubmit(uint32_t queue, const uint32_t* dcb, uint32_t dwords, const char* api);
[[nodiscard]] bool TryAckWaitRegMem(uint64_t addr);
} // namespace Phase68

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
