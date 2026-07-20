#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_GUESTIMAGEWRITETRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_GUESTIMAGEWRITETRACKER_H_

#include "common/common.h"

#include <cstdint>

namespace Libs::Graphics {

// Phase 48 (SharpEmu #447): monotonic write generation for CPU-rewritten guest images.
// Survives dirty-flag consume so VideoOut / texture cache can force a fresh upload.
class GuestImageWriteTracker {
public:
	static void Track(uint64_t address, uint64_t size);
	static void Untrack(uint64_t address);
	static void NoteCpuWrite(uint64_t address, uint64_t size);
	[[nodiscard]] static bool TryGetWriteGeneration(uint64_t address, int64_t* generation);
	static void RecordUploadedGeneration(uint64_t address, int64_t generation);
	[[nodiscard]] static bool NeedsReupload(uint64_t address);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_GUESTIMAGEWRITETRACKER_H_
