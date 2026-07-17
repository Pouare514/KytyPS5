#include "common/common.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

// Prefer discrete GPU on hybrid laptops (NVIDIA Optimus / AMD Switchable Graphics).
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement                 = 0x00000001;
__declspec(dllexport) unsigned long AmdPowerXpressRequestHighPerformance = 0x00000001;
}

#endif
