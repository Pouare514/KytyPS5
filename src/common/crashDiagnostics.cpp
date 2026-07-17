#include "common/crashDiagnostics.h"

#include "common/common.h"

#include <cstdio>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#endif

namespace Common {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

using SecerrHandlerFunc = void(__cdecl*)(int, void*);
using SetSecerrHandlerFunc = SecerrHandlerFunc(__cdecl*)(SecerrHandlerFunc);

static void SecurityFailureHandler(int code, void* return_address) {
	std::fprintf(stderr, "Security failure: code=%d, retaddr=%p\n", code, return_address);
	std::fflush(stderr);
}

#endif

void InstallCrashDiagnostics() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const char* const modules[] = {"ucrtbase.dll", "vcruntime140.dll", "msvcrt.dll"};
	for (const char* module_name : modules) {
		const HMODULE module = GetModuleHandleA(module_name);
		if (module == nullptr) {
			continue;
		}
		const auto set_handler = reinterpret_cast<SetSecerrHandlerFunc>(
		    GetProcAddress(module, "_set_security_error_handler"));
		if (set_handler != nullptr) {
			(void)set_handler(&SecurityFailureHandler);
			return;
		}
	}
#endif
}

} // namespace Common
