#include "common/crashDiagnostics.h"

#include "common/common.h"
#include "common/fatalLog.h"

#include <cinttypes>
#include <cstdio>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#endif

namespace Common {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

using SecerrHandlerFunc = void(__cdecl*)(int, void*);
using SetSecerrHandlerFunc = SecerrHandlerFunc(__cdecl*)(SecerrHandlerFunc);

static void SecurityFailureHandler(int code, void* return_address) {
	char module_name[MAX_PATH] = {};
	if (return_address != nullptr) {
		HMODULE owner_module = nullptr;
		if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                       static_cast<LPCSTR>(return_address), &owner_module) != 0 &&
		    owner_module != nullptr) {
			(void)GetModuleFileNameA(owner_module, module_name, MAX_PATH);
		}
	}

	char message[512];
	std::snprintf(message, sizeof(message),
	              "Security failure: code=%d, retaddr=%p, module=%s", code, return_address,
	              module_name[0] != '\0' ? module_name : "(unknown)");
	LogFatalToFile(message);
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
