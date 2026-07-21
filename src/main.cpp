#include "common/common.h"
#include "common/crashDiagnostics.h"
#include "common/commonSubsystem.h"
#include "common/dateTime.h"
#include "common/debug.h"
#include "common/emulatorConfig.h"
#include "common/fatalLog.h"
#include "common/file.h"
#include "common/magicEnum.h"
#include "common/platform/sysDbg.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/vulkanLayerWorkaround.h"
#include "emulator.h"
#include "kytyGitVersion.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <fmt/format.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <intrin.h>
#include <windows.h>
#endif

using namespace Common;
using namespace Emulator;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// CFG stubs sanitize RCX so allow-all never calls null / guest-stack / non-X memory.
// Replace PE guard slots only — never patch ntdll bodies (caused RIP=0 crashes).

extern "C" void KytyCfgSafeCallTarget() {}

// Park only the current thread (CFG walker runaway). Do not TerminateProcess — GPU
// producer threads must keep running.
extern "C" void KytyParkThreadForever() {
	for (;;) {
		Sleep(1000);
	}
}

static bool KytyHostCodeIsExecutable(uint64_t addr) {
	if (addr < 0x10000ull) {
		return false;
	}
	MEMORY_BASIC_INFORMATION mbi {};
	if (VirtualQuery(reinterpret_cast<const void*>(addr), &mbi, sizeof(mbi)) == 0 ||
	    mbi.State != MEM_COMMIT) {
		return false;
	}
	const DWORD prot = mbi.Protect & 0xffu;
	return prot == PAGE_EXECUTE || prot == PAGE_EXECUTE_READ || prot == PAGE_EXECUTE_READWRITE ||
	       prot == PAGE_EXECUTE_WRITECOPY;
}

static bool KytyCfgCallerWillInvokeRcx(uint64_t return_addr) {
	if (return_addr < 0x10000ull || !KytyHostCodeIsExecutable(return_addr)) {
		return false;
	}
	const auto* p = reinterpret_cast<const uint8_t*>(return_addr);
	return p[0] == 0xff && (p[1] == 0xd1 || p[1] == 0xe1);
}

// target in RCX, CFG caller's resume address in RDX (passed by naked stubs).
extern "C" uint64_t KytyCfgSanitizeTarget(uint64_t target, uint64_t caller_ret) {
	// Walkers validate RAs without calling them — leave RCX intact (Exp B / DS Main freeze).
	if (caller_ret != 0 && !KytyCfgCallerWillInvokeRcx(caller_ret)) {
		return target;
	}
	if (!KytyHostCodeIsExecutable(target)) {
		return reinterpret_cast<uint64_t>(&KytyCfgSafeCallTarget);
	}
	return target;
}

#if defined(__clang__)
__attribute__((naked)) extern "C" void KytyCfgAllowAll() {
	__asm__ volatile(
	    "subq $0x28, %rsp\n"
	    "movq 0x28(%rsp), %rdx\n" // CFG caller's return address
	    "callq KytyCfgSanitizeTarget\n"
	    "movq %rax, %rcx\n"
	    "addq $0x28, %rsp\n"
	    "ret\n");
}

__attribute__((naked)) extern "C" void KytyCfgDispatchAll() {
	__asm__ volatile(
	    "subq $0x28, %rsp\n"
	    // Force invoke path: rdx=0 → Sanitize always retargets non-exec.
	    "xorl %edx, %edx\n"
	    "callq KytyCfgSanitizeTarget\n"
	    "addq $0x28, %rsp\n"
	    "jmpq *%rax\n");
}
#else
extern "C" void KytyCfgAllowAll() {}
extern "C" void KytyCfgDispatchAll() {}
#endif

static void DisableHostControlFlowGuard() {
	// PE load-config dir.Size is often 0x140 while current SDK sizeof(...) is larger
	// (GuardMemcpy/Uma). Requiring full sizeof caused a silent early-return — CFG stayed on.
	constexpr DWORD kMinLoadConfigForCfg = 0x80; // through GuardCFDispatchFunctionPointer
	auto* const     module = reinterpret_cast<uint8_t*>(GetModuleHandleW(nullptr));
	int             patched_slots = 0;
	ULONGLONG       check_slot    = 0;
	ULONGLONG       dispatch_slot = 0;
	DWORD           guard_flags   = 0;
	const char*     status        = "ok";

	auto write_log = [&]() {
		wchar_t exe_path[MAX_PATH] = {};
		GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
		std::wstring log_path(exe_path);
		const auto   slash = log_path.find_last_of(L"\\/");
		if (slash != std::wstring::npos) {
			log_path.resize(slash + 1);
		}
		log_path += L"_kyty_cfg_disable.txt";
		FILE* f = nullptr;
		_wfopen_s(&f, log_path.c_str(), L"w");
		if (f == nullptr) {
			fopen_s(&f, "_kyty_cfg_disable.txt", "w");
		}
		if (f != nullptr) {
			std::fprintf(f,
			             "status=%s slots=%d check_slot=0x%llx dispatch_slot=0x%llx flags=0x%lx "
			             "expect_check=%p expect_dispatch=%p\n",
			             status, patched_slots, static_cast<unsigned long long>(check_slot),
			             static_cast<unsigned long long>(dispatch_slot),
			             static_cast<unsigned long>(guard_flags),
			             reinterpret_cast<void*>(&KytyCfgAllowAll),
			             reinterpret_cast<void*>(&KytyCfgDispatchAll));
			if (check_slot != 0) {
				auto** slot = reinterpret_cast<void**>(static_cast<uintptr_t>(check_slot));
				std::fprintf(f, "check_fn_now=%p\n", slot != nullptr ? *slot : nullptr);
			}
			std::fclose(f);
		}
	};

	if (module == nullptr) {
		status = "no_module";
		write_log();
		return;
	}
	const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(module);
	if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
		status = "bad_dos";
		write_log();
		return;
	}
	const auto* nt =
	    reinterpret_cast<const IMAGE_NT_HEADERS64*>(module + static_cast<uint32_t>(dos->e_lfanew));
	if (nt->Signature != IMAGE_NT_SIGNATURE) {
		status = "bad_nt";
		write_log();
		return;
	}
	const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG];
	if (dir.VirtualAddress == 0 || dir.Size < kMinLoadConfigForCfg) {
		status = "no_load_config";
		write_log();
		return;
	}

	// Read only fields present in dir.Size — do not require full SDK struct.
	auto*       lc_bytes = module + dir.VirtualAddress;
	const DWORD lc_size  = dir.Size;
	auto read_u64 = [&](DWORD off) -> ULONGLONG {
		if (off + sizeof(ULONGLONG) > lc_size) {
			return 0;
		}
		ULONGLONG v = 0;
		std::memcpy(&v, lc_bytes + off, sizeof(v));
		return v;
	};
	auto read_u32 = [&](DWORD off) -> DWORD {
		if (off + sizeof(DWORD) > lc_size) {
			return 0;
		}
		DWORD v = 0;
		std::memcpy(&v, lc_bytes + off, sizeof(v));
		return v;
	};
	auto write_u32 = [&](DWORD off, DWORD v) {
		if (off + sizeof(DWORD) > lc_size) {
			return;
		}
		DWORD old = 0;
		if (VirtualProtect(lc_bytes + off, sizeof(DWORD), PAGE_READWRITE, &old) != 0) {
			std::memcpy(lc_bytes + off, &v, sizeof(v));
			VirtualProtect(lc_bytes + off, sizeof(DWORD), old, &old);
		}
	};

	check_slot                   = read_u64(0x70);
	dispatch_slot                = read_u64(0x78);
	guard_flags                  = read_u32(0x90);
	const ULONGLONG xfg_check    = read_u64(0x118);
	const ULONGLONG xfg_dispatch = read_u64(0x120);

	auto patch_ntdll_check_to_stub = [&](ULONGLONG slot_va) {
		if (slot_va == 0) {
			return;
		}
		auto** slot = reinterpret_cast<void**>(static_cast<uintptr_t>(slot_va));
		if (slot == nullptr || *slot == nullptr) {
			return;
		}
		HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
		HMODULE owner = nullptr;
		if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
		                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                       reinterpret_cast<LPCSTR>(*slot), &owner) == 0 ||
		    owner == nullptr || owner != ntdll) {
			return;
		}
		// Absolute jmp: FF 25 00 00 00 00 + imm64 (rel32 cannot reach our image from ntdll).
		auto*             start = reinterpret_cast<uint8_t*>(*slot);
		constexpr size_t  kLen  = 14;
		uint8_t           buf[kLen] = {0xff, 0x25, 0x00, 0x00, 0x00, 0x00};
		const uint64_t    dest      = reinterpret_cast<uint64_t>(&KytyCfgAllowAll);
		std::memcpy(buf + 6, &dest, sizeof(dest));
		DWORD old = 0;
		if (VirtualProtect(start, kLen, PAGE_EXECUTE_READWRITE, &old) != 0) {
			std::memcpy(start, buf, kLen);
			FlushInstructionCache(GetCurrentProcess(), start, kLen);
			VirtualProtect(start, kLen, old, &old);
		}
	};
	// Redirect ntdll check through our sanitizer (other modules still call ntdll).
	patch_ntdll_check_to_stub(check_slot);
	patch_ntdll_check_to_stub(xfg_check);

	auto patch_slot = [&](ULONGLONG slot_va, void* stub) {
		if (slot_va == 0) {
			return;
		}
		auto** slot = reinterpret_cast<void**>(static_cast<uintptr_t>(slot_va));
		DWORD  old  = 0;
		if (VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old) != 0) {
			*slot = stub;
			VirtualProtect(slot, sizeof(void*), old, &old);
			++patched_slots;
		}
	};
	patch_slot(check_slot, reinterpret_cast<void*>(&KytyCfgAllowAll));
	patch_slot(dispatch_slot, reinterpret_cast<void*>(&KytyCfgDispatchAll));
	patch_slot(xfg_check, reinterpret_cast<void*>(&KytyCfgAllowAll));
	patch_slot(xfg_dispatch, reinterpret_cast<void*>(&KytyCfgDispatchAll));

	guard_flags &=
	    ~(0x100u | 0x200u | 0x400u | 0x4000u | 0x8000u | 0x10000u | 0x20000u | 0x40000u);
	write_u32(0x90, guard_flags);
	write_log();
	::printf("Kyty: host CFG disabled (slots=%d)\n", patched_slots);
}
#endif

static std::string GetBuildString() {
	Date date = Date::FromMacros(std::string(__DATE__));

#if KYTY_BUILD == KYTY_BUILD_DEBUG
	std::string type = "Debug";
#elif KYTY_BUILD == KYTY_BUILD_RELEASE
	std::string type = "Release";
#else
	std::string type = "????";
#endif

	std::string compiler =
	    Debug::GetCompiler() + "-" + Debug::GetLinker() + "-" + Debug::GetBitness();

	std::string str =
	    fmt::format("{}, {}, ver = {}, git = {}, date = {}", type.c_str(), compiler.c_str(),
	                KYTY_VERSION, KYTY_GIT_VERSION, date.ToString().c_str());

	return str;
}

static void PrintUsage() {
	::printf("%s\n", GetBuildString().c_str());
	::printf("kyty_emulator --game <dir|elf> [options]\n\n");
	::printf("Options:\n");
	::printf("  --game <dir|elf>                     Game directory or ELF to load.\n");
	::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
	::printf("  --screen-height <num>                Window height. Default: 720.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --vulkan-validation <true|false>     Enable Vulkan validation.\n");
	::printf("  --shader-validation <true|false>     Enable shader validation.\n");
	::printf("  --shader-optimization-type <value>   None, Size, or Performance.\n");
	::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
	::printf("  --shader-log-folder <path>           Shader log output folder.\n");
	::printf("  --command-buffer-dump <true|false>   Enable command buffer dumps.\n");
	::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
	::printf("  --graphics-debug-dump <true|false>   Enable graphics debug dumps.\n");
	::printf("  --printf-direction <value>           Silent, Console, or File.\n");
	::printf("  --printf-output-file <path>          Guest printf output file.\n");
	::printf("  --profiler-direction <value>         None or Network.\n");
	::printf("  --spirv-debug-printf <true|false>    Enable SPIR-V debug printf.\n");
	::printf("  --ngg-rectlist-draw <true|false>     Draw rect-list auto draws using the NGG "
	         "4-vertex path.\n");
	::printf("  --rd                                 Enable RenderDoc capture.\n");
	::printf("  --save-data-folder <path>            Host folder for save data. Default: "
	         "_SaveData.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
	if (index + 1 >= argc) {
		return false;
	}

	index++;
	out = argv[index];
	return true;
}

static bool ParseBool(const std::string& value, bool& out) {
	if (Common::EqualNoCase(value, "true") || value == "1" || Common::EqualNoCase(value, "yes") ||
	    Common::EqualNoCase(value, "on")) {
		out = true;
		return true;
	}

	if (Common::EqualNoCase(value, "false") || value == "0" || Common::EqualNoCase(value, "no") ||
	    Common::EqualNoCase(value, "off")) {
		out = false;
		return true;
	}

	return false;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
	auto enum_value = magic_enum::enum_cast<E>(value.c_str());
	if (!enum_value.has_value()) {
		return false;
	}

	out = enum_value.value();
	return true;
}

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
	show_help = false;

	for (int i = 1; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		std::string value;

		if (arg == "--help" || arg == "-h") {
			show_help = true;
			continue;
		}

		if (arg == "--rd") {
			options.config.renderdoc_enabled = true;
			continue;
		}

		if (!Common::StartsWith(arg, "--")) {
			::printf("game input must be provided with --game\n");
			return false;
		}

		if (!NextArg(argc, argv, i, value)) {
			::printf("missing value for %s\n", arg.c_str());
			return false;
		}

		if (arg == "--game") {
			if (!options.app0_dir.empty()) {
				::printf("--game can only be specified once\n");
				return false;
			}

			value = Common::FixFilenameSlash(value);
			if (Common::File::IsDirectoryExisting(value)) {
				options.app0_dir = value;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::File::IsFileExisting(value)) {
				options.app0_dir = Common::DirectoryWithoutFilename(value);
				if (options.app0_dir.empty()) {
					options.app0_dir = ".";
				}
				options.elf = "/app0/" + Common::FilenameWithoutDirectory(value);
			} else {
				::printf("--game must point to an existing directory or ELF: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--screen-width") {
			options.config.screen_width = static_cast<uint32_t>(Common::ToInt32(value));
		} else if (arg == "--screen-height") {
			options.config.screen_height = static_cast<uint32_t>(Common::ToInt32(value));
		} else if (arg == "--vblank-frequency") {
			const int32_t vblank_frequency = Common::ToInt32(value);
			options.config.vblank_frequency =
			    static_cast<uint32_t>(vblank_frequency < 0 ? 0 : vblank_frequency);
		} else if (arg == "--vulkan-validation") {
			if (!ParseBool(value, options.config.vulkan_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-validation") {
			if (!ParseBool(value, options.config.shader_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-optimization-type") {
			if (!ParseEnum(value, options.config.shader_optimization_type)) {
				::printf("invalid shader optimization type: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-direction") {
			if (!ParseEnum(value, options.config.shader_log_direction)) {
				::printf("invalid shader log direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-folder") {
			options.config.shader_log_folder = value;
		} else if (arg == "--command-buffer-dump") {
			if (!ParseBool(value, options.config.command_buffer_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--command-buffer-dump-folder") {
			options.config.command_buffer_dump_folder = value;
		} else if (arg == "--graphics-debug-dump") {
			if (!ParseBool(value, options.config.graphics_debug_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--printf-direction") {
			if (!ParseEnum(value, options.config.printf_direction)) {
				::printf("invalid printf direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = value;
		} else if (arg == "--profiler-direction") {
			if (!ParseEnum(value, options.config.profiler_direction)) {
				::printf("invalid profiler direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--spirv-debug-printf") {
			if (!ParseBool(value, options.config.spirv_debug_printf_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--ngg-rectlist-draw") {
			if (!ParseBool(value, options.config.ngg_rectlist_draw_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--save-data-folder") {
			options.config.save_data_folder = value;
		} else {
			::printf("unknown option: %s\n", arg.c_str());
			return false;
		}
	}

	return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

int main(int argc, char* argv[]) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (Common::RunExitWatcherIfRequested(argc, argv)) {
		return 0;
	}
#endif

	// Launcher probes version by spawning with no args and reading stdout. Keep this
	// path free of crash hooks / subsystem teardown (those can AV on exit and lose
	// the buffered usage text → "Can't find emulator").
	if (argc < 2) {
		PrintUsage();
		std::fflush(stdout);
		return 0;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	DisableHostControlFlowGuard();
	Common::InstallCrashDiagnostics();
	Common::DisableKnownVulkanLayers();
#endif

	auto& slist = *SubsystemsList::Instance();

	slist.SetArgs(argc, argv);

	auto* core    = CommonSubsystem::Instance();
	auto* threads = ThreadsSubsystem::Instance();

	slist.Add(core, {});
	slist.Add(threads, {core});

	if (!slist.InitAll(false)) {
		::printf("Failed to initialize '%s' subsystem: %s\n", slist.GetFailName(),
		         slist.GetFailMsg());
		return 1;
	}

	RunOptions options;
	bool       show_help = false;

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		std::fflush(stdout);
		slist.DestroyAll(false);
		return 1;
	}

	if (show_help) {
		PrintUsage();
		std::fflush(stdout);
		slist.DestroyAll(false);
		return 0;
	}

	if (!options.config.printf_output_file.empty()) {
		Common::SetFatalLogMirrorPath(options.config.printf_output_file.string().c_str());
	}

	Run(options);

	slist.DestroyAll(false);

	return 0;
}
