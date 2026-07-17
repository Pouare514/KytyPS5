#include "common/fatalLog.h"

#include <cstdio>
#include <mutex>
#include <string>

namespace Common {

namespace {

std::mutex  g_fatal_log_mutex;
std::string g_fatal_log_mirror_path;

void AppendFatalLine(const char* path, const char* message) {
	if (path == nullptr || path[0] == '\0' || message == nullptr) {
		return;
	}
	if (FILE* file = std::fopen(path, "a")) {
		std::fputs(message, file);
		std::fputc('\n', file);
		std::fclose(file);
	}
}

} // namespace

void SetFatalLogMirrorPath(const char* path) {
	std::lock_guard lock(g_fatal_log_mutex);
	if (path != nullptr && path[0] != '\0') {
		g_fatal_log_mirror_path = path;
	} else {
		g_fatal_log_mirror_path.clear();
	}
}

void LogFatalToFile(const char* message) {
	if (message == nullptr) {
		return;
	}

	std::fputs(message, stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);

	std::lock_guard lock(g_fatal_log_mutex);
	AppendFatalLine("_kyty_fatal.txt", message);
	if (!g_fatal_log_mirror_path.empty()) {
		AppendFatalLine(g_fatal_log_mirror_path.c_str(), message);
	}
}

} // namespace Common
