#ifndef KYTY_COMMON_CRASH_DIAGNOSTICS_H_
#define KYTY_COMMON_CRASH_DIAGNOSTICS_H_

namespace Common {

// Installs runtime hooks that log /GS security failures and fatal failfast before process death.
void InstallCrashDiagnostics();

// When enabled, ACCESS_VIOLATION in guest VA ranges is logged to the fatal file (diag only).
void EnableGuestAccessViolationLogging(bool enable);

// Lightweight ring of recent HLE entries (always on). Flushed on fatal failfast /gs.
void NoteHleCall(const char* library, const char* module, const char* func);
void FlushHleRingToFatal(const char* reason);

} // namespace Common

#endif /* KYTY_COMMON_CRASH_DIAGNOSTICS_H_ */
