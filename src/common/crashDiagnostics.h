#ifndef KYTY_COMMON_CRASH_DIAGNOSTICS_H_
#define KYTY_COMMON_CRASH_DIAGNOSTICS_H_

namespace Common {

// Installs runtime hooks that log /GS security failures before __fastfail (Windows only).
void InstallCrashDiagnostics();

} // namespace Common

#endif /* KYTY_COMMON_CRASH_DIAGNOSTICS_H_ */
