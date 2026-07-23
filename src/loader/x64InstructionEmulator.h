#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstddef>
#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);

// Soft-continue guest AVs on poisoned / non-canonical addresses (e.g. Read[ffffffffffffffff]):
// skip the faulting memory op, zero GP/XMM destination on reads, advance RIP.
[[nodiscard]] bool TrySoftContinuePoisonAccess(void* native_context, uint64_t fault_vaddr,
                                               bool is_write, bool force = false,
                                               bool allow_system_module = false);

// Usermode `int imm8` / `ud2` / `int3` — Windows often reports these as AV Read[-1].
// Not soft-continuable; classify as guest_abort_trap (do not treat as poison memop).
[[nodiscard]] bool DescribeGuestAbortTrap(uint64_t rip, char* detail, size_t detail_size);

// SSE #GP (movaps/movdqa) on misaligned guest RSP is often reported as AV[-1].
// Align RSP to 16 bytes and retry the instruction (do not skip).
[[nodiscard]] bool TryFixMisalignedSseAccess(void* native_context, uint64_t fault_vaddr);

// ntdll!LdrpValidateUserCallTarget bitmap AV: simulate `ret` from the check (allow), and
// retarget RCX to a safe stub when the call target is not host-executable (e.g. guest stack).
[[nodiscard]] bool TrySoftContinueCfgBitmap(void* native_context, uint64_t fault_vaddr);

// ntdll!RtlEnterCriticalSection(NULL): AV Write[8] (lock bts [rdi+8],0). Skip-one-insn is
// insufficient (next [rdi+10] faults). Unwind one frame so the caller resumes as if Enter
// returned without locking. Logs ret=mod+off for root-cause tracking.
[[nodiscard]] bool TrySoftContinueNullCriticalSection(void* native_context,
                                                      uint64_t fault_vaddr);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
