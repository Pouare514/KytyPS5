/* Windows x64 fiber save/restore — MS ABI (rcx=ctx, rdx=ret). No prologue. */
.globl FiberSaveContext
.globl FiberRestoreContext

.text

/* int FiberSaveContext(FiberCpuContext* ctx); returns 0; resume returns 1 via rax */
FiberSaveContext:
	movq %rbx, 0(%rcx)
	movq %rbp, 8(%rcx)
	movq %rdi, 16(%rcx)
	movq %rsi, 24(%rcx)
	movq %r12, 32(%rcx)
	movq %r13, 40(%rcx)
	movq %r14, 48(%rcx)
	movq %r15, 56(%rcx)
	leaq 8(%rsp), %r11
	movq %r11, 64(%rcx)
	movq (%rsp), %r11
	movq %r11, 72(%rcx)
	xorl %eax, %eax
	ret

/* void FiberRestoreContext(FiberCpuContext* ctx, uint64_t ret); noreturn */
FiberRestoreContext:
	movq %rcx, %r10
	movq %rdx, %rax
	movq 72(%r10), %r11
	movq 0(%r10), %rbx
	movq 8(%r10), %rbp
	movq 16(%r10), %rdi
	movq 24(%r10), %rsi
	movq 32(%r10), %r12
	movq 40(%r10), %r13
	movq 48(%r10), %r14
	movq 56(%r10), %r15
	movq 64(%r10), %rsp
	jmpq *%r11
