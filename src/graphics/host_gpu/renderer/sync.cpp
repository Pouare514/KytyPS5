#include "graphics/host_gpu/renderer/sync.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/objects/label.h"
#include "graphics/host_gpu/renderer/bufferCache.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/displayBuffer.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <array>
#include <limits>
#include <optional>

namespace Libs::Graphics::Sync {

namespace SubmitTrace {
std::atomic<uint64_t> submit_dcb {0};
std::atomic<uint64_t> submit_acb {0};
std::atomic<uint64_t> add_eq {0};
std::atomic<uint64_t> eop {0};
std::atomic<bool>     bootstrap_eop_done {false};
std::atomic<bool>     bootstrap_seen_irq0 {false};
std::atomic<bool>     bootstrap_seen_eop {false};
std::atomic<bool>     bootstrap_extended_irq {false};

void NoteSubmitDcb() {
	const auto n = submit_dcb.fetch_add(1, std::memory_order_relaxed) + 1;
	LOGF("SubmitTrace: GraphicsDriverSubmitDcb count=%" PRIu64 "\n", n);
}

void NoteSubmitAcb() {
	const auto n = submit_acb.fetch_add(1, std::memory_order_relaxed) + 1;
	LOGF("SubmitTrace: GraphicsDriverSubmitAcb count=%" PRIu64 "\n", n);
}

void NoteAddEq() {
	const auto n = add_eq.fetch_add(1, std::memory_order_relaxed) + 1;
	LOGF("SubmitTrace: GraphicsDriverAddEqEvent count=%" PRIu64 "\n", n);
}

void NoteEop() {
	const auto n = eop.fetch_add(1, std::memory_order_relaxed) + 1;
	LOGF("SubmitTrace: TriggerEopEvent count=%" PRIu64 "\n", n);
}

[[nodiscard]] bool NoSubmitsYet() {
	return submit_dcb.load(std::memory_order_relaxed) == 0 &&
	       submit_acb.load(std::memory_order_relaxed) == 0;
}

void LogNdJobSyncTimeout() {
	static std::atomic<uint32_t> logs {0};
	const auto                   n = logs.fetch_add(1, std::memory_order_relaxed);
	if (n >= 8 && (n % 32) != 0) {
		return;
	}
	LOGF("SubmitTrace: NdJobSyncTimeout submit_dcb=%" PRIu64 " submit_acb=%" PRIu64
	     " add_eq=%" PRIu64 " eop=%" PRIu64 " bootstrap=%d extended=%d\n",
	     submit_dcb.load(std::memory_order_relaxed), submit_acb.load(std::memory_order_relaxed),
	     add_eq.load(std::memory_order_relaxed), eop.load(std::memory_order_relaxed),
	     bootstrap_eop_done.load(std::memory_order_relaxed) ? 1 : 0,
	     bootstrap_extended_irq.load(std::memory_order_relaxed) ? 1 : 0);
	if (n < 8) {
		fprintf(stderr,
		        "SubmitTrace: NdJobSyncTimeout submit_dcb=%" PRIu64 " submit_acb=%" PRIu64
		        " add_eq=%" PRIu64 " eop=%" PRIu64 " bootstrap=%d extended=%d\n",
		        submit_dcb.load(std::memory_order_relaxed),
		        submit_acb.load(std::memory_order_relaxed), add_eq.load(std::memory_order_relaxed),
		        eop.load(std::memory_order_relaxed),
		        bootstrap_eop_done.load(std::memory_order_relaxed) ? 1 : 0,
		        bootstrap_extended_irq.load(std::memory_order_relaxed) ? 1 : 0);
	}
}
} // namespace SubmitTrace

constexpr int      GRAPHICS_EVENT_QUEUED_GRAPHICS_INTERRUPT = 0x00;
constexpr int      GRAPHICS_EVENT_EOP                       = 0x40;
constexpr uint64_t GRAPHICS_REFERENCE_CLOCK_FREQUENCY       = 100000000;

namespace {
std::mutex                                                            g_graphics_eq_ids_mutex;
std::unordered_map<LibKernel::EventQueue::KernelEqueue, std::unordered_set<int>>
    g_graphics_eq_ids;

void TrackGraphicsEqId(LibKernel::EventQueue::KernelEqueue eq, int id) {
	std::lock_guard lock(g_graphics_eq_ids_mutex);
	g_graphics_eq_ids[eq].insert(id);
}

size_t CountGraphicsEqIds(LibKernel::EventQueue::KernelEqueue eq) {
	std::lock_guard lock(g_graphics_eq_ids_mutex);
	const auto      it = g_graphics_eq_ids.find(eq);
	if (it == g_graphics_eq_ids.end()) {
		return 0;
	}
	return it->second.size();
}

void UntrackGraphicsEqId(LibKernel::EventQueue::KernelEqueue eq, int id) {
	std::lock_guard lock(g_graphics_eq_ids_mutex);
	const auto      it = g_graphics_eq_ids.find(eq);
	if (it == g_graphics_eq_ids.end()) {
		return;
	}
	it->second.erase(id);
	if (it->second.empty()) {
		g_graphics_eq_ids.erase(it);
	}
}
} // namespace

static void MaybeExtendedBootstrapIrq(LibKernel::EventQueue::KernelEqueue eq);

// Phase 28: bootstrap IRQ0 + AGC_USER only — do NOT fake EOP 0x40 before any real DCB.
static void TriggerBootstrapIrq0(LibKernel::EventQueue::KernelEqueue eq) {
	EXIT_IF(eq == nullptr);
	auto result = LibKernel::EventQueue::KernelTriggerEvent(
	    eq, static_cast<uintptr_t>(GRAPHICS_EVENT_QUEUED_GRAPHICS_INTERRUPT),
	    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS, reinterpret_cast<void*>(0));
	EXIT_NOT_IMPLEMENTED(result != OK && result != LibKernel::KERNEL_ERROR_ENOENT);
	TriggerAgcUserInterrupt();
}

static bool TrySyntheticBootstrapIrq0(LibKernel::EventQueue::KernelEqueue eq) {
	if (eq == nullptr || !SubmitTrace::NoSubmitsYet()) {
		return false;
	}
	if (!SubmitTrace::bootstrap_seen_irq0.load(std::memory_order_relaxed) ||
	    !SubmitTrace::bootstrap_seen_eop.load(std::memory_order_relaxed)) {
		return false;
	}
	if (SubmitTrace::bootstrap_eop_done.exchange(true, std::memory_order_acq_rel)) {
		return false;
	}
	const auto tracked = CountGraphicsEqIds(eq);
	LOGF("SubmitTrace: SyntheticBootstrapIrq0 tracked=%zu eq=0x%016" PRIx64 "\n", tracked,
	     reinterpret_cast<uint64_t>(eq));
	fprintf(stderr, "SubmitTrace: SyntheticBootstrapIrq0 tracked=%zu\n", tracked);
	TriggerBootstrapIrq0(eq);
	// One-shot wake only at bootstrap (Phase 28: do not spam on every NdJob timeout).
	const auto woken = LibKernel::PthreadWakeSubmissionCondWaiters();
	LOGF("SubmitTrace: WakeSubmissionCond after bootstrap woken=%zu\n", woken);
	fprintf(stderr, "SubmitTrace: WakeSubmissionCond after bootstrap woken=%zu\n", woken);
	return true;
}

static void MaybeSyntheticBootstrapEop(LibKernel::EventQueue::KernelEqueue eq, int id) {
	if (id == GRAPHICS_EVENT_QUEUED_GRAPHICS_INTERRUPT) {
		SubmitTrace::bootstrap_seen_irq0.store(true, std::memory_order_relaxed);
	} else if (id == GRAPHICS_EVENT_EOP) {
		SubmitTrace::bootstrap_seen_eop.store(true, std::memory_order_relaxed);
	} else {
		return;
	}

	TrySyntheticBootstrapIrq0(eq);
}

void OnNdJobSyncTimeout(LibKernel::EventQueue::KernelEqueue eq) {
	SubmitTrace::LogNdJobSyncTimeout();
	if (!SubmitTrace::NoSubmitsYet()) {
		return;
	}
	LibKernel::PthreadDumpSubmissionThreads("NdJobSyncTimeout");
	if (!SubmitTrace::bootstrap_eop_done.load(std::memory_order_relaxed)) {
		TrySyntheticBootstrapIrq0(eq);
		return;
	}
	MaybeExtendedBootstrapIrq(eq);
}

bool ScaleReferenceClock(uint64_t host_ticks, uint64_t host_frequency, uint64_t* value) {
	if (host_frequency == 0 || value == nullptr) {
		return false;
	}

	const auto     whole_seconds = host_ticks / host_frequency;
	const auto     remainder     = host_ticks % host_frequency;
	constexpr auto MAX_VALUE     = std::numeric_limits<uint64_t>::max();
	if (whole_seconds > MAX_VALUE / GRAPHICS_REFERENCE_CLOCK_FREQUENCY ||
	    remainder > MAX_VALUE / GRAPHICS_REFERENCE_CLOCK_FREQUENCY) {
		return false;
	}

	const auto whole_value      = whole_seconds * GRAPHICS_REFERENCE_CLOCK_FREQUENCY;
	const auto fractional_value = (remainder * GRAPHICS_REFERENCE_CLOCK_FREQUENCY) / host_frequency;
	if (whole_value > MAX_VALUE - fractional_value) {
		return false;
	}
	*value = whole_value + fractional_value;
	return true;
}

uint64_t ReadReferenceClock() {
	const auto host_frequency = LibKernel::KernelGetTscFrequency();
	const auto host_ticks     = LibKernel::KernelReadTsc();
	uint64_t   value          = 0;
	if (!ScaleReferenceClock(host_ticks, host_frequency, &value)) {
		EXIT("cannot scale host clock, ticks=0x%016" PRIx64 " frequency=%" PRIu64 "\n", host_ticks,
		     host_frequency);
	}
	return value;
}

static void SubmitLabel(CommandBuffer* buffer, LabelCallback callback_1 = nullptr,
                        LabelCallback callback_2 = nullptr, const uint64_t* args = nullptr) {
	auto* label = LabelCreate(g_render_ctx->GetGraphicCtx(), callback_1, callback_2, args);
	LabelSet(buffer, label);
	LabelDelete(label);
}

static bool CompleteDisplayBufferFlip(const uint64_t* args) {
	if (g_render_ctx == nullptr || args == nullptr) {
		EXIT("GPU flip completion has invalid state, render_ctx=%p args=%p\n",
		     static_cast<const void*>(g_render_ctx), static_cast<const void*>(args));
	}
	Presentation::DisplayBufferCompleteFlipFromGpu(args[0]);
	return true;
}

enum class EndOfPipeCompletion { None, Interrupt, Flip, FlipAndInterrupt };

struct EndOfPipeSignal {
	CommandBuffer*          buffer          = nullptr;
	uint64_t                submit_id       = 0;
	CommandBufferDebugOp    debug_operation = CommandBufferDebugOp::Unknown;
	std::array<uint32_t, 4> debug_args      = {};
	uint64_t                debug_data      = 0;
	std::optional<uint64_t> destination;
	EndOfPipeCompletion     completion      = EndOfPipeCompletion::None;
	uint64_t                completion_data = 0;
};

enum class EndOfPipeWriteSize : uint32_t { Dword = 4, Qword = 8 };
enum class EndOfPipeWriteAction { Write, WriteBack, Interrupt, InterruptWriteBack };

static bool TriggerEopEventCallback(const uint64_t* args) {
	EXIT_IF(g_render_ctx == nullptr);
	EXIT_IF(args == nullptr);
	g_render_ctx->TriggerEopEvent(static_cast<uint32_t>(args[0]));
	return true;
}

static bool TriggerDefaultEopEventCallback(const uint64_t* /*args*/) {
	EXIT_IF(g_render_ctx == nullptr);
	g_render_ctx->TriggerEopEvent(0);
	return true;
}

static void ValidateEndOfPipeSignal(const EndOfPipeSignal& signal) {
	EXIT_IF(g_render_ctx == nullptr);
	if (signal.destination.has_value()) {
		EXIT_IF(*signal.destination == 0);
	}
	EXIT_IF(signal.buffer == nullptr);
	(void)signal.buffer->Handle();
}

static void RecordEndOfPipeSignal(const EndOfPipeSignal& signal) {
	ValidateEndOfPipeSignal(signal);
	signal.buffer->SetDebugInfo(static_cast<uint32_t>(signal.debug_operation), signal.submit_id,
	                            signal.debug_args[0], signal.debug_args[1], signal.debug_args[2],
	                            signal.debug_args[3], signal.debug_data);

	const uint64_t args[LABEL_ARGS_MAX] = {signal.completion_data};
	switch (signal.completion) {
		case EndOfPipeCompletion::None: return;
		case EndOfPipeCompletion::Interrupt:
			SubmitLabel(signal.buffer, nullptr, TriggerEopEventCallback, args);
			return;
		case EndOfPipeCompletion::Flip:
			SubmitLabel(signal.buffer, CompleteDisplayBufferFlip, nullptr, args);
			return;
		case EndOfPipeCompletion::FlipAndInterrupt:
			SubmitLabel(signal.buffer, CompleteDisplayBufferFlip, TriggerDefaultEopEventCallback,
			            args);
			return;
	}
}

static CommandBufferDebugOp DebugOperation(EndOfPipeWriteAction action) {
	switch (action) {
		case EndOfPipeWriteAction::Write: return CommandBufferDebugOp::EopWrite;
		case EndOfPipeWriteAction::WriteBack:
		case EndOfPipeWriteAction::InterruptWriteBack: return CommandBufferDebugOp::EopWriteBack;
		case EndOfPipeWriteAction::Interrupt: return CommandBufferDebugOp::EopInterrupt;
	}
	EXIT("unsupported end-of-pipe write action\n");
	return CommandBufferDebugOp::Unknown;
}

static bool TriggersInterrupt(EndOfPipeWriteAction action) {
	return action == EndOfPipeWriteAction::Interrupt ||
	       action == EndOfPipeWriteAction::InterruptWriteBack;
}

static void RecordEndOfPipeWrite(uint64_t submit_id, CommandBuffer* buffer, uint64_t destination,
                                 uint64_t value, EndOfPipeWriteSize size,
                                 EndOfPipeWriteAction action, uint32_t context_id = 0) {
	const auto width      = static_cast<uint32_t>(size);
	const auto value_low  = static_cast<uint32_t>(value);
	const auto value_high = static_cast<uint32_t>(value >> 32u);
	const bool interrupt  = TriggersInterrupt(action);

	EndOfPipeSignal signal {
	    .buffer          = buffer,
	    .submit_id       = submit_id,
	    .debug_operation = DebugOperation(action),
	    .debug_args      = interrupt ? std::array {width, context_id, value_low, value_high}
	                                 : std::array {width, value_low, value_high, 0u},
	    .debug_data      = destination,
	    .destination     = destination,
	    .completion      = interrupt ? EndOfPipeCompletion::Interrupt : EndOfPipeCompletion::None,
	    .completion_data = context_id != 0 ? context_id : value,
	};
	RecordEndOfPipeSignal(signal);
}

void TriggerAgcUserInterrupt() {
	auto tsc    = LibKernel::KernelReadTsc();
	auto result = LibKernel::EventQueue::KernelTriggerUserEventForAll(AGC_USER_INTERRUPT_EVENT,
	                                                                  reinterpret_cast<void*>(tsc));
	EXIT_NOT_IMPLEMENTED(result != OK && result != LibKernel::KERNEL_ERROR_ENOENT);
}

void TriggerEopEvent(uint32_t context_id) {
	EXIT_IF(g_render_ctx == nullptr);
	SubmitTrace::NoteEop();
	LOGF("SubmitTrace: TriggerEopEvent context_id=%" PRIu32 "\n", context_id);
	g_render_ctx->TriggerEopEvent(context_id);
}

void WriteAtEndOfPipe32(uint64_t submit_id, CommandBuffer* buffer, uint32_t* dst_gpu_addr,
                        uint32_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::Write);
}

void WriteAtEndOfPipeGds32(uint64_t submit_id, CommandBuffer* buffer, uint32_t* dst_gpu_addr,
                           uint32_t dw_offset, uint32_t dw_num) {
	const auto destination = reinterpret_cast<uint64_t>(dst_gpu_addr);
	RecordEndOfPipeSignal({
	    .buffer          = buffer,
	    .submit_id       = submit_id,
	    .debug_operation = CommandBufferDebugOp::EopWrite,
	    .debug_args      = {dw_offset, dw_num, 0, 0},
	    .debug_data      = destination,
	    .destination     = destination,
	});
}

void WriteAtEndOfPipe64(uint64_t submit_id, CommandBuffer* buffer, uint64_t* dst_gpu_addr,
                        uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Write);
}

void WriteAtEndOfPipeClockCounter(uint64_t submit_id, CommandBuffer* buffer, uint64_t* dst_gpu_addr,
                                  uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), 0,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Write);

	LOGF_COLOR(Log::Color::BrightGreen,
	           "EndOfPipe Signal!!! [0x%016" PRIx64 "] <- Clock: 0x%016" PRIx64 "\n",
	           reinterpret_cast<uint64_t>(dst_gpu_addr), value);
}

void WriteAtEndOfPipeClockCounterWithWriteBack(uint64_t submit_id, CommandBuffer* buffer,
                                               uint64_t* dst_gpu_addr, uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), 0,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::WriteBack);

	LOGF_COLOR(Log::Color::BrightGreen,
	           "EndOfPipe Signal!!! [0x%016" PRIx64 "] <- Clock: 0x%016" PRIx64 "\n",
	           reinterpret_cast<uint64_t>(dst_gpu_addr), value);
}

void WriteAtEndOfPipeWithWriteBack64(uint64_t submit_id, CommandBuffer* buffer,
                                     uint64_t* dst_gpu_addr, uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::WriteBack);
}

void WriteAtEndOfPipeWithWriteBack32(uint64_t submit_id, CommandBuffer* buffer,
                                     uint32_t* dst_gpu_addr, uint32_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::WriteBack);
}

void WriteAtEndOfPipeWithInterruptWriteBack64(uint64_t submit_id, CommandBuffer* buffer,
                                              uint64_t* dst_gpu_addr, uint64_t value,
                                              uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::InterruptWriteBack,
	                     context_id);
}

void WriteAtEndOfPipeWithInterruptWriteBack32(uint64_t submit_id, CommandBuffer* buffer,
                                              uint32_t* dst_gpu_addr, uint32_t value,
                                              uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::InterruptWriteBack,
	                     context_id);
}

void WriteAtEndOfPipeWithInterrupt64(uint64_t submit_id, CommandBuffer* buffer,
                                     uint64_t* dst_gpu_addr, uint64_t value, uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Interrupt, context_id);
}

void WriteAtEndOfPipeWithInterrupt32(uint64_t submit_id, CommandBuffer* buffer,
                                     uint32_t* dst_gpu_addr, uint32_t value, uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::Interrupt, context_id);
}

uint64_t PrepareDisplayBufferFlip(CommandBuffer* buffer, int handle, int index, int flip_mode,
                                  int64_t flip_arg) {
	for (;;) {
		uint64_t   request_id = 0;
		const auto result     = Presentation::DisplayBufferSubmitFlipFromGpu(
		    buffer, handle, index, flip_mode, flip_arg, &request_id);
		if (result == OK) {
			EXIT_IF(request_id == 0);
			if (boot_trace_log()) {
				LOGF("FlipTrace: PrepareDisplayBufferFlip OK id=%" PRIu64 "\n", request_id);
			}
			return request_id;
		}
		if (result != VideoOut::VIDEO_OUT_ERROR_FLIP_QUEUE_FULL) {
			EXIT("GPU flip submission failed, result=%d handle=%d index=%d mode=%d arg=%" PRId64
			     "\n",
			     result, handle, index, flip_mode, flip_arg);
		}
		Presentation::DisplayBufferWaitForFlipQueueSlot();
	}
}

void WriteAtEndOfPipeWithInterruptWriteBackFlip32(uint64_t submit_id, CommandBuffer* buffer,
                                                  uint32_t* dst_gpu_addr, uint32_t value,
                                                  int handle, int index, int flip_mode,
                                                  int64_t flip_arg, uint64_t request_id) {
	const auto destination = reinterpret_cast<uint64_t>(dst_gpu_addr);
	RecordEndOfPipeSignal({
	    .buffer          = buffer,
	    .submit_id       = submit_id,
	    .debug_operation = CommandBufferDebugOp::EopWriteBackFlip,
	    .debug_args      = {static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                        static_cast<uint32_t>(flip_mode), value},
	    .debug_data      = static_cast<uint64_t>(flip_arg),
	    .destination     = destination,
	    .completion      = EndOfPipeCompletion::FlipAndInterrupt,
	    .completion_data = request_id,
	});
}

void WriteAtEndOfPipeWithFlip32(uint64_t submit_id, CommandBuffer* buffer, uint32_t* dst_gpu_addr,
                                uint32_t value, int handle, int index, int flip_mode,
                                int64_t flip_arg, uint64_t request_id) {
	const auto destination = reinterpret_cast<uint64_t>(dst_gpu_addr);
	RecordEndOfPipeSignal({
	    .buffer          = buffer,
	    .submit_id       = submit_id,
	    .debug_operation = CommandBufferDebugOp::EopFlip,
	    .debug_args      = {static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                        static_cast<uint32_t>(flip_mode), value},
	    .debug_data      = static_cast<uint64_t>(flip_arg),
	    .destination     = destination,
	    .completion      = EndOfPipeCompletion::Flip,
	    .completion_data = request_id,
	});
}

void WriteAtEndOfPipeOnlyFlip(uint64_t submit_id, CommandBuffer* buffer, int handle, int index,
                              int flip_mode, int64_t flip_arg, uint64_t request_id) {
	RecordEndOfPipeSignal({
	    .buffer          = buffer,
	    .submit_id       = submit_id,
	    .debug_operation = CommandBufferDebugOp::EopOnlyFlip,
	    .debug_args      = {static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                        static_cast<uint32_t>(flip_mode), 0},
	    .debug_data      = static_cast<uint64_t>(flip_arg),
	    .completion      = EndOfPipeCompletion::Flip,
	    .completion_data = request_id,
	});
}

void TriggerEopEventAtEndOfPipe(CommandBuffer* buffer, uint32_t context_id) {
	ValidateEndOfPipeSignal({.buffer = buffer});

	uint64_t args[LABEL_ARGS_MAX] = {static_cast<uint64_t>(context_id)};
	SubmitLabel(buffer, nullptr, TriggerEopEventCallback, args);
}

static void EopEventResetFunc(LibKernel::EventQueue::KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	event->triggered    = false;
	event->event.fflags = 0;
	event->event.data   = 0;
}

static void EopEventDeleteFunc(LibKernel::EventQueue::KernelEqueue       eq,
                               LibKernel::EventQueue::KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	EXIT_IF(g_render_ctx == nullptr);
	EXIT_NOT_IMPLEMENTED(event->event.filter != LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS);
	if (event->event.ident == GRAPHICS_EVENT_QUEUED_GRAPHICS_INTERRUPT ||
	    event->event.ident == GRAPHICS_EVENT_EOP) {
		g_render_ctx->DeleteEopEq(eq, static_cast<int>(event->event.ident));
	}
}

static void EopEventTriggerFunc(LibKernel::EventQueue::KernelEqueueEvent* event,
                                void*                                     trigger_data) {
	EXIT_IF(event == nullptr);

	auto triggered_event = event->event;
	triggered_event.fflags++;
	triggered_event.data = reinterpret_cast<intptr_t>(trigger_data);
	if (event->triggered) {
		event->pending_events.push_back(triggered_event);
	} else {
		event->event     = triggered_event;
		event->triggered = true;
	}
}

int AddEqEvent(LibKernel::EventQueue::KernelEqueue eq, int id, void* udata) {
	EXIT_IF(g_render_ctx == nullptr);

	LibKernel::EventQueue::KernelEqueueEvent event;
	event.triggered                = false;
	event.event.ident              = static_cast<uintptr_t>(id);
	event.event.filter             = LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS;
	event.event.udata              = udata;
	event.event.fflags             = 0;
	event.event.data               = id;
	event.filter.delete_event_func = EopEventDeleteFunc;
	event.filter.reset_func        = EopEventResetFunc;
	event.filter.trigger_func      = EopEventTriggerFunc;
	event.filter.data              = nullptr;

	int result = LibKernel::EventQueue::KernelAddEvent(eq, event);

	if (result == 0 &&
	    (id == GRAPHICS_EVENT_QUEUED_GRAPHICS_INTERRUPT || id == GRAPHICS_EVENT_EOP)) {
		g_render_ctx->AddEopEq(eq, id);
		// Ensure TriggerEopEvent's AGC_USER 0x1800 can land on this EQ (TLOU never AddUserEvent 0x1800).
		// Edge (EV_CLEAR): level-triggered AddUserEvent sticky-livelocked the IRQ handler (Phase 26).
		static std::atomic<bool> agc_user_added {false};
		if (!agc_user_added.exchange(true, std::memory_order_acq_rel)) {
			const auto user_result =
			    LibKernel::EventQueue::KernelAddUserEventEdge(eq, AGC_USER_INTERRUPT_EVENT);
			LOGF("SubmitTrace: EnsureAgcUserEvent eq=0x%016" PRIx64 " id=0x1800 edge=1 result=%d\n",
			     reinterpret_cast<uint64_t>(eq), user_result);
			fprintf(stderr, "SubmitTrace: EnsureAgcUserEvent edge result=%d\n", user_result);
		}
	}
	MaybeSyntheticBootstrapEop(eq, id);
	SubmitTrace::NoteAddEq();
	LOGF("SubmitTrace: AddEqEvent id=%d result=%d\n", id, result);

	return result;
}

int DeleteEqEvent(LibKernel::EventQueue::KernelEqueue eq, int id) {
	EXIT_IF(g_render_ctx == nullptr);
	UntrackGraphicsEqId(eq, id);

	int result = LibKernel::EventQueue::KernelDeleteEvent(
	    eq, static_cast<uintptr_t>(id), LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS);

	return result;
}

void ReadGds(uint32_t* dst, uint32_t dw_offset, uint32_t dw_size) {
	EXIT_IF(g_render_ctx == nullptr);

	g_render_ctx->GetGdsBuffer()->Read(g_render_ctx->GetGraphicCtx(), dst, dw_offset, dw_size);
}

void DeleteBuffers() {
	EXIT_IF(g_render_ctx == nullptr);
	g_render_ctx->GetBufferCache()->ResetNullBuffer();
}

} // namespace Libs::Graphics::Sync
