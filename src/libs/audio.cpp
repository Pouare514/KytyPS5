#include "libs/audio.h"

#include "SDL.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/magicEnum.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "kernel/semaphore.h"
#include "libs/audio_internal.h"
#include "libs/errno.h"
#include "libs/libs.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

extern "C" {
#include "libatrac9.h"
}

namespace Libs::Audio {

namespace {

constexpr int AUDIO_OUT_PORT_TYPE_MAIN      = 0;
constexpr int AUDIO_OUT_PORT_TYPE_BGM       = 1;
constexpr int AUDIO_OUT_PORT_TYPE_VOICE     = 2;
constexpr int AUDIO_OUT_PORT_TYPE_PERSONAL  = 3;
constexpr int AUDIO_OUT_PORT_TYPE_PADSPK    = 4;
constexpr int AUDIO_OUT_PORT_TYPE_VIBRATION = 10;
constexpr int AUDIO_OUT_PORT_TYPE_AUDIO3D   = 126;
constexpr int AUDIO_OUT_PORT_TYPE_AUX       = 127;

constexpr uint32_t AUDIO_OUT_PARAM_FORMAT_MASK = 0x000000ffu;

static bool audio_out_port_type_is_valid(int type) {
	return (type >= AUDIO_OUT_PORT_TYPE_MAIN && type <= AUDIO_OUT_PORT_TYPE_PADSPK) ||
	       type == AUDIO_OUT_PORT_TYPE_VIBRATION || type == AUDIO_OUT_PORT_TYPE_AUDIO3D ||
	       type == AUDIO_OUT_PORT_TYPE_AUX;
}

} // namespace

class Audio {
public:
	using Format = AudioInternal::Format;

	class Id {
	public:
		explicit Id(int id): m_id(id - 1) {}
		[[nodiscard]] int  ToInt() const { return m_id + 1; }
		[[nodiscard]] bool IsValid() const { return m_id >= 0; }

		friend class Audio;

	private:
		Id() = default;
		static Id Invalid() { return {}; }
		static Id Create(int audio_id) {
			Id r;
			r.m_id = audio_id;
			return r;
		}
		[[nodiscard]] int GetId() const { return m_id; }

		int m_id = -1;
	};

	struct OutputParam {
		Id          handle;
		const void* data = nullptr;
	};

	Audio() = default;
	virtual ~Audio();

	KYTY_CLASS_NO_COPY(Audio);

	Id       AudioOutOpen(int type, uint32_t samples_num, uint32_t freq, Format format);
	bool     AudioOutClose(Id handle);
	bool     AudioOutValid(Id handle);
	bool     AudioOutSetVolume(Id handle, uint32_t bitflag, const int* volume);
	uint32_t AudioOutOutputs(OutputParam* params, uint32_t num, bool blocking = true);
	bool     AudioOutGetStatus(Id handle, int* type, int* channels_num);

	Id       AudioInOpen(uint32_t type, uint32_t samples_num, uint32_t freq, Format format);
	bool     AudioInValid(Id handle);
	uint32_t AudioInInput(Id handle, void* dest);

	static constexpr int OUT_PORTS_MAX = 32;
	static constexpr int IN_PORTS_MAX  = 8;

private:
	struct PortOut {
		bool     used             = false;
		int      type             = 0;
		uint32_t samples_num      = 0;
		uint32_t freq             = 0;
		Format   format           = Format::Unknown;
		uint64_t last_output_time = 0;
		int      channels_num     = 0;
		int      volume[8]        = {};

		SDL_AudioDeviceID audio_device = 0;
		SDL_AudioSpec     audio_spec   = {};
	};

	struct PortIn {
		bool     used            = false;
		uint32_t type            = 0;
		uint32_t samples_num     = 0;
		uint32_t freq            = 0;
		Format   format          = Format::Unknown;
		uint64_t last_input_time = 0;
	};

	Common::Mutex m_mutex;
	PortOut       m_out_ports[OUT_PORTS_MAX];
	PortIn        m_in_ports[IN_PORTS_MAX];

	static bool            FormatIsFloat(Format format);
	static bool            FormatIsStd(Format format);
	static uint32_t        BytesPerSample(Format format);
	static uint32_t        FrameSize(const PortOut& port);
	static SDL_AudioFormat SdlFormat(Format format);
	static bool            OpenSdlDevice(PortOut* port);
	static void            CloseSdlDevice(PortOut* port);
	static const void*     PrepareOutputBuffer(const PortOut& port, const void* data,
	                                           std::vector<uint8_t>* buffer);
	static bool            QueueSdlAudio(PortOut* port, const void* data, bool blocking);
};

static Audio* g_audio = nullptr;

namespace AudioInternal {

int AudioOutOpen(int type, uint32_t samples_num, uint32_t freq, Format format) {
	if (g_audio == nullptr) {
		return 0;
	}

	auto id = g_audio->AudioOutOpen(type, samples_num, freq, format);
	return id.IsValid() ? id.ToInt() : 0;
}

void AudioOutClose(int handle) {
	if (g_audio != nullptr && handle > 0) {
		(void)g_audio->AudioOutClose(Audio::Id(handle));
	}
}

uint32_t AudioOutOutputs(const OutputParam* params, uint32_t num, bool blocking) {
	if (g_audio == nullptr || params == nullptr || num == 0) {
		return 0;
	}

	std::vector<Audio::OutputParam> output_params;
	output_params.reserve(num);
	for (uint32_t i = 0; i < num; i++) {
		if (params[i].handle > 0 && params[i].data != nullptr) {
			output_params.push_back(
			    Audio::OutputParam {Audio::Id(params[i].handle), params[i].data});
		}
	}

	if (output_params.empty()) {
		return 0;
	}

	return g_audio->AudioOutOutputs(output_params.data(),
	                                static_cast<uint32_t>(output_params.size()), blocking);
}

} // namespace AudioInternal

KYTY_SUBSYSTEM_INIT(Audio) {
	EXIT_IF(g_audio != nullptr);

	g_audio = new Audio;
}

KYTY_SUBSYSTEM_UNEXPECTED_SHUTDOWN(Audio) {}

KYTY_SUBSYSTEM_DESTROY(Audio) {
	delete g_audio;
	g_audio = nullptr;
}

Audio::~Audio() {
	for (auto& port: m_out_ports) {
		CloseSdlDevice(&port);
	}
}

bool Audio::FormatIsFloat(Format format) {
	return (format == Format::FloatMono || format == Format::FloatStereo ||
	        format == Format::Float8Ch || format == Format::Float8ChStd);
}

bool Audio::FormatIsStd(Format format) {
	return (format == Format::Signed16bit8ChStd || format == Format::Float8ChStd);
}

uint32_t Audio::BytesPerSample(Format format) {
	return FormatIsFloat(format) ? sizeof(float) : sizeof(int16_t);
}

uint32_t Audio::FrameSize(const PortOut& port) {
	return BytesPerSample(port.format) * port.channels_num;
}

SDL_AudioFormat Audio::SdlFormat(Format format) {
	return FormatIsFloat(format) ? AUDIO_F32SYS : AUDIO_S16SYS;
}

bool Audio::OpenSdlDevice(PortOut* port) {
	EXIT_IF(port == nullptr);

	if (SDL_InitSubSystem(SDL_INIT_AUDIO) < 0) {
		LOGF("AudioOut: SDL audio init failed: %s\n", SDL_GetError());
		return false;
	}

	SDL_AudioSpec desired {};
	desired.freq     = static_cast<int>(port->freq);
	desired.format   = SdlFormat(port->format);
	desired.channels = static_cast<Uint8>(port->channels_num);
	desired.samples  = static_cast<Uint16>(port->samples_num);
	desired.callback = nullptr;

	SDL_AudioSpec obtained {};

	port->audio_device =
	    SDL_OpenAudioDevice(nullptr, 0, &desired, &obtained, SDL_AUDIO_ALLOW_ANY_CHANGE);
	if (port->audio_device == 0) {
		LOGF("AudioOut: SDL_OpenAudioDevice failed: %s\n", SDL_GetError());
		return false;
	}

	port->audio_spec = obtained;
	SDL_PauseAudioDevice(port->audio_device, 0);

	LOGF("AudioOut: opened SDL device (%d Hz, %u ch, format 0x%04x)\n", obtained.freq,
	     obtained.channels, obtained.format);
	return true;
}

void Audio::CloseSdlDevice(PortOut* port) {
	EXIT_IF(port == nullptr);

	if (port->audio_device != 0 && SDL_WasInit(SDL_INIT_AUDIO) != 0) {
		SDL_ClearQueuedAudio(port->audio_device);
		SDL_CloseAudioDevice(port->audio_device);
	}

	port->audio_device = 0;
	port->audio_spec   = {};
}

const void* Audio::PrepareOutputBuffer(const PortOut& port, const void* data,
                                       std::vector<uint8_t>* buffer) {
	EXIT_IF(data == nullptr);
	EXIT_IF(buffer == nullptr);

	const auto frames           = port.samples_num;
	const auto channels         = static_cast<uint32_t>(port.channels_num);
	const auto bytes_per_sample = BytesPerSample(port.format);
	const auto src_size         = frames * channels * bytes_per_sample;

	bool volume_changed = false;
	for (uint32_t ch = 0; ch < channels; ch++) {
		if (port.volume[ch] != 32768) {
			volume_changed = true;
			break;
		}
	}

	if (!volume_changed && !FormatIsStd(port.format)) {
		return data;
	}

	buffer->resize(src_size);

	static constexpr uint32_t STD_8CH_MAP[8] = {0, 1, 2, 3, 6, 7, 4, 5};

	if (FormatIsFloat(port.format)) {
		auto*       dst = reinterpret_cast<float*>(buffer->data());
		const auto* src = static_cast<const float*>(data);

		for (uint32_t frame = 0; frame < frames; frame++) {
			for (uint32_t ch = 0; ch < channels; ch++) {
				const auto src_ch =
				    (FormatIsStd(port.format) && channels == 8 ? STD_8CH_MAP[ch] : ch);
				dst[frame * channels + ch] = src[frame * channels + src_ch] *
				                             (static_cast<float>(port.volume[ch]) / 32768.0f);
			}
		}
	} else {
		auto*       dst = reinterpret_cast<int16_t*>(buffer->data());
		const auto* src = static_cast<const int16_t*>(data);

		for (uint32_t frame = 0; frame < frames; frame++) {
			for (uint32_t ch = 0; ch < channels; ch++) {
				const auto src_ch =
				    (FormatIsStd(port.format) && channels == 8 ? STD_8CH_MAP[ch] : ch);
				int64_t sample =
				    static_cast<int64_t>(src[frame * channels + src_ch]) * port.volume[ch] / 32768;
				if (sample > std::numeric_limits<int16_t>::max()) {
					sample = std::numeric_limits<int16_t>::max();
				} else if (sample < std::numeric_limits<int16_t>::min()) {
					sample = std::numeric_limits<int16_t>::min();
				}
				dst[frame * channels + ch] = static_cast<int16_t>(sample);
			}
		}
	}

	return buffer->data();
}

bool Audio::QueueSdlAudio(PortOut* port, const void* data, bool blocking) {
	EXIT_IF(port == nullptr);

	if (port->audio_device == 0 || data == nullptr) {
		return false;
	}

	std::vector<uint8_t> prepared_buffer;
	const void*          prepared_data = PrepareOutputBuffer(*port, data, &prepared_buffer);
	const auto           prepared_size = FrameSize(*port) * port->samples_num;

	std::vector<uint8_t> convert_buffer;
	const void*          queue_data = prepared_data;
	uint32_t             queue_size = prepared_size;

	SDL_AudioCVT cvt {};
	const int    cvt_result =
	    SDL_BuildAudioCVT(&cvt, SdlFormat(port->format), static_cast<Uint8>(port->channels_num),
	                      static_cast<int>(port->freq), port->audio_spec.format,
	                      port->audio_spec.channels, port->audio_spec.freq);

	if (cvt_result < 0) {
		LOGF("AudioOut: SDL_BuildAudioCVT failed: %s\n", SDL_GetError());
		return false;
	}

	if (cvt_result > 0) {
		convert_buffer.resize(prepared_size * cvt.len_mult);
		std::memcpy(convert_buffer.data(), prepared_data, prepared_size);

		cvt.buf = convert_buffer.data();
		cvt.len = static_cast<int>(prepared_size);

		if (SDL_ConvertAudio(&cvt) < 0) {
			LOGF("AudioOut: SDL_ConvertAudio failed: %s\n", SDL_GetError());
			return false;
		}

		queue_data = cvt.buf;
		queue_size = static_cast<uint32_t>(cvt.len_cvt);
	}

	if (blocking) {
		const auto min_queued_size = queue_size * 2u;
		const auto wait_start      = LibKernel::KernelGetProcessTime();
		while (SDL_GetQueuedAudioSize(port->audio_device) > min_queued_size) {
			if (LibKernel::KernelGetProcessTime() - wait_start > 200000) {
				SDL_ClearQueuedAudio(port->audio_device);
				break;
			}
			Common::Thread::SleepMicro(1000);
		}
	}

	if (SDL_QueueAudio(port->audio_device, queue_data, queue_size) < 0) {
		LOGF("AudioOut: SDL_QueueAudio failed: %s\n", SDL_GetError());
		return false;
	}

	return true;
}

Audio::Id Audio::AudioOutOpen(int type, uint32_t samples_num, uint32_t freq, Format format) {
	Common::LockGuard lock(m_mutex);

	for (int id = 0; id < OUT_PORTS_MAX; id++) {
		if (!m_out_ports[id].used) {
			auto& port = m_out_ports[id];

			port.used             = true;
			port.type             = type;
			port.samples_num      = samples_num;
			port.freq             = freq;
			port.format           = format;
			port.last_output_time = 0;

			switch (format) {
				case Format::Signed16bitMono:
				case Format::FloatMono: port.channels_num = 1; break;
				case Format::Signed16bitStereo:
				case Format::FloatStereo: port.channels_num = 2; break;
				case Format::Signed16bit8Ch:
				case Format::Float8Ch:
				case Format::Signed16bit8ChStd:
				case Format::Float8ChStd: port.channels_num = 8; break;
				default: EXIT("unknown format");
			}

			for (int i = 0; i < port.channels_num; i++) {
				port.volume[i] = 32768;
			}

			if (type != AUDIO_OUT_PORT_TYPE_VIBRATION) {
				if (type == AUDIO_OUT_PORT_TYPE_AUDIO3D) {
					LOGF("AudioOut: Audio3D port mapped to stereo SDL output\n");
				}
				OpenSdlDevice(&port);
			}

			return Id::Create(id);
		}
	}

	return Id::Invalid();
}

bool Audio::AudioOutClose(Id handle) {
	Common::LockGuard lock(m_mutex);

	if (AudioOutValid(handle)) {
		auto& port = m_out_ports[handle.GetId()];

		CloseSdlDevice(&port);
		port = {};

		return true;
	}

	return false;
}

bool Audio::AudioOutValid(Id handle) {
	Common::LockGuard lock(m_mutex);

	return (handle.GetId() >= 0 && handle.GetId() < OUT_PORTS_MAX &&
	        m_out_ports[handle.GetId()].used);
}

bool Audio::AudioOutGetStatus(Id handle, int* type, int* channels_num) {
	Common::LockGuard lock(m_mutex);

	if (AudioOutValid(handle)) {
		auto& port = m_out_ports[handle.GetId()];

		*type         = port.type;
		*channels_num = port.channels_num;

		return true;
	}

	return false;
}

bool Audio::AudioOutSetVolume(Id handle, uint32_t bitflag, const int* volume) {
	Common::LockGuard lock(m_mutex);

	if (AudioOutValid(handle)) {
		auto& port = m_out_ports[handle.GetId()];

		for (int i = 0; i < port.channels_num; i++, bitflag >>= 1u) {
			auto bit = bitflag & 0x1u;

			if (bit == 1) {
				int src_index = i;
				if (port.format == Format::Float8ChStd ||
				    port.format == Format::Signed16bit8ChStd) {
					switch (i) {
						case 4: src_index = 6; break;
						case 5: src_index = 7; break;
						case 6: src_index = 4; break;
						case 7: src_index = 5; break;
						default:;
					}
				}
				port.volume[i] = volume[src_index];

				LOGF("\t port.volume[%d] = volume[%d] (%d)\n", i, src_index, volume[src_index]);
			}
		}

		return true;
	}

	return false;
}

uint32_t Audio::AudioOutOutputs(OutputParam* params, uint32_t num, bool blocking) {
	EXIT_NOT_IMPLEMENTED(num == 0);
	EXIT_NOT_IMPLEMENTED(!AudioOutValid(params[0].handle));

	const auto& first_port = m_out_ports[params[0].handle.GetId()];

	uint64_t block_time   = (1000000 * first_port.samples_num) / first_port.freq;
	uint64_t current_time = LibKernel::KernelGetProcessTime();

	uint64_t max_wait_time = 0;

	for (uint32_t i = 0; i < num; i++) {
		uint64_t next_time = m_out_ports[params[i].handle.GetId()].last_output_time + block_time;
		uint64_t wait_time = (next_time > current_time ? next_time - current_time : 0);
		max_wait_time      = (wait_time > max_wait_time ? wait_time : max_wait_time);
	}

	if (blocking && max_wait_time != 0) {
		Common::Thread::SleepMicro(max_wait_time);
	}

	for (uint32_t i = 0; i < num; i++) {
		auto& port = m_out_ports[params[i].handle.GetId()];

		QueueSdlAudio(&port, params[i].data, blocking);
	}

	for (uint32_t i = 0; i < num; i++) {
		m_out_ports[params[i].handle.GetId()].last_output_time = LibKernel::KernelGetProcessTime();
	}

	return first_port.samples_num;
}

Audio::Id Audio::AudioInOpen(uint32_t type, uint32_t samples_num, uint32_t freq, Format format) {
	Common::LockGuard lock(m_mutex);

	for (int id = 0; id < IN_PORTS_MAX; id++) {
		if (!m_in_ports[id].used) {
			auto& port = m_in_ports[id];

			port.used        = true;
			port.type        = type;
			port.samples_num = samples_num;
			port.freq        = freq;
			port.format      = format;

			switch (format) {
				case Format::Signed16bitMono:
				case Format::Signed16bitStereo: break;
				default: EXIT("unknown format");
			}

			return Id::Create(id);
		}
	}

	return Id::Invalid();
}

bool Audio::AudioInValid(Id handle) {
	Common::LockGuard lock(m_mutex);

	return (handle.GetId() >= 0 && handle.GetId() < IN_PORTS_MAX &&
	        m_in_ports[handle.GetId()].used);
}

uint32_t Audio::AudioInInput(Id handle, void* dest) {
	EXIT_NOT_IMPLEMENTED(!AudioInValid(handle));
	EXIT_NOT_IMPLEMENTED(dest == nullptr);

	const auto& port = m_in_ports[handle.GetId()];

	uint64_t block_time   = (1000000 * port.samples_num) / port.freq;
	uint64_t current_time = LibKernel::KernelGetProcessTime();

	uint64_t next_time = m_in_ports[handle.GetId()].last_input_time + block_time;
	uint64_t wait_time = (next_time > current_time ? next_time - current_time : 0);

	// TODO(): Audio input is not yet implemented, so simulate audio delay
	Common::Thread::SleepMicro(wait_time);

	m_in_ports[handle.GetId()].last_input_time = LibKernel::KernelGetProcessTime();

	return port.samples_num;
}

namespace AudioOut {

LIB_NAME("AudioOut", "AudioOut");

struct AudioOutOutputParam {
	int         handle;
	const void* ptr;
};

struct AudioOutPortState {
	uint16_t output;
	uint8_t  channel;
	uint8_t  reserved1[1];
	int16_t  volume;
	uint16_t reroute_counter;
	uint64_t flag;
	uint64_t reserved2[2];
};

int KYTY_SYSV_ABI AudioOutInit() {
	PRINT_NAME();

	return OK;
}

int KYTY_SYSV_ABI AudioOutOpen(int user_id, int type, int index, uint32_t len, uint32_t freq,
                               uint32_t param) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n"
	     "\t len     = %u\n"
	     "\t freq    = %u\n",
	     user_id, type, index, len, freq);

	if (!audio_out_port_type_is_valid(type)) {
		return AUDIO_OUT_ERROR_INVALID_PORT_TYPE;
	}
	EXIT_NOT_IMPLEMENTED(index != 0);

	Audio::Format format       = Audio::Format::Unknown;
	const auto    format_param = param & AUDIO_OUT_PARAM_FORMAT_MASK;

	switch (format_param) {
		case 0: format = Audio::Format::Signed16bitMono; break;
		case 1: format = Audio::Format::Signed16bitStereo; break;
		case 2: format = Audio::Format::Signed16bit8Ch; break;
		case 3: format = Audio::Format::FloatMono; break;
		case 4: format = Audio::Format::FloatStereo; break;
		case 5: format = Audio::Format::Float8Ch; break;
		case 6: format = Audio::Format::Signed16bit8ChStd; break;
		case 7: format = Audio::Format::Float8ChStd; break;
		default:;
	}

	LOGF("\t param   = %u (format=%u, %s)\n", param, format_param,
	     Common::EnumName(format).c_str());

	EXIT_NOT_IMPLEMENTED(format == Audio::Format::Unknown);

	EXIT_IF(g_audio == nullptr);

	auto id = g_audio->AudioOutOpen(type, len, freq, format);

	if (!id.IsValid()) {
		return AUDIO_OUT_ERROR_PORT_FULL;
	}

	return id.ToInt();
}

int KYTY_SYSV_ABI AudioOutClose(int handle) {
	PRINT_NAME();

	if (!g_audio->AudioOutClose(Audio::Id(handle))) {
		return AUDIO_OUT_ERROR_INVALID_PORT;
	}

	return OK;
}

int KYTY_SYSV_ABI AudioOutGetPortState(int handle, AudioOutPortState* state) {
	PRINT_NAME();

	int type         = 0;
	int channels_num = 0;

	if (!g_audio->AudioOutGetStatus(Audio::Id(handle), &type, &channels_num)) {
		return AUDIO_OUT_ERROR_INVALID_PORT;
	}

	EXIT_NOT_IMPLEMENTED(state == nullptr);

	state->reroute_counter = 0;
	state->volume          = 127;

	switch (type) {
		case AUDIO_OUT_PORT_TYPE_MAIN:
		case AUDIO_OUT_PORT_TYPE_BGM:
		case AUDIO_OUT_PORT_TYPE_AUDIO3D:
			state->output  = 1;
			state->channel = (channels_num > 2 ? 2 : channels_num);
			break;
		case AUDIO_OUT_PORT_TYPE_VOICE:
		case AUDIO_OUT_PORT_TYPE_PERSONAL:
			state->output  = 0x40;
			state->channel = 1;
			break;
		case AUDIO_OUT_PORT_TYPE_PADSPK:
		case AUDIO_OUT_PORT_TYPE_VIBRATION:
			state->output  = 4;
			state->channel = 1;
			break;
		case AUDIO_OUT_PORT_TYPE_AUX:
			state->output  = 0x80;
			state->channel = 0;
			break;
		default: EXIT("unknown port type: %d\n", type);
	}

	LOGF("\t output  = %" PRIu16 "\n"
	     "\t channel = %" PRIu8 "\n",
	     state->output, state->channel);

	return OK;
}

int KYTY_SYSV_ABI AudioOutSetVolume(int handle, uint32_t flag, int* vol) {
	PRINT_NAME();

	LOGF("\t handle = %d\n"
	     "\t flag   = %u\n",
	     handle, flag);

	EXIT_IF(g_audio == nullptr);
	EXIT_NOT_IMPLEMENTED(vol == nullptr);

	if (!g_audio->AudioOutSetVolume(Audio::Id(handle), flag, vol)) {
		return AUDIO_OUT_ERROR_INVALID_PORT;
	}

	return OK;
}

int KYTY_SYSV_ABI AudioOutOutputs(AudioOutOutputParam* param, uint32_t num) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(param == nullptr);

	Audio::OutputParam params[Audio::OUT_PORTS_MAX];

	EXIT_IF(g_audio == nullptr);

	for (uint32_t i = 0; i < num; i++) {
		params[i].handle = Audio::Id(param[i].handle);
		params[i].data   = param[i].ptr;

		if (!g_audio->AudioOutValid(params[i].handle)) {
			return AUDIO_OUT_ERROR_INVALID_PORT;
		}
	}

	return static_cast<int>(g_audio->AudioOutOutputs(params, num));
}

int KYTY_SYSV_ABI AudioOutOutput(int handle, const void* ptr) {
	// EXIT_NOT_IMPLEMENTED(ptr == nullptr);

	Audio::OutputParam params[1];

	EXIT_IF(g_audio == nullptr);

	params[0].handle = Audio::Id(handle);
	params[0].data   = ptr;

	if (!g_audio->AudioOutValid(params[0].handle)) {
		return AUDIO_OUT_ERROR_INVALID_PORT;
	}

	return static_cast<int>(g_audio->AudioOutOutputs(params, 1));
}

} // namespace AudioOut

namespace AudioIn {

LIB_NAME("AudioIn", "AudioIn");

int KYTY_SYSV_ABI AudioInOpen(int user_id, uint32_t type, uint32_t index, uint32_t len,
                              uint32_t freq, uint32_t param) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %u\n"
	     "\t index   = %d\n"
	     "\t len     = %u\n"
	     "\t freq    = %u\n",
	     user_id, type, index, len, freq);

	if (user_id != 255 && user_id != 1) {
		LOGF("\t temporary: accepting unsupported audio input user_id %d\n", user_id);
	}
	EXIT_NOT_IMPLEMENTED(type != 1);
	EXIT_NOT_IMPLEMENTED(index != 0);

	Audio::Format format = Audio::Format::Unknown;

	switch (param) {
		case 0: format = Audio::Format::Signed16bitMono; break;
		case 2: format = Audio::Format::Signed16bitStereo; break;
		default:
			LOGF("\t temporary: using signed 16-bit stereo for unsupported audio input param %u\n",
			     param);
			format = Audio::Format::Signed16bitStereo;
			break;
	}

	LOGF("\t param   = %u (%s)\n", param, Common::EnumName(format).c_str());

	EXIT_IF(g_audio == nullptr);

	auto id = g_audio->AudioInOpen(type, len, freq, format);

	if (!id.IsValid()) {
		return AUDIO_IN_ERROR_PORT_FULL;
	}

	return id.ToInt();
}

int KYTY_SYSV_ABI AudioInInput(int handle, void* dest) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(dest == nullptr);

	EXIT_IF(g_audio == nullptr);

	if (!g_audio->AudioInValid(Audio::Id(handle))) {
		return AUDIO_IN_ERROR_INVALID_HANDLE;
	}

	return static_cast<int>(g_audio->AudioInInput(Audio::Id(handle), dest));
}

} // namespace AudioIn

namespace VoiceQoS {

LIB_NAME("VoiceQoS", "VoiceQoS");

int KYTY_SYSV_ABI VoiceQoSInit(void* mem_block, uint32_t mem_size, int32_t app_type) {
	PRINT_NAME();

	LOGF("\t mem_block = %016" PRIx64 "\n"
	     "\t mem_size = %" PRIu32 "\n"
	     "\t app_type = %" PRId32 "\n",
	     reinterpret_cast<uint64_t>(mem_block), mem_size, app_type);

	return OK;
}

} // namespace VoiceQoS

namespace Acm {

LIB_NAME("Acm", "Acm");

struct AcmBatchInfo {
	void*  buffer;
	size_t offset;
	size_t buffer_size;
};

struct AcmBatchError {
	uint32_t reserved[8];
};

static std::atomic_uint32_t g_acm_next_context {1};
static std::atomic_uint32_t g_acm_next_batch {1};

static void acm_advance_batch(AcmBatchInfo* info, size_t bytes) {
	if (info == nullptr || info->buffer == nullptr || info->buffer_size == 0) {
		return;
	}

	info->offset = std::min(info->buffer_size, info->offset + bytes);
}

int KYTY_SYSV_ABI AcmContextCreate(AcmContextId* context) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(context == nullptr);

	*context = g_acm_next_context.fetch_add(1, std::memory_order_relaxed);

	LOGF("\t context = %" PRIu32 "\n", *context);

	return OK;
}

int KYTY_SYSV_ABI AcmContextDestroy(AcmContextId context) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 "\n", context);
	return OK;
}

int KYTY_SYSV_ABI AcmBatchStartBuffer(AcmContextId context, const void* batch_commands,
                                      size_t batch_size, AcmBatchError* batch_error,
                                      AcmBatchId* batch) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(batch == nullptr);

	if (batch_error != nullptr) {
		std::memset(batch_error, 0, sizeof(AcmBatchError));
	}

	*batch = g_acm_next_batch.fetch_add(1, std::memory_order_relaxed);

	return OK;
}

int KYTY_SYSV_ABI AcmBatchStartBuffers(AcmContextId context, uint32_t batch_info_count,
                                       const AcmBatchInfo* const batch_info[],
                                       AcmBatchError* batch_error, AcmBatchId* batch) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(batch_info_count != 0 && batch_info == nullptr);
	EXIT_NOT_IMPLEMENTED(batch == nullptr);

	if (batch_error != nullptr) {
		std::memset(batch_error, 0, sizeof(AcmBatchError));
	}

	*batch = g_acm_next_batch.fetch_add(1, std::memory_order_relaxed);

	return OK;
}

int KYTY_SYSV_ABI AcmBatchWait(AcmContextId context, AcmBatchId batch, uint32_t timeout) {
	return OK;
}

int KYTY_SYSV_ABI AcmBatchJobNotification(AcmBatchInfo* batch_info) {
	PRINT_NAME();
	acm_advance_batch(batch_info, 2 * 16);
	return OK;
}

int KYTY_SYSV_ABI AcmConvReverbSharedInput(AcmBatchInfo* batch_info, uint32_t block_count, void* in,
                                           uint32_t count, const void* const ir[],
                                           const float* gain, void* const out[]) {
	PRINT_NAME();
	(void)block_count;
	(void)in;
	(void)count;
	(void)ir;
	(void)gain;
	(void)out;
	acm_advance_batch(batch_info, 1024);
	return OK;
}

int KYTY_SYSV_ABI AcmConvReverbSharedIr(AcmBatchInfo* batch_info, uint32_t block_count,
                                        const void* ir, uint32_t count, void* const in[],
                                        const float* gain, void* const out[]) {
	PRINT_NAME();
	(void)block_count;
	(void)ir;
	(void)count;
	(void)in;
	(void)gain;
	(void)out;
	acm_advance_batch(batch_info, 1024);
	return OK;
}

int KYTY_SYSV_ABI AcmFft(AcmBatchInfo* batch_info, int size, int count, int input_format,
                         const void* const input[], int output_format, void* const output[],
                         uint32_t flags) {
	PRINT_NAME();
	(void)size;
	(void)count;
	(void)input_format;
	(void)input;
	(void)output_format;
	(void)output;
	(void)flags;
	acm_advance_batch(batch_info, 256);
	return OK;
}

int KYTY_SYSV_ABI AcmIfft(AcmBatchInfo* batch_info, int size, int count, int input_format,
                          const void* const input[], int output_format, void* const output[],
                          uint32_t flags) {
	PRINT_NAME();
	(void)size;
	(void)count;
	(void)input_format;
	(void)input;
	(void)output_format;
	(void)output;
	(void)flags;
	acm_advance_batch(batch_info, 256);
	return OK;
}

int KYTY_SYSV_ABI AcmPanner(AcmBatchInfo* batch_info, uint32_t in_count, const float* const in[],
                            uint32_t biquad_count, uint32_t biquad_update_count, uint32_t out_count,
                            const void* const parameter[], void* const state[],
                            const float* const out_init[], float* const out[]) {
	PRINT_NAME();
	(void)in_count;
	(void)in;
	(void)biquad_count;
	(void)biquad_update_count;
	(void)out_count;
	(void)parameter;
	(void)state;
	(void)out_init;
	(void)out;
	acm_advance_batch(batch_info, 512);
	return OK;
}

} // namespace Acm

namespace Audio3d {

LIB_NAME("Audio3d", "Audio3d");

namespace Semaphore = LibKernel::Semaphore;

struct Audio3dOpenParameters {
	size_t   size        = 0x20;
	uint32_t granularity = 256;
	uint32_t rate        = 0;
	uint32_t max_objects = 512;
	uint32_t queue_depth = 2;
	uint32_t buffer_mode = 2;
	uint32_t pad         = 0;
	// uint32_t num_beds;
};

struct Audio3dData {
	enum class State { Empty, Ready, Play };

	std::atomic<State> state = State::Empty;
};

struct Audio3dInternal {
	Audio3dData*          data                        = nullptr;
	Common::Mutex*        data_mutex                  = nullptr;
	uint64_t              data_delay                  = 0;
	Semaphore::KernelSema playback_sema               = nullptr;
	Audio3dOpenParameters params                      = {};
	int                   user_id                     = 0;
	float                 late_reverb_level           = 0.0f;
	float                 downmix_spread_radius       = 2.0f;
	int                   downmix_spread_height_aware = 0;
	uint32_t              data_index                  = 0;
	bool                  used                        = false;
	std::atomic_bool      playback_finished           = false;
};

constexpr uint32_t MAX_PORTS = 4;

static Audio3dInternal g_ports[MAX_PORTS] = {};

static void playback_simulate(void* arg) {
	auto* port = static_cast<Audio3dInternal*>(arg);
	EXIT_IF(port == nullptr);
	EXIT_IF(port->data_mutex == nullptr);
	EXIT_IF(port->data == nullptr);

	for (;;) {
		int result = Semaphore::KernelWaitSema(port->playback_sema, 1, nullptr);

		if (result != OK) {
			break;
		}

		Audio3dData* play_data = nullptr;

		port->data_mutex->Lock();
		{
			for (uint32_t i = 0; i < port->params.queue_depth; i++) {
				uint32_t index = (port->data_index + i) % port->params.queue_depth;

				if (port->data[index].state == Audio3dData::State::Play) {
					play_data = &port->data[index];
					break;
				}
			}
		}
		port->data_mutex->Unlock();

		EXIT_IF(play_data == nullptr);

		if (play_data != nullptr) {
			// TODO(): Audio output is not yet implemented, so simulate audio delay
			Common::Thread::SleepMicro(port->data_delay);
			play_data->state = Audio3dData::State::Empty;
		}
	}

	port->playback_finished = true;
}

int KYTY_SYSV_ABI Audio3dInitialize(int64_t reserved) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(reserved != 0);

	return OK;
}

void KYTY_SYSV_ABI Audio3dGetDefaultOpenParameters(Audio3dOpenParameters* p) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(sizeof(Audio3dOpenParameters) != 0x20);

	*p = Audio3dOpenParameters();
}

int KYTY_SYSV_ABI Audio3dPortOpen(int user_id, const Audio3dOpenParameters* parameters,
                                  uint32_t* id) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(parameters == nullptr);
	EXIT_NOT_IMPLEMENTED(id == nullptr);
	EXIT_NOT_IMPLEMENTED(parameters->size != 0x20);

	LOGF("\t user_id     = %d\n"
	     "\t granularity = %u\n"
	     "\t rate        = %u\n"
	     "\t max_objects = %u\n"
	     "\t queue_depth = %u\n"
	     "\t buffer_mode = %u\n",
	     user_id, parameters->granularity, parameters->rate, parameters->max_objects,
	     parameters->queue_depth, parameters->buffer_mode);

	EXIT_NOT_IMPLEMENTED(parameters->buffer_mode != 2);
	EXIT_NOT_IMPLEMENTED(user_id != 255 && user_id != 1);

	uint32_t port = 0;
	for (; port < MAX_PORTS; port++) {
		if (!g_ports[port].used) {
			break;
		}
	}

	EXIT_NOT_IMPLEMENTED(port >= MAX_PORTS);

	g_ports[port].user_id = user_id;
	g_ports[port].params  = *parameters;
	g_ports[port].used    = true;

	EXIT_IF(g_ports[port].data != nullptr);
	EXIT_IF(g_ports[port].data_mutex != nullptr);
	EXIT_IF(g_ports[port].playback_sema != nullptr);

	g_ports[port].data       = new Audio3dData[parameters->queue_depth];
	g_ports[port].data_index = 0;
	g_ports[port].data_mutex = new Common::Mutex;
	g_ports[port].data_delay = (1000000 * static_cast<uint64_t>(parameters->granularity)) / 48000;

	for (uint32_t d = 0; d < parameters->queue_depth; d++) {
		g_ports[port].data[d].state = Audio3dData::State::Empty;
	}

	int result = Semaphore::KernelCreateSema(&g_ports[port].playback_sema, "audio3d_play", 0x01, 0,
	                                         static_cast<int>(parameters->queue_depth), nullptr);
	EXIT_NOT_IMPLEMENTED(result != OK);

	g_ports[port].playback_finished = false;
	Common::Thread playback_thread(playback_simulate, &g_ports[port]);
	playback_thread.Detach();

	*id = port;

	return OK;
}

int KYTY_SYSV_ABI Audio3dPortSetAttribute(uint32_t port_id, uint32_t attribute_id,
                                          const void* attribute, size_t attribute_size) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(port_id >= MAX_PORTS);
	EXIT_NOT_IMPLEMENTED(!g_ports[port_id].used);
	EXIT_NOT_IMPLEMENTED(attribute == nullptr);

	LOGF("\t attribute_id = 0x%" PRIx32 "\n", attribute_id);

	switch (attribute_id) {
		case 0x10001:
			EXIT_NOT_IMPLEMENTED(attribute_size != 4);
			g_ports[port_id].late_reverb_level = *static_cast<const float*>(attribute);
			LOGF("\t late_reverb_level = %f\n", g_ports[port_id].late_reverb_level);
			break;
		case 0x10002:
			EXIT_NOT_IMPLEMENTED(attribute_size != 4);
			g_ports[port_id].downmix_spread_radius = *static_cast<const float*>(attribute);
			LOGF("\t downmix_spread_radius = %f\n", g_ports[port_id].downmix_spread_radius);
			break;
		case 0x10003:
			EXIT_NOT_IMPLEMENTED(attribute_size != 4);
			g_ports[port_id].downmix_spread_height_aware = *static_cast<const int*>(attribute);
			LOGF("\t downmix_spread_height_aware = %d\n",
			     g_ports[port_id].downmix_spread_height_aware);
			break;
		default: EXIT("unknown attribute: 0x%" PRIx32 "\n", attribute_id);
	}

	return OK;
}

int KYTY_SYSV_ABI Audio3dPortGetQueueLevel(uint32_t port_id, uint32_t* queue_level,
                                           uint32_t* queue_available) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(port_id >= MAX_PORTS);
	EXIT_NOT_IMPLEMENTED(!g_ports[port_id].used);
	EXIT_NOT_IMPLEMENTED(queue_level == nullptr && queue_available == nullptr);

	auto* port = &g_ports[port_id];

	uint32_t empty_num = 0;

	port->data_mutex->Lock();
	{
		for (uint32_t i = 0; i < port->params.queue_depth; i++) {
			uint32_t index = (port->data_index + i) % port->params.queue_depth;

			if (port->data[index].state == Audio3dData::State::Empty) {
				empty_num++;
			} else {
				break;
			}
		}
	}
	port->data_mutex->Unlock();

	EXIT_IF(empty_num > port->params.queue_depth);

	LOGF("\t queue_available = %u\n", empty_num);

	if (queue_level != nullptr) {
		*queue_level = port->params.queue_depth - empty_num;
	}
	if (queue_available != nullptr) {
		*queue_available = empty_num;
	}

	return OK;
}

int KYTY_SYSV_ABI Audio3dPortAdvance(uint32_t port_id) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(port_id >= MAX_PORTS);
	EXIT_NOT_IMPLEMENTED(!g_ports[port_id].used);

	auto* port = &g_ports[port_id];

	port->data_mutex->Lock();
	{
		uint32_t current_index = port->data_index;
		uint32_t next_index    = (current_index + 1) % port->params.queue_depth;

		if (port->data[current_index].state == Audio3dData::State::Empty) {
			port->data[current_index].state = Audio3dData::State::Ready;
		}

		EXIT_NOT_IMPLEMENTED(port->data[current_index].state != Audio3dData::State::Ready);

		port->data_index = next_index;

		LOGF("\t %u -> %u\n", current_index, next_index);
	}
	port->data_mutex->Unlock();

	return OK;
}

int KYTY_SYSV_ABI Audio3dPortPush(uint32_t port_id, uint32_t blocking) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(port_id >= MAX_PORTS);
	EXIT_NOT_IMPLEMENTED(!g_ports[port_id].used);

	auto* port = &g_ports[port_id];

	EXIT_NOT_IMPLEMENTED(blocking != 1);

	LOGF("\t blocking = %u\n", blocking);

	int          data_num   = 0;
	Audio3dData* first_data = nullptr;

	port->data_mutex->Lock();
	{
		first_data = port->data + port->data_index;

		for (uint32_t i = 0; i < port->params.queue_depth; i++) {
			uint32_t index = (port->data_index + i) % port->params.queue_depth;

			if (port->data[index].state == Audio3dData::State::Ready) {
				port->data[index].state = Audio3dData::State::Play;
				data_num++;
			}
		}
	}
	port->data_mutex->Unlock();

	LOGF("\t push num = %d\n", data_num);

	if (data_num > 0) {
		Semaphore::KernelSignalSema(port->playback_sema, data_num);

		if (blocking == 1) {
			auto wait_time = port->data_delay / 8;
			while (first_data->state != Audio3dData::State::Empty) {
				Common::Thread::SleepMicro(wait_time);
			}
		}
	}

	return OK;
}

} // namespace Audio3d

namespace Ngs2 {

LIB_NAME("Ngs2", "Ngs2");

struct Ngs2SystemOption {
	size_t    size                     = 0;
	char      name[64]                 = {};
	uintptr_t job_scheduler_options[4] = {};
	uint32_t  flags                    = 0;
	uint32_t  max_grain_samples        = 0;
	uint32_t  num_grain_samples        = 0;
	uint32_t  sample_rate              = 0;
	uint32_t  max_voice_channels       = 0;
	uint32_t  reserved[5]              = {};
};

struct Ngs2RackOption {
	size_t   size                   = 0;
	char     name[64]               = {};
	uint32_t flags                  = 0;
	uint32_t max_grain_samples      = 0;
	uint32_t max_voices             = 0;
	uint32_t max_input_delay_blocks = 0;
	uint32_t max_matrices           = 0;
	uint32_t max_ports              = 0;
	uint32_t max_voice_channels     = 0;
	uint32_t max_output_channels    = 0;
	uint32_t reserved[18]           = {};
};

struct Ngs2MasteringRackOption {
	Ngs2RackOption rack_option;
	uint32_t       max_channels          = 0;
	uint32_t       num_peak_meter_blocks = 0;
};

struct Ngs2SubmixerRackOption {
	Ngs2RackOption rack_option;
	uint32_t       max_channels          = 0;
	uint32_t       max_envelope_points   = 0;
	uint32_t       max_filters           = 0;
	uint32_t       max_inputs            = 0;
	uint32_t       num_peak_meter_blocks = 0;
};

struct Ngs2SamplerRackOption {
	Ngs2RackOption rack_option;
	uint32_t       max_channel_works        = 0;
	uint32_t       max_codec_caches         = 0;
	uint32_t       max_waveform_blocks      = 0;
	uint32_t       max_envelope_points      = 0;
	uint32_t       max_filters              = 0;
	uint32_t       max_atrac9_decoders      = 0;
	uint32_t       max_atrac9_channel_works = 0;
	uint32_t       max_ajm_atrac9_decoders  = 0;
	uint32_t       num_peak_meter_blocks    = 0;
};

struct Ngs2ReverbRackOption {
	Ngs2RackOption rack_option;
	uint32_t       max_channels = 0;
	uint32_t       reverb_size  = 0;
};

struct Ngs2CustomModuleOption {
	uint32_t size = 0;
};

struct Ngs2CustomRackModuleInfo {
	const Ngs2CustomModuleOption* option           = nullptr;
	uint32_t                      module_id        = 0;
	uint32_t                      source_buffer_id = 0;
	uint32_t                      extra_buffer_id  = 0;
	uint32_t                      dest_buffer_id   = 0;
	uint32_t                      state_offset     = 0;
	uint32_t                      state_size       = 0;
	uint32_t                      reserved         = 0;
	uint32_t                      reserved2        = 0;
};

struct Ngs2CustomRackPortInfo {
	uint32_t source_buffer_id = 0;
	uint32_t reserved         = 0;
};

struct Ngs2CustomRackOption {
	Ngs2RackOption           rack_option;
	uint32_t                 state_size  = 0;
	uint32_t                 num_buffers = 0;
	uint32_t                 num_modules = 0;
	uint32_t                 reserved    = 0;
	Ngs2CustomRackModuleInfo module[24];
	Ngs2CustomRackPortInfo   port[16];
};

struct Ngs2CustomSubmixerRackOption {
	Ngs2CustomRackOption custom_rack_option;
	uint32_t             max_channels = 0;
	uint32_t             max_inputs   = 0;
};

struct Ngs2CustomSamplerRackOption {
	Ngs2CustomRackOption custom_rack_option;
	uint32_t             max_channel_works        = 0;
	uint32_t             max_waveform_blocks      = 0;
	uint32_t             max_atrac9_decoders      = 0;
	uint32_t             max_atrac9_channel_works = 0;
	uint32_t             max_ajm_atrac9_decoders  = 0;
	uint32_t             max_codec_caches         = 0;
};

union Ngs2RackOptionUnion {
	Ngs2RackOption               common;
	Ngs2SamplerRackOption        sampler;
	Ngs2MasteringRackOption      mastering;
	Ngs2SubmixerRackOption       submixer;
	Ngs2ReverbRackOption         reverb;
	Ngs2CustomSubmixerRackOption custom_submixer;
	Ngs2CustomSamplerRackOption  custom_sampler;
};

struct Ngs2ContextBufferInfo {
	void*     host_buffer      = nullptr;
	size_t    host_buffer_size = 0;
	uintptr_t reserved[5]      = {};
	uintptr_t user_data        = 0;
};

struct Ngs2RenderBufferInfo {
	void*    buffer        = nullptr;
	size_t   buffer_size   = 0;
	uint32_t waveform_type = 0;
	uint32_t num_channels  = 0;
};

struct Ngs2WaveformFormat {
	uint32_t waveform_type = 0;
	uint32_t num_channels  = 0;
	uint32_t sample_rate   = 0;
	uint32_t config_data   = 0;
	uint32_t frame_margin  = 0;
	uint32_t frame_offset  = 0;
};

struct Ngs2WaveformBlock {
	uintptr_t data_offset      = 0;
	size_t    data_size        = 0;
	uint32_t  num_repeats      = 0;
	uint32_t  num_skip_samples = 0;
	uint32_t  num_samples      = 0;
	uint32_t  reserved         = 0;
	uintptr_t user_data        = 0;
};

struct Ngs2WaveformInfo {
	Ngs2WaveformFormat format;
	uint32_t           data_offset              = 0;
	uint32_t           data_size                = 0;
	uint32_t           loop_begin_position      = 0;
	uint32_t           loop_end_position        = 0;
	uint32_t           num_samples              = 0;
	uint32_t           audio_unit_size          = 0;
	uint32_t           num_audio_unit_samples   = 0;
	uint32_t           num_audio_unit_per_frame = 0;
	uint32_t           audio_frame_size         = 0;
	uint32_t           num_audio_frame_samples  = 0;
	uint32_t           num_delay_samples        = 0;
	uint32_t           num_blocks               = 0;
	Ngs2WaveformBlock  blocks[4];
};

struct Ngs2PanParam {
	float angle     = 0.0f;
	float distance  = 0.0f;
	float fbw_level = 0.0f;
	float lfe_level = 0.0f;
};

struct Ngs2PanWork {
	float    speaker_angles[8] = {};
	float    unit_angle        = 0.0f;
	uint32_t num_speakers      = 0;
};

struct Ngs2GeomVector {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
};

struct Ngs2GeomCone {
	float inner_level = 0.0f;
	float inner_angle = 0.0f;
	float outer_level = 0.0f;
	float outer_angle = 0.0f;
};

struct Ngs2GeomRolloff {
	uint32_t model              = 0;
	float    max_distance       = 0.0f;
	float    rolloff_factor     = 0.0f;
	float    reference_distance = 0.0f;
};

struct Ngs2GeomListenerParam {
	Ngs2GeomVector position;
	Ngs2GeomVector orient_front;
	Ngs2GeomVector orient_up;
	Ngs2GeomVector velocity;
	float          sound_speed = 0.0f;
	uint32_t       reserved[2] = {};
};

struct Ngs2GeomListenerWork {
	float          matrix[4][4] = {};
	Ngs2GeomVector velocity;
	float          sound_speed = 0.0f;
	uint32_t       coordinate  = 0;
	uint32_t       reserved[3] = {};
};

struct Ngs2GeomSourceParam {
	Ngs2GeomVector  position;
	Ngs2GeomVector  velocity;
	Ngs2GeomVector  direction;
	Ngs2GeomCone    cone;
	Ngs2GeomRolloff rolloff;
	float           doppler_factor = 0.0f;
	float           fbw_level      = 0.0f;
	float           lfe_level      = 0.0f;
	float           max_level      = 0.0f;
	float           min_level      = 0.0f;
	float           radius         = 0.0f;
	uint32_t        num_speakers   = 0;
	uint32_t        matrix_format  = 0;
	uint32_t        reserved[2]    = {};
};

struct Ngs2GeomA3dAttribute {
	Ngs2GeomVector position;
	float          volume      = 0.0f;
	uint32_t       reserved[4] = {};
};

struct Ngs2GeomAttribute {
	float                pitch_ratio = 0.0f;
	float                level[64]   = {};
	Ngs2GeomA3dAttribute a3d_attrib;
	uint32_t             reserved[4] = {};
};

using Ngs2BufferAllocHandler = int32_t KYTY_SYSV_ABI (*)(Ngs2ContextBufferInfo*);
using Ngs2BufferFreeHandler  = int32_t  KYTY_SYSV_ABI (*)(Ngs2ContextBufferInfo*);

struct Ngs2BufferAllocator {
	Ngs2BufferAllocHandler alloc_handler = nullptr;
	Ngs2BufferFreeHandler  free_handler  = nullptr;
	uintptr_t              user_data     = 0;
};

struct Ngs2Internal {
	Ngs2SystemOption    option;
	Ngs2BufferAllocator allocator;
	Ngs2Internal*       next = nullptr;
	Common::Mutex       mutex;
};

enum class Ngs2RackType {
	Sampler,
	Submixer,
	Mastering,
	Reverb,
	CustomSubmixer,
	CustomSampler,
};

struct Ngs2RackInternal {
	Ngs2Internal*       ngs  = nullptr;
	Ngs2RackInternal*   next = nullptr;
	Ngs2RackType        type = Ngs2RackType::Sampler;
	Ngs2RackOptionUnion option;
	Ngs2BufferAllocator allocator;
};

enum class Ngs2VoicePlayState { Empty, Playing, Paused, Stopped };

enum class Ngs2VoicePlayEvent { None, Play, Pause, Resume, Stop, StopImm, Kill };

struct Ngs2VoiceInternal {
	Ngs2VoicePlayEvent   event          = Ngs2VoicePlayEvent::None;
	Ngs2VoicePlayState   state          = Ngs2VoicePlayState::Empty;
	Ngs2RackInternal*    rack           = nullptr;
	uintptr_t            callback       = 0;
	uintptr_t            callback_data  = 0;
	uint32_t             callback_flags = 0;
	Ngs2WaveformFormat   format {};
	std::vector<int16_t> pcm;
	uint32_t             pcm_rate     = 48000;
	uint32_t             pcm_cursor   = 0;
	int32_t              loop_start   = -1;
	int32_t              loop_end     = -1;
	float                pitch_ratio  = 1.0f;
	float                gain         = 1.0f;
};

// Guest OrbisNgs2WaveformFormat / WaveformBlock layouts (public ABI).
struct GuestWaveformFormat {
	uint32_t waveform_type = 0;
	uint32_t num_channels  = 0;
	uint32_t sample_rate   = 0;
	uint32_t config_data   = 0;
	uint32_t frame_offset  = 0;
	uint32_t frame_margin  = 0;
};

struct GuestWaveformBlock {
	uint32_t data_offset      = 0;
	uint32_t data_size        = 0;
	uint32_t num_repeats      = 0;
	uint32_t num_skip_samples = 0;
	uint32_t num_samples      = 0;
	uint32_t reserved         = 0;
	uint64_t user_data        = 0;
};

static_assert(sizeof(GuestWaveformFormat) == 24);
static_assert(sizeof(GuestWaveformBlock) == 32);

struct Ngs2VoiceParamHeader {
	uint16_t size;
	int16_t  next;
	uint32_t id;
};

constexpr uint32_t kWaveformTypePcmI16    = 0x01u;
constexpr uint32_t kWaveformTypePcmF32    = 0x02u;
constexpr uint32_t kWaveformTypeVag       = 0x03u;
constexpr uint32_t kWaveformTypeAtrac9    = 0x04u;
constexpr uint32_t kWaveformTypePcmI16Alt = 0x12u;
constexpr uint32_t kMaxWaveformBytes      = 8u * 1024u * 1024u;
constexpr uint32_t kVagHeaderSize         = 0x30u;
constexpr uint32_t kVagMagic              = 0x56414770u; // "VAGp" BE

struct Ngs2DecodedWaveform {
	std::vector<int16_t> samples;
	uint32_t             sample_rate = 48000;
	int32_t              loop_start  = -1;
	int32_t              loop_end    = -1;
};

static uint32_t Ngs2ReadBe32(const uint8_t* p) {
	return (static_cast<uint32_t>(p[0]) << 24u) | (static_cast<uint32_t>(p[1]) << 16u) |
	       (static_cast<uint32_t>(p[2]) << 8u) | static_cast<uint32_t>(p[3]);
}

static uint16_t Ngs2ReadLe16(const uint8_t* p) {
	return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8u);
}

static uint32_t Ngs2ReadLe32(const uint8_t* p) {
	return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8u) |
	       (static_cast<uint32_t>(p[2]) << 16u) | (static_cast<uint32_t>(p[3]) << 24u);
}

static bool Ngs2HasNonZeroSample(const std::vector<int16_t>& samples) {
	const auto n = std::min<size_t>(samples.size(), 4096);
	for (size_t i = 0; i < n; i++) {
		if (samples[i] != 0) {
			return true;
		}
	}
	if (samples.size() > n) {
		const auto start = samples.size() - std::min<size_t>(256, samples.size() - n);
		for (size_t i = start; i < samples.size(); i++) {
			if (samples[i] != 0) {
				return true;
			}
		}
	}
	return false;
}

static bool Ngs2IsVag(const uint8_t* raw, size_t raw_size) {
	return raw != nullptr && raw_size >= kVagHeaderSize && Ngs2ReadBe32(raw) == kVagMagic;
}

static bool Ngs2DecodeVagFrames(const uint8_t* frames, size_t frame_bytes, uint32_t sample_rate,
                                Ngs2DecodedWaveform* out) {
	static constexpr int kCoeff0[] = {0, 60, 115, 98, 122};
	static constexpr int kCoeff1[] = {0, 0, -52, -55, -60};

	if (frames == nullptr || out == nullptr || frame_bytes < 16) {
		return false;
	}

	const auto frame_count = frame_bytes / 16u;
	out->samples.clear();
	out->samples.reserve(frame_count * 28u);
	out->sample_rate = (sample_rate != 0 ? sample_rate : 48000u);
	out->loop_start  = -1;
	out->loop_end    = -1;

	int  hist1 = 0;
	int  hist2 = 0;
	bool ended = false;
	for (size_t frame = 0; frame < frame_count && !ended; frame++) {
		const auto* f      = frames + frame * 16u;
		const auto  shift  = f[0] & 0x0fu;
		auto        filter = (f[0] >> 4u) & 0x0fu;
		if (filter > 4u) {
			filter = 0;
		}
		const auto flags      = f[1];
		const auto block_start = static_cast<int32_t>(out->samples.size());
		if (flags == 0x03) {
			out->loop_start = block_start;
		}

		const auto f0 = kCoeff0[filter];
		const auto f1 = kCoeff1[filter];
		for (uint32_t i = 0; i < 14u; i++) {
			const auto d = f[2 + i];
			for (uint32_t nibble = 0; nibble < 2u; nibble++) {
				const auto raw_n = (nibble == 0 ? (d & 0x0fu) : (d >> 4u));
				const auto s     = static_cast<int16_t>(static_cast<int16_t>(raw_n << 12u) >> shift);
				const auto predicted = (hist1 * f0 + hist2 * f1) >> 6;
				const auto sample =
				    static_cast<int>(std::clamp(s + predicted, static_cast<int>(INT16_MIN),
				                                static_cast<int>(INT16_MAX)));
				out->samples.push_back(static_cast<int16_t>(sample));
				hist2 = hist1;
				hist1 = sample;
			}
		}

		if (flags == 0x06) {
			out->loop_end = static_cast<int32_t>(out->samples.size());
		} else if (flags == 0x01 || flags == 0x07) {
			ended = true;
		}
	}

	if (out->loop_start >= 0 && out->loop_end <= out->loop_start) {
		out->loop_end = static_cast<int32_t>(out->samples.size());
	}
	return !out->samples.empty() && Ngs2HasNonZeroSample(out->samples);
}

static bool Ngs2TryDecodeVag(const uint8_t* raw, size_t raw_size, Ngs2DecodedWaveform* out) {
	if (!Ngs2IsVag(raw, raw_size) || out == nullptr) {
		return false;
	}

	const auto declared_size = static_cast<int>(Ngs2ReadBe32(raw + 0x0c));
	auto       sample_rate   = Ngs2ReadBe32(raw + 0x10);
	if (sample_rate == 0) {
		sample_rate = 48000;
	}

	const auto* body      = raw + kVagHeaderSize;
	const auto  body_size = raw_size - kVagHeaderSize;
	const auto  available = body_size - (body_size % 16u);
	size_t      frame_bytes = available;
	if (declared_size > 0 && static_cast<size_t>(declared_size) <= available) {
		frame_bytes = static_cast<size_t>(declared_size) - (static_cast<size_t>(declared_size) % 16u);
	}
	if (frame_bytes == 0) {
		return false;
	}
	return Ngs2DecodeVagFrames(body, frame_bytes, sample_rate, out);
}

static bool Ngs2DecodeRawPcm16(const uint8_t* raw, size_t raw_size, uint32_t channels,
                               uint32_t num_samples_hint, std::vector<int16_t>* out_mono) {
	if (raw == nullptr || out_mono == nullptr || channels == 0 || channels > 8 ||
	    raw_size < channels * 2u) {
		return false;
	}

	const auto frame_bytes = channels * 2u;
	auto       frames      = static_cast<uint32_t>(raw_size / frame_bytes);
	if (num_samples_hint != 0) {
		frames = std::min(frames, num_samples_hint);
	}
	if (frames == 0) {
		return false;
	}

	out_mono->resize(frames);
	bool any_nonzero = false;
	for (uint32_t i = 0; i < frames; i++) {
		int32_t sum = 0;
		for (uint32_t c = 0; c < channels; c++) {
			int16_t sample = 0;
			std::memcpy(&sample, raw + (i * frame_bytes) + (c * 2u), sizeof(sample));
			sum += sample;
		}
		const auto mono = static_cast<int16_t>(sum / static_cast<int32_t>(channels));
		(*out_mono)[i]  = mono;
		any_nonzero     = any_nonzero || mono != 0;
	}
	return any_nonzero;
}

static bool Ngs2DecodeRawFloat32(const uint8_t* raw, size_t raw_size, uint32_t channels,
                                 uint32_t num_samples_hint, std::vector<int16_t>* out_mono) {
	if (raw == nullptr || out_mono == nullptr || channels == 0 || channels > 8 ||
	    raw_size < channels * 4u) {
		return false;
	}

	const auto frame_bytes = channels * 4u;
	auto       frames      = static_cast<uint32_t>(raw_size / frame_bytes);
	if (num_samples_hint != 0) {
		frames = std::min(frames, num_samples_hint);
	}
	if (frames == 0) {
		return false;
	}

	out_mono->resize(frames);
	bool any_nonzero = false;
	for (uint32_t i = 0; i < frames; i++) {
		float sum = 0.0f;
		for (uint32_t c = 0; c < channels; c++) {
			float sample = 0.0f;
			std::memcpy(&sample, raw + (i * frame_bytes) + (c * 4u), sizeof(sample));
			sum += sample;
		}
		const auto mono =
		    static_cast<int16_t>(std::clamp(sum / static_cast<float>(channels), -1.0f, 1.0f) *
		                         32767.0f);
		(*out_mono)[i] = mono;
		any_nonzero    = any_nonzero || mono != 0;
	}
	return any_nonzero;
}

static bool Ngs2TryDecodeAt9Raw(const uint8_t* raw, size_t raw_size, const uint8_t config[4],
                                int32_t sample_count_hint, int32_t encoder_delay,
                                Ngs2DecodedWaveform* out) {
	if (raw == nullptr || config == nullptr || out == nullptr || raw_size == 0) {
		return false;
	}

	auto* handle = Atrac9GetHandle();
	if (handle == nullptr) {
		return false;
	}

	uint8_t config_data[ATRAC9_CONFIG_DATA_SIZE] {};
	std::memcpy(config_data, config, 4);
	if (Atrac9InitDecoder(handle, config_data) != 0) {
		Atrac9ReleaseHandle(handle);
		return false;
	}

	Atrac9CodecInfo info {};
	if (Atrac9GetCodecInfo(handle, &info) != 0 || info.channels <= 0 || info.frameSamples <= 0 ||
	    info.superframeSize <= 0 || info.framesInSuperframe <= 0) {
		Atrac9ReleaseHandle(handle);
		return false;
	}

	const auto channels           = static_cast<uint32_t>(info.channels);
	const auto frame_samples      = static_cast<uint32_t>(info.frameSamples);
	const auto frames_in_sf       = static_cast<uint32_t>(info.framesInSuperframe);
	const auto superframe_bytes   = static_cast<uint32_t>(info.superframeSize);
	const auto superframe_samples = frame_samples * frames_in_sf;
	if (superframe_bytes == 0 || superframe_samples == 0) {
		Atrac9ReleaseHandle(handle);
		return false;
	}

	auto sample_count = sample_count_hint;
	if (sample_count <= 0) {
		sample_count = static_cast<int32_t>((raw_size / superframe_bytes) * superframe_samples);
	}
	if (sample_count <= 0) {
		Atrac9ReleaseHandle(handle);
		return false;
	}

	auto superframe_count =
	    (static_cast<uint32_t>(sample_count + encoder_delay) + superframe_samples - 1u) /
	    superframe_samples;
	superframe_count =
	    std::min(superframe_count, static_cast<uint32_t>(raw_size / superframe_bytes));
	if (superframe_count == 0) {
		Atrac9ReleaseHandle(handle);
		return false;
	}

	std::vector<int16_t> pcm_buffer(static_cast<size_t>(channels) * superframe_samples);
	out->samples.clear();
	out->samples.reserve(static_cast<size_t>(sample_count));
	out->sample_rate = static_cast<uint32_t>(info.samplingRate > 0 ? info.samplingRate : 48000);
	out->loop_start  = -1;
	out->loop_end    = -1;

	int64_t decoded_index = 0;
	int32_t written       = 0;
	for (uint32_t f = 0; f < superframe_count && written < sample_count; f++) {
		int bytes_used = 0;
		// isAligned=0 → interleaved PCM for the whole superframe.
		if (Atrac9Decode(handle, raw + (f * superframe_bytes), pcm_buffer.data(), &bytes_used, 0) !=
		    0) {
			break;
		}
		for (uint32_t s = 0; s < superframe_samples && written < sample_count; s++) {
			if (decoded_index >= encoder_delay) {
				int32_t sum = 0;
				for (uint32_t c = 0; c < channels; c++) {
					sum += pcm_buffer[static_cast<size_t>(s) * channels + c];
				}
				out->samples.push_back(static_cast<int16_t>(sum / static_cast<int32_t>(channels)));
				written++;
			}
			decoded_index++;
		}
	}

	Atrac9ReleaseHandle(handle);
	return written > 0 && Ngs2HasNonZeroSample(out->samples);
}

static bool Ngs2TryDecodeRiffWave(const uint8_t* raw, size_t raw_size, Ngs2DecodedWaveform* out) {
	if (raw == nullptr || out == nullptr || raw_size < 12 ||
	    std::memcmp(raw, "RIFF", 4) != 0 || std::memcmp(raw + 8, "WAVE", 4) != 0) {
		return false;
	}

	uint8_t  config_data[4] {};
	bool     has_config    = false;
	bool     is_at9        = false;
	int32_t  sample_count  = 0;
	int32_t  encoder_delay = 0;
	int32_t  data_offset   = -1;
	int32_t  data_size     = 0;
	uint16_t channels      = 0;
	uint32_t sample_rate   = 0;
	uint16_t bits          = 0;
	uint16_t format_tag    = 0;

	// GUID {47E142D2-36BA-4D8D-88FC-61654F8C836C}
	static constexpr uint8_t kAt9Guid[16] = {0xd2, 0x42, 0xe1, 0x47, 0xba, 0x36, 0x8d, 0x4d,
	                                         0x88, 0xfc, 0x61, 0x65, 0x4f, 0x8c, 0x83, 0x6c};

	size_t pos = 12;
	while (pos + 8 <= raw_size) {
		const auto chunk_size = static_cast<int32_t>(Ngs2ReadLe32(raw + pos + 4));
		const auto chunk_start = pos + 8;
		if (chunk_size < 0 || chunk_start + static_cast<size_t>(chunk_size) > raw_size) {
			break;
		}

		if (std::memcmp(raw + pos, "fmt ", 4) == 0) {
			format_tag  = Ngs2ReadLe16(raw + chunk_start);
			channels    = Ngs2ReadLe16(raw + chunk_start + 2);
			sample_rate = Ngs2ReadLe32(raw + chunk_start + 4);
			bits        = (chunk_size >= 16 ? Ngs2ReadLe16(raw + chunk_start + 14) : 0);
			if (format_tag == 0xFFFE && chunk_size >= 40 &&
			    std::memcmp(raw + chunk_start + 24, kAt9Guid, 16) == 0) {
				is_at9 = true;
				if (chunk_size >= 48) {
					std::memcpy(config_data, raw + chunk_start + 44, 4);
					has_config = true;
				}
			}
		} else if (std::memcmp(raw + pos, "fact", 4) == 0) {
			if (chunk_size >= 4) {
				sample_count = static_cast<int32_t>(Ngs2ReadLe32(raw + chunk_start));
			}
			if (chunk_size >= 12) {
				encoder_delay = static_cast<int32_t>(Ngs2ReadLe32(raw + chunk_start + 8));
			}
		} else if (std::memcmp(raw + pos, "data", 4) == 0) {
			data_offset = static_cast<int32_t>(chunk_start);
			data_size   = chunk_size;
		}

		pos = chunk_start + static_cast<size_t>(chunk_size) + (chunk_size & 1);
	}

	if (data_offset < 0 || data_size <= 0) {
		return false;
	}

	const auto* data = raw + static_cast<size_t>(data_offset);
	if (is_at9 && has_config) {
		return Ngs2TryDecodeAt9Raw(data, static_cast<size_t>(data_size), config_data, sample_count,
		                           encoder_delay, out);
	}

	if ((format_tag == 1 || format_tag == 0xFFFE) && bits == 16 && channels > 0 && channels <= 8 &&
	    sample_rate > 0) {
		std::vector<int16_t> mono;
		if (!Ngs2DecodeRawPcm16(data, static_cast<size_t>(data_size), channels, 0, &mono)) {
			return false;
		}
		out->samples     = std::move(mono);
		out->sample_rate = sample_rate;
		out->loop_start  = -1;
		out->loop_end    = -1;
		return true;
	}

	if (bits == 32 && channels > 0 && channels <= 8 && sample_rate > 0) {
		std::vector<int16_t> mono;
		if (!Ngs2DecodeRawFloat32(data, static_cast<size_t>(data_size), channels, 0, &mono)) {
			return false;
		}
		out->samples     = std::move(mono);
		out->sample_rate = sample_rate;
		out->loop_start  = -1;
		out->loop_end    = -1;
		return true;
	}

	return false;
}

static bool Ngs2TryDecodePayload(const uint8_t* raw, size_t raw_size,
                                 const Ngs2WaveformFormat& format, uint32_t channels,
                                 uint32_t sample_rate, uint32_t num_samples_hint,
                                 Ngs2DecodedWaveform* out) {
	if (raw == nullptr || out == nullptr || raw_size == 0) {
		return false;
	}

	if (Ngs2TryDecodeVag(raw, raw_size, out)) {
		return true;
	}
	if (raw_size >= 12 && Ngs2TryDecodeRiffWave(raw, raw_size, out)) {
		return true;
	}

	if (format.waveform_type == kWaveformTypeAtrac9 || format.config_data != 0) {
		uint8_t config[4] {};
		std::memcpy(config, &format.config_data, 4);
		const auto hint =
		    (num_samples_hint != 0 ? static_cast<int32_t>(num_samples_hint) : 0);
		if (Ngs2TryDecodeAt9Raw(raw, raw_size, config, hint, 0, out)) {
			return true;
		}
	}

	if (format.waveform_type == kWaveformTypePcmF32 ||
	    (num_samples_hint > 0 && raw_size >= num_samples_hint * channels * 4u)) {
		std::vector<int16_t> mono;
		if (Ngs2DecodeRawFloat32(raw, raw_size, channels, num_samples_hint, &mono)) {
			out->samples     = std::move(mono);
			out->sample_rate = sample_rate;
			out->loop_start  = -1;
			out->loop_end    = -1;
			return true;
		}
	}

	if (format.waveform_type == kWaveformTypePcmI16 || format.waveform_type == kWaveformTypePcmI16Alt ||
	    format.waveform_type == kWaveformTypeVag || format.waveform_type == 0 ||
	    format.config_data == 0) {
		std::vector<int16_t> mono;
		if (Ngs2DecodeRawPcm16(raw, raw_size, channels, num_samples_hint, &mono)) {
			out->samples     = std::move(mono);
			out->sample_rate = sample_rate;
			out->loop_start  = -1;
			out->loop_end    = -1;
			return true;
		}
	}

	return false;
}

static bool Ngs2TryArmFromGuestAddress(Ngs2VoiceInternal* voice, uint64_t address, int depth);

static bool Ngs2ArmVoiceFromBlocks(Ngs2VoiceInternal* voice, const void* data_base,
                                   const GuestWaveformBlock* blocks, uint32_t block_count) {
	if (voice == nullptr || data_base == nullptr || blocks == nullptr || block_count == 0) {
		return false;
	}

	auto channels = voice->format.num_channels;
	if (channels == 0 || channels > 8) {
		channels = 2;
	}
	auto rate = voice->format.sample_rate;
	if (rate == 0 || rate > 192000) {
		rate = 48000;
	}

	std::vector<int16_t> combined;
	int32_t              loop_start = -1;
	int32_t              loop_end   = -1;
	uint32_t             out_rate   = rate;

	for (uint32_t i = 0; i < block_count; i++) {
		auto block = blocks[i];
		if (block.data_size == 0 && block.num_samples > 0 &&
		    block.num_samples <= kMaxWaveformBytes) {
			block.data_size = block.num_samples * channels * 2u;
		}
		if (block.data_size == 0 || block.data_size > kMaxWaveformBytes) {
			continue;
		}

		const auto*        raw = static_cast<const uint8_t*>(data_base) + block.data_offset;
		Ngs2DecodedWaveform piece {};
		if (!Ngs2TryDecodePayload(raw, block.data_size, voice->format, channels, rate,
		                          block.num_samples, &piece)) {
			continue;
		}
		if (loop_start < 0 && piece.loop_start >= 0) {
			loop_start = static_cast<int32_t>(combined.size()) + piece.loop_start;
			loop_end =
			    (piece.loop_end >= 0 ? static_cast<int32_t>(combined.size()) + piece.loop_end
			                         : -1);
		}
		out_rate = piece.sample_rate;
		combined.insert(combined.end(), piece.samples.begin(), piece.samples.end());
	}

	if (combined.empty()) {
		const auto base_addr = reinterpret_cast<uint64_t>(data_base);
		return Ngs2TryArmFromGuestAddress(voice, base_addr, 0);
	}

	voice->pcm        = std::move(combined);
	voice->pcm_rate   = out_rate;
	voice->pcm_cursor = 0;
	voice->loop_start = loop_start;
	voice->loop_end   = loop_end;
	return true;
}

static bool Ngs2TryArmFromGuestAddress(Ngs2VoiceInternal* voice, uint64_t address, int depth) {
	if (voice == nullptr || address <= 0x10000ull || depth > 2) {
		return false;
	}

	const auto* head = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(address));

	if (Ngs2IsVag(head, kVagHeaderSize)) {
		const auto declared = Ngs2ReadBe32(head + 0x0c);
		const auto total =
		    kVagHeaderSize +
		    std::min<uint32_t>(declared != 0 ? declared : kMaxWaveformBytes, kMaxWaveformBytes);
		Ngs2DecodedWaveform decoded {};
		if (Ngs2TryDecodeVag(head, total, &decoded)) {
			voice->pcm        = std::move(decoded.samples);
			voice->pcm_rate   = decoded.sample_rate;
			voice->pcm_cursor = 0;
			voice->loop_start = decoded.loop_start;
			voice->loop_end   = decoded.loop_end;
			return true;
		}
	}

	if (std::memcmp(head, "RIFF", 4) == 0) {
		const auto riff_size = static_cast<int32_t>(Ngs2ReadLe32(head + 4));
		const auto total =
		    std::clamp<size_t>(8u + static_cast<size_t>(std::max(riff_size, 0)), 12u,
		                       kMaxWaveformBytes);
		Ngs2DecodedWaveform decoded {};
		if (Ngs2TryDecodeRiffWave(head, total, &decoded)) {
			voice->pcm        = std::move(decoded.samples);
			voice->pcm_rate   = decoded.sample_rate;
			voice->pcm_cursor = 0;
			voice->loop_start = decoded.loop_start;
			voice->loop_end   = decoded.loop_end;
			return true;
		}
	}

	if (depth < 2) {
		constexpr size_t kScan = 0x80;
		for (size_t o = 0; o + 8 <= kScan; o += 8) {
			uint64_t nested = 0;
			std::memcpy(&nested, head + o, sizeof(nested));
			if (nested > 0x10000ull && nested != address &&
			    Ngs2TryArmFromGuestAddress(voice, nested, depth + 1)) {
				return true;
			}
		}
	}

	return false;
}

static void Ngs2ApplySetupParam(Ngs2VoiceInternal* voice, const Ngs2VoiceParamHeader* param) {
	if (param == nullptr || param->size < sizeof(Ngs2VoiceParamHeader) + sizeof(GuestWaveformFormat)) {
		return;
	}
	GuestWaveformFormat format {};
	std::memcpy(&format, reinterpret_cast<const uint8_t*>(param) + sizeof(Ngs2VoiceParamHeader),
	            sizeof(format));
	if (format.num_channels == 0 || format.num_channels > 8 || format.sample_rate == 0 ||
	    format.sample_rate > 192000) {
		return;
	}
	voice->format.waveform_type = format.waveform_type;
	voice->format.num_channels  = format.num_channels;
	voice->format.sample_rate   = format.sample_rate;
	voice->format.config_data   = format.config_data;
	voice->format.frame_offset  = format.frame_offset;
	voice->format.frame_margin  = format.frame_margin;
	voice->pcm_rate             = format.sample_rate;
}

static void Ngs2ApplyWaveformBlocksParam(Ngs2VoiceInternal*         voice,
                                         const Ngs2VoiceParamHeader* param) {
	if (param == nullptr || param->size < 32) {
		return;
	}

	const auto* bytes = reinterpret_cast<const uint8_t*>(param);
	uint64_t    data_addr  = 0;
	uint32_t    flags      = 0;
	uint32_t    num_blocks = 0;
	uint64_t    block_addr = 0;
	std::memcpy(&data_addr, bytes + 8, sizeof(data_addr));
	std::memcpy(&flags, bytes + 16, sizeof(flags));
	std::memcpy(&num_blocks, bytes + 20, sizeof(num_blocks));
	std::memcpy(&block_addr, bytes + 24, sizeof(block_addr));
	(void)flags;

	if (data_addr == 0) {
		voice->pcm.clear();
		voice->pcm_cursor = 0;
		voice->loop_start = -1;
		voice->loop_end   = -1;
		return;
	}
	if (data_addr <= 0x10000ull) {
		return;
	}

	num_blocks = std::min(num_blocks, 16u);
	GuestWaveformBlock blocks[16] {};
	uint32_t           block_count = 0;
	if (num_blocks > 0 && block_addr > 0x10000ull) {
		const auto* guest_blocks = reinterpret_cast<const GuestWaveformBlock*>(
		    static_cast<uintptr_t>(block_addr));
		for (uint32_t i = 0; i < num_blocks; i++) {
			blocks[block_count]             = guest_blocks[i];
			blocks[block_count].num_repeats = 0;
			block_count++;
		}
	}
	if (block_count == 0) {
		blocks[0].data_offset = 0;
		blocks[0].data_size   = 64u * 1024u;
		block_count           = 1;
	}

	if (Ngs2ArmVoiceFromBlocks(voice, reinterpret_cast<const void*>(static_cast<uintptr_t>(data_addr)),
	                           blocks, block_count)) {
		if (voice->state == Ngs2VoicePlayState::Empty) {
			voice->state = Ngs2VoicePlayState::Playing;
		}
	} else if (Ngs2TryArmFromGuestAddress(voice, data_addr, 0)) {
		if (voice->state == Ngs2VoicePlayState::Empty) {
			voice->state = Ngs2VoicePlayState::Playing;
		}
	}
}

static void Ngs2MixVoiceIntoGrain(float* accum, uint32_t frames, uint32_t channels,
                                  Ngs2VoiceInternal* voice, uint32_t dest_rate) {
	if (accum == nullptr || voice == nullptr || voice->pcm.empty() || frames == 0 ||
	    channels == 0) {
		return;
	}
	if (voice->state != Ngs2VoicePlayState::Playing) {
		return;
	}

	const auto  pcm_rate = (voice->pcm_rate != 0 ? voice->pcm_rate : dest_rate);
	const float pitch =
	    (voice->pitch_ratio > 0.0f && std::isfinite(voice->pitch_ratio) ? voice->pitch_ratio
	                                                                   : 1.0f);
	const float step =
	    (static_cast<float>(pcm_rate) / static_cast<float>(dest_rate != 0 ? dest_rate : 48000u)) *
	    pitch;
	const float gain = (voice->gain > 0.0f && std::isfinite(voice->gain) ? voice->gain : 1.0f);
	float       cursor = static_cast<float>(voice->pcm_cursor);

	for (uint32_t f = 0; f < frames; f++) {
		auto index = static_cast<uint32_t>(cursor);
		const auto loop_end =
		    (voice->loop_end > 0 ? static_cast<uint32_t>(voice->loop_end) : voice->pcm.size());
		if (index >= loop_end || index >= voice->pcm.size()) {
			if (voice->loop_start >= 0 && voice->loop_end > voice->loop_start) {
				cursor = static_cast<float>(voice->loop_start);
				index  = static_cast<uint32_t>(cursor);
			} else {
				voice->state      = Ngs2VoicePlayState::Stopped;
				voice->pcm_cursor = static_cast<uint32_t>(voice->pcm.size());
				return;
			}
		}
		const float sample = (static_cast<float>(voice->pcm[index]) / 32768.0f) * gain * 0.5f;
		for (uint32_t c = 0; c < std::min(channels, 2u); c++) {
			accum[f * channels + c] += sample;
		}
		cursor += step;
	}
	voice->pcm_cursor = static_cast<uint32_t>(cursor);
}

struct Ngs2VoiceEventParam {
	Ngs2VoiceParamHeader header;
	uint32_t             event_id;
};

struct Ngs2VoicePatchParam {
	Ngs2VoiceParamHeader header;
	uint32_t             port;
	uint32_t             dest_input_id;
	uintptr_t            dest_handle;
};

struct Ngs2VoiceMatrixLevelsParam {
	Ngs2VoiceParamHeader header;
	uint32_t             matrix_id;
	uint32_t             num_levels;
	const float*         levels;
};

struct Ngs2VoicePortMatrixParam {
	Ngs2VoiceParamHeader header;
	uint32_t             port;
	int32_t              matrix_id;
};

struct Ngs2VoicePortVolumeParam {
	Ngs2VoiceParamHeader header;
	uint32_t             port;
	float                level;
};

struct Ngs2VoicePortDelayParam {
	Ngs2VoiceParamHeader header;
	uint32_t             port;
	uint32_t             num_samples;
};

struct Ngs2VoiceCallbackParam {
	Ngs2VoiceParamHeader header;
	uintptr_t            callback;
	uintptr_t            callback_data;
	uint32_t             flags;
	uint32_t             reserved;
};

struct Ngs2VoiceState {
	uint32_t state_flags;
};

struct Ngs2SamplerVoiceState {
	Ngs2VoiceState voice_state;
	float          envelope_height;
	float          peak_height;
	uint32_t       reserved;
	uint64_t       num_decoded_samples;
	uint64_t       decoded_data_size;
	uint64_t       user_data;
	const void*    waveform_data;
};

static Ngs2Internal*     g_ngs_list   = nullptr;
static Ngs2RackInternal* g_racks_list = nullptr;

static_assert(sizeof(Ngs2SystemOption) == 144);
static_assert(sizeof(Ngs2RackOption) == 176);

static uint32_t Ngs2GetStateFlags(const Ngs2VoiceInternal* voice) {
	switch (voice->state) {
		case Ngs2VoicePlayState::Empty: return 0;
		case Ngs2VoicePlayState::Playing: return 0x3;
		case Ngs2VoicePlayState::Paused: return 0x5;
		case Ngs2VoicePlayState::Stopped: return 0xb;
	}

	return 0;
}

static Ngs2SystemOption Ngs2DefaultSystemOption() {
	Ngs2SystemOption option {};
	option.size              = sizeof(Ngs2SystemOption);
	option.max_grain_samples = 512;
	option.num_grain_samples = 256;
	option.sample_rate       = 48000;
	return option;
}

int KYTY_SYSV_ABI Ngs2SystemResetOption(Ngs2SystemOption* option) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(option == nullptr);

	*option = Ngs2DefaultSystemOption();
	return OK;
}

static Ngs2Internal* Ngs2CreateSystemInternal(const Ngs2SystemOption* option, void* host_buffer) {
	auto* ngs = new (host_buffer) Ngs2Internal;

	ngs->option = *option;
	ngs->next   = g_ngs_list;
	g_ngs_list  = ngs;

	return ngs;
}

static bool Ngs2RackIsCustom(Ngs2RackType type) {
	switch (type) {
		case Ngs2RackType::CustomSubmixer:
		case Ngs2RackType::CustomSampler: return true;
		default: return false;
	}
}

int KYTY_SYSV_ABI Ngs2SystemQueryBufferSize(const Ngs2SystemOption* option,
                                            Ngs2ContextBufferInfo*  buffer_info) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(buffer_info == nullptr);

	auto default_option = Ngs2DefaultSystemOption();
	if (option == nullptr) {
		option = &default_option;
		LOGF("\t option            = nullptr, using reset defaults\n");
	}

	EXIT_NOT_IMPLEMENTED(option->size != sizeof(Ngs2SystemOption));

	std::memset(buffer_info, 0, sizeof(Ngs2ContextBufferInfo));
	buffer_info->host_buffer_size = sizeof(Ngs2Internal);

	return OK;
}

int KYTY_SYSV_ABI Ngs2SystemCreate(const Ngs2SystemOption*      option,
                                   const Ngs2ContextBufferInfo* buffer_info, uintptr_t* handle) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(buffer_info == nullptr);
	EXIT_NOT_IMPLEMENTED(handle == nullptr);
	EXIT_NOT_IMPLEMENTED(buffer_info->host_buffer == nullptr);
	EXIT_NOT_IMPLEMENTED(buffer_info->host_buffer_size < sizeof(Ngs2Internal));

	auto default_option = Ngs2DefaultSystemOption();
	if (option == nullptr) {
		option = &default_option;
		LOGF("\t option            = nullptr, using reset defaults\n");
	}

	EXIT_NOT_IMPLEMENTED(option->size != sizeof(Ngs2SystemOption));

	auto* ngs = Ngs2CreateSystemInternal(option, buffer_info->host_buffer);

	*handle = reinterpret_cast<uintptr_t>(ngs);

	return OK;
}

static void Ngs2FillDefaultRackOption(uint32_t rack_id, Ngs2RackOptionUnion* option) {
	EXIT_NOT_IMPLEMENTED(option == nullptr);

	*option = {};

	switch (rack_id) {
		case 0x1000:
			option->sampler.rack_option.size                   = sizeof(Ngs2SamplerRackOption);
			option->sampler.rack_option.max_grain_samples      = 512;
			option->sampler.rack_option.max_voices             = 256;
			option->sampler.rack_option.max_input_delay_blocks = 0;
			option->sampler.rack_option.max_matrices           = 1;
			option->sampler.rack_option.max_ports              = 8;
			option->sampler.max_channel_works                  = 256;
			option->sampler.max_codec_caches                   = 32;
			option->sampler.max_waveform_blocks                = 4;
			option->sampler.max_envelope_points                = 4;
			option->sampler.max_filters                        = 8;
			option->sampler.max_atrac9_decoders                = 256;
			option->sampler.max_atrac9_channel_works           = 256;
			option->sampler.max_ajm_atrac9_decoders            = 0;
			option->sampler.num_peak_meter_blocks              = 8;
			break;
		case 0x2000:
			option->submixer.rack_option.size                   = sizeof(Ngs2SubmixerRackOption);
			option->submixer.rack_option.max_grain_samples      = 512;
			option->submixer.rack_option.max_voices             = 1;
			option->submixer.rack_option.max_input_delay_blocks = 1;
			option->submixer.rack_option.max_matrices           = 1;
			option->submixer.rack_option.max_ports              = 8;
			option->submixer.max_channels                       = 8;
			option->submixer.max_envelope_points                = 4;
			option->submixer.max_filters                        = 8;
			option->submixer.max_inputs                         = 1;
			option->submixer.num_peak_meter_blocks              = 8;
			break;
		case 0x2001:
			option->reverb.rack_option.size                   = sizeof(Ngs2ReverbRackOption);
			option->reverb.rack_option.max_grain_samples      = 512;
			option->reverb.rack_option.max_voices             = 1;
			option->reverb.rack_option.max_input_delay_blocks = 1;
			option->reverb.rack_option.max_matrices           = 1;
			option->reverb.rack_option.max_ports              = 8;
			option->reverb.max_channels                       = 8;
			option->reverb.reverb_size                        = 1;
			break;
		case 0x3000:
			option->mastering.rack_option.size                   = sizeof(Ngs2MasteringRackOption);
			option->mastering.rack_option.max_grain_samples      = 512;
			option->mastering.rack_option.max_voices             = 1;
			option->mastering.rack_option.max_input_delay_blocks = 1;
			option->mastering.rack_option.max_matrices           = 0;
			option->mastering.rack_option.max_ports              = 0;
			option->mastering.max_channels                       = 8;
			option->mastering.num_peak_meter_blocks              = 8;
			break;
		case 0x4002:
			// FIXME: Temporary PS5 progress fallback. This mirrors Prospero reset helper's
			// common custom-submixer defaults, but the custom module internals are still stubbed.
			option->custom_submixer.custom_rack_option.rack_option.size =
			    sizeof(Ngs2CustomSubmixerRackOption);
			option->custom_submixer.custom_rack_option.rack_option.max_grain_samples      = 512;
			option->custom_submixer.custom_rack_option.rack_option.max_voices             = 1;
			option->custom_submixer.custom_rack_option.rack_option.max_input_delay_blocks = 1;
			option->custom_submixer.custom_rack_option.rack_option.max_matrices           = 1;
			option->custom_submixer.custom_rack_option.rack_option.max_ports              = 8;
			option->custom_submixer.custom_rack_option.num_buffers                        = 1;
			option->custom_submixer.max_channels                                          = 8;
			option->custom_submixer.max_inputs                                            = 1;
			break;
		case 0x4001:
			option->custom_sampler.custom_rack_option.rack_option.size =
			    sizeof(Ngs2CustomSamplerRackOption);
			option->custom_sampler.custom_rack_option.rack_option.max_grain_samples      = 512;
			option->custom_sampler.custom_rack_option.rack_option.max_voices             = 256;
			option->custom_sampler.custom_rack_option.rack_option.max_input_delay_blocks = 0;
			option->custom_sampler.custom_rack_option.rack_option.max_matrices           = 1;
			option->custom_sampler.custom_rack_option.rack_option.max_ports              = 8;
			option->custom_sampler.custom_rack_option.num_buffers                        = 1;
			option->custom_sampler.max_channel_works                                     = 256;
			option->custom_sampler.max_waveform_blocks                                   = 4;
			option->custom_sampler.max_atrac9_decoders                                   = 256;
			option->custom_sampler.max_atrac9_channel_works                              = 256;
			option->custom_sampler.max_ajm_atrac9_decoders                               = 0;
			option->custom_sampler.max_codec_caches                                      = 32;
			break;
		default: EXIT("unknown rack_id for default option: 0x%" PRIx32 "\n", rack_id);
	}
}

int KYTY_SYSV_ABI Ngs2RackQueryBufferSize(uint32_t rack_id, const Ngs2RackOption* option,
                                          Ngs2ContextBufferInfo* buffer_info) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(buffer_info == nullptr);

	Ngs2RackOptionUnion default_option {};
	if (option == nullptr) {
		Ngs2FillDefaultRackOption(rack_id, &default_option);
		option = &default_option.common;
		LOGF("\t option     = nullptr, using reset defaults for rack_id 0x%" PRIx32 "\n", rack_id);
	}

	LOGF("\t rack_id    = 0x%" PRIx32 "\n"
	     "\t max_voices = %u\n",
	     rack_id, option->max_voices);

	buffer_info->host_buffer_size =
	    sizeof(Ngs2RackInternal) + sizeof(Ngs2VoiceInternal) * option->max_voices;

	return OK;
}

int KYTY_SYSV_ABI Ngs2SystemCreateWithAllocator(const Ngs2SystemOption*    option,
                                                const Ngs2BufferAllocator* allocator,
                                                uintptr_t*                 handle) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(allocator == nullptr);
	EXIT_NOT_IMPLEMENTED(handle == nullptr);
	EXIT_NOT_IMPLEMENTED(allocator->alloc_handler == nullptr);
	EXIT_NOT_IMPLEMENTED(allocator->free_handler == nullptr);

	auto default_option = Ngs2DefaultSystemOption();
	if (option == nullptr) {
		option = &default_option;
		LOGF("\t option            = nullptr, using reset defaults\n");
	}

	EXIT_NOT_IMPLEMENTED(option->size != sizeof(Ngs2SystemOption));

	LOGF("\t name              = %.64s\n"
	     "\t flags             = %u\n"
	     "\t max_grain_samples = %u\n"
	     "\t num_grain_samples = %u\n"
	     "\t sample_rate       = %u\n"
	     "\t max_voice_channels = %u\n"
	     "\t alloc_handler     = 0x%016" PRIx64 "\n"
	     "\t free_handler      = 0x%016" PRIx64 "\n"
	     "\t user_data         = 0x%016" PRIx64 "\n",
	     option->name, option->flags, option->max_grain_samples, option->num_grain_samples,
	     option->sample_rate, option->max_voice_channels,
	     reinterpret_cast<uint64_t>(allocator->alloc_handler),
	     reinterpret_cast<uint64_t>(allocator->free_handler),
	     reinterpret_cast<uint64_t>(allocator->user_data));

	Ngs2ContextBufferInfo buf {};
	buf.host_buffer      = nullptr;
	buf.host_buffer_size = sizeof(Ngs2Internal);
	buf.user_data        = allocator->user_data;

	int result = allocator->alloc_handler(&buf);

	EXIT_NOT_IMPLEMENTED(result != OK);
	EXIT_NOT_IMPLEMENTED(buf.host_buffer == nullptr);

	auto* ngs      = Ngs2CreateSystemInternal(option, buf.host_buffer);
	ngs->allocator = *allocator;

	*handle = reinterpret_cast<uintptr_t>(ngs);

	return OK;
}

int KYTY_SYSV_ABI Ngs2SystemSetGrainSamples(uintptr_t system_handle, uint32_t num_samples) {
	PRINT_NAME();
	LOGF("\t system_handle = 0x%016" PRIx64 "\n"
	     "\t num_samples   = %u\n",
	     static_cast<uint64_t>(system_handle), num_samples);

	EXIT_NOT_IMPLEMENTED(system_handle == 0);

	auto* ngs                     = reinterpret_cast<Ngs2Internal*>(system_handle);
	ngs->option.num_grain_samples = num_samples;

	return OK;
}

int KYTY_SYSV_ABI Ngs2SystemDestroy(uintptr_t system_handle, Ngs2ContextBufferInfo* buffer_info) {
	PRINT_NAME();
	LOGF("\t system_handle = 0x%016" PRIx64 "\n", static_cast<uint64_t>(system_handle));

	if (buffer_info != nullptr) {
		std::memset(buffer_info, 0, sizeof(Ngs2ContextBufferInfo));
	}

	return OK;
}

int KYTY_SYSV_ABI Ngs2RackCreate(uintptr_t system_handle, uint32_t rack_id,
                                 const Ngs2RackOption*        option,
                                 const Ngs2ContextBufferInfo* buffer_info, uintptr_t* handle) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(buffer_info == nullptr);
	EXIT_NOT_IMPLEMENTED(handle == nullptr);
	EXIT_NOT_IMPLEMENTED(buffer_info->host_buffer == nullptr);
	EXIT_NOT_IMPLEMENTED(buffer_info->host_buffer_size == 0);
	EXIT_NOT_IMPLEMENTED(system_handle == 0);

	Ngs2RackOptionUnion default_option {};
	if (option == nullptr) {
		Ngs2FillDefaultRackOption(rack_id, &default_option);
		option = &default_option.common;
		LOGF("\t option                 = nullptr, using reset defaults for rack_id 0x%" PRIx32
		     "\n",
		     rack_id);
	}

	EXIT_NOT_IMPLEMENTED(option->size < sizeof(Ngs2RackOption));

	LOGF("\t rack_id                = 0x%" PRIx32 "\n"
	     "\t name                   = %.64s\n"
	     "\t flags                  = %u\n"
	     "\t max_grain_samples      = %u\n"
	     "\t max_voices             = %u\n"
	     "\t max_input_delay_blocks = %u\n"
	     "\t max_matrices           = %u\n"
	     "\t max_ports              = %u\n"
	     "\t max_voice_channels     = %u\n"
	     "\t max_output_channels    = %u\n"
	     "\t host_buffer            = 0x%016" PRIx64 "\n"
	     "\t host_buffer_size      = 0x%016" PRIx64 "\n",
	     rack_id, option->name, option->flags, option->max_grain_samples, option->max_voices,
	     option->max_input_delay_blocks, option->max_matrices, option->max_ports,
	     option->max_voice_channels, option->max_output_channels,
	     reinterpret_cast<uint64_t>(buffer_info->host_buffer),
	     reinterpret_cast<uint64_t>(buffer_info->host_buffer_size));

	auto* ngs    = reinterpret_cast<Ngs2Internal*>(system_handle);
	auto* rack   = static_cast<Ngs2RackInternal*>(buffer_info->host_buffer);
	auto* voices = reinterpret_cast<Ngs2VoiceInternal*>(rack + 1);

	Common::LockGuard lock(ngs->mutex);

	switch (rack_id) {
		case 0x1000:
			EXIT_NOT_IMPLEMENTED(option->size != sizeof(Ngs2SamplerRackOption));
			rack->option.sampler = *reinterpret_cast<const Ngs2SamplerRackOption*>(option);
			rack->type           = Ngs2RackType::Sampler;
			break;
		case 0x2000:
			EXIT_NOT_IMPLEMENTED(option->size != sizeof(Ngs2SubmixerRackOption));
			rack->option.submixer = *reinterpret_cast<const Ngs2SubmixerRackOption*>(option);
			rack->type            = Ngs2RackType::Submixer;
			break;
		case 0x2001:
			EXIT_NOT_IMPLEMENTED(option->size != sizeof(Ngs2ReverbRackOption));
			rack->option.reverb = *reinterpret_cast<const Ngs2ReverbRackOption*>(option);
			rack->type          = Ngs2RackType::Reverb;
			break;
		case 0x3000:
			EXIT_NOT_IMPLEMENTED(option->size != sizeof(Ngs2MasteringRackOption));
			rack->option.mastering = *reinterpret_cast<const Ngs2MasteringRackOption*>(option);
			rack->type             = Ngs2RackType::Mastering;
			break;
		case 0x4002:
			EXIT_NOT_IMPLEMENTED(option->size != sizeof(Ngs2CustomSubmixerRackOption));
			rack->option.custom_submixer =
			    *reinterpret_cast<const Ngs2CustomSubmixerRackOption*>(option);
			rack->type = Ngs2RackType::CustomSubmixer;
			break;
		case 0x4001:
			EXIT_NOT_IMPLEMENTED(option->size != sizeof(Ngs2CustomSamplerRackOption));
			rack->option.custom_sampler =
			    *reinterpret_cast<const Ngs2CustomSamplerRackOption*>(option);
			rack->type = Ngs2RackType::CustomSampler;
			break;
		default: EXIT("unknown rack_id: 0x%" PRIx32 "\n", rack_id);
	}

	LOGF("\t type                   = %s\n", Common::EnumName(rack->type).c_str());

	rack->allocator = Ngs2BufferAllocator();
	rack->ngs       = ngs;

	rack->next   = g_racks_list;
	g_racks_list = rack;

	for (uint32_t i = 0; i < option->max_voices; i++) {
		new (voices + i) Ngs2VoiceInternal();
		voices[i].rack  = rack;
		voices[i].event = Ngs2VoicePlayEvent::None;
		voices[i].state = Ngs2VoicePlayState::Empty;
	}

	*handle = reinterpret_cast<uintptr_t>(rack);

	return OK;
}

int KYTY_SYSV_ABI Ngs2RackCreateWithAllocator(uintptr_t system_handle, uint32_t rack_id,
                                              const Ngs2RackOption*      option,
                                              const Ngs2BufferAllocator* allocator,
                                              uintptr_t*                 handle) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(allocator == nullptr);
	EXIT_NOT_IMPLEMENTED(handle == nullptr);
	EXIT_NOT_IMPLEMENTED(allocator->alloc_handler == nullptr);
	EXIT_NOT_IMPLEMENTED(allocator->free_handler == nullptr);
	EXIT_NOT_IMPLEMENTED(system_handle == 0);

	Ngs2RackOptionUnion default_option {};
	if (option == nullptr) {
		Ngs2FillDefaultRackOption(rack_id, &default_option);
		option = &default_option.common;
		LOGF("\t option                 = nullptr, using reset defaults for rack_id 0x%" PRIx32
		     "\n",
		     rack_id);
	}

	EXIT_NOT_IMPLEMENTED(option->size < sizeof(Ngs2RackOption));

	LOGF("\t rack_id                = 0x%" PRIx32 "\n"
	     "\t name                   = %.64s\n"
	     "\t flags                  = %u\n"
	     "\t max_grain_samples      = %u\n"
	     "\t max_voices             = %u\n"
	     "\t max_input_delay_blocks = %u\n"
	     "\t max_matrices           = %u\n"
	     "\t max_ports              = %u\n"
	     "\t max_voice_channels     = %u\n"
	     "\t max_output_channels    = %u\n"
	     "\t alloc_handler          = 0x%016" PRIx64 "\n"
	     "\t free_handler           = 0x%016" PRIx64 "\n"
	     "\t user_data              = 0x%016" PRIx64 "\n",
	     rack_id, option->name, option->flags, option->max_grain_samples, option->max_voices,
	     option->max_input_delay_blocks, option->max_matrices, option->max_ports,
	     option->max_voice_channels, option->max_output_channels,
	     reinterpret_cast<uint64_t>(allocator->alloc_handler),
	     reinterpret_cast<uint64_t>(allocator->free_handler),
	     reinterpret_cast<uint64_t>(allocator->user_data));

	Ngs2ContextBufferInfo buf {};
	buf.host_buffer      = nullptr;
	buf.host_buffer_size = 0;
	buf.user_data        = allocator->user_data;

	Ngs2RackQueryBufferSize(rack_id, option, &buf);

	EXIT_NOT_IMPLEMENTED(buf.host_buffer_size == 0);

	int result = allocator->alloc_handler(&buf);

	EXIT_NOT_IMPLEMENTED(result != OK);
	EXIT_NOT_IMPLEMENTED(buf.host_buffer == nullptr);

	result = Ngs2RackCreate(system_handle, rack_id, option, &buf, handle);

	if (result == OK) {
		auto* rack      = static_cast<Ngs2RackInternal*>(buf.host_buffer);
		rack->allocator = *allocator;
	}

	return result;
}

int KYTY_SYSV_ABI Ngs2RackDestroy(uintptr_t rack_handle, Ngs2ContextBufferInfo* buffer_info) {
	PRINT_NAME();
	LOGF("\t rack_handle = 0x%016" PRIx64 "\n", static_cast<uint64_t>(rack_handle));

	if (rack_handle != 0) {
		auto* rack   = reinterpret_cast<Ngs2RackInternal*>(rack_handle);
		auto* voices = reinterpret_cast<Ngs2VoiceInternal*>(rack + 1);
		const auto max_voices = rack->option.common.max_voices;
		for (uint32_t i = 0; i < max_voices; i++) {
			voices[i].~Ngs2VoiceInternal();
		}

		if (g_racks_list == rack) {
			g_racks_list = rack->next;
		} else {
			for (auto* it = g_racks_list; it != nullptr; it = it->next) {
				if (it->next == rack) {
					it->next = rack->next;
					break;
				}
			}
		}
	}

	if (buffer_info != nullptr) {
		std::memset(buffer_info, 0, sizeof(Ngs2ContextBufferInfo));
	}

	return OK;
}

int KYTY_SYSV_ABI Ngs2RackLock(uintptr_t rack_handle) {
	PRINT_NAME();
	LOGF("\t rack_handle = 0x%016" PRIx64 "\n", static_cast<uint64_t>(rack_handle));

	EXIT_NOT_IMPLEMENTED(rack_handle == 0);

	auto* rack = reinterpret_cast<Ngs2RackInternal*>(rack_handle);

	EXIT_NOT_IMPLEMENTED(rack->ngs == nullptr);

	rack->ngs->mutex.Lock();

	return OK;
}

int KYTY_SYSV_ABI Ngs2RackUnlock(uintptr_t rack_handle) {
	PRINT_NAME();
	LOGF("\t rack_handle = 0x%016" PRIx64 "\n", static_cast<uint64_t>(rack_handle));

	EXIT_NOT_IMPLEMENTED(rack_handle == 0);

	auto* rack = reinterpret_cast<Ngs2RackInternal*>(rack_handle);

	EXIT_NOT_IMPLEMENTED(rack->ngs == nullptr);

	rack->ngs->mutex.Unlock();

	return OK;
}

int KYTY_SYSV_ABI Ngs2SystemRender(uintptr_t system_handle, const Ngs2RenderBufferInfo* buffer_info,
                                   uint32_t num_buffer_info) {
	static std::atomic_uint32_t render_log_count = 0;
	const auto log_index     = render_log_count.fetch_add(1, std::memory_order_relaxed);
	const bool log_this_call = (log_index < 16 || (log_index % 600) == 0);

	if (log_this_call) {
		PRINT_NAME();
		LOGF("\t call_count      = %" PRIu32 "\n", log_index + 1);
	}

	EXIT_NOT_IMPLEMENTED(buffer_info == nullptr);
	EXIT_NOT_IMPLEMENTED(system_handle == 0);
	EXIT_NOT_IMPLEMENTED(num_buffer_info == 0);

	auto* ngs = reinterpret_cast<Ngs2Internal*>(system_handle);

	Common::LockGuard lock(ngs->mutex);

	for (uint32_t i = 0; i < num_buffer_info; i++) {
		if (buffer_info[i].buffer == nullptr || buffer_info[i].buffer_size == 0) {
			continue;
		}
		std::memset(buffer_info[i].buffer, 0, buffer_info[i].buffer_size);

		auto channels = buffer_info[i].num_channels;
		if (channels == 0 || channels > 8) {
			channels = 2;
		}
		const auto dest_rate = (ngs->option.sample_rate != 0 ? ngs->option.sample_rate : 48000u);
		const auto grain =
		    (ngs->option.num_grain_samples != 0 ? ngs->option.num_grain_samples : 256u);
		const auto capacity_frames = std::min<uint32_t>(
		    grain, static_cast<uint32_t>(buffer_info[i].buffer_size / (channels * sizeof(float))));
		if (capacity_frames == 0) {
			continue;
		}

		std::vector<float> accum(static_cast<size_t>(capacity_frames) * channels, 0.0f);
		for (auto* rack = g_racks_list; rack != nullptr; rack = rack->next) {
			if (rack->ngs != ngs) {
				continue;
			}
			auto* voices = reinterpret_cast<Ngs2VoiceInternal*>(rack + 1);
			for (uint32_t v = 0; v < rack->option.common.max_voices; v++) {
				Ngs2MixVoiceIntoGrain(accum.data(), capacity_frames, channels, &voices[v],
				                      dest_rate);
			}
		}
		std::memcpy(buffer_info[i].buffer, accum.data(),
		            capacity_frames * channels * sizeof(float));
	}

	for (auto* rack = g_racks_list; rack != nullptr; rack = rack->next) {
		if (rack->ngs == ngs) {
			auto* voices = reinterpret_cast<Ngs2VoiceInternal*>(rack + 1);

			for (uint32_t i = 0; i < rack->option.common.max_voices; i++) {
				auto& voice = voices[i];
				switch (voice.event) {
					case Ngs2VoicePlayEvent::None:
						if (voice.state == Ngs2VoicePlayState::Playing ||
						    voice.state == Ngs2VoicePlayState::Stopped) {
							if (voice.pcm.empty()) {
								voice.state = Ngs2VoicePlayState::Empty;
							}
						}
						break;
					case Ngs2VoicePlayEvent::Play:
						if (voice.state == Ngs2VoicePlayState::Empty ||
						    voice.state == Ngs2VoicePlayState::Stopped) {
							voice.state      = Ngs2VoicePlayState::Playing;
							voice.pcm_cursor = 0;
						}
						break;
					case Ngs2VoicePlayEvent::Pause:
						if (voice.state == Ngs2VoicePlayState::Playing) {
							voice.state = Ngs2VoicePlayState::Paused;
						}
						break;
					case Ngs2VoicePlayEvent::Resume:
						if (voice.state == Ngs2VoicePlayState::Paused) {
							voice.state = Ngs2VoicePlayState::Playing;
						}
						break;
					case Ngs2VoicePlayEvent::Stop:
						if (voice.state == Ngs2VoicePlayState::Playing) {
							voice.state = Ngs2VoicePlayState::Stopped;
						}
						break;
					case Ngs2VoicePlayEvent::StopImm:
					case Ngs2VoicePlayEvent::Kill:
						voice.state      = Ngs2VoicePlayState::Empty;
						voice.pcm.clear();
						voice.pcm_cursor = 0;
						voice.loop_start = -1;
						voice.loop_end   = -1;
						break;
				}
				voice.event = Ngs2VoicePlayEvent::None;
			}
		}
	}

	return OK;
}

int KYTY_SYSV_ABI Ngs2ParseWaveformData(const void* data, size_t data_size,
                                        Ngs2WaveformInfo* info) {
	PRINT_NAME();
	LOGF("\t data = 0x%016" PRIx64 ", data_size = 0x%016" PRIx64 "\n",
	     reinterpret_cast<uint64_t>(data), static_cast<uint64_t>(data_size));

	EXIT_NOT_IMPLEMENTED(info == nullptr);

	std::memset(info, 0, sizeof(Ngs2WaveformInfo));
	info->format.waveform_type = 0x80;
	info->format.num_channels  = 1;
	info->format.sample_rate   = 48000;
	info->data_size =
	    static_cast<uint32_t>(std::min<size_t>(data_size, std::numeric_limits<uint32_t>::max()));
	info->num_audio_unit_samples   = 1;
	info->num_audio_unit_per_frame = 1;
	info->num_audio_frame_samples  = 1;
	return OK;
}

int KYTY_SYSV_ABI Ngs2CalcWaveformBlock(const Ngs2WaveformFormat* format, uint32_t sample_pos,
                                        uint32_t num_samples, Ngs2WaveformBlock* block) {
	PRINT_NAME();
	LOGF("\t format = 0x%016" PRIx64 ", sample_pos = %" PRIu32 ", num_samples = %" PRIu32 "\n",
	     reinterpret_cast<uint64_t>(format), sample_pos, num_samples);

	EXIT_NOT_IMPLEMENTED(block == nullptr);

	std::memset(block, 0, sizeof(Ngs2WaveformBlock));
	block->num_samples = num_samples;
	return OK;
}

int KYTY_SYSV_ABI Ngs2PanInit(Ngs2PanWork* work, const float* speaker_angles, float unit_angle,
                              uint32_t num_speakers) {
	PRINT_NAME();
	LOGF("\t work = 0x%016" PRIx64 ", num_speakers = %" PRIu32 "\n",
	     reinterpret_cast<uint64_t>(work), num_speakers);

	EXIT_NOT_IMPLEMENTED(work == nullptr);

	std::memset(work, 0, sizeof(Ngs2PanWork));
	work->unit_angle   = unit_angle;
	work->num_speakers = std::min<uint32_t>(num_speakers, 8);
	if (speaker_angles != nullptr) {
		for (uint32_t i = 0; i < work->num_speakers; i++) {
			work->speaker_angles[i] = speaker_angles[i];
		}
	}
	return OK;
}

int KYTY_SYSV_ABI Ngs2PanGetVolumeMatrix(Ngs2PanWork* work, const Ngs2PanParam* params,
                                         uint32_t num_params, uint32_t matrix_format,
                                         float* out_volume_matrix) {
	PRINT_NAME();
	LOGF("\t work = 0x%016" PRIx64 ", params = 0x%016" PRIx64 ", num_params = %" PRIu32
	     ", matrix_format = %" PRIu32 "\n",
	     reinterpret_cast<uint64_t>(work), reinterpret_cast<uint64_t>(params), num_params,
	     matrix_format);

	EXIT_NOT_IMPLEMENTED(out_volume_matrix == nullptr && num_params != 0);

	const auto channels = (matrix_format == 0 ? 2u : std::min<uint32_t>(matrix_format, 8));
	for (uint32_t p = 0; p < num_params; p++) {
		for (uint32_t c = 0; c < channels; c++) {
			out_volume_matrix[p * channels + c] = (c == 0 ? 1.0f : 0.0f);
		}
	}

	return OK;
}

int KYTY_SYSV_ABI Ngs2GeomResetListenerParam(Ngs2GeomListenerParam* out_listener_param) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(out_listener_param == nullptr);

	std::memset(out_listener_param, 0, sizeof(Ngs2GeomListenerParam));
	out_listener_param->orient_front.z = 1.0f;
	out_listener_param->orient_up.y    = 1.0f;
	out_listener_param->sound_speed    = 343.0f;

	return OK;
}

int KYTY_SYSV_ABI Ngs2GeomResetSourceParam(Ngs2GeomSourceParam* out_source_param) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(out_source_param == nullptr);

	std::memset(out_source_param, 0, sizeof(Ngs2GeomSourceParam));
	out_source_param->direction.z                = 1.0f;
	out_source_param->cone.inner_level           = 1.0f;
	out_source_param->cone.inner_angle           = 360.0f;
	out_source_param->cone.outer_level           = 1.0f;
	out_source_param->cone.outer_angle           = 360.0f;
	out_source_param->rolloff.model              = 0;
	out_source_param->rolloff.max_distance       = 1000000.0f;
	out_source_param->rolloff.rolloff_factor     = 1.0f;
	out_source_param->rolloff.reference_distance = 1.0f;
	out_source_param->doppler_factor             = 1.0f;
	out_source_param->fbw_level                  = 1.0f;
	out_source_param->lfe_level                  = 1.0f;
	out_source_param->max_level                  = 1.0f;
	out_source_param->min_level                  = 0.0f;
	out_source_param->num_speakers               = 2;
	out_source_param->matrix_format              = 2;

	return OK;
}

int KYTY_SYSV_ABI Ngs2GeomCalcListener(const Ngs2GeomListenerParam* param,
                                       Ngs2GeomListenerWork* out_work, uint32_t flags) {
	PRINT_NAME();
	LOGF("\t flags = 0x%08" PRIx32 "\n", flags);

	EXIT_NOT_IMPLEMENTED(param == nullptr);
	EXIT_NOT_IMPLEMENTED(out_work == nullptr);

	std::memset(out_work, 0, sizeof(Ngs2GeomListenerWork));
	for (uint32_t i = 0; i < 4; i++) {
		out_work->matrix[i][i] = 1.0f;
	}
	out_work->velocity    = param->velocity;
	out_work->sound_speed = (param->sound_speed > 0.0f ? param->sound_speed : 343.0f);
	out_work->coordinate  = flags & 0x1u;

	return OK;
}

int KYTY_SYSV_ABI Ngs2GeomApply(const Ngs2GeomListenerWork* listener,
                                const Ngs2GeomSourceParam* source, Ngs2GeomAttribute* out_attrib,
                                uint32_t flags) {
	PRINT_NAME();
	LOGF("\t flags = 0x%08" PRIx32 "\n", flags);

	EXIT_NOT_IMPLEMENTED(listener == nullptr);
	EXIT_NOT_IMPLEMENTED(source == nullptr);
	EXIT_NOT_IMPLEMENTED(out_attrib == nullptr);

	std::memset(out_attrib, 0, sizeof(Ngs2GeomAttribute));
	out_attrib->pitch_ratio         = 1.0f;
	out_attrib->a3d_attrib.position = source->position;
	out_attrib->a3d_attrib.volume   = std::max(source->min_level, source->max_level);

	const auto channels =
	    std::min<uint32_t>((source->matrix_format == 0 ? 2u : source->matrix_format), 8);
	const auto level = (source->max_level > 0.0f ? source->max_level : 1.0f);
	for (uint32_t ch = 0; ch < channels; ch++) {
		out_attrib->level[ch * 8 + ch] = level;
	}

	return OK;
}

int KYTY_SYSV_ABI Ngs2RackGetVoiceHandle(uintptr_t rack_handle, uint32_t voice_id,
                                         uintptr_t* handle) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(handle == nullptr);
	EXIT_NOT_IMPLEMENTED(rack_handle == 0);

	LOGF("\t voice_id = %u\n", voice_id);

	auto* rack   = reinterpret_cast<Ngs2RackInternal*>(rack_handle);
	auto* voices = reinterpret_cast<Ngs2VoiceInternal*>(rack_handle + sizeof(Ngs2RackInternal));

	if (voice_id >= rack->option.common.max_voices) {
		LOGF("\t warning: voice_id %u >= max_voices %u, using last available stub voice\n",
		     voice_id, rack->option.common.max_voices);
		if (rack->option.common.max_voices == 0) {
			return -1;
		}
		voice_id = rack->option.common.max_voices - 1;
	}

	EXIT_IF(voices[voice_id].rack != rack);

	*handle = reinterpret_cast<uintptr_t>(voices + voice_id);

	return OK;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int KYTY_SYSV_ABI Ngs2VoiceControl(uintptr_t voice_handle, const Ngs2VoiceParamHeader* param_list) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(param_list == nullptr);
	EXIT_NOT_IMPLEMENTED(voice_handle == 0);

	auto* voice = reinterpret_cast<Ngs2VoiceInternal*>(voice_handle);

	Common::LockGuard lock(voice->rack->ngs->mutex);

	const auto* param = param_list;

	for (;;) {
		LOGF("\t id   = 0x%08" PRIx32 "\n"
		     "\t size = %" PRIu16 "\n"
		     "\t next = %" PRId16 "\n",
		     param->id, param->size, param->next);

		auto rack_id = param->id >> 16u;

		EXIT_NOT_IMPLEMENTED(((param->id >> 15u) & 0x1u) != 0);

		switch (rack_id) {
			case 0x0000: {
				auto cid = param->id & 0x7fffu;
				switch (cid) {
					case 0x0001: {
						EXIT_NOT_IMPLEMENTED(param->size != sizeof(Ngs2VoiceMatrixLevelsParam));
						const auto* ml = reinterpret_cast<const Ngs2VoiceMatrixLevelsParam*>(param);
						LOGF("\t matrix_id  = %u\n"
						     "\t num_levels = %u\n"
						     "\t levels     = 0x%016" PRIx64 "\n",
						     ml->matrix_id, ml->num_levels, reinterpret_cast<uint64_t>(ml->levels));
						break;
					}
					case 0x0002: {
						EXIT_NOT_IMPLEMENTED(param->size != sizeof(Ngs2VoicePortVolumeParam));
						const auto* volume =
						    reinterpret_cast<const Ngs2VoicePortVolumeParam*>(param);
						LOGF("\t port  = %u\n"
						     "\t level = %f\n",
						     volume->port, volume->level);
						break;
					}
					case 0x0003: {
						EXIT_NOT_IMPLEMENTED(param->size != sizeof(Ngs2VoicePortMatrixParam));
						const auto* pm = reinterpret_cast<const Ngs2VoicePortMatrixParam*>(param);
						LOGF("\t port      = %u\n"
						     "\t matrix_id = %d\n",
						     pm->port, pm->matrix_id);
						break;
					}
					case 0x0004: {
						EXIT_NOT_IMPLEMENTED(param->size != sizeof(Ngs2VoicePortDelayParam));
						const auto* delay = reinterpret_cast<const Ngs2VoicePortDelayParam*>(param);
						LOGF("\t port        = %u\n"
						     "\t num_samples = %u\n",
						     delay->port, delay->num_samples);
						break;
					}
					case 0x0005: {
						EXIT_NOT_IMPLEMENTED(param->size != sizeof(Ngs2VoicePatchParam));
						const auto* patch = reinterpret_cast<const Ngs2VoicePatchParam*>(param);
						LOGF("\t connect->port          = %u\n"
						     "\t connect->dest_input_id = %u\n"
						     "\t connect->dest_handle   = 0x%016" PRIx64 "\n",
						     patch->port, patch->dest_input_id, patch->dest_handle);
						break;
					}
					case 0x0006: {
						EXIT_NOT_IMPLEMENTED(param->size != sizeof(Ngs2VoiceEventParam));
						const auto* event = reinterpret_cast<const Ngs2VoiceEventParam*>(param);
						switch (event->event_id) {
							case 0x0001: voice->event = Ngs2VoicePlayEvent::Play; break;
							case 0x0002: voice->event = Ngs2VoicePlayEvent::Stop; break;
							case 0x0004: voice->event = Ngs2VoicePlayEvent::StopImm; break;
							case 0x0008: voice->event = Ngs2VoicePlayEvent::Kill; break;
							case 0x0010: voice->event = Ngs2VoicePlayEvent::Pause; break;
							case 0x0020: voice->event = Ngs2VoicePlayEvent::Resume; break;
							default: EXIT("unknown event_id: 0x%08" PRIx32 "\n", event->event_id);
						}
						LOGF("\t event = %u\n", event->event_id);
						break;
					}
					case 0x0007: {
						EXIT_NOT_IMPLEMENTED(param->size != sizeof(Ngs2VoiceCallbackParam));
						const auto* callback =
						    reinterpret_cast<const Ngs2VoiceCallbackParam*>(param);
						voice->callback       = callback->callback;
						voice->callback_data  = callback->callback_data;
						voice->callback_flags = callback->flags;
						LOGF("\t callback      = 0x%016" PRIx64 "\n"
						     "\t callback_data = 0x%016" PRIx64 "\n"
						     "\t flags         = 0x%08" PRIx32 "\n",
						     static_cast<uint64_t>(voice->callback),
						     static_cast<uint64_t>(voice->callback_data), voice->callback_flags);
						break;
					}
					default: EXIT("unknown id: 0x%04" PRIx32 "\n", cid);
				}
				break;
			}
			case 0x1000: {
				EXIT_NOT_IMPLEMENTED(voice->rack->type != Ngs2RackType::Sampler);
				auto cid = param->id & 0xffffu;
				switch (cid) {
					case 0x0000: Ngs2ApplySetupParam(voice, param); break;
					case 0x0001: Ngs2ApplyWaveformBlocksParam(voice, param); break;
					case 0x0004:
					case 0x0005:
						if (param->size >= sizeof(Ngs2VoiceParamHeader) + sizeof(float)) {
							float pitch = 1.0f;
							std::memcpy(&pitch,
							            reinterpret_cast<const uint8_t*>(param) +
							                sizeof(Ngs2VoiceParamHeader),
							            sizeof(pitch));
							if (std::isfinite(pitch) && pitch > 0.0f) {
								voice->pitch_ratio = pitch;
							}
						}
						break;
					default: break;
				}
				break;
			}
			case 0x2000: EXIT_NOT_IMPLEMENTED(voice->rack->type != Ngs2RackType::Submixer); break;
			case 0x2001: EXIT_NOT_IMPLEMENTED(voice->rack->type != Ngs2RackType::Reverb); break;
			case 0x3000: EXIT_NOT_IMPLEMENTED(voice->rack->type != Ngs2RackType::Mastering); break;
			case 0x4000: {
				EXIT_NOT_IMPLEMENTED(!Ngs2RackIsCustom(voice->rack->type));
				auto cid       = param->id & 0xffffu;
				auto module_id = (cid >> 8u) & 0xffu;
				auto ctl_id    = (cid >> 5u) & 0x7u;
				auto module_no = cid & 0x1fu;
				LOGF("\t custom module_id = 0x%02" PRIx32 ", ctl_id = 0x%" PRIx32
				     ", module_no = %" PRIu32 "\n",
				     module_id, ctl_id, module_no);
				if (cid == 0x1300u && param->size >= 16) {
					// Continuous-control trailing floats often encode gain.
					const auto* floats = reinterpret_cast<const float*>(
					    reinterpret_cast<const uint8_t*>(param) + sizeof(Ngs2VoiceParamHeader));
					const auto count =
					    (param->size - sizeof(Ngs2VoiceParamHeader)) / sizeof(float);
					for (size_t i = 0; i < count; i++) {
						if (std::isfinite(floats[i]) && floats[i] > 0.0f && floats[i] <= 4.0f) {
							voice->gain = floats[i];
						}
					}
				}
				break;
			}
			case 0x4001: {
				EXIT_NOT_IMPLEMENTED(voice->rack->type != Ngs2RackType::CustomSampler);
				auto cid = param->id & 0xffffu;
				switch (cid) {
					case 0x0000: Ngs2ApplySetupParam(voice, param); break;
					case 0x0001: Ngs2ApplyWaveformBlocksParam(voice, param); break;
					case 0x0004:
					case 0x0005:
						if (param->size >= sizeof(Ngs2VoiceParamHeader) + sizeof(float)) {
							float pitch = 1.0f;
							std::memcpy(&pitch,
							            reinterpret_cast<const uint8_t*>(param) +
							                sizeof(Ngs2VoiceParamHeader),
							            sizeof(pitch));
							if (std::isfinite(pitch) && pitch > 0.0f) {
								voice->pitch_ratio = pitch;
							}
						}
						break;
					default: break;
				}
				break;
			}
			case 0x4002:
				EXIT_NOT_IMPLEMENTED(voice->rack->type != Ngs2RackType::CustomSubmixer);
				break;
			default: EXIT("unknown rack_id: 0x%" PRIx32 "\n", rack_id);
		}

		if (param->next == 0) {
			break;
		}
		param = reinterpret_cast<const Ngs2VoiceParamHeader*>(reinterpret_cast<uintptr_t>(param) +
		                                                      param->next);
	}

	return OK;
}

int KYTY_SYSV_ABI Ngs2VoiceRunCommands(uintptr_t voice_handle, const void* commands,
                                       uint32_t num_commands, uint32_t flags) {
	PRINT_NAME();

	(void)voice_handle;
	(void)commands;
	(void)num_commands;
	(void)flags;

	return OK;
}

int KYTY_SYSV_ABI Ngs2VoiceGetState(uintptr_t voice_handle, Ngs2VoiceState* state,
                                    size_t state_size) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(state == nullptr);
	EXIT_NOT_IMPLEMENTED(voice_handle == 0);

	auto* voice = reinterpret_cast<Ngs2VoiceInternal*>(voice_handle);

	Common::LockGuard lock(voice->rack->ngs->mutex);

	switch (voice->rack->type) {
		case Ngs2RackType::Sampler:
		case Ngs2RackType::CustomSampler: {
			if (state_size != sizeof(Ngs2SamplerVoiceState)) {
				LOGF("\t warning: sampler state_size = 0x%016" PRIx64 ", expected 0x%016" PRIx64
				     "\n",
				     static_cast<uint64_t>(state_size),
				     static_cast<uint64_t>(sizeof(Ngs2SamplerVoiceState)));
			}
			std::memset(state, 0, state_size);

			state->state_flags = Ngs2GetStateFlags(voice);
			if (state_size < sizeof(Ngs2SamplerVoiceState)) {
				LOGF("\t state_flags = %u\n", state->state_flags);
				break;
			}

			auto* sampler                = reinterpret_cast<Ngs2SamplerVoiceState*>(state);
			sampler->envelope_height     = 1.0f;
			sampler->peak_height         = 0.0f;
			sampler->reserved            = 0;
			sampler->num_decoded_samples = 0;
			sampler->user_data           = 0;
			sampler->waveform_data       = nullptr;
			LOGF("\t state_flags = %u\n", sampler->voice_state.state_flags);
			break;
		}
		default: EXIT("unknown type: %s\n", Common::EnumName(voice->rack->type).c_str());
	}

	return OK;
}

int KYTY_SYSV_ABI Ngs2VoiceGetStateFlags(uintptr_t voice_handle, uint32_t* state_flags) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(state_flags == nullptr);
	EXIT_NOT_IMPLEMENTED(voice_handle == 0);

	auto* voice = reinterpret_cast<Ngs2VoiceInternal*>(voice_handle);

	Common::LockGuard lock(voice->rack->ngs->mutex);

	*state_flags = Ngs2GetStateFlags(voice);

	LOGF("\t state_flags = %u\n", *state_flags);

	return OK;
}

} // namespace Ngs2

} // namespace Libs::Audio
