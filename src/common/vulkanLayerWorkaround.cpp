#include "common/vulkanLayerWorkaround.h"

#include "common/common.h"

#include <cstdlib>
#include <string>
#include <string_view>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <stdlib.h> // IWYU pragma: keep
#endif

namespace Common {

namespace {

constexpr const char* kKnownVulkanLayerDisableGlobs[] = {
    "VK_LAYER_MEDAL_capture",
    "VK_LAYER_MEDAL_HOOK",
    "VK_LAYER_NV_present",
    "*nvspcap*",
    "*Steam*Overlay*",
    "*steam*overlay*",
    "VK_LAYER_VALVE_steam_overlay",
    "*EOS_Overlay*",
    "VK_LAYER_OW_OVERLAY",
    "*Discord*",
    "*discord*",
    "*obs*",
    "*OBS*",
};
constexpr const char* kLoaderLayersDisable = "VK_LOADER_LAYERS_DISABLE";

bool LayerListContains(std::string_view list, std::string_view layer) {
	if (layer.empty()) {
		return false;
	}

	for (size_t pos = 0; pos <= list.size();) {
		const size_t end = list.find(',', pos);
		const auto   token =
		    list.substr(pos, end == std::string_view::npos ? list.size() - pos : end - pos);
		if (token == layer) {
			return true;
		}
		if (end == std::string_view::npos) {
			break;
		}
		pos = end + 1;
	}

	return false;
}

} // namespace

void DisableKnownVulkanLayers() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::string value;
	if (const char* existing = std::getenv(kLoaderLayersDisable);
	    existing != nullptr && existing[0] != '\0') {
		value = existing;
	}

	for (const char* layer : kKnownVulkanLayerDisableGlobs) {
		if (!LayerListContains(value, layer)) {
			if (!value.empty()) {
				value += ',';
			}
			value += layer;
		}
	}

	if (!value.empty()) {
		(void)::_putenv_s(kLoaderLayersDisable, value.c_str());
	}
#else
	(void)kKnownVulkanLayerDisableGlobs;
	(void)kLoaderLayersDisable;
#endif
}

void DisableMedalVulkanLayer() {
	DisableKnownVulkanLayers();
}

const char* GetVulkanLayersDisableEnv() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return std::getenv(kLoaderLayersDisable);
#else
	return nullptr;
#endif
}

} // namespace Common
