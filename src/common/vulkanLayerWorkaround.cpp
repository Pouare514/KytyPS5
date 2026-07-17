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

constexpr const char* kMedalVulkanLayerNames[] = {
    "VK_LAYER_MEDAL_capture",
    "VK_LAYER_MEDAL_HOOK",
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

void DisableMedalVulkanLayer() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::string value;
	if (const char* existing = std::getenv(kLoaderLayersDisable); existing != nullptr && existing[0] != '\0') {
		value = existing;
	}

	for (const char* layer : kMedalVulkanLayerNames) {
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
	(void)kMedalVulkanLayerNames;
	(void)kLoaderLayersDisable;
#endif
}

} // namespace Common
