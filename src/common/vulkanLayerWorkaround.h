#ifndef KYTY_COMMON_VULKAN_LAYER_WORKAROUND_H_
#define KYTY_COMMON_VULKAN_LAYER_WORKAROUND_H_

namespace Common {

// Disables known third-party Vulkan layers for this process (Medal, NVIDIA capture, Steam, etc.).
void DisableKnownVulkanLayers();

// Backward-compatible alias.
void DisableMedalVulkanLayer();

// Returns getenv("VK_LOADER_LAYERS_DISABLE") after DisableKnownVulkanLayers (may be null).
const char* GetVulkanLayersDisableEnv();

} // namespace Common

#endif /* KYTY_COMMON_VULKAN_LAYER_WORKAROUND_H_ */
