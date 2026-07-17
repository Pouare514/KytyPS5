#ifndef KYTY_COMMON_VULKAN_LAYER_WORKAROUND_H_
#define KYTY_COMMON_VULKAN_LAYER_WORKAROUND_H_

namespace Common {

// Disables known Medal Vulkan layers for this process only (orphaned registry entries).
void DisableMedalVulkanLayer();

} // namespace Common

#endif /* KYTY_COMMON_VULKAN_LAYER_WORKAROUND_H_ */
