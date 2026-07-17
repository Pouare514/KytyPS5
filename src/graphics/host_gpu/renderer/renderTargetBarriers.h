#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERTARGETBARRIERS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERTARGETBARRIERS_H_

#include <vulkan/vulkan_core.h>

namespace Libs::Graphics {

struct VulkanImage;

void GraphicsRenderTextureBarrier(VkCommandBuffer vk_buffer, VulkanImage* image);
void GraphicsRenderColorImageBarrier(VkCommandBuffer vk_buffer, VulkanImage* image,
                                     VkImageLayout new_layout);
void GraphicsRenderDepthStencilImageBarrier(VkCommandBuffer vk_buffer, VulkanImage* image,
                                            VkImageLayout new_layout);
void GraphicsRenderDepthStencilBarrier(VkCommandBuffer vk_buffer, VulkanImage* image);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERTARGETBARRIERS_H_
