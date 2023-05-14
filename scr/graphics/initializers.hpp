#ifndef CLOCKWORK_INITIALIZERS_HPP
#define CLOCKWORK_INITIALIZERS_HPP

#include "types.hpp"

namespace cw::graphics::init {
    VkCommandBufferAllocateInfo commandBufferAllocateInfo(VkCommandPool pool, uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    VkCommandPoolCreateInfo commandPoolCreateInfo(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags = 0);

    VkFenceCreateInfo fenceCreateInfo(VkFenceCreateFlags flags = 0);

    VkSemaphoreCreateInfo semaphoreCreateInfo(VkSemaphoreCreateFlags flags = 0);

    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(VkShaderStageFlagBits stage, VkShaderModule shaderModule);

    VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo(VkPrimitiveTopology topology);

    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo(VkPolygonMode polygonMode);

    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo();

    VkPipelineColorBlendAttachmentState colorBlendAttachmentState();

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo();
}

#endif //CLOCKWORK_INITIALIZERS_HPP
