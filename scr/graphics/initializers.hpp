#ifndef CLOCKWORK_INITIALIZERS_HPP
#define CLOCKWORK_INITIALIZERS_HPP

#include "types.hpp"

namespace cw::graphics::init {

    VkCommandPoolCreateInfo commandPoolCreateInfo(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags = 0) ;

    VkCommandBufferAllocateInfo commandBufferAllocateInfo(VkCommandPool pool, uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) ;

    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(VkShaderStageFlagBits stage, VkShaderModule shaderModule) ;

    VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo() ;

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo(VkPrimitiveTopology topology) ;

    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo(VkPolygonMode polygonMode) ;

    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo() ;

    VkPipelineColorBlendAttachmentState colorBlendAttachmentState() ;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo() ;

    VkFenceCreateInfo fenceCreateInfo(VkFenceCreateFlags flags = 0) ;

    VkSemaphoreCreateInfo semaphoreCreateInfo(VkSemaphoreCreateFlags flags = 0) ;

    VkSubmitInfo submitInfo(VkCommandBuffer *cmd) ;

    VkPresentInfoKHR presentInfo() ;

    VkRenderPassBeginInfo
    renderPassBeginInfo(VkRenderPass renderPass, VkExtent2D windowExtent, VkFramebuffer framebuffer) ;

    VkImageCreateInfo imageCreateInfo(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent) ;

    VkImageViewCreateInfo imageViewCreateInfo(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags) ;

    VkPipelineDepthStencilStateCreateInfo
    depthStencilStateCreateInfo(bool bDepthTest, bool bDepthWrite, VkCompareOp compareOp) ;

    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding(VkDescriptorType type, VkShaderStageFlags stageFlags, uint32_t binding);

    VkWriteDescriptorSet writeDescriptorBuffer(VkDescriptorType type, VkDescriptorSet dstSet, VkDescriptorBufferInfo* bufferInfo , uint32_t binding);
}

#endif //CLOCKWORK_INITIALIZERS_HPP