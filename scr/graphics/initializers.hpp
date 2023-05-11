#ifndef CLOCKWORK_INITIALIZERS_HPP
#define CLOCKWORK_INITIALIZERS_HPP

#include "types.hpp"

namespace cw::graphics::init {
    VkCommandPoolCreateInfo commandPoolCreateInfo(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags = 0);

    VkCommandBufferAllocateInfo commandBufferAllocateInfo(VkCommandPool pool, uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);
}

#endif //CLOCKWORK_INITIALIZERS_HPP
