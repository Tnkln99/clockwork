#ifndef CLOCKWORK_TYPES_HPP
#define CLOCKWORK_TYPES_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vk_mem_alloc.h>

struct AllocatedBuffer {
    VkBuffer mBuffer;
    VmaAllocation mAllocation;
};

#endif //CLOCKWORK_TYPES_HPP
