#ifndef CLOCKWORK_TYPES_HPP
#define CLOCKWORK_TYPES_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vk_mem_alloc.h>

struct AllocatedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
};

struct AllocatedImage {
    VkImage image;
    VmaAllocation allocation;
};


#endif //CLOCKWORK_TYPES_HPP
