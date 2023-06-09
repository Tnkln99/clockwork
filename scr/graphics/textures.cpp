#include "textures.hpp"
#include "initializers.hpp"

#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace cw::graphics::utils{
    bool loadImageFromFile(Engine &engine, const char *file, AllocatedImage &outImage) {
        int texWidth, texHeight, texChannels;

        stbi_uc* pixels = stbi_load(file, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            std::cout << "Failed to load texture file " << file << std::endl;
            return false;
        }

        void* pixel_ptr = pixels;
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        VkFormat image_format = VK_FORMAT_R8G8B8A8_SRGB;

        AllocatedBuffer stagingBuffer = engine.createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

        void* data;
        vmaMapMemory(engine.mAllocator, stagingBuffer.allocation, &data);

        memcpy(data, pixel_ptr, static_cast<size_t>(imageSize));

        vmaUnmapMemory(engine.mAllocator, stagingBuffer.allocation);

        stbi_image_free(pixels);

        VkExtent3D imageExtent;
        imageExtent.width = static_cast<uint32_t>(texWidth);
        imageExtent.height = static_cast<uint32_t>(texHeight);
        imageExtent.depth = 1;

        VkImageCreateInfo dimg_info = init::imageCreateInfo(image_format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, imageExtent);

        AllocatedImage newImage{};

        VmaAllocationCreateInfo dimg_allocinfo = {};
        dimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        //allocate and create the image
        vmaCreateImage(engine.mAllocator, &dimg_info,
                       &dimg_allocinfo,&newImage.image,
                       &newImage.allocation, nullptr);

        //transition image to transfer-receiver
        engine.immediateSubmit([&](VkCommandBuffer cmd) {
            VkImageSubresourceRange range;
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.baseMipLevel = 0;
            range.levelCount = 1;
            range.baseArrayLayer = 0;
            range.layerCount = 1;

            VkImageMemoryBarrier imageBarrier_toTransfer = {};
            imageBarrier_toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

            imageBarrier_toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageBarrier_toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageBarrier_toTransfer.image = newImage.image;
            imageBarrier_toTransfer.subresourceRange = range;

            imageBarrier_toTransfer.srcAccessMask = 0;
            imageBarrier_toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            //barrier the image into the transfer-receive layout
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 0, nullptr,
                                 0, nullptr,
                                 1, &imageBarrier_toTransfer);

            VkBufferImageCopy copyRegion = {};
            copyRegion.bufferOffset = 0;
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;

            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.mipLevel = 0;
            copyRegion.imageSubresource.baseArrayLayer = 0;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageExtent = imageExtent;

            //copy the buffer into the image
            vkCmdCopyBufferToImage(cmd, stagingBuffer.buffer, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

            VkImageMemoryBarrier imageBarrier_toReadable = imageBarrier_toTransfer;

            imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imageBarrier_toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            //barrier the image into the shader readable layout
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);
        });


        engine.mMainDeletionQueue.push_function([=]() {
            vmaDestroyImage(engine.mAllocator, newImage.image, newImage.allocation);
        });

        vmaDestroyBuffer(engine.mAllocator, stagingBuffer.buffer, stagingBuffer.allocation);

        std::cout << "Texture loaded successfully " << file << std::endl;

        outImage = newImage;
        return true;
    }
}
