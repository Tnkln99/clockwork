#ifndef CLOCKWORK_ENGINE_HPP
#define CLOCKWORK_ENGINE_HPP

#include "pipeline.hpp"

#include <vector>
#include <deque>
#include <functional>

namespace cw::graphics {
    struct DeletionQueue
    {
        std::deque<std::function<void()>> deletors;

        void push_function(std::function<void()>&& function) {
            deletors.push_back(function);
        }

        void flush() {
            // reverse iterate the deletion queue to execute all the functions
            for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
                (*it)(); //call the function
            }

            deletors.clear();
        }
    };


    class Engine {
    public:
        void init();

        void run();
        void draw();

        void cleanup();

        VmaAllocator mAllocator; //vma lib allocator

        VkSwapchainKHR mSwapchain{}; // from other articles
        // image format expected by the windowing system
        VkFormat mSwapchainImageFormat{};
        // array of images from the swapchain
        std::vector<VkImage> mSwapchainImages{};
        // array of image-views from the swapchain
        std::vector<VkImageView> mSwapchainImageViews{};


        VkInstance mInstance{};
        VkDebugUtilsMessengerEXT mDebugMessenger{};
        VkPhysicalDevice mChosenGpu{};
        VkDevice mDevice{}; // vulkan device for commands
        VkSurfaceKHR mSurface{};

        VkQueue mGraphicsQueue{}; //queue we will submit to
        uint32_t mGraphicsQueueFamily; //family of that queue

        VkCommandPool mCommandPool; //the command pool for our commands
        VkCommandBuffer mMainCommandBuffer; //the buffer we will record into

        VkRenderPass mRenderpass;
        std::vector<VkFramebuffer> mFramebuffers;

        VkSemaphore mPresentSemaphore, mRenderSemaphore;
        VkFence mRenderFence;

        VkPipelineLayout mTrianglePipelineLayout;
        VkPipeline mTrianglePipeline;
        VkPipeline mRedTrianglePipeline;

        DeletionQueue mMainDeletionQueue;
    private:
        void initWindow();
        void initVulkan();
        void initSwapchain();
        void initCommands();
        void initDefaultRenderpass();
        void initFramebuffers();
        void initSyncStructure();
        void initPipelines();

        //loads a shader module from a spir-v file. Returns false if it errors
        bool loadShaderModule(const char* filePath, VkShaderModule* outShaderModule) const;

        bool mIsInit {false};
        int mFrameNumber {0};

        VkExtent2D mWindowExtent {800, 600};
        GLFWwindow* mWindow{ nullptr };
    };
}

#endif //CLOCKWORK_ENGINE_HPP
