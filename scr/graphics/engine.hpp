#ifndef CLOCKWORK_ENGINE_HPP
#define CLOCKWORK_ENGINE_HPP

#include "types.hpp"
#include <vector>

namespace cw::graphics {

    class Engine {
    public:
        void init();

        void run();
        void draw();

        void cleanup();

        VkSwapchainKHR m_Swapchain{}; // from other articles
        // image format expected by the windowing system
        VkFormat m_SwapchainImageFormat{};
        //array of images from the swapchain
        std::vector<VkImage> m_SwapchainImages{};
        //array of image-views from the swapchain
        std::vector<VkImageView> m_SwapchainImageViews{};


        VkInstance m_Instance{};
        VkDebugUtilsMessengerEXT m_DebugMessenger{};
        VkPhysicalDevice m_ChosenGpu{};
        VkDevice m_Device{}; // vulkan device for commands
        VkSurfaceKHR m_Surface{};

        VkQueue m_GraphicsQueue{};
        uint32_t m_GraphicsQueueFamily{};
        
    private:
        void initVulkan();
        void initSwapchain();

        bool m_IsInit {false};
        int m_FrameNumber {0};

        VkExtent2D m_WindowExtent {800, 600};
        GLFWwindow* m_Window{ nullptr };
    };

}

#endif //CLOCKWORK_ENGINE_HPP
