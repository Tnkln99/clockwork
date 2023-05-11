#include "engine.hpp"
#include "initializers.hpp"

#include <iostream>
#include <VkBootstrap.h>


//we want to immediately abort when there is an error. In normal engines this would give an error message to the user, or perform a dump of state.
#define VK_CHECK(x)                                                    \
	do                                                                 \
	{                                                                  \
		VkResult err = x;                                              \
		if (err)                                                       \
		{                                                              \
			std::cout <<"Detected Vulkan error: " << err << std::endl; \
			abort();                                                   \
		}                                                              \
	} while (0)

namespace cw::graphics {
    void Engine::init() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_Window = glfwCreateWindow(static_cast<int>(m_WindowExtent.width), static_cast<int>(m_WindowExtent.height), "clockwork engine", nullptr, nullptr);

        initVulkan();
        initSwapchain();
        initVulkan();

        m_IsInit = true;
    }

    void Engine::cleanup() {
        if(m_IsInit){
            vkDestroySwapchainKHR(m_Device, m_Swapchain, nullptr);

            //destroy swapchain resources
            for (auto & m_SwapchainImageView : m_SwapchainImageViews) {
                vkDestroyImageView(m_Device, m_SwapchainImageView, nullptr);
            }

            vkDestroyDevice(m_Device, nullptr);
            vkDestroySurfaceKHR(m_Instance, m_Surface, nullptr);
            vkb::destroy_debug_utils_messenger(m_Instance, m_DebugMessenger);
            vkDestroyInstance(m_Instance, nullptr);

            glfwDestroyWindow(m_Window);
            glfwTerminate();
        }
    }

    void Engine::run() {
        while(!glfwWindowShouldClose(m_Window)){
            glfwPollEvents();
            draw();
        }
    }

    void Engine::draw() {

    }

    void Engine::initVulkan() {
        vkb::InstanceBuilder builder;

        //make the Vulkan instance, with basic debug features
        auto instanceRet = builder.set_app_name("clockwork")
                .request_validation_layers(true)
                .require_api_version(1, 1, 0)
                .use_default_debug_messenger()
                .build();

        vkb::Instance vkbInst = instanceRet.value();

        //store the instance
        m_Instance = vkbInst.instance;
        //store the debug messenger
        m_DebugMessenger = vkbInst.debug_messenger;

        if (glfwCreateWindowSurface(m_Instance, m_Window, nullptr, &m_Surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan surface");
        }

        //use vkbootstrap to select a GPU.
        //We want a GPU that can write to the GLFW surface and supports Vulkan 1.1
        vkb::PhysicalDeviceSelector selector{ vkbInst };
        vkb::PhysicalDevice physicalDevice = selector
                .set_minimum_version(1, 1)
                .set_surface(m_Surface)
                .select()
                .value();

        //create the final Vulkan device
        vkb::DeviceBuilder deviceBuilder{ physicalDevice };

        vkb::Device vkbDevice = deviceBuilder.build().value();

        // Get the VkDevice handle used in the rest of a Vulkan application
        m_Device = vkbDevice.device;
        m_ChosenGpu = physicalDevice.physical_device;

        // use vkbootstrap to get a Graphics queue
        m_GraphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
        m_GraphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
    }

    void Engine::initSwapchain() {
        vkb::SwapchainBuilder swapchainBuilder{m_ChosenGpu,m_Device,m_Surface };

        vkb::Swapchain vkbSwapchain = swapchainBuilder
                .use_default_format_selection()
                //use vsync present mode
                .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                .set_desired_extent(m_WindowExtent.width, m_WindowExtent.height)
                .build()
                .value();

        //store swapchain and its related images
        m_Swapchain = vkbSwapchain.swapchain;
        m_SwapchainImages = vkbSwapchain.get_images().value();
        m_SwapchainImageViews = vkbSwapchain.get_image_views().value();

        m_SwapchainImageFormat = vkbSwapchain.image_format;
    }

    void Engine::initCommands() {
        //create a command pool for commands submitted to the graphics queue.
        //we also want the pool to allow for resetting of individual command buffers
        VkCommandPoolCreateInfo commandPoolInfo = init::commandPoolCreateInfo(m_GraphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

        VK_CHECK(vkCreateCommandPool(m_Device, &commandPoolInfo, nullptr, &m_CzommandPool));

        //allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = init::commandBufferAllocateInfo(m_CommandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &m_MainCommandBuffer));    }
}