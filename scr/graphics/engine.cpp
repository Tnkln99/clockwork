#include "engine.hpp"

#include <iostream>
#include <VkBootstrap.h>
#include <valarray>
#include <fstream>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>


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
        initWindow();
        initVulkan();
        initSwapchain();
        initCommands();
        initDefaultRenderpass();
        initFramebuffers();
        initSyncStructure();

        initPipelines();

        mIsInit = true;
    }

    void Engine::cleanup() {
        if (mIsInit) {
            //make sure the GPU has stopped doing its things
            vkWaitForFences(mDevice, 1, &mRenderFence, true, 1000000000);

            mMainDeletionQueue.flush();

            vkDestroyDevice(mDevice, nullptr);
            vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
            vkb::destroy_debug_utils_messenger(mInstance, mDebugMessenger);
            vkDestroyInstance(mInstance, nullptr);

            glfwDestroyWindow(mWindow);
            glfwTerminate();
        }
    }

    void Engine::run() {
        while (!glfwWindowShouldClose(mWindow)) {
            glfwPollEvents();
            draw();
        }
    }

    void Engine::draw() {
        //wait until the GPU has finished rendering the last frame. Timeout of 1 second
        VK_CHECK(vkWaitForFences(mDevice, 1, &mRenderFence, true, 1000000000));
        VK_CHECK(vkResetFences(mDevice, 1, &mRenderFence));

        //request image from the swapchain, one second timeout
        uint32_t swapchainImageIndex;
        VK_CHECK(vkAcquireNextImageKHR(mDevice, mSwapchain, 1000000000, mPresentSemaphore, nullptr, &swapchainImageIndex));

        //now that we are sure that the commands finished executing, we can safely reset the command buffer to begin recording again.
        VK_CHECK(vkResetCommandBuffer(mMainCommandBuffer, 0));

        //naming it cmd for shorter writing
        VkCommandBuffer cmd = mMainCommandBuffer;

        //begin the command buffer recording. We will use this command buffer exactly once, so we want to let Vulkan know that
        VkCommandBufferBeginInfo cmdBeginInfo = {};
        cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmdBeginInfo.pNext = nullptr;

        cmdBeginInfo.pInheritanceInfo = nullptr;
        cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

        //make a clear-color from frame number. This will flash with a 120*pi frame period.
        VkClearValue clearValue;
        float flash = abs(std::sin((float)mFrameNumber / 120.f));
        clearValue.color = { { 0.0f, 0.0f, flash, 1.0f } };

        //start the main renderpass.
        //We will use the clear color from above, and the framebuffer of the index the swapchain gave us
        VkRenderPassBeginInfo rpInfo = {};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.pNext = nullptr;

        rpInfo.renderPass = mRenderpass;
        rpInfo.renderArea.offset.x = 0;
        rpInfo.renderArea.offset.y = 0;
        rpInfo.renderArea.extent = mWindowExtent;
        rpInfo.framebuffer = mFramebuffers[swapchainImageIndex];

        //connect clear values
        rpInfo.clearValueCount = 1;
        rpInfo.pClearValues = &clearValue;

        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

        //once we start adding rendering commands, they will go here
        if(glfwGetKey(mWindow, GLFW_KEY_Y)){
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mRedTrianglePipeline);
        }
        else{
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mTrianglePipeline);
        }
        vkCmdDraw(cmd, 3, 1, 0, 0);

        //finalize the render pass
        vkCmdEndRenderPass(cmd);
        //finalize the command buffer (we can no longer add commands, but it can now be executed)
        VK_CHECK(vkEndCommandBuffer(cmd));

        //prepare the submission to the queue.
        //we want to wait on the mPresentSemaphore, as that semaphore is signaled when the swapchain is ready
        //we will signal the mRenderSemaphore, to signal that rendering has finished

        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.pNext = nullptr;

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        submit.pWaitDstStageMask = &waitStage;

        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &mPresentSemaphore;

        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &mRenderSemaphore;

        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;

        //submit command buffer to the queue and execute it.
        // _renderFence will now block until the graphic commands finish execution
        VK_CHECK(vkQueueSubmit(mGraphicsQueue, 1, &submit, mRenderFence));

        // this will put the image we just rendered into the visible window.
        // we want to wait on the _renderSemaphore for that,
        // as it's necessary that drawing commands have finished before the image is displayed to the user
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.pNext = nullptr;

        presentInfo.pSwapchains = &mSwapchain;
        presentInfo.swapchainCount = 1;

        presentInfo.pWaitSemaphores = &mRenderSemaphore;
        presentInfo.waitSemaphoreCount = 1;

        presentInfo.pImageIndices = &swapchainImageIndex;

        VK_CHECK(vkQueuePresentKHR(mGraphicsQueue, &presentInfo));

        //increase the number of frames drawn
        mFrameNumber++;
    }

    void Engine::initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        mWindow = glfwCreateWindow(static_cast<int>(mWindowExtent.width), static_cast<int>(mWindowExtent.height),
                                   "clockwork engine", nullptr, nullptr);
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
        mInstance = vkbInst.instance;
        //store the debug messenger
        mDebugMessenger = vkbInst.debug_messenger;

        if (glfwCreateWindowSurface(mInstance, mWindow, nullptr, &mSurface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan surface");
        }

        //use vkbootstrap to select a GPU.
        //We want a GPU that can write to the GLFW surface and supports Vulkan 1.1
        vkb::PhysicalDeviceSelector selector{vkbInst};
        vkb::PhysicalDevice physicalDevice = selector
                .set_minimum_version(1, 1)
                .set_surface(mSurface)
                .select()
                .value();

        //create the final Vulkan device
        vkb::DeviceBuilder deviceBuilder{physicalDevice};

        vkb::Device vkbDevice = deviceBuilder.build().value();

        // Get the VkDevice handle used in the rest of a Vulkan application
        mDevice = vkbDevice.device;
        mChosenGpu = physicalDevice.physical_device;

        // use vkbootstrap to get a Graphics queue
        mGraphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
        mGraphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.physicalDevice = mChosenGpu;
        allocatorInfo.device = mDevice;
        allocatorInfo.instance = mInstance;
        vmaCreateAllocator(&allocatorInfo, &mAllocator);
    }

    void Engine::initSwapchain() {
        vkb::SwapchainBuilder swapchainBuilder{mChosenGpu, mDevice, mSurface};

        vkb::Swapchain vkbSwapchain = swapchainBuilder
                .use_default_format_selection()
                //use vsync present mode
                .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                .set_desired_extent(mWindowExtent.width, mWindowExtent.height)
                .build()
                .value();

        //store swapchain and its related images
        mSwapchain = vkbSwapchain.swapchain;
        mSwapchainImages = vkbSwapchain.get_images().value();
        mSwapchainImageViews = vkbSwapchain.get_image_views().value();

        mSwapchainImageFormat = vkbSwapchain.image_format;

        mMainDeletionQueue.push_function([=]() {
            vkDestroySwapchainKHR(mDevice, mSwapchain, nullptr);
        });
    }

    void Engine::initCommands() {
        //create a command pool for commands submitted to the graphics queue.
        //we also want the pool to allow for resetting of individual command buffers
        VkCommandPoolCreateInfo commandPoolInfo = init::commandPoolCreateInfo(mGraphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

        VK_CHECK(vkCreateCommandPool(mDevice, &commandPoolInfo, nullptr, &mCommandPool));

        //allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = init::commandBufferAllocateInfo(mCommandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(mDevice, &cmdAllocInfo, &mMainCommandBuffer));

        mMainDeletionQueue.push_function([=]() {
            vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
        });
    }

    void Engine::initDefaultRenderpass() {
        // the renderpass will use this color attachment.
        VkAttachmentDescription colorAttachment = {};
        //the attachment will have the format needed by the swapchain
        colorAttachment.format = mSwapchainImageFormat;
        //1 sample, we won't be doing MSAA (Multisample Anti-Aliasing)
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        // we Clear when this attachment is loaded
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // we keep the attachment stored when the renderpass ends
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        //we don't care about stencil
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        //we don't know or care about the starting layout of the attachment
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        //after the renderpass ends, the image has to be on a layout ready for display
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef = {};
        //attachment number will index into the pAttachments array in the parent renderpass itself
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        //we are going to create 1 subpass, which is the minimum you can do
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

        //connect the color attachment to the info
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        //connect the subpass to the info
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        VK_CHECK(vkCreateRenderPass(mDevice, &renderPassInfo, nullptr, &mRenderpass));


        mMainDeletionQueue.push_function([=]() {
            vkDestroyRenderPass(mDevice, mRenderpass, nullptr);
        });
    }

    void Engine::initFramebuffers() {
        //create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
        VkFramebufferCreateInfo fbInfo = {};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.pNext = nullptr;

        fbInfo.renderPass = mRenderpass;
        fbInfo.attachmentCount = 1;
        fbInfo.width = mWindowExtent.width;
        fbInfo.height = mWindowExtent.height;
        fbInfo.layers = 1;

        //grab how many images we have in the swapchain
        const uint32_t swapChainImageCount = mSwapchainImages.size();
        mFramebuffers = std::vector<VkFramebuffer>(swapChainImageCount);

        //create framebuffers for each of the swapchain image views
        for (int i = 0; i < swapChainImageCount; i++) {
            fbInfo.pAttachments = &mSwapchainImageViews[i];
            VK_CHECK(vkCreateFramebuffer(mDevice, &fbInfo, nullptr, &mFramebuffers[i]));

            mMainDeletionQueue.push_function([=]() {
                vkDestroyFramebuffer(mDevice, mFramebuffers[i], nullptr);
                vkDestroyImageView(mDevice, mSwapchainImageViews[i], nullptr);
            });
        }
    }

    void Engine::initSyncStructure() {
        VkFenceCreateInfo fenceCreateInfo = init::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);

        VK_CHECK(vkCreateFence(mDevice, &fenceCreateInfo, nullptr, &mRenderFence));

        //enqueue the destruction of the fence
        mMainDeletionQueue.push_function([=]() {
            vkDestroyFence(mDevice, mRenderFence, nullptr);
        });

        VkSemaphoreCreateInfo semaphoreCreateInfo = init::semaphoreCreateInfo();

        VK_CHECK(vkCreateSemaphore(mDevice, &semaphoreCreateInfo, nullptr, &mPresentSemaphore));
        VK_CHECK(vkCreateSemaphore(mDevice, &semaphoreCreateInfo, nullptr, &mRenderSemaphore));

        //enqueue the destruction of semaphores
        mMainDeletionQueue.push_function([=]() {
            vkDestroySemaphore(mDevice, mPresentSemaphore, nullptr);
            vkDestroySemaphore(mDevice, mRenderSemaphore, nullptr);
        });
    }

    void Engine::initPipelines() {
        VkShaderModule triangleFragShader;
        if (!loadShaderModule("res/shaders/triangle.frag.spv", &triangleFragShader))
        {
            std::cout << "Error when building the triangle fragment shader module" << std::endl;
        }
        else {
            std::cout << "Triangle fragment shader successfully loaded" << std::endl;
        }

        VkShaderModule triangleVertexShader;
        if (!loadShaderModule("res/shaders/triangle.vert.spv", &triangleVertexShader))
        {
            std::cout << "Error when building the triangle vertex shader module" << std::endl;
        }
        else {
            std::cout << "Triangle vertex shader successfully loaded" << std::endl;
        }

        //compile red triangle modules
        VkShaderModule redTriangleFragShader;
        if (!loadShaderModule("res/shaders/triangle_red.frag.spv", &redTriangleFragShader))
        {
            std::cout << "Error when building the triangle fragment shader module" << std::endl;
        }
        else {
            std::cout << "Red Triangle fragment shader successfully loaded" << std::endl;
        }

        VkShaderModule redTriangleVertShader;
        if (!loadShaderModule("res/shaders/triangle_red.vert.spv", &redTriangleVertShader))
        {
            std::cout << "Error when building the triangle vertex shader module" << std::endl;
        }
        else {
            std::cout << "Red Triangle vertex shader successfully loaded" << std::endl;
        }

        //build the pipeline layout that controls the inputs/outputs of the shader
        //we are not using descriptor sets or other systems yet, so no need to use anything other than empty default
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = init::pipelineLayoutCreateInfo();

        VK_CHECK(vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &mTrianglePipelineLayout));

        //build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
        Pipeline pipelineBuilder;

        pipelineBuilder.mShaderStages.push_back(
                init::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, triangleVertexShader));

        pipelineBuilder.mShaderStages.push_back(
                init::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));


        //vertex input controls how to read vertices from vertex buffers. We aren't using it yet
        pipelineBuilder.mVertexInputInfo = init::vertexInputStateCreateInfo();

        //input assembly is the configuration for drawing triangle lists, strips, or individual points.
        //we are just going to draw triangle list
        pipelineBuilder.mInputAssembly = init::inputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        //build viewport and scissor from the swapchain extents
        pipelineBuilder.mViewport.x = 0.0f;
        pipelineBuilder.mViewport.y = 0.0f;
        pipelineBuilder.mViewport.width = (float)mWindowExtent.width;
        pipelineBuilder.mViewport.height = (float)mWindowExtent.height;
        pipelineBuilder.mViewport.minDepth = 0.0f;
        pipelineBuilder.mViewport.maxDepth = 1.0f;

        pipelineBuilder.mScissor.offset = { 0, 0 };
        pipelineBuilder.mScissor.extent = mWindowExtent;

        //configure the rasterizer to draw filled triangles
        pipelineBuilder.mRasterizer = init::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);

        //we don't use multisampling, so just run the default one
        pipelineBuilder.mMultisampling = init::multisampleStateCreateInfo();

        //a single blend attachment with no blending and writing to RGBA
        pipelineBuilder.mColorBlendAttachment = init::colorBlendAttachmentState();

        //use the triangle layout we created
        pipelineBuilder.mPipelineLayout = mTrianglePipelineLayout;

        //finally build the pipeline
        mTrianglePipeline = pipelineBuilder.buildPipeline(mDevice, mRenderpass);

        //clear the shader stages for the builder
        pipelineBuilder.mShaderStages.clear();

        //add the other shaders
        pipelineBuilder.mShaderStages.push_back(
                init::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, redTriangleVertShader));

        pipelineBuilder.mShaderStages.push_back(
                init::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, redTriangleFragShader));

        //build the red triangle pipeline
        mRedTrianglePipeline = pipelineBuilder.buildPipeline(mDevice, mRenderpass);

        //destroy all shader modules, outside the queue
        vkDestroyShaderModule(mDevice, redTriangleVertShader, nullptr);
        vkDestroyShaderModule(mDevice, redTriangleFragShader, nullptr);
        vkDestroyShaderModule(mDevice, triangleFragShader, nullptr);
        vkDestroyShaderModule(mDevice, triangleVertexShader, nullptr);

        mMainDeletionQueue.push_function([=]() {
            //destroy the 2 pipelines we have created
            vkDestroyPipeline(mDevice, mRedTrianglePipeline, nullptr);
            vkDestroyPipeline(mDevice, mTrianglePipeline, nullptr);

            //destroy the pipeline layout that they use
            vkDestroyPipelineLayout(mDevice, mTrianglePipelineLayout, nullptr);
        });
    }

    bool Engine::loadShaderModule(const char *filePath, VkShaderModule *outShaderModule) const {
        //open the file. With cursor at the end
        std::ifstream file(filePath, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            return false;
        }

        //find what the size of the file is by looking up the location of the cursor
        //because the cursor is at the end, it gives the size directly in bytes
        size_t fileSize = (size_t)file.tellg();

        //spirv expects the buffer to be on uint32, so make sure to reserve an int vector big enough for the entire file
        std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

        //put file cursor at beginning
        file.seekg(0);

        //load the entire file into the buffer
        file.read((char*)buffer.data(), fileSize);

        //now that the file is loaded into the buffer, we can close it
        file.close();

        //create a new shader module, using the buffer we loaded
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pNext = nullptr;

        //codeSize has to be in bytes, so multiply the ints in the buffer by size of int to know the real size of the buffer
        createInfo.codeSize = buffer.size() * sizeof(uint32_t);
        createInfo.pCode = buffer.data();

        //check that the creation goes well.
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(mDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            return false;
        }
        *outShaderModule = shaderModule;
        return true;
    }
}