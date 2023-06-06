#include "engine.hpp"
#include "textures.hpp"

#include <iostream>
#include <VkBootstrap.h>
#include <valarray>
#include <fstream>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <glm/gtx/transform.hpp>

//we want to immediately abort when there is an error. In normal engines this would give an error message to the user, or perform a dump of state.
#define VK_CHECK(x)                                                    \
	do                                                                 \
	{                                                                  \
		VkResult err = x;                                              \
		if (err)                                                       \
		{                                                              \
			std::cout <<"Detected Vulkan error: " << err << "at " << std::to_string(__LINE__) << std::endl; \
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
        initDescriptors();
        initPipelines();

        loadImages();
        loadMeshes();

        initScenes();

        mIsInit = true;
    }

    void Engine::cleanup() {
        if (mIsInit) {
            //make sure the GPU has stopped doing its things
            vkDeviceWaitIdle(mDevice);

            mMainDeletionQueue.flush();

            vkDestroySurfaceKHR(mInstance, mSurface, nullptr);

            vkDestroyDevice(mDevice, nullptr);
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
        VK_CHECK(vkWaitForFences(mDevice, 1, &getCurrentFrame().renderFence, true, 1000000000));
        VK_CHECK(vkResetFences(mDevice, 1, &getCurrentFrame().renderFence));

        //request image from the swapchain, one second timeout
        uint32_t swapchainImageIndex;
        VK_CHECK(vkAcquireNextImageKHR(mDevice, mSwapchain, 1000000000, getCurrentFrame().presentSemaphore, nullptr, &swapchainImageIndex));

        //now that we are sure that the commands finished executing, we can safely reset the command buffer to begin recording again.
        VK_CHECK(vkResetCommandBuffer(getCurrentFrame().mainCommandBuffer, 0));

        //naming it cmd for shorter writing
        VkCommandBuffer cmd = getCurrentFrame().mainCommandBuffer;

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

        //clear depth at 1
        VkClearValue depthClear;
        depthClear.depthStencil.depth = 1.f;

        //start the main renderpass.
        //We will use the clear color from above, and the framebuffer of the index the swapchain gave us
        VkRenderPassBeginInfo rpInfo = init::renderPassBeginInfo(mRenderpass, mWindowExtent, mFramebuffers[swapchainImageIndex]);

        //connect clear values
        rpInfo.clearValueCount = 2;

        VkClearValue clearValues[] = { clearValue, depthClear };

        rpInfo.pClearValues = &clearValues[0];

        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

        drawObjects(cmd);

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
        submit.pWaitSemaphores = &getCurrentFrame().presentSemaphore;

        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &getCurrentFrame().renderSemaphore;

        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;

        //submit command buffer to the queue and execute it.
        // _renderFence will now block until the graphic commands finish execution
        VK_CHECK(vkQueueSubmit(mGraphicsQueue, 1, &submit, getCurrentFrame().renderFence));

        // this will put the image we just rendered into the visible window.
        // we want to wait on the _renderSemaphore for that,
        // as it's necessary that drawing commands have finished before the image is displayed to the user
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.pNext = nullptr;

        presentInfo.pSwapchains = &mSwapchain;
        presentInfo.swapchainCount = 1;

        presentInfo.pWaitSemaphores = &getCurrentFrame().renderSemaphore;
        presentInfo.waitSemaphoreCount = 1;

        presentInfo.pImageIndices = &swapchainImageIndex;

        VK_CHECK(vkQueuePresentKHR(mGraphicsQueue, &presentInfo));

        //increase the number of frames drawn
        mFrameNumber++;
    }

    void Engine::immediateSubmit(std::function<void(VkCommandBuffer)> &&function) {
        VkCommandBuffer cmd = mUploadContext.commandBuffer;

        //begin the command buffer recording. We will use this command buffer exactly once before resetting, so we tell vulkan that
        VkCommandBufferBeginInfo cmdBeginInfo = init::commandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

        //execute the function
        function(cmd);

        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo submit = init::submitInfo(&cmd);

        //submit command buffer to the queue and execute it.
        // _uploadFence will now block until the graphic commands finish execution
        VK_CHECK(vkQueueSubmit(mGraphicsQueue, 1, &submit, mUploadContext.uploadFence));

        vkWaitForFences(mDevice, 1, &mUploadContext.uploadFence, true, 9999999999);
        vkResetFences(mDevice, 1, &mUploadContext.uploadFence);

        // reset the command buffers inside the command pool
        vkResetCommandPool(mDevice, mUploadContext.commandPool, 0);
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

        VkPhysicalDeviceShaderDrawParametersFeatures shadowDrawParametersFeature = {};
        shadowDrawParametersFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
        shadowDrawParametersFeature.pNext = nullptr;
        shadowDrawParametersFeature.shaderDrawParameters = VK_TRUE;


        vkb::Device vkbDevice = deviceBuilder.
                add_pNext(&shadowDrawParametersFeature).build().value();

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

        mMainDeletionQueue.push_function([&]() {
            vmaDestroyAllocator(mAllocator);
        });

        mGpuProperties = vkbDevice.physical_device.properties;
        std::cout << "The GPU has a minimum buffer alignment of " << mGpuProperties.limits.minUniformBufferOffsetAlignment << std::endl;
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

        //depth image size will match the window
        VkExtent3D depthImageExtent = {
                mWindowExtent.width,
                mWindowExtent.height,
                1
        };

        //hardcoding the depth format to 32-bit float
        mDepthFormat = VK_FORMAT_D32_SFLOAT;

        //the depth image will be an image with the format we selected and Depth Attachment usage flag
        VkImageCreateInfo imgInfo = init::imageCreateInfo(mDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

        //for the depth image, we want to allocate it from GPU local memory
        VmaAllocationCreateInfo imgAllocInfo = {};
        imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        imgAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        //allocate and create the image
        vmaCreateImage(mAllocator, &imgInfo, &imgAllocInfo, &mDepthImage.image, &mDepthImage.allocation, nullptr);

        //build an image-view for the depth image to use for rendering
        VkImageViewCreateInfo dview_info = init::imageViewCreateInfo(mDepthFormat, mDepthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

        VK_CHECK(vkCreateImageView(mDevice, &dview_info, nullptr, &mDepthImageView));

        //add to deletion queues
        mMainDeletionQueue.push_function([=]() {
            vkDestroyImageView(mDevice, mDepthImageView, nullptr);
            vmaDestroyImage(mAllocator, mDepthImage.image, mDepthImage.allocation);
        });
    }

    void Engine::initCommands() {
        //create a command pool for commands submitted to the graphics queue.
        //we also want the pool to allow for resetting of individual command buffers
        VkCommandPoolCreateInfo commandPoolInfo = init::commandPoolCreateInfo(mGraphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

        for (int i = 0; i < cFrameOverlap; i++) {
            VK_CHECK(vkCreateCommandPool(mDevice, &commandPoolInfo, nullptr, &mFrames[i].commandPool));

            //allocate the default command buffer that we will use for rendering
            VkCommandBufferAllocateInfo cmdAllocInfo = init::commandBufferAllocateInfo(mFrames[i].commandPool, 1);

            VK_CHECK(vkAllocateCommandBuffers(mDevice, &cmdAllocInfo, &mFrames[i].mainCommandBuffer));

            mMainDeletionQueue.push_function([=]() {
                vkDestroyCommandPool(mDevice, mFrames[i].commandPool, nullptr);
            });
        }

        VkCommandPoolCreateInfo uploadCommandPoolInfo = init::commandPoolCreateInfo(mGraphicsQueueFamily);
        //create pool for upload context
        VK_CHECK(vkCreateCommandPool(mDevice, &uploadCommandPoolInfo, nullptr, &mUploadContext.commandPool));

        mMainDeletionQueue.push_function([=]() {
            vkDestroyCommandPool(mDevice, mUploadContext.commandPool, nullptr);
        });

        //allocate the default command buffer that we will use for the instant commands
        VkCommandBufferAllocateInfo cmdAllocInfo = init::commandBufferAllocateInfo(mUploadContext.commandPool, 1);

        VkCommandBuffer cmd;
        VK_CHECK(vkAllocateCommandBuffers(mDevice, &cmdAllocInfo, &mUploadContext.commandBuffer));
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

        VkAttachmentDescription depthAttachment = {};
        // Depth attachment
        depthAttachment.flags = 0;
        depthAttachment.format = mDepthFormat;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef = {};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        //we are going to create 1 subpass, which is the minimum you can do
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        //hook the depth attachment into the subpass
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        //1 dependency, which is from "outside" into the subpass. And we can read or write color
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        //dependency from outside to the subpass, making this subpass dependent on the previous renderpasses
        VkSubpassDependency depthDependency = {};
        depthDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        depthDependency.dstSubpass = 0;
        depthDependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        depthDependency.srcAccessMask = 0;
        depthDependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        depthDependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        //array of 2 dependencies, one for color, two for depth
        VkSubpassDependency dependencies[2] = { dependency, depthDependency };

        //array of 2 attachments, one for the color, and other for depth
        VkAttachmentDescription attachments[2] = { colorAttachment,depthAttachment };


        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

        //connect the color attachment to the info
        renderPassInfo.attachmentCount = 2;
        renderPassInfo.pAttachments = &attachments[0];
        //connect the subpass to the info
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        renderPassInfo.dependencyCount = 2;
        renderPassInfo.pDependencies = &dependencies[0];

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
            VkImageView attachments[2];
            attachments[0] = mSwapchainImageViews[i];
            attachments[1] = mDepthImageView;

            fbInfo.pAttachments = attachments;
            fbInfo.attachmentCount = 2;
            VK_CHECK(vkCreateFramebuffer(mDevice, &fbInfo, nullptr, &mFramebuffers[i]));
            mMainDeletionQueue.push_function([=]() {
                vkDestroyFramebuffer(mDevice, mFramebuffers[i], nullptr);
                vkDestroyImageView(mDevice, mSwapchainImageViews[i], nullptr);
            });
        }
    }

    void Engine::initSyncStructure() {
        VkFenceCreateInfo fenceCreateInfo = init::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);

        VkSemaphoreCreateInfo semaphoreCreateInfo = init::semaphoreCreateInfo();

        for (int i = 0; i < cFrameOverlap; i++) {
            VK_CHECK(vkCreateFence(mDevice, &fenceCreateInfo, nullptr, &mFrames[i].renderFence));

            //enqueue the destruction of the fence
            mMainDeletionQueue.push_function([=]() {
                vkDestroyFence(mDevice, mFrames[i].renderFence, nullptr);
            });


            VK_CHECK(vkCreateSemaphore(mDevice, &semaphoreCreateInfo, nullptr, &mFrames[i].presentSemaphore));
            VK_CHECK(vkCreateSemaphore(mDevice, &semaphoreCreateInfo, nullptr, &mFrames[i].renderSemaphore));

            //enqueue the destruction of semaphores
            mMainDeletionQueue.push_function([=]() {
                vkDestroySemaphore(mDevice, mFrames[i].presentSemaphore, nullptr);
                vkDestroySemaphore(mDevice, mFrames[i].renderSemaphore, nullptr);
            });
        }

        VkFenceCreateInfo uploadFenceCreateInfo = init::fenceCreateInfo();

        VK_CHECK(vkCreateFence(mDevice, &uploadFenceCreateInfo, nullptr, &mUploadContext.uploadFence));
        mMainDeletionQueue.push_function([=]() {
            vkDestroyFence(mDevice, mUploadContext.uploadFence, nullptr);
        });
    }

    void Engine::initPipelines() {
        //compile mesh vertex shader
        VkShaderModule meshVertShader;
        if (!loadShaderModule("../res/shaders/tri_mesh.vert.spv", &meshVertShader)) {
            std::cout << "Error when building the triangle vertex shader module" << std::endl;
        }
        else {
            std::cout << "Mesh Triangle vertex shader successfully loaded" << std::endl;
        }
        VkShaderModule triangleFragShader;
        if (!loadShaderModule("../res/shaders/default_lit.frag.spv", &triangleFragShader))
        {
            std::cout << "Error when building the triangle fragment shader module" << std::endl;
        }
        else
        {
            std::cout << "Triangle fragment shader successfully loaded" << std::endl;
        }
        VkShaderModule texturedMeshShader;
        if (!loadShaderModule("../res/shaders/textured_lit.frag.spv", &texturedMeshShader))
        {
            std::cout << "Error when building the textured mesh shader" << std::endl;
        }
        else
        {
            std::cout << "Texture mesh shader successfully loaded" << std::endl;
        }

        //build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
        Pipeline pipelineBuilder;

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

        pipelineBuilder.mDepthStencil = init::depthStencilStateCreateInfo(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

        //build the mesh pipeline
        VertexInputDescription vertexDescription = Vertex::getVertexDescription();

        //connect the pipeline builder vertex input info to the one we get from Vertex
        pipelineBuilder.mVertexInputInfo.pVertexAttributeDescriptions = vertexDescription.mAttributes.data();
        pipelineBuilder.mVertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.mAttributes.size();

        pipelineBuilder.mVertexInputInfo.pVertexBindingDescriptions = vertexDescription.mBindings.data();
        pipelineBuilder.mVertexInputInfo.vertexBindingDescriptionCount = vertexDescription.mBindings.size();

        //add the other shaders
        pipelineBuilder.mShaderStages.push_back(
                init::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

        //make sure that triangleFragShader is holding the compiled colored_triangle.frag
        pipelineBuilder.mShaderStages.push_back(
                init::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));

        //we start from just the default empty pipeline layout info
        VkPipelineLayoutCreateInfo meshPipelineLayoutInfo = init::pipelineLayoutCreateInfo();

        //setup push constants
        VkPushConstantRange pushConstant;
        //this push constant range starts at the beginning
        pushConstant.offset = 0;
        //this push constant range takes up the size of a MeshPushConstants struct
        pushConstant.size = sizeof(MeshPushConstants);
        //this push constant range is accessible only in the vertex shader
        pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        meshPipelineLayoutInfo.pPushConstantRanges = &pushConstant;
        meshPipelineLayoutInfo.pushConstantRangeCount = 1;

        VkDescriptorSetLayout setLayouts[] = { mGlobalSetLayout, mObjectSetLayout };

        //hook the global set layout
        meshPipelineLayoutInfo.setLayoutCount = 2;
        meshPipelineLayoutInfo.pSetLayouts = setLayouts;

        VkPipelineLayout meshPipelineLayout;
        VkPipeline meshPipeline;

        VK_CHECK(vkCreatePipelineLayout(mDevice, &meshPipelineLayoutInfo, nullptr, &meshPipelineLayout));

        pipelineBuilder.mPipelineLayout = meshPipelineLayout;
        meshPipeline = pipelineBuilder.buildPipeline(mDevice, mRenderpass);
        createMaterial(meshPipeline, meshPipelineLayout, "defaultmesh");

        //create pipeline layout for the textured mesh, which has 3 descriptor sets
        //we start from  the normal mesh layout
        VkPipelineLayoutCreateInfo textured_pipeline_layout_info = meshPipelineLayoutInfo;

        VkDescriptorSetLayout texturedSetLayouts[] = { mGlobalSetLayout, mObjectSetLayout, mSingleTextureSetLayout };

        textured_pipeline_layout_info.setLayoutCount = 3;
        textured_pipeline_layout_info.pSetLayouts = texturedSetLayouts;

        VkPipelineLayout texturedPipeLayout;
        VK_CHECK(vkCreatePipelineLayout(mDevice, &textured_pipeline_layout_info, nullptr, &texturedPipeLayout));

        pipelineBuilder.mShaderStages.clear();
        pipelineBuilder.mShaderStages.push_back(
                init::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

        pipelineBuilder.mShaderStages.push_back(
                init::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, texturedMeshShader));

        //connect the new pipeline layout to the pipeline builder
        pipelineBuilder.mPipelineLayout = texturedPipeLayout;
        VkPipeline texPipeline = pipelineBuilder.buildPipeline(mDevice, mRenderpass);
        createMaterial(texPipeline, texturedPipeLayout, "texturedmesh");


        //destroy all shader modules, outside the queue
        vkDestroyShaderModule(mDevice, meshVertShader, nullptr);
        vkDestroyShaderModule(mDevice, triangleFragShader, nullptr);
        vkDestroyShaderModule(mDevice, texturedMeshShader, nullptr);

        mMainDeletionQueue.push_function([=]() {
            //destroy pipelines we have created
            vkDestroyPipeline(mDevice, meshPipeline, nullptr);
            vkDestroyPipeline(mDevice, texPipeline, nullptr);

            //destroy the pipeline layout that they use
            vkDestroyPipelineLayout(mDevice, meshPipelineLayout, nullptr);
            vkDestroyPipelineLayout(mDevice, texturedPipeLayout, nullptr);
        });
    }


    void Engine::initDescriptors() {
        //create a descriptor pool that will hold 10 uniform buffers
        std::vector<VkDescriptorPoolSize> sizes =
                {
                        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10 },
                        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10 },
                        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 },
                        //add combined-image-sampler descriptor types to the pool
                        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 }
                };

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = 0;
        poolInfo.maxSets = 10;
        poolInfo.poolSizeCount = (uint32_t)sizes.size();
        poolInfo.pPoolSizes = sizes.data();

        vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mDescriptorPool);

        //binding for camera data at 0
        VkDescriptorSetLayoutBinding cameraBind = init::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,VK_SHADER_STAGE_VERTEX_BIT,0);

        //binding for scene data at 1
        VkDescriptorSetLayoutBinding sceneBind = init::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1);

        VkDescriptorSetLayoutBinding bindings[] = { cameraBind,sceneBind };


        VkDescriptorSetLayoutCreateInfo setInfo = {};
        setInfo.bindingCount = 2;
        setInfo.flags = 0;
        setInfo.pNext = nullptr;
        setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setInfo.pBindings = bindings;

        vkCreateDescriptorSetLayout(mDevice, &setInfo, nullptr, &mGlobalSetLayout);

        VkDescriptorSetLayoutBinding objectBind = init::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);

        VkDescriptorSetLayoutCreateInfo set2info = {};
        set2info.bindingCount = 1;
        set2info.flags = 0;
        set2info.pNext = nullptr;
        set2info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set2info.pBindings = &objectBind;

        vkCreateDescriptorSetLayout(mDevice, &set2info, nullptr, &mObjectSetLayout);

        //another set, one that holds a single texture
        VkDescriptorSetLayoutBinding textureBind = init::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);

        VkDescriptorSetLayoutCreateInfo set3info = {};
        set3info.bindingCount = 1;
        set3info.flags = 0;
        set3info.pNext = nullptr;
        set3info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        set3info.pBindings = &textureBind;

        vkCreateDescriptorSetLayout(mDevice, &set3info, nullptr, &mSingleTextureSetLayout);

        const size_t sceneParamBufferSize = cFrameOverlap * padUniformBufferSize(sizeof(GPUSceneData));
        mSceneParameterBuffer = createBuffer(sceneParamBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

        for (int i = 0; i < cFrameOverlap; i++)
        {
            const int maxObjects = 10000;
            mFrames[i].objectBuffer = createBuffer(sizeof(GPUObjectData) * maxObjects, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            mFrames[i].cameraBuffer = createBuffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

            //allocate one descriptor set for each frame
            VkDescriptorSetAllocateInfo globalSetAllocInfo ={};
            globalSetAllocInfo.pNext = nullptr;
            globalSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            //using the pool we just set
            globalSetAllocInfo.descriptorPool = mDescriptorPool;
            //only 1 descriptor
            globalSetAllocInfo.descriptorSetCount = 1;
            //using the global data layout
            globalSetAllocInfo.pSetLayouts = &mGlobalSetLayout;

            vkAllocateDescriptorSets(mDevice, &globalSetAllocInfo, &mFrames[i].globalDescriptor);

            //allocate the descriptor set that will point to object buffer
            VkDescriptorSetAllocateInfo objectSetAlloc = {};
            objectSetAlloc.pNext = nullptr;
            objectSetAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            objectSetAlloc.descriptorPool = mDescriptorPool;
            objectSetAlloc.descriptorSetCount = 1;
            objectSetAlloc.pSetLayouts = &mObjectSetLayout;

            vkAllocateDescriptorSets(mDevice, &objectSetAlloc, &mFrames[i].objectDescriptor);

            VkDescriptorBufferInfo cameraInfo;
            cameraInfo.buffer = mFrames[i].cameraBuffer.buffer;
            cameraInfo.offset = 0;
            cameraInfo.range = sizeof(GPUCameraData);

            VkDescriptorBufferInfo sceneInfo;
            sceneInfo.buffer = mSceneParameterBuffer.buffer;
            sceneInfo.offset = 0;
            sceneInfo.range = sizeof(GPUSceneData);

            VkDescriptorBufferInfo objectInfo;
            objectInfo.buffer = mFrames[i].objectBuffer.buffer;
            objectInfo.offset = 0;
            objectInfo.range = sizeof(GPUObjectData) * maxObjects;

            VkWriteDescriptorSet cameraWrite = init::writeDescriptorBuffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, mFrames[i].globalDescriptor,&cameraInfo,0);

            VkWriteDescriptorSet sceneWrite = init::writeDescriptorBuffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, mFrames[i].globalDescriptor, &sceneInfo, 1);

            VkWriteDescriptorSet objectWrite = init::writeDescriptorBuffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mFrames[i].objectDescriptor, &objectInfo, 0);

            VkWriteDescriptorSet setWrites[] = { cameraWrite,sceneWrite, objectWrite };

            vkUpdateDescriptorSets(mDevice, 3, setWrites, 0, nullptr);
        }

        // add buffers to deletion queues
        mMainDeletionQueue.push_function([&]() {
            vmaDestroyBuffer(mAllocator, mSceneParameterBuffer.buffer, mSceneParameterBuffer.allocation);

            vkDestroyDescriptorSetLayout(mDevice, mObjectSetLayout, nullptr);
            vkDestroyDescriptorSetLayout(mDevice, mGlobalSetLayout, nullptr);
            vkDestroyDescriptorSetLayout(mDevice, mSingleTextureSetLayout, nullptr);

            vkDestroyDescriptorPool(mDevice, mDescriptorPool, nullptr);

            for (int i = 0; i < cFrameOverlap; i++)
            {
                vmaDestroyBuffer(mAllocator, mFrames[i].cameraBuffer.buffer, mFrames[i].cameraBuffer.allocation);
                vmaDestroyBuffer(mAllocator, mFrames[i].objectBuffer.buffer, mFrames[i].objectBuffer.allocation);
            }

        });
    }

    void Engine::loadMeshes() {
        Mesh triangleMesh{};
        Mesh monkeyMesh{};
        Mesh lostEmpire{};

        //make the array 3 vertices long
        triangleMesh.mVertices.resize(3);

        //vertex positions
        triangleMesh.mVertices[0].position = { 1.f, 1.f, 0.0f };
        triangleMesh.mVertices[1].position = {-1.f, 1.f, 0.0f };
        triangleMesh.mVertices[2].position = { 0.f,-1.f, 0.0f };

        //vertex colors all green
        triangleMesh.mVertices[1].color = { 1.f, 1.f, 0.0f }; //pure green
        triangleMesh.mVertices[2].color = { 1.f, 1.f, 0.0f }; //pure green
        triangleMesh.mVertices[0].color = { 1.f, 1.f, 0.0f }; //pure green

        monkeyMesh.loadFromObj("../res/meshes/monkey_smooth.obj");
        lostEmpire.loadFromObj("../res/meshes/lost_empire.obj");

        uploadMesh(triangleMesh);
        uploadMesh(monkeyMesh);
        uploadMesh(lostEmpire);

        mMeshes["monkey"] = monkeyMesh;
        mMeshes["triangle"] = triangleMesh;
        mMeshes["empire"] = lostEmpire;
    }

    void Engine::uploadMesh(Mesh &mesh) {
        const size_t bufferSize= mesh.mVertices.size() * sizeof(Vertex);
        //allocate staging buffer
        VkBufferCreateInfo stagingBufferInfo = {};
        stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingBufferInfo.pNext = nullptr;

        stagingBufferInfo.size = bufferSize;
        stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        //let the VMA library know that this data should be on CPU RAM
        VmaAllocationCreateInfo vmaallocInfo = {};
        vmaallocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

        AllocatedBuffer stagingBuffer;

        //allocate the buffer
        VK_CHECK(vmaCreateBuffer(mAllocator, &stagingBufferInfo, &vmaallocInfo,
                                 &stagingBuffer.buffer,
                                 &stagingBuffer.allocation,
                                 nullptr));

        //copy vertex data
        void* data;
        vmaMapMemory(mAllocator, stagingBuffer.allocation, &data);
        memcpy(data, mesh.mVertices.data(), mesh.mVertices.size() * sizeof(Vertex));
        vmaUnmapMemory(mAllocator, stagingBuffer.allocation);

        //allocate vertex buffer
        VkBufferCreateInfo vertexBufferInfo = {};
        vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vertexBufferInfo.pNext = nullptr;
        //this is the total size, in bytes, of the buffer we are allocating
        vertexBufferInfo.size = bufferSize;
        //this buffer is going to be used as a Vertex Buffer
        vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        //let the VMA library know that this data should be GPU native
        vmaallocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        //allocate the buffer
        VK_CHECK(vmaCreateBuffer(mAllocator, &vertexBufferInfo, &vmaallocInfo,
                                 &mesh.mVertexBuffer.buffer,
                                 &mesh.mVertexBuffer.allocation,
                                 nullptr));

        immediateSubmit([=](VkCommandBuffer cmd) {
            VkBufferCopy copy;
            copy.dstOffset = 0;
            copy.srcOffset = 0;
            copy.size = bufferSize;
            vkCmdCopyBuffer(cmd, stagingBuffer.buffer, mesh.mVertexBuffer.buffer, 1, &copy);
        });

        //add the destruction of mesh buffer to the deletion queue
        mMainDeletionQueue.push_function([=]() {
            vmaDestroyBuffer(mAllocator, mesh.mVertexBuffer.buffer, mesh.mVertexBuffer.allocation);
        });

        vmaDestroyBuffer(mAllocator, stagingBuffer.buffer, stagingBuffer.allocation);
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

    Material* Engine::createMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string &name) {
        Material mat{};
        mat.pipeline = pipeline;
        mat.pipelineLayout = layout;
        mMaterials[name] = mat;
        return &mMaterials[name];
    }

    Material* Engine::getMaterial(const std::string &name) {
        //search for the object, and return nullptr if not found
        auto it = mMaterials.find(name);
        if (it == mMaterials.end()) {
            return nullptr;
        }
        else {
            return &(*it).second;
        }
    }

    Mesh* Engine::getMesh(const std::string &name) {
        auto it = mMeshes.find(name);
        if (it == mMeshes.end()) {
            return nullptr;
        }
        else {
            return &(*it).second;
        }
    }

    void Engine::drawObjects(VkCommandBuffer cmd) {
        //make a model view matrix for rendering the object
        //camera view
        glm::vec3 camPos = { 0.f,-6.f,-10.f };

        glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
        //camera projection
        glm::mat4 projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.0f);
        projection[1][1] *= -1;

        //fill a GPU camera data struct
        GPUCameraData camData{};
        camData.proj = projection;
        camData.view = view;
        camData.viewProj = projection * view;

        //and copy it to the buffer
        void* data;
        vmaMapMemory(mAllocator, getCurrentFrame().cameraBuffer.allocation, &data);
        memcpy(data, &camData, sizeof(GPUCameraData));
        vmaUnmapMemory(mAllocator, getCurrentFrame().cameraBuffer.allocation);

        float framed = (mFrameNumber / 120.f);
        mSceneParameters.ambientColor = { sin(framed),0,cos(framed),1 };
        char* sceneData;
        vmaMapMemory(mAllocator, mSceneParameterBuffer.allocation , (void**)&sceneData);
        int frameIndex = mFrameNumber % cFrameOverlap;
        sceneData += padUniformBufferSize(sizeof(GPUSceneData)) * frameIndex;
        memcpy(sceneData, &mSceneParameters, sizeof(GPUSceneData));
        vmaUnmapMemory(mAllocator, mSceneParameterBuffer.allocation);

        void* objectData;
        vmaMapMemory(mAllocator, getCurrentFrame().objectBuffer.allocation, &objectData);
        auto* objectSSBO = (GPUObjectData*)objectData;
        for (int i = 0; i < mRenderables.size(); i++)
        {
            RenderObject& object = mRenderables[i];
            objectSSBO[i].modelMatrix = object.transformMatrix;
        }
        vmaUnmapMemory(mAllocator, getCurrentFrame().objectBuffer.allocation);

        Mesh* lastMesh = nullptr;
        Material* lastMaterial = nullptr;
        int count = 0;
        for (auto & object : mRenderables)
        {
            //only bind the pipeline if it doesn't match with the already bound one
            if (object.material != lastMaterial) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
                lastMaterial = object.material;
                //offset for our scene buffer
                uint32_t uniformOffset = padUniformBufferSize(sizeof(GPUSceneData)) * frameIndex;
                //bind the descriptor set when changing pipeline
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 0, 1, &getCurrentFrame().globalDescriptor, 1, &uniformOffset);
                //object data descriptor
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 1, 1, &getCurrentFrame().objectDescriptor, 0, nullptr);
                if(object.material->textureSet != VK_NULL_HANDLE){
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 2, 1, &object.material->textureSet, 0, nullptr);
                }
            }

            MeshPushConstants constants{};
            constants.renderMatrix = object.transformMatrix;

            //upload the mesh to the GPU via push constants
            vkCmdPushConstants(cmd, object.material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

            //only bind the mesh if it's a different one from last bind
            if (object.mesh != lastMesh) {
                //bind the mesh vertex buffer with offset 0
                VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->mVertexBuffer.buffer, &offset);
                lastMesh = object.mesh;
            }
            //we can now draw
            vkCmdDraw(cmd, object.mesh->mVertices.size(), 1, 0, count);
            count++;
        }
    }

    void Engine::initScenes() {
        RenderObject monkey{};
        monkey.mesh = getMesh("monkey");
        monkey.material = getMaterial("defaultmesh");
        monkey.transformMatrix = glm::mat4{ 1.0f };

        mRenderables.push_back(monkey);

        for (int x = -20; x <= 20; x++) {
            for (int y = -20; y <= 20; y++) {

                RenderObject tri{};
                tri.mesh = getMesh("triangle");
                tri.material = getMaterial("defaultmesh");
                glm::mat4 translation = glm::translate(glm::mat4{ 1.0 }, glm::vec3(x, 0, y));
                glm::mat4 scale = glm::scale(glm::mat4{ 1.0 }, glm::vec3(0.2, 0.2, 0.2));
                tri.transformMatrix = translation * scale;

                mRenderables.push_back(tri);
            }
        }

        RenderObject map{};
        map.mesh = getMesh("empire");
        map.material = getMaterial("texturedmesh");
        map.transformMatrix = glm::translate(glm::vec3{ 5,-10,0 });

        mRenderables.push_back(map);

        auto texturedMat=	getMaterial("texturedmesh");

        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.pNext = nullptr;
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = mDescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &mSingleTextureSetLayout;

        vkAllocateDescriptorSets(mDevice, &allocInfo, &texturedMat->textureSet);

        VkSamplerCreateInfo samplerInfo = init::samplerCreateInfo(VK_FILTER_NEAREST);

        VkSampler blockySampler;
        vkCreateSampler(mDevice, &samplerInfo, nullptr, &blockySampler);

        mMainDeletionQueue.push_function([=]() {
            vkDestroySampler(mDevice, blockySampler, nullptr);
        });

        VkDescriptorImageInfo imageBufferInfo;
        imageBufferInfo.sampler = blockySampler;
        imageBufferInfo.imageView = mLoadedTextures["empire_diffuse"].imageView;
        imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet texture1 = init::writeDescriptorImage(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texturedMat->textureSet, &imageBufferInfo, 0);

        vkUpdateDescriptorSets(mDevice, 1, &texture1, 0, nullptr);
    }

    FrameData &Engine::getCurrentFrame() {
        return mFrames[mFrameNumber % cFrameOverlap];
    }

    AllocatedBuffer Engine::createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) {
        //allocate vertex buffer
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.pNext = nullptr;

        bufferInfo.size = allocSize;
        bufferInfo.usage = usage;


        VmaAllocationCreateInfo vmaAllocInfo = {};
        vmaAllocInfo.usage = memoryUsage;

        AllocatedBuffer newBuffer{};

        //allocate the buffer
        VK_CHECK(vmaCreateBuffer(mAllocator, &bufferInfo, &vmaAllocInfo,
                                 &newBuffer.buffer,
                                 &newBuffer.allocation,
                                 nullptr));

        return newBuffer;
    }

    size_t Engine::padUniformBufferSize(size_t originalSize)
    {
        // Calculate required alignment based on minimum device offset alignment
        size_t minUboAlignment = mGpuProperties.limits.minUniformBufferOffsetAlignment;
        size_t alignedSize = originalSize;
        if (minUboAlignment > 0) {
            alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
        }
        return alignedSize;
    }

    void Engine::loadImages() {
        Texture lostEmpire{};

        if(utils::loadImageFromFile(*this, "../res/textures/lost_empire-RGBA.png", lostEmpire.image)){
            VkImageViewCreateInfo imageinfo = init::imageViewCreateInfo(VK_FORMAT_R8G8B8A8_SRGB, lostEmpire.image.image, VK_IMAGE_ASPECT_COLOR_BIT);
            vkCreateImageView(mDevice, &imageinfo, nullptr, &lostEmpire.imageView);
        }
        else{
            std::cout << "Error loading lost empire image " << std::endl;
            exit(1);
        }

        mMainDeletionQueue.push_function([=]() {
            vkDestroyImageView(mDevice, lostEmpire.imageView, nullptr);
        });

        mLoadedTextures["empire_diffuse"] = lostEmpire;
    }
}