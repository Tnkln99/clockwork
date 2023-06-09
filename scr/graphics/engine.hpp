#ifndef CLOCKWORK_ENGINE_HPP
#define CLOCKWORK_ENGINE_HPP

#include "pipeline.hpp"
#include "mesh.hpp"

#include <vector>
#include <deque>
#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <memory>

namespace cw::graphics {
    struct Texture {
        AllocatedImage image;
        VkImageView imageView;
    };

    struct UploadContext {
        VkFence uploadFence;
        VkCommandPool commandPool;
        VkCommandBuffer commandBuffer;
    };

    struct GPUObjectData{
        glm::mat4 modelMatrix;
    };

    struct GPUSceneData {
        glm::vec4 fogColor; // w is for exponent
        glm::vec4 fogDistances; //x for min, y for max, zw unused.
        glm::vec4 ambientColor;
        glm::vec4 sunlightDirection; //w for sun power
        glm::vec4 sunlightColor;
    };

    struct GPUCameraData{
        glm::mat4 view;
        glm::mat4 proj;
        glm::mat4 viewProj;
    };

    struct FrameData {
        VkSemaphore presentSemaphore, renderSemaphore;
        VkFence renderFence;

        VkCommandPool commandPool;
        VkCommandBuffer mainCommandBuffer;

        //buffer that holds a single GPUCameraData to use when rendering
        AllocatedBuffer cameraBuffer;
        VkDescriptorSet globalDescriptor;

        AllocatedBuffer objectBuffer;
        VkDescriptorSet objectDescriptor;
    };

    struct MeshPushConstants {
        glm::vec4 data;
        glm::mat4 renderMatrix;
    };

    struct Material {
        VkDescriptorSet textureSet{VK_NULL_HANDLE}; //texture defaulted to null
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
    };

    struct RenderObject {
        Mesh* mesh;
        Material* material;
        glm::mat4 transformMatrix;
    };

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

    constexpr unsigned int cFrameOverlap = 2;

    class Engine {
    public:
        void init();

        void run();
        void draw();

        void cleanup();

        void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);

        UploadContext mUploadContext;

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

        // frame storage
        FrameData mFrames[cFrameOverlap];

        VkRenderPass mRenderpass;
        std::vector<VkFramebuffer> mFramebuffers;

        VkImageView mDepthImageView;
        AllocatedImage mDepthImage;

        //the format for the depth image
        VkFormat mDepthFormat;

        DeletionQueue mMainDeletionQueue;

        //default array of renderable objects
        std::vector<RenderObject> mRenderables;

        std::unordered_map<std::string,Material> mMaterials;
        std::unordered_map<std::string,Mesh> mMeshes;

        VkDescriptorSetLayout mGlobalSetLayout;
        VkDescriptorSetLayout mObjectSetLayout;
        VkDescriptorSetLayout mSingleTextureSetLayout;
        VkDescriptorPool mDescriptorPool;

        GPUSceneData mSceneParameters;
        AllocatedBuffer mSceneParameterBuffer;

        //create material and add it to the map
        Material* createMaterial(VkPipeline pipeline, VkPipelineLayout layout,const std::string& name);

        //returns nullptr if it can't be found
        Material* getMaterial(const std::string& name);

        //returns nullptr if it can't be found
        Mesh* getMesh(const std::string& name);

        // getter for the frame we are rendering right now
        FrameData& getCurrentFrame();

        //our draw function
        void drawObjects(VkCommandBuffer cmd);

        AllocatedBuffer createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

        std::unordered_map<std::string, Texture> mLoadedTextures;

        void loadImages();
    private:
        void initWindow();
        void initVulkan();
        void initSwapchain();
        void initCommands();
        void initDefaultRenderpass();
        void initFramebuffers();
        void initSyncStructure();
        void initPipelines();
        void initDescriptors();

        void loadMeshes();
        void uploadMesh(Mesh & mesh);

        void initScenes();

        //loads a shader module from a spir-v file. Returns false if it errors
        bool loadShaderModule(const char* filePath, VkShaderModule* outShaderModule) const;
        size_t padUniformBufferSize(size_t originalSize);

        bool mIsInit {false};
        int mFrameNumber {0};

        VkExtent2D mWindowExtent {800, 600};
        GLFWwindow* mWindow{ nullptr };

        VkPhysicalDeviceProperties mGpuProperties;
    };
}

#endif //CLOCKWORK_ENGINE_HPP
