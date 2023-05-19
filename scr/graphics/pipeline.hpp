#ifndef CLOCKWORK_PIPELINE_HPP
#define CLOCKWORK_PIPELINE_HPP

#include "initializers.hpp"

#include <vector>

namespace cw::graphics{
    class Pipeline {
    private:

    public:
        std::vector<VkPipelineShaderStageCreateInfo> mShaderStages;
        VkPipelineVertexInputStateCreateInfo mVertexInputInfo;
        VkPipelineInputAssemblyStateCreateInfo mInputAssembly;
        VkViewport mViewport;
        VkRect2D mScissor;
        VkPipelineRasterizationStateCreateInfo mRasterizer;
        VkPipelineColorBlendAttachmentState mColorBlendAttachment;
        VkPipelineMultisampleStateCreateInfo mMultisampling;
        VkPipelineLayout mPipelineLayout;
        VkPipelineDepthStencilStateCreateInfo mDepthStencil;

        VkPipeline buildPipeline(VkDevice device, VkRenderPass pass);
    };
}

#endif //CLOCKWORK_PIPELINE_HPP
