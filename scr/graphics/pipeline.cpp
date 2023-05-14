#include <iostream>
#include "pipeline.hpp"

namespace cw::graphics{

    VkPipeline Pipeline::buildPipeline(VkDevice device, VkRenderPass pass) {
        //make viewport state from our stored viewport and scissor.
        //at the moment we won't support multiple viewports or scissors
        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.pNext = nullptr;

        viewportState.viewportCount = 1;
        viewportState.pViewports = &mViewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &mScissor;

        //setup dummy color blending. We aren't using transparent objects yet
        //the blending is just "no blend", but we do write to the color attachment
        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.pNext = nullptr;

        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &mColorBlendAttachment;

        //build the actual pipeline
        //we now use all the info structs we have been writing into this one to create the pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = nullptr;

        pipelineInfo.stageCount = mShaderStages.size();
        pipelineInfo.pStages = mShaderStages.data();
        pipelineInfo.pVertexInputState = &mVertexInputInfo;
        pipelineInfo.pInputAssemblyState = &mInputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &mRasterizer;
        pipelineInfo.pMultisampleState = &mMultisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = mPipelineLayout;
        pipelineInfo.renderPass = pass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        //it's easy to error out on create graphics pipeline, so we handle it a bit better than the common VK_CHECK case
        VkPipeline newPipeline;
        if (vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
            std::cout << "failed to create pipeline\n";
            return VK_NULL_HANDLE; // failed to create graphics pipeline
        }
        else
        {
            return newPipeline;
        }
    }
}