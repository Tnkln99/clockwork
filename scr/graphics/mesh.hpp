#ifndef CLOCKWORK_MESH_HPP
#define CLOCKWORK_MESH_HPP

#include "types.hpp"

#include <vector>
#include <glm/vec3.hpp>

namespace cw::graphics{
    struct VertexInputDescription {
        std::vector<VkVertexInputBindingDescription> mBindings;
        std::vector<VkVertexInputAttributeDescription> mAttributes;

        VkPipelineVertexInputStateCreateFlags mFlags = 0;
    };

    struct Vertex{
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec3 color;

        static VertexInputDescription getVertexDescription();
    };

    struct Mesh{
        std::vector<Vertex> mVertices;
        AllocatedBuffer mVertexBuffer;
    };
}


#endif //CLOCKWORK_MESH_HPP
