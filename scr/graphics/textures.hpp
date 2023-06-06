#ifndef CLOCKWORK_TEXTURES_HPP
#define CLOCKWORK_TEXTURES_HPP

#include "types.hpp"
#include "engine.hpp"

namespace cw::graphics::utils{
    bool loadImageFromFile(Engine& engine, const char* file, AllocatedImage& outImage);
}

#endif //CLOCKWORK_TEXTURES_HPP
