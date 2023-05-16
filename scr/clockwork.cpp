#include "graphics/engine.hpp"

using namespace cw::graphics;

int main() {
    Engine engine;

    engine.init();

    engine.run();

    engine.cleanup();

    return 0;
}