#pragma once
#include <cstdint>
#include <vector>

namespace engine::vision {
    struct PngInfo {
        uint32_t width;
        uint32_t height;
        uint8_t bit_depth;
        uint8_t color_type;
    };

    bool load_png_pixels(const char* filename, PngInfo& out_info, std::vector<uint8_t>& out_pixels);
}