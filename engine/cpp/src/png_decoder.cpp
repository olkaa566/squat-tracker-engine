#include <iostream>
#include <vector>
#include "../include/png_decoder.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h" 

namespace engine::vision {

bool load_png_pixels(const char* filename, PngInfo& out_info, std::vector<uint8_t>& out_pixels) {
    std::cout << "Loading Image via stb_image." << std::endl;

    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 3);

    if (!data) {
        std::cerr << "Fatal Error: Could not load or decode the image." << std::endl;
        return false;
    }

    out_info.width = static_cast<uint32_t>(width);
    out_info.height = static_cast<uint32_t>(height);
    out_info.bit_depth = 8;
    out_info.color_type = 2; 

    size_t total_bytes = width * height * 3;
    out_pixels.assign(data, data + total_bytes);
    stbi_image_free(data);

    std::cout << "Width: " << out_info.width << " px" << std::endl;
    std::cout << "Height: " << out_info.height << " px" << std::endl;
    std::cout << "Extracted " << out_pixels.size() << " bytes of raw pixel data." << std::endl;

    return true;
}

} 