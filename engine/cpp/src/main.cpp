#include "../include/png_decoder.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "--- Squat Tracker Engine Booting ---" << std::endl;
    
    const char* file_path = "../assets/test_image.png"; 
    
    engine::vision::PngInfo image_info;
    std::vector<uint8_t> raw_pixels;
    
    if (engine::vision::load_png_pixels(file_path, image_info, raw_pixels)) {
        std::cout << "Engine is ready to process computer vision matrix!" << std::endl;
    }
    
    return 0;
}