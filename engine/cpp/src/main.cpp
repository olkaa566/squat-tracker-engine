#include "../include/png_decoder.hpp"
#include "../include/vision.hpp"
#include <iostream>
#include <vector>

int main() {
    const char* input_file = "assets/test_image.png"; 
    const char* output_file = "assets/output_edges.png";
    
    engine::vision::PngInfo image_info;
    std::vector<uint8_t> raw_pixels;
    
    // Extract the raw pixels using stb_image (via png_decoder)
    if (!engine::vision::load_png_pixels(input_file, image_info, raw_pixels)) {
        return -1; // Stop if we can't find the image
    }

    // Bridge the pixels into a float matrix
    Matrix gray_img = image_to_matrix(raw_pixels, image_info.width, image_info.height);
    
    // Run the edge detection 
    float threshold = 0.5f; 
    Matrix edge_map = edge_detection(gray_img, threshold);
  
    
    // Save the final matrix back to a file using stb_image_write
    if (matrix_to_image(edge_map, output_file)) {
        std::cout << "See assets folder for output." << std::endl;
    }
    
    return 0;
}