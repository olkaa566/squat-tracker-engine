#define _CRT_SECURE_NO_WARNINGS
#include <array>
#include <algorithm>
#include <cmath>
#include "image_utils.hpp"
#include "matrix.hpp"
#include "vision_constants.hpp"
#include "vision.hpp"
#include <cstdint>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

using namespace std;

int get_index(int x, int y, int total_image_width){
    return (y * total_image_width) + x;
}

bool bounds_validation(int x, int y, int width, int height){
    return ((x >= 0 && x < width-1) && (y >= 0 && y < height-1));
}

float grayscale_normalizer(int raw_pixel_intensity){
    return raw_pixel_intensity / 255.0f;
}

std::array<Point, 8> get_neighbours(int x, int y, int width, int height){
    std::array<Point, 8> neighbours;

    for(int i = 0; i < 8; i++){
        neighbours[i].isvalid = false;
    }
    for(int count = 0; count < 8; count++){
        for(int i = -1; i <= 1; i++){
            for(int j = -1; j <= 1; j++){
                if(i == 0 && j == 0) continue;

                int neighbour_x = x + i;
                int neihgbour_y = y + j;

                if (bounds_validation(neighbour_x, neihgbour_y, width, height)){
                    neighbours[count].x = neighbour_x;
                    neighbours[count].y = neihgbour_y;
                    neighbours[count].isvalid = true;
                }
            }
        }
    }
    return neighbours;
} 

void convolve(const Matrix& input, const Kernel& kernel, Matrix& output){
    if (output.get_columns() != input.get_columns() || output.get_rows() != input.get_rows()) {
        output = Matrix(input.get_columns(), input.get_rows());
    }

    for(int i = 1; i < input.get_rows()-1; ++i){
        for(int j = 1; j < input.get_columns()-1; ++j){
            float running_sum = 0.0f;

            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    
                    int kernel_index = (ki + 1) * 3 + (kj + 1);
                    float weight = kernel.weights[kernel_index];
                    
                    float pixel = input.at(j + kj, i + ki); 
                    running_sum += (weight * pixel);
                }
            }
            // normalization
            float normalized_value = running_sum / kernel.divisor;
            
            // clamping - 0-255
            //float clamped_value = std::clamp(normalized_value, 0.0f, 255.0f);
            
            output.at(j, i) = normalized_value;
        }
    }
}

Matrix image_to_matrix(const vector<uint8_t>& raw_rgb, int width, int height){
    Matrix processed_matrix = Matrix(height, width);

    //read image left to right
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            int pos = y * width + x;
            pos *= 3; //3 sequential slots: red, green, blue

            float red_value = raw_rgb.at(pos);
            float green_value = raw_rgb.at(pos+1);
            float blue_value = raw_rgb.at(pos+2);

            float pixel_brightness = red_value*(0.299f) + green_value*(0.587f) + blue_value*(0.114f);
            pixel_brightness = grayscale_normalizer(static_cast<int>(pixel_brightness));

            processed_matrix.at(x, y) = pixel_brightness;

        }
    }
    return processed_matrix;
}

Matrix edge_detection(const Matrix& grayscale_image, float treshold){
    int width = grayscale_image.get_columns();
    int height = grayscale_image.get_rows();

    Matrix gradient_x(height, width);
    Matrix gradient_y(height, width);
    Matrix output_matrix(height, width);

    convolve(grayscale_image, engine::vision::constants::SOBEL_X, gradient_x);
    convolve(grayscale_image, engine::vision::constants::SOBEL_Y, gradient_y);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float gx = gradient_x.at(x, y);
            float gy = gradient_y.at(x, y);
            float magnitude = std::hypot(gx, gy);

            if(magnitude > treshold){
                output_matrix.at(x, y) = 255.0f;
            }
            else{
                output_matrix.at(x, y) = 0.0f;
            }
        }
    }

    //float treshold = 0.5f;
    /*
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            float curr_pixel = output_matrix.at(j, i);
            if(curr_pixel > treshold){
                output_matrix.at(j, i) = 1.0f;
            }
            else{
                output_matrix.at(j, i) = 0.0f;
            }
        }
    }
    */

    return output_matrix;
}

bool matrix_to_image(const Matrix& matrix, const char* output_filename){
    int height = matrix.get_rows();
    int width = matrix.get_columns();

    vector<uint8_t> image_data(width * height);

    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            float curr_value = matrix.at(x, y);
            curr_value = (int8_t)curr_value;

            int index = (y * width) + x;
            image_data[index] = static_cast<uint8_t>(curr_value);
        }
    }

    int success = stbi_write_png(output_filename, width, height, 1, image_data.data(), width);

    if (success != 0) {
        return true;  
    } else {
        return false; 
    }
}