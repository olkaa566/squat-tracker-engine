#pragma once
#include <array>

struct Kernel {
    float weights[9]; 
    float divisor;    
};

namespace engine::vision::constants {
    
    //light changes from left to right, highlights vertical lines
    inline constexpr Kernel SOBEL_X = {
        {-1, 0, 1, -2, 0, 2, -1, 0, 1}, 1.0f 
    };

    //light changes from right to left, highlights horizontal lines
    inline constexpr Kernel SOBEL_Y = {
        {-1, -2, -1, 0, 0, 0, 1, 2, 1}, 1.0f 
    };
    
    inline constexpr Kernel GAUSSIAN_BLUR = {
        {1, 2, 1, 2, 4, 2, 1, 2, 1}, 16.0f 
    };
}