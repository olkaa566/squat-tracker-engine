#include "geometry.hpp"
#include <cmath> 

float calculate_knee_angle(float hip_x, float hip_y, float knee_x, float knee_y, float ankle_x, float ankle_y) {
    float v1_x = hip_x - knee_x;
    float v1_y = hip_y - knee_y;

    float v2_x = ankle_x - knee_x;
    float v2_y = ankle_y - knee_y;

    float mag_v1 = std::sqrt((v1_x * v1_x) + (v1_y * v1_y));
    float mag_v2 = std::sqrt((v2_x * v2_x) + (v2_y * v2_y));

    float dot_product = (v1_x * v2_x) + (v1_y * v2_y);

    float cos_angle = dot_product / (mag_v1 * mag_v2);
    
    if (cos_angle > 1.0f){
        cos_angle = 1.0f;
    } 
    
    if (cos_angle < -1.0f) {
        cos_angle = -1.0f;
    }

    float angle_radians = std::acos(cos_angle);
    float angle_degrees = angle_radians * (180.0f / 3.14159265f);

    return angle_degrees;
}