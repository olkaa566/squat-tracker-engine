#include <array>
#include "matrix.hpp"
#include "vision_constants.hpp"

void convolve(const Matrix& input, const Kernel& kernel, Matrix& ouput);
Matrix image_to_matrix(const std::vector<uint8_t>& raw_rgb, int width, int height);
Matrix edge_detection(const Matrix& grayscale_image, float treshold);
bool matrix_to_image(const Matrix& matrix, const char* output_filename);