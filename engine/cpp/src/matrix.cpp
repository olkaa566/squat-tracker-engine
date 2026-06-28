#include "matrix.hpp"
//#include "image_utils.hpp"
//#include "vision.hpp"

#include <stdexcept>
#include <cmath> 

using namespace std;

// Constructor
Matrix::Matrix(int r, int c) : rows(r), columns(c) {
    data.resize(r * c, 0.0f);
}

// Constructor with input data
Matrix::Matrix(int r, int c, const vector<float>& input_data) : rows(r), columns(c), data(input_data) {
    if (input_data.size() != r * c) {
        throw invalid_argument("Data size does not match matrix dimensions.");
    }
}

// Getters
int Matrix::get_rows() const {
    return rows;
}

int Matrix::get_columns() const {
    return columns;
}

std::vector<float>& Matrix::get_data() {
    return data; 
}

const std::vector<float>& Matrix::get_data() const {
    return data;
}

// Transpose
Matrix Matrix::transpose() {
    Matrix result(columns, rows);

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < columns; ++x) {
            //  x becomes the row (y-axis) in the new matrix
            result.at(y, x) = this->at(x, y); 
        }
    }
    return result;
}

// Dot Product
Matrix Matrix::dot(const Matrix& other) {
    if (this->columns != other.get_rows()) {
        throw std::invalid_argument("Matrix dot product failed: Incompatible dimensions.");
    }

    // The resulting matrix takes the outer dimensions: (A rows x B cols)
    Matrix result(this->rows, other.get_columns());

    for (int y = 0; y < result.get_rows(); ++y) {
        for (int x = 0; x < result.get_columns(); ++x) {
            float sum = 0.0f;
            
            // Calculate the dot product of the current row of 'this' 
            // and the current column of 'other'
            for (int k = 0; k < this->columns; ++k) {
                sum += this->at(k, y) * other.at(x, k);
            }
            result.at(x, y) = sum;
        }
    }
    return result;
}