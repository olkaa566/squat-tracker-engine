#pragma once
#include <vector>

class Matrix{
private:
    int rows;
    int columns;
    std::vector<float> data;

public:

    // Constructors
    Matrix(int r, int c);
    Matrix(int r, int c, const std::vector<float>& data);

    // Getters
    int get_rows() const;
    int get_columns() const;
    
    inline float at(int x, int y) const {
        return data[(y * columns) + x];
    }

    inline float& at(int x, int y) {
        return data[(y * columns) + x];
    }

   const std::vector<float>& get_data() const;
   std::vector<float>& get_data();

    // Math Operations
    Matrix transpose();
    Matrix dot(const Matrix& other);
};