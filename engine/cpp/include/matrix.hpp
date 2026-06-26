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

    std::vector<float> get_data() const;

    // Math Operations
    Matrix transpose();
    Matrix dot(const Matrix& other);
};

class MultiLayerPerceptron{
private:
    Matrix weights_hidden;
    Matrix biases_hidden;
    Matrix weights_output;
    Matrix biases_output;

    float learning_rate;

    float sigmoid(float x);
    float sigmoid_derivative(float x);

public:
    MultiLayerPerceptron(int input_nodes, int hidden_nodes, int output_nodes, float learning_rate = 0.01f);

    Matrix forward(Matrix& inputs);
    void train(Matrix& inputs, Matrix& targets);
};

float calculate_joint_angle(float ax, float ay, float bx, float by, float cx, float cy);