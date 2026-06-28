#pragma once
#include <vector>
#include "matrix.hpp"

class MultiLayerPerceptron{
private:
    Matrix weights_hidden;
    Matrix biases_hidden;
    Matrix weights_output;
    Matrix biases_output;
    Matrix hidden_buffer; 
    Matrix output_buffer;

    float learning_rate;

    float sigmoid(float x);
    float sigmoid_derivative(float x);

    float relu(float x) const;
    float relu_derivative(float x) const;

public:
    MultiLayerPerceptron(int input_nodes, int hidden_nodes, int output_nodes, float learning_rate = 0.01f);

    Matrix& forward(Matrix& inputs);
    void train(Matrix& inputs, Matrix& targets);
};