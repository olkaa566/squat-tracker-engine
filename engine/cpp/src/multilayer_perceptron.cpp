//#include <vector>
#include <stdexcept>
#include <cmath> 
#include "multilayer_perceptron.hpp"

using namespace std;

// Constructor: Initialize the network architecture
MultiLayerPerceptron::MultiLayerPerceptron(int input_nodes, int hidden_nodes, int output_nodes, float lr) 
    : weights_hidden(input_nodes, hidden_nodes), biases_hidden(1, hidden_nodes), weights_output(hidden_nodes, output_nodes),
      biases_output(1, output_nodes), learning_rate(lr), hidden_buffer(1, hidden_nodes), output_buffer(1, output_nodes) {}

// Sigmoid squish function
float MultiLayerPerceptron::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Derivative of Sigmoid
float MultiLayerPerceptron::sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

float MultiLayerPerceptron::relu(float x) const {
    return std::max(0.0f, x);
}

float MultiLayerPerceptron::relu_derivative(float x) const {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Forward pass
Matrix& MultiLayerPerceptron::forward(Matrix& inputs) {
    hidden_buffer = inputs.dot(weights_hidden);
    
    std::vector<float>& h_data = hidden_buffer.get_data(); 
    const std::vector<float>& b_data = biases_hidden.get_data(); 
    
    for (size_t i = 0; i < h_data.size(); i++) {
        h_data[i] = relu(h_data[i] + b_data[i % biases_hidden.get_columns()]); 
    }

    output_buffer = hidden_buffer.dot(weights_output);
    
    std::vector<float>& o_data = output_buffer.get_data();
    const std::vector<float>& bo_data = biases_output.get_data();
    
    for (size_t i = 0; i < o_data.size(); i++) {
        o_data[i] = sigmoid(o_data[i] + bo_data[i % biases_output.get_columns()]);
    }
    
    return output_buffer;
}

// full dense propagation
void MultiLayerPerceptron::train(Matrix& inputs, Matrix& targets) {
    // forward pass
    Matrix hidden_layer = inputs.dot(weights_hidden);
    std::vector<float> h_data = hidden_layer.get_data();
    std::vector<float> b_hidden_data = biases_hidden.get_data();
    
    for (size_t i = 0; i < h_data.size(); i++) {
        h_data[i] = sigmoid(h_data[i] + b_hidden_data[i % biases_hidden.get_columns()]);
    }

    Matrix activated_hidden(hidden_layer.get_rows(), hidden_layer.get_columns(), h_data);

    Matrix output_layer = activated_hidden.dot(weights_output);
    std::vector<float> o_data = output_layer.get_data();
    std::vector<float> b_output_data = biases_output.get_data();
    
    for (size_t i = 0; i < o_data.size(); i++) {
        o_data[i] = sigmoid(o_data[i] + b_output_data[i % biases_output.get_columns()]);
    }

    // output layer error and gradients
    std::vector<float> t_data = targets.get_data();
    std::vector<float> output_gradients_data(o_data.size());
    
    for (size_t i = 0; i < o_data.size(); i++) {
        float error = t_data[i] - o_data[i];
        output_gradients_data[i] = error * sigmoid_derivative(o_data[i]);
    }

    Matrix gradients_output(1, output_gradients_data.size(), output_gradients_data);

    // update outuput weights and biases
    Matrix hidden_T = activated_hidden.transpose();
    Matrix weight_output_deltas = hidden_T.dot(gradients_output);
    
    std::vector<float> wo_data = weights_output.get_data();
    std::vector<float> wod_data = weight_output_deltas.get_data();
    for (size_t i = 0; i < wo_data.size(); i++) { 
        wo_data[i] += wod_data[i] * learning_rate; 
    }

    weights_output = Matrix(weights_output.get_rows(), weights_output.get_columns(), wo_data);

    for (size_t i = 0; i < b_output_data.size(); i++) { 
        b_output_data[i] += output_gradients_data[i] * learning_rate; 
    }

    biases_output = Matrix(1, biases_output.get_columns(), b_output_data);

    // chain rule
    Matrix weights_output_T = weights_output.transpose();
    Matrix hidden_errors = gradients_output.dot(weights_output_T);
    
    std::vector<float> he_data = hidden_errors.get_data();
    std::vector<float> hidden_gradients_data(he_data.size());
    
    for (size_t i = 0; i < he_data.size(); i++) {
        hidden_gradients_data[i] = he_data[i] * sigmoid_derivative(h_data[i]);
    }

    Matrix gradients_hidden(1, hidden_gradients_data.size(), hidden_gradients_data);

    // update hidden weights and biases
    Matrix inputs_T = inputs.transpose();
    Matrix weight_hidden_deltas = inputs_T.dot(gradients_hidden);

    std::vector<float> wh_data = weights_hidden.get_data();
    std::vector<float> whd_data = weight_hidden_deltas.get_data();
    for (size_t i = 0; i < wh_data.size(); i++) { 
        wh_data[i] += whd_data[i] * learning_rate; 
    }

    weights_hidden = Matrix(weights_hidden.get_rows(), weights_hidden.get_columns(), wh_data);

    for (size_t i = 0; i < b_hidden_data.size(); i++) { 
        b_hidden_data[i] += hidden_gradients_data[i] * learning_rate; 
    }

    biases_hidden = Matrix(1, biases_hidden.get_columns(), b_hidden_data);
}