//#include <vector>
#include <stdexcept>
#include <cmath> 
#include "multilayer_perceptron.hpp"
#include <iostream>
#include <random>
#include <fstream>

using namespace std;

// Constructor: Initialize the network architecture
MultiLayerPerceptron::MultiLayerPerceptron(int input_nodes, int hidden_nodes, int output_nodes, float lr) 
    : weights_hidden(input_nodes, hidden_nodes), biases_hidden(1, hidden_nodes), weights_output(hidden_nodes, output_nodes),
      biases_output(1, output_nodes), learning_rate(lr), hidden_buffer(1, hidden_nodes), output_buffer(1, output_nodes) {
    
    // 1. Setup random number generator (Xavier/He style distribution)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f); 

    // 2. Helper function to modify matrix data in-place
    auto randomize_matrix = [&](Matrix& m) {
        std::vector<float>& data = m.get_data();
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dis(gen);
        }
    };

    // 3. Apply random values to all weights and biases
    randomize_matrix(weights_hidden);
    randomize_matrix(biases_hidden);
    randomize_matrix(weights_output);
    randomize_matrix(biases_output);
}

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
    forward(inputs);

    std::vector<float>& o_data = output_buffer.get_data();
    std::vector<float>& h_data = hidden_buffer.get_data();
    std::vector<float>& t_data = targets.get_data();

    // output layer error and gradient
    std::vector<float> output_gradients_data(o_data.size());

    for (size_t i = 0; i < o_data.size(); i++) {
        float error = t_data[i] - o_data[i];
        output_gradients_data[i] = error * sigmoid_derivative(o_data[i]);
    }

    Matrix gradients_output(1, output_gradients_data.size(), output_gradients_data);

    // output weight deltas
    Matrix hidden_T = hidden_buffer.transpose();
    Matrix weight_output_deltas = hidden_T.dot(gradients_output);

    // Update output weights and biases (Używamy += z racji na wzór error = target - output)
    std::vector<float>& wo_data = weights_output.get_data();
    std::vector<float>& wod_data = weight_output_deltas.get_data();

    for (size_t i = 0; i < wo_data.size(); i++) { 
        wo_data[i] += wod_data[i] * learning_rate; // <-- ZMIENIONO NA +=
    }

    std::vector<float>& b_output_data = biases_output.get_data();
    
    for (size_t i = 0; i < b_output_data.size(); i++) { 
        b_output_data[i] += output_gradients_data[i] * learning_rate; // <-- ZMIENIONO NA +=
    }

    // chain rule + relu, hidden layer error + gradients
    Matrix weights_output_T = weights_output.transpose();
    Matrix hidden_errors = gradients_output.dot(weights_output_T);
    
    std::vector<float>& he_data = hidden_errors.get_data();
    std::vector<float> hidden_gradients_data(he_data.size());
    
    for (size_t i = 0; i < he_data.size(); i++) {
        hidden_gradients_data[i] = he_data[i] * relu_derivative(h_data[i]);
    }

    Matrix gradients_hidden(1, hidden_gradients_data.size(), hidden_gradients_data);

    // hidden weight deltas
    Matrix inputs_T = inputs.transpose();
    Matrix weight_hidden_deltas = inputs_T.dot(gradients_hidden);

    // Update hidden weights and biases (Używamy += z racji na wzór error = target - output)
    std::vector<float>& wh_data = weights_hidden.get_data();
    std::vector<float>& whd_data = weight_hidden_deltas.get_data();

    for (size_t i = 0; i < wh_data.size(); i++) { 
        wh_data[i] += whd_data[i] * learning_rate; // <-- ZMIENIONO NA +=
    }

    std::vector<float>& b_hidden_data = biases_hidden.get_data();

    for (size_t i = 0; i < b_hidden_data.size(); i++) { 
        b_hidden_data[i] += hidden_gradients_data[i] * learning_rate; // <-- ZMIENIONO NA +=
    }
}

float MultiLayerPerceptron::calculate_mse(const Matrix& target, const Matrix& output) const {
    const std::vector<float>& t_data = target.get_data();
    const std::vector<float>& o_data = output.get_data();
    
    float sum_squared_errors = 0.0f;

    for (size_t i = 0; i < t_data.size(); i++) {
        float error = t_data[i] - o_data[i];
        sum_squared_errors += (error * error); 
    }

    return sum_squared_errors / t_data.size();
}

void MultiLayerPerceptron::fit(std::vector<Matrix>& training_data, std::vector<Matrix>& target_data, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_epoch_loss = 0.0f;

        for (size_t i = 0; i < training_data.size(); i++) {
            train(training_data[i], target_data[i]);
            float sample_loss = calculate_mse(target_data[i], output_buffer);
            total_epoch_loss += sample_loss;
        }

        float average_loss = total_epoch_loss / training_data.size();

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            cout << "Epoch: " << epoch << " | Loss: " << average_loss << "\n";
        }
    }
}

void MultiLayerPerceptron::save_weights(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Nie mozna otworzyc pliku do zapisu: " + filename);
    }
    
    auto write_matrix = [&out](const Matrix& m) {
        const std::vector<float>& data = m.get_data();
        for (size_t i = 0; i < data.size(); i++) {
            out << data[i] << (i == data.size() - 1 ? "" : " ");
        }
        out << "\n";
    };

    write_matrix(weights_hidden);
    write_matrix(biases_hidden);
    write_matrix(weights_output);
    write_matrix(biases_output);
    
    out.close();
}

void MultiLayerPerceptron::load_weights(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Nie mozna otworzyc pliku do odczytu: " + filename);
    }

    auto read_matrix = [&in](Matrix& m) {
        std::vector<float>& data = m.get_data();
        for (size_t i = 0; i < data.size(); i++) {
            in >> data[i];
        }
    };

    read_matrix(weights_hidden);
    read_matrix(biases_hidden);
    read_matrix(weights_output);
    read_matrix(biases_output);
    
    in.close();
}