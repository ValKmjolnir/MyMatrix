#include <matrix.hpp>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

class neural_network {
private:
    std::vector<matrix<float>> input_data;
    matrix<float> hidden_weight;
    matrix<float> hidden_bias;
    std::vector<matrix<float>> hidden_results;
    matrix<float> output_weight;
    matrix<float> output_bias;
    std::vector<matrix<float>> output_results;
    std::vector<matrix<float>> output_data;

public:
    neural_network():
        hidden_weight(2, 8), hidden_bias(1, 8),
        output_weight(8, 1), output_bias(1, 1) {
        hidden_weight.random_init();
        hidden_weight /= 10.0;
        hidden_bias.random_init();
        hidden_bias /= 10.0;

        output_weight.random_init();
        output_weight /= 10.0;
        output_bias.random_init();
        output_bias /= 10.0;

        matrix<float> data = matrix<float>(2, 1);
        matrix<float> out = matrix<float>(1, 1);

        data[0][0] = 0;
        data[0][1] = 0;
        out[0][0] = 0;
        input_data.push_back(data);
        hidden_results.push_back(matrix<float>(1, 1));
        output_results.push_back(matrix<float>(1, 1));
        output_data.push_back(out);

        data[0][0] = 0;
        data[0][1] = 1;
        out[0][0] = 1;
        input_data.push_back(data);
        hidden_results.push_back(matrix<float>(1, 1));
        output_results.push_back(matrix<float>(1, 1));
        output_data.push_back(out);

        data[0][0] = 1;
        data[0][1] = 0;
        out[0][0] = 1;
        input_data.push_back(data);
        hidden_results.push_back(matrix<float>(1, 1));
        output_results.push_back(matrix<float>(1, 1));
        output_data.push_back(out);

        data[0][0] = 1;
        data[0][1] = 1;
        out[0][0] = 0;
        input_data.push_back(data);
        hidden_results.push_back(matrix<float>(1, 1));
        output_results.push_back(matrix<float>(1, 1));
        output_data.push_back(out);
    }

    void train() {
        const float learning_rate = 0.5;

        for (int i = 0; i < 10000; ++i) {
            float total_error = 0;
            for (int j = 0; j < input_data.size(); ++j) {
                hidden_results[j] = (input_data[j].transpose() * hidden_weight + hidden_bias).tanh();
                output_results[j] = (hidden_results[j] * output_weight + output_bias).sigmoid();

                auto error = output_results[j] - output_data[j];
                total_error += error.pow(2).sum();

                auto diff = error;
                diff *= -0.5 * learning_rate;
                auto output_diff = output_results[j].sigmoid_derivative().hadamard(diff);
                output_weight = output_weight + hidden_results[j].transpose() * output_diff;
                output_bias = output_bias + output_diff;
                auto hidden_diff = (output_diff * output_weight.transpose()).hadamard(hidden_results[j].tanh_derivative());
                hidden_weight = hidden_weight + input_data[j] * hidden_diff;
                hidden_bias = hidden_bias + hidden_diff;
            }

            std::cout << "Total error: " << total_error << std::endl;
            if (total_error < 0.01) {
                std::cout << "Training complete after " << i << " times" << std::endl;
                break;
            }
        }
    }

    void calc() {
        for (int i = 0; i < input_data.size(); ++i) {
            hidden_results[i] = (input_data[i].transpose() * hidden_weight + hidden_bias).tanh();
            output_results[i] = (hidden_results[i] * output_weight + output_bias).sigmoid();
            std::cout << "Input: " << input_data[i].transpose();
            std::cout << "Output: " << output_results[i] << std::endl;
        }
    }
};

int main() {
    srand(unsigned(time(nullptr)));

    neural_network nn;
    nn.train();
    nn.calc();
    return 0;
}