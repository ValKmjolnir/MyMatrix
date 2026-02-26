#include "matrix.hpp"
#include <iostream>
#include <cassert>

void test_large_softmax() {
    // Create a test matrix
    matrix<double> test_matrix(512, 1024);
    test_matrix.random_init();

    matrix<double> softmax_result = test_matrix.softmax();
    // Verify that each row of softmax result sums to 1
    for (size_t i = 0; i < softmax_result.get_row(); ++i) {
        double row_sum = 0;
        for (size_t j = 0; j < softmax_result.get_col(); ++j) {
            row_sum += softmax_result[i][j];
        }
        assert(std::abs(row_sum - 1.0) < 1e-6); // Check if row sum is close to 1
    }

    std::cout << "Large Softmax Verification Test Passed!" << std::endl;
}

void test_softmax_cross_entropy_gradient() {
    matrix<double> test_matrix(2, 3);
    test_matrix.random_init();

    matrix<double> output = test_matrix.softmax();

    matrix<double> label(2, 3);
    label[0][0] = 1; label[0][1] = 0; label[0][2] = 0;
    label[1][0] = 0; label[1][1] = 1; label[1][2] = 0;

    matrix<double> gradient = output.softmax_cross_entropy_gradient(label);

    std::cout << "Output:" << std::endl;
    std::cout << output << std::endl;
    std::cout << "Label:" << std::endl;
    std::cout << label << std::endl;
    std::cout << "Gradient:" << std::endl;
    std::cout << gradient << std::endl;

    for (size_t i = 0; i < gradient.get_row(); ++i) {
        for (size_t j = 0; j < gradient.get_col(); ++j) {
            auto value = output[i][j] - label[i][j];
            assert(std::abs(value - gradient[i][j]) < 1e-10); // Check if gradient is close to zero
        }
    }

    std::cout << "Softmax Cross Entropy Gradient Test Passed!" << std::endl;
}

int main() {
    // Create a test matrix
    matrix<double> test_matrix(2, 3);
    test_matrix.random_init();

    std::cout << "Original Matrix:" << std::endl;
    std::cout << test_matrix << std::endl;

    // Test softmax function
    matrix<double> softmax_result = test_matrix.softmax();
    std::cout << "Softmax Result:" << std::endl;
    std::cout << softmax_result << std::endl;

    test_large_softmax();
    test_softmax_cross_entropy_gradient();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}