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

    // Test softmax_derivative function
    matrix<double> softmax_derivative_result = softmax_result.softmax_derivative();
    std::cout << "Softmax Derivative Result:" << std::endl;
    std::cout << softmax_derivative_result << std::endl;

    test_large_softmax();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}