#include <iostream>
#include <ctime>
#include <chrono>

#include "matrix.hpp"

int main() {
    using clk = std::chrono::high_resolution_clock;
    const auto den = clk::duration::period::den;

    srand(unsigned(time(nullptr)));

    matrix<float> m(6, 2);
    m.random_init();
    m /= 10;

    auto n = m.transpose();
    auto multi = m * n;
    auto copy = m;
    copy = m;

    std::cout << "origin:\n" << m << std::endl;
    std::cout << "copy:\n" << copy << std::endl;
    std::cout << "transpose:\n" << n << std::endl;
    std::cout << "multi:\n" << multi << std::endl;
    std::cout << "sub:\n" << m - copy << std::endl;
    std::cout << "add:\n" << m + copy << std::endl;
    std::cout << "hadamard:\n" << m.hadamard(copy) << std::endl;
    std::cout << "sigmoid:\n" << m.sigmoid() << std::endl;
    std::cout << "sigmoid.derivate:\n" << m.sigmoid().sigmoid_derivative() << std::endl;
    std::cout << "tanh:\n" << m.tanh() << std::endl;
    std::cout << "tanh.derivate:\n" << m.tanh().tanh_derivative() << std::endl;
    std::cout << "relu:\n" << m.relu() << std::endl;
    std::cout << "relu.derivate:\n" << m.relu().relu_derivative() << std::endl;
    std::cout << "softmax:\n" << m.softmax() << std::endl;
    std::cout << "softmax.derivate:\n" << m.softmax().softmax_derivative() << std::endl;
    std::cout << "softmax.sum: " << m.softmax().sum() << std::endl;
    std::cout << "m[3][1]: " << m[3][1] << std::endl;

    matrix<float> large(2048, 256);
    large.random_init();

    auto begin = clk::now();
    auto res = large * large.transpose();
    auto end = clk::now();
    auto t = static_cast<float>((end - begin).count()) / den;
    std::cout << "time (omp parallel): " << t << " s" << std::endl;

    auto begin_no_parallel = clk::now();
    auto res_no_parallel = large.no_parallel_mult(large.transpose());
    auto end_no_parallel = clk::now();
    auto t_no_parallel = static_cast<float>((end_no_parallel - begin_no_parallel).count()) / den;
    std::cout << "time (no parallel): " << t_no_parallel << " s" << std::endl;
    return 0;
}