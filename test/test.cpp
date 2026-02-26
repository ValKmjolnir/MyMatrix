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
    std::cout << "softmax.sum: " << m.softmax().sum() << std::endl;
    std::cout << "m[3][1]: " << m[3][1] << std::endl;

    matrix<float> large(2048, 4096);
    large.random_init();

    auto begin = clk::now();
    auto res = large * large.transpose();
    auto end = clk::now();
    auto t = static_cast<float>((end - begin).count()) / den;
    std::cout << "time (omp parallel): " << t << " s "
              << "(" << large.get_col() << " x " << large.get_row() << ")\n";

    auto begin_seq = clk::now();
    auto res_seq = large.mult_sequential(large.transpose());
    auto end_seq = clk::now();
    auto t_seq = static_cast<float>((end_seq - begin_seq).count()) / den;
    std::cout << "time (no parallel): " << t_seq << " s "
              << "(" << large.get_col() << " x " << large.get_row() << ")\n";
    return 0;
}