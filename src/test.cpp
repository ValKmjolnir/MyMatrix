#include "Matrix.hpp"

#include <iostream>
#include <ctime>

int main() {
    srand(unsigned(time(nullptr)));

    Matrix<float> m(6, 2);
    m.random_init();

    auto n = m.transpose();
    auto multi = m * n;
    auto copy = m;

    std::cout << "origin:\n" << m << std::endl;
    std::cout << "transpose:\n" << n << std::endl;
    std::cout << "multi:\n" << multi << std::endl;
    std::cout << "copy:\n" << copy << std::endl;
    std::cout << "sub:\n" << m - copy << std::endl;
    std::cout << "add:\n" << m + copy << std::endl;
    std::cout << "dot:\n" << m.hadamard(copy) << std::endl;
    return 0;
}