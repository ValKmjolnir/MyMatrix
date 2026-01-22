/* matrix.hpp by ValKmjolnir 2020/5/3             */
/* Rewrite    by ValKmjolnir 2022/11/16           */
/* Update     by ValKmjolnir 2025/01/19           */
/*            by ValKmjolnir 2026/01/22           */

#pragma once

#include <omp.h>

#include <iostream>
#include <cmath>
#include <sstream>
#include <cstring>
#include <cstdint>
#include <cstdlib>

template<typename T>
class matrix {
private:
    size_t row;
    size_t col;
    T* num;

private:
    [[noreturn]]
    void report(const char* calc, const matrix<T>& Temp) const {
        if (Temp.row != row || Temp.col != col) {
            std::cout << "Error: matrix size not match! ";
            std::cout << "In calculation " << calc << ": expect (";
            std::cout << row << "x" << col << "), but get (";
            std::cout << Temp.row << "x" << Temp.col << ").\n";
        } else {
            std::cout << "Error: unknown error! ";
            std::cout << "In calculation " << calc << ": expect (";
            std::cout << row << "x" << col << "), but get (";
            std::cout << Temp.row << "x" << Temp.col << ").\n";
        }
        std::exit(-1);
    }

public:
    matrix(const size_t __row, const size_t __col) {
        row = __row;
        col = __col;
        if (row > 0 && col > 0) {
            num = new T [row * col];
        } else {
            row = 0;
            col = 0;
            num = nullptr;
        }
        return;
    }

    matrix(const matrix<T>& Temp) {
        row = Temp.row;
        col = Temp.col;
        if (row > 0 && col > 0) {
            num = new T [row * col];
            #pragma omp parallel for
            for (size_t i = 0; i < row * col; ++i)
                num[i] = Temp.num[i];
        } else {
            row = 0;
            col = 0;
            num = nullptr;
        }
        return;
    }

    ~matrix() {
        if (num) {
            delete[] num;
        }
        return;
    }

public:
    matrix operator+(const matrix<T>& B) {
        if (this->row == B.row && this->col == B.col) {
            auto Temp = *this;
            #pragma omp parallel for
            for (size_t i = 0; i < row * col; ++i)
                Temp.num[i] += B.num[i];
            return Temp;
        } else {
            report("+", B);
        }
    }

    matrix operator-(const matrix<T>& B) {
        if (this->row == B.row && this->col == B.col) {
            auto Temp = *this;
            #pragma omp parallel for
            for (size_t i = 0; i < row * col; ++i)
                Temp.num[i] -= B.num[i];
            return Temp;
        } else {
            report("-", B);
        }
    }

    matrix operator*(const matrix<T>& B) {
        if (!this->row || !this->col || !B.row || !B.col) {
            report("*", B);
        } else if (this->col != B.row) {
            report("*", B);
        }

        matrix<T> Temp(this->row, B.col);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < Temp.row; ++i)
            for (size_t j = 0; j < Temp.col; ++j) {
                T trans = 0;
                for (size_t k = 0; k < this->col; ++k)
                    trans += this->num[i * this->col + k] * B.num[k * B.col + j];
                Temp.num[i * Temp.col + j] = trans;
            }
        return Temp;
    }

    matrix& operator=(const matrix<T>& B) {
        if (num) {
            delete[] num;
        }

        row = B.row;
        col = B.col;
        if (row > 0 && col > 0) {
            num = new T [row * col];
            #pragma omp parallel for
            for (size_t i = 0; i < row * col; ++i)
                num[i] = B.num[i];
        } else {
            row = 0;
            col = 0;
            num = nullptr;
        }

        return *this;
    }

    matrix& operator+=(const matrix<T>& B) {
        if (this->row == B.row && this->col == B.col) {
            #pragma omp parallel for
            for (size_t i = 0; i < row * col; ++i)
                num[i] += B.num[i];
            return *this;
        } else {
            report("+=", B);
        }
    }

    matrix& operator*=(const T B) {
        #pragma omp parallel for
        for (size_t i = 0; i < row * col; ++i)
            num[i] *= B;
        return *this;
    }

    matrix& operator/=(const T B) {
        #pragma omp parallel for
        for (size_t i = 0; i < row * col; ++i)
            num[i] /= B;
        return *this;
    }

public:
    T* operator[](const size_t addr) {
        return addr >= row ? nullptr : &this->num[addr * col];
    }

public:
    T sum() const {
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < row * col; ++i)
            sum += num[i];
        return sum;
    }

    matrix hadamard(const matrix<T>& B) {
        if (!this->row || !this->col || !B.row || !B.col) {
            report("hadamard", B);
        } else if (this->row != B.row || this->col != B.col) {
            report("hadamard", B);
        }

        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = this->num[i] * B.num[i];
        return temp;
    }

    matrix transpose() {
        matrix<T> temp(this->col, this->row);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < this->row; ++i)
            for (size_t j = 0; j < this->col; ++j)
                temp.num[j * this->row + i] = this->num[i * this->col + j];
        return temp;
    }

    matrix no_parallel_mult(const matrix<T>& B) {
        if (!this->row || !this->col || !B.row || !B.col) {
            report("*", B);
        } else if (this->col != B.row) {
            report("*", B);
        }

        matrix<T> Temp(this->row, B.col);
        for (size_t i = 0; i < Temp.row; ++i)
            for (size_t j = 0; j < Temp.col; ++j) {
                T trans = 0;
                for (size_t k = 0; k < this->col; ++k)
                    trans += this->num[i * this->col + k] * B.num[k * B.col + j];
                Temp.num[i * Temp.col + j] = trans;
            }
        return Temp;
    }

public:
    void random_init()  {
        for (size_t i = 0; i < row * col; ++i)
            num[i] = ((rand() & 1)? 1 : -1) * static_cast<T>((rand() + 1) % 10);
    }

public:
    template<typename _T>
    friend std::ostream& operator<<(std::ostream& out, const matrix<_T>& m) {
        for (size_t i = 0; i < m.row; ++i)
            for (size_t j = 0; j < m.col; ++j)
                out << m.num[i * m.col + j] << ((char)(j == m.col - 1)? '\n' : ' ');
        return out;
    }

    template<typename _T>
    friend std::istream& operator>>(std::istream& in, const matrix<_T>& m) {
        for (size_t i = 0; i < m.row; ++i)
            for (size_t j = 0; j < m.col; ++j)
                in >> m.num[i * m.col + j];
        return in;
    }

public:
    matrix sigmoid() {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = 1 / (1 + std::exp(-this->num[i]));
        return temp;
    }

    matrix sigmoid_derivative() {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = this->num[i] * (1 - this->num[i]);
        return temp;
    }

    matrix tanh() {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = std::tanh(this->num[i]);
        return temp;
    }

    matrix tanh_derivative() {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = 1 - std::pow(std::tanh(this->num[i]), 2);
        return temp;
    }

    matrix relu() {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = this->num[i] > 0 ? this->num[i] : 0;
        return temp;
    }

    matrix relu_derivative() {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = this->num[i] > 0 ? 1 : 0;
        return temp;
    }

    matrix softmax() {
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < this->row * this->col; ++i)
            sum += std::exp(this->num[i]);
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = std::exp(this->num[i]) / sum;
        return temp;
    }

    matrix softmax_derivative() {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = this->num[i] * (1 - this->num[i]);
        return temp;
    }

    matrix pow(const T B) {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = std::pow(this->num[i], B);
        return temp;
    }

    matrix l1_norm() {
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < this->row * this->col; ++i)
            sum += std::abs(this->num[i]);
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = this->num[i]/sum;
        return temp;
    }

    matrix l2_norm() {
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < this->row * this->col; ++i)
            sum += std::pow(this->num[i], 2);
        sum = std::sqrt(sum);
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row* this->col; ++i)
            temp.num[i] = this->num[i]/sum;
        return temp;
    }

    void save(std::ostream& out) const {
        out.write((char*)&this->row, sizeof(size_t));
        out.write((char*)&this->col, sizeof(size_t));
        out.write((char*)this->num, sizeof(T) * this->row * this->col);
    }

    void load(std::istream& in) {
        in.read((char*)&this->row, sizeof(size_t));
        in.read((char*)&this->col, sizeof(size_t));

        auto tmp = matrix<T>(this->row, this->col);
        in.read((char*)tmp.num, sizeof(T) * this->row * this->col);
        *this = tmp;
    }
};
