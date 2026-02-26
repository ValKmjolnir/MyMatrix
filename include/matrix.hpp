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
#include <stdexcept>
#include <random>
#include <type_traits>

template<typename T>
class matrix {
    static_assert(std::is_floating_point<T>::value, "T must be floating point type");
private:
    static constexpr size_t SMALL_MATRIX_THRESHOLD = 100;

    size_t row;
    size_t col;
    T* num;

private:
    void report(const char* calc, const matrix<T>& Temp) const {
        std::ostringstream oss;
        oss << "Error: matrix size not match! In calculation " << calc
            << ": expect (" << row << " x " << col << "), but get ("
            << Temp.row << " x " << Temp.col << ").";
        throw std::runtime_error(oss.str());
    }

    void report_zero_size(const char* calc) const {
        std::ostringstream oss;
        oss << "Error: matrix size is zero! (" << calc << ")";
        throw std::runtime_error(oss.str());
    }

    void copy_data(const T* source, T* destination, size_t size) {
        if (size <= SMALL_MATRIX_THRESHOLD) {
            for (size_t i = 0; i < size; ++i) {
                destination[i] = source[i];
            }
        } else {
            #pragma omp parallel for
            for (size_t i = 0; i < size; ++i) {
                destination[i] = source[i];
            }
        }
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
            copy_data(Temp.num, num, row * col);
        } else {
            row = 0;
            col = 0;
            num = nullptr;
        }
        return;
    }

    matrix(matrix<T>&& Temp) noexcept {
        row = Temp.row;
        col = Temp.col;
        num = Temp.num;

        Temp.row = 0;
        Temp.col = 0;
        Temp.num = nullptr;
    }

    ~matrix() {
        if (num) {
            delete[] num;
        }
        return;
    }

public:
    auto get_row() const {
        return row;
    }

    auto get_col() const {
        return col;
    }

public:
    matrix operator+(const matrix<T>& B) const {
        if (this->row != B.row || this->col != B.col) {
            report("+", B);
        }
        matrix<T> Temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < row * col; ++i)
            Temp.num[i] = this->num[i] + B.num[i];
        return Temp;
    }

    matrix operator-(const matrix<T>& B) const {
        if (this->row != B.row || this->col != B.col) {
            report("-", B);
        }
        matrix<T> Temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < row * col; ++i)
            Temp.num[i] = this->num[i] - B.num[i];
        return Temp;
    }

    matrix operator*(const matrix<T>& B) const {
        if (!this->row || !this->col || !B.row || !B.col) {
            report_zero_size("*");
        } else if (this->col != B.row) {
            report("*", B);
        }

        matrix<T> Temp(this->row, B.col);
        #pragma omp parallel for
        for (size_t idx = 0; idx < Temp.row * Temp.col; ++idx)
            Temp.num[idx] = 0;

        const size_t BLOCK = 64;
        #pragma omp parallel for collapse(2)
        for (size_t ii = 0; ii < Temp.row; ii += BLOCK)
            for (size_t jj = 0; jj < Temp.col; jj += BLOCK)
                for (size_t kk = 0; kk < this->col; kk += BLOCK) {
                    const size_t i_end = std::min(ii + BLOCK, Temp.row);
                    const size_t j_end = std::min(jj + BLOCK, Temp.col);
                    const size_t k_end = std::min(kk + BLOCK, this->col);

                    for (size_t i = ii; i < i_end; ++i) {
                        const size_t tbase = i * Temp.col;
                        const size_t abase = i * this->col;
                        for (size_t k = kk; k < k_end; ++k) {
                            const T a = this->num[abase + k];
                            const size_t bbase = k * B.col;
                            for (size_t j = jj; j < j_end; ++j) {
                                Temp.num[tbase + j] += a * B.num[bbase + j];
                            }
                        }
                    }
                }
        return Temp;
    }

    matrix& operator=(const matrix<T>& B) {
        if (this == &B) {
            return *this; // self-assignment check
        }

        if (num) {
            delete[] num;
        }

        row = B.row;
        col = B.col;
        if (row > 0 && col > 0) {
            num = new T [row * col];
            copy_data(B.num, num, row * col);
        } else {
            row = 0;
            col = 0;
            num = nullptr;
        }

        return *this;
    }

    matrix& operator=(matrix<T>&& B) noexcept {
        if (this == &B) {
            return *this; // self-assignment check
        }

        if (num) {
            delete[] num;
        }

        row = B.row;
        col = B.col;
        num = B.num;

        B.row = 0;
        B.col = 0;
        B.num = nullptr;
        return *this;
    }

    matrix& operator+=(const matrix<T>& B) {
        if (this->row != B.row || this->col != B.col) {
            report("+=", B);
        }
        #pragma omp parallel for
        for (size_t i = 0; i < row * col; ++i)
            num[i] += B.num[i];
        return *this;
    }

    matrix& operator-=(const matrix<T>& B) {
        if (this->row != B.row || this->col != B.col) {
            report("-=", B);
        }
        #pragma omp parallel for
        for (size_t i = 0; i < row * col; ++i)
            num[i] -= B.num[i];
        return *this;
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

    const T* operator[](const size_t addr) const {
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

    matrix hadamard(const matrix<T>& B) const {
        if (!this->row || !this->col || !B.row || !B.col) {
            report_zero_size("hadamard");
        } else if (this->row != B.row || this->col != B.col) {
            report("hadamard", B);
        }

        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = this->num[i] * B.num[i];
        return temp;
    }

    matrix transpose() const {
        matrix<T> temp(this->col, this->row);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < this->row; ++i)
            for (size_t j = 0; j < this->col; ++j)
                temp.num[j * this->row + i] = this->num[i * this->col + j];
        return temp;
    }

    matrix mult_sequential(const matrix<T>& B) const {
        if (!this->row || !this->col || !B.row || !B.col) {
            report_zero_size("*");
        } else if (this->col != B.row) {
            report("*", B);
        }

        matrix<T> Temp(this->row, B.col);
        for (size_t idx = 0; idx < Temp.row * Temp.col; ++idx)
            Temp.num[idx] = 0;

        const size_t BLOCK = 64;
        for (size_t ii = 0; ii < Temp.row; ii += BLOCK)
            for (size_t jj = 0; jj < Temp.col; jj += BLOCK)
                for (size_t kk = 0; kk < this->col; kk += BLOCK) {
                    const size_t i_end = std::min(ii + BLOCK, Temp.row);
                    const size_t j_end = std::min(jj + BLOCK, Temp.col);
                    const size_t k_end = std::min(kk + BLOCK, this->col);

                    for (size_t i = ii; i < i_end; ++i) {
                        const size_t tbase = i * Temp.col;
                        const size_t abase = i * this->col;
                        for (size_t k = kk; k < k_end; ++k) {
                            const T a = this->num[abase + k];
                            const size_t bbase = k * B.col;
                            for (size_t j = jj; j < j_end; ++j) {
                                Temp.num[tbase + j] += a * B.num[bbase + j];
                            }
                        }
                    }
                }
        return Temp;
    }

public:
    void random_init() {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        static thread_local std::uniform_real_distribution<T> dis(-1.0, 1.0);

        for (size_t i = 0; i < row * col; ++i)
            num[i] = dis(gen);
    }

public:
    template<typename _T>
    friend std::ostream& operator<<(std::ostream& out, const matrix<_T>& m) {
        static_assert(std::is_floating_point<_T>::value, "_T must be floating point type");
        for (size_t i = 0; i < m.row; ++i)
            for (size_t j = 0; j < m.col; ++j)
                out << m.num[i * m.col + j] << ((char)(j == m.col - 1)? '\n' : ' ');
        return out;
    }

    template<typename _T>
    friend std::istream& operator>>(std::istream& in, matrix<_T>& m) {
        static_assert(std::is_floating_point<_T>::value, "_T must be floating point type");
        for (size_t i = 0; i < m.row; ++i)
            for (size_t j = 0; j < m.col; ++j)
                in >> m.num[i * m.col + j];
        return in;
    }

public:
    matrix sigmoid() const {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = 1 / (1 + std::exp(-this->num[i]));
        return temp;
    }

    matrix sigmoid_derivative() const {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = this->num[i] * (1 - this->num[i]);
        return temp;
    }

    matrix tanh() const {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = std::tanh(this->num[i]);
        return temp;
    }

    matrix tanh_derivative() const {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = 1 - std::pow(this->num[i], 2);
        return temp;
    }

    matrix relu() const {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = this->num[i] > 0 ? this->num[i] : 0;
        return temp;
    }

    matrix relu_derivative() const {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = this->num[i] > 0 ? 1 : 0;
        return temp;
    }

    matrix softmax() const {
        matrix<T> temp(this->row, this->col);
        for (size_t i = 0; i < this->row; ++i) {
            T max_val = this->num[i * this->col];  // find max value of each line
            for (size_t j = 1; j < this->col; ++j) {
                if (this->num[i * this->col + j] > max_val) {
                    max_val = this->num[i * this->col + j];
                }
            }

            T sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (size_t j = 0; j < this->col; ++j) {
                T exp_val = std::exp(this->num[i * this->col + j] - max_val);
                sum += exp_val;
                temp.num[i * temp.col + j] = exp_val;
            }

            #pragma omp parallel for
            for (size_t j = 0; j < this->col; ++j)
                temp.num[i * temp.col + j] /= sum;
        }
        return temp;
    }

    matrix softmax_cross_entropy_gradient(const matrix<T>& label) const {
        if (this->row != label.row || this->col != label.col) {
            report("matrix size mismatch", label);
        }
        // this->num must be softmax output (not raw logits)
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i) {
            temp.num[i] = this->num[i] - label.num[i];
        }
        return temp;
    }

    matrix pow(const T B) const {
        matrix<T> temp(this->row, this->col);
        #pragma omp parallel for
        for (size_t i = 0; i < this->row * this->col; ++i)
            temp.num[i] = std::pow(this->num[i], B);
        return temp;
    }

    matrix l1_normalize() const {
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

    matrix l2_normalize() const {
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