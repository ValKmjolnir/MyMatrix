/* By ValKmjolnir 2020/5/3                        */
/* Rewrite by ValKmjolnir 2022/11/16              */
/* Updated by ValKmjolnir 2025/01/19              */

#pragma once

#include <omp.h>

#include <iostream>
#include <cmath>
#include <sstream>
#include <cstring>
#include <cstdint>
#include <cstdlib>

template<typename T>
class Matrix {
private:
    size_t row;
    size_t col;
    T* num;

public:
    Matrix(const size_t, const size_t);
    Matrix(const Matrix<T>&);
    ~Matrix();

public:
    Matrix  operator+ (const Matrix<T>&);
    Matrix  operator- (const Matrix<T>&);
    Matrix  operator* (const Matrix<T>&);
    Matrix& operator= (const Matrix<T>&);
    T*      operator[](const size_t);
    Matrix  Hadamard  (const Matrix<T>&);
    Matrix  Transpose ();

public:
    void random_init();

public:
    template<typename _T>
    friend std::ostream& operator<<(std::ostream&, const Matrix<_T>&);
    
    template<typename _T>
    friend std::istream& operator>>(std::istream&, const Matrix<_T>&);
};

template<typename T>
Matrix<T>::Matrix(const size_t __row, const size_t __col) {
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

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& Temp) {
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

template<typename T>
Matrix<T>::~Matrix() {
    if (num) {
        delete[] num;
    }
    return;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& B) {
    if (this->row == B.row && this->col==B.col) {
        auto Temp = *this;
        #pragma omp parallel for
        for (size_t i = 0; i < row * col; ++i)
            Temp.num[i] += B.num[i];
        return Temp;
    } else {
        throw "No matching matrix";
    }
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& B) {
    if (this->row == B.row && this->col == B.col) {
        auto Temp = *this;
        #pragma omp parallel for
        for (size_t i = 0; i < row * col; ++i)
            Temp.num[i] -= B.num[i];
        return Temp;
    } else {
        throw "No matching matrix";
    }
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& B) {
    if (!this->row || !this->col || !B.row || !B.col) {
        throw "No matching matrix";
    } else if (this->col != B.row) {
        throw "No matching matrix";
    }

    Matrix<T> Temp(this->row, B.col);
    #pragma omp parallel for
    for (size_t i = 0; i < Temp.row; ++i)
        for (size_t j = 0; j < Temp.col; ++j) {
            T trans = 0;
            for (size_t k = 0; k < this->col; ++k)
                trans += this->num[i * this->col + k] * B.num[k * B.col + j];
            Temp.num[i * Temp.col + j] = trans;
        }
    return Temp;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& B) {
    if (num) {
        delete[] num;
    }

    row = B.row;
    col = B.col;
    if (row > 0 && col > 0) {
        num = new T [row];
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

template<typename T>
T* Matrix<T>::operator[](const size_t addr) {
    return addr >= this->row ? nullptr : this->num[addr];
}

template<typename T>
Matrix<T> Matrix<T>::Hadamard(const Matrix<T>& B) {
    if (!this->row || !this->col || !B.row || !B.col) {
        throw "No matching matrix";
    } else if (this->row != B.row || this->col != B.col) {
        throw "No matching matrix";
    }

    Matrix<T> temp(this->row, this->col);
    #pragma omp parallel for
    for (size_t i = 0; i < this->row * this->col; ++i)
        temp.num[i] = this->num[i] * B.num[i];
    return temp;
}

template<typename T>
Matrix<T> Matrix<T>::Transpose() {
    Matrix<T> temp(this->col, this->row);
    #pragma omp parallel for
    for (size_t i = 0; i < this->row; ++i)
        for (size_t j = 0; j < this->col; ++j)
            temp.num[j * this->row + i] = this->num[i * this->col + j];
    return temp;
}

template<typename T>
void Matrix<T>::random_init() {
    for (size_t i = 0; i < row * col; ++i)
        num[i] = static_cast<T>(rand() % 10);
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& m) {
    for (size_t i = 0; i < m.row; ++i)
        for (size_t j = 0; j < m.col; ++j)
            out << m.num[i * m.col + j] << ((char)(j == m.col - 1)? '\n' : ' ');
    return out;
}

template<typename T>
std::istream& operator>>(std::istream& in, const Matrix<T>& m) {
    for (size_t i = 0; i < m.row; ++i)
        for (size_t j = 0; j < m.col; ++j)
            in >> m.num[i * m.col + j];
    return in;
}
