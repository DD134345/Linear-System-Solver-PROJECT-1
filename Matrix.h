#pragma once
#include "Vector.h"
#include <cassert>
#include <utility>

class Matrix {
private:
    int      mNumRows;
    int      mNumCols;
    double** mData;

    void Allocate();
    void Deallocate();

public:
    Matrix(int rows, int cols);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix();

    int GetNumRows() const { return mNumRows; }
    int GetNumCols() const { return mNumCols; }

    // 1-based access
    double& operator()(int i, int j);
    double  operator()(int i, int j) const;

    Matrix& operator=(const Matrix& m);
    Matrix  operator+(const Matrix& m) const;
    Matrix  operator-(const Matrix& m) const;
    Matrix  operator*(const Matrix& m) const;
    Matrix  operator*(double s)        const;
    Vector  operator*(const Vector& v) const;

    Matrix    Transpose()    const;
    double    Determinant()  const;
    Matrix    Inverse()      const;
    Matrix    PseudoInverse() const;  // Moore-Penrose: (A^T A)^{-1} A^T
};

Matrix operator*(double s, const Matrix& m);
