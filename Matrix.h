#pragma once
#include "Vector.h"

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
    Matrix& operator=(const Matrix& other);
    ~Matrix();

    int GetNumRows() const { return mNumRows; }
    int GetNumCols() const { return mNumCols; }

    double& operator()(int i, int j);
    double  operator()(int i, int j) const;

    Matrix operator+(const Matrix& m) const;
    Matrix operator-(const Matrix& m) const;
    Matrix operator*(const Matrix& m) const;
    Matrix operator*(double s) const;
    Vector operator*(const Vector& v) const;

    Matrix Transpose() const;
    double Determinant() const;
    Matrix Inverse() const;
    Matrix PseudoInverse() const;
    Matrix TikhonovInverse(double lambda) const;
};

Matrix operator*(double s, const Matrix& m);
