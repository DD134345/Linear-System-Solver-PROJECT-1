#include "Matrix.h"
#include <cassert>
#include <cmath>
#include <algorithm>

void Matrix::Allocate() {
    mData = new double*[mNumRows];
    for (int i = 0; i < mNumRows; ++i)
        mData[i] = new double[mNumCols]();
}

void Matrix::Deallocate() {
    for (int i = 0; i < mNumRows; ++i) delete[] mData[i];
    delete[] mData;
}

Matrix::Matrix(int rows, int cols) : mNumRows(rows), mNumCols(cols) {
    assert(rows > 0 && cols > 0);
    Allocate();
}

Matrix::Matrix(const Matrix& other) : mNumRows(other.mNumRows), mNumCols(other.mNumCols) {
    Allocate();
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            mData[i][j] = other.mData[i][j];
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        Deallocate();
        mNumRows = other.mNumRows;
        mNumCols = other.mNumCols;
        Allocate();
        for (int i = 0; i < mNumRows; ++i)
            for (int j = 0; j < mNumCols; ++j)
                mData[i][j] = other.mData[i][j];
    }
    return *this;
}

Matrix::~Matrix() { Deallocate(); }

double& Matrix::operator()(int i, int j) {
    assert(i >= 1 && i <= mNumRows && j >= 1 && j <= mNumCols);
    return mData[i - 1][j - 1];
}

double Matrix::operator()(int i, int j) const {
    assert(i >= 1 && i <= mNumRows && j >= 1 && j <= mNumCols);
    return mData[i - 1][j - 1];
}

Matrix Matrix::operator+(const Matrix& m) const {
    assert(mNumRows == m.mNumRows && mNumCols == m.mNumCols);
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] + m.mData[i][j];
    return result;
}

Matrix Matrix::operator-(const Matrix& m) const {
    assert(mNumRows == m.mNumRows && mNumCols == m.mNumCols);
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] - m.mData[i][j];
    return result;
}

Matrix Matrix::operator*(const Matrix& m) const {
    assert(mNumCols == m.mNumRows);
    Matrix result(mNumRows, m.mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < m.mNumCols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < mNumCols; ++k)
                sum += mData[i][k] * m.mData[k][j];
            result.mData[i][j] = sum;
        }
    return result;
}

Matrix Matrix::operator*(double s) const {
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] * s;
    return result;
}

Vector Matrix::operator*(const Vector& v) const {
    assert(mNumCols == v.GetSize());
    Vector result(mNumRows);
    for (int i = 0; i < mNumRows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < mNumCols; ++j)
            sum += mData[i][j] * v[j];
        result[i] = sum;
    }
    return result;
}

Matrix operator*(double s, const Matrix& m) { return m * s; }

Matrix Matrix::Transpose() const {
    Matrix result(mNumCols, mNumRows);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[j][i] = mData[i][j];
    return result;
}

double Matrix::Determinant() const {
    assert(mNumRows == mNumCols);
    int n = mNumRows;
    Matrix A(*this);
    double det = 1.0;

    for (int col = 0; col < n; ++col) {
        int pivotRow = col;
        for (int row = col + 1; row < n; ++row)
            if (std::abs(A.mData[row][col]) > std::abs(A.mData[pivotRow][col]))
                pivotRow = row;

        if (pivotRow != col) {
            std::swap(A.mData[col], A.mData[pivotRow]);
            det *= -1.0;
        }

        if (std::abs(A.mData[col][col]) < 1e-14) return 0.0;
        det *= A.mData[col][col];

        for (int row = col + 1; row < n; ++row) {
            double factor = A.mData[row][col] / A.mData[col][col];
            for (int k = col; k < n; ++k)
                A.mData[row][k] -= factor * A.mData[col][k];
        }
    }
    return det;
}

Matrix Matrix::Inverse() const {
    assert(mNumRows == mNumCols);
    int n = mNumRows;
    Matrix aug(n, 2 * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            aug.mData[i][j] = mData[i][j];
        aug.mData[i][n + i] = 1.0;
    }

    for (int col = 0; col < n; ++col) {
        int pivotRow = col;
        for (int row = col + 1; row < n; ++row)
            if (std::abs(aug.mData[row][col]) > std::abs(aug.mData[pivotRow][col]))
                pivotRow = row;

        if (pivotRow != col) std::swap(aug.mData[col], aug.mData[pivotRow]);

        double pivot = aug.mData[col][col];
        assert(std::abs(pivot) > 1e-14);

        for (int k = 0; k < 2 * n; ++k) aug.mData[col][k] /= pivot;

        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double factor = aug.mData[row][col];
            for (int k = 0; k < 2 * n; ++k)
                aug.mData[row][k] -= factor * aug.mData[col][k];
        }
    }

    Matrix result(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result.mData[i][j] = aug.mData[i][n + j];
    return result;
}

Matrix Matrix::PseudoInverse() const {
    if (mNumRows >= mNumCols) {
        Matrix At = Transpose();
        Matrix AtA = At * (*this);
        return AtA.Inverse() * At;
    } else {
        Matrix At = Transpose();
        Matrix AAt = (*this) * At;
        return At * AAt.Inverse();
    }
}

Matrix Matrix::TikhonovInverse(double lambda) const {
    Matrix At = Transpose();
    Matrix AtA = At * (*this);
    int n = AtA.GetNumRows();
    Matrix I(n, n);
    for (int i = 1; i <= n; ++i) I(i, i) = 1.0;
    Matrix regularized = AtA + I * (lambda * lambda);
    return regularized.Inverse() * At;
}
