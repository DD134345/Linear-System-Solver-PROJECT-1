#include "Matrix.h"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ──────────────────────────── memory helpers ────────────────────────────────

void Matrix::Allocate() {
    mData = new double*[mNumRows];
    for (int i = 0; i < mNumRows; ++i)
        mData[i] = new double[mNumCols]();
}

void Matrix::Deallocate() {
    if (mData) {
        for (int i = 0; i < mNumRows; ++i) delete[] mData[i];
        delete[] mData;
        mData = nullptr;
    }
}

// ──────────────────────────── constructors ──────────────────────────────────

Matrix::Matrix(int rows, int cols) : mNumRows(rows), mNumCols(cols), mData(nullptr) {
    if (rows <= 0 || cols <= 0) throw std::invalid_argument("Matrix dimensions must be positive");
    Allocate();
}

Matrix::Matrix(const Matrix& other) : mNumRows(other.mNumRows), mNumCols(other.mNumCols), mData(nullptr) {
    Allocate();
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            mData[i][j] = other.mData[i][j];
}

Matrix::Matrix(Matrix&& other) noexcept : mNumRows(other.mNumRows), mNumCols(other.mNumCols), mData(other.mData) {
    other.mNumRows = 0;
    other.mNumCols = 0;
    other.mData = nullptr;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;
    if (mNumRows != other.mNumRows || mNumCols != other.mNumCols) {
        Deallocate();
        mNumRows = other.mNumRows;
        mNumCols = other.mNumCols;
        Allocate();
    }
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            mData[i][j] = other.mData[i][j];
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this == &other) return *this;
    Deallocate();
    mNumRows = other.mNumRows;
    mNumCols = other.mNumCols;
    mData = other.mData;
    other.mNumRows = 0;
    other.mNumCols = 0;
    other.mData = nullptr;
    return *this;
}

Matrix::~Matrix() { Deallocate(); }

// ──────────────────────────── element access ────────────────────────────────

double& Matrix::operator()(int i, int j) {
    if (i < 1 || i > mNumRows || j < 1 || j > mNumCols)
        throw std::out_of_range("Matrix index out of range");
    return mData[i - 1][j - 1];
}

double Matrix::operator()(int i, int j) const {
    if (i < 1 || i > mNumRows || j < 1 || j > mNumCols)
        throw std::out_of_range("Matrix index out of range");
    return mData[i - 1][j - 1];
}

// ──────────────────────────── operators ─────────────────────────────────────

Matrix Matrix::operator+(const Matrix& m) const {
    if (mNumRows != m.mNumRows || mNumCols != m.mNumCols)
        throw std::invalid_argument("Matrix dimension mismatch");
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] + m.mData[i][j];
    return result;
}

Matrix Matrix::operator-(const Matrix& m) const {
    if (mNumRows != m.mNumRows || mNumCols != m.mNumCols)
        throw std::invalid_argument("Matrix dimension mismatch");
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] - m.mData[i][j];
    return result;
}

Matrix Matrix::operator*(const Matrix& m) const {
    if (mNumCols != m.mNumRows)
        throw std::invalid_argument("Matrix multiplication dimension mismatch");
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
    if (mNumCols != v.GetSize())
        throw std::invalid_argument("Matrix-Vector multiplication dimension mismatch");
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

// ──────────────────────────── transpose ─────────────────────────────────────

Matrix Matrix::Transpose() const {
    Matrix result(mNumCols, mNumRows);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[j][i] = mData[i][j];
    return result;
}

// ──────────────────────────── determinant (LU) ──────────────────────────────

double Matrix::Determinant() const {
    if (mNumRows != mNumCols)
        throw std::invalid_argument("Determinant requires square matrix");
    int n = mNumRows;

    // Working copy
    Matrix A(*this);
    double det = 1.0;

    for (int col = 0; col < n; ++col) {
        // Partial pivot
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

// ──────────────────────────── inverse (Gauss-Jordan) ────────────────────────

Matrix Matrix::Inverse() const {
    if (mNumRows != mNumCols)
        throw std::invalid_argument("Inverse requires square matrix");
    int n = mNumRows;

    // Augmented [A | I]
    Matrix aug(n, 2 * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            aug.mData[i][j] = mData[i][j];
        aug.mData[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        int pivotRow = col;
        for (int row = col + 1; row < n; ++row)
            if (std::abs(aug.mData[row][col]) > std::abs(aug.mData[pivotRow][col]))
                pivotRow = row;

        if (pivotRow != col) std::swap(aug.mData[col], aug.mData[pivotRow]);

        double pivot = aug.mData[col][col];
        if (std::abs(pivot) < 1e-14)
            throw std::runtime_error("Matrix is singular and cannot be inverted");

        for (int k = 0; k < 2 * n; ++k) aug.mData[col][k] /= pivot;

        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double factor = aug.mData[row][col];
            for (int k = 0; k < 2 * n; ++k)
                aug.mData[row][k] -= factor * aug.mData[col][k];
        }
    }

    // Extract right half
    Matrix result(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result.mData[i][j] = aug.mData[i][n + j];
    return result;
}

// ──────────────────────────── pseudo-inverse ────────────────────────────────

Matrix Matrix::PseudoInverse() const {
    // For overdetermined (rows >= cols): A^+ = (A^T A)^{-1} A^T
    // For underdetermined (rows < cols): A^+ = A^T (A A^T)^{-1}
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
