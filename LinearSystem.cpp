#include "LinearSystem.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

// ─────────────────────── LinearSystem ───────────────────────────────────────

LinearSystem::LinearSystem(const Matrix& A, const Vector& b)
    : mSize(b.GetSize()),
      mpA(new Matrix(A)),
      mpb(new Vector(b))
{
    assert(A.GetNumRows() == A.GetNumCols());
    assert(A.GetNumRows() == b.GetSize());
}

LinearSystem::~LinearSystem() {
    delete mpA;
    delete mpb;
}

// Gaussian elimination with partial pivoting
Vector LinearSystem::Solve() {
    int n = mSize;

    // Working copies
    Matrix A(*mpA);
    Vector b(*mpb);

    for (int col = 0; col < n; ++col) {
        // Partial pivot: find row with largest absolute value in this column
        int pivotRow = col;
        for (int row = col + 1; row < n; ++row)
            if (std::abs(A(row + 1, col + 1)) > std::abs(A(pivotRow + 1, col + 1)))
                pivotRow = row;

        // Swap rows in A and b
        if (pivotRow != col) {
            for (int k = 1; k <= n; ++k)
                std::swap(A(col + 1, k), A(pivotRow + 1, k));
            std::swap(b[col], b[pivotRow]);
        }

        double pivot = A(col + 1, col + 1);
        assert(std::abs(pivot) > 1e-14); // singular matrix

        // Eliminate below
        for (int row = col + 1; row < n; ++row) {
            double factor = A(row + 1, col + 1) / pivot;
            for (int k = col; k < n; ++k)
                A(row + 1, k + 1) -= factor * A(col + 1, k + 1);
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    Vector x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j)
            sum -= A(i + 1, j + 1) * x[j];
        x[i] = sum / A(i + 1, i + 1);
    }
    return x;
}

// ─────────────────────── PosSymLinSystem ────────────────────────────────────

PosSymLinSystem::PosSymLinSystem(const Matrix& A, const Vector& b)
    : LinearSystem(A, b) {}

bool PosSymLinSystem::IsSymmetric(double tol) const {
    for (int i = 1; i <= mSize; ++i)
        for (int j = i + 1; j <= mSize; ++j)
            if (std::abs((*mpA)(i, j) - (*mpA)(j, i)) > tol)
                return false;
    return true;
}

// Conjugate Gradient method
Vector PosSymLinSystem::Solve() {
    if (!IsSymmetric())
        std::cerr << "Warning: matrix is not symmetric — CG may not converge.\n";

    int n = mSize;
    Vector x(n);          // initial guess: zero vector

    Vector r = *mpb - (*mpA) * x;   // r = b - Ax
    Vector p(r);                      // p = r
    double rsOld = r.Dot(r);

    const double tol = 1e-10;
    const int    maxIter = 10 * n;

    for (int iter = 0; iter < maxIter; ++iter) {
        if (std::sqrt(rsOld) < tol) break;

        Vector Ap = (*mpA) * p;
        double alpha = rsOld / p.Dot(Ap);

        x = x + alpha * p;
        r = r - alpha * Ap;

        double rsNew = r.Dot(r);
        double beta  = rsNew / rsOld;
        p = r + beta * p;
        rsOld = rsNew;
    }
    return x;
}
