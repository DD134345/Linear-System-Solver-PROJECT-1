#include "LinearSystem.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>

LinearSystem::LinearSystem(const Matrix& A, const Vector& b)
    : mSize(b.GetSize()), mpA(new Matrix(A)), mpb(new Vector(b))
{
    assert(A.GetNumRows() == A.GetNumCols());
    assert(A.GetNumRows() == b.GetSize());
}

LinearSystem::~LinearSystem() {
    delete mpA;
    delete mpb;
}

Vector LinearSystem::Solve() {
    int n = mSize;
    Matrix A(*mpA);
    Vector b(*mpb);

    for (int col = 0; col < n; ++col) {
        int pivotRow = col;
        for (int row = col + 1; row < n; ++row)
            if (std::abs(A(row + 1, col + 1)) > std::abs(A(pivotRow + 1, col + 1)))
                pivotRow = row;

        if (pivotRow != col) {
            for (int k = 1; k <= n; ++k)
                std::swap(A(col + 1, k), A(pivotRow + 1, k));
            std::swap(b[col], b[pivotRow]);
        }

        double pivot = A(col + 1, col + 1);
        assert(std::abs(pivot) > 1e-12);

        for (int row = col + 1; row < n; ++row) {
            double factor = A(row + 1, col + 1) / pivot;
            for (int k = col; k < n; ++k)
                A(row + 1, k + 1) -= factor * A(col + 1, k + 1);
            b[row] -= factor * b[col];
        }
    }

    Vector x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j)
            sum -= A(i + 1, j + 1) * x[j];
        x[i] = sum / A(i + 1, i + 1);
    }
    return x;
}

PosSymLinSystem::PosSymLinSystem(const Matrix& A, const Vector& b)
    : LinearSystem(A, b) {}

bool PosSymLinSystem::IsSymmetric() const {
    int n = mSize;
    for (int i = 1; i <= n; ++i)
        for (int j = i + 1; j <= n; ++j)
            if (std::abs((*mpA)(i, j) - (*mpA)(j, i)) > 1e-12)
                return false;
    return true;
}

Vector PosSymLinSystem::Solve() {
    if (!IsSymmetric())
        std::cerr << "Warning: matrix is not symmetric.\n";

    int n = mSize;
    Vector x(n);
    Vector r = *mpb - (*mpA) * x;
    Vector p(r);
    double rsOld = r.Dot(r);

    const int maxIter = 10 * n;

    for (int iter = 0; iter < maxIter; ++iter) {
        if (std::sqrt(rsOld) < 1e-10) break;

        Vector Ap = (*mpA) * p;
        double alpha = rsOld / p.Dot(Ap);

        x = x + alpha * p;
        r = r - alpha * Ap;

        double rsNew = r.Dot(r);
        double beta = rsNew / rsOld;
        p = r + beta * p;
        rsOld = rsNew;
    }
    return x;
}
