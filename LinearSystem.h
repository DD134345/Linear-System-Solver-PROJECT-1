#pragma once
#include "Matrix.h"
#include "Vector.h"

// ─── Base class: Gaussian elimination with partial pivoting ─────────────────

class LinearSystem {
protected:
    int     mSize;
    Matrix* mpA;
    Vector* mpb;

private:
    LinearSystem();                          // disabled
    LinearSystem(const LinearSystem&);       // disabled
    LinearSystem& operator=(const LinearSystem&); // disabled

public:
    LinearSystem(const Matrix& A, const Vector& b);
    virtual ~LinearSystem();

    virtual Vector Solve();   // Gaussian elimination + partial pivot
};

// ─── Derived: Conjugate Gradient for symmetric positive definite systems ─────

class PosSymLinSystem : public LinearSystem {
public:
    PosSymLinSystem(const Matrix& A, const Vector& b);
    Vector Solve() override;   // conjugate gradient method

private:
    bool IsSymmetric(double tol = 1e-10) const;
};
