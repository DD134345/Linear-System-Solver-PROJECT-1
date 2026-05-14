#pragma once
#include "Matrix.h"
#include "Vector.h"

// ─── Base class: Gaussian elimination with partial pivoting ─────────────────

class LinearSystem {
protected:
    int     mSize;
    Matrix* mpA;
    Vector* mpb;
    double  mTolerance;  // configurable tolerance for singularity check

private:
    LinearSystem();                          // disabled
    LinearSystem(const LinearSystem&);       // disabled
    LinearSystem& operator=(const LinearSystem&); // disabled

public:
    LinearSystem(const Matrix& A, const Vector& b, double tolerance = 1e-12);
    virtual ~LinearSystem();

    virtual Vector Solve();   // Gaussian elimination + partial pivot
    void SetTolerance(double tol) { mTolerance = tol; }
    double GetTolerance() const { return mTolerance; }
};

// ─── Derived: Conjugate Gradient for symmetric positive definite systems ─────

class PosSymLinSystem : public LinearSystem {
public:
    PosSymLinSystem(const Matrix& A, const Vector& b, double tolerance = 1e-10);
    Vector Solve() override;   // conjugate gradient method

private:
    bool IsSymmetric(double tol) const;
};
