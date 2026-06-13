#pragma once
#include "Matrix.h"
#include "Vector.h"

class LinearSystem {
protected:
    int     mSize;
    Matrix* mpA;
    Vector* mpb;

public:
    LinearSystem(const Matrix& A, const Vector& b);
    virtual ~LinearSystem();

    virtual Vector Solve();
};

class PosSymLinSystem : public LinearSystem {
public:
    PosSymLinSystem(const Matrix& A, const Vector& b);
    Vector Solve() override;

private:
    bool IsSymmetric() const;
};
