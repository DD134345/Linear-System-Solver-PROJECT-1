#pragma once
#include <cassert>
#include <cmath>
#include <utility>

class Vector {
private:
    int     mSize;
    double* mData;

public:
    explicit Vector(int size);
    Vector(const Vector& other);
    Vector(Vector&& other) noexcept;
    Vector& operator=(const Vector& other);
    Vector& operator=(Vector&& other) noexcept;
    ~Vector();

    int GetSize() const { return mSize; }

    // 0-based, bounds-checked via assert
    double& operator[](int i);
    double  operator[](int i) const;

    // 1-based access (project spec)
    double& operator()(int i);
    double  operator()(int i) const;

    Vector& operator=(const Vector& v);
    Vector  operator+(const Vector& v) const;
    Vector  operator-(const Vector& v) const;
    Vector  operator-()                const;
    Vector  operator*(double s)        const;

    double Dot(const Vector& v) const;
    double Norm()               const;
};

Vector operator*(double s, const Vector& v);
