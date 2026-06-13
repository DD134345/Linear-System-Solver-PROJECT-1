#pragma once
#include <cassert>

class Vector {
private:
    int     mSize;
    double* mData;

public:
    Vector(int size);
    Vector(const Vector& other);
    Vector& operator=(const Vector& other);
    ~Vector();

    int GetSize() const { return mSize; }

    double& operator[](int i);
    double  operator[](int i) const;

    double& operator()(int i);
    double  operator()(int i) const;

    Vector operator+(const Vector& v) const;
    Vector operator-(const Vector& v) const;
    Vector operator-() const;
    Vector operator*(double s) const;

    double Dot(const Vector& v) const;
    double Norm() const;
};

Vector operator*(double s, const Vector& v);
