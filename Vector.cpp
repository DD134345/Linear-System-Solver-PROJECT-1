#include "Vector.h"
#include <cmath>

Vector::Vector(int size) : mSize(size), mData(new double[size]()) {}

Vector::Vector(const Vector& other) : mSize(other.mSize), mData(new double[other.mSize]) {
    for (int i = 0; i < mSize; ++i) mData[i] = other.mData[i];
}

Vector& Vector::operator=(const Vector& other) {
    if (this != &other) {
        delete[] mData;
        mSize = other.mSize;
        mData = new double[mSize];
        for (int i = 0; i < mSize; ++i) mData[i] = other.mData[i];
    }
    return *this;
}

Vector::~Vector() { delete[] mData; }

double& Vector::operator[](int i) {
    assert(i >= 0 && i < mSize);
    return mData[i];
}

double Vector::operator[](int i) const {
    assert(i >= 0 && i < mSize);
    return mData[i];
}

double& Vector::operator()(int i) {
    assert(i >= 1 && i <= mSize);
    return mData[i - 1];
}

double Vector::operator()(int i) const {
    assert(i >= 1 && i <= mSize);
    return mData[i - 1];
}

Vector Vector::operator+(const Vector& v) const {
    assert(mSize == v.mSize);
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i) result.mData[i] = mData[i] + v.mData[i];
    return result;
}

Vector Vector::operator-(const Vector& v) const {
    assert(mSize == v.mSize);
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i) result.mData[i] = mData[i] - v.mData[i];
    return result;
}

Vector Vector::operator-() const {
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i) result.mData[i] = -mData[i];
    return result;
}

Vector Vector::operator*(double s) const {
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i) result.mData[i] = mData[i] * s;
    return result;
}

double Vector::Dot(const Vector& v) const {
    assert(mSize == v.mSize);
    double sum = 0.0;
    for (int i = 0; i < mSize; ++i) sum += mData[i] * v.mData[i];
    return sum;
}

double Vector::Norm() const { return std::sqrt(Dot(*this)); }

Vector operator*(double s, const Vector& v) { return v * s; }
