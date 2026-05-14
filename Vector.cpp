#include "Vector.h"
#include <cassert>
#include <cmath>
#include <stdexcept>

Vector::Vector(int size) : mSize(size), mData(new double[size]()) {
    if (size < 0) throw std::invalid_argument("Vector size must be non-negative");
}

Vector::Vector(const Vector& other) : mSize(other.mSize), mData(new double[other.mSize]) {
    for (int i = 0; i < mSize; ++i) mData[i] = other.mData[i];
}

Vector::Vector(Vector&& other) noexcept : mSize(other.mSize), mData(other.mData) {
    other.mSize = 0;
    other.mData = nullptr;
}

Vector& Vector::operator=(const Vector& other) {
    if (this == &other) return *this;
    if (mSize != other.mSize) throw std::invalid_argument("Vector size mismatch in assignment");
    for (int i = 0; i < mSize; ++i) mData[i] = other.mData[i];
    return *this;
}

Vector& Vector::operator=(Vector&& other) noexcept {
    if (this == &other) return *this;
    delete[] mData;
    mSize = other.mSize;
    mData = other.mData;
    other.mSize = 0;
    other.mData = nullptr;
    return *this;
}

Vector::~Vector() { delete[] mData; }

double& Vector::operator[](int i) {
    if (i < 0 || i >= mSize) throw std::out_of_range("Vector index out of range");
    return mData[i];
}
double Vector::operator[](int i) const {
    if (i < 0 || i >= mSize) throw std::out_of_range("Vector index out of range");
    return mData[i];
}

// 1-based
double& Vector::operator()(int i) {
    if (i < 1 || i > mSize) throw std::out_of_range("Vector index out of range");
    return mData[i - 1];
}
double Vector::operator()(int i) const {
    if (i < 1 || i > mSize) throw std::out_of_range("Vector index out of range");
    return mData[i - 1];
}

Vector Vector::operator+(const Vector& v) const {
    if (mSize != v.mSize) throw std::invalid_argument("Vector size mismatch");
    Vector result(mSize);
    for (int i = 0; i < mSize; ++i) result.mData[i] = mData[i] + v.mData[i];
    return result;
}

Vector Vector::operator-(const Vector& v) const {
    if (mSize != v.mSize) throw std::invalid_argument("Vector size mismatch");
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
    if (mSize != v.mSize) throw std::invalid_argument("Vector size mismatch");
    double sum = 0.0;
    for (int i = 0; i < mSize; ++i) sum += mData[i] * v.mData[i];
    return sum;
}

double Vector::Norm() const {
    return std::sqrt(Dot(*this));
}

Vector operator*(double s, const Vector& v) { return v * s; }
