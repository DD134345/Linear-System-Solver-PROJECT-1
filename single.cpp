#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>

using namespace std;

// === Vector ===
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

// Vector implementation
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

// === Matrix ===
class Matrix {
private:
    int      mNumRows;
    int      mNumCols;
    double** mData;

    void Allocate();
    void Deallocate();

public:
    Matrix(int rows, int cols);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    ~Matrix();

    int GetNumRows() const { return mNumRows; }
    int GetNumCols() const { return mNumCols; }

    double& operator()(int i, int j);
    double  operator()(int i, int j) const;

    Matrix operator+(const Matrix& m) const;
    Matrix operator-(const Matrix& m) const;
    Matrix operator*(const Matrix& m) const;
    Matrix operator*(double s) const;
    Vector operator*(const Vector& v) const;

    Matrix Transpose() const;
    double Determinant() const;
    Matrix Inverse() const;
    Matrix PseudoInverse() const;
    Matrix TikhonovInverse(double lambda) const;
};

Matrix operator*(double s, const Matrix& m);

// Matrix implementation
void Matrix::Allocate() {
    mData = new double*[mNumRows];
    for (int i = 0; i < mNumRows; ++i)
        mData[i] = new double[mNumCols]();
}

void Matrix::Deallocate() {
    for (int i = 0; i < mNumRows; ++i) delete[] mData[i];
    delete[] mData;
}

Matrix::Matrix(int rows, int cols) : mNumRows(rows), mNumCols(cols) {
    assert(rows > 0 && cols > 0);
    Allocate();
}

Matrix::Matrix(const Matrix& other) : mNumRows(other.mNumRows), mNumCols(other.mNumCols) {
    Allocate();
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            mData[i][j] = other.mData[i][j];
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        Deallocate();
        mNumRows = other.mNumRows;
        mNumCols = other.mNumCols;
        Allocate();
        for (int i = 0; i < mNumRows; ++i)
            for (int j = 0; j < mNumCols; ++j)
                mData[i][j] = other.mData[i][j];
    }
    return *this;
}

Matrix::~Matrix() { Deallocate(); }

double& Matrix::operator()(int i, int j) {
    assert(i >= 1 && i <= mNumRows && j >= 1 && j <= mNumCols);
    return mData[i - 1][j - 1];
}

double Matrix::operator()(int i, int j) const {
    assert(i >= 1 && i <= mNumRows && j >= 1 && j <= mNumCols);
    return mData[i - 1][j - 1];
}

Matrix Matrix::operator+(const Matrix& m) const {
    assert(mNumRows == m.mNumRows && mNumCols == m.mNumCols);
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] + m.mData[i][j];
    return result;
}

Matrix Matrix::operator-(const Matrix& m) const {
    assert(mNumRows == m.mNumRows && mNumCols == m.mNumCols);
    Matrix result(mNumRows, mNumCols);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[i][j] = mData[i][j] - m.mData[i][j];
    return result;
}

Matrix Matrix::operator*(const Matrix& m) const {
    assert(mNumCols == m.mNumRows);
    Matrix result(mNumRows, mNumCols);
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
    assert(mNumCols == v.GetSize());
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

Matrix Matrix::Transpose() const {
    Matrix result(mNumCols, mNumRows);
    for (int i = 0; i < mNumRows; ++i)
        for (int j = 0; j < mNumCols; ++j)
            result.mData[j][i] = mData[i][j];
    return result;
}

double Matrix::Determinant() const {
    assert(mNumRows == mNumCols);
    int n = mNumRows;
    Matrix A(*this);
    double det = 1.0;

    for (int col = 0; col < n; ++col) {
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

Matrix Matrix::Inverse() const {
    assert(mNumRows == mNumCols);
    int n = mNumRows;
    Matrix aug(n, 2 * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            aug.mData[i][j] = mData[i][j];
        aug.mData[i][n + i] = 1.0;
    }

    for (int col = 0; col < n; ++col) {
        int pivotRow = col;
        for (int row = col + 1; row < n; ++row)
            if (std::abs(aug.mData[row][col]) > std::abs(aug.mData[pivotRow][col]))
                pivotRow = row;

        if (pivotRow != col) std::swap(aug.mData[col], aug.mData[pivotRow]);

        double pivot = aug.mData[col][col];
        assert(std::abs(pivot) > 1e-14);

        for (int k = 0; k < 2 * n; ++k) aug.mData[col][k] /= pivot;

        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double factor = aug.mData[row][col];
            for (int k = 0; k < 2 * n; ++k)
                aug.mData[row][k] -= factor * aug.mData[col][k];
        }
    }

    Matrix result(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result.mData[i][j] = aug.mData[i][n + j];
    return result;
}

Matrix Matrix::PseudoInverse() const {
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

Matrix Matrix::TikhonovInverse(double lambda) const {
    Matrix At = Transpose();
    Matrix AtA = At * (*this);
    int n = AtA.GetNumRows();
    Matrix I(n, n);
    for (int i = 1; i <= n; ++i) I(i, i) = 1.0;
    Matrix regularized = AtA + I * (lambda * lambda);
    return regularized.Inverse() * At;
}

// === LinearSystem ===
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

// LinearSystem implementation
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

// === Application code (main) ===

void PrintVector(const Vector& v, const string& name = "") {
    if (!name.empty()) cout << name << " = [";
    else               cout << "[";
    for (int i = 1; i <= v.GetSize(); ++i) {
        cout << fixed << setprecision(4) << v(i);
        if (i < v.GetSize()) cout << ", ";
    }
    cout << "]\n";
}

void PrintMatrix(const Matrix& M, const string& name = "") {
    if (!name.empty()) cout << name << ":\n";
    for (int i = 1; i <= M.GetNumRows(); ++i) {
        cout << "  [ ";
        for (int j = 1; j <= M.GetNumCols(); ++j)
            cout << setw(10) << fixed << setprecision(4) << M(i, j);
        cout << " ]\n";
    }
}

struct Record {
    double myct, mmin, mmax, cach, chmin, chmax;
    double prp;
};

vector<Record> LoadDataset(const string& filename) {
    vector<Record> records;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "ERROR: Cannot open " << filename << "\n";
        return records;
    }
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream ss(line);
        string tok;
        vector<string> tokens;
        while (getline(ss, tok, ',')) tokens.push_back(tok);
        if (tokens.size() < 10) continue;

        Record r;
        r.myct  = stod(tokens[2]);
        r.mmin  = stod(tokens[3]);
        r.mmax  = stod(tokens[4]);
        r.cach  = stod(tokens[5]);
        r.chmin = stod(tokens[6]);
        r.chmax = stod(tokens[7]);
        r.prp   = stod(tokens[8]);
        records.push_back(r);
    }
    return records;
}

double ComputeRMSE(const Vector& params, const vector<Record>& data) {
    double sumSq = 0.0;
    for (const auto& r : data) {
        double pred = params(1) * r.myct + params(2) * r.mmin
                    + params(3) * r.mmax + params(4) * r.cach
                    + params(5) * r.chmin + params(6) * r.chmax;
        double err = pred - r.prp;
        sumSq += err * err;
    }
    return sqrt(sumSq / data.size());
}

void RunRegression(const string& dataPath) {
    cout << "\n=== Part B: CPU Performance Linear Regression ===\n\n";

    auto all = LoadDataset(dataPath);
    if (all.empty()) return;

    cout << "Loaded " << all.size() << " records." << "\n";

    mt19937 rng(42);
    shuffle(all.begin(), all.end(), rng);

    int nTrain = static_cast<int>(all.size() * 0.8);
    int nTest  = static_cast<int>(all.size()) - nTrain;

    vector<Record> trainSet(all.begin(), all.begin() + nTrain);
    vector<Record> testSet (all.begin() + nTrain, all.end());

    cout << "Training: " << nTrain << "  |  Testing: " << nTest << "\n\n";

    Matrix A(nTrain, 6);
    Vector b(nTrain);
    for (int i = 0; i < nTrain; ++i) {
        const auto& r = trainSet[i];
        A(i+1, 1) = r.myct;
        A(i+1, 2) = r.mmin;
        A(i+1, 3) = r.mmax;
        A(i+1, 4) = r.cach;
        A(i+1, 5) = r.chmin;
        A(i+1, 6) = r.chmax;
        b(i+1)    = r.prp;
    }

    cout << "--- Method 1: Pseudo-inverse (A^+ b) ---\n";
    Matrix Aplus   = A.PseudoInverse();
    Vector params1 = Aplus * b;
    PrintVector(params1, "params");
    cout << "Train RMSE: " << fixed << setprecision(4) << ComputeRMSE(params1, trainSet) << "\n";
    cout << "Test  RMSE: " << ComputeRMSE(params1, testSet) << "\n\n";

    cout << "--- Method 2: Normal equations (Gaussian elimination) ---\n";
    Matrix AtA = A.Transpose() * A;
    Vector Atb = A.Transpose() * b;
    LinearSystem normalEq(AtA, Atb);
    Vector params2 = normalEq.Solve();
    PrintVector(params2, "params");
    cout << "Train RMSE: " << ComputeRMSE(params2, trainSet) << "\n";
    cout << "Test  RMSE: " << ComputeRMSE(params2, testSet) << "\n\n";

    cout << "--- Method 3: Normal equations (Conjugate Gradient) ---\n";
    PosSymLinSystem cgEq(AtA, Atb);
    Vector params3 = cgEq.Solve();
    PrintVector(params3, "params");
    cout << "Train RMSE: " << ComputeRMSE(params3, trainSet) << "\n";
    cout << "Test  RMSE: " << ComputeRMSE(params3, testSet) << "\n\n";

    cout << "--- Test set: Predicted vs Actual (first 10) ---\n";
    cout << setw(12) << "Predicted" << setw(10) << "Actual" << setw(12) << "Error\n";
    cout << string(34, '-') << "\n";
    for (int i = 0; i < min(10, nTest); ++i) {
        const auto& r = testSet[i];
        double pred = params2(1)*r.myct + params2(2)*r.mmin
                    + params2(3)*r.mmax + params2(4)*r.cach
                    + params2(5)*r.chmin + params2(6)*r.chmax;
        cout << setw(12) << fixed << setprecision(1) << pred
             << setw(10) << r.prp
             << setw(12) << (pred - r.prp) << "\n";
    }

    cout << "\nFeature coefficients:\n";
    cout << "  x1 (MYCT)  = " << params2(1) << "\n";
    cout << "  x2 (MMIN)  = " << params2(2) << "\n";
    cout << "  x3 (MMAX)  = " << params2(3) << "\n";
    cout << "  x4 (CACH)  = " << params2(4) << "\n";
    cout << "  x5 (CHMIN) = " << params2(5) << "\n";
    cout << "  x6 (CHMAX) = " << params2(6) << "\n";
}

void TestPartA() {
    cout << "=== Part A: Linear System Solver Tests ===\n\n";

    cout << "--- Vector Operations ---\n";
    Vector a(3), b(3);
    a(1) = 1; a(2) = 2; a(3) = 3;
    b(1) = 4; b(2) = 5; b(3) = 6;
    PrintVector(a, "a");
    PrintVector(b, "b");
    PrintVector(a + b, "a + b");
    PrintVector(a - b, "a - b");
    PrintVector(a * 2.0, "a * 2");
    cout << "a . b = " << a.Dot(b) << "\n";
    cout << "||a|| = " << a.Norm() << "\n\n";

    cout << "--- Matrix Operations ---\n";
    Matrix A(3, 3);
    A(1,1)=2; A(1,2)=1; A(1,3)=0;
    A(2,1)=1; A(2,2)=3; A(2,3)=1;
    A(3,1)=0; A(3,2)=1; A(3,3)=2;
    PrintMatrix(A, "A");
    cout << "det(A) = " << A.Determinant() << "\n";
    Matrix Ainv = A.Inverse();
    PrintMatrix(Ainv, "A inverse");
    Matrix I = A * Ainv;
    PrintMatrix(I, "A * A inverse\n");

    cout << "--- LinearSystem: Gaussian Elimination ---\n";
    Matrix M(3, 3);
    Vector rhs(3);
    M(1,1)=2; M(1,2)=1; M(1,3)=0;
    M(2,1)=1; M(2,2)=3; M(2,3)=1;
    M(3,1)=0; M(3,2)=1; M(3,3)=2;
    rhs(1)=5; rhs(2)=10; rhs(3)=7;
    LinearSystem ls(M, rhs);
    Vector x = ls.Solve();
    PrintVector(x, "x (Gaussian)\n");

    cout << "--- PosSymLinSystem: Conjugate Gradient ---\n";
    Matrix S(3, 3);
    Vector rhs2(3);
    S(1,1)=4; S(1,2)=1; S(1,3)=0;
    S(2,1)=1; S(2,2)=3; S(2,3)=1;
    S(3,1)=0; S(3,2)=1; S(3,3)=4;
    rhs2(1)=6; rhs2(2)=5; rhs2(3)=5;
    PosSymLinSystem cg(S, rhs2);
    Vector xcg = cg.Solve();
    PrintVector(xcg, "x (CG)");
    Vector residual = rhs2 - S * xcg;
    cout << "Residual ||b - Sx|| = " << residual.Norm() << "\n\n";

    cout << "--- Pseudo-Inverse: Overdetermined System ---\n";
    Matrix O(4, 2);
    Vector rhs3(4);
    O(1,1)=1; O(1,2)=1;
    O(2,1)=1; O(2,2)=2;
    O(3,1)=1; O(3,2)=3;
    O(4,1)=1; O(4,2)=4;
    rhs3(1)=3; rhs3(2)=5; rhs3(3)=7; rhs3(4)=9;
    Matrix Oplus = O.PseudoInverse();
    Vector xpinv = Oplus * rhs3;
    PrintVector(xpinv, "x (pseudo-inverse)");
    cout << "Expected: [1, 2]\n\n";
}

int main(int argc, char* argv[]) {
    TestPartA();

    string dataPath = (argc > 1) ? argv[1] : "machine.data";
    RunRegression(dataPath);

    cout << "\nDone.\n";
    return 0;
}
