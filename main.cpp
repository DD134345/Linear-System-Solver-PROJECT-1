#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include "Vector.h"
#include "Matrix.h"
#include "LinearSystem.h"

using namespace std;

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

    cout << "Loaded " << all.size() << " records.\n";

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
