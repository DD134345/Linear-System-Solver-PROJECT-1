#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <random>

#include "Vector.h"
#include "Matrix.h"
#include "LinearSystem.h"
using namespace std;

// ═══════════════════════════════════════════════════════════════════════════
//  Printing helpers
// ═══════════════════════════════════════════════════════════════════════════

void PrintVector(const Vector& v, const std::string& name = "") {
    if (!name.empty()) std::cout << name << " = [";
    else               std::cout << "[";
    for (int i = 1; i <= v.GetSize(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << v(i);
        if (i < v.GetSize()) std::cout << ", ";
    }
    std::cout << "]\n";
}

void PrintMatrix(const Matrix& M, const std::string& name = "") {
    if (!name.empty()) std::cout << name << ":\n";
    for (int i = 1; i <= M.GetNumRows(); ++i) {
        std::cout << "  [ ";
        for (int j = 1; j <= M.GetNumCols(); ++j)
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << M(i, j);
        std::cout << " ]\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 1: Vector operations
// ═══════════════════════════════════════════════════════════════════════════

void TestVector() {
    std::cout << "\n╔══════════════════════════════════════╗\n";
    std::cout <<   "║  TEST 1: Vector Operations           ║\n";
    std::cout <<   "╚══════════════════════════════════════╝\n";

    Vector a(3), b(3);
    a(1) = 1; a(2) = 2; a(3) = 3;
    b(1) = 4; b(2) = 5; b(3) = 6;

    PrintVector(a, "a");
    PrintVector(b, "b");
    PrintVector(a + b,   "a + b");
    PrintVector(a - b,   "a - b");
    PrintVector(a * 2.0, "a * 2");
    std::cout << "a . b = " << a.Dot(b) << "\n";
    std::cout << "||a|| = " << a.Norm()  << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 2: Matrix operations + det + inverse
// ═══════════════════════════════════════════════════════════════════════════

void TestMatrix() {
    std::cout << "\n╔══════════════════════════════════════╗\n";
    std::cout <<   "║  TEST 2: Matrix Operations           ║\n";
    std::cout <<   "╚══════════════════════════════════════╝\n";

    // 3×3 invertible matrix
    Matrix A(3, 3);
    A(1,1)=2; A(1,2)=1; A(1,3)=0;
    A(2,1)=1; A(2,2)=3; A(2,3)=1;
    A(3,1)=0; A(3,2)=1; A(3,3)=2;

    PrintMatrix(A, "A");
    std::cout << "det(A) = " << A.Determinant() << "  (expected 8)\n";

    Matrix Ainv = A.Inverse();
    PrintMatrix(Ainv, "A^{-1}");

    Matrix I = A * Ainv;
    std::cout << "A * A^{-1} (should be identity):\n";
    PrintMatrix(I);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 3: LinearSystem — Gaussian elimination
// ═══════════════════════════════════════════════════════════════════════════

void TestLinearSystem() {
    std::cout << "\n╔══════════════════════════════════════╗\n";
    std::cout <<   "║  TEST 3: LinearSystem (Gauss+pivot)  ║\n";
    std::cout <<   "╚══════════════════════════════════════╝\n";

    // 3x + y = 9
    // x + 2y = 8
    // Expected: x=2, y=3   ... wait, let's use a 3x3 known system:
    // 2x + y       = 5
    //  x + 3y + z  = 10
    //      y + 2z  = 7
    // Solution: x=1, y=3, z=2
    Matrix A(3, 3);
    Vector b(3);
    A(1,1)=2; A(1,2)=1; A(1,3)=0;
    A(2,1)=1; A(2,2)=3; A(2,3)=1;
    A(3,1)=0; A(3,2)=1; A(3,3)=2;
    b(1)=5; b(2)=10; b(3)=7;

    PrintMatrix(A, "A");
    PrintVector(b, "b");

    LinearSystem ls(A, b);
    Vector x = ls.Solve();
    PrintVector(x, "x (Gaussian)");
    std::cout << "Expected: [1.5, 2, 2.5]\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 4: PosSymLinSystem — Conjugate Gradient
// ═══════════════════════════════════════════════════════════════════════════

void TestCGSystem() {
    std::cout << "\n╔══════════════════════════════════════╗\n";
    std::cout <<   "║  TEST 4: PosSymLinSystem (CG)        ║\n";
    std::cout <<   "╚══════════════════════════════════════╝\n";

    // SPD matrix (diag-dominant symmetric)
    Matrix A(3, 3);
    Vector b(3);
    A(1,1)=4; A(1,2)=1; A(1,3)=0;
    A(2,1)=1; A(2,2)=3; A(2,3)=1;
    A(3,1)=0; A(3,2)=1; A(3,3)=4;
    b(1)=6; b(2)=5; b(3)=5;

    PosSymLinSystem cg(A, b);
    Vector x = cg.Solve();
    PrintVector(x, "x (CG)");

    // Verify residual
    Vector r = b - A * x;
    std::cout << "Residual ||b - Ax|| = " << r.Norm() << " (should be ~0)\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 5: Pseudo-inverse on a small overdetermined system
// ═══════════════════════════════════════════════════════════════════════════

void TestPseudoInverse() {
    std::cout << "\n╔══════════════════════════════════════╗\n";
    std::cout <<   "║  TEST 5: Pseudo-inverse (4×2)        ║\n";
    std::cout <<   "╚══════════════════════════════════════╝\n";

    // Overdetermined: 4 equations, 2 unknowns
    // True answer: [1, 2]
    Matrix A(4, 2);
    Vector b(4);
    A(1,1)=1; A(1,2)=1;
    A(2,1)=1; A(2,2)=2;
    A(3,1)=1; A(3,2)=3;
    A(4,1)=1; A(4,2)=4;
    b(1)=3; b(2)=5; b(3)=7; b(4)=9;   // exactly [1,2]

    Matrix Aplus = A.PseudoInverse();
    Vector x = Aplus * b;
    PrintVector(x, "x (pseudo-inverse)");
    std::cout << "Expected: [1, 2]\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  Part B: Linear regression on CPU performance dataset
// ═══════════════════════════════════════════════════════════════════════════

struct Record {
    double myct, mmin, mmax, cach, chmin, chmax;
    double prp;
};

std::vector<Record> LoadDataset(const std::string& filename) {
    std::vector<Record> records;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open " << filename << "\n";
        return records;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> tokens;
        while (std::getline(ss, tok, ',')) tokens.push_back(tok);
        if (tokens.size() < 10) continue;

        Record r;
        r.myct  = std::stod(tokens[2]);
        r.mmin  = std::stod(tokens[3]);
        r.mmax  = std::stod(tokens[4]);
        r.cach  = std::stod(tokens[5]);
        r.chmin = std::stod(tokens[6]);
        r.chmax = std::stod(tokens[7]);
        r.prp   = std::stod(tokens[8]);
        records.push_back(r);
    }
    return records;
}

double ComputeRMSE(const Vector& params, const std::vector<Record>& data) {
    double sumSq = 0.0;
    for (const auto& r : data) {
        double pred = params(1) * r.myct + params(2) * r.mmin
                    + params(3) * r.mmax + params(4) * r.cach
                    + params(5) * r.chmin + params(6) * r.chmax;
        double err = pred - r.prp;
        sumSq += err * err;
    }
    return std::sqrt(sumSq / data.size());
}

void RunRegression(const std::string& dataPath) {
    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout <<   "║  PART B: CPU Performance Linear Regression   ║\n";
    std::cout <<   "╚══════════════════════════════════════════════╝\n";

    auto all = LoadDataset(dataPath);
    if (all.empty()) return;

    std::cout << "Loaded " << all.size() << " records.\n";

    // Shuffle with fixed seed for reproducibility
    std::mt19937 rng(42);
    std::shuffle(all.begin(), all.end(), rng);

    int nTrain = static_cast<int>(all.size() * 0.8);
    int nTest  = static_cast<int>(all.size()) - nTrain;

    std::vector<Record> trainSet(all.begin(), all.begin() + nTrain);
    std::vector<Record> testSet (all.begin() + nTrain, all.end());

    std::cout << "Train: " << nTrain << "  |  Test: " << nTest << "\n";

    // Build A (nTrain × 6) and b (nTrain)
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

    // ── Method 1: Pseudo-inverse  x = A^+ b ─────────────────────────────
    std::cout << "\n── Method 1: Pseudo-inverse (A^+ b) ──\n";
    Matrix Aplus  = A.PseudoInverse();
    Vector params1 = Aplus * b;
    PrintVector(params1, "params [x1..x6]");
    std::cout << "Train RMSE : " << std::fixed << std::setprecision(4)
              << ComputeRMSE(params1, trainSet) << "\n";
    std::cout << "Test  RMSE : " << ComputeRMSE(params1, testSet)  << "\n";

    // ── Method 2: Normal equations via Gaussian elimination ─────────────
    // (A^T A) x = A^T b  →  square 6×6 system
    std::cout << "\n── Method 2: Normal equations (Gaussian elim) ──\n";
    Matrix AtA = A.Transpose() * A;
    Vector Atb = A.Transpose() * b;
    LinearSystem normalEq(AtA, Atb);
    Vector params2 = normalEq.Solve();
    PrintVector(params2, "params [x1..x6]");
    std::cout << "Train RMSE : " << ComputeRMSE(params2, trainSet) << "\n";
    std::cout << "Test  RMSE : " << ComputeRMSE(params2, testSet)  << "\n";

    // ── Method 3: Normal equations via Conjugate Gradient ───────────────
    // A^T A is symmetric positive semi-definite
    std::cout << "\n── Method 3: Normal equations (Conjugate Gradient) ──\n";
    PosSymLinSystem cgEq(AtA, Atb);
    Vector params3 = cgEq.Solve();
    PrintVector(params3, "params [x1..x6]");
    std::cout << "Train RMSE : " << ComputeRMSE(params3, trainSet) << "\n";
    std::cout << "Test  RMSE : " << ComputeRMSE(params3, testSet)  << "\n";

    // ── Prediction preview ───────────────────────────────────────────────
    std::cout << "\n── Test set sample: Predicted vs Actual (first 10) ──\n";
    std::cout << std::setw(10) << "Predicted"
              << std::setw(10) << "Actual"
              << std::setw(12) << "Error\n";
    std::cout << std::string(32, '-') << "\n";
    for (int i = 0; i < std::min(10, nTest); ++i) {
        const auto& r = testSet[i];
        double pred = params2(1)*r.myct + params2(2)*r.mmin
                    + params2(3)*r.mmax + params2(4)*r.cach
                    + params2(5)*r.chmin + params2(6)*r.chmax;
        std::cout << std::setw(10) << std::fixed << std::setprecision(1) << pred
                  << std::setw(10) << r.prp
                  << std::setw(12) << (pred - r.prp) << "\n";
    }

    std::cout << "\nFeature interpretation:\n";
    std::cout << "  x1 (MYCT  - cycle time)  = " << params2(1) << "\n";
    std::cout << "  x2 (MMIN  - min memory)  = " << params2(2) << "\n";
    std::cout << "  x3 (MMAX  - max memory)  = " << params2(3) << "\n";
    std::cout << "  x4 (CACH  - cache)       = " << params2(4) << "\n";
    std::cout << "  x5 (CHMIN - min channels)= " << params2(5) << "\n";
    std::cout << "  x6 (CHMAX - max channels)= " << params2(6) << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║       Project 1: Linear System Solver          ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";

    TestVector();
    TestMatrix();
    TestLinearSystem();
    TestCGSystem();
    TestPseudoInverse();

    // Dataset path: first arg, or default
    std::string dataPath = (argc > 1) ? argv[1] : "machine.data";
    RunRegression(dataPath);

    std::cout << "\nDone.\n";
    return 0;
}
