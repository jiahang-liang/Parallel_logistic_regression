#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct SparseMatrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};

struct Dataset {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<int> labels;
    SparseMatrix csr;
};

struct Metrics {
    double loss = 0.0;
    double accuracy = 0.0;
};

Dataset load_svmlight_dataset(
    const std::string& data_path,
    int max_rows = -1,
    int rank = 0,
    int world_size = 1
);

double sigmoid(double x);
double sparse_dot(const SparseMatrix& matrix, std::size_t row, const std::vector<double>& weights);

Metrics evaluate_dataset(const Dataset& dataset, const std::vector<double>& weights, double bias, double lambda);
