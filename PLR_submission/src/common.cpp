#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
    int normalize_label_value(double raw_label) {
        return raw_label > 0.0 ? 1 : 0;
    }
}

Dataset load_svmlight_dataset(const std::string& data_path, int max_rows, int rank, int world_size) {
    Dataset dataset;
    dataset.csr.row_ptr.push_back(0);

    std::ifstream stream(data_path);
    if (!stream) {
        throw std::runtime_error("Cannot open svmlight data file: " + data_path);
    }

    std::string line;
    int kept_rows = 0;
    int global_row_index = 0;

    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }

        const bool keep = (global_row_index % world_size) == rank;
        global_row_index += 1;
        if (!keep) {
            continue;
        }

        std::istringstream row_stream(line);
        std::string token;
        if (!(row_stream >> token)) {
            continue;
        }
        try {
            double label_val = std::stod(token);
            dataset.labels.push_back(normalize_label_value(label_val));
        } catch (const std::exception& e) {
            continue; // Skip invalid lines
        }

        while (row_stream >> token) {
            const std::size_t colon_pos = token.find(':');
            if (colon_pos == std::string::npos) {
                continue;
            }
            try {
                const int col = std::stoi(token.substr(0, colon_pos)) - 1;
                const double value = std::stod(token.substr(colon_pos + 1));
                dataset.csr.col_idx.push_back(col);
                dataset.csr.values.push_back(value);
                dataset.cols = std::max(dataset.cols, static_cast<std::size_t>(col + 1));
            } catch (const std::exception& e) {
                continue; // Skip invalids
            }
        }

        dataset.csr.row_ptr.push_back(static_cast<int>(dataset.csr.values.size()));
        kept_rows += 1;

        if (max_rows > 0 && kept_rows >= max_rows) {
            break;
        }
    }

    dataset.rows = dataset.labels.size();
    dataset.csr.rows = dataset.rows;
    dataset.csr.cols = dataset.cols;
    return dataset;
}

double sigmoid(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

double sparse_dot(const SparseMatrix& matrix, std::size_t row, const std::vector<double>& weights) {
    const int begin = matrix.row_ptr[row];
    const int end = matrix.row_ptr[row + 1];
    double sum = 0.0;
    for (int k = begin; k < end; ++k) {
        sum += matrix.values[k] * weights[matrix.col_idx[k]];
    }
    return sum;
}

Metrics evaluate_dataset(const Dataset& dataset, const std::vector<double>& weights, double bias, double lambda) {
    Metrics metrics;
    if (dataset.rows == 0) {
        return metrics;
    }

    double total_loss = 0.0;
    double correct = 0.0;

    for (std::size_t i = 0; i < dataset.rows; ++i) {
        const double score = sparse_dot(dataset.csr, i, weights) + bias;
        const double probability = std::clamp(sigmoid(score), 1e-12, 1.0 - 1e-12);
        const int label = dataset.labels[i];
        const int predicted = probability >= 0.5 ? 1 : 0;

        total_loss += -(label * std::log(probability) + (1 - label) * std::log(1.0 - probability));
        if (predicted == label) {
            correct += 1.0;
        }
    }

    double reg = 0.0;
    for (double weight : weights) {
        reg += weight * weight;
    }

    metrics.loss = total_loss / static_cast<double>(dataset.rows) + 0.5 * lambda * reg;
    metrics.accuracy = correct / static_cast<double>(dataset.rows);
    return metrics;
}
