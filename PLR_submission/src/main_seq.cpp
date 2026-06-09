#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <train_data> <test_data> <epochs> <learning_rate> <lambda>\n";
        return 1;
    }

    std::string train_data = argv[1];
    std::string test_data = argv[2];
    int epochs = std::stoi(argv[3]);
    double learning_rate = std::stod(argv[4]);
    double lambda = std::stod(argv[5]);

    Dataset train = load_svmlight_dataset(train_data);
    Dataset test = load_svmlight_dataset(test_data);

    std::size_t feature_count = std::max(train.cols, test.cols);
    if (feature_count == 0 || train.rows == 0) {
        std::cerr << "Training dataset is empty.\n";
        return 1;
    }

    std::vector<double> weights(feature_count, 0.0);
    double bias = 0.0;

    auto start_time = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::vector<double> gradient(feature_count, 0.0);
        double bias_gradient = 0.0;

        for (std::size_t row = 0; row < train.rows; ++row) {
            int begin = train.csr.row_ptr[row];
            int end = train.csr.row_ptr[row + 1];

            double dot = bias;
            for (int k = begin; k < end; ++k) {
                dot += weights[train.csr.col_idx[k]] * train.csr.values[k];
            }

            // Sigmoid
            double probability = 1.0 / (1.0 + std::exp(-dot));

            double error = probability - train.labels[row];

            bias_gradient += error;
            for (int k = begin; k < end; ++k) {
                gradient[train.csr.col_idx[k]] += error * train.csr.values[k];
            }
        }

        double scale = 1.0 / train.rows;
        for (std::size_t col = 0; col < feature_count; ++col) {
            gradient[col] = gradient[col] * scale + lambda * weights[col];
            weights[col] -= learning_rate * gradient[col];
        }
        bias -= learning_rate * bias_gradient * scale;

        Metrics train_metrics = evaluate_dataset(train, weights, bias, lambda);
        Metrics test_metrics = evaluate_dataset(test, weights, bias, lambda);

        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

        std::cout << "[seq] epoch " << epoch 
                  << " time " << elapsed << "s"
                  << " train_loss " << train_metrics.loss 
                  << " train_acc " << train_metrics.accuracy
                  << " test_loss " << test_metrics.loss 
                  << " test_acc " << test_metrics.accuracy << "\n";
    }

    return 0;
}
