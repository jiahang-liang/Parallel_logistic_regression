#include "common.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <string>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <train_path> <test_path> <epochs> <learning_rate> <lambda> <threads>\n";
        return 1;
    }

    std::string train_path = argv[1];
    std::string test_path = argv[2];
    int epochs = std::stoi(argv[3]);
    double learning_rate = std::stod(argv[4]);
    double lambda = std::stod(argv[5]);
    int num_threads = std::stoi(argv[6]);

    omp_set_num_threads(num_threads);

    Dataset train = load_svmlight_dataset(train_path);
    Dataset test = load_svmlight_dataset(test_path);

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

        #pragma omp parallel
        {
            std::vector<double> local_gradient(feature_count, 0.0);
            double local_bias_gradient = 0.0;

            #pragma omp for
            for (int i = 0; i < static_cast<int>(train.rows); ++i) {
                double score = sparse_dot(train.csr, i, weights) + bias;
                double prob = sigmoid(score);
                double error = prob - train.labels[i];

                local_bias_gradient += error;

                int begin = train.csr.row_ptr[i];
                int end = train.csr.row_ptr[i + 1];
                for (int k = begin; k < end; ++k) {
                    local_gradient[train.csr.col_idx[k]] += error * train.csr.values[k];
                }
            }

            #pragma omp critical
            {
                bias_gradient += local_bias_gradient;
                for (std::size_t j = 0; j < feature_count; ++j) {
                    gradient[j] += local_gradient[j];
                }
            }
        }

        const double scale = 1.0 / static_cast<double>(train.rows);
        for (std::size_t j = 0; j < feature_count; ++j) {
            gradient[j] = gradient[j] * scale + lambda * weights[j];
            weights[j] -= learning_rate * gradient[j];
        }
        bias -= learning_rate * bias_gradient * scale;

        Metrics train_metrics = evaluate_dataset(train, weights, bias, lambda);
        Metrics test_metrics = evaluate_dataset(test, weights, bias, lambda);
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

        std::cout << "[omp] epoch " << epoch
                  << " time " << elapsed << "s"
                  << " train_loss " << train_metrics.loss
                  << " train_acc " << train_metrics.accuracy
                  << " test_loss " << test_metrics.loss
                  << " test_acc " << test_metrics.accuracy << "\n";
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "OpenMP threads used: " << omp_get_max_threads() << "\n";
    std::cout << "Training time: " << elapsed.count() << " seconds\n";

    return 0;
}
