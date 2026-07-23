#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 5) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <train_data> <test_data> <epochs> <learning_rate> [lambda] [overlap]\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string train_data = argv[1];
    std::string test_data = argv[2];
    int epochs = std::stoi(argv[3]);
    double learning_rate = std::stod(argv[4]);
    double lambda = argc > 5 ? std::stod(argv[5]) : 0.1;
    bool overlap = argc > 6 ? std::stoi(argv[6]) != 0 : false;

    if (rank == 0) {
        std::cout << "MPI processes: " << world_size << "\n";
        std::cout << "Dataset: " << train_data << "\n";
        std::cout << "Epochs: " << epochs << "\n";
        std::cout << "Learning rate: " << learning_rate << "\n";
        std::cout << "Lambda: " << lambda << "\n";
        std::cout << "Overlap: " << (overlap ? "enabled" : "disabled") << "\n";
    }

    Dataset train = load_svmlight_dataset(train_data, -1, rank, world_size);
    Dataset test;
    if (rank == 0) {
        test = load_svmlight_dataset(test_data);
    }

    std::size_t local_feature_count = train.cols;
    if (rank == 0) {
        local_feature_count = std::max(local_feature_count, test.cols);
    }

    unsigned long long local_cols = static_cast<unsigned long long>(local_feature_count);
    unsigned long long global_cols = 0;
    MPI_Allreduce(&local_cols, &global_cols, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    std::size_t feature_count = static_cast<std::size_t>(global_cols);

    long long local_rows = static_cast<long long>(train.rows);
    long long global_rows = 0;
    MPI_Allreduce(&local_rows, &global_rows, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (global_rows == 0 || feature_count == 0) {
        if (rank == 0) {
            std::cerr << "Error: Dataset is empty.\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0 && test.rows == 0) {
        std::cerr << "Error: Test dataset is empty.\n";
        MPI_Finalize();
        return 1;
    }

    std::vector<double> weights(feature_count, 0.0);
    double bias = 0.0;

    const auto start_time = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::vector<double> local_gradient(feature_count, 0.0);
        double local_bias_gradient = 0.0;

        for (std::size_t i = 0; i < train.rows; ++i) {
            const double score = sparse_dot(train.csr, i, weights) + bias;
            const double probability = std::clamp(sigmoid(score), 1e-12, 1.0 - 1e-12);
            const double error = probability - static_cast<double>(train.labels[i]);

            local_bias_gradient += error;

            const int begin = train.csr.row_ptr[i];
            const int end = train.csr.row_ptr[i + 1];
            for (int k = begin; k < end; ++k) {
                local_gradient[train.csr.col_idx[k]] += error * train.csr.values[k];
            }
        }

        std::vector<double> global_gradient(feature_count, 0.0);
        double global_bias_gradient = 0.0;

        if (overlap) {
            MPI_Request requests[2];
            MPI_Iallreduce(local_gradient.data(), global_gradient.data(), static_cast<int>(feature_count), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &requests[0]);
            MPI_Iallreduce(&local_bias_gradient, &global_bias_gradient, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &requests[1]);

            std::vector<double> regularization(feature_count, 0.0);
            for (std::size_t col = 0; col < feature_count; ++col) {
                regularization[col] = lambda * weights[col];
            }

            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

            const double scale = 1.0 / static_cast<double>(global_rows);
            for (std::size_t col = 0; col < feature_count; ++col) {
                global_gradient[col] = global_gradient[col] * scale + regularization[col];
                weights[col] -= learning_rate * global_gradient[col];
            }
            bias -= learning_rate * global_bias_gradient * scale;
        } else {
            MPI_Allreduce(local_gradient.data(), global_gradient.data(), static_cast<int>(feature_count), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_bias_gradient, &global_bias_gradient, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            const double scale = 1.0 / static_cast<double>(global_rows);
            for (std::size_t col = 0; col < feature_count; ++col) {
                global_gradient[col] = global_gradient[col] * scale + lambda * weights[col];
                weights[col] -= learning_rate * global_gradient[col];
            }
            bias -= learning_rate * global_bias_gradient * scale;
        }

        Metrics local_metrics = evaluate_dataset(train, weights, bias, lambda);
        
        double local_loss_sum = local_metrics.loss * train.rows;
        double local_correct = local_metrics.accuracy * train.rows;

        double global_loss_sum = 0.0;
        double global_correct = 0.0;

        MPI_Allreduce(&local_loss_sum, &global_loss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_correct, &global_correct, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double global_loss = global_loss_sum / static_cast<double>(global_rows);
        double global_accuracy = global_correct / static_cast<double>(global_rows);

        if (rank == 0) {
            Metrics test_metrics = evaluate_dataset(test, weights, bias, lambda);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

            std::cout << "[mpi] epoch " << epoch
                      << " time " << elapsed << "s"
                      << " train_loss " << global_loss
                      << " train_acc " << global_accuracy
                      << " test_loss " << test_metrics.loss
                      << " test_acc " << test_metrics.accuracy << "\n";
        }
    }

    if (rank == 0) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        std::cout << "Training finished in " << elapsed << " seconds.\n";
    }

    MPI_Finalize();
    return 0;
}
