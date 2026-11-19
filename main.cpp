#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Matrix = std::vector<std::vector<double>>;
using HighClock = std::chrono::high_resolution_clock;
using Tick = HighClock::time_point;
using Rows = void (*)(const Matrix&, const Matrix&, Matrix&, int,
                      int);

Matrix make_matrix(int n, double value) {
    return Matrix(n, std::vector<double>(n, value));
}

void multiply_ikj(const Matrix& A, const Matrix& B, Matrix& C,
                  int start_row, int end_row) {
    const int n = static_cast<int>(A.size());
    for (int i = start_row; i < end_row; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][k] * B[k][j];
}

void multiply_ijk(const Matrix& A, const Matrix& B, Matrix& C,
                  int start_row, int end_row) {
    const int n = static_cast<int>(A.size());
    for (int i = start_row; i < end_row; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

double milliseconds(Tick start, Tick finish) {
    const double elapsed =
        std::chrono::duration<double, std::milli>(finish - start)
            .count();
    if (elapsed <= 0.0)
        throw std::runtime_error("Invalid clock interval");
    return elapsed;
}

double measure_serial(const Matrix& A, const Matrix& B,
                      Matrix& C, Rows multiply) {
    const auto start = HighClock::now();
    multiply(A, B, C, 0, static_cast<int>(A.size()));
    return milliseconds(start, HighClock::now());
}

double measure_parallel(const Matrix& A, const Matrix& B,
                        Matrix& C, int p, Rows multiply,
                        bool gated) {
    const int n = static_cast<int>(A.size());
    const int base = n / p;
    const int extra = n % p;
    std::atomic<bool> start_flag{false};
    std::atomic<int> ready{0};
    std::vector<std::thread> workers;
    workers.reserve(p);

    Tick start;
    if (!gated)
        start = HighClock::now();

    try {
        int first = 0;
        for (int t = 0; t < p; ++t) {
            const int last = first + base + (t < extra ? 1 : 0);
            workers.emplace_back([&, first, last] {
                if (gated) {
                    ready.fetch_add(1,
                                    std::memory_order_release);
                    while (!start_flag.load(
                        std::memory_order_acquire))
                        std::this_thread::yield();
                }
                multiply(A, B, C, first, last);
            });
            first = last;
        }
    } catch (...) {
        start_flag.store(true, std::memory_order_release);
        for (auto& worker : workers)
            worker.join();
        throw;
    }

    if (gated) {
        while (ready.load(std::memory_order_acquire) != p)
            std::this_thread::yield();
        start = HighClock::now();
        start_flag.store(true, std::memory_order_release);
    }

    for (auto& worker : workers)
        worker.join();
    return milliseconds(start, HighClock::now());
}

void check(const Matrix& C) {
    const double expected = 2.0 * C.size();
    for (const auto& row : C)
        for (double value : row)
            if (value != expected)
                throw std::runtime_error("Incorrect matrix");
}

struct Case {
    const char* id;
    int threads;
    Rows multiply;
    bool gated;
};

double run_cases(int n, const std::vector<Case>& cases,
                 bool compare_four) {
    constexpr int repeats = 5;
    const Matrix A = make_matrix(n, 1.0);
    const Matrix B = make_matrix(n, 2.0);
    std::vector<std::array<double, repeats>> samples(
        cases.size());

    // Round 0 is a warm-up. Rotate the variant order in each
    // round.
    for (int round = 0; round <= repeats; ++round) {
        for (std::size_t offset = 0; offset < cases.size();
             ++offset) {
            const auto index = (offset + round) % cases.size();
            const auto& test = cases[index];
            Matrix C = make_matrix(n, 0.0);
            const double ms =
                test.threads == 0
                    ? measure_serial(A, B, C, test.multiply)
                    : measure_parallel(A, B, C, test.threads,
                                       test.multiply,
                                       test.gated);
            check(C);
            if (round > 0)
                samples[index][round - 1] = ms;
        }
    }

    for (auto& values : samples)
        std::sort(values.begin(), values.end());
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& values = samples[i];
        const double median = values[repeats / 2];
        std::cout << cases[i].id << " (n=" << n << ", threads="
                  << std::max(1, cases[i].threads)
                  << "): " << median << " ms (min "
                  << values.front() << ", max " << values.back()
                  << "), check OK\n";
    }
    if (compare_four) {
        std::cout << "T1/T4 = "
                  << samples[0][repeats / 2] /
                         samples[1][repeats / 2]
                  << "x\n\n";
    }
    return samples[0][repeats / 2];
}

double parse_time(const std::string& value) {
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), ',', '.');
    std::size_t end = 0;
    const double time = std::stod(normalized, &end);
    if (end != normalized.size() || !std::isfinite(time) ||
        time <= 0.0) {
        throw std::invalid_argument(
            "Stage 1 time must be positive (ms)");
    }
    return time;
}

void run_stage(char stage, double baseline) {
    constexpr int n = 800;
    const unsigned hardware =
        std::thread::hardware_concurrency();
    const int p = static_cast<int>(std::min(
        static_cast<unsigned>(n), std::max(1u, hardware)));
    const Case serial{"serial", 0, multiply_ikj, false};

    std::cout << "Stage " << stage
              << ", hardware_concurrency=" << hardware
              << ", clock_is_steady=" << HighClock::is_steady
              << '\n'
              << std::fixed << std::setprecision(6);
    switch (stage) {
        case '1':
            run_cases(n, {serial}, false);
            break;
        case '2':
            run_cases(n, {{"naive", p, multiply_ikj, false}},
                      false);
            break;
        case '3':
            run_cases(n, {{"gated", p, multiply_ikj, true}},
                      false);
            break;
        case '4':
            for (int size : {300, 500, n, 1500}) {
                run_cases(
                    size,
                    {serial, {"scale_4", 4, multiply_ikj, true}},
                    true);
            }
            break;
        case '5':
            run_cases(n, {{"ijk", p, multiply_ijk, true}},
                      false);
            break;
        case '6': {
            const double final_time = run_cases(
                n, {{"final", p, multiply_ikj, true}}, false);
            if (baseline > 0.0) {
                std::cout << "T1/T6 = " << baseline / final_time
                          << "x\n";
            } else {
                std::cout
                    << "Speedup = stage 1 time / final time.\n";
            }
            break;
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::string stage = argc > 1 ? argv[1] : "";
        if (stage.size() != 1 || stage[0] < '1' ||
            stage[0] > '6' || argc > 3 ||
            (argc == 3 && stage != "6")) {
            std::cerr << "Usage: ./sem7 <1-6> "
                         "[stage_1_time_ms_for_stage_6]\n";
            return 1;
        }
        run_stage(stage[0],
                  argc == 3 ? parse_time(argv[2]) : 0.0);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
