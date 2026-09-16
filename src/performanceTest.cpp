#include "performanceTest.h"

#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

#include <thread>
#include <chrono>

#include <cmath>
#include <iostream>

void PerformanceSuite::runPerformanceTest(int numSamples)
{
    printf("\n");
    printf("***********************\n");
    printf("** PERFORMANCE TESTS **\n");
    printf("***********************\n");

    std::vector<int> sizes = {
        (1 << 8) - 3,
        (1 << 12) - 3,
        (1 << 16) - 3,
        (1 << 20) - 3,
        (1 << 24) - 3
    };

    std::cout << "\Sizes: [(1<<8) - 3, (1<<12) - 3, (1<<16) - 3, (1<<20) - 3, (1<<24) - 3]\n";

    runIndividualPerformanceTest("CPU Scan", 
        [](int n, int* out, int* in) {
            StreamCompaction::CPU::scan(n, out, in, true);
        },
        []() {
            return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        }, numSamples, sizes);

    runIndividualPerformanceTest("Naive Scan",
        [](int n, int* out, int* in) {
            StreamCompaction::Naive::scan(n, out, in);
        },
        []() {
            return StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
        }, numSamples, sizes);

    runIndividualPerformanceTest("Efficient Scan",
        [](int n, int* out, int* in) {
            StreamCompaction::Efficient::scan(n, out, in);
        },
        []() {
            return StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
        }, numSamples, sizes);

    runIndividualPerformanceTest("Thrust Scan",
        [](int n, int* out, int* in) {
            StreamCompaction::Thrust::scan(n, out, in);
        },
        []() {
            return StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
        }, numSamples, sizes);

    runIndividualPerformanceTest("CPU Compact w/o scan",
        [](int n, int* out, int* in) {
            StreamCompaction::CPU::compactWithoutScan(n, out, in);
        },
        []() {
            return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        }, numSamples, sizes);

    runIndividualPerformanceTest("CPU Compact w/ scan",
        [](int n, int* out, int* in) {
            StreamCompaction::CPU::compactWithScan(n, out, in);
        },
        []() {
            return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        }, numSamples, sizes);

    runIndividualPerformanceTest("GPU Compact",
        [](int n, int* out, int* in) {
            StreamCompaction::Efficient::compact(n, out, in);
        },
        []() {
            return StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
        }, numSamples, sizes);
}

void PerformanceSuite::runIndividualPerformanceTest(const std::string& testName,
    std::function<void(int, int*, int*)> functionToTest,
    std::function<float()> timerFunction,
    int numSamples,
    std::vector<int> sizes)
{
    std::vector<float> meanRuntimes(sizes.size());
    std::vector<float> standardDeviations(sizes.size());

    for (int i = 0; i < sizes.size(); i++) {
        int* a = new int[sizes[i]];
        int* b = new int[sizes[i]];

        genArray(sizes[i] - 1, a, 50);
        a[sizes[i] - 1] = 0;

        float meanRuntime = 0;
        float stdDevRuntime = 0;
        std::vector<float> runtimes(numSamples);

        // Run numSamples + 1 throwaway tests
        for (int j = -3; j < numSamples; j++) {
            functionToTest(sizes[i], b, a);
            if (j >= 0) {
                runtimes[j] = timerFunction();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        getMeanStandardDeviation(runtimes, meanRuntimes[i], standardDeviations[i]);

        delete[] a;
        delete[] b;
    }

    std::cout << testName << "\n";
    std::cout << "\tRuntimes (ms): [";
    for (size_t i = 0; i < meanRuntimes.size(); ++i) {
        std::cout << meanRuntimes[i];
        if (i < meanRuntimes.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << "\tStandard Deviations: [";
    for (size_t i = 0; i < standardDeviations.size(); ++i) {
        std::cout << standardDeviations[i];
        if (i < standardDeviations.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

void PerformanceSuite::getMeanStandardDeviation(std::vector<float> runtimes, float& outMean, float& outStandardDeviation)
{
    if (runtimes.empty()) {
        outMean = 0.0f;
        outStandardDeviation = 0.0f;
        return;
    }

    // Mean calculation
    float sum = 0.0;
    for (float time : runtimes) {
        sum += time;
    }
    float mean = sum / runtimes.size();
    outMean = mean;

    // No standard deviation if only a single element
    if (runtimes.size() == 1) {
        outStandardDeviation = 0.0f;
        return;
    }

    // Standard deviation formula
    float squareSum = 0.0;
    for (float time : runtimes) {
        squareSum += (time - mean) * (time - mean);
    }

    float variance = squareSum / (runtimes.size() - 1);
    outStandardDeviation = std::sqrt(variance);
}

