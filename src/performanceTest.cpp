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

void PerformanceSuite::runPerformanceTest(int numSamples, int SIZE, int NPOT, int* a, int* b, int* c)
{
    printf("\n");
    printf("***********************\n");
    printf("** PERFORMANCE TESTS **\n");
    printf("***********************\n");

    runIndividualPerformanceTest("CPU Scan", 
        [](int n, int* out, int* in) {
            StreamCompaction::CPU::scan(n, out, in, true);
        },
        []() {
            return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        }, numSamples, SIZE, NPOT, a, b, c);

    runIndividualPerformanceTest("Naive Scan",
        [](int n, int* out, int* in) {
            StreamCompaction::Naive::scan(n, out, in);
        },
        []() {
            return StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
        }, numSamples, SIZE, NPOT, a, b, c);

    runIndividualPerformanceTest("Efficient Scan",
        [](int n, int* out, int* in) {
            StreamCompaction::Efficient::scan(n, out, in);
        },
        []() {
            return StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
        }, numSamples, SIZE, NPOT, a, b, c);

    runIndividualPerformanceTest("Thrust Scan",
        [](int n, int* out, int* in) {
            StreamCompaction::Thrust::scan(n, out, in);
        },
        []() {
            return StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
        }, numSamples, SIZE, NPOT, a, b, c);

    runIndividualPerformanceTest("CPU Compact w/o scan",
        [](int n, int* out, int* in) {
            StreamCompaction::CPU::compactWithoutScan(n, out, in);
        },
        []() {
            return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        }, numSamples, SIZE, NPOT, a, b, c);

    runIndividualPerformanceTest("CPU Compact w/ scan",
        [](int n, int* out, int* in) {
            StreamCompaction::CPU::compactWithScan(n, out, in);
        },
        []() {
            return StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        }, numSamples, SIZE, NPOT, a, b, c);

    runIndividualPerformanceTest("GPU Compact",
        [](int n, int* out, int* in) {
            StreamCompaction::Efficient::compact(n, out, in);
        },
        []() {
            return StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
        }, numSamples, SIZE, NPOT, a, b, c);
}

void PerformanceSuite::runIndividualPerformanceTest(const std::string& testName, std::function<void(int, int*, int*)> functionToTest, std::function<float()> timerFunction, int numSamples, int SIZE, int NPOT, int* a, int* b, int* c)
{
    genArray(SIZE - 1, a, 50);
    a[SIZE - 1] = 0;

    std::cout << testName << "\n";
    float meanRuntime = 0;
    float stdDevRuntime = 0;
    std::vector<float> runtimes(numSamples);

    // Test power of 2 size
    for (int i = -1; i < numSamples; i++) {
        functionToTest(SIZE, b, a);
        if (i >= 0) {
            runtimes[i] = timerFunction();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    getMeanStandardDeviation(runtimes, meanRuntime, stdDevRuntime);
    std::cout << "\t[Power-of-Two] Mean: " << meanRuntime << " ms, StdDev: " << stdDevRuntime << " ms\n";

    // Test non-power of 2 size
    for (int i = -1; i < numSamples; i++) {
        functionToTest(NPOT, c, a);
        if (i >= 0) {
            runtimes[i] = timerFunction();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    getMeanStandardDeviation(runtimes, meanRuntime, stdDevRuntime);
    std::cout << "\t[Non-Power-of-Two] Mean: " << meanRuntime << " ms, StdDev: " << stdDevRuntime << " ms\n";
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

